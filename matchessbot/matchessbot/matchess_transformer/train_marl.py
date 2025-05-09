# This code is an adaptation of: https://github.com/sgrvinod/chess-transformers

import os
import pathlib
import math
import time

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler
import torch.nn.functional as F

from torch.nn import HuberLoss

from matchess_transformer.config import import_config
from matchess_transformer.dataset import MATChessDataset
from matchess_transformer.model import MATChessTransformerEncoder, LabelSmoothedCE, huber_loss
from matchess_transformer.tokenizer import Tokenizer
from matchess_transformer.vocab_uci_dicts import CHESS_PIECE_AGENTS



# def training_inference_phase(stepsize, batch_size, n_agents, n_episodes, steps_per_episode):
#     for episode in range(n_episodes):
#         for t in range(steps_per_episode):




def get_lr(step, d_model, warmup_steps, schedule="vaswani", decay=0.06, fixed=1e-5):
    """
    The LR schedule.

    Args:

        step (int): Training step number.

        d_model (int): Size of vectors throughout the transformer model.

        warmup_steps (int): Number of warmup steps where learning rate
        is increased linearly; twice the value in the paper, as in the
        official T2T repo.

    Returns:

        float: Updated learning rate.

    Args:

        step (int): Training step number.

        d_model (int): Size of vectors throughout the transformer model.

        warmup_steps (int): Number of warmup steps where learning rate
        is increased linearly.

        schedule (str, optional): The learning rate schedule. Defaults
        to "vaswani", in which case the schedule in "Attention Is All
        You Need", by Vasvani et. al. is followed. This version below is
        twice the definition in the paper, as used in the official T2T
        repository. If the schedule is "exp_decay", the learning rate is
        exponentially decayed after the warmup stage.

        decay (float, optional): The decay rate per 10000 training steps
        for the "exp_decay" schedule. Defaults to 0.06, i.e. 6%.

    Raises:

        NotImplementedError: If the schedule is not one of "vaswani" or
        "exp_decay".

    Returns:

        float: Updated learning rate.
    """
    if schedule == "vaswani":
        lr = (
            2.0
            * math.pow(d_model, -0.5)
            * min(math.pow(step, -0.5), step * math.pow(warmup_steps, -1.5))
        )
    elif schedule == "exp_decay":
        if step <= warmup_steps:
            lr = 1e-3 * step / warmup_steps
        else:
            lr = 1e-3 * ((1 - decay) ** ((step - warmup_steps) / 10000))
    elif schedule == "fixed":
        lr = fixed
    else:
        raise NotImplementedError

    return lr


def save_checkpoint(epoch, model, optimizer, config_name, checkpoint_folder, prefix=""):
    """
    Checkpoint saver. Each save overwrites any previous save.

    Args:

        epoch (int): The epoch number (0-indexed).

        model (torch.nn.Module): The transformer model.

        optimizer (torch.optim.adam.Adam): The optimizer.

        config_name (str): The configuration name.

        checkpoint_folder (str): The folder where checkpoints must be
        saved.

        prefix (str, optional): The checkpoint filename prefix. Defaults
        to "".
    """
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    pathlib.Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
    filename = prefix + config_name + ".pt"
    torch.save(state, os.path.join(checkpoint_folder, filename))
    print("Checkpoint saved.\n")

def change_lr(optimizer, new_lr):
    """
    Change learning rate to a specified value.

    Args:

        optimizer (torch.optim.adam.Adam): Optimizer whose learning rate
        must be changed.

        new_lr (float): New learning rate.
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def topk_accuracy(logits, targets, other_logits=None, other_targets=None, k=[1, 3, 5]):
    """
    Compute "top-k" accuracies for multiple values of "k".

    Optionally, a second set of logits and targets, for a second
    predicted variable, can be provided. In this case, probabilities
    associated with both sets of logits are combined to arrive at the
    best combinations of both predicted variables. A correct prediction
    occurs when the combination of the targets is present in the top "k"
    predicted combinations.

    Args:

        logits (torch.FloatTensor): Predicted logits, of size (N,
        vocab_size).

        targets (torch.LongTensor): Actual targets, of size (N).

        other_logits (torch.FloatTensor, optional): Predicted logits for
        a second predicted variable, if any, of size (N,
        other_vocab_size). Defaults to None.

        other_targets (torch.LongTensor, optional): Actual targets for a
        second predicted variable, if any, of size (N). Defaults to
        None.

        k (list, optional): Values of "k". Defaults to [1, 3, 5].

    Returns:

        list: "Top-k" accuracies.
    """
    with torch.no_grad():
        batch_size = logits.shape[0]
        if other_logits is not None:
            # Get indices corresponding to top-max(k) scores
            probabilities = F.softmax(logits, dim=-1).unsqueeze(2)  # (N, vocab_size, 1)
            other_probabilities = F.softmax(other_logits, dim=-1).unsqueeze(
                1
            )  # (N, 1, other_vocab_size)
            combined_probabilities = torch.bmm(probabilities, other_probabilities).view(
                batch_size, -1
            )  # (N, vocab_size * other_vocab_size)
            _, flattened_indices = combined_probabilities.topk(
                k=max(k), dim=1
            )  # (N, max(k))
            indices = flattened_indices // other_logits.shape[-1]  # (N, max(k))
            other_indices = flattened_indices % other_logits.shape[-1]  # (N, max(k))

            # Expand targets to the same shape
            targets = targets.unsqueeze(1).expand_as(indices)  # (N, max(k))
            other_targets = other_targets.unsqueeze(1).expand_as(
                other_indices
            )  # (N, max(k))

            # Get correct predictions
            correct_predictions = (indices == targets) * (
                other_indices == other_targets
            )  # (N, max(k))

        else:
            # Get indices corresponding to top-max(k) scores
            _, indices = logits.topk(k=max(k), dim=1)  # (N, max(k))

            # Expand targets to the same shape
            targets = targets.unsqueeze(1).expand_as(indices)  # (N, max(k))

            # Get correct predictions
            correct_predictions = indices == targets  # (N, max(k))

        # Calculate top-k accuracies
        topk_accuracies = [
            correct_predictions[:, :k_value].sum().item() / batch_size for k_value in k
        ]

        return topk_accuracies


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if type(val) != float:
            raise TypeError(f"{val=}:{type(val)}")
        if type(n) != int:
            raise TypeError(f"{n=}:{type(n)}")
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def train_model(CONFIG):
    """
    Training and validation.

    Args:

        config (dict): Configuration. See ./configs.
    """


    writer = SummaryWriter(log_dir=config['LOGS_FOLDER'])

    tokenizer = Tokenizer()


    # Initialize data-loaders
    
    data = MATChessDataset(
        tokenizer,
        dataset_path=CONFIG['DATA_FOLDER'],
        n_datapoints=CONFIG['MAX_DATASET_SIZE']
    )
    
    data_len = len(data)
    train_len = int(data_len * 0.8)

    train_valid_split_generater = torch.Generator().manual_seed(42)

    train_data, val_data = random_split(
        data,
        [train_len, data_len - train_len],
        generator=train_valid_split_generater
    )

    train_loader = DataLoader(
        train_data,
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        shuffle=True
    )
    

    ##### Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = MATChessTransformerEncoder(CONFIG)
    model = model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        params=[p for p in model.parameters() if p.requires_grad],
        lr=get_lr(
            step=CONFIG['STEP'],
            d_model=CONFIG['D_MODEL'],
            warmup_steps=CONFIG['WARMUP_STEPS'],
            schedule=CONFIG['LR_SCHEDULE'],
            decay=CONFIG['LR_DECAY'],
            ),
        betas=CONFIG['BETAS'],
        eps=CONFIG['EPSILON'],
    )

    # Load checkpoint if available
    if CONFIG['TRAINING_CHECKPOINT'] is not None:
        checkpoint = torch.load(
            os.path.join(CONFIG['CHECKPOINT_FOLDER'], CONFIG['TRAINING_CHECKPOINT']),
            weights_only=True,
        )
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
    else:
        start_epoch = 0

    # # Loss function
    # criterion = LabelSmoothedCE(
    #     eps=CONFIG['LABEL_SMOOTHING'], n_predictions=CONFIG['N_MOVES']
    # )
    criterion = LabelSmoothedCE(
        eps=CONFIG['LABEL_SMOOTHING'], n_predictions=CONFIG['N_AGENTS']
    )
    criterion = criterion.to(device)
    value_criterion = HuberLoss(
        reduction=CONFIG['HUBER_LOSS_REDUCTION'], delta=CONFIG['VALUE_LOSS_COEF']
    )
    value_criterion = value_criterion.to(device)

    # AMP scaler
    scaler = GradScaler(device=device, enabled=CONFIG['USE_AMP'])

    # Find total epochs to train
    # print(f"CONFIG['N_STEPS']={CONFIG['N_STEPS']}\tlen(train_loader)={len(train_loader)}\tCONFIG['BATCHES_PER_STEP']={CONFIG['BATCHES_PER_STEP']}")
    # print(f"(len(train_loader) / CONFIG['BATCHES_PER_STEP'])={(len(train_loader) / CONFIG['BATCHES_PER_STEP'])}")
    epochs = (CONFIG['N_STEPS'] // (len(train_loader) // CONFIG['BATCHES_PER_STEP'])) + 1



    # Epochs
    for epoch in range(start_epoch, epochs):
        # Step
        step = epoch * len(train_loader) // CONFIG['BATCHES_PER_STEP']

        # One epoch's training
        train_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            value_criterion=value_criterion,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            epochs=epochs,
            step=step,
            writer=writer,
            device=device,
            CONFIG=CONFIG,
        )

        # One epoch's validation
        validate_epoch(
            val_loader=val_loader,
            model=model,
            criterion=criterion,
            value_criterion=value_criterion,
            epoch=epoch,
            writer=writer,
            device=device,
            CONFIG=CONFIG,
        )

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, CONFIG['NAME'], CONFIG['CHECKPOINT_FOLDER'], prefix="checkpoint_epoch_" + str(epoch) + "_")



def train_epoch(
    train_loader,
    model,
    criterion,
    value_criterion,
    optimizer,
    scaler,
    epoch,
    epochs,
    step,
    writer,
    device,
    CONFIG,
):
    """
    One epoch's training.

    Args:

        train_loader (torch.utils.data.DataLoader): Loader for training
        data.

        model (torch.nn.Module): Model.

        criterion (torch.nn.Module): Loss criterion.

        optimizer (torch.optim.adam.Adam): Optimizer.

        scaler (torch.cuda.amp.GradScaler): AMP scaler.

        epoch (int): Epoch number.

        epochs (int): Total number of epochs.

        step (int): Step number.

        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard
        writer.

        CONFIG (dict): Configuration.
    """
    model.train()  # training mode enables dropout

    # Track some metrics
    data_time = AverageMeter()  # data loading time
    step_time = AverageMeter()  # forward prop. + back prop. time
    losses = AverageMeter()  # loss
    top1_accuracies = AverageMeter()  # top-1 accuracy of first move
    top3_accuracies = AverageMeter()  # top-3 accuracy of first move
    top5_accuracies = AverageMeter()  # top-5 accuracy of first move

    top1_accuracies_per_agent = []  # top-1 accuracy of each agent's output move
    top3_accuracies_per_agent = []  # top-3 accuracy of each agent's output move
    top5_accuracies_per_agent = []  # top-5 accuracy of each agent's output move

    for agent in range(CONFIG['N_AGENTS']):
        top1_accuracies_per_agent.append(AverageMeter())
        top3_accuracies_per_agent.append(AverageMeter())
        top5_accuracies_per_agent.append(AverageMeter())


    # Starting time
    start_data_time = time.time()
    start_step_time = time.time()

    # Track Policy loss and value Loss:
    losses_policy = AverageMeter()
    losses_value = AverageMeter()


    # Keep track of the rewards during training:
    reward_per_agent_individual_survival = []
    reward_per_agent_individual_mobility = []
    reward_per_agent_individual_aggressiveness = []
    reward_per_agent_piece_solidarity = []
    reward_per_agent_individual_defensiveness = []
    reward_per_agent_team_goal = []
    reward_per_agent_group_survival = []
    reward_per_agent_society_aggresiveness = []
    reward_per_agent_patience = []
    reward_per_agent_group_mobility = []
    reward_per_agent_society_defensiveness = []

    for agent in range(CONFIG['N_AGENTS']):
        reward_per_agent_individual_survival.append(AverageMeter())
        reward_per_agent_individual_mobility.append(AverageMeter())
        reward_per_agent_individual_aggressiveness.append(AverageMeter())
        reward_per_agent_piece_solidarity.append(AverageMeter())
        reward_per_agent_individual_defensiveness.append(AverageMeter())
        reward_per_agent_team_goal.append(AverageMeter())
        reward_per_agent_group_survival.append(AverageMeter())
        reward_per_agent_society_aggresiveness.append(AverageMeter())
        reward_per_agent_patience.append(AverageMeter())
        reward_per_agent_group_mobility.append(AverageMeter())
        reward_per_agent_society_defensiveness.append(AverageMeter())


    # Batches
    # for i, batch in enumerate(train_loader):
    for i, batch in tqdm(
            enumerate(train_loader), desc=f"Training epoch: {epoch}/{epochs}", total=len(train_loader)
        ):
        # Move to default device
        for key in batch:
            batch[key] = batch[key].to(device)

        # Time taken to load data
        data_time.update(time.time() - start_data_time)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=CONFIG['USE_AMP']
        ):
            # (Direct) Move prediction models
            if CONFIG['NAME'].startswith(("MATChessFormer-Homogeneous-")):
                # Forward prop.
                predicted_moves = model(batch)  # (N, n_agents, move_vocab_size)

                # Loss
                loss = criterion(
                    predicted=predicted_moves,  # (N, n_agents, move_vocab_size)
                    targets=batch["moves"],  # (N, n_agents)
                    lengths=batch["n_agents"].view(-1,1),  # (N, n_predictions_per_agent)
                )  # scalar
                # print(f"\nloss={loss}\n")

            elif CONFIG['NAME'].startswith(("MATChessFormer-Heterogeneous-")):
                # Forward prop.
                predicted_moves, predicted_rewards = model(
                    batch
                )   # (N, n_agents, move_vocab_size), (N,n_agents, n_rewards_per_agent)

                policy_loss = criterion(
                    predicted=predicted_moves,
                    targets=batch["moves"],
                    lengths= batch["n_agents"].view(-1,1),  # (N, n_predictions_per_agent)
                )
                value_loss = value_criterion(
                    input=predicted_rewards,
                    target=batch["rewards"],  # (N, n_rewards_per_agent)
                )
                # print(value_loss.shape)

                loss = policy_loss + (CONFIG['VALUE_LOSS_COEF'] * value_loss) #value_criterion(predicted_rewards, batch["agents_rewards"]) #np.mean(value_loss)

            # Other models
            else:
                raise NotImplementedError

            loss = loss / CONFIG['BATCHES_PER_STEP']

        # Backward prop.
        scaler.scale(loss).backward()

        # # Keep track of losses
        # losses.update(
        #     loss.item() * CONFIG['BATCHES_PER_STEP'], batch["lengths"].sum().item()
        # )

        # Keep track of losses
        losses.update(
            loss.item() * CONFIG['BATCHES_PER_STEP'], batch["n_agents"].sum().item()
        )

        # # Keep track of accuracy (Direct) Move prediction models
        # if CONFIG['NAME'].startswith(("CT-ED-", "CT-E-")):
        #     top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
        #         logits=predicted_moves[:, 0, :],  # (N, move_vocab_size)
        #         targets=batch["moves"][:, 1],  # (N)
        #         k=[1, 3, 5],
        #     )
        # elif CONFIG['NAME'].startswith(("CT-EFT-")):
        #     top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
        #         logits=predicted_from_squares[:, 0, :],  # (N, 64)
        #         targets=batch["from_squares"].squeeze(1),  # (N)
        #         other_logits=predicted_to_squares[:, 0, :],  # (N, 64)
        #         other_targets=batch["to_squares"].squeeze(1),  # (N)
        #         k=[1, 3, 5],
        #     )
        
        # Keep track of accuracy (Direct) Move prediction models
        if CONFIG['NAME'].startswith(("MATChessFormer-Homogeneous-")):
            top_k_accuracies = []
            top_k_train_batch_accuracy = [0,0,0]
            for agent_idx in range(CONFIG['N_AGENTS']):
                top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predicted_moves[:, agent_idx, :],  # (N, 1, move_vocab_size)
                    targets=batch["moves"][:, 1],  # (N)
                    k=[1, 3, 5],
                )
                top_k_accuracies.append([top1_accuracy, top3_accuracy, top5_accuracy])
                # top1_accuracies_per_agent[agent_idx].update(top1_accuracy, CONFIG['BATCH_SIZE'])
                # top3_accuracies_per_agent[agent_idx].update(top3_accuracy, CONFIG['BATCH_SIZE'])
                # top5_accuracies_per_agent[agent_idx].update(top5_accuracy, CONFIG['BATCH_SIZE'])

                top_k_train_batch_accuracy[0] += top1_accuracy
                top_k_train_batch_accuracy[1] += top3_accuracy
                top_k_train_batch_accuracy[2] += top5_accuracy

        elif CONFIG['NAME'].startswith(("MATChessFormer-Heterogeneous-")):
            # Keep track of policy and value losses
            losses_policy.update(policy_loss.item(), batch["n_agents"].sum().item())
            losses_value.update(value_loss.item(), batch["n_agents"].sum().item())
            
            top_k_accuracies = []
            top_k_train_batch_accuracy = [0,0,0]
            for agent_idx in range(CONFIG['N_AGENTS']):
                top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                    logits=predicted_moves[:, agent_idx, :],  # (N, 1, move_vocab_size)
                    targets=batch["moves"][:, 1],  # (N)
                    k=[1, 3, 5],
                )
                top_k_accuracies.append([top1_accuracy, top3_accuracy, top5_accuracy])
                # top1_accuracies_per_agent[agent_idx].update(top1_accuracy, CONFIG['BATCH_SIZE'])
                # top3_accuracies_per_agent[agent_idx].update(top3_accuracy, CONFIG['BATCH_SIZE'])
                # top5_accuracies_per_agent[agent_idx].update(top5_accuracy, CONFIG['BATCH_SIZE'])

                top_k_train_batch_accuracy[0] += top1_accuracy
                top_k_train_batch_accuracy[1] += top3_accuracy
                top_k_train_batch_accuracy[2] += top5_accuracy
        else:
            raise NotImplementedError
        
        
        top1_accuracies.update(top_k_train_batch_accuracy[0] / CONFIG['N_AGENTS'], CONFIG['BATCH_SIZE'])
        top3_accuracies.update(top_k_train_batch_accuracy[1] / CONFIG['N_AGENTS'], CONFIG['BATCH_SIZE'])
        top5_accuracies.update(top_k_train_batch_accuracy[2] / CONFIG['N_AGENTS'], CONFIG['BATCH_SIZE'])

        for agent_idx in range(CONFIG['N_AGENTS']):
            top1_accuracies_per_agent[agent_idx].update(top_k_accuracies[agent_idx][0], CONFIG['BATCH_SIZE'])
            top3_accuracies_per_agent[agent_idx].update(top_k_accuracies[agent_idx][1], CONFIG['BATCH_SIZE'])
            top5_accuracies_per_agent[agent_idx].update(top_k_accuracies[agent_idx][2], CONFIG['BATCH_SIZE'])

            # top1_accuracies.update(top_k_accuracies[agent_idx][0], CONFIG['BATCH_SIZE'])
            # top3_accuracies.update(top_k_accuracies[agent_idx][1], CONFIG['BATCH_SIZE'])
            # top5_accuracies.update(top_k_accuracies[agent_idx][2], CONFIG['BATCH_SIZE'])

        # TODO: Update the 11 rewards per agent and accout for batch_size
        # batch['rewards'] = (N, )
        for agent_idx in range(CONFIG['N_AGENTS']):
            reward_per_agent_individual_survival[agent_idx].update(torch.sum(batch['rewards'][agent_idx][0]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_individual_mobility[agent_idx].update(torch.sum(batch['rewards'][agent_idx][1]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_individual_aggressiveness[agent_idx].update(torch.sum(batch['rewards'][agent_idx][2]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_piece_solidarity[agent_idx].update(torch.sum(batch['rewards'][agent_idx][3]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_individual_defensiveness[agent_idx].update(torch.sum(batch['rewards'][agent_idx][4]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_team_goal[agent_idx].update(torch.sum(batch['rewards'][agent_idx][5]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_group_survival[agent_idx].update(torch.sum(batch['rewards'][agent_idx][6]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_society_aggresiveness[agent_idx].update(torch.sum(batch['rewards'][agent_idx][7]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_patience[agent_idx].update(torch.sum(batch['rewards'][agent_idx][8]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_group_mobility[agent_idx].update(torch.sum(batch['rewards'][agent_idx][9]).item(), CONFIG['BATCH_SIZE'])
            reward_per_agent_society_defensiveness[agent_idx].update(torch.sum(batch['rewards'][agent_idx][10]).item(), CONFIG['BATCH_SIZE'])


        # Update model (i.e. perform a training step) only after
        # gradients are accumulated from batches_per_step batches
        if (i + 1) % CONFIG['BATCHES_PER_STEP'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # This step is now complete
            step += 1

            # Update learning rate after each step
            change_lr(
                optimizer,
                new_lr=get_lr(
                    step=step,
                    d_model=CONFIG['D_MODEL'],
                    warmup_steps=CONFIG['WARMUP_STEPS'],
                    schedule=CONFIG['LR_SCHEDULE'],
                    decay=CONFIG['LR_DECAY'],
                ),
            )

            # Time taken for this training step
            step_time.update(time.time() - start_step_time)

            # Print status
            if CONFIG['PRINT_FREQUENCY'] is not None:
                if step % CONFIG['PRINT_FREQUENCY'] == 0:
                    print(
                        "Epoch {0}/{1}---"
                        "Batch {2}/{3}---"
                        "Step {4}/{5}---"
                        "Data Time {data_time.val:.3f} ({data_time.avg:.3f})---"
                        "Step Time {step_time.val:.3f} ({step_time.avg:.3f})---"
                        "Loss {losses.val:.4f} ({losses.avg:.4f})---"
                        "Top-5 {top5s.val:.4f} ({top5s.avg:.4f})".format(
                            epoch + 1,
                            epochs,
                            i + 1,
                            len(train_loader),
                            step,
                            CONFIG['N_STEPS'],
                            step_time=step_time,
                            data_time=data_time,
                            losses=losses,
                            top5s=top5_accuracies,
                        )
                    )

            # Log to tensorboard
            writer.add_scalar(
                tag="train/loss", scalar_value=losses.val, global_step=step
            )
            
            writer.add_scalar(
                tag="train/loss_policy", scalar_value=losses_policy.val, global_step=step
            )
            writer.add_scalar(
                tag="train/loss_value", scalar_value=losses_value.val, global_step=step
            )

            writer.add_scalar(
                tag="train/lr",
                scalar_value=optimizer.param_groups[0]["lr"],
                global_step=step,
            )
            writer.add_scalar(
                tag="train/data_time", scalar_value=data_time.val, global_step=step
            )
            writer.add_scalar(
                tag="train/step_time", scalar_value=step_time.val, global_step=step
            )
            writer.add_scalar(
                tag="train/top1_accuracy",
                scalar_value=top1_accuracies.val,
                global_step=step,
            )
            writer.add_scalar(
                tag="train/top3_accuracy",
                scalar_value=top3_accuracies.val,
                global_step=step,
            )
            writer.add_scalar(
                tag="train/top5_accuracy",
                scalar_value=top5_accuracies.val,
                global_step=step,
            )

            #################################################
            #### Creates a lot of .tf files in seperate sub-directories, but creates 3 graphs on tensorboard with all the top-1, top-3, and top-5 accuracies for all agents, respectively.
            reward_per_agent_dict = {
                'SURVIVAL_OF_THE_AGENT': {},
                'AGENT_CONTRIBUTION': {},
                'INDIVIDUAL_AGGRESIVENESS': {},
                'PIECE_TYPE_SOLIDARITY': {},
                'INDIVIDUAL_DEFENSIVENESS': {},
                'TEAM_GOAL': {},
                'TEAM_SURVIVAL': {},
                'SOCIETY_AGGRESIVENESS': {},
                'SOCIETY_PATIENCE': {},
                'GROUP_MOBILITY': {},
                'SOCIETY_DEFENSIVENESS': {}
            }
            for key, value in CHESS_PIECE_AGENTS.items():
                reward_per_agent_dict['SURVIVAL_OF_THE_AGENT'][key] = reward_per_agent_individual_survival[value].val
                reward_per_agent_dict['AGENT_CONTRIBUTION'][key] = reward_per_agent_individual_mobility[value].val
                reward_per_agent_dict['INDIVIDUAL_AGGRESIVENESS'][key] = reward_per_agent_individual_aggressiveness[value].val 
                reward_per_agent_dict['PIECE_TYPE_SOLIDARITY'][key] = reward_per_agent_piece_solidarity[value].val
                reward_per_agent_dict['INDIVIDUAL_DEFENSIVENESS'][key] = reward_per_agent_individual_defensiveness[value].val
                reward_per_agent_dict['TEAM_GOAL'][key] = reward_per_agent_team_goal[value].val
                reward_per_agent_dict['TEAM_SURVIVAL'][key] = reward_per_agent_group_survival[value].val
                reward_per_agent_dict['SOCIETY_AGGRESIVENESS'][key] = reward_per_agent_society_aggresiveness[value].val
                reward_per_agent_dict['SOCIETY_PATIENCE'][key] = reward_per_agent_patience[value].val
                reward_per_agent_dict['GROUP_MOBILITY'][key] = reward_per_agent_group_mobility[value].val
                reward_per_agent_dict['SOCIETY_DEFENSIVENESS'][key] = reward_per_agent_society_defensiveness[value].val

            # for reward_key, reward_item in reward_per_agent_dict.items():
            #     writer.add_scalars(
            #         main_tag="train_reward_" + str(reward_key) + "/",
            #         tag_scalar_dict= reward_item, #reward_per_agent_dict[reward_key],
            #         global_step=step
            #     )
            writer.add_scalars(
                main_tag="train_reward/SURVIVAL_OF_THE_AGENT/",
                tag_scalar_dict= reward_per_agent_dict['SURVIVAL_OF_THE_AGENT'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/AGENT_CONTRIBUTION/",
                tag_scalar_dict= reward_per_agent_dict['AGENT_CONTRIBUTION'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/INDIVIDUAL_AGGRESIVENESS/",
                tag_scalar_dict= reward_per_agent_dict['INDIVIDUAL_AGGRESIVENESS'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/PIECE_TYPE_SOLIDARITY/",
                tag_scalar_dict= reward_per_agent_dict['PIECE_TYPE_SOLIDARITY'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/INDIVIDUAL_DEFENSIVENESS/",
                tag_scalar_dict= reward_per_agent_dict['INDIVIDUAL_DEFENSIVENESS'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/TEAM_GOAL/",
                tag_scalar_dict= reward_per_agent_dict['TEAM_GOAL'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/TEAM_SURVIVAL/",
                tag_scalar_dict= reward_per_agent_dict['TEAM_SURVIVAL'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/SOCIETY_AGGRESIVENESS/",
                tag_scalar_dict= reward_per_agent_dict['SOCIETY_AGGRESIVENESS'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/SOCIETY_PATIENCE/",
                tag_scalar_dict= reward_per_agent_dict['SOCIETY_PATIENCE'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/GROUP_MOBILITY/",
                tag_scalar_dict= reward_per_agent_dict['GROUP_MOBILITY'],
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_reward/SOCIETY_DEFENSIVENESS/",
                tag_scalar_dict= reward_per_agent_dict['SOCIETY_DEFENSIVENESS'],
                global_step=step
            )
            #################################################

            
            #################################################
            #### Creates a lot of .tf files in seperate sub-directories, but creates 3 graphs on tensorboard with all the top-1, top-3, and top-5 accuracies for all agents, respectively.
            top1_accuracy_per_agent_dict = dict()
            top3_accuracy_per_agent_dict = dict()
            top5_accuracy_per_agent_dict = dict()
            for key, value in CHESS_PIECE_AGENTS.items():
                top1_accuracy_per_agent_dict[key] = top1_accuracies_per_agent[value].val
                top3_accuracy_per_agent_dict[key] = top3_accuracies_per_agent[value].val
                top5_accuracy_per_agent_dict[key] = top5_accuracies_per_agent[value].val

            writer.add_scalars(
                main_tag="train_per_agent/top1_accuracy",
                tag_scalar_dict=top1_accuracy_per_agent_dict,
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_per_agent/top3_accuracy",
                tag_scalar_dict=top3_accuracy_per_agent_dict,
                global_step=step
            )
            writer.add_scalars(
                main_tag="train_per_agent/top5_accuracy",
                tag_scalar_dict=top5_accuracy_per_agent_dict,
                global_step=step
            )
            #################################################


            # for key, value in CHESS_PIECE_AGENTS.items():
            #     writer.add_scalar(
            #         tag="train_per_agent_top1_accuracy/" + str(key),
            #         scalar_value=top1_accuracies_per_agent[value].val,
            #         global_step=i
            #     )
            #     writer.add_scalar(
            #         tag="train_per_agent_top3_accuracy/" + str(key),
            #         scalar_value=top1_accuracies_per_agent[value].val,
            #         global_step=i
            #     )
            #     writer.add_scalar(
            #         tag="train_per_agent_top5_accuracy/" + str(key),
            #         scalar_value=top1_accuracies_per_agent[value].val,
            #         global_step=i
            #     )

            
            # for key, value in CHESS_PIECE_AGENTS.items():
            #     writer.add_scalar(
            #         tag="train/top1_accuracy" + key,
            #         scalar_value=top5_accuracies.val,
            #         global_step=step,
            #     )
            #     # top1_accuracy_per_agent_dict[key] = top1_accuracies_per_agent[value].val
            #     # top3_accuracy_per_agent_dict[key] = top3_accuracies_per_agent[value].val
            #     # top5_accuracy_per_agent_dict[key] = top5_accuracies_per_agent[value].val
            

            # train_losses.append(loss.detach().cpu().numpy())
            # writer.add_scalar(
            #     tag="train/train_loss", scalar_value=np.mean(train_losses), global_step=step
            # )

            # Reset step time
            start_step_time = time.time()

            # If this step is marked for saving a checkpoint for averaging, save checkpoint
            if step in CONFIG['AVERAGE_STEPS']:
                save_checkpoint(
                    epoch,
                    model,
                    optimizer,
                    CONFIG['NAME'],
                    CONFIG['CHECKPOINT_FOLDER'],
                    prefix="step" + str(step) + "_",
                )

        # print("\nAverage training loss: %.3f" % losses.avg)
        # print("Training top-1 accuracy: %.3f" % top1_accuracies.avg)
        # print("Training top-3 accuracy: %.3f" % top3_accuracies.avg)
        # print("Training top-5 accuracy: %.3f\n" % top5_accuracies.avg)
        
        # Reset data time
        start_data_time = time.time()

        # if CONFIG['SAVE_CHECKPOINT_EPOCH_FEQUENCY'] is not None:
        #     if epoch % CONFIG['SAVE_CHECKPOINT_EPOCH_FEQUENCY'] == 0:
        #         save_checkpoint(
        #             epoch,
        #             model,
        #             optimizer,
        #             CONFIG['NAME'],
        #             CONFIG['CHECKPOINT_FOLDER'],
        #             prefix="step" + str(step) + "_",
        #         )

    
    writer.add_scalar(
        tag="train_epoch/avg_loss",
        scalar_value=losses.avg,
        global_step=epoch + 1,
    )

    writer.add_scalar(
        tag="train_epoch/avg_loss_policy",
        scalar_value=losses_policy.avg,
        global_step=epoch + 1,
    )
    writer.add_scalar(
        tag="train_epoch/avg_loss_value",
        scalar_value=losses_value.avg,
        global_step=epoch + 1,
    )


    writer.add_scalar(
        tag="train_epoch/avg_accuracy_top5",
        scalar_value=top5_accuracies.avg,
        global_step=epoch + 1,
    )
    writer.add_scalar(
        tag="train_epoch/avg_accuracy_top3",
        scalar_value=top3_accuracies.avg,
        global_step=epoch + 1,
    )
    writer.add_scalar(
        tag="train_epoch/avg_accuracy_top1",
        scalar_value=top1_accuracies.avg,
        global_step=epoch + 1,
    )
    # writer.add_scalar(
    #     tag="train_epoch/train_loss",
    #     scalar_value=np.mean(train_losses),
    #     global_step=epoch + 1
    # )

    ##################################################
    ##### Creates a lot of .tf files in seperate sub-directories, but creates 3 graphs on tensorboard with all the top-1, top-3, and top-5 accuracies for all agents, respectively.
    epoch_top1_accuracy_per_agent_dict = dict()
    epoch_top3_accuracy_per_agent_dict = dict()
    epoch_top5_accuracy_per_agent_dict = dict()
    for key, value in CHESS_PIECE_AGENTS.items():
        epoch_top1_accuracy_per_agent_dict[key] = top1_accuracies_per_agent[value].avg
        epoch_top3_accuracy_per_agent_dict[key] = top3_accuracies_per_agent[value].avg
        epoch_top5_accuracy_per_agent_dict[key] = top5_accuracies_per_agent[value].avg

    writer.add_scalars(
        main_tag="train_epoch_per_agent/top1_accuracy",
        tag_scalar_dict=epoch_top1_accuracy_per_agent_dict,
        global_step=epoch + 1
    )
    writer.add_scalars(
        main_tag="train_epoch_per_agent/top3_accuracy",
        tag_scalar_dict=epoch_top3_accuracy_per_agent_dict,
        global_step=epoch + 1
    )
    writer.add_scalars(
        main_tag="train_epoch_per_agent/top5_accuracy",
        tag_scalar_dict=epoch_top5_accuracy_per_agent_dict,
        global_step=epoch + 1
    )
    ##################################################
    
    print(
            "Training Epoch {0}/{1}---"
            "Batch {2}/{3}---"
            "Step {4}/{5}---"
            "Avg Load Data Time {data_time.avg:.3f}---"
            "Avg Step Time {step_time.avg:.3f}---"
            "Avg Loss {losses.avg:.4f}---"
            "\nAccuracies: Avg Top-5={top5s.avg:.4f}---"
            "Avg Top-3 Accuracy={top3s.avg:.4f}---"
            "Avg Top-1 Accuracy={top1s.avg:.4f}".format(
                epoch + 1,
                epochs,
                i + 1,
                len(train_loader),
                step,
                CONFIG['N_STEPS'],
                step_time=step_time,
                data_time=data_time,
                losses=losses,
                top5s=top5_accuracies,
                top3s=top3_accuracies,
                top1s=top1_accuracies,
            )
        )


def validate_epoch(val_loader, model, criterion, value_criterion, epoch, writer, device, CONFIG):
    """
    One epoch's validation.

    Args:

        val_loader (torch.utils.data.DataLoader): Loader for validation
        data

        model (torch.nn.Module): Model

        criterion (torch.nn.Module): Loss criterion.

        value_criterion (torch.nn.Module): Value Loss criterion

        epoch (int): Epoch number.

        writer (torch.utils.tensorboard.SummaryWriter): TensorBoard
        writer.

        CONFIG (dict): Configuration.
    """
    print("\n")
    model.eval()  # eval mode disables dropout

    # valid_losses = []

    # Prohibit gradient computation explicitly
    with torch.no_grad():
        losses = AverageMeter()
        top1_accuracies = AverageMeter()  # top-1 accuracy of first move
        top3_accuracies = AverageMeter()  # top-3 accuracy of first move
        top5_accuracies = AverageMeter()  # top-5 accuracy of first move

        top1_accuracies_per_agent = []  # top-1 accuracy of each agent's output move
        top3_accuracies_per_agent = []  # top-3 accuracy of each agent's output move
        top5_accuracies_per_agent = []  # top-5 accuracy of each agent's output move
        for agent in range(CONFIG['N_AGENTS']):
            top1_accuracies_per_agent.append(AverageMeter())
            top3_accuracies_per_agent.append(AverageMeter())
            top5_accuracies_per_agent.append(AverageMeter())
        
        # Track Policy loss and value Loss:
        losses_policy = AverageMeter()
        losses_value = AverageMeter()

        # Batches
        for i, batch in tqdm(
            enumerate(val_loader), desc="Validating", total=len(val_loader)
        ):
            # Move to default device
            for key in batch:
                batch[key] = batch[key].to(device)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=CONFIG['USE_AMP']
            ):
                # (Direct) Move prediction models
                if CONFIG['NAME'].startswith(("MATChessFormer-Homogeneous-")):
                    # Forward prop.
                    predicted_moves = model(batch)  # (N, n_agents, move_vocab_size)

                    loss = criterion(
                        predicted=predicted_moves,  # (N, n_agents, move_vocab_size)
                        targets=batch["moves"],     # (N, n_agents)
                        lengths=batch["n_agents"].view(-1,1),  # (N, n_predictions_per_agent)=(N,1)
                    )  # scalar

                # Imitation learning with next state reward prediction as  auxiliary targets:
                elif CONFIG['NAME'].startswith(("MATChessFormer-Heterogeneous-")):
                    # Forward prop.
                    predicted_moves, predicted_rewards = model(
                        batch
                    )  # (N, 1, 64), (N, 1, 64)

                    policy_loss = criterion(
                        predicted=predicted_moves,
                        targets=batch["moves"],
                        lengths= batch["n_agents"].view(-1,1),
                    )
                    # pred_reward_errors = predicted_rewards - batch["agents_rewards"]
                    # value_loss = huber_loss(pred_reward_errors, CONFIG['HUBER_DELTA']
                    # )  # SHOULD BE: scalar
                    value_loss = value_criterion(
                        input=predicted_rewards,
                        target=batch["rewards"],  # (N, n_rewards_per_agent)
                    )

                    loss = policy_loss + CONFIG['VALUE_LOSS_COEF'] * value_loss #value_criterion(predicted_rewards, batch["agents_rewards"]) # np.mean(value_loss) # scalar
                # Other models
                else:
                    raise NotImplementedError

            # Keep track of losses
            losses.update(loss.item(), batch["n_agents"].sum().item())


            # Keep track of accuracy of move prediction:
            if CONFIG['NAME'].startswith(("MATChessFormer-Homogeneous-")):
                top_k_accuracies = []
                top_k_val_batch_accuracy = [0,0,0]
                for agent_idx in range(CONFIG['N_AGENTS']):
                    top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                        logits=predicted_moves[:, agent_idx, :],  # (N, 1, move_vocab_size)
                        targets=batch["moves"][:, 1],  # (N)
                        k=[1, 3, 5],
                    )
                    top_k_accuracies.append([top1_accuracy, top3_accuracy, top5_accuracy])
                    # top1_accuracies_per_agent[agent_idx].update(top1_accuracy, CONFIG['BATCH_SIZE'])
                    # top3_accuracies_per_agent[agent_idx].update(top3_accuracy, CONFIG['BATCH_SIZE'])
                    # top5_accuracies_per_agent[agent_idx].update(top5_accuracy, CONFIG['BATCH_SIZE'])

                    top_k_val_batch_accuracy[0] += top1_accuracy
                    top_k_val_batch_accuracy[1] += top3_accuracy
                    top_k_val_batch_accuracy[2] += top5_accuracy
            elif CONFIG['NAME'].startswith(("MATChessFormer-Heterogeneous-")):
                # Keep track of policy and value losses
                losses_policy.update(policy_loss.item(), batch["n_agents"].sum().item())
                losses_value.update(value_loss.item(), batch["n_agents"].sum().item())

                top_k_accuracies = []
                top_k_val_batch_accuracy = [0,0,0]
                for agent_idx in range(CONFIG['N_AGENTS']):
                    top1_accuracy, top3_accuracy, top5_accuracy = topk_accuracy(
                        logits=predicted_moves[:, agent_idx, :],  # (N, 1, move_vocab_size)
                        targets=batch["moves"][:, 1],  # (N)
                        k=[1, 3, 5],
                    )
                    top_k_accuracies.append([top1_accuracy, top3_accuracy, top5_accuracy])

                    top_k_val_batch_accuracy[0] += top1_accuracy
                    top_k_val_batch_accuracy[1] += top3_accuracy
                    top_k_val_batch_accuracy[2] += top5_accuracy
            else:
                raise NotImplementedError
            
            top1_accuracies.update(top_k_val_batch_accuracy[0] / CONFIG['N_AGENTS'], CONFIG['BATCH_SIZE'])
            top3_accuracies.update(top_k_val_batch_accuracy[1] / CONFIG['N_AGENTS'], CONFIG['BATCH_SIZE'])
            top5_accuracies.update(top_k_val_batch_accuracy[2] / CONFIG['N_AGENTS'], CONFIG['BATCH_SIZE'])
            
            for agent_idx in range(CONFIG['N_AGENTS']):
                top1_accuracies_per_agent[agent_idx].update(top_k_accuracies[agent_idx][0], CONFIG['BATCH_SIZE'])
                top3_accuracies_per_agent[agent_idx].update(top_k_accuracies[agent_idx][1], CONFIG['BATCH_SIZE'])
                top5_accuracies_per_agent[agent_idx].update(top_k_accuracies[agent_idx][2], CONFIG['BATCH_SIZE'])


        # Log to tensorboard
        writer.add_scalar(
            tag="val/loss", scalar_value=losses.avg, global_step=epoch + 1
        )
        writer.add_scalar(
            tag="val/loss_policy", scalar_value=losses_policy.avg, global_step=epoch + 1
        )
        writer.add_scalar(
            tag="val/loss_value", scalar_value=losses_value.avg, global_step=epoch +1
        )
    
        writer.add_scalar(
            tag="val/top1_accuracy",
            scalar_value=top1_accuracies.avg,
            global_step=epoch + 1,
        )
        writer.add_scalar(
            tag="val/top3_accuracy",
            scalar_value=top3_accuracies.avg,
            global_step=epoch + 1,
        )
        writer.add_scalar(
            tag="val/top5_accuracy",
            scalar_value=top5_accuracies.avg,
            global_step=epoch + 1,
        )
        
        ##################################################
        ##### Creates a lot of .tf files in seperate sub-directories, but creates 3 graphs on tensorboard with all the top-1, top-3, and top-5 accuracies for all agents, respectively.
        top1_accuracy_per_agent_dict = dict()
        top3_accuracy_per_agent_dict = dict()
        top5_accuracy_per_agent_dict = dict()
        for key, value in CHESS_PIECE_AGENTS.items():
            top1_accuracy_per_agent_dict[key] = top1_accuracies_per_agent[value].avg
            top3_accuracy_per_agent_dict[key] = top3_accuracies_per_agent[value].avg
            top5_accuracy_per_agent_dict[key] = top5_accuracies_per_agent[value].avg

        writer.add_scalars(
            main_tag="val_per_agent/top1_accuracy",
            tag_scalar_dict=top1_accuracy_per_agent_dict,
            global_step=epoch + 1
        )
        writer.add_scalars(
            main_tag="val_per_agent/top3_accuracy",
            tag_scalar_dict=top3_accuracy_per_agent_dict,
            global_step=epoch + 1
        )
        writer.add_scalars(
            main_tag="val_per_agent/top5_accuracy",
            tag_scalar_dict=top5_accuracy_per_agent_dict,
            global_step=epoch + 1
        )
        ##################################################

        # for key, value in CHESS_PIECE_AGENTS.items():
        #     writer.add_scalar(
        #         tag="val_per_agent_top1_accuracy/" + str(key),
        #         scalar_value=top1_accuracies_per_agent[value].avg,
        #         global_step=epoch + 1
        #     )
        #     writer.add_scalar(
        #         tag="val_per_agent_top3_accuracy/" + str(key),
        #         scalar_value=top1_accuracies_per_agent[value].avg,
        #         global_step=epoch + 1
        #     )
        #     writer.add_scalar(
        #         tag="val_per_agent_top5_accuracy/" + str(key),
        #         scalar_value=top1_accuracies_per_agent[value].avg,
        #         global_step=epoch + 1
        #     )

        print("\nValidation loss total: %.4f" % losses.avg)
        print("Validation loss policy: %.4f" % losses_policy.avg)
        print("Validation loss value: %.4f" % losses_value.avg)
        print("\nValidation top-1 accuracy: %.4f" % top1_accuracies.avg)
        print("Validation top-3 accuracy: %.4f" % top3_accuracies.avg)
        print("Validation top-5 accuracy: %.4f\n" % top5_accuracies.avg)


if __name__ == "__main__":
    # Get configuration
    # config = import_config(model_config_name="MATChessFormer-Heterogeneous-20", run_number=2)

    config = import_config(model_config_name="MATChessFormer-Heterogeneous-20-hubersum", run_number=3)

    # Train model
    train_model(config)
