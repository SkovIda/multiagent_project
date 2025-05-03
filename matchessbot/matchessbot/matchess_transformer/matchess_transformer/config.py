import yaml

# from .vocab_uci_dicts import get_vocab_sizes
from .vocab_uci_dicts import get_vocab_sizes

def import_config(model_config_name: str="MATChessFormer-20", run_number: int=0):
    if model_config_name=="MATChessFormer-Homogeneous-20":
        return {
            "NAME": model_config_name,
            "VOCAB_SIZES": get_vocab_sizes(),
            "D_MODEL": 512,
            "N_HEADS": 8,
            "D_QUERIES": 64,
            "D_VALUES": 64,
            "D_INNER": 2048,
            "N_LAYERS": 6,
            "DROPOUT": 0.1, 
            "N_MOVES": 1,           # expected maximum length of move sequences in the model, <= MAX_MOVE_SEQUENCE_LENGTH
            "N_AGENTS": 16,
            "BATCH_SIZE": 8, #128,
            "NUM_WORKERS": 0, #2,       # number of workers to use for dataloading
            "RUN_NUMBER": run_number,
            "CHECKPOINT_FOLDER": "training_runs/" + model_config_name + "/model_checkpoints/run_" + str(run_number) + "/",
            "DATA_FOLDER": "dataset/gen_dataset_test.json",
            #"H5_FILE": "LE22ct.h5",
            "LOGS_FOLDER": "training_runs/" + model_config_name + "/logs/run_" + str(run_number),
            "MAX_MOVE_SEQUENCE_LENGTH": 1,
            "PREFETCH_FACTOR": None,   # number of batches to prefetch per worker
            "PIN_MEMORY": False,    # pin to GPU memory when dataloading?"
            "SAMPLING_K": 1,        # k in top-k sampling model predictions during play
            "PRINT_FREQUENCY": None, # 50, # 1, # print status once every so many steps
            "N_STEPS": 100000,      # number of training steps
            "WARMUP_STEPS": 4000,   # 800,   # number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official transformer repo.
            "STEP": 1,              # the step number, start from 1 to prevent math error in the 'LR' line
            "LR_SCHEDULE": "fixed", #"vaswani",  # the learning rate schedule; see utils.py for learning rate schedule
            "LR_DECAY": None,       # the decay rate for 'exp_decay' schedule
            "START_EPOCH": 0,       # start at this epoch
            "BETAS": (0.9, 0.98),   # beta coefficients in the Adam optimizer
            "EPSILON": 1e-9,  # epsilon term in the Adam optimizer
            "LABEL_SMOOTHING": 0.1, # label smoothing co-efficient in the Cross Entropy loss
            "BOARD_STATUS_LENGTH": 84,  # total length of input sequence
            "USE_AMP": True,        # use automatic mixed precision training?
            #"CRITERION" = LabelSmoothedCE  # training criterion (loss)
            #"OPTIMIZER": "", #torch.optim.Adam  # optimizer
            "BATCHES_PER_STEP": (64), #(16),# perform a training step, i.e. update parameters, once every so many batches
            "TRAINING_CHECKPOINT": None, # path to model checkpoint (NAME + ".pt") to resume training, None if none
            "CHECKPOINT_AVG_PREFIX": "step",
            "CHECKPOINT_AVG_SUFFIX": ".pt",  # checkpoint end string to match checkpoints saved for averaging
            "EVAL_GAMES_FOLDER": "training_runs/" + model_config_name + "/evaluate_games/run_" + str(run_number),  # folder where evaluation games are saved in PGN files
            "FINAL_CHECKPOINT": "averaged_" + model_config_name + "_run_" + str(run_number) + ".pt", # final checkpoint to be used for eval/inference
            "AVERAGE_STEPS": {9000, 9250, 9550, 9700, 9850, 10000}, #{491000, 492500, 494000, 495500, 497000, 498500, 500000}
            "SAVE_CHECKPOINT_EPOCH_FEQUENCY": 1,
            # "MODEL_CHECKPOINT_FOLDER": "training_runs/" + model_config_name + "/model_epoch_checkpoints/"
            "MAX_DATASET_SIZE": 10000 #None #1000
            }
    elif model_config_name=="MATChessFormer-Heterogeneous-20":
        return {
            "NAME": model_config_name,
            "VOCAB_SIZES": get_vocab_sizes(),
            "D_MODEL": 512,
            "N_HEADS": 8,
            "D_QUERIES": 64,
            "D_VALUES": 64,
            "D_INNER": 2048,
            "N_LAYERS": 6,
            "DROPOUT": 0.1, 
            "N_MOVES": 1,           # expected maximum length of move sequences in the model, <= MAX_MOVE_SEQUENCE_LENGTH
            "BATCH_SIZE": 128,
            "NUM_WORKERS": 2,       # number of workers to use for dataloading
            "RUN_NUMBER": run_number,
            "CHECKPOINT_FOLDER": "training_runs/" + model_config_name + "/model_checkpoints/run_" + str(run_number) + "/",
            "DATA_FOLDER": "dataset/gen_dataset_test.json",
            #"H5_FILE": "LE22ct.h5",
            "LOGS_FOLDER": "training_runs/" + model_config_name + "/logs/run_" + str(run_number),
            "MAX_MOVE_SEQUENCE_LENGTH": 10,
            "PREFETCH_FACTOR": 1,   # number of batches to prefetch per worker
            "PIN_MEMORY": False,    # pin to GPU memory when dataloading?"
            "SAMPLING_K": 1,        # k in top-k sampling model predictions during play
            "PRINT_FREQUENCY": None, # 50, # 1, # print status once every so many steps
            "N_STEPS": 100000,      # number of training steps
            "WARMUP_STEPS": 4000,   # 800,   # number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official transformer repo.
            "STEP": 1,              # the step number, start from 1 to prevent math error in the 'LR' line
            "LR_SCHEDULE": "fixed", #"vaswani",  # the learning rate schedule; see utils.py for learning rate schedule
            "LR_DECAY": None,       # the decay rate for 'exp_decay' schedule
            "START_EPOCH": 0,       # start at this epoch
            "BETAS": (0.9, 0.98),   # beta coefficients in the Adam optimizer
            "EPSILON": 1e-9,  # epsilon term in the Adam optimizer
            "LABEL_SMOOTHING": 0.1, # label smoothing co-efficient in the Cross Entropy loss
            "BOARD_STATUS_LENGTH": 70,  # total length of input sequence
            "USE_AMP": True,        # use automatic mixed precision training?
            #"CRITERION" = LabelSmoothedCE  # training criterion (loss)
            #"OPTIMIZER": "", #torch.optim.Adam  # optimizer
            "BATCHES_PER_STEP": (16),# perform a training step, i.e. update parameters, once every so many batches
            "TRAINING_CHECKPOINT": None, # path to model checkpoint (NAME + ".pt") to resume training, None if none
            "CHECKPOINT_AVG_PREFIX": "step",
            "CHECKPOINT_AVG_SUFFIX": ".pt",  # checkpoint end string to match checkpoints saved for averaging
            "EVAL_GAMES_FOLDER": "training_runs/" + model_config_name + "/evaluate_games/run_" + str(run_number),  # folder where evaluation games are saved in PGN files
            "FINAL_CHECKPOINT": "averaged_" + model_config_name + "_run_" + str(run_number) + ".pt", # final checkpoint to be used for eval/inference
            "AVERAGE_STEPS": {9000, 9250, 9550, 9700, 9850, 10000}, #{491000, 492500, 494000, 495500, 497000, 498500, 500000}
            "SAVE_CHECKPOINT_EPOCH_FEQUENCY": 1,
            # "MODEL_CHECKPOINT_FOLDER": "training_runs/" + model_config_name + "/model_epoch_checkpoints/"
            "MAX_DATASET_SIZE": None
            }
    elif model_config_name=="CT-E-20_inference":
        return {
            "NAME": model_config_name,
            "VOCAB_SIZES": get_vocab_sizes(),
            "D_MODEL": 512,
            "N_HEADS": 8,
            "D_QUERIES": 64,
            "D_VALUES": 64,
            "D_INNER": 2048,
            "N_LAYERS": 6,
            "DROPOUT": 0.1, 
            "N_MOVES": 1,           # expected maximum length of move sequences in the model, <= MAX_MOVE_SEQUENCE_LENGTH
            "BATCH_SIZE": 1,
            "NUM_WORKERS": 2,       # number of workers to use for dataloading
            "RUN_NUMBER": run_number,
            "CHECKPOINT_FOLDER": "training_runs/" + "CT-E-20" + "/model_checkpoints/run_" + str(run_number) + "/",
            "DATA_FOLDER": "dataset/",
            "H5_FILE": "LE22ct.h5",
            "LOGS_FOLDER": "training_runs/" + "CT-E-20" + "/logs/run_" + str(run_number),
            "MAX_MOVE_SEQUENCE_LENGTH": 10,
            "PREFETCH_FACTOR": 1,   # number of batches to prefetch per worker
            "PIN_MEMORY": False,    # pin to GPU memory when dataloading?"
            "SAMPLING_K": 1,        # k in top-k sampling model predictions during play
            "PRINT_FREQUENCY": None, # 50, # 1, # print status once every so many steps
            "N_STEPS": 10000,      # number of training steps
            "WARMUP_STEPS": 800,   # number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official transformer repo.
            "STEP": 1,              # the step number, start from 1 to prevent math error in the 'LR' line
            "LR_SCHEDULE": "vaswani",  # the learning rate schedule; see utils.py for learning rate schedule
            "LR_DECAY": None,       # the decay rate for 'exp_decay' schedule
            "START_EPOCH": 0,       # start at this epoch
            "BETAS": (0.9, 0.98),   # beta coefficients in the Adam optimizer
            "EPSILON": 1e-9,  # epsilon term in the Adam optimizer
            "LABEL_SMOOTHING": 0.1, # label smoothing co-efficient in the Cross Entropy loss
            "BOARD_STATUS_LENGTH": 70,  # total length of input sequence
            "USE_AMP": True,        # use automatic mixed precision training?
            #"CRITERION" = LabelSmoothedCE  # training criterion (loss)
            #"OPTIMIZER": "", #torch.optim.Adam  # optimizer
            "BATCHES_PER_STEP": (16),# perform a training step, i.e. update parameters, once every so many batches
            "TRAINING_CHECKPOINT": "checkpoint_epoch_8_CT-E-20.pt", # path to model checkpoint (NAME + ".pt") to resume training, None if none
            "CHECKPOINT_AVG_PREFIX": "step",
            "CHECKPOINT_AVG_SUFFIX": ".pt",  # checkpoint end string to match checkpoints saved for averaging
            "EVAL_GAMES_FOLDER": "training_runs/evaluate_games/" + "CT-E-20",  # folder where evaluation games are saved in PGN files
            "FINAL_CHECKPOINT": "averaged_" + "CT-E-20" + ".pt", # final checkpoint to be used for eval/inference
            "AVERAGE_STEPS": {1550, 1700, 1850, 2000, 2150, 2300}, #{9000, 9250, 9550, 9700, 9850, 10000}, #{491000, 492500, 494000, 495500, 497000, 498500, 500000}
            "SAVE_CHECKPOINT_EPOCH_FEQUENCY": 1
            # "MODEL_CHECKPOINT_FOLDER": "training_runs/" + model_config_name + "/model_epoch_checkpoints/"
            }