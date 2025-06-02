import os
# import sys
import chess
import chess.svg
# import urllib
import pathlib
# import markdown
# import textwrap
import chess.pgn
import chess.engine
import torch.utils.data
import torch.nn.functional as F
# from tqdm import tqdm
# from datetime import date
# from tabulate import tabulate
# from bs4 import BeautifulSoup
# from IPython.display import display
# from contextlib import contextmanager
# from colorama import Fore, Back, Style

import torch
from torch.utils.data import DataLoader

import numpy as np


#from chess_transformers.play.utils import write_pgns
# from chess_transformers.play import model_v_engine, warm_up, load_model, load_engine

# import os
import re
# from collections import Counter



import cairosvg
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# import matplotlib.colors
from matplotlib.colors import rgb2hex
import matplotlib as mpl


from config import import_config
from vocab_uci_dicts import RANKS, FILES, SQUARES, TURN, PIECES, UCI_MOVES, BOOL
from dataset import ChessDataset
from model import ChessTransformerEncoder, LabelSmoothedCE
from train import get_lr



DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


TOKEN_2_UCI_MOVES = [move for move in UCI_MOVES]

BOARD_2_PIECE = [piece for piece in PIECES]

SQUARE_IDX_2_SQUARE_NAME = [square for square in SQUARES]


def replace_number(match):
    """
    Replaces numbers in a string with as many periods.

    For example, "3" will be replaced by "...".

    Args:

        match (regex.match): A RegEx match for a number.

    Returns:

        str: The replacement string.
    """
    return int(match.group()) * "."


def square_index(square):
    """
    Gets the index of a chessboard square, counted from the top-left
    corner (a8) of the chessboard.

    Args:

        square (str): The square.

    Returns:

        int: The index for this square.
    """
    file = square[0]
    rank = square[1]

    return (7 - RANKS.index(rank)) * 8 + FILES.index(file)


def assign_ep_square(board, ep_square):
    """
    Notate a board position with an En Passan square.

    Args:

        board (str): The board position.

        ep_square (str): The En Passant square.

    Returns:

        str: The modified board position.
    """
    i = square_index(ep_square)

    return board[:i] + "," + board[i + 1 :]


def get_castling_rights(castling_rights):
    """
    Get individual color/side castling rights from the FEN notation of
    castling rights.

    Args:

        castling_rights (str): The castling rights component of the FEN
        notation.

    Returns:

        bool: Can white castle kingside?

        bool: Can white castle queenside?

        bool: Can black castle kingside?

        bool: Can black castle queenside?
    """
    white_kingside = "K" in castling_rights
    white_queenside = "Q" in castling_rights
    black_kingside = "k" in castling_rights
    black_queenside = "q" in castling_rights

    return white_kingside, white_queenside, black_kingside, black_queenside



def is_pawn_promotion(board, move):
    """
    Check if a move (in UCI notation) corresponds to a pawn promotion on
    the given board.

    Args:

        board (chess.Board): The chessboard in its current state.

        move (str): The move un UCI notation.

    Returns:

        bool: Is the move a pawn promotion?
    """

    m = chess.Move.from_uci(move)
    if board.piece_type_at(m.from_square) == chess.PAWN and chess.square_rank(
        m.to_square
    ) in [0, 7]:
        return True
    else:
        return False





def parse_fen(fen):
    """
    Parse the FEN notation at a given board position.

    Args:

        fen (str): The FEN notation.

    Returns:

        str: The player to move next, one of "w" or "b".

        str: The board position.

        bool: Can white castle kingside?

        bool: Can white castle queenside?

        bool: Can black castle kingside?

        bool: Can black castle queenside?
    """
    board, turn, castling_rights, ep_square, _, __ = fen.split()
    board = re.sub(r"\d", replace_number, board.replace("/", ""))
    if ep_square != "-":
        board = assign_ep_square(board, ep_square)
    (
        white_kingside,
        white_queenside,
        black_kingside,
        black_queenside,
    ) = get_castling_rights(castling_rights)

    return turn, board, white_kingside, white_queenside, black_kingside, black_queenside

def encode(item, vocabulary):
    """
    Encode an item with its index in the vocabulary its from.

    Args:

        item (list, str, bool): The item.

        vocabulary (dict): The vocabulary.

    Raises:

        NotImplementedError: If the item is not one of the types
        specified above.

    Returns:

        list, str: The item, encoded.
    """
    if isinstance(item, list):  # move sequence
        return [vocabulary[it] for it in item]
    elif isinstance(item, str):  # turn or board position or square
        return (
            vocabulary[item] if item in vocabulary else [vocabulary[it] for it in item]
        )
    elif isinstance(item, bool):  # castling rights
        return vocabulary[item]
    else:
        raise NotImplementedError



def decode_tokens_to_uci(token: int):
    # token_2_uci_str = [uci_move for uci_move in UCI_MOVES.keys()]
    if token < len(TOKEN_2_UCI_MOVES):
        return TOKEN_2_UCI_MOVES[token] #uci_move
    else:
        raise ValueError(f"Token: {token} is NOT in the vocabilary!")





def write_pgns(pgns, pgn_file):
    """
    Write PGNs to a file.

    Args:

        pgns (str): The PGNs.

        pgn_file (str): The path to write as a file.
    """
    parent_folder = pathlib.Path(pgn_file).parent.resolve()
    parent_folder.mkdir(parents=True, exist_ok=True)
    with open(pgn_file, "w") as f:
        f.write(pgns)









def get_model_inputs(board):
    """
    Get inputs to be fed to a model.

    Args:

        board (chess.Board): The chessboard in its current state.

    Returns:

        dict: The inputs to be fed to the model.
    """
    model_inputs = dict()

    t, b, wk, wq, bk, bq = parse_fen(board.fen())
    model_inputs["turns"] = (
        torch.IntTensor([encode(t, vocabulary=TURN)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["board_positions"] = (
        torch.IntTensor(encode(b, vocabulary=PIECES)).unsqueeze(0).to(DEVICE)
    )
    model_inputs["white_kingside_castling_rights"] = (
        torch.IntTensor([encode(wk, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["white_queenside_castling_rights"] = (
        torch.IntTensor([encode(wq, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["black_kingside_castling_rights"] = (
        torch.IntTensor([encode(bk, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["black_queenside_castling_rights"] = (
        torch.IntTensor([encode(bq, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["moves"] = (
        torch.LongTensor(
            [
                UCI_MOVES["<move>"],
                UCI_MOVES["<pad>"],
            ]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["lengths"] = torch.LongTensor([1]).unsqueeze(0).to(DEVICE)

    return model_inputs



def load_model(CONFIG):
    """
    Load model for inference.

    Args:

        CONFIG (dict): The configuration of the model.

    Returns:

        torch.nn.Module: The model.
    """

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChessTransformerEncoder(CONFIG)
    model = model.to(device)

    # # Load checkpoint:
    # checkpoint_folder = (
    #     pathlib.Path(__file__).parent.parent.resolve() / "checkpoints" / CONFIG.NAME
    # )
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

    if CONFIG['TRAINING_CHECKPOINT'] is not None:
        checkpoint = torch.load(
            os.path.join(CONFIG['CHECKPOINT_FOLDER'], CONFIG['TRAINING_CHECKPOINT']),
            weights_only=True,
        )
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
    else:
        raise ValueError(f"Could not load model {CONFIG['TRAINING_CHECKPOINT']} for filepath {CONFIG['CHECKPOINT_FOLDER']}")

    print("\nModel loaded!\n")

    return model


def topk_sampling(logits, k=1):
    """
    Randomly sample from the multinomial distribution formed from the
    "top-k" logits only.

    Args:

        logits (torch.FloatTensor): Predicted logits, of size (N,
        vocab_size).

        k (int, optional): Value of "k". Defaults to 1.

    Returns:

        torch.LongTensor: Samples (indices), of size (N).
    """
    k = min(k, logits.shape[1])

    with torch.no_grad():
        min_topk_logit_values = logits.topk(k=k, dim=1)[0][:, -1:]  # (N, 1)
        logits[logits < min_topk_logit_values] = -float("inf")  #  (N, vocab_size)
        probabilities = F.softmax(logits, dim=1)  #  (N, vocab_size)
        samples = torch.multinomial(probabilities, num_samples=1).squeeze(1)  #  (N)

    return samples


def model_next_move(
    model,
    board,
    use_amp,
    k,
    model_name=None,
    opponent_name=None,
    show_board=True,
):
    """
    Have the model make the next move on the board.

    Args:

        config_name (str): The name of the model configuration (which
        describes the type of model).

        model (torch.nn.Module): The model.

        board (chess.Board): The chessboard in its current state.

        use_amp (bool): Use automatic mixed precision?

        k (int): The "k" in "top-k" sampling, for sampling the model's
        predicted moves.

        model_name (str, optional): The name of the model, for
        displaying in status messages. Defaults to None.

        opponent_name (str, optional): The name of the model's opponent,
        for displaying in status messages. Defaults to None.

        show_board (bool, optional): Display the board (along with a
        status message) upon making the move?

    Returns:

        chess.Board: The chessboard after the model makes its move.
    """
    # Get predictions
    model.eval()
    with torch.no_grad():
        # Get list of legal moves for the current position
        legal_moves = [move.uci() for move in board.legal_moves]

        # Get model inputs
        model_inputs = get_model_inputs(board)

        # (Direct) Move prediction models
        # if model.code in {"E", "ED"}:
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp
        ):
            predicted_moves = model(model_inputs)
        predicted_moves = predicted_moves[:, 0, :]  # (1, move_vocab_size)

        # Filter out move indices corresponding to illegal moves
        legal_move_indices = [UCI_MOVES[m] for m in legal_moves]

        # Perform top-k sampling to obtain a legal predicted move
        legal_move_index = topk_sampling(
            logits=predicted_moves[:, legal_move_indices],
            k=k,
        ).item()
        model_move = legal_moves[legal_move_index]

        # # "From" and "To" square prediction models
        # elif model.code in {"EFT"}:
        #     with torch.autocast(
        #         device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp
        #     ):
        #         predicted_from_squares, predicted_to_squares = model(
        #             model_inputs
        #         )  # (1, 1, 64), (1, 1, 64)
        #     predicted_from_squares = predicted_from_squares[:, 0, :]  # (1, 64)
        #     predicted_to_squares = predicted_to_squares[:, 0, :]  # (1, 64)

        #     # Convert "From" and "To" square predictions to move predictions
        #     predicted_from_log_probabilities = torch.log_softmax(
        #         predicted_from_squares, dim=-1
        #     ).unsqueeze(
        #         2
        #     )  # (1, 64, 1)
        #     predicted_to_log_probabilities = torch.log_softmax(
        #         predicted_to_squares, dim=-1
        #     ).unsqueeze(
        #         1
        #     )  # (1, 1, 64)
        #     predicted_moves = (
        #         predicted_from_log_probabilities + predicted_to_log_probabilities
        #     ).view(
        #         1, -1
        #     )  # (1, 64 * 64)

        #     # Filter out move indices corresponding to illegal moves
        #     legal_moves = list(
        #         set([m[:4] for m in legal_moves])
        #     )  # for handing pawn promotions manually, remove pawn promotion targets
        #     legal_move_indices = list()
        #     for m in legal_moves:
        #         from_square = m[:2]
        #         to_square = m[2:4]
        #         legal_move_indices.append(
        #             SQUARES[from_square] * 64 + SQUARES[to_square]
        #         )

        #     # Perform top-k sampling to obtain a legal predicted move
        #     legal_move_index = topk_sampling(
        #         logits=predicted_moves[:, legal_move_indices],
        #         k=k,
        #     ).item()
        #     model_move = legal_moves[legal_move_index]

        #     # Handle pawn promotion manually if "model_move" is a pawn promotion move
        #     if is_pawn_promotion(board, model_move):
        #         model_move = model_move + "q"  # always promote to a queen

        # # Other models
        # else:
        #     raise NotImplementedError

        # # Move
        # board.push_uci(model_move)
        # if show_board:
        #     clear_output(wait=True)
        #     if opponent_name and len(board.move_stack) > 1:
        #         msg = "# {} played ***{}***. {} plays ***{}***.".format(
        #             opponent_name,
        #             board.move_stack[-2],
        #             model_name if model_name else "Model",
        #             board.move_stack[-1],
        #         )
        #     else:
        #         msg = "# {} plays ***{}***.".format(
        #             model_name if model_name else "Model", board.move_stack[-1]
        #         )
        #     display(Markdown(msg)) if in_notebook() else print_text(msg)
        #     display(board) if in_notebook() else print_board(board)

        return model_move #decode_tokens_to_uci(model_move) # board



def board_positions_2_chess_board(board_positions):
    
    board = chess.Board()
    board.clear()

    temp_board_pieces_list = []
    board_pieces_dict = {}

    for square_idx, piece_token in enumerate(board_positions):
        if piece_token not in [0, 1]:
            # print(f"piece token:\t{piece_token}")
            piece_str = BOARD_2_PIECE[piece_token]
            # print(f"piece str:\t{piece_str}")
            chess_piece = chess.Piece.from_symbol(piece_str)

            temp_board_pieces_list.append(chess_piece)



            square_name = SQUARE_IDX_2_SQUARE_NAME[square_idx]
            chess_square = chess.parse_square(square_name)

            board_pieces_dict[chess_square] = chess_piece
        else:
                non_piece_token = BOARD_2_PIECE[piece_token]
                square_name = SQUARE_IDX_2_SQUARE_NAME[square_idx]
                chess_square = chess.parse_square(square_name)
                # print(f"square {square_name} is set to token \"{non_piece_token}\"")

    # print(temp_board_pieces_list)
    # print(board_pieces_dict)

    board.set_piece_map(board_pieces_dict)

    return board




def run_validation(val_loader, model, criterion, device, CONFIG , num_examples=2):
    model.eval()
    count = 0

    # n_topk_predictions_to_choose_from = 10
    # TODO: Visualize the attention maps for the different squares on the board as well!

    print(f"========== Running inference for {num_examples} examples")
    pathlib.Path("test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"])).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            count += 1

            # Move to default device
            for key in batch:
                batch[key] = batch[key].to(device)

            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=CONFIG['USE_AMP']
            ):
                # (Direct) Move prediction models
                if CONFIG['NAME'].startswith(("CT-ED-", "CT-E-")):
                    # Forward prop.
                    predicted_moves = model(batch)  # (N, n_moves, move_vocab_size)
                    # Note: n_moves is how many moves into the future we
                    # are targeting for modeling. For an Encoder-Decoder
                    # model, this might be max_move_sequence_length. For
                    # an Encoder-only model, this will be 1.

                    # Loss
                    loss = criterion(
                        predicted=predicted_moves,  # (N, n_moves, move_vocab_size)
                        targets=batch["moves"][:, 1:],  # (N, n_moves)
                        lengths=batch["lengths"],  # (N, 1)
                    )  # scalar
                    # Note: We don't pass the first move (the prompt
                    # "<move>") as it is not a target/next-move of
                    # anything

                    # Print the input/output of the model and the target output to terminal:
                    batch_moves = batch["moves"]
                    token_batch_move_list = batch_moves.detach().cpu().numpy()[0]
                    batch_moves_uci = [decode_tokens_to_uci(tok) for tok in token_batch_move_list]
                    output_token_values = predicted_moves[:, 0, :]  # (1, move_vocab_size)
                    next_move_token = topk_sampling(logits=output_token_values,k=1)[0]
                    uci_move = decode_tokens_to_uci(next_move_token.item()) #UCI_MOVES.get(int(next_move_token.item()), 'Not Found')
                    
                    
                    board_positions = batch["board_positions"]
                    
                    board_positions_numpy = board_positions.detach().cpu().numpy()[0]
                    chess_board = board_positions_2_chess_board(board_positions_numpy)

                    batch_turns = batch["turns"].detach().cpu()
                    chess_board.turn = batch_turns[0].item() # batch_turns.item()

                    print(chess_board)

                    print(f"\nExample #{count}:")
                    print(f"\tMove sequence tokens:\t{token_batch_move_list}")
                    print(f"\tMove sequence uci_moves:\t{batch_moves_uci}")

                    print(f"\tPredicted move token:\t{next_move_token.item()}")
                    print(f"\tPredicted UCI move:\ttype: {type(output_token_values)}\tshape: {output_token_values.shape}")
                    print(f"\tPredicted UCI move:\t{uci_move}")
                    print(f"\tTargets:\t{batch_moves[:, 1:]}")
                    # print(f"\tBoard positions:\n\ttype: {type(board_positions)}\tshape: {board_positions.shape}")
                    print(f"Board positions:\t{board_positions}")

                    print(f"\tchess_board_turn:\t{chess_board.turn}")
                    

                    # print(f"\tbatch info:\n\ttype: {batch}")
                    # pathlib.Path("./test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/attention_visual_example_" + str(i)).mkdir(parents=True, exist_ok=True)
                    pathlib.Path("test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/attention_visual_example_" + str(i)).mkdir(exist_ok=True)
                    
                    pathlib.Path("test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/attention_visual_example_" + str(i) + "/attn_map_per_cls_heads").mkdir(exist_ok=True)

                    avg_attn_per_head = [[],[],[],[],[],[],[],[]]

                    attn_scores_all_cls_tokens_per_head = [] #[[],[],[],[],[],[],[],[]]
                    for cls_head in range(69):
                        attn_scores_all_cls_tokens_per_head.append([[],[],[],[],[],[],[],[]])
                    
                    # head = 0
                    for layer_i, enc_layer in enumerate(model.board_encoder.encoder_layers):
                        # print(f"\nLayer{layer_i + 1}:")
                        # print(f"n_heads:\t{enc_layer[0].n_heads}")
                        for head_i in range(enc_layer[0].n_heads):
                            # # # print(type(enc_layer[0].attention_scores))

                            # print(f"Layer{layer_i + 1}\tHead\t{head_i}: attn_scores.shape={enc_layer[0].attention_scores.shape}")

                            # # print(enc_layer[0].attention_scores[0, head_i].data)

                            # # chess_move = chess.Move.from_uci(uci_move)
                            # # chess_move_from_square =chess_move.from_square
                            
                            # attn_scores_dim_01 = enc_layer[0].attention_scores[head_i, 0].data.detach().cpu().numpy()
                            # attn_scores_dim_02 = enc_layer[0].attention_scores[head_i, :, 0].data.detach().cpu().numpy()
                            # attn_scores_dim_12 = enc_layer[0].attention_scores[0, head_i].data.detach().cpu().numpy()
                            # print(f"\nattn_scores_dim_01:\n{attn_scores_dim_01}")
                            # print(f"\nattn_scores_dim_02:\n{attn_scores_dim_02}")
                            # print(f"\nattn_scores_dim_12:\n{attn_scores_dim_12}")

                            # NOTE: Use index [head_i, 0] because index zero of second dimension gives us the attention scores of the turn token in the input sequence???
                            attn_score = enc_layer[0].attention_scores[head_i, 0].data.detach().cpu().numpy() # attn_scores_dim_01 #enc_layer[0].attention_scores[0, head_i].data.detach().cpu().numpy()

                            for cls_head_i in range(69):
                                # attn_scores_all_cls_tokens.append()
                                attn_score_cls_head_i = enc_layer[0].attention_scores[head_i, cls_head_i].data.detach().cpu().numpy()
                                attn_scores_all_cls_tokens_per_head[cls_head_i][head_i].append(attn_score_cls_head_i)
                                # visualize_attention_maps(chess_board, filename="./test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/attention_visual_example_" + str(i) + "/attn_map_per_cls_heads/" + "cls_head_" + str(cls_head_i) + "_layer_" + str(layer_i) + "_head_" + str(head_i), attention_scores=attn_score_cls_head_i[5:], filetype=".png")

                            avg_attn_per_head[head_i].append(attn_score)

                            visualize_attention_maps(chess_board, filename="./test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/attention_visual_example_" + str(i) + "/attn_map_layer_" + str(layer_i) + "_head_" + str(head_i), attention_scores=attn_score[5:],filetype=".png")
                        

                        # for head_i in enumerate(enc_layer[0].n_heads):
                        #     print(f"head:\t{enc_layer + 1}")

                    numpy_attn_map = np.array(avg_attn_per_head)

                    attention_per_head = np.sum(numpy_attn_map,axis=1) #/ 6.0
                    print(attention_per_head.shape)

                    

                    for idx in range(len(attention_per_head)):
                        visualize_attention_maps(chess_board, filename="./test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/attention_visual_example_" + str(i) + "/attn_map_avg_head_" + str(idx), attention_scores=attention_per_head[idx,5:],filetype=".png")
                    
                    # Apply the top legal uci_move to the board before visualizing the chess board as imgage:
                    legal_uci_move = model_next_move(model=model, board=chess_board, use_amp=True, k=1)
                    print(f"Predicted move:\t{uci_move}\tApplying best legal uci move:\t{legal_uci_move}")
                    uci_move = legal_uci_move
                    
                    

                    print(f"\tAttention map for the \"from square\" of chosen uci move:\t{uci_move}")
                    uci_move_frm_square_idx = int(chess.Move.from_uci(uci_move).from_square)
                    print(f"\tFrom square:\t{chess.square_name(chess.Move.from_uci(uci_move).from_square)}\tas int:\t{uci_move_frm_square_idx}")
                    numpy_attn_map_cls_from_square = np.array(attn_scores_all_cls_tokens_per_head)
                    # # attention_per_head_cls_from_square = np.sum(numpy_attn_map_cls_from_square,axis=2) #/ 6.0
                    print(numpy_attn_map_cls_from_square.shape)
                    print(np.array(attn_scores_all_cls_tokens_per_head).shape)
                    # # print(attn_scores_all_cls_tokens_per_head[uci_move_frm_square_idx+5][0])
                    
                    attn_score_cls_head_from_square = attn_scores_all_cls_tokens_per_head[uci_move_frm_square_idx+5]
                    for layer_i in range(6):
                        for head_i in range(8):
                            attn_score_cls_head_i = attn_score_cls_head_from_square[head_i][layer_i]
                            visualize_attention_maps(chess_board, filename="./test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/attention_visual_example_" + str(i) + "/attn_map_per_cls_heads/" + "cls_head_" + str(uci_move_frm_square_idx+5) + "_layer_" + str(layer_i) + "_head_" + str(head_i), attention_scores=attn_score_cls_head_i[5:], filetype=".png")
                    
                    chess_board.push_uci(uci_move)
                    save_chess_board_img(chess_board, filename="./test_results/CT-E-20/run_" + str(CONFIG["RUN_NUMBER"]) + "/inference_example_" + str(i), filetype=".png")


                    # for i, avg_head_attn in enumerate(avg_attn_per_head):
                    #     avg_head_attention = np.sum(avg_head_attn)
                    #     print(f"\nHead_i: {i}:\t{len(avg_head_attn)}\tavg. attention scores:\n{avg_head_attention}")



            # print(f"\tloss:{loss}")

            if count >= num_examples:
                break
            
            # # model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            # model_out_beam = beam_search_decode(model, 4, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            # source_text = batch["src_text"][0]
            # target_text = batch["tgt_text"][0]
            # # model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            # model_out_text = tokenizer_tgt.decode(model_out_beam.detach().cpu().numpy())

            # source_texts.append(source_text)
            # expected.append(target_text)
            # predicted.append(model_out_text)
            
            # # Print the source, target and model output
            # print_msg('-'*console_width)
            # print_msg(f"{f'SOURCE: ':>12}{source_text}")
            # print_msg(f"{f'TARGET: ':>12}{target_text}")
            # print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            # ##### Calulate the validation loss:
            # decoder_input = batch['decoder_input'].to(device) # (B, seq_len)
            # decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq_len, seq_len)

            # # Run the tensors through the encoder, decoder and the projection layer
            # encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq_len, d_model)
            # decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq_len, d_model)
            # proj_output = model.project(decoder_output) # (B, seq_len, vocab_size)

            # # Compare the output with the label
            # label = batch['label'].to(device) # (B, seq_len)

            # # Compute the loss using a simple cross entropy
            # valid_loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            # valid_losses.append(valid_loss.item())


    # if writer is not None:
    #     print("Validation:")
    #     # Evaluate the character error rate
    #     # Compute the char error rate 
    #     metric = CharErrorRate() #torchmetrics.CharErrorRate()
    #     cer = metric(predicted, expected)
    #     writer.add_scalar('validation cer', cer, global_step)
    #     writer.flush()

    #     # Compute the word error rate
    #     metric = WordErrorRate() ##torchmetrics.WordErrorRate()
    #     wer = metric(predicted, expected)
    #     writer.add_scalar('validation wer', wer, global_step)
    #     writer.flush()

    #     # Compute the BLEU metric
    #     metric = BLEUScore()    #torchmetrics.BLEUScore()
    #     bleu = metric(predicted, expected)
    #     writer.add_scalar('validation BLEU', bleu, global_step)
    #     writer.flush()

    #     print(cer)
    #     print(wer)
    #     print(bleu)

    #     ##################
    #     # print("metric: sacrebleu")
    #     # metric = huggin.load_metric('sacrebleu')
    #     # metric.add_batch(predictions=predicted, references=expected)
    #     # score = metric.compute()
    #     # print(score)

    #     print("SacreBLEUScore - ver 2:")
    #     sacre_bleu = SacreBLEUScore()
    #     sacrebleu_score = sacre_bleu(predicted, target_text)
    #     print(sacrebleu_score)


    #     bleu_eval = evaluate.load("bleu")
    #     results = bleu_eval.compute(predictions=predicted, references=expected)
    #     print(results)

    #     # bertscore = evaluate_lib.load("bertscore")
    #     # results = bertscore.compute(predictions=predicted, references=expected)
    #     # print("BERTScore from evaluate lib:")
    #     # print(results)
        
    #     # # BERTScore calculation
    #     # scorer = BERTScorer(model_type="")
    #     # P, R, F1 = scorer.score(predicted, expected)
    #     # print(f"BERTScore Precision: {P.mean():.4f}, Recall: {R.mean():.4f}, F1: {F1.mean():.4f}")

    return # np.mean(valid_losses), cer.item(), wer.item(), bleu.item(), sacrebleu_score.item(), results



# def get_attention_maps(model, layers=[0], heads=[0]):
#     for layer in layers:
#         for head in heads:


def save_chess_board_img(board, filename, filetype=".png"):
    svg_board = chess.svg.board(
        board,
        lastmove=board.peek(),
        # fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
        # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
        # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
        size=350,
        )
    cairosvg.svg2png(bytestring=svg_board,write_to=filename + filetype)
    return
    

# def visualize_attention_maps(board, filename, attention_scores, filetype=".png"):
#     # TODO: color squares based on the attention weights and save img as png file.
#     # NOTE: Use matplotlib.colormap 'plasma' for the color: https://matplotlib.org/stable/users/explain/colors/colormaps.html
#     # attention_scores
#     cmap_plasma = mpl.colormaps['plasma']

#     # viridis = mpl.colormaps['viridis'].resampled(256)
#     # newcolors = viridis(np.linspace(0, 1, 256))
#     # pink = np.array([248/256, 24/256, 148/256, 1])
#     # newcolors[:25, :] = pink
#     # newcmp = ListedColormap(newcolors)

#     # plot_examples([viridis, newcmp])

#     # print(attention_scores.shape)

#     attn_map_dict = {}

#     hex_col_map = []
#     for idx, val in enumerate(attention_scores):

#         hex_col_map.append(rgb2hex(cmap_plasma(val), keep_alpha=True))

#         square_name = SQUARE_IDX_2_SQUARE_NAME[idx]
#         chess_square = chess.parse_square(square_name)

#         attn_map_dict[chess_square] = rgb2hex(cmap_plasma(val))

#     # print(hex_col_map)
#     # print(type(attn_map_dict))
#     # print(attn_map_dict)

#     svg_board = chess.svg.board(
#         board,
#         # lastmove=board.peek(),
#         fill=attn_map_dict, #dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
#         # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
#         # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
#         size=350,
#         )
#     cairosvg.svg2png(bytestring=svg_board,write_to=filename + filetype)
#     return

def visualize_attention_maps(board, filename, attention_scores, filetype=".png", cls_square=None):
    # TODO: color squares based on the attention weights and save img as png file.
    # NOTE: Use matplotlib.colormap 'plasma' for the color: https://matplotlib.org/stable/users/explain/colors/colormaps.html
    # attention_scores
    cmap_plasma = mpl.colormaps['plasma']

    # viridis = mpl.colormaps['viridis'].resampled(256)
    # newcolors = viridis(np.linspace(0, 1, 256))
    # pink = np.array([248/256, 24/256, 148/256, 1])
    # newcolors[:25, :] = pink
    # newcmp = ListedColormap(newcolors)

    # plot_examples([viridis, newcmp])

    # print(attention_scores.shape)

    attn_map_dict = {}

    hex_col_map = []
    for idx, val in enumerate(attention_scores):

        hex_col_map.append(rgb2hex(cmap_plasma(val), keep_alpha=True))

        square_name = SQUARE_IDX_2_SQUARE_NAME[idx]
        chess_square = chess.parse_square(square_name)

        attn_map_dict[chess_square] = rgb2hex(cmap_plasma(val))

    # print(hex_col_map)
    # print(type(attn_map_dict))
    # print(attn_map_dict)
    if cls_square is None:

        svg_board = chess.svg.board(
            board,
            # lastmove=board.peek(),
            fill=attn_map_dict, #dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
            # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
            # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
            size=350,
            )
        cairosvg.svg2png(bytestring=svg_board,write_to=filename + filetype)
    return




def generate_model_description(CONFIG):
    """
    Generate a description of the trained model in terms of input/output and dimension of all the layers/heads in the trained model.

    Args:

        CONFIG (dict): Configuration. See ./configs.
    """
    
    # Load model:
    model = load_model(CONFIG)

    # model.eval()  # eval mode disables dropout

    print(
        "There are %d learnable parameters in this model."
        % sum([p.numel() for p in model.parameters()])
    )

    # model.board_encoder.encoder_layers[:]
    # get_attention_maps(model, layers=[0], heads=[0])

    # layer = 0
    # head = 0
    # print(type(model.board_encoder.encoder_layers[layer][0]))

    # Loss function
    criterion = LabelSmoothedCE(
        eps=CONFIG['LABEL_SMOOTHING'], n_predictions=CONFIG['N_MOVES']
    )
    criterion = criterion.to(DEVICE)


    # Initialize the test dataset:
    test_loader = DataLoader(
        dataset=ChessDataset(
            data_folder=CONFIG['DATA_FOLDER'],
            h5_file=CONFIG['H5_FILE'],
            split="val",
            n_moves=CONFIG['N_MOVES'],
        ),
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY'],
        prefetch_factor=CONFIG['PREFETCH_FACTOR'],
        shuffle=False,
    )

    run_validation(test_loader, model=model, CONFIG=CONFIG, device=DEVICE, criterion=criterion,num_examples=5)

    # model.board_encoder.encoder_layers #.get_submodule()

    # for layer_i, enc_layer in enumerate(model.board_encoder.encoder_layers):
    #     print(f"Layer:\t{layer_i + 1}")
    #     print(enc_layer.attention_weights)
    #     print(type(enc_layer.positional_embeddings.weight))
    #     # for head_i in enumerate(enc_layer[0].n_heads):
    #     #     print(f"head:\t{enc_layer + 1}")
            





def evaluate_model(CONFIG):
    """
    Evaluation.

    Args:

        CONFIG (dict): Configuration. See ./configs.
    """
    # Load
    model = load_model(CONFIG)

    # Initialize the test dataset:
    test_loader = DataLoader(
        dataset=ChessDataset(
            data_folder=CONFIG['DATA_FOLDER'],
            h5_file=CONFIG['H5_FILE'],
            split="val",
            n_moves=CONFIG['N_MOVES'],
        ),
        batch_size=CONFIG['BATCH_SIZE'],
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY'],
        prefetch_factor=CONFIG['PREFETCH_FACTOR'],
        shuffle=False,
    )

    # # Warmup model
    # warm_up(
    #     model=model,
    # )
    model.eval()  # eval mode disables dropout


    # # Evaluate
    # for LL in range(1, 7):
    #     for model_color in ["w", "b"]:
    #         # Play
    #         wins, losses, draws, pgns = model_v_engine(
    #             model=model,
    #             k=CONFIG.SAMPLING_K,
    #             use_amp=CONFIG.USE_AMP,
    #             model_color=model_color,
    #             engine=engine,
    #             time_limit=CONFIG.LICHESS_LEVELS[LL]["TIME_CONSTRAINT"],
    #             depth_limit=CONFIG.LICHESS_LEVELS[LL]["DEPTH"],
    #             uci_options={
    #                 "Skill Level": CONFIG.LICHESS_LEVELS[LL]["SKILL"],
    #                 "Threads": 8,
    #                 "Hash": 8000,
    #             },
    #             rounds=500,
    #             clock=None,
    #             white_player_name="Fairy Stockfish @ LL {}".format(LL)
    #             if model_color == "b"
    #             else CONFIG.NAME,
    #             black_player_name="Fairy Stockfish @ LL {}".format(LL)
    #             if model_color == "w"
    #             else CONFIG.NAME,
    #             event=CONFIG.NAME + " v. Fairy Stockfish @ LL {}".format(LL)
    #             if model_color == "w"
    #             else "Fairy Stockfish @ LL {} v. ".format(LL) + CONFIG.NAME,
    #         )

    #         # Write games to PGN
    #         write_pgns(
    #             pgns,
    #             pgn_file=os.path.join(
    #                 CONFIG.EVAL_GAMES_FOLDER,
    #                 (
    #                     "LL {} | "
    #                     + CONFIG.NAME
    #                     + " as {} | GAMES {} |  W {} |  L {} |  D {} | {}.pgn"
    #                 ).format(
    #                     LL,
    #                     model_color.upper(),
    #                     500,
    #                     wins,
    #                     losses,
    #                     draws,
    #                     cpuinfo.get_cpu_info()["brand_raw"],
    #                 ),
    #             ),
    #         )


if __name__ == "__main__":
    # Get configuration
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_name", type=str, help="Name of configuration file.")
    # args = parser.parse_args()
    # CONFIG = import_config(model_config_name="CT-E-20_inference", run_number=7)
    CONFIG = import_config(model_config_name="CT-E-20_run_9", run_number=9)


    # Evaluate model
    generate_model_description(CONFIG)