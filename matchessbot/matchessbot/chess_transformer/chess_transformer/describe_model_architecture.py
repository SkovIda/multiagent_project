import os
import chess
import chess.svg
import pathlib
import chess.pgn
import chess.engine
import torch.utils.data
import torch.nn.functional as F

import torch
from torch.utils.data import DataLoader

import numpy as np


import re
import json

import cairosvg
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from matplotlib.colors import rgb2hex
import matplotlib as mpl


from config import import_config
from vocab_uci_dicts import RANKS, FILES, SQUARES, TURN, PIECES, UCI_MOVES, BOOL, get_vocab_sizes
from dataset import ChessDataset
from model import ChessTransformerEncoder, LabelSmoothedCE
from train import get_lr




DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


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
        direct_model_output = predicted_moves
        predicted_moves = predicted_moves[:, 0, :]  # (1, move_vocab_size)

        # Filter out move indices corresponding to illegal moves
        legal_move_indices = [UCI_MOVES[m] for m in legal_moves]

        # Perform top-k sampling to obtain a legal predicted move
        legal_move_index = topk_sampling(
            logits=predicted_moves[:, legal_move_indices],
            k=k,
        ).item()
        model_move = legal_moves[legal_move_index]

        return model_move, direct_model_output


def save_chess_board_img(board, filename, filetype=".png", show_last_move=False):
    if show_last_move:
        svg_board = chess.svg.board(
            board,
            lastmove=board.peek(),
            # fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
            # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
            # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
            size=350,
            )
    else:
        svg_board = chess.svg.board(
            board,
            size=350,
            )
    cairosvg.svg2png(bytestring=svg_board,write_to=filename + filetype)
    return


def describe_model_architecture(CONFIG):
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

    fen_pos_example_1 = "r1bqkb1r/pppppppp/2n2n2/8/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 3 3"
    # fen_pos_example_1 = "r4rk1/pbp2ppp/4p3/6q1/3P4/2N5/PPP2PPK/R2Q3R b - - 3 16"
    path_example_1 = "./test_results/CT-E-20/run_7/report_figures/model_architecture_example_1/"


    chess_board = chess.Board()
    chess_board.reset()
    chess_board.set_fen(fen=fen_pos_example_1)

    save_chess_board_img(board=chess_board, filename=path_example_1 + "model_input_board", filetype=".png", show_last_move=False)

    # Get model inputs
    model_inputs = get_model_inputs(chess_board)

    print(f"Model input for example 1:\n\t")
    print(model_inputs)

    k_samples = 5
    model_uci_move, model_output = model_next_move(model=model,board=chess_board, use_amp=CONFIG['USE_AMP'], k=k_samples)


    print(f"model_uci_move:\t{model_uci_move}")
    chess_board.push_uci(model_uci_move)

    save_chess_board_img(board=chess_board,filename=path_example_1 + "model_next_move_board", filetype=".png", show_last_move=True)

    print(f"Model output:\n\tshape:\t{model_output.shape}\n\tmodel output:\n\t{model_output}")
    # print(f"Model output (predictions):\n\t{model_output[:,0,:]}")
    print(f"Model output (predictions):\n\t{model_output[:,:1,:]}")
    print(f"Model output dimentions (1,move_vocab size):\n\t{(1,get_vocab_sizes()['moves'])}")

    # Convert model input/output to dict and write JSON object to file
    model_input_dict = {
            "turns": model_inputs['turns'].detach().cpu().numpy()[0].tolist(),
            "white_kingside_castling_rights": model_inputs['white_kingside_castling_rights'].detach().cpu().numpy()[0].tolist(),
            "white_queenside_castling_rights": model_inputs['white_queenside_castling_rights'].detach().cpu().numpy()[0].tolist(),
            "black_kingside_castling_rights": model_inputs['black_kingside_castling_rights'].detach().cpu().numpy()[0].tolist(),
            "black_queenside_castling_rights": model_inputs['black_queenside_castling_rights'].detach().cpu().numpy()[0].tolist(),
            "board_positions": model_inputs['board_positions'].detach().cpu().numpy()[0].tolist(),
            "moves": model_inputs['moves'].detach().cpu().numpy()[0].tolist(),
            "lengths": model_inputs['lengths'].detach().cpu().numpy()[0].tolist(),
        }

    with open(path_example_1 + "model_input_dict.json", "w") as outfile: 
        json.dump(model_input_dict, outfile)

    # model_output_dict = {

    # }







if __name__ == "__main__":
    # Get configuration
    CONFIG = import_config(model_config_name="CT-E-20_inference", run_number=7)

    describe_model_architecture(CONFIG)
