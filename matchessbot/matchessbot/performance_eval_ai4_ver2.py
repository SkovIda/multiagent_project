import os

import torch
import numpy as np


import chess
import chess.engine
from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils


from chessformers.chessformers.configuration import get_configuration
from chessformers.chessformers.model import Transformer
from chessformers.chessformers.tokenizer import Tokenizer


from chess_transformer.chess_transformer.model import ChessTransformerEncoder

from chess_transformer.chess_transformer.config import import_config
# from chess_transformer.chess_transformer.inference import model_next_move

from chess_transformer.chess_transformer.vocab_uci_dicts import RANKS, FILES, SQUARES, TURN, PIECES, UCI_MOVES, BOOL


from astar_search.astar_search import astar_search

import time

import json

from tqdm import tqdm

###########################################################

import os
import chess
import pathlib

import torch.nn.functional as F


import numpy as np

import re


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



def load_encoder_model(CONFIG):
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
    # optimizer = torch.optim.Adam(
    #     params=[p for p in model.parameters() if p.requires_grad],
    #     lr=get_lr(
    #         step=CONFIG['STEP'],
    #         d_model=CONFIG['D_MODEL'],
    #         warmup_steps=CONFIG['WARMUP_STEPS'],
    #         schedule=CONFIG['LR_SCHEDULE'],
    #         decay=CONFIG['LR_DECAY'],
    #         ),
    #     betas=CONFIG['BETAS'],
    #     eps=CONFIG['EPSILON'],
    # )

    if CONFIG['TRAINING_CHECKPOINT'] is not None:
        checkpoint = torch.load(
            os.path.join(CONFIG['CHECKPOINT_FOLDER'], CONFIG['TRAINING_CHECKPOINT']),
            weights_only=True,
        )
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)
    else:
        raise ValueError(f"Could not load model {CONFIG['TRAINING_CHECKPOINT']} for filepath {CONFIG['CHECKPOINT_FOLDER']}")

    # print("\nModel loaded!\n")

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

        return model_move #decode_tokens_to_uci(model_move) # board


##############################################################

# Base class for chess a chess player:
class Player(object):
    def __init__(self):
        return NotImplementedError
    
    def get_next_move(self):
        return NotImplementedError
    
    def shut_down(self):
        return NotImplementedError


# Player with random move selection:
class RandomPlayer(Player):
    def __init__(self, name: str='player_0'):
        self.player_name = name
        self.player_type='random'
        return

    def get_next_move(self, chess_board_fen, action_mask, move_hist_san, move_hist_uci):
        chess_board = chess.Board(fen=chess_board_fen)
        legal_moves_uci_list = []
        for move in chess_board.legal_moves:
            legal_moves_uci_list.append(chess.Move.uci(move))
        # print(f'random_player.legal_moves: {legal_moves_uci_list}')
        return legal_moves_uci_list[np.random.randint(0, len(legal_moves_uci_list))]
    
    

class AStarPlayer(Player):
    def __init__(self, name: str='player_0', max_search_depth: int=2):
        self.player_name = name
        self.max_search_depth = max_search_depth
        self.player_type='astar-' + str(max_search_depth)

        return

    def get_next_move(self, chess_board_fen, action_mask, move_hist_san, move_hist_uci):
        move_hist_uci_input = [move for move in move_hist_uci]
        # print(f"\nBEFORE AStarPlayer.get_next_move\tfen={chess_board_fen}\n\tmovehist={move_hist_uci_input}")
        next_uci_move = astar_search(max_search_depth=self.max_search_depth, move_hist_uci=move_hist_uci_input, start_fen=chess_board_fen)
        # print(f"\nAFTER AStarPlayer.get_next_move\tfen={chess_board_fen}\n\tmovehist={move_hist_uci_input}")
        # print(f"\tastar_search(args) => \tuci_move: {next_uci_move}\n\tlength: {len(next_uci_move)}\t type: {type(next_uci_move)}")
        return next_uci_move




# Player that utilizes the Stockfish chess engine for selecting its moves:
class StockfishPlayer(Player):
    SKILL_LEVEL_ELO = {1: 1350}
    def __init__(self, name: str='player_0', skill_level = 1):
        self.player_name = name
        # self.player_type='stockfish-level-' + str(skill_level)
        self.player_type='stockfish-elo' + str(self.SKILL_LEVEL_ELO[skill_level])
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
        # print(dict(self.engine.options).items())
        self.engine.configure({"Skill Level": skill_level, "UCI_LimitStrength": True, "UCI_Elo": self.SKILL_LEVEL_ELO[skill_level]})
        
    def get_next_move(self, chess_board_fen, action_mask, move_hist_san, move_hist_uci):
        chess_board = chess.Board(fen=chess_board_fen)
        engine_result = self.engine.play(chess_board, chess.engine.Limit(time=0.1))
        next_move_uci = engine_result.move.uci()
        return next_move_uci
    
    def shut_down(self):
        self.engine.close()
        return
    

# Player that utilizes the Stockfish chess engine for selecting its moves:
class FairyStockfishPlayer(Player):
    SKILL_LEVEL_ELO = {1: 500}
    def __init__(self, name: str='player_0', skill_level = 1):
        self.player_name = name
        # self.player_type='stockfish-level-' + str(skill_level)
        self.player_type='fairystockfish-elo' + str(self.SKILL_LEVEL_ELO[skill_level])
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/home/ida/Downloads/fairy-stockfish-largeboard_x86-64")
        # print(dict(self.engine.options).items())
        # self.engine.configure({"Skill Level": skill_level})
        self.engine.configure({"UCI_LimitStrength": True, "UCI_Elo": self.SKILL_LEVEL_ELO[skill_level]})
        
    def get_next_move(self, chess_board_fen, action_mask, move_hist_san, move_hist_uci):
        chess_board = chess.Board(fen=chess_board_fen)
        engine_result = self.engine.play(chess_board, chess.engine.Limit(time=0.1))
        next_move_uci = engine_result.move.uci()
        return next_move_uci
    
    def shut_down(self):
        self.engine.close()
        return
    

# Player that utilizes a Transformer Decoder-Only model trained for move selection:
class DecoderChessformerPlayer(Player):
    def __init__(self, name='player_0'):
        self.player_name=name
        self.player_type='decformer'

    def load_model(self, cfg):
        self.cfg = cfg

        self.config = get_configuration(self.cfg['config'])
        self.tokenizer = Tokenizer(self.cfg['tokenizer'])
        self.model = Transformer(self.tokenizer,
                            num_tokens=self.tokenizer.vocab_size(),
                            dim_model=self.config["model"]["dim_model"],
                            d_hid=self.config["model"]["d_hid"],
                            num_heads=self.config["model"]["num_heads"],
                            num_layers=self.config["model"]["num_layers"],
                            dropout_p=self.config["model"]["dropout_p"],
                            n_positions=self.config["model"]["n_positions"],
                            )
        self.model.load_state_dict(torch.load(self.cfg['load_model'], weights_only=True))
    
    def get_next_move(self, chess_board_fen, action_mask, move_hist_san, move_hist_uci):
        # Convert move history from UCI notation to SAN:
        input_string = "<bos>"
        if move_hist_san:
            for move in move_hist_san:
                input_string += " " + move

        prediction = self.model.model_next_move(
            input_string, 
            stop_at_next_move=True, 
            temperature=0.2,
            )
        
        pred_move_list = prediction.split(" ")
        # print(f'Prediced move sequence:\t{pred_move_list}')

        next_move = "0000"
        next_move_pred = pred_move_list[-1]
        
        chess_board_from_fen = chess.Board(fen=chess_board_fen)
        if next_move_pred != self.tokenizer.eos_token:
            try:
                move = chess_board_from_fen.parse_san(next_move_pred)
                if move in chess_board_from_fen.legal_moves:
                    next_move = chess_board_from_fen.uci(move=move)
            except chess.IllegalMoveError:
                print('illegal move error ...')
        else: #next_move_pred == self.tokenizer.eos_token:
            # if len(pred_move_list) > 2:
            try:
                move = chess_board_from_fen.parse_san(pred_move_list[-2])
                next_move = chess_board_from_fen.uci(move=move)
            except chess.IllegalMoveError:
                print('illegal move error ...')

        return next_move
    
    def shut_down(self):
        return


# Player that utilizes a Transformer Decoder-Only model trained for move selection:
class EncoderChessformerPlayer(Player):
    def __init__(self, name='player_0'):
        self.player_name=name
        self.player_type='encformer'

    def load_model(self, cfg):
        # Load model:
        """
        Load model for inference.

        Args:

            CONFIG (dict): The configuration of the model.

        Returns:

            torch.nn.Module: The model.
        """

        # Model
        CONFIG = cfg
        model = ChessTransformerEncoder(CONFIG)
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(DEVICE)

        # # Load checkpoint:
        # checkpoint_folder = (
        #     pathlib.Path(__file__).parent.parent.resolve() / "checkpoints" / CONFIG.NAME
        # )
        
        checkpoint = torch.load(
            os.path.join("./chess_transformer/chess_transformer", CONFIG['CHECKPOINT_FOLDER'], CONFIG['TRAINING_CHECKPOINT']),
            weights_only=True,
        )
        # start_epoch = 50
        model.load_state_dict(checkpoint["model_state_dict"])
        # print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)

        self.model = model

    def get_next_move(self, chess_board_fen, action_mask, move_hist_san, move_hist_uci):
        chess_board = chess.Board(fen=chess_board_fen)
        next_move = model_next_move(model=self.model, board=chess_board, use_amp=True, k=1)
        # print(next_move)
        return next_move





def print_game_log(game_log_dict):
    print(f'GAME LOG:')
    for key, item in game_log_dict.items():
        print(f"\t{key}:\t{item}")

    return

    
def play_chess_game(player_0: Player=RandomPlayer, player_1: Player=RandomPlayer):
    env = chess_v6.env(render_mode="ansi")
    env.reset(seed=42)

    players_in_game = {}

    game_log = {
        'player_0': None,
        'player_1': None,
        'uci_moves': [],
        'san_moves': [],
        'game_result': None,
        'termination': None
        }
    
    # Player white:
    players_in_game[player_0.player_name] = player_0
    game_log[player_0.player_name] = player_0.player_type
    
    # Player Black:
    players_in_game[player_1.player_name] = player_1
    game_log[player_1.player_name] = player_1.player_type

    # print(players_in_game)
    # print_game_log(game_log)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        # print_game_log(game_log)

        if termination or truncation:
            action = None

        else:
            obs = observation['observation']
            mask = observation["action_mask"]
            
            uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask, move_hist_san=game_log['san_moves'], move_hist_uci=game_log['uci_moves'])

            # print(f"################# {agent}:\t{uci_move}")
            # print(f"game_log['uci_moves']\t{game_log['uci_moves']}")

            pz_legal_actions = chess_utils.legal_moves(env.env.board)
            uci_move_to_action = {"0000": None}
            for possible_action in pz_legal_actions:
                possible_chess_move = chess_utils.action_to_move(env.env.board,possible_action, int(str(agent[-1])))
                uci_move_to_action[possible_chess_move.uci()] = possible_action

                
            action = uci_move_to_action[uci_move]
            

            game_log['uci_moves'].append(uci_move)

            chess_move = chess.Move.from_uci(uci_move)
            san_move = env.env.board.san(chess_move)
            game_log['san_moves'].append(san_move)
        
        env.step(action)

        outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
        if outcome is not None:
                game_log['game_result'] = outcome.result()
                game_log['termination'] = outcome.termination
        # if env.env.board.is_game_over():
        #     outcome = chess.Board(fen=env.env.board.fen()).outcome()
        #     if outcome is not None:
        #             game_log['game_result'] = outcome.result()
        #             game_log['termination'] = outcome.termination
    
    # # print(game_log)
    # print_game_log(game_log)
    # print('\n\n')
    

    env.close()

    if isinstance(player_0, (StockfishPlayer, FairyStockfishPlayer)):
        player_0.shut_down()
    if isinstance(player_1, (StockfishPlayer, FairyStockfishPlayer)):
        player_1.shut_down()
    
    return game_log


def result_2_winner_idx(result):
    """
    returns white=0, black=1
    """
    if result == "1-0":
        return 0
    if result == "0-1":
        return 1
    return None



# NOTE: ELO rating for the different skill_levels of Stockfish can be found at: https://github.com/official-stockfish/Stockfish/commit/a08b8d4
# ELO ratings for skill level 1-20 approx. equal 1300-3000 ??? Otherwise look at the arg='UCI_Elo', which overwrites the skill-level value if it is given???

if __name__ == '__main__':

    cfg_dict = {'config': "./chessformers/configs/config_pgn_checkmate.yaml",
                    'load_model': "./chessformers/model/pgn_checkmate_dataset_lr25e-6/chessformer_epoch_50.pth",
                    'tokenizer': "./chessformers/vocabs/kaggle2_vocab.txt"}
    
    n_games_per_color = 50

    methods = ["decformer", "encformer", "astar"]

    # opponent_names = ["random", "stockfish", "fairystockfish"]
    opponent_name = "random" #"fairystockfish" # options=["stockfish", "fairystockfish"]

    for method in methods:

        save_dir = "./test_results/compare_methods_to_fairystockfish/"
        save_file = "performance_eval_" + method + "_vs_" + opponent_name + "_" + str(time.time_ns()) + ".json"

        # AI win stats for white and black:
        ai_win_stats = [0, 0]
        ai_loss_stats = [0, 0]

        engine_skill_level = 1
        astar_max_search_depth = 2

        for ai_player_idx in range(2):
            for game_i in tqdm(range(n_games_per_color)):

                game_log = {
                    'player_0': None,
                    'player_1': None,
                    'uci_moves': [],
                    'san_moves': [],
                    'game_result': None,
                    'termination': None
                    }

                opponent_idx = int(not ai_player_idx)
                # # stockfish = StockfishPlayer(name='player_' + str(opponent_idx), skill_level=1)
                # fairystockfish = FairyStockfishPlayer(name='player_' + str(opponent_idx), skill_level=1)

                opponent_player = None #Player()
                if opponent_name == "random":
                    opponent_player = RandomPlayer(name="player_" + str(opponent_idx))
                elif opponent_name == "stockfish":
                    opponent_player = StockfishPlayer(name='player_' + str(opponent_idx), skill_level=engine_skill_level)
                elif opponent_name == "fairystockfish":
                    opponent_player = FairyStockfishPlayer(name='player_' + str(opponent_idx), skill_level=engine_skill_level)


                if method == "astar":
                    astar = AStarPlayer(name='player_' + str(ai_player_idx), max_search_depth=astar_max_search_depth)
                    game_log = play_chess_game(player_0=opponent_player, player_1=astar)
                elif method == "decformer":
                    decoderchessformer = DecoderChessformerPlayer(name='player_' + str(ai_player_idx))
                    decoderchessformer.load_model(cfg_dict)
                    game_log = play_chess_game(player_0=opponent_player, player_1=decoderchessformer)
                elif method== "encformer":
                    encoderchessformer = EncoderChessformerPlayer(name='player_' + str(ai_player_idx))
                    encoderchessformer.load_model(cfg=import_config(model_config_name="CT-E-20_inference", run_number=7))
                    game_log = play_chess_game(player_0=opponent_player, player_1=encoderchessformer)

                # opponent_idx = int(not ai_player_idx)
                # # stockfish = StockfishPlayer(name='player_' + str(opponent_idx), skill_level=1)
                # fairystockfish = FairyStockfishPlayer(name='player_' + str(opponent_idx), skill_level=1)
                # # random = RandomPlayer(name="player_" + str(opponent_idx))

                winner_idx = result_2_winner_idx(game_log['game_result'])
                if winner_idx is not None:
                    if winner_idx == ai_player_idx:
                        ai_win_stats[winner_idx] += 1
                    else:
                        ai_loss_stats[ai_player_idx] += 1

                with open(save_dir + save_file, 'a') as outfile:
                    entry = {
                        'player_white': game_log['player_0'],
                        'player_black': game_log['player_1'],
                        'game_result': game_log['game_result'],
                        'termination': str(game_log['termination']),
                        'uci_moves': game_log['uci_moves'],
                        'san_moves': game_log['san_moves']
                        }
                    # print_game_log(entry)
                    json.dump(entry, outfile)
                    outfile.write('\n')
                    outfile.close()
            

        print(f"==================== {method} ====================")
        ai_player_name = 'player_' + str(ai_player_idx)
        opponent_player_name = 'player_' + str(opponent_idx)
        print(f"WIN/LOSS Stats for {game_log[ai_player_name]} VS {game_log[opponent_player_name]}:")
        print(f"\twins as white: {ai_win_stats[0]}\n\twins as black: {ai_win_stats[1]}")
        print(f"\tloss as white: {ai_loss_stats[0]}\n\tloss as black: {ai_loss_stats[1]}")




    
        
        