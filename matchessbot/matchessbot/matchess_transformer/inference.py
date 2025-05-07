import os

import torch

import torch.nn.functional as F
import numpy as np

import copy

import chess
import chess.engine


import enum

from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils


from matchess_transformer.config import import_config
from matchess_transformer.dataset import MATChessDataset
from matchess_transformer.model import MATChessTransformerEncoder, ChessTransformerEncoder, LabelSmoothedCE, huber_loss
from matchess_transformer.tokenizer import Tokenizer
from matchess_transformer.vocab_uci_dicts import CHESS_PIECE_AGENTS, UCI_MOVES
from matchess_transformer.reward import Reward, RewardType


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


class MATChessGameState:
    white_chess_piece_agents_start_pos = {
        'king': chess.E1,
        'queen': chess.D1,
        'rook0': chess.A1,
        'rook7': chess.H1,
        'knight1': chess.B1,
        'knight6': chess.G1,
        'bishop2': chess.C1,
        'bishop5': chess.F1,
        'pawn0': chess.A2,
        'pawn1': chess.B2,
        'pawn2': chess.C2,
        'pawn3': chess.D2,
        'pawn4': chess.E2,
        'pawn5': chess.F2,
        'pawn6': chess.G2,
        'pawn7': chess.H2
        }
    
    black_chess_piece_agents_start_pos = {
        'king': chess.E8,
        'queen': chess.D8,
        'rook0': chess.A8,
        'rook7': chess.H8,
        'knight1': chess.B8,
        'knight6': chess.G8,
        'bishop2': chess.C8,
        'bishop5': chess.F8,
        'pawn0': chess.A7,
        'pawn1': chess.B7,
        'pawn2': chess.C7,
        'pawn3': chess.D7,
        'pawn4': chess.E7,
        'pawn5': chess.F7,
        'pawn6': chess.G7,
        'pawn7': chess.H7
        }
    
    reward_list_team_order = ['white', 'black']
    reward_list_agent_order = ['king', 'queen', 'rook0', 'rook7', 'knight1', 'knight6', 'bishop2', 'bishop5', 'pawn0', 'pawn1', 'pawn2', 'pawn3', 'pawn4', 'pawn5', 'pawn6', 'pawn7']
    
    def __init__(self):
        self.white_chess_piece_agents_pos = {
            'king': chess.E1,
            'queen': chess.D1,
            'rook0': chess.A1,
            'rook7': chess.H1,
            'knight1': chess.B1,
            'knight6': chess.G1,
            'bishop2': chess.C1,
            'bishop5': chess.F1,
            'pawn0': chess.A2,
            'pawn1': chess.B2,
            'pawn2': chess.C2,
            'pawn3': chess.D2,
            'pawn4': chess.E2,
            'pawn5': chess.F2,
            'pawn6': chess.G2,
            'pawn7': chess.H2
        }
        self.black_chess_piece_agents_pos = {
            'king': chess.E8,
            'queen': chess.D8,
            'rook0': chess.A8,
            'rook7': chess.H8,
            'knight1': chess.B8,
            'knight6': chess.G8,
            'bishop2': chess.C8,
            'bishop5': chess.F8,
            'pawn0': chess.A7,
            'pawn1': chess.B7,
            'pawn2': chess.C7,
            'pawn3': chess.D7,
            'pawn4': chess.E7,
            'pawn5': chess.F7,
            'pawn6': chess.G7,
            'pawn7': chess.H7
        }

        self.state_attributes_per_agent_before_action = {
            'white': {
                'king': None,
                'queen': None,
                'rook0': None,
                'rook7': None,
                'knight1': None,
                'knight6': None,
                'bishop2': None,
                'bishop5': None,
                'pawn0': None,
                'pawn1': None,
                'pawn2': None,
                'pawn3': None,
                'pawn4': None,
                'pawn5': None,
                'pawn6': None,
                'pawn7': None
            },
            'black': {
                'king': None,
                'queen': None,
                'rook0': None,
                'rook7': None,
                'knight1': None,
                'knight6': None,
                'bishop2': None,
                'bishop5': None,
                'pawn0': None,
                'pawn1': None,
                'pawn2': None,
                'pawn3': None,
                'pawn4': None,
                'pawn5': None,
                'pawn6': None,
                'pawn7': None
            }
        }

        self.agent_order_in_state_attr_lists = []
        for team_name in self.reward_list_team_order:
                for agent_name in self.reward_list_agent_order:
                    self.agent_order_in_state_attr_lists.append(team_name + '_' + agent_name)


    def init_new_matchess_game(self):
        self.board = chess.Board()
        self.board.reset()

        # Create structure for storing the state-action paris:
        self.all_board_fens = []
        self.all_uci_moves = []
        self.all_white_chess_pieces_pos = []
        self.all_black_chess_pieces_pos = []

        self.state_action_captures = []

        # Set the first entry in the state lists to the starting state (used to generate the state-action-reward sequences):
        board_fen = self.board.fen()
        self.all_board_fens.append(board_fen)

        self.all_white_chess_pieces_pos.append(dict(self.white_chess_piece_agents_pos))
        self.all_black_chess_pieces_pos.append(dict(self.black_chess_piece_agents_pos))

        self.update_agent_state_attributes(chess_board=self.board)
        
        self.all_state_attributes_per_agent_before_action = []

        self.all_state_attributes_per_agent_before_action.append(copy.deepcopy(self.state_attributes_per_agent_before_action))
        self.all_global_state_attributes_before_action = []

        global_state_attributes_before_action = {
                'possible_en_pessant_target_present': False,
                'ep_square_idx': -1,
                'castling_rights_kingside': False,
                'castling_rights_queenside': False,
                'opponent_castling_rights_kingside': False,
                'opponent_castling_rights_queenside': False,
                'legal_uci_moves': [],
            }

        if self.board.ep_square is not None:
            global_state_attributes_before_action['possible_en_pessant_target_present'] = True
            global_state_attributes_before_action['ep_square_idx'] = self.board.ep_square
        
        # Update global_state_attributes_before_action with castling rights for the input state (for both the active player and their opponent)
        if self.board.has_kingside_castling_rights(self.board.turn):
            global_state_attributes_before_action['castling_rights_kingside'] = True

        if self.board.has_queenside_castling_rights(self.board.turn):
            global_state_attributes_before_action['castling_rights_queenside'] = True

        if self.board.has_kingside_castling_rights(not self.board.turn):
            global_state_attributes_before_action['opponent_castling_rights_kingside'] = True

        if self.board.has_queenside_castling_rights(not self.board.turn):
            global_state_attributes_before_action['opponent_castling_rights_queenside'] = True
        
        global_state_attributes_before_action['legal_uci_moves'] = [legal_move.uci() for legal_move in self.board.legal_moves]
        self.all_global_state_attributes_before_action.append(copy.deepcopy(global_state_attributes_before_action))
        
        return
    
    def update_agent_state_attributes(self, chess_board: chess.Board):
        # The following code stores the state attributes as sets of squares, which are internally represented as a 64 bit integer masks of the included squares in the set:
        for key_team, item_team in self.state_attributes_per_agent_before_action.items():
            for key, item in item_team.items():
                agent_pos = None
                if key_team == 'white':
                    agent_pos = self.white_chess_piece_agents_pos[key]
                elif key_team == 'black':
                    agent_pos = self.black_chess_piece_agents_pos[key]

                if agent_pos is not None:
                
                    agent_color = chess_board.color_at(square=agent_pos)
                    opponent_color = chess.WHITE if agent_color==chess.BLACK else chess.BLACK

                    agent_attack_squares = chess_board.attacks(square=agent_pos)
                    agent_under_attacked_from_square = chess_board.attackers(color=opponent_color, square=agent_pos)
                    agent_defended_by_agent_on_square = chess_board.attackers(color=agent_color, square=agent_pos)
                    agent_chess_piece = chess_board.piece_at(square=agent_pos)
                    same_piece_type_count = len(chess_board.pieces(piece_type=agent_chess_piece.piece_type, color=agent_color))

                    agent_attributes_dict = {
                        'is_alive': True,
                        'agent_pos': agent_pos,
                        'attack_squares': agent_attack_squares,
                        'attacked_by_agents_on_square': agent_under_attacked_from_square,
                        'defended_by_agents_on_square': agent_defended_by_agent_on_square,
                        'same_piece_type_count': same_piece_type_count,
                    }

                    self.state_attributes_per_agent_before_action[key_team][key] = dict(agent_attributes_dict)
                else:
                    agent_attributes_dict = {
                        'is_alive': False,
                        'agent_pos': None,
                        'attack_squares': chess.SquareSet(chess.BB_EMPTY),
                        'attacked_by_agents_on_square': chess.SquareSet(chess.BB_EMPTY),
                        'defended_by_agents_on_square': chess.SquareSet(chess.BB_EMPTY),
                        'same_piece_type_count': 0,
                    }

                    self.state_attributes_per_agent_before_action[key_team][key] = dict(agent_attributes_dict)
        return
    
    def matchess_env_step(self, uci_move):
        # Update Individual reward state attributes:
        capture_with_action = {
            'move_idx': len(self.all_uci_moves),
            'piece_was_captured': False,
            'capturing_agent_team_color': self.board.turn,
            'capturing_agent_id': None,
            'captured_agent_id': None,
            'captured_agent_type': None,
        }

        # Update the positions of the chess piece agents:
        move_piece_id = None
        chess_move = chess.Move.from_uci(uci_move)
        if self.board.turn == chess.WHITE:
            for key, item in self.white_chess_piece_agents_pos.items():
                if chess_move.from_square == item:
                    self.white_chess_piece_agents_pos[key] = chess_move.to_square
                    move_piece_id = str(key)
                    break

            for key, item in self.black_chess_piece_agents_pos.items():
                if chess.Move.from_uci(uci_move).to_square == item:
                    self.black_chess_piece_agents_pos[key] = None

                    capture_with_action['piece_was_captured'] = True
                    capture_with_action['capturing_agent_id'] = move_piece_id
                    capture_with_action['captured_agent_id'] = str(key)
                    capture_with_action['captured_agent_type'] = self.board.piece_type_at(chess_move.to_square)

                    break
        else:
            for key, item in self.black_chess_piece_agents_pos.items():
                if chess_move.from_square == item:
                    self.black_chess_piece_agents_pos[key] = chess_move.to_square
                    move_piece_id = str(key)
                    break

            for key, item in self.white_chess_piece_agents_pos.items():
                if chess_move.to_square == item:
                    self.white_chess_piece_agents_pos[key] = None

                    capture_with_action['piece_was_captured'] = True
                    capture_with_action['capturing_agent_id'] = move_piece_id
                    capture_with_action['captured_agent_id'] = str(key)
                    capture_with_action['captured_agent_type'] = self.board.piece_type_at(chess_move.to_square)

                    break
        
        self.state_action_captures.append(dict(capture_with_action)) # Used for computing the next-state rewards

        # Handle updating the position of the rook agents for castling moves:
        if self.board.is_castling(chess_move):
            # update rook position for castling moves:
            if self.board.is_kingside_castling(chess_move):
                if self.board.turn == chess.WHITE:
                    self.white_chess_piece_agents_pos['rook7'] = chess.F1
                else:
                    self.black_chess_piece_agents_pos['rook7'] = chess.F8
            elif self.board.is_queenside_castling(chess_move):
                if self.board.turn == chess.WHITE:
                    self.white_chess_piece_agents_pos['rook0'] = chess.D1
                else:
                    self.black_chess_piece_agents_pos['rook0'] = chess.D8
        
        # Handle updating position of pawns captured with en passant:
        if self.board.is_en_passant(chess_move):
            ep_captured_agent_rank = chess.square_rank(chess_move.from_square)
            ep_captured_agent_file = chess.square_file(chess_move.to_square)
            ep_captured_agent_square = chess.square(file_index=ep_captured_agent_file, rank_index=ep_captured_agent_rank)
            if self.board.turn == chess.WHITE:
                for key, item in self.black_chess_piece_agents_pos.items():
                    if item == ep_captured_agent_square:
                        self.black_chess_piece_agents_pos[key] = None
            elif self.board.turn == chess.BLACK:
                for key, item in self.white_chess_piece_agents_pos.items():
                    if item == ep_captured_agent_square:
                        self.white_chess_piece_agents_pos[key] = None


        # Apply the move to the board in game:
        self.board.push(chess_move)
        
        # Store a copy of all the moves in UCI format (convenient when generating the state-action-reward sequence)
        self.all_uci_moves.append(chess_move.uci())

        # Get the next board state (represented as a fen string) and append it to the list of all board states:
        board_fen = self.board.fen()
        self.all_board_fens.append(board_fen)

        # Update the global board state attributes before next action:
        global_state_attributes_before_action = {
            'possible_en_pessant_target_present': False,
            'ep_square_idx': -1,
            'castling_rights_kingside': False,
            'castling_rights_queenside': False,
            'opponent_castling_rights_kingside': False,
            'opponent_castling_rights_queenside': False,
            'legal_uci_moves': [],
        }

        if self.board.ep_square is not None:
            global_state_attributes_before_action['possible_en_pessant_target_present'] = True
            global_state_attributes_before_action['ep_square_idx'] = self.board.ep_square
        
        # Update global_state_attributes_before_action with castling rights for the input state (for both the active player and their opponent)
        if self.board.has_kingside_castling_rights(self.board.turn):
            global_state_attributes_before_action['castling_rights_kingside'] = True

        if self.board.has_queenside_castling_rights(self.board.turn):
            global_state_attributes_before_action['castling_rights_queenside'] = True

        if self.board.has_kingside_castling_rights(not self.board.turn):
            global_state_attributes_before_action['opponent_castling_rights_kingside'] = True

        if self.board.has_queenside_castling_rights(not self.board.turn):
            global_state_attributes_before_action['opponent_castling_rights_queenside'] = True
        global_state_attributes_before_action['legal_uci_moves'] = [legal_move.uci() for legal_move in self.board.legal_moves]
        self.all_global_state_attributes_before_action.append(copy.deepcopy(global_state_attributes_before_action))
        
        self.update_agent_state_attributes(chess_board=self.board)
        self.all_state_attributes_per_agent_before_action.append(copy.deepcopy(self.state_attributes_per_agent_before_action))


        # Append the new positions of all the pieces to the lists of all the positions of the pieces:
        self.all_white_chess_pieces_pos.append(dict(self.white_chess_piece_agents_pos))
        self.all_black_chess_pieces_pos.append(dict(self.black_chess_piece_agents_pos))


    
    def get_game_state(self):
        team_name = 'white' if self.board.turn == chess.WHITE else 'black'

        agent_pos_square_idx_current_state = []
        # agent_pos_square_idx_reward_state = []
        for agent_name in self.reward_list_agent_order:
            agent_pos_for_state_action_pair_idx = self.all_state_attributes_per_agent_before_action[-1][team_name][agent_name]['agent_pos']
            if agent_pos_for_state_action_pair_idx is not None:
                agent_pos_square_idx_current_state.append(agent_pos_for_state_action_pair_idx)
            else:
                agent_pos_square_idx_current_state.append(-1)

        return {
            'state': {
                'fen': self.all_board_fens[-1],
                'state_idx': len(self.all_board_fens) - 1,
                'team_color': team_name,
                'agent_pos_square_idx': agent_pos_square_idx_current_state,
                'possible_en_pessant_target_present': int(self.all_global_state_attributes_before_action[-1]['possible_en_pessant_target_present']),
                'ep_square_idx': self.all_global_state_attributes_before_action[-1]['ep_square_idx'],
                'castling_rights_kingside': int(self.all_global_state_attributes_before_action[-1]['castling_rights_kingside']),
                'castling_rights_queenside': int(self.all_global_state_attributes_before_action[-1]['castling_rights_queenside']),
                'opponent_castling_rights_kingside': int(self.all_global_state_attributes_before_action[-1]['opponent_castling_rights_kingside']),
                'opponent_castling_rights_queenside': int(self.all_global_state_attributes_before_action[-1]['opponent_castling_rights_queenside']),
            }
        }
    
    # def update_game_state(self, action):

    #     raise NotImplementedError
    


class ModelType(enum.Enum):
    NONE = -1,
    MABC = 0,
    MARL = 1,



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


# MATChessTransformerEncoder chess player:
class MATChessPlayer(Player):
    chess_piece_agent_ids = ['king', 'queen', 'rook0', 'rook7', 'knight1', 'knight6', 'bishop2', 'bishop5', 'pawn0', 'pawn1', 'pawn2', 'pawn3', 'pawn4', 'pawn5', 'pawn6', 'pawn7']
    reward_types_len = 11
    # randomize_agent_priority = np.array(range(1.0, 12.0, 1.1), dtype=float)
    # random.Random(42).shuffle(randomize_agent_priority)
    # print(f"\nRandomized agent priority{randomize_agent_priority}")
    chess_piece_agent_reward_weights = [[1.0] * reward_types_len] * len(chess_piece_agent_ids)
    
    valid_model_types = [ModelType.MABC.name, ModelType.MARL.name]

    voting_scheme = 'democracy'

    def __init__(self, name='player_0'):
        self.player_name=name
        self.player_type='matchess-encformer'

        self.matchess_game_state = MATChessGameState()
        self.matchess_game_state.init_new_matchess_game()

        self.tokenizer = Tokenizer()

        self.model_type = ModelType.NONE

        self.n_agents = 0


    def load_model(self, CONFIG):
        # Load model:
        """
        Load model for inference.

        Args:

            CONFIG (dict): The configuration of the model.

        Returns:

            torch.nn.Module: The model.
        """

        # Model
        # CONFIG = cfg
        if CONFIG['NAME'].startswith(("MATChessFormer-Homogeneous-")):
            self.model_type = ModelType.MABC
            model = ChessTransformerEncoder(CONFIG)
        elif CONFIG['NAME'].startswith(("MATChessFormer-Heterogeneous-")):
            model = MATChessTransformerEncoder(CONFIG)
            self.model_type = ModelType.MARL
        else:
            raise NotImplementedError

        self.n_agents = CONFIG['N_AGENTS']
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.DEVICE)

        # # Load checkpoint:
        # checkpoint_folder = (
        #     pathlib.Path(__file__).parent.parent.resolve() / "checkpoints" / CONFIG.NAME
        # )
        
        # checkpoint = torch.load(
        #     os.path.join("./matchess_transformer/matchchess_transformer", CONFIG['CHECKPOINT_FOLDER'], CONFIG['TRAINING_CHECKPOINT']),
        #     weights_only=True,
        # )
        checkpoint = torch.load(
            os.path.join(CONFIG['CHECKPOINT_FOLDER'], CONFIG['LOAD_MODEL_CHECKPOINT']),
            weights_only=True,
        )
        

        # start_epoch = 50
        model.load_state_dict(checkpoint["model_state_dict"])
        # print("\nLoaded checkpoint from epoch %d.\n" % start_epoch)

        self.model = model

    def get_next_move(self, chess_board_fen, action_mask, move_hist_san, move_hist_uci):
        # chess_board = chess.Board(fen=chess_board_fen)
        next_move = self.model_next_move(use_amp=True, k=1, show_board=False)
        # print(next_move)
        return next_move

    def model_next_move(
        self,
        # model,
        # board,
        use_amp,
        k,
        # model_name=None,
        # opponent_name=None,
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
        agent_token_move_preds = []
        # Get predictions
        self.model.eval()
        with torch.no_grad():
            # Get list of legal moves for the current position
            legal_moves = [move.uci() for move in self.matchess_game_state.board.legal_moves]

            # Get model inputs
            
            game_state = self.matchess_game_state.get_game_state()

            # print(game_state)

            model_inputs = self.tokenizer.encode_model_input_v2(game_state=game_state, chess_piece_agent_ids=self.chess_piece_agent_ids, chess_piece_agent_reward_weights=self.chess_piece_agent_reward_weights)

            # print(model_inputs)

            # for key, value in model_inputs.items():
            #     print(f"key: {key}\tvalue.shape: {value.shape}")

            # Move to default device
            for key in model_inputs:
                model_inputs[key] = model_inputs[key].to(self.DEVICE)

            # (Direct) Move prediction models
            # if model.code in {"E", "ED"}:
            with torch.autocast(
                device_type=self.DEVICE.type, dtype=torch.float16, enabled=use_amp
            ):
                if self.model_type == ModelType.MABC:
                    predicted_moves = self.model(model_inputs)
                elif self.model_type == ModelType.MARL:
                    predicted_moves, predicted_rewards = self.model(model_inputs)
                else:
                    raise NotImplementedError(f"The loaded model is not a valid model type! Valid model types are: {self.valid_model_types}")
                
            # Filter out move indices corresponding to illegal moves
            legal_move_indices = [UCI_MOVES[m] for m in legal_moves]

            move_votes = {}

            for agent_idx in range(self.n_agents):
                # agents_predicted_moves = predicted_moves[:, agent_idx, :]  # (1, move_vocab_size)
                # print(agents_predicted_moves.shape)

                # Perform top-k sampling to obtain a legal predicted move
                if self.voting_scheme == 'democracy':
                    legal_move_index = topk_sampling(
                        logits= predicted_moves[:, agent_idx, legal_move_indices], #predicted_moves[:, legal_move_indices],
                        k=k,
                    ).item()
                    
                    # Keep track of all agent's votes:
                    agent_vote = legal_moves[legal_move_index]
                    if not agent_vote in move_votes.keys():
                        move_votes[agent_vote] = 0
                    move_votes[agent_vote] += 1

            # model_move = legal_moves[legal_move_index]
            model_move = max(move_votes, key=move_votes.get)

            # self.matchess_game_state.update_game_state()

            return model_move #decode_tokens_to_uci(model_move) # board
        


def play_chess_game(player_0: Player=RandomPlayer, player_1: Player=RandomPlayer):
    # env = chess_v6.env(render_mode="ansi")
    env = chess_v6.env(render_mode="human")
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
        
        if isinstance(player_0, MATChessPlayer):
            player_0.matchess_game_state.matchess_env_step(uci_move)
        if isinstance(player_1, MATChessPlayer):
            player_1.matchess_game_state.matchess_env_step(uci_move)

        env.step(action)

        outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
        if outcome is not None:
            game_log['game_result'] = outcome.result()
            game_log['termination'] = outcome.termination

    env.close()

    # if isinstance(player_0, (FairyStockfishPlayer)):
    #     player_0.shut_down()
    # if isinstance(player_1, (FairyStockfishPlayer)):
    #     player_1.shut_down()
    
    return game_log


def print_game_log(game_log_dict):
    print(f'GAME LOG:')
    for key, item in game_log_dict.items():
        print(f"\t{key}:\t{item}")

    return

if __name__=='__main__':
    # TODO: test that the MATChess player model can be used to generate an episode with the current policy to generate on-policy self-play RL training data and save in a PGN file???

    # TODO: Use the PGN file to train the value head of the model with RL???
    # TODO: Log: rewards, per-reward loss, chosen action, value loss, tok-k legal_moves accuracy, for all agents during traing???

    matchessformer = MATChessPlayer(name='player_0')
    cfg = import_config(model_config_name="MATChessFormer-Homogeneous-20", run_number=2, inference=True)
    matchessformer.load_model(cfg)

    # opponent_player = RandomPlayer(name="player_1")
    # opponent_player = FairyStockfishPlayer(name="player_1")

    matchessformer_opponent = MATChessPlayer(name='player_1')
    # cfg = import_config(model_config_name="MATChessFormer-Homogeneous-20", run_number=2, inference=True)
    matchessformer_opponent.load_model(cfg)

    game_log = play_chess_game(player_0=matchessformer, player_1=matchessformer_opponent)

    print_game_log(game_log)

    

