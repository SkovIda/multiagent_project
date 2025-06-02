import os

import torch
import torch.nn.functional as F

import copy
import enum

import chess

from .matchess_transformer.matchess_transformer.config import import_config, ROS_CFG_HETEROGENEOUS_MAS_INFERENCE, ROS_CFG_IMITATIVE_RL_INFERENCE, ROS_CFG_STANDARD_IL_INFERENCE
from .matchess_transformer.matchess_transformer.model import MATChessTransformerEncoder, ChessTransformerEncoder #, LabelSmoothedCE, huber_loss
from .matchess_transformer.matchess_transformer.tokenizer import Tokenizer
from .matchess_transformer.matchess_transformer.vocab_uci_dicts import CHESS_PIECE_AGENTS, UCI_MOVES


class ModelType(enum.Enum):
    NONE = -1,
    MABC = 0,
    MARL = 1,
    SIL = 2,
    IRL = 3,

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
        self.init_new_matchess_game()
        return
    
    def init_new_matchess_game(self):
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
                
                    agent_color = self.board.color_at(square=agent_pos) #chess_board.color_at(square=agent_pos)
                    if agent_color is None:
                        print(chess_board)
                        raise AttributeError(f"Agent at square {agent_pos} has color={agent_color}")
                        
                    opponent_color = chess.WHITE if agent_color==chess.BLACK else chess.BLACK
                    
                    agent_attack_squares = self.board.attacks(square=agent_pos)
                    agent_under_attacked_from_square = self.board.attackers(color=opponent_color, square=agent_pos)
                    agent_defended_by_agent_on_square = self.board.attackers(color=agent_color, square=agent_pos)
                    agent_chess_piece = self.board.piece_at(square=agent_pos)
                    same_piece_type_count = len(self.board.pieces(piece_type=agent_chess_piece.piece_type, color=agent_color))

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
    





############################## MATChess Transformer Model Wrapper ##############################
class MATChessTransformer:
    chess_piece_agent_ids = ['king', 'queen', 'rook0', 'rook7', 'knight1', 'knight6', 'bishop2', 'bishop5', 'pawn0', 'pawn1', 'pawn2', 'pawn3', 'pawn4', 'pawn5', 'pawn6', 'pawn7']
    reward_types_len = 11
    # randomize_agent_priority = np.array(range(1.0, 12.0, 1.1), dtype=float)
    # random.Random(42).shuffle(randomize_agent_priority)
    # print(f"\nRandomized agent priority{randomize_agent_priority}")
    chess_piece_agent_reward_weights = [[1.0] * reward_types_len] * len(chess_piece_agent_ids)
    
    valid_model_types = [ModelType.MABC.name, ModelType.MARL.name, ModelType.IRL.name, ModelType.SIL.name]

    voting_scheme = 'democracy'

    model_checkpoint_path_prefix = "matchess_transformer/"

    def __init__(self):
        # self.player_type='matchess-rl-encformer'
        self.model_name = 'matchess_transformer'

        self.matchess_game_state = MATChessGameState()
        self.matchess_game_state.init_new_matchess_game()

        self.tokenizer = Tokenizer()

        self.model_type = ModelType.NONE

        self.n_agents = 0


    def load_model(self, model_type: ModelType=ModelType.MARL):
        # Load model:
        """
        Load model for inference.

        Args:

            CONFIG (dict): The configuration of the model.

        Returns:

            torch.nn.Module: The model.
        """

        # Init MATChess Transformer Model
        self.model_type = model_type
        if self.model_type == ModelType.MARL:
            #config_path = os.path.join(self.model_checkpoint_path_prefix, )
            CONFIG = ROS_CFG_HETEROGENEOUS_MAS_INFERENCE #import_config(model_config_name="MATChessFormer-Heterogeneous-20", run_number=2, inference=True)
            model = MATChessTransformerEncoder(CONFIG)
        elif self.model_type == ModelType.IRL:
            CONFIG = ROS_CFG_IMITATIVE_RL_INFERENCE #import_config(model_config_name="MATChessFormer-Heterogeneous-20", run_number=2, inference=True)
            model = MATChessTransformerEncoder(CONFIG)
        elif self.model_type == ModelType.MABC or self.model_type == ModelType.SIL:
            CONFIG = ROS_CFG_STANDARD_IL_INFERENCE #import_config(model_config_name="MATChessFormer-Homogeneous-20", run_number=1, inference=True)
            model = ChessTransformerEncoder(CONFIG)
        else:
            raise NotImplementedError(f"Unknown model type. Valid model types are: {self.valid_model_types}")

        self.model_name = self.model_name + '_' + str(self.model_type.name)

        self.n_agents = CONFIG['N_AGENTS']
        self.batch_size = CONFIG['BATCH_SIZE']
        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(self.DEVICE)

        # checkpoint = torch.load(
        #     os.path.join(CONFIG['CHECKPOINT_FOLDER'], CONFIG['LOAD_MODEL_CHECKPOINT']),
        #     weights_only=True,
        # )

        # use ament package tools to get own share dir
        from ament_index_python.packages import get_package_share_directory
        checkpoint = torch.load(
            os.path.join(get_package_share_directory('matchessbot'), CONFIG['CHECKPOINT_FOLDER'], CONFIG['LOAD_MODEL_CHECKPOINT']),
            map_location=self.DEVICE
        )
        
        
        model.load_state_dict(checkpoint["model_state_dict"])
        self.model = model


    def sample_action(self,policy_logits, reward_logits, batch_size, k=1):
        k = min(k, policy_logits.shape[1])
        
        # Get indices corresponding to top-max(k) scores
        probabilities = F.softmax(policy_logits, dim=-1).unsqueeze(2)  # (N, vocab_size, 1)
        other_probabilities = F.softmax(reward_logits, dim=-1).unsqueeze(
            1
        )  # (N, 1, other_vocab_size)

        combined_probabilities = torch.bmm(probabilities, other_probabilities).view(
            k, -1
        )  # (N, vocab_size * other_vocab_size)
        _, flattened_indices = combined_probabilities.topk(
            k=k, dim=1
        )  # (N, max(k))
        policy_indices = flattened_indices // reward_logits.shape[-1]  # (N, max(k))

        return policy_indices[0, :k]
    
    def topk_sampling(self,logits, k=1):
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

    def play(self):
        next_move = self.model_next_move(use_amp=True, k=1, show_board=False)
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
            model_inputs = self.tokenizer.encode_model_input_v2(game_state=game_state, chess_piece_agent_ids=self.chess_piece_agent_ids, chess_piece_agent_reward_weights=self.chess_piece_agent_reward_weights)

            # Move to default device
            for key in model_inputs:
                model_inputs[key] = model_inputs[key].to(self.DEVICE)
            

            # with torch.autocast(
            with torch.cuda.amp.autocast(       # torch version 1.8.0
                enabled=use_amp
            ):
                
                # Filter out move indices corresponding to illegal moves
                legal_move_indices = [UCI_MOVES[m] for m in legal_moves]

                move_votes = {}

                if self.model_type == ModelType.MABC or self.model_type == ModelType.SIL:
                    predicted_moves = self.model(model_inputs)

                    for agent_idx in range(self.n_agents):
                        # Perform action sampling to obtain a legal predicted move
                        if self.voting_scheme == 'democracy':
                            legal_move_index = self.topk_sampling(
                                logits= predicted_moves[:, agent_idx, legal_move_indices], #predicted_moves[:, legal_move_indices],
                                k=k,
                            ).item()
                            
                            # Keep track of all agent's votes:
                            agent_vote = legal_moves[legal_move_index]
                            if not agent_vote in move_votes.keys():
                                move_votes[agent_vote] = 0
                            move_votes[agent_vote] += 1

                elif self.model_type == ModelType.MARL or self.model_type == ModelType.IRL:
                    predicted_moves, predicted_rewards = self.model(model_inputs)
                    
                    for agent_idx in range(self.n_agents):
                        # Perform action sampling to obtain a legal predicted move
                        if self.voting_scheme == 'democracy':
                            legal_move_index = self.sample_action(
                                policy_logits= predicted_moves[:, agent_idx, legal_move_indices], #predicted_moves[:, legal_move_indices],
                                reward_logits=predicted_rewards[:, agent_idx, :],
                                batch_size=self.batch_size,
                                k=k,
                            ).item()
                            
                            # Keep track of all agent's votes:
                            agent_vote = legal_moves[legal_move_index]
                            if not agent_vote in move_votes.keys():
                                move_votes[agent_vote] = 0
                            move_votes[agent_vote] += 1
                else:
                    raise NotImplementedError(f"The loaded model is not a valid model type! Valid model types are: {self.valid_model_types}")
            
            
            print(move_votes)

            model_move = max(move_votes, key=move_votes.get)

            return model_move
    
    def game_env_turn_color(self):
        return self.matchess_game_state.board.turn

    def game_env_legal_moves(self):
        return self.matchess_game_state.board.legal_moves
    
    def game_env_is_game_over(self, claim_draw_allowed=True):
        return self.matchess_game_state.board.is_game_over(claim_draw=claim_draw_allowed)
    
    def game_env_reset(self):
        self.matchess_game_state.init_new_matchess_game()
        return
    
    def mas_play(self):
        agent_move_votes = self.model_agent_move_votes(use_amp=True, k=1, show_board=False)
        return agent_move_votes
    
    def model_agent_move_votes(
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
            model_inputs = self.tokenizer.encode_model_input_v2(game_state=game_state, chess_piece_agent_ids=self.chess_piece_agent_ids, chess_piece_agent_reward_weights=self.chess_piece_agent_reward_weights)

            # Move to default device
            for key in model_inputs:
                model_inputs[key] = model_inputs[key].to(self.DEVICE)
            

            # with torch.autocast(
            with torch.cuda.amp.autocast(       # torch version 1.8.0
                enabled=use_amp
            ):
                
                # Filter out move indices corresponding to illegal moves
                legal_move_indices = [UCI_MOVES[m] for m in legal_moves]

                # move_votes = {}
                
                agent_votes = {}
                agent_team_color = 'white' if self.game_env_turn_color() == chess.WHITE else 'black'

                if self.model_type == ModelType.MABC or self.model_type == ModelType.SIL:
                    predicted_moves = self.model(model_inputs)

                    for agent_idx in range(self.n_agents):
                        # Perform action sampling to obtain a legal predicted move
                        if not self.matchess_game_state.all_state_attributes_per_agent_before_action[-1][agent_team_color][self.chess_piece_agent_ids[agent_idx]]['is_alive']:
                            continue

                        if self.voting_scheme == 'democracy':
                            legal_move_index = self.topk_sampling(
                                logits= predicted_moves[:, agent_idx, legal_move_indices], #predicted_moves[:, legal_move_indices],
                                k=k,
                            ).item()
                            
                            # Keep track of all agent's votes:
                            agent_votes[self.chess_piece_agent_ids[agent_idx]] = legal_moves[legal_move_index]
                            # agent_vote = legal_moves[legal_move_index]
                            # if not agent_vote in move_votes.keys():
                            #     move_votes[agent_vote] = 0
                            # move_votes[agent_vote] += 1

                    
                elif self.model_type == ModelType.MARL or self.model_type == ModelType.IRL:
                    predicted_moves, predicted_rewards = self.model(model_inputs)

                    for agent_idx in range(self.n_agents):
                        # Perform action sampling to obtain a legal predicted move
                        if not self.matchess_game_state.all_state_attributes_per_agent_before_action[-1][agent_team_color][self.chess_piece_agent_ids[agent_idx]]['is_alive']:
                            continue
                        
                        if self.voting_scheme == 'democracy':
                            legal_move_index = self.sample_action(
                                policy_logits= predicted_moves[:, agent_idx, legal_move_indices], #predicted_moves[:, legal_move_indices],
                                reward_logits=predicted_rewards[:, agent_idx, :],
                                batch_size=self.batch_size,
                                k=k,
                            ).item()
                            
                            # # Keep track of all agent's votes:
                            agent_votes[self.chess_piece_agent_ids[agent_idx]] = legal_moves[legal_move_index]

                            # agent_vote = legal_moves[legal_move_index]
                            # if not agent_vote in move_votes.keys():
                            #     move_votes[agent_vote] = 0
                            # move_votes[agent_vote] += 1
                    
                    # print(move_votes)

                    # model_move = max(move_votes, key=move_votes.get)

                    # return model_move

                else:
                    raise NotImplementedError(f"The loaded model is not a valid model type! Valid model types are: {self.valid_model_types}")
                
            # agent_votes[self.chess_piece_agent_ids[agent_idx]] = legal_moves[legal_move_index]
            return agent_votes

    
if __name__=='__main__':

    model_type = ModelType['MARL']
    model = MATChessTransformer()
    model.load_model(model_type)
