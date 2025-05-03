import os
import torch

import chess
# import json

from tqdm import tqdm
import re

from .vocab_uci_dicts import get_vocab_sizes, RANKS, FILES, SQUARES, TURN, PIECES, UCI_MOVES, BOOL, CHESS_PIECE_AGENTS, AGENT_POS
from .reward import RewardType

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



class Tokenizer:
    # pad_token_index: int = 1970
    # bos_token_index: int = 1968
    # eos_token_index: int = 1969
    # unk_token_index: int = 3

    pad_token: str = "<pad>"
    # bos_token: str = "<move>"
    # eos_token: str = "<loss>"
    # unk_token: str = "<unk>"

    
    def __init__(self, torch_device="cuda") -> None:
        if torch_device == "cuda":
            # Try using cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        return
    
    def encode(self, item, vocabulary):
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
        
    
    def encode_pgn_dataset_entry(self, pgn_dataset_entry, chess_piece_agent_ids, chess_piece_agent_reward_weights):
        """
        encode model input and training targets for dataset creation.

        Args:
            pgn_dataset_entry (dict): Dict containing the following items:
                    'pgn_filename': string with the filename of the PGN file that this datapoint was extracted from.
                    'game_id': The Lichess.org game ID, which is the last part of the URL of the game on: "Lichess.org/<game_id>".
                    'game_offset_in_pgn': the offset that can be used to extract the PGN for the game in the original PGN file.
                    'pgn_header_info': dict containing the same info as the header for that game in the PGN file,
                    'state': "state":
                        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
                        "state_idx": The idx of this state in the PGN game, where "0" is the first state in the game (the starting setup) 
                        "team_color": The color of the team whose turn it is to make a move: can be either "white" or "black"
                        "agent_pos_square_idx": list of the idx of the square wheare each of the agents are standing on. -1 if the agent is not present on the board.
                        "possible_en_pessant_target_present": true/false
                        "ep_square_idx": int in range [0,63] if possible_en_pessant_target_present=true, else -1
                        "castling_rights_kingside": true/false
                        "castling_rights_queenside": true/false
                        "opponent_castling_rights_kingside": true/false
                        "opponent_castling_rights_queenside": true/false
                    'action': The name of the chess move that the human player chose to make in UCI notation (str)
                    'rewards': A dict where each item corresponds to the rewards for all agents for a specific type of reward
                }

        Returns:

            dict: Encoded item in the dataset. Containst the encoded input to the model and the corresponding output targets
        """
        model_inputs = dict()
        board = chess.Board()
        board.set_fen(pgn_dataset_entry['state']['fen'])

        # If blacks turn, mirror the board and invert colors of the pieces => "model always thinks it is playing as white"
        if board.turn is not chess.WHITE:
            board.apply_mirror()
            agent_states = [chess.square_mirror(agent_pos) for agent_pos in agent_states]

        ##### Castling Rights:
        board_str, turn, castling_rights, ep_square, _, __ = board.fen().split() # NOTE: The ep_square output from this is wrong!: print(ep_square)
        
        castling_rights_kingside = "K" in castling_rights
        castling_rights_queenside = "Q" in castling_rights
        opponent_castling_rights_kingside = "k" in castling_rights
        opponent_castling_rights_queenside = "q" in castling_rights

        board_encoded = [0] * 64
        for square_idx, chess_piece in board.piece_map().items():
            board_encoded[self.encode(chess.SQUARE_NAMES[square_idx], SQUARES)] = self.encode(chess.Piece.symbol(chess_piece),PIECES) 

        ##### EP Square:
        if board.ep_square is not None:
            board_encoded[self.encode(chess.SQUARE_NAMES[board.ep_square], SQUARES)] = self.encode(",", PIECES)


        board_posiiton = (
            torch.IntTensor(board_encoded).unsqueeze(0)
        ) # (64)
        kingside_castling_rights = torch.IntTensor(
            [self.encode(castling_rights_kingside, vocabulary=BOOL)]
        ) # (1)
        queenside_castling_rights = torch.IntTensor(
                [self.encode(castling_rights_queenside, vocabulary=BOOL)]
        ) # (1)
        opponent_castling_rights_kingside = torch.IntTensor(
            [self.encode(opponent_castling_rights_kingside, vocabulary=BOOL)]
        ) # (1)
        opponent_castling_rights_queenside = torch.IntTensor(
            [self.encode(opponent_castling_rights_queenside, vocabulary=BOOL)]
        ) # (1)
        agent_ids = torch.IntTensor(
            self.encode(chess_piece_agent_ids,vocabulary=CHESS_PIECE_AGENTS)
        ) # (16)
        agents_pos_temp_list = [square_idx if square_idx >= 0 else self.encode(self.pad_token, vocabulary=AGENT_POS) for square_idx in pgn_dataset_entry['state']['agent_pos_square_idx']]
        agents_pos = torch.IntTensor(
            agents_pos_temp_list #pgn_dataset_entry['state']['agent_pos_square_idx']
        ) # (16)
        reward_weights = torch.FloatTensor(
            chess_piece_agent_reward_weights
        ) # Agent reward weights: (n_agents, n_reward_types)

        moves = torch.LongTensor(
            [self.encode(pgn_dataset_entry['action'], vocabulary=UCI_MOVES)] * len(chess_piece_agent_ids)
        ) # (n_agents) = (16)

        target_rewards = []
        for key, reward_weights_agent_i in pgn_dataset_entry['rewards'].items():
            target_rewards.append(reward_weights_agent_i)

        weighted_target_rewards = torch.FloatTensor(
                chess_piece_agent_reward_weights).mul(torch.FloatTensor(target_rewards).transpose(1,0)
        )

        return {
            'board_positions': board_posiiton,
            "kingside_castling_rights": kingside_castling_rights,
            "queenside_castling_rights": queenside_castling_rights,
            "opponent_castling_rights_kingside": opponent_castling_rights_kingside,
            "opponent_castling_rights_queenside": opponent_castling_rights_queenside,
            "agent_ids": agent_ids,
            "agents_pos": agents_pos,
            "reward_weights": reward_weights,
            "moves": moves,
            "rewards": weighted_target_rewards,
            "n_agents": len(chess_piece_agent_ids),
        }


    def encode_model_input(self, fen_str, agent_ids, agent_pos, agent_reward_weights):
        """
        Get inputs to be fed to a model.

        Args:
            board (chess.Board): The chessboard in its current state.

        Returns:

            dict: The inputs to be fed to the model.
        """
        model_inputs = dict()

        # t, b, wk, wq, bk, bq = parse_fen(board.fen())
        # board_str = "................................................................" # string len(64)
        board = chess.Board()
        board.set_fen(fen_str)

        # print(f"\nInput to function: fen_str, agent states")
        # print(f"fen: {board.fen()}")
        # print(f"board state from fen_str:\n{board}")
        # print(f"ep square name:\t{chess.SQUARE_NAMES[board.ep_square]}")
        # print(f"Agent pos squares:\n{chess.SquareSet(agent_states)}")
        # print([chess.SQUARE_NAMES[new_agent_pos] for new_agent_pos in agent_states])

        # If blacks turn, mirror the board and invert colors of the pieces => "model always thinks it is playing as white"
        if board.turn is not chess.WHITE:
            board.apply_mirror()
            agent_states = [chess.square_mirror(agent_pos) for agent_pos in agent_states]

            # print("Blacks turn => mirror board => new input to encode:")
            # print(f"fen: {board.fen()}")
            # print(f"board state from fen_str:\n{board}")
            # print(f"Agent pos squares:\n{chess.SquareSet(agent_states)}")
            # print([chess.SQUARE_NAMES[new_agent_pos] for new_agent_pos in agent_states])

        ##### Castling Rights:
        board_str, turn, castling_rights, ep_square, _, __ = board.fen().split()

        # NOTE: The ep_square output from this is wrong!: print(ep_square)
        
        castling_rights_kingside = "K" in castling_rights
        castling_rights_queenside = "Q" in castling_rights
        opponent_castling_rights_kingside = "k" in castling_rights
        opponent_castling_rights_queenside = "q" in castling_rights

        # print(f"Castling Rights:\tK={castling_rights_kingside}\tQ={castling_rights_queenside}\tk={opponent_castling_rights_kingside}\tq={opponent_castling_rights_queenside}")
        # print(castling_rights_kingside)


        board_encoded = [0] * 64
        # board_encoded_square_names = ['.'] * 64
        for square_idx, chess_piece in board.piece_map().items():
            board_encoded[self.encode(chess.SQUARE_NAMES[square_idx], SQUARES)] = self.encode(chess.Piece.symbol(chess_piece),PIECES) 
            # board_encoded_square_names[self.encode(chess.SQUARE_NAMES[square_idx], SQUARES)] = chess.SQUARE_NAMES[square_idx]

        # print("BEFORE EP SQUARE UPDATE:")
        # print(board_encoded)
        # # print(board_encoded_square_names)

        ##### EP Square:
        if board.ep_square is not None:
            # print(f"ep square name:\t{chess.SQUARE_NAMES[board.ep_square]}")
            board_encoded[self.encode(chess.SQUARE_NAMES[board.ep_square], SQUARES)] = self.encode(",", PIECES)
            # # board_encoded_square_names[self.encode(chess.SQUARE_NAMES[board.ep_square], SQUARES)] = ','

        # print("AFTER EP SQUARE UPDATE:")
        # print(board_encoded)
        # # print(board_encoded_square_names)


        # TODO: Encode agent states with reward attributes or just 16 one-hot encodeded vectors of their current position???
        model_inputs["board_positions"] = (
            torch.IntTensor(board_encoded).unsqueeze(0).to(self.device)
        ) # (64)
        model_inputs["kingside_castling_rights"] = (
            torch.IntTensor([self.encode(castling_rights_kingside, vocabulary=BOOL)]).unsqueeze(0).to(self.device)
        ) # (1)
        model_inputs["queenside_castling_rights"] = (
            torch.IntTensor([self.encode(castling_rights_queenside, vocabulary=BOOL)]).unsqueeze(0).to(self.device)
        ) # (1)
        model_inputs["opponent_castling_rights_kingside"] = (
            torch.IntTensor([self.encode(opponent_castling_rights_kingside, vocabulary=BOOL)]).unsqueeze(0).to(self.device)
        ) # (1)
        model_inputs["opponent_castling_rights_queenside"] = (
            torch.IntTensor([self.encode(opponent_castling_rights_queenside, vocabulary=BOOL)]).unsqueeze(0).to(self.device)
        ) # (1)
        model_inputs["agent_ids"] = (
            torch.IntTensor(self.encode(agent_ids,vocabulary=CHESS_PIECE_AGENTS)).unsqueeze(0).to(self.device)
        ) # (16)
        # model_inputs["agents_pos"] = (
        #     torch.IntTensor(self.encode(agent_pos,vocabulary=AGENT_POS)).unsqueeze(0).to(self.device)
        # ) # (1)
        model_inputs["agents_pos"] = (
            torch.IntTensor(agent_pos).unsqueeze(0).to(self.device)
        ) # (16)
        
        # print(type(agent_reward_weights))
        model_inputs["reward_weights"] = (
            torch.FloatTensor(agent_reward_weights).unsqueeze(0).to(self.device)
        ) # Agent reward weights flattened: (n_agents , n_reward_types)
        # print(type(agent_reward_weights))
        

        return model_inputs
    
    ###################################################################################################
    # def encode_model_targets(self, target_moves, auxiliary_taget_rewards, agent_reward_weights):
    #     model_targets = dict()

    #     # print(self.encode(target_moves, vocabulary=UCI_MOVES))
    #     model_targets["moves"] = (
    #         torch.LongTensor([self.encode(target_moves, vocabulary=UCI_MOVES)]).unsqueeze(0).to(self.device)
    #     ) # (1) OR (len(target_moves)) if target moves is a list of e.g. all legal moves

    #     # print(type(model_targets["moves"]))
    #     # print(model_targets["moves"].shape)

    #     # print(torch.FloatTensor(agent_reward_weights))
    #     # print(torch.FloatTensor(target_rewards))
    #     # print(torch.FloatTensor(target_rewards).transpose(1,0))
        
    #     # agent_reward_weight_tensor = torch.FloatTensor(agent_reward_weights).mul(torch.FloatTensor(target_rewards).transpose(1,0))


    #     # # for key, target_rewards_per_rewardtype in auxiliary_taget_rewards.items():
    #     # #     # weighted_rewards_all_agents_per_rewardtype = [reward_target_agent_j * reward_weight_agent_i for reward_target_agent_j, reward_weight_agent_i in zip(target_rewards_per_rewardtype, agent_reward_weights)]
    #     # #     for agent_i_reward in target_rewards_per_rewardtype:
    #     # #         weighted_rewards_all_agents = agent_i_reward
    #     # #         weighted_rewards.append(weighted_rewards_all_agents_per_rewardtype)
        
    #     # model_targets["rewards"] = torch.IntTensor(
    #     #     [agent_reward_weight_tensor]).unsqueeze(0).to(self.device
    #     # )
    #     target_rewards = []
    #     for key, reward_weights_agent_i in auxiliary_taget_rewards.items():
    #         target_rewards.append(reward_weights_agent_i)

    #     model_targets["rewards"] = (
    #         torch.FloatTensor(agent_reward_weights).mul(torch.FloatTensor(target_rewards).transpose(1,0)).unsqueeze(0).to(self.device)
    #     )


    #     return model_targets
    
    ##########################################################################################
    
    # def encode(self, agent_posisions: dict, board: chess.Board):
    #     """
    #     Get inputs to be fed to the model.

    #     Args:

    #         board (chess.Board): The chessboard in its current state.

    #     Returns:

    #         dict: The inputs to be fed to the model.
    #     """
    #     encoded = dict()

    #     encoded["board_positions"] = (
    #         torch.IntTensor(encode(b, vocabulary=PIECES)).unsqueeze(0).to(self.device_name)
    #     )
    #     model_inputs["white_kingside_castling_rights"] = (
    #         torch.IntTensor([encode(wk, vocabulary=BOOL)]).unsqueeze(0).to(self.device_name)
    #     )
    #     model_inputs["white_queenside_castling_rights"] = (
    #         torch.IntTensor([encode(wq, vocabulary=BOOL)]).unsqueeze(0).to(self.device_name)
    #     )
    #     model_inputs["black_kingside_castling_rights"] = (
    #         torch.IntTensor([encode(bk, vocabulary=BOOL)]).unsqueeze(0).to(self.device_name)
    #     )
    #     model_inputs["black_queenside_castling_rights"] = (
    #         torch.IntTensor([encode(bq, vocabulary=BOOL)]).unsqueeze(0).to(self.device_name)
    #     )
    #     model_inputs["moves"] = (
    #         torch.LongTensor(
    #             [
    #                 UCI_MOVES["<move>"],
    #                 UCI_MOVES["<pad>"],
    #             ]
    #         )
    #         .unsqueeze(0)
    #         .to(self.device_name)
    #     )
    #     # model_inputs["lengths"] = torch.LongTensor([1]).unsqueeze(0).to(self.device_name)


    #     return encoded

    def decode_tokens_to_uci(token: int):
        # # token_2_uci_str = [uci_move for uci_move in UCI_MOVES.keys()]
        # if token < len(TOKEN_2_UCI_MOVES):
        #     return TOKEN_2_UCI_MOVES[token] #uci_move
        # else:
        #     raise ValueError(f"Token: {token} is NOT in the vocabilary!")
        raise NotImplementedError


    def load_uci_vocab(self, vocabs_dir="vocabs/uci_move_vocab.txt"):
        # vocab_filepath = os.path.join(args.vocab_dir, args.vocab_filename) 
        # # vocab_filepath = pathlib.Path.joinpath(args.vocab_dir, args.vocab_filename) #'vocabs/uci_vocab.txt'

        # # with open(vocab_filepath, 'w') as outfile:
        # #     print(f'deleted the content of vocab file: {vocab_filepath}')
        # with open(vocab_filepath, 'w') as outfile:
        #     for move in moves:
        #         entry = move
        #         outfile.write(entry)
        #         outfile.write('\n')
        #     print(f'Generated the uci vocab file: {vocab_filepath}')
        
        with open(vocabs_dir, "r") as uci_move_vocab_file:
            for idx,line in enumerate(tqdm(uci_move_vocab_file)):
                self.UCI_MOVES[re.sub("\n", "", line).strip()] = idx
        
        # Add special tokens:
        for token in self.special_tokens:
            idx += 1
            self.UCI_MOVES[token] = idx
            
        print(self.UCI_MOVES)


    @classmethod
    def vocab_sizes(self) -> int:
        return get_vocab_sizes() #self.VOCAB_SIZES



if __name__ == "__main__":
    vocab_sizes = Tokenizer.vocab_sizes()
    print(f"vocab_sizes:\n{vocab_sizes}")

    tokenizer = Tokenizer()

    black_pieces_start_pos_dict = {
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
    
    black_agent_state_names = []
    black_agent_states = []
    print("Generate agent positions from (agent_id,chess.Square):\n\tkey:\tpos:")
    for key, square_idx in black_pieces_start_pos_dict.items():
        print(f"\t{key}\t{square_idx}")
        black_agent_state_names.append(key)
        black_agent_states.append(square_idx)
    
    print(black_pieces_start_pos_dict)
    print(black_agent_state_names)
    print(black_agent_states)

    chess_piece_agent_reward_weights = [[1.0] * 11] * len(16)
    
    # fen_str = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
    # tokenizer.encode_model_input(fen_str=fen_str, agent_ids=range(16), agent_pos=black_agent_states, agent_reward_weights=chess_piece_agent_reward_weights)

    # fen_str_no_black_castle = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQ e3 0 1"
    # tokenizer.encode_model_input(fen_str=fen_str, agent_ids=range(16), agent_pos=black_agent_states, agent_reward_weights=chess_piece_agent_reward_weights)

