import os
import argparse


from tqdm import tqdm
import numpy as np

import chess
import chess.pgn


import enum
import re

import typing
import copy


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare pgn data for generating a chess dataset for Multi-agent RL')
    
    parser.add_argument('--pgn_dir', type=str, default="lichess_elite_database",
                        help='directory of the PGN files')
    parser.add_argument('--pgn_file', type=str, default="lichess_elite_2025-02.pgn",
                        help='name of the PGN file')
    
    args = parser.parse_args()
    return args


class RewardType(enum.Enum):
    NONE = -1,
    SURVIVAL_OF_THE_AGENT = 0,      # The agent receives a reward is based on wether or not the agent survives to the end of the game #has been captured or not.
    AGENT_CONTRIBUTION = 1,         # The agent receives a reward is based on the number of squares, that the agent is attacking at the current state of the game (i.e. squares that the agent can move to and thereby capture opponent pieces on)
    INDIVIDUAL_AGGRESIVENESS = 2,   # The agent receives a rewards based on the number of opponent pieces that it captures during the game
    PIECE_TYPE_SOLIDARITY = 3,      # The agent receives a rewards based on the number of agents on its team, that are the same type of chess piece as itself, that are still alive at the end of the game (irregardless of whether the team wins or not)
    INDIVIDUAL_DEFENSIVENESS = 4,   # The agent recieves a rewards based on the number of agents on the team that help defend it from being captured by an opponent piece.
    TEAM_GOAL = 5                   # All the agents on the team receive the same reward based on the result of the game: win/draw/loose
    TEAM_SURVIVAL = 6,              # All the agents on the team receive the same reward based on the number of agents on the team that survive until the end of the game (irregardless of whether the team wins or not)
    SOCIETY_AGGRESIVENESS = 7,      # All the agents on the team receive the same reward based on the number of opponent pieces that they capture during the game
    SOCIETY_PATIENCE = 8,           # All the agents on the team receive the same reward, where the value of the reward is based on how fast the team manages to end the game
    GROUP_MOBILITY = 9,             # All the agents on the team receive the same reward based on how many squares that an agent on the team can move to on their next turn
    SOCIETY_DEFENSIVENESS = 10,     # All the agents on the team receive the same reward, where the value of that reward is based on the number of agents that are defended by other agents on the team




# class MATChessTeamState:

#     chess_piece_agents_pos = {
#         'king': None,
#         'queen': None,
#         'rook0': None,
#         'rook7': None,
#         'knight1': None,
#         'knight6': None,
#         'bishop2': None,
#         'bishop5': None,
#         'pawn0': None,
#         'pawn1': None,
#         'pawn2': None,
#         'pawn3': None,
#         'pawn4': None,
#         'pawn5': None,
#         'pawn6': None,
#         'pawn7': None
#         }
    
#     # def __init__(self, prev_chess_piece_agents_pos, prev_board_fen):
#     #     self.chess_piece_agents_pos = prev_chess_piece_agents_pos
#     #     self.prev_board_fen = prev_board_fen
    
#     def __init__(self, halfmove_count, prev_chess_piece_agents_pos, prev_board_fen):
#         self.halfmove_count = halfmove_count
#         self.chess_piece_agents_pos = prev_chess_piece_agents_pos
#         self.prev_board_fen = prev_board_fen
#         self.chess_piece_agents_pos = {}

#     def agent_pos_next_state(self, team_action_uci, opponent_action_uci):
#         board = chess.Board()
#         board.reset_board()
#         board.set_fen(fen=self.prev_board_fen)

#         team_move = chess.Move.from_uci(team_action_uci)
#         opponent_move = chess.Move.from_uci(opponent_action_uci)

#         try:
#             board.push(team_move)
#             board.push(opponent_move)
#         except:
#             raise ValueError(f"The team's chosen move \"{team_action_uci}\" or the opponent's move \"{opponent_action_uci}\" are not valid actions for the current game state with fen= {self.prev_board_fen}")
#             return
        
#         for key, item in self.chess_piece_agents_pos.items():
#             if team_move.from_square == item:
#                 self.chess_piece_agents_pos[key] = team_move.to_square
#                 break
        
#         for key, item in self.chess_piece_agents_pos.items():
#             if opponent_move.to_square == item:
#                 self.chess_piece_agents_pos[key] = None
#                 break
    
#         return self.chess_piece_agents_pos



class Reward(object):
    individual_reward_types = [RewardType.SURVIVAL_OF_THE_AGENT, RewardType.AGENT_CONTRIBUTION, RewardType.INDIVIDUAL_AGGRESIVENESS, RewardType.PIECE_TYPE_SOLIDARITY, RewardType.INDIVIDUAL_DEFENSIVENESS]
    sociely_reward_types = [RewardType.TEAM_GOAL, RewardType.TEAM_SURVIVAL, RewardType.SOCIETY_AGGRESIVENESS, RewardType.SOCIETY_PATIENCE, RewardType.GROUP_MOBILITY, RewardType.SOCIETY_DEFENSIVENESS]
    all_reward_types = RewardType._member_names_

    # Pawn = 10, Knight = 32, Bishop = 33, Rook = 50, Queen = 90, King = 100

    chess_piece_values = {
        0: 0,
        chess.KING: 100,
        chess.QUEEN: 9,
        chess.ROOK: 5,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.PAWN: 1,
    }

    # chess_piece_type_start_count = {
    #     0: 0,
    #     chess.KING: 1,
    #     chess.QUEEN: 1,
    #     chess.ROOK: 2,
    #     chess.KNIGHT: 2,
    #     chess.BISHOP: 2,
    #     chess.PAWN: 8,
    # }

    # chess_piece_type_start_count = {
    #     'king': 1,
    #     'queen': 1,
    #     'rook0': 2,
    #     'rook7': 2,    
    #     'knight1': 2,   
    #     'knight6': 2,
    #     'bishop2': 2,
    #     'bishop5': 2,
    #     'pawn0': 8,
    #     'pawn1': 8,
    #     'pawn2': 8,
    #     'pawn3': 8,
    #     'pawn4': 8,
    #     'pawn5': 8,
    #     'pawn6': 8,
    #     'pawn7': 8,
    # }
    chess_piece_type_start_count = [1,1,2,2,2,2,2,2,8,8,8,8,8,8,8,8]

    chess_piece_agent_values = {
        'king': chess_piece_values[chess.KING],
        'queen': chess_piece_values[chess.QUEEN],
        'rook0': chess_piece_values[chess.ROOK],
        'rook7': chess_piece_values[chess.ROOK],
        'knight1': chess_piece_values[chess.KNIGHT],
        'knight6': chess_piece_values[chess.KNIGHT],
        'bishop2': chess_piece_values[chess.BISHOP],
        'bishop5': chess_piece_values[chess.BISHOP],
        'pawn0': chess_piece_values[chess.PAWN],
        'pawn1': chess_piece_values[chess.PAWN],
        'pawn2': chess_piece_values[chess.PAWN],
        'pawn3': chess_piece_values[chess.PAWN],
        'pawn4': chess_piece_values[chess.PAWN],
        'pawn5': chess_piece_values[chess.PAWN],
        'pawn6': chess_piece_values[chess.PAWN],
        'pawn7': chess_piece_values[chess.PAWN],
    }

    chess_piece_agent_piece_type = {
        'king': chess.KING,
        'queen': chess.QUEEN,
        'rook0': chess.ROOK,
        'rook7': chess.ROOK,
        'knight1': chess.KNIGHT,
        'knight6': chess.KNIGHT,
        'bishop2': chess.BISHOP,
        'bishop5': chess.BISHOP,
        'pawn0': chess.PAWN,
        'pawn1': chess.PAWN,
        'pawn2': chess.PAWN,
        'pawn3': chess.PAWN,
        'pawn4': chess.PAWN,
        'pawn5': chess.PAWN,
        'pawn6': chess.PAWN,
        'pawn7': chess.PAWN,
    }
    

    maximum_number_of_attack_squares_per_agent = {
        'king': 8,      # 8
        'queen': 27,    # 7+7+7+6 
        'rook0': 14,    # 7+7
        'rook7': 14,    
        'knight1': 8,   
        'knight6': 8,
        'bishop2': 13,  # 7+6
        'bishop5': 13,
        'pawn0': 2,     #
        'pawn1': 2,
        'pawn2': 2,
        'pawn3': 2,
        'pawn4': 2,
        'pawn5': 2,
        'pawn6': 2,
        'pawn7': 2,
    }

    win_reward = 100

    def __init__(self):
        # self.reward_type = RewardType.NONE
        return

    def print_all_reward_types(self):
        print(self.all_reward_types)
        return
    
    def get_rewards_from_reward_state_attributes(self, reward_attr_dict):
        # print("Reward Attributes:")
        # for key, item in reward_attr_dict.items():
        #     print(f'Reward Attr:\t{key.name}:\t{item}')

        agent_attack_square_scale_factor = [1.0 / float(item) for key,item in self.maximum_number_of_attack_squares_per_agent.items()]

        # rewards = {
        #     RewardType.SURVIVAL_OF_THE_AGENT: [float(reward_attr) for idx,reward_attr in enumerate(reward_attr_dict[RewardType.SURVIVAL_OF_THE_AGENT])], #[0.0] * 16,
        #     RewardType.AGENT_CONTRIBUTION: [float(reward_attr) * agent_attack_square_scale_factor[idx] for idx,reward_attr in enumerate(reward_attr_dict[RewardType.AGENT_CONTRIBUTION])], #[1.0 / float(item) for key,item in self.maximum_number_of_attack_squares_per_agent.items()],
        #     RewardType.INDIVIDUAL_AGGRESIVENESS: [self.chess_piece_values[captured_piece_type] for captured_piece_type in reward_attr_dict[RewardType.INDIVIDUAL_AGGRESIVENESS]], #[0.0]*16,
        #     RewardType.PIECE_TYPE_SOLIDARITY: [float(reward_attr) / float(self.chess_piece_type_start_count[idx]) for idx,reward_attr in enumerate(reward_attr_dict[RewardType.PIECE_TYPE_SOLIDARITY])], #[self.chess_piece_type_start_count[piece_type_start_count] for piece_type_start_count in reward_attr_dict[RewardType.INDIVIDUAL_AGGRESIVENESS]],#[0.0]*16,
        #     RewardType.INDIVIDUAL_DEFENSIVENESS: [float(reward_attr) / 16.0 for idx,reward_attr in enumerate(reward_attr_dict[RewardType.INDIVIDUAL_DEFENSIVENESS])],
        #     RewardType.TEAM_GOAL: reward_attr_dict[RewardType.TEAM_GOAL],
        #     RewardType.TEAM_SURVIVAL: [reward_attr_dict[RewardType.TEAM_SURVIVAL] / 16.0] * 16,
        #     RewardType.SOCIETY_AGGRESIVENESS: [reward_attr_dict[RewardType.SOCIETY_AGGRESIVENESS] / 16.0] * 16,
        #     RewardType.SOCIETY_PATIENCE: [float(reward_attr_dict[RewardType.SOCIETY_PATIENCE])] * 16,
        #     RewardType.GROUP_MOBILITY: [reward_attr_dict[RewardType.GROUP_MOBILITY] / 16.0] * 16,
        #     RewardType.SOCIETY_DEFENSIVENESS: [reward_attr_dict[RewardType.SOCIETY_DEFENSIVENESS] / 16.0] * 16,
        #     }
        rewards = {
            RewardType.SURVIVAL_OF_THE_AGENT.name: reward_attr_dict[RewardType.SURVIVAL_OF_THE_AGENT], #[float(reward_attr) for reward_attr in reward_attr_dict[RewardType.SURVIVAL_OF_THE_AGENT]], #[0.0] * 16,
            RewardType.AGENT_CONTRIBUTION.name: [float(reward_attr) * agent_attack_square_scale_factor[idx] for idx,reward_attr in enumerate(reward_attr_dict[RewardType.AGENT_CONTRIBUTION])], #[1.0 / float(item) for key,item in self.maximum_number_of_attack_squares_per_agent.items()],
            RewardType.INDIVIDUAL_AGGRESIVENESS.name: [self.chess_piece_values[captured_piece_type] for captured_piece_type in reward_attr_dict[RewardType.INDIVIDUAL_AGGRESIVENESS]], #[0.0]*16,
            RewardType.PIECE_TYPE_SOLIDARITY.name: [float(reward_attr) / float(self.chess_piece_type_start_count[idx]) for idx,reward_attr in enumerate(reward_attr_dict[RewardType.PIECE_TYPE_SOLIDARITY])], #[self.chess_piece_type_start_count[piece_type_start_count] for piece_type_start_count in reward_attr_dict[RewardType.INDIVIDUAL_AGGRESIVENESS]],#[0.0]*16,
            RewardType.INDIVIDUAL_DEFENSIVENESS.name: [float(reward_attr) / 16.0 for idx,reward_attr in enumerate(reward_attr_dict[RewardType.INDIVIDUAL_DEFENSIVENESS])],
            RewardType.TEAM_GOAL.name: reward_attr_dict[RewardType.TEAM_GOAL],
            RewardType.TEAM_SURVIVAL.name: [reward_attr_dict[RewardType.TEAM_SURVIVAL] / 16.0] * 16,
            RewardType.SOCIETY_AGGRESIVENESS.name: [reward_attr_dict[RewardType.SOCIETY_AGGRESIVENESS] / 16.0] * 16,
            RewardType.SOCIETY_PATIENCE.name: [float(reward_attr_dict[RewardType.SOCIETY_PATIENCE])] * 16,
            RewardType.GROUP_MOBILITY.name: [reward_attr_dict[RewardType.GROUP_MOBILITY] / 16.0] * 16,
            RewardType.SOCIETY_DEFENSIVENESS.name: [reward_attr_dict[RewardType.SOCIETY_DEFENSIVENESS] / 16.0] * 16,
            }
        
        # print("\nRewards:")
        # for key, item in rewards.items():
        #     print(f'Reward:\t{key}:\t{item}')

        return rewards




class GameResult(enum.Enum):
    NONE = -1
    WHITE_WIN = 0,
    DRAW=1,
    BLACK_WIN = 2

def result_str_2_enum(result: str="?"):
    if result == "1-0":
        return GameResult.WHITE_WIN
    elif result == "1/2-1/2":
        return GameResult.DRAW
    elif result == "0-1":
        return GameResult.BLACK_WIN
    else:
        return GameResult.NONE

def lichessURL_2_gameid(lichess_url: str="?"):
    ''''
    lichess_url: The URL for the source of the game.
    return: The game ID, which is jsut the characters after the last fron-slash in the Lichess URL
    '''
    return re.split("/",lichess_url)[-1]


class GamePGN:
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
    
    def __init__(self, game_pgn: chess.pgn.Game):
        # Store all relevant info from the PGN file header for this game:
        self.pgn_chess_game = game_pgn
        self.event_name = game_pgn.headers.get("Event", "?") #game_pgn['Event']
        self.game_result_str = game_pgn.headers.get("Result", "?")
        self.game_result_enum = result_str_2_enum(self.game_result_str)
        self.lichess_game_url = game_pgn.headers.get("LichessURL", "?")
        self.game_id = lichessURL_2_gameid(self.lichess_game_url)
        
        # TODO: Finish adding the remaining info from the PGN file headers
        self.white_player_username = game_pgn.headers.get("White", "?")
        self.black_player_username = game_pgn.headers.get("Black", "?")
        self.white_elo = game_pgn.headers.get("WhiteElo", "?")
        self.black_elo = game_pgn.headers.get("BlackElo", "?")
        self.pgn_termination = game_pgn.headers.get("Termination", "?")
        self.white_player_title = game_pgn.headers.get("WhiteTitle", "?")
        self.black_player_title = game_pgn.headers.get("BlackTitle", "?")


        self.date = game_pgn.headers.get("Date", "?")
        self.eco = game_pgn.headers.get("ECO", "?")     # Encyclopaedia of Chess Openings (ECO)
        self.opening = game_pgn.headers.get("Opening", "?")
        self.time_control = game_pgn.headers.get("TimeControl", "?")

        # Init variables required to compute the board state attributes for all states in the game: 
        self.uci_moves = [move.uci() for move in game_pgn.mainline_moves()]

        self.chess_termination = self.pgn_game_chess_termination()
        
        # self.white_chess_piece_agents_pos = dict(self.white_chess_piece_agents_start_pos)
        # self.black_chess_piece_agents_pos = dict(self.black_chess_piece_agents_start_pos)
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

    
    def print_pgn_info(self):
        # print('\n\n======================')
        print(f'Event:\t{self.event_name}')
        print(f'Result:\t{self.game_result_enum.name}')
        print(f'LichessURL:\t{self.lichess_game_url}')
        print(f'Game ID:\t{self.game_id}')
        print(f'White username:\t{self.white_player_username}')
        print(f'Black username:\t{self.black_player_username}')
        print(f'White ELO rating:\t{self.white_elo}')
        print(f'Black ELO rating:\t{self.black_elo}')

        print(f'UCI Moves:\n\t{self.uci_moves}')

    def pgn_header_info(self) -> dict:
        return {
            'event': self.event_name,
            'result': self.game_result_enum.name,
            'pgn_termination': self.pgn_termination,
            'chess_termination': self.chess_termination,
            #'lichess_url': self.lichess_game_url,
            #'game_id': self.game_id,
            'white_username': self.white_player_username,
            'black_username': self.black_player_username,
            'white_player_title': self.white_player_title,
            'black_player_title': self.black_player_title,
            'white_elo': self.white_elo,
            'black_elo': self.black_elo,
            'uci_moves:': self.uci_moves,
        }
    
    def pgn_game_chess_termination(self):
        chess_board = chess.Board()
        chess_board.reset()
        for move in self.pgn_chess_game.mainline_moves():
            chess_board.push(move)
        
        game_outcome = chess_board.outcome(claim_draw=True)
        if game_outcome is not None:
            if game_outcome.termination is not None:
                return game_outcome.termination.name
        
        return 'NONE'



    
    def get_game_id(self):
        return self.game_id



    def update_agent_state_attributes(self, chess_board: chess.Board):
        # The following code stores the state attributes as sets of squares, which are internally represented as a 64 bit integer masks of the included squares in the set:
        for key_team, item_team in self.state_attributes_per_agent_before_action.items():
            for key, item in item_team.items():
                # agent_pos = self.white_chess_piece_agents_pos[key]
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

                    # print(f'\nboard:\n{chess_board}')
                    # print(f'\nagent_pos:\n{chess.SquareSet([agent_pos])}')
                    # print(f'\tagent_attack_squares:\n{agent_attack_squares}')
                    # print(f'\tagent_under_attacked_from_square:\n{agent_under_attacked_from_square}')
                    # print(f'\tagent_defended_by_agent_on_square:\n{agent_defended_by_agent_on_square}')

                    agent_attributes_dict = {
                        'is_alive': True,
                        'agent_pos': agent_pos,
                        'attack_squares': agent_attack_squares,
                        'attacked_by_agents_on_square': agent_under_attacked_from_square,
                        'defended_by_agents_on_square': agent_defended_by_agent_on_square,
                        # 'captures_in_game_excl_action': self.state_attributes_per_agent_before_action[key_team][key]
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
                        # 'captures_in_game_excl_action': self.state_attributes_per_agent_before_action[key_team][key]
                        'same_piece_type_count': 0,
                    }

                    self.state_attributes_per_agent_before_action[key_team][key] = dict(agent_attributes_dict)

        return

    def all_reward_attributes_for_game(self, only_winning_players_moves=True):
        # Init new chess game:
        board = chess.Board()
        board.reset()

        # Create a cpoy of all the moves played throughout the game:
        game_move_hist = [move for move in self.pgn_chess_game.mainline_moves()]
        if len(game_move_hist) < 2:
            raise AttributeError(f"Game hist should be longer than two for the PGN to be a valid completed game!")

        ##### Generate state-action pairs for each state in the PGN game #####
        # Create structure for storing the state-action paris:
        all_board_fens = []
        all_uci_moves = []
        all_white_chess_pieces_pos = []
        all_black_chess_pieces_pos = []

        # all_boards_as_ucicode = []
        # all_boards_as_ucicode.append(board.unicode())

        all_state_attributes_per_agent_before_action = []

        # Set the first entry in the state lists to the starting state (used to generate the state-action-reward sequences):
        board_fen = board.fen()
        all_board_fens.append(board_fen)

        all_white_chess_pieces_pos.append(dict(self.white_chess_piece_agents_pos))
        all_black_chess_pieces_pos.append(dict(self.black_chess_piece_agents_pos))

        self.update_agent_state_attributes(chess_board=board)
        #all_state_attributes_per_agent_before_action.append(dict(self.state_attributes_per_agent_before_action))
        all_state_attributes_per_agent_before_action.append(copy.deepcopy(self.state_attributes_per_agent_before_action))
        # print(all_state_attributes_per_agent_before_action[-1]['white']['king'])
        state_action_captures = []

        # global_state_attributes_before_action = {
        #     'possible_en_pessant_target_present': False,
        #     'ep_square_idx': -1,
        #     'castling_rights_kingside': True,
        #     'castling_rights_queenside': False,
        #     'opponent_castling_rights_kingside': False,
        #     'opponent_castling_rights_queenside': False,
        #     'legal_uci_moves': [],
        # }

        all_global_state_attributes_before_action = []
        # all_global_state_attributes_before_action.append(copy.deepcopy(global_state_attributes_before_action))

        # Apply the move to the chess piece agents by updating their positions (this is required to keep track of each agent during the game):
        for move_idx in range(len(game_move_hist)):

            global_state_attributes_before_action = {
                'possible_en_pessant_target_present': False,
                'ep_square_idx': -1,
                'castling_rights_kingside': False,
                'castling_rights_queenside': False,
                'opponent_castling_rights_kingside': False,
                'opponent_castling_rights_queenside': False,
                'legal_uci_moves': [],
            }

            if board.ep_square is not None:
                global_state_attributes_before_action['possible_en_pessant_target_present'] = True
                global_state_attributes_before_action['ep_square_idx'] = board.ep_square
            
            # Update global_state_attributes_before_action with castling rights for the input state (for both the active player and their opponent)
            if board.has_kingside_castling_rights(board.turn):
                global_state_attributes_before_action['castling_rights_kingside'] = True

            if board.has_queenside_castling_rights(board.turn):
                global_state_attributes_before_action['castling_rights_queenside'] = True

            if board.has_kingside_castling_rights(not board.turn):
                global_state_attributes_before_action['opponent_castling_rights_kingside'] = True

            if board.has_queenside_castling_rights(not board.turn):
                global_state_attributes_before_action['opponent_castling_rights_queenside'] = True
            global_state_attributes_before_action['legal_uci_moves'] = [legal_move.uci() for legal_move in board.legal_moves]
            all_global_state_attributes_before_action.append(copy.deepcopy(global_state_attributes_before_action))

            # Update Individual reward state attributes:
            capture_with_action = {
                'move_idx': move_idx,
                'piece_was_captured': False,
                'capturing_agent_team_color': board.turn,
                'capturing_agent_id': None,
                'captured_agent_id': None,
                'captured_agent_type': None,
            }

            self.update_agent_state_attributes(chess_board=board)
            # all_state_attributes_per_agent_before_action.append(dict(self.state_attributes_per_agent_before_action))
            all_state_attributes_per_agent_before_action.append(copy.deepcopy(self.state_attributes_per_agent_before_action))
            
            # print(f'\nHalfmove #{move_idx}\tMove:\t{game_move_hist[move_idx]}')
            # print(all_state_attributes_per_agent_before_action[-1]['white']['king'])

            # Update the positions of the chess piece agents:
            move_piece_id = None
            if board.turn == chess.WHITE:
                for key, item in self.white_chess_piece_agents_pos.items():
                    if game_move_hist[move_idx].from_square == item:
                        self.white_chess_piece_agents_pos[key] = game_move_hist[move_idx].to_square
                        move_piece_id = str(key)
                        break

                for key, item in self.black_chess_piece_agents_pos.items():
                    if game_move_hist[move_idx].to_square == item:
                        self.black_chess_piece_agents_pos[key] = None

                        capture_with_action['piece_was_captured'] = True
                        capture_with_action['capturing_agent_id'] = move_piece_id
                        capture_with_action['captured_agent_id'] = str(key)
                        capture_with_action['captured_agent_type'] = board.piece_type_at(game_move_hist[move_idx].to_square)

                        break
            else:
                for key, item in self.black_chess_piece_agents_pos.items():
                    if game_move_hist[move_idx].from_square == item:
                        self.black_chess_piece_agents_pos[key] = game_move_hist[move_idx].to_square
                        move_piece_id = str(key)
                        break

                for key, item in self.white_chess_piece_agents_pos.items():
                    if game_move_hist[move_idx].to_square == item:
                        self.white_chess_piece_agents_pos[key] = None

                        capture_with_action['piece_was_captured'] = True
                        capture_with_action['capturing_agent_id'] = move_piece_id
                        capture_with_action['captured_agent_id'] = str(key)
                        capture_with_action['captured_agent_type'] = board.piece_type_at(game_move_hist[move_idx].to_square)

                        break
            
            state_action_captures.append(dict(capture_with_action))

            # Handle updating the position of the rook agents for castling moves:
            if board.is_castling(game_move_hist[move_idx]):
                # update rook position for castling moves:
                if board.is_kingside_castling(game_move_hist[move_idx]):
                    if board.turn == chess.WHITE:
                        self.white_chess_piece_agents_pos['rook7'] = chess.F1
                    else:
                        self.black_chess_piece_agents_pos['rook7'] = chess.F8
                elif board.is_queenside_castling(game_move_hist[move_idx]):
                    if board.turn == chess.WHITE:
                        self.white_chess_piece_agents_pos['rook0'] = chess.D1
                    else:
                        self.black_chess_piece_agents_pos['rook0'] = chess.D8
            
            # Handle updating position of pawns captured with en passant:
            if board.is_en_passant(game_move_hist[move_idx]):
                ep_captured_agent_rank = chess.square_rank(game_move_hist[move_idx].from_square)
                ep_captured_agent_file = chess.square_file(game_move_hist[move_idx].to_square)
                ep_captured_agent_square = chess.square(file_index=ep_captured_agent_file, rank_index=ep_captured_agent_rank)
                if board.turn == chess.WHITE:
                    for key, item in self.black_chess_piece_agents_pos.items():
                        if item == ep_captured_agent_square:
                            self.black_chess_piece_agents_pos[key] = None
                elif board.turn == chess.BLACK:
                    for key, item in self.white_chess_piece_agents_pos.items():
                        if item == ep_captured_agent_square:
                            self.white_chess_piece_agents_pos[key] = None


            # Apply the move to the board in game:
            board.push(game_move_hist[move_idx])

            # Get the next board state (represented as a fen string) and append it to the list of all board states:
            board_fen = board.fen()
            all_board_fens.append(board_fen)

            # all_boards_as_ucicode.append(board.unicode())

            # Append the new positions of all the pieces to the lists of all the positions of the pieces:
            all_white_chess_pieces_pos.append(dict(self.white_chess_piece_agents_pos))
            all_black_chess_pieces_pos.append(dict(self.black_chess_piece_agents_pos))

            # Store a copy of all the moves in UCI format (convenient when generating the state-action-reward sequence)
            all_uci_moves.append(game_move_hist[move_idx].uci())

        # # print(f'\nAll game states:\n\t{all_board_fens}')
        # # print(f'\nAll white pieces pos:\t{all_white_chess_pieces_pos}')
        # # print(f'\nAll black pieces pos:\t{all_black_chess_pieces_pos}')
        # # print(f'\nAll uci moves:\t{all_uci_moves}')
        
        self.update_agent_state_attributes(chess_board=board)
        # all_state_attributes_per_agent_before_action.append(dict(self.state_attributes_per_agent_before_action))
        all_state_attributes_per_agent_before_action.append(copy.deepcopy(self.state_attributes_per_agent_before_action))
        
        # print('\nGame Over state:')
        # print(all_state_attributes_per_agent_before_action[-1]['white']['king'])

        ##### Generate the state-action-reward sequence for the game #####
        # NOTE: The "reward state" is either:
        # a) The state of the game AFTER BOTH the players move AND the opponent's response (their subsequent move) has been applied to the board.
        # b) Or the terminal state (i.e. the game ends right after the player's move due to game over or the opponent claiming draw?).
        # NOTE 1: The "reward state" is used for computing the "reward state attributes" for a given state-action pair.
        # Using this as the "next state" makes sense, since this is the state that best represent the consequences of its last action, and 
        # it is the next state where the agent is allowed to make another action (if the game is not over).
        #  NOTE 2: If the same state-action pair occures multiple times in the dataset, but the opponet responds differntly in some instances, then this should be used when computing the consequences of the chosen action.
        attributes_for_all_states_actions_rewards = []
        terminal_state_idx = len(all_board_fens) - 1

        # # Uncomment the following line to print all the attributes for each state of the game!
        # print(all_state_attributes_per_agent_before_action[0])

        # print("\nAgent\'s state-attributes for the first and last state in game BEFOR REWARD CALC:")
        # print(all_state_attributes_per_agent_before_action[0]['white']['king'])
        # print(all_state_attributes_per_agent_before_action[-1]['white']['king'])

        team_size = 16

        # winner_color = board.result(claim_draw=True)
        winning_team_name = None
        game_outcome = board.outcome(claim_draw=True)
        if game_outcome is not None:
            if game_outcome.winner is not None:
                winning_team_name = 'white' if game_outcome.winner is chess.WHITE else chess.BLACK


        for state_action_pair_idx in range(len(all_uci_moves)):
            reward_state_idx = state_action_pair_idx + 2
            if reward_state_idx > terminal_state_idx:
                reward_state_idx = state_action_pair_idx + 1

            # TODO: Compute the state attributes for all agents on the team for the state at reward_state_idx
            # DO NOT include info about previous/future states in the game since its too complicated to define which actions in the past leads to which specific consequences in the future!
            # agent_order_in_state_attr_lists = []
            state_attr_survival_of_the_agents = []
            state_attr_individual_contribution = []

            
            # state_attr_individual_aggressiveness = [0] * (team_size * 2)   # Creates a list of size 32 filled with zeros
            state_attr_individual_aggressiveness = [0] * (team_size)   # Creates a list of size 32 filled with zeros

            # state_attr_piece_type_solidarity = {
            #     'white': [0] * len(chess.PIECE_TYPES),
            #     'black': [0] * len(chess.PIECE_TYPES)
            # }
            state_attr_piece_type_solidarity = []
            state_attr_individual_defensiveness = []

            state_attr_society_team_goal = [] #[0] * (2*team_size)  # Creates a list of size 32 filled with zeros
            state_attr_society_team_patience = []

            if state_action_captures[state_action_pair_idx]['piece_was_captured'] is True:
                # capture_agent_team_idx_offset = self.reward_list_team_order.index(typing.cast(str, chess.COLOR_NAMES[state_action_captures[state_action_pair_idx]['capturing_agent_team_color']]))
                capture_agent_team_idx_offset = 0
                capture_agent_idx_offset_within_team = self.reward_list_agent_order.index(state_action_captures[state_action_pair_idx]['capturing_agent_id'])

                # # Add a "1" to the aggressiveness reward at the index belonging to the agent that captured it:
                # state_attr_individual_aggressiveness[capture_agent_team_idx_offset * team_size + capture_agent_idx_offset_within_team] = 1.0 # state_action_captures[state_action_pair_idx]['captured_agent_type']

                # NOTE: Uncomment the line below to this to add the enum value for the piece type that was captured at the index belonging to the agent that captured it instead of just setting the reward to 1:
                state_attr_individual_aggressiveness[capture_agent_team_idx_offset * team_size + capture_agent_idx_offset_within_team] = state_action_captures[state_action_pair_idx]['captured_agent_type']

            team_name = 'white' if state_action_pair_idx % 2 == 0 else 'black'

            agent_pos_square_idx_current_state = []
            agent_pos_square_idx_reward_state = []
            for agent_name in self.reward_list_agent_order:
                # agent_order_in_state_attr_lists.append(team_name + '_' + agent_name)

                agent_pos_for_state_action_pair_idx = all_state_attributes_per_agent_before_action[state_action_pair_idx][team_name][agent_name]['agent_pos']
                if agent_pos_for_state_action_pair_idx is not None:
                    agent_pos_square_idx_current_state.append(agent_pos_for_state_action_pair_idx)
                else:
                    agent_pos_square_idx_current_state.append(-1)
                
                agent_pos_for_reward_state_idx = all_state_attributes_per_agent_before_action[reward_state_idx][team_name][agent_name]['agent_pos']
                if agent_pos_for_state_action_pair_idx is not None:
                    agent_pos_square_idx_reward_state.append(agent_pos_for_reward_state_idx)
                else:
                    agent_pos_square_idx_reward_state.append(-1)
                
                # state_attr_survival_of_the_agents.append(all_state_attributes_per_agent_before_action[reward_state_idx][team_name][agent_name]['is_alive'])
                state_attr_survival_of_the_agents.append(int(all_state_attributes_per_agent_before_action[reward_state_idx][team_name][agent_name]['is_alive']))

                state_attr_individual_contribution.append(len(list(all_state_attributes_per_agent_before_action[reward_state_idx][team_name][agent_name]['attack_squares'])))

                # Add the count of all pieces of the same color and type as this one to the "all_state_attributes_per_agent_before_action" dict:
                state_attr_piece_type_solidarity.append(all_state_attributes_per_agent_before_action[reward_state_idx][team_name][agent_name]['same_piece_type_count'])

                # state_attr_individual_defensiveness.append(all_state_attributes_per_agent_before_action[reward_state_idx][team_name][agent_name]['defended_by_agents_on_square'])
                state_attr_individual_defensiveness.append(len(list(all_state_attributes_per_agent_before_action[reward_state_idx][team_name][agent_name]['defended_by_agents_on_square'])))

                if reward_state_idx == terminal_state_idx and winning_team_name is not None:
                    # if winner_color == state_attr_society_team_goal.append(self.game_result_enum)
                    # if winning_team_name is None:
                    
                    if winning_team_name == team_name:
                        state_attr_society_team_goal.append(1)
                    else:
                        state_attr_society_team_goal.append(-1)                        
                else:
                    state_attr_society_team_goal.append(0)

            #################################################################
            
            if reward_state_idx != terminal_state_idx:
                state_attr_society_team_patience.append(-1) # Negative reward for not finishing the game
            else:
                state_attr_society_team_patience.append(0)  # no/neutral reward for ending the game irregardless of wether the team wins or not (i.e. reward=0 when the agent choose an action that will terminate the game)


            
            # print(f'\nstate-action at time step = {state_action_pair_idx}\treward_state_idx={reward_state_idx}')
            # print(f'\taction:\t{game_move_hist[state_action_pair_idx]}')
            
            # print(self.agent_order_in_state_attr_lists[team_start_idx_in_angent_order_in_state_attribute_lists_name:team_start_idx_in_angent_order_in_state_attribute_lists_name+team_size])
            # print(state_attr_survival_of_the_agents)
            # print(state_attr_individual_contribution)
            # print(state_attr_individual_aggressiveness)
            # print(state_attr_piece_type_solidarity)
            # print(state_attr_individual_defensiveness)
            # print(state_attr_society_team_goal)

            # print(board)

            team_start_idx_in_angent_order_in_state_attribute_lists_name = 0 if state_action_pair_idx % 2 == 0 else team_size
            # team_name = 'white' if state_action_pair_idx % 2 == 0 else 'black'
            reward_function = Reward()

            state_action_reward = {
                'state': {
                    'fen': all_board_fens[state_action_pair_idx],
                    'state_idx': state_action_pair_idx,
                    'team_color': team_name,
                    # 'agent_ids': self.reward_list_agent_order,
                    'agent_pos_square_idx': agent_pos_square_idx_current_state,
                    # DONE: Add info in global_state_attributes_before_action to include en passant and castling information in the input state!
                    'possible_en_pessant_target_present': int(all_global_state_attributes_before_action[state_action_pair_idx]['possible_en_pessant_target_present']),
                    'ep_square_idx': all_global_state_attributes_before_action[state_action_pair_idx]['ep_square_idx'],
                    'castling_rights_kingside': int(all_global_state_attributes_before_action[state_action_pair_idx]['castling_rights_kingside']),
                    'castling_rights_queenside': int(all_global_state_attributes_before_action[state_action_pair_idx]['castling_rights_queenside']),
                    'opponent_castling_rights_kingside': int(all_global_state_attributes_before_action[state_action_pair_idx]['opponent_castling_rights_kingside']),
                    'opponent_castling_rights_queenside': int(all_global_state_attributes_before_action[state_action_pair_idx]['opponent_castling_rights_queenside']),
                    #'legal_uci_moves': all_global_state_attributes_before_action[state_action_pair_idx]['legal_uci_moves'],
                    # TODO?: Add the token value for each of the 64 squares in the board for the input state OR just use the board part of the fen_str as input for the model Dataset? 
                },
                'reward_state_info': {
                    'fen': all_board_fens[reward_state_idx],
                    'state_idx': reward_state_idx,
                    'agent_ids': self.reward_list_agent_order,
                    'agent_pos_square_idx': agent_pos_square_idx_reward_state
                },
                # 'matchess_state_attributes_before_action': dict(all_state_attributes_per_agent_before_action[state_action_pair_idx]),
                'action': {
                    'uci_move': all_uci_moves[state_action_pair_idx],
                    'halfmove_count': state_action_pair_idx,
                },
                # 'reward_state_info': {
                #     'game_state_unique_id': None,
                #     'is_terminal_state': None,
                # },
                # 'reward_state_halfmove': reward_state_idx,
                'reward_state_attributes': {
                    RewardType.SURVIVAL_OF_THE_AGENT: state_attr_survival_of_the_agents, # typing.cast(list[int], state_attr_survival_of_the_agents),
                    RewardType.AGENT_CONTRIBUTION: state_attr_individual_contribution,
                    RewardType.INDIVIDUAL_AGGRESIVENESS: state_attr_individual_aggressiveness,
                    RewardType.PIECE_TYPE_SOLIDARITY: state_attr_piece_type_solidarity,
                    RewardType.INDIVIDUAL_DEFENSIVENESS: state_attr_individual_defensiveness,
                    RewardType.TEAM_GOAL: state_attr_society_team_goal,
                    RewardType.TEAM_SURVIVAL: sum(state_attr_survival_of_the_agents), #[sum(state_attr_survival_of_the_agents[0:team_size]), sum(state_attr_survival_of_the_agents[team_size:])],
                    RewardType.SOCIETY_AGGRESIVENESS: sum(state_attr_individual_aggressiveness), #[sum(state_attr_individual_aggressiveness[0:team_size]), sum(state_attr_individual_aggressiveness[team_size:])],
                    RewardType.SOCIETY_PATIENCE: state_attr_society_team_patience[0],
                    RewardType.GROUP_MOBILITY: sum(state_attr_individual_contribution), #[sum(state_attr_individual_contribution[0:team_size]), sum(state_attr_individual_contribution[team_size:])],       # Changed to number of squares on the board that each player controls - i.e. sum of the number of squares that each agent control
                    RewardType.SOCIETY_DEFENSIVENESS: sum(state_attr_individual_defensiveness), #[sum(state_attr_individual_defensiveness[0:team_size]), sum(state_attr_individual_defensiveness[team_size:])],
                },
                # 'rewards': reward_function.get_rewards_from_reward_state_attributes(all_global_state_attributes_before_action[state_action_pair_idx]['reward_state_attributes'])
                # # TODO: Compute the reward values by uding the Reward class above (OBS! TODO in Reward Class needs to be finished first!)
                # 'rewards': {
                #     RewardType.SURVIVAL_OF_THE_AGENT: [float(i)/len(state_attr_survival_of_the_agents) for i in state_attr_survival_of_the_agents],
                #     RewardType.AGENT_CONTRIBUTION: [],
                #     RewardType.INDIVIDUAL_AGGRESIVENESS: [],
                #     RewardType.PIECE_TYPE_SOLIDARITY: [],
                #     RewardType.INDIVIDUAL_DEFENSIVENESS: [],
                #     RewardType.TEAM_GOAL: 0,
                #     RewardType.TEAM_SURVIVAL: 0,
                #     RewardType.SOCIETY_AGGRESIVENESS: 0,
                #     RewardType.SOCIETY_PATIENCE: 0,
                #     RewardType.GROUP_MOBILITY: 0,
                #     RewardType.SOCIETY_DEFENSIVENESS: 0
                # }
            }
            state_action_reward['rewards'] = reward_function.get_rewards_from_reward_state_attributes(state_action_reward['reward_state_attributes'])
            # attributes_for_all_states_actions_rewards.append(state_action_reward)

            # Only add the winning players state-action-pairs to the list:
            if only_winning_players_moves:
                if state_action_reward['state']['team_color'] == winning_team_name:
                    attributes_for_all_states_actions_rewards.append(state_action_reward)
            else:
                attributes_for_all_states_actions_rewards.append(state_action_reward)
        
            # print("\nState-attributes for first and last state in game:")
            # print(attributes_for_all_states_actions_rewards[0]['matchess_state_attributes'])
            # print(attributes_for_all_states_actions_rewards[-1]['matchess_state_attributes'])

            # print(game_outcome)
            # print(winning_team_name)

            # print("\nreward_state_attributes for first and last reward-state in the game:")
            # print(attributes_for_all_states_actions_rewards[0]['reward_state_attributes'])
            # print(attributes_for_all_states_actions_rewards[-1]['reward_state_attributes'])

            # print(f'\nreward_state_attributes for the last state-action pair for each team in the game:')
            # second_last_move_player_name = attributes_for_all_states_actions_rewards[-2]['state']['team_color']
            # print(f'state-action-reward for player\t{second_last_move_player_name}:')
            # print(attributes_for_all_states_actions_rewards[-2])
            # last_move_player_name = attributes_for_all_states_actions_rewards[-1]['state']['team_color']
            # print(f'state-action-reward for player\t{last_move_player_name}:')
            # print(attributes_for_all_states_actions_rewards[-1])
            ########################################################################################

            # for idx, move in enumerate(game_move_hist, start=1):
            #     # TODO: compute attributes for all agents for all states in the game and append to attribute list.

            #     board_fen = board.fen()
            #     team_agents_pos = self.white_chess_piece_agents_pos if board.turn == chess.WHITE else self.black_chess_piece_agents_pos

            #     board.push(move)

            # state_action_reward = {
            #     'state': MATChessTeamState(halfmove_count=idx, prev_board_fen=board_fen,prev_chess_piece_agents_pos=team_agents_pos),
            #     'action': move.uci(),
            #     'rewards': {
            #         RewardType.SURVIVAL_OF_THE_AGENT: [],
            #         RewardType.AGENT_CONTRIBUTION: [],
            #         RewardType.INDIVIDUAL_AGGRESIVENESS: [],
            #         RewardType.PIECE_TYPE_SOLIDARITY: [],
            #         RewardType.INDIVIDUAL_DEFENSIVENESS: [],
            #         RewardType.TEAM_GOAL: 0,
            #         RewardType.TEAM_SURVIVAL: 0,
            #         RewardType.SOCIETY_AGGRESIVENESS: 0,
            #         RewardType.SOCIETY_PATIENCE: 0,
            #         RewardType.GROUP_MOBILITY: 0,
            #         RewardType.SOCIETY_DEFENSIVENESS: 0
            #     }
            # }
            # attributes_for_all_states_actions_rewards.append(state_action_reward)

        return attributes_for_all_states_actions_rewards


if __name__ == '__main__':
    args = _parse_args()

    print(f'pgn dir:\t{os.path.join(args.pgn_dir, args.pgn_file)}')

    pgn_filename = os.path.join(args.pgn_dir, args.pgn_file)

    with open(pgn_filename) as pgn:
        done_reading_pgn_file = False
        game_offsets = []

        max_dataset_entries = 1    # set max_dataset_entries to inf to process all the games in the PGN file.
        game_count = 0
        while not done_reading_pgn_file:

            if game_count >= max_dataset_entries:
                break

            game_offset = pgn.tell()
            # game_header = chess.pgn.read_headers(pgn)
            
            # print(f'Game PGN offset:\t{game_offset}\n\t{game_header}')

            # if game_header is None:
            #     done_reading_pgn_file = True
            #     break

            # if "1-0" in game_header.get("Result", "?"):
            # game_offsets.append(game_offset)
            

            pgn_game = chess.pgn.read_game(pgn)
            if pgn_game is None:
                done_reading_pgn_file = True
                break

            game_offsets.append(game_offset)
            pgn_game_datapoint = GamePGN(pgn_game)
            # pgn_game_datapoint.print_pgn_info()

            pgn_game_state_action_reward_state_attr = pgn_game_datapoint.all_reward_attributes_for_game()

            # Print data for input state:
            # print(pgn_game_state_action_reward_state_attr[0]['state'])

            
            print("\n\n========== PGN GAME DATA ==========")
            pgn_game_datapoint.print_pgn_info()

            last_input_state_idx = len(pgn_game_state_action_reward_state_attr) - 1
            print(f"Game # {game_count}\nHalfmove = {last_input_state_idx}")
            for key, item in pgn_game_state_action_reward_state_attr[last_input_state_idx].items():
                print(f'\n{key}:\t{item}')

            reward_function = Reward()
            pgn_game_rewards = reward_function.get_rewards_from_reward_state_attributes(pgn_game_state_action_reward_state_attr[last_input_state_idx]['reward_state_attributes'])
            print("\nRewards:")
            for key, item in pgn_game_rewards.items():
                print(f'Reward:\t{key.name}:\t{item}')
            ##############
            # TODO: Convert this to a dataset and save it in a json file:
            # Each entry is a dictonary of pgn_game_info (incl. info to find the state in the orinigal pgn game dataset file), 
            # state-action-rewards dictionary, plus all state-attributes for the input state and the reward state
            ##############
            

            game_count += 1
        
        print(f'\n\n#Games in PGN file:\t{len(game_offsets)}')
    
    

    # pgn.close()

    # pgn = open(pgn_filename)

    # for game_idx in tqdm(game_offsets):
    #     pgn.seek(game_idx)
    #     print(f'Game PGN offset:\t{game_idx}')
    #     pgn_game = chess.pgn.read_game(pgn)
    #     pgn_game_datapoint = GamePGN(pgn_game.headers)
    #     print(f'\t{game_header}')
    #     pgn_game_datapoint.print_pgn_info()
