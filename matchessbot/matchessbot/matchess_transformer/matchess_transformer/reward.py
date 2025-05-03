import enum

import chess



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


class Reward(object):
    individual_reward_types = [RewardType.SURVIVAL_OF_THE_AGENT, RewardType.AGENT_CONTRIBUTION, RewardType.INDIVIDUAL_AGGRESIVENESS, RewardType.PIECE_TYPE_SOLIDARITY, RewardType.INDIVIDUAL_DEFENSIVENESS]
    sociely_reward_types = [RewardType.TEAM_GOAL, RewardType.TEAM_SURVIVAL, RewardType.SOCIETY_AGGRESIVENESS, RewardType.SOCIETY_PATIENCE, RewardType.GROUP_MOBILITY, RewardType.SOCIETY_DEFENSIVENESS]
    all_reward_types = RewardType._member_names_

    # Alternative piece value system: Pawn = 10, Knight = 32, Bishop = 33, Rook = 50, Queen = 90, King = 100
    chess_piece_values = {
        0: 0,
        chess.KING: 100,
        chess.QUEEN: 9,
        chess.ROOK: 5,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.PAWN: 1,
    }

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
        'pawn0': 2,
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
        return

    @classmethod
    def print_all_reward_types(self):
        print(self.all_reward_types)
        return
    
    def get_rewards_from_reward_state_attributes(self, reward_attr_dict):
        # print("Reward Attributes:")
        # for key, item in reward_attr_dict.items():
        #     print(f'Reward Attr:\t{key.name}:\t{item}')

        agent_attack_square_scale_factor = [1.0 / float(item) for key,item in self.maximum_number_of_attack_squares_per_agent.items()]

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