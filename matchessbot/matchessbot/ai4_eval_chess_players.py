# import neat.reporting
import numpy as np

# import neat
# import visualize
# import pickle

# import sys
# sys.path.append("../")

from tqdm import tqdm

import json
import time

from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils

import chess
import chess.engine


import torch
# import torch.nn as nn
# import torch.nn.functional as F



class Player(object):
    def __init__(self):
        return NotImplementedError
    
    def get_next_move(self):
        return NotImplementedError
    
    def shut_down(self):
        return NotImplementedError


class RandomPlayer(Player):
    def __init__(self, name: str='player_0'):
        self.player_name = name
        return

    # def get_next_move(self, state_obs, action_mask):
    #     legal_actions_list = np.nonzero(action_mask)
    #     return legal_actions_list[np.random.randint(0, len(legal_actions_list))]
    
    # def get_next_move(self, chess_board_fen, action_mask):
    #     legal_actions_list = np.nonzero(action_mask)
    #     chosen_action = legal_actions_list[np.random.randint(0, len(legal_actions_list))]
    #     chess_board = chess.Board(fen=chess_board_fen)
    #     return chess.Move.uci(chess_utils.action_to_move(chess_board, chosen_action, int(self.player_name[-1])))

    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        chess_board = chess.Board(fen=chess_board_fen)
        legal_moves_uci_list = []
        for move in chess_board.legal_moves:
            legal_moves_uci_list.append(chess.Move.uci(move))
        # print(f'random_player.legal_moves: {legal_moves_uci_list}')
        return legal_moves_uci_list[np.random.randint(0, len(legal_moves_uci_list))]
    
    # def shut_down(self):
    #     return


# class NEATPlayer(Player):
#     def __init__(self, name: str='player_0'):
#         self.player_name = name
#         # print(f'\nInit NEATPlayer: {self.player_name}')

#         # print('buliding mlp to learn input embeddings???...')
#         # self.mlp = self.build_mlp(input_dim=1280, n_hidden_neurons=1000, output_dim=64)

#         return
    
#     def load_neat_net(self, _genome, _config):
#         # print(f'\tCreating network ...')
#         self.genome = _genome
#         self.config = _config
#         self.net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
#         return
    
    
#     # def get_next_move(self, state_obs, action_mask=None):
#     #     # net_input = np.reshape(state_obs, (1,-1))   # [1, 8*8*111]

#     #     state_obs_prev = state_obs[:,:,0:20]
#     #     net_input = np.reshape(state_obs_prev, (1,-1))   # [1, 8*8*20]
#     #     net_output = self.net.activate(net_input)

#     #     if action_mask is not None:
#     #         net_output = np.array(net_output) * np.array(action_mask)

#     #     print(f'nonzero(masked_net_output):\n{np.nonzero(net_output)}')
        
#     #     next_move_sorted = np.argsort(net_output)
#     #     # next_move_idx = np.nonzero()
#     #     print(f'\n\tDone sorting all possible actions!')

#     #     next_move = []
#     #     for move in next_move_sorted[::-1]:
#     #         if action_mask[move] != 0:
#     #             next_move = move
#     #             break
        
#     #     return next_move
    
#     def get_value_of_move(self, next_state):
#         state_obs_prev = next_state[:,:,0:20]
#         # net_input = np.reshape(state_obs_prev, (1,448))   # [1, 8*8*20]
#         net_input = list(np.reshape(state_obs_prev, 1280))
#         net_output = self.net.activate(net_input)

#         return net_output

#     def get_next_move(self, chess_board_fen, action_mask):
#         # this is where you would insert your policy
        
#         # Get list of all legal moves:
#         chess_board = chess.Board(fen=chess_board_fen)
#         legal_moves = chess_board.legal_moves

#         # print(f'list of all legal moves: {legal_moves}')

#         # next_boards_obs = []
#         possible_next_moves = {'move_ucis': [], 'move_scores': []} # {'move_ucis': [], 'move_scores': []}

#         for possible_move in legal_moves:
#             next_chess_board = chess.Board(fen=chess_board_fen)
#             next_chess_board.push(possible_move)

#             player_idx = self.player_name[-1]

#             possible_next_obs = chess_utils.get_observation(next_chess_board, player_idx)
#             # next_boards_obs.append(possible_next_obs)
#             # score = self.get_value_of_move(next_state=possible_next_obs[:, :, :7])
#             score = self.get_value_of_move(next_state=possible_next_obs)

#             possible_next_moves['move_ucis'].append(chess.Move.uci(possible_move))
#             possible_next_moves['move_scores'].append(score[0])
        
#         next_move_sorted = np.argsort(possible_next_moves['move_scores'])
#         # next_move_idx = np.nonzero()
#         # print(f'\n\tDone sorting all possible actions!')

#         # print(possible_next_moves['move_scores'])
#         # print(next_move_sorted)
#         # print(possible_next_moves['move_ucis'])

#         next_move_uci = possible_next_moves['move_ucis'][next_move_sorted[-1]]
#         for move_idx in next_move_sorted[::-1]:
#             if chess.Move.from_uci(possible_next_moves['move_ucis'][move_idx]) in legal_moves:
#                 next_move_uci = possible_next_moves['move_ucis'][move_idx]
#                 break
        
#         return next_move_uci

    
#     # def build_mlp(self, input_dim: int, n_hidden_neurons: int, output_dim: int):
#     #     return nn.Sequential(
#     #         nn.Flatten(start_dim=1),  # (N, d_data)
#     #         nn.Linear(input_dim, n_hidden_neurons), nn.ReLU(),
#     #         nn.Linear(n_hidden_neurons, output_dim)
#     #     )



# TODO NOTE: DOES NOT WORK YET - NEEDS DEBUGGING!!!

class PlayerStockfish(Player):
    def __init__(self, name: str='player_0'):
        self.player_name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
        
    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        chess_board = chess.Board(fen=chess_board_fen)
        engine_result = self.engine.play(chess_board, chess.engine.Limit(time=0.1))
        next_move_uci = engine_result.move.uci()
        return next_move_uci
    
    def shut_down(self):
        self.engine.close()
        return
    


class PlayerToga2(Player):
    def __init__(self, name: str='player_0'):
        self.player_name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/toga2")
        
    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        chess_board = chess.Board(fen=chess_board_fen)
        engine_result = self.engine.play(chess_board, chess.engine.Limit(time=0.1))
        next_move_uci = engine_result.move.uci()
        return next_move_uci
    
    def shut_down(self):
        self.engine.close()
        return


from chessformers.chessformers.configuration import get_configuration
from chessformers.chessformers.model import Transformer
from chessformers.chessformers.tokenizer import Tokenizer

class ChessFormerPlayer(Player):
    def __init__(self, name='player_0'):
        self.player_name=name

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
        self.model.load_state_dict(torch.load(self.cfg['load_model']))

        self.input_string = "<bos>"
        # self.boards = [self.input_string]
        self.prev_input_string = self.input_string


    def feed_info(self, move_obs) -> str:
        self.prev_input_string = self.input_string
        self.input_string += " " + move_obs
        # self.model.feedinfo(info)
        return self.input_string
        
    def get_info(self):
        try:
            self.input_string = self.model.predict(
                self.input_string, 
                stop_at_next_move=True, 
                temperature=0.2,
                )
            # self.boards.append(self.input_string)
            # print("BLACK MOVE:", self.input_string.split(" ")[-1])
        except ValueError:
            self.input_string = self.prev_input_string
            print("ILLEGAL MOVE. Please, try again.")
        except Exception as e:
            print(f"UNHANDLED EXCEPTION. Please, try again.: {e}")
        
        # return self.model.info
        return self.input_string.split(" ")[-1]
    
    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        # Convert move history from UCI notation to SAN:
        board_hist = chess.Board()
        if move_hist:
            san_move_list = []
            for uci_move_i in move_hist:
                move_i = chess.Move.from_uci(uci_move_i)
                san_move_list.append(board_hist.san(move_i))
                board_hist.push(move_i)

            self.feed_info(san_move_list[-1])
            # print(f'san move list:\t{san_move_list}')

        next_move_pred = self.get_info()

        # next_move_uci_hist_str = self.model.predict(
        #         self.input_string,
        #         stop_at_next_move=True, 
        #         temperature=0.2,
        #         )
        
        # next_move_str = next_move_uci_hist_str.split(" ")[-1]
        
        next_move = "0000"

        
        chess_board_from_fen = chess.Board(fen=chess_board_fen)
        if next_move_pred != self.tokenizer.eos_token:
            try:
                move = chess_board_from_fen.parse_san(next_move_pred)
                if move in chess_board_from_fen.legal_moves:
                    next_move = chess_board_from_fen.uci(move=move)
            except chess.IllegalMoveError:
                print('illegal move error ...')
        else:
            pred_moves_san_list = self.input_string.split(" ")
            if len(pred_moves_san_list) > 2:
                if len(move_hist)  == len(pred_moves_san_list) - 2:
                    print(f'\tlenght(input move sequence) == lenght(output move sequence)\n\t\t=> ChessFormer surrenders because the model does not know what do')
                elif len(move_hist) + 1 == len(pred_moves_san_list) - 2:
                    try:
                        move = chess_board_from_fen.parse_san(pred_moves_san_list[-2])
                        if move in chess_board_from_fen.legal_moves:
                            next_move = chess_board_from_fen.uci(move=move)
                    except chess.IllegalMoveError:
                        print('illegal move error ...')
                    
                    # print(f'\tlenght(input move sequence) + 1 == lenght(output move sequence)\n\t\t=> ChessFormer predicted Game Over after move:\t{pred_moves_san_list[-2]}')
                    # print(f'predicted output sequence (san str): {self.input_string}')
                    last_move_color = "WHITE" if len(pred_moves_san_list) % 2 == 1 else 'BLACK'
                    print(f'\tPredicted Game Over with {last_move_color}s move: {pred_moves_san_list[-2]}')
                    # print(f'\t\t#input_moves=\t{len(move_hist)}\n\t#output_moves={len(pred_moves_san_list)}')
                

        
        # if next_move_pred != self.tokenizer.eos_token:
        #     chess_board = chess.Board(fen=chess_board_fen)

        #     try:
        #         move = chess_board.parse_san(next_move_pred)
        #         if move in chess_board.legal_moves:
        #             next_move = chess_board.uci(move=move)

        #     except chess.IllegalMoveError:
        #         print('illegal move error ...')

        # else:
        #     chess_board = chess.Board(fen=chess_board_fen)
        #     pred_moves_san_list = self.input_string.split(" ")

        #     last_move_color = "WHITE" if len(pred_moves_san_list) % 2 == 0 else 'BLACK'
        #     print(f'\tPredicted Game Over with {last_move_color}s move: {pred_moves_san_list[-2]}')
        #     print(f'\t\t#input_moves=\t{len(move_hist)}\n\t#output_moves={len(pred_moves_san_list)}')

        #     if len(pred_moves_san_list) > 2:
        #         try:
        #             move = chess_board.parse_san(pred_moves_san_list[-2])
        #             if move in chess_board.legal_moves:
        #                 next_move = chess_board.uci(move=move)
        #         except chess.IllegalMoveError:
        #             print('illegal move error ...')
                    
        #         print(f'\tChessFormer predicted {next_move_pred} after move:\t{pred_moves_san_list[-2]}?')
        #         # last_move_color = "WHITE" if len(pred_moves_san_list) % 2 == 0 else 'BLACK'
        #         # print(f'\t\tPredicted Game Over with {last_move_color}s move: {pred_moves_san_list[-2]}')


        # # if next_move_pred != self.tokenizer.eos_token:
        # #     chess_board = chess.Board(fen=chess_board_fen)

        # #     try:
        # #         move = chess_board.parse_san(next_move_pred)
        # #         if move in chess_board.legal_moves:
        # #             next_move = chess_board.uci(move=move)
        # #     except chess.IllegalMoveError:
        # #         print('illegal move error ...')
        # # else:

        # #     pred_game_over_board = chess.Board(fen=chess_board_fen)
        # #     # game_is_over = pred_game_over_board.is_game_over(claim_draw=True)
        # #     print(f'\tChessFormer predicted move {next_move_pred}... for sequence:\n\t\t{self.input_string}')
        # #     print(f'\tBOARD STATE FEN:\n\t\t{pred_game_over_board.fen()}')
            

        # #     outcome = pred_game_over_board.outcome(claim_draw=True)
        # #     print(f'\tGAME OUTCOME:\t{outcome}')

        # #     if outcome is not None:
        # #         # print(f'\tBOARD STATUS:\t{pred_game_over_board.status()}')
        # #         print(f"\t\tChessFormer thinks the game is over, which is True?")
        # #         # print(f'\tLEGAL MOVES CHESSFORMER COULD PLAY: {[move.uci() for move in chess.Board(fen=chess_board_fen).legal_moves]}')
        # #     else:
        # #         print(f"\t\tChessFormer thinks the game is over, which is False?")
        # #         # print(f"\tChessFormer thinks the game is over, which is {bool(pred_game_over_board.is_game_over(claim_draw=True))}!")



        # # if next_move_pred == self.tokenizer.eos_token:
        # #     next_move = "0000"
        # # else:
        # #     chess_board = chess.Board(fen=chess_board_fen)
        # #     move = chess_board.parse_san(next_move_pred)
        # #     if move in chess_board.legal_moves:
        # #         next_move = chess_board.uci(move=move)

        # # # move = chess_board.san(next_move_uci)
        # # # next_move_uci = chess_board.uci(move)
        # # # chess.Move.from_uci()
        # # move = chess_board.parse_san(next_move_str)
        # # print(move.uci())

        # # next_move_uci = move.uci()
        
        
        # # move = chess.Move.from_uci(next_move_uci)
        
        # # if move not in chess_board.legal_moves:
        # #     print(f"{next_move_uci} IS NOT A VALID UCI MOVE!!!")

        # # try:
        # #     self.input_string = self.model.predict(
        # #         self.input_string, 
        # #         stop_at_next_move=True, 
        # #         temperature=0.2,
        # #         )
        # #     # self.boards.append(self.input_string)
        # #     # print("BLACK MOVE:", self.input_string.split(" ")[-1])
        # # except ValueError:
        # #     self.input_string = self.prev_input_string
        # #     print("ILLEGAL MOVE. Please, try again.")
        # # except Exception as e:
        # #     print(f"UNHANDLED EXCEPTION. Please, try again.: {e}")
        
        # # return self.model.info
        return next_move
    
    def shut_down(self):
        return
    
    # def is_not_game_over(self):
    #     return (len(self.input_string.split(" ")) < self.config["model"]["n_positions"] 
    #             and self.input_string.split(" ")[-1] != self.tokenizer.eos_token)
    
    # def get_attn_matrix(self):
    #     return self.model.get_attn_matrix(self.input_string)


# # #######################################################
# # class module(nn.Module):
# #     def __init__(self, hidden_size):
# #         super(module, self).__init__()
# #         self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
# #         self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
# #         self.bn1 = nn.BatchNorm2d(hidden_size)
# #         self.bn2 = nn.BatchNorm2d(hidden_size)
# #         self.activation1 = nn.SELU()
# #         self.activation2 = nn.SELU()

# #     def forward(self, x):
# #         x_input = torch.clone(x)
# #         x = self.conv1(x)
# #         x = self.bn1(x)
# #         x = self.activation1(x)
# #         x = self.conv2(x)
# #         x = self.bn2(x)
# #         x = x + x_input
# #         x = self.activation2(x)
# #         return x


# # class ChessNet(nn.Module):
# #     def __init__(self, hidden_layers=4, hidden_size=200):
# #         super(ChessNet, self).__init__()
# #         self.hidden_layers = hidden_layers
# #         self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
# #         self.module_list = nn.ModuleList([module(hidden_size) for i in range(hidden_layers)])
# #         self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

# #     def forward(self, x):
# #         x = self.input_layer(x)
# #         x = F.relu(x)

# #         for i in range(self.hidden_layers):
# #             x = self.module_list[i](x)

# #         x = self.output_layer(x)

# #         return x
# # #######################################################


# ##########
# # class MLP(nn.Module):
# #   def __init__(self, n_hidden_neurons: int):
# #     super().__init__()
# #     self.fc1 = nn.Linear(32 * 32 * 3, n_hidden_neurons)
# #     self.fc2 = nn.Linear(n_hidden_neurons, 10)

# #   def forward(self, x: torch.Tensor):
# #     x = x.flatten(start_dim=1)  # (N, d_data)
# #     h = F.relu(self.fc1(x))
# #     logits = self.fc2(h)
# #     return logits

# # def run_mlp():
# #     all_losses, all_accuracies = {}, {}
# #     best_accuracy = 0
# #     best_model = None


# ##########
# # def play_chess_game(genome, config, neat_player_idx: int=0):

# #     env = chess_v6.env(render_mode="ansi")
# #     env.reset(seed=42)

# #     players_in_game = {}
# #     for idx, agent_name in enumerate(env.agents):
# #         if idx==neat_player_idx:
# #             players_in_game[agent_name] = NEATPlayer(name=agent_name)
# #             players_in_game[agent_name].load_neat_net(genome, config)
# #         else:
# #             players_in_game[agent_name] = RandomPlayer(name=agent_name)

# #     game_result = {'rewards': None, 'termination': None, 'truncation': None, 'info': None}


# #     for agent in env.agent_iter():
# #         observation, reward, termination, truncation, info = env.last()

# #         if termination or truncation:
# #             action = None
# #         else:
# #             obs = observation['observation']
# #             mask = observation["action_mask"]

# #             # action = players_in_game[agent].get_next_move(state_obs=obs, action_mask=mask)

# #             # if isinstance(players_in_game[agent], RandomPlayer):
# #             #     action = players_in_game[agent].get_next_move(state_obs=obs, action_mask=mask)

# #             # elif isinstance(players_in_game[agent], NEATPlayer):
# #             #     # action = players_in_game[agent].get_next_move(state_obs=obs, action_mask=mask)
# #             #     print(f'fen_str: {env.env.board.fen()}')

# #             #     uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask)
# #             #     # action = env.action_space(agent).sample(mask)

# #             #     # chess_utils.make_move_mapping(uci_move)
# #             #     print(f'chosen move: {uci_move}')
# #             #     action = chess_utils.moves_to_actions[uci_move]

# #             print(f'fen_str: {env.env.board.fen()}')
# #             print(env.env.board.legal_moves)

# #             uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask)
# #             # action = env.action_space(agent).sample(mask)

# #             # print(f'chosen move: {uci_move}')
# #             pz_legal_actions = chess_utils.legal_moves(env.env.board)
# #             # print(f'chess_utils.legal_actions: {pz_legal_actions}')
# #             # pz_legal_moves = []
            
# #             # print('===== convert uci str to PZ action =====')
# #             # print(f'\tUCI Move:\tPZ Action:')
# #             uci_move_to_action = {}
# #             for possible_action in pz_legal_actions:
# #                 possible_uci_move = chess_utils.action_to_move(env.env.board,possible_action, int(str(agent[-1]))).uci()
# #                 # print(f'\t{possible_uci_move}\t\t{possible_action}')
# #                 uci_move_to_action[possible_uci_move] = possible_action
# #                 # pz_legal_moves.append(chess_utils.action_to_move(env.env.board,possible_action, int(str(agent[-1]))))
            
# #             # print(f'chess_utils.legal_moves: {pz_legal_moves}')
                
# #             action = uci_move_to_action[uci_move]

# #             # print(f'pz action: {action}')
# #             # print(f'chess_utils.action_to_move: {chess_utils.action_to_move(env.env.board, action, agent)}')

# #         env.step(action)
    
# #     # Store rewards and infos from env:
# #     game_result['rewards'] = env.rewards
# #     game_result['info'] = env.infos

# #     print(game_result)

# #     env.close()

# #     # return game_result

# #     # return player idx of the player that won the game:
# #     player_0_reward = game_result['rewards']['player_0']
# #     player_1_reward = game_result['rewards']['player_1']
# #     if player_0_reward > player_1_reward:
# #         return 0
# #     else:
# #         return 1


# def play_chess_game(genome, config, neat_player_idx: int=0):

#     env = chess_v6.env(render_mode="ansi")
#     env.reset(seed=42)

#     players_in_game = {}
#     player_rewards = {}

#     for idx, agent_name in enumerate(env.agents):
#         if idx==neat_player_idx:
#             players_in_game[agent_name] = NEATPlayer(name=agent_name)
#             players_in_game[agent_name].load_neat_net(genome, config)
#         else:
#             players_in_game[agent_name] = RandomPlayer(name=agent_name)
        
#         player_rewards[agent_name] = []

#     # game_result = {'rewards': None, 'termination': None, 'truncation': None, 'info': None}

#     game_result = {'players_in_game': players_in_game, 'uci_moves': [], 'game_info': [], 'game_over_info': None}

    

#     for agent in env.agent_iter():
#         observation, reward, termination, truncation, info = env.last()

#         player_rewards[agent].append(env.env.rewards[agent])

#         # game_result['game_info'].append({'agent': agent, 'reward': reward, 'termination': termination, 'truncation': truncation, 'info': info})

#         if termination or truncation:
#             action = None

#             # print('\nGAME OVER:')
#             # print(f'\tinfo:\t{info}')
#             # print(f'\treward:\t{reward}')

#         else:
#             obs = observation['observation']
#             mask = observation["action_mask"]
            
#             # print(f'fen_str: {env.env.board.fen()}')
#             # print(env.env.board.legal_moves)

#             uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask)
            
#             pz_legal_actions = chess_utils.legal_moves(env.env.board)
#             uci_move_to_action = {}
#             for possible_action in pz_legal_actions:
#                 possible_uci_move = chess_utils.action_to_move(env.env.board,possible_action, int(str(agent[-1]))).uci()
#                 uci_move_to_action[possible_uci_move] = possible_action
                
#             action = uci_move_to_action[uci_move]

#             game_result['uci_moves'].append(uci_move)

#         # game_result['game_over_info'] = {'agents': env.agents, 'rewards': env.rewards, 'termination': env.terminations, 'truncation': env.truncations, 'info': env.infos}

#         env.step(action)

        
#         ########## PRINTS OUTCOME OF GAME TO SCREEN ##########
#         # if action is None:
#         #     outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
            
#         #     if outcome is not None:
#         #         result = outcome.result()
#         #         print(f'result={result}')
#         #         print(f'result int={chess_utils.result_to_int(result)}')
#         #         print(f'winner={outcome.winner}')
#         #         if outcome.winner is not None:
#         #             color = 'white' if outcome.winner else 'black'
#         #             print(f'winner color={color}')
#         #     # print(type(outcome))
#         #     game_result['game_over_info'] = {'outcome': outcome}
#         ############################################################

#         game_result['game_over_info'] = {'outcome': chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)}
    
#     # # Store rewards and infos from env:
#     # game_result['rewards'] = env.env.rewards
#     # game_result['info'] = env.env.infos

#     # print(game_result['game_info'][-1])

#     # print(game_result['players_in_game'])
#     # print(game_result['game_over_info'])


#     # # print(game_uci_hist)

#     env.close()

#     # return game_result

#     # return player idx of the player that won the game:
#     # player_0_reward = game_result['rewards']['player_0']
#     # player_1_reward = game_result['rewards']['player_1']
#     # if player_0_reward > player_1_reward:
#     #     return 0#, game_result, game_uci_hist
#     # else:
#     #     return 1#, game_result, game_uci_hist

#     # print(player_rewards)

#     # if sum(player_rewards['player_0']) > sum(player_rewards['player_1']):
#     #     return 1
#     # elif sum(player_rewards['player_0']) < sum(player_rewards['player_1']):
#     #     return -1
#     # else:
#     #     print('GAME ENDED IN A DRAW!')
#     #     return 0

#     # print(player_rewards)

#     winner_idx = None
#     if player_rewards['player_0'][-1] > 0:
#         winner_idx = 0
#     elif player_rewards['player_1'][-1] > 0:
#         winner_idx = 1

#     return player_rewards['player_' + str(neat_player_idx)][-1], winner_idx


# def eval_genomes_vs_random(genomes, config, n_eval_games=10):
#     genome_win_stats = np.zeros(len(genomes))
#     round_count_per_player = np.zeros(len(genomes))

#     genome_accumulative_rewards = np.zeros(len(genomes))

#     genomes_win_stats_per_color = np.zeros((2,len(genomes)))

#     print('========== RUN NEAT EVAL GAMES ==========')

#     # Loop through all genomes:
#     for genome_idx, (genome_id, genome) in enumerate(tqdm(genomes)):
#         # print(f'\nGENOME #{genome_idx}')
#         for neat_player_start_pos in range(2):
#             # print(f'\nNEAT PLAYER START POS:\t{neat_player_start_pos}')
#             for game_i in range(n_eval_games):
#                 # winner_player_idx = play_chess_game(genome, config, neat_player_start_pos)
#                 neat_player_reward, winner_player_idx = play_chess_game(genome, config, neat_player_start_pos)
#                 genome_accumulative_rewards[genome_idx] += neat_player_reward
#                 # print(f'WINNING PLAYER:\t{winner_player_idx}')
#                 if winner_player_idx is not None:
#                     if winner_player_idx == neat_player_start_pos:
#                         genome_win_stats[genome_idx] += 1
#                         genomes_win_stats_per_color[neat_player_start_pos][genome_idx] += 1
#                 round_count_per_player[genome_idx] += 1
    
#     print(f'GENOME ACCUMULATIVE REWARDS:\t{genome_accumulative_rewards}')
#     print(f'GENOME WIN STATS:\t{genome_win_stats}')
#     print(f'ROUND COUNT PER GENOME:\t{set(round_count_per_player)}')
#     print(f'\n\nTOTAL #ROUNDS IN THIS TURNAMENT:\t{sum(round_count_per_player)}')

#     print(f'GENOME WIN STATS per color [white, black]:\t{genomes_win_stats_per_color}')

#     # for i, (genome_id, genome) in enumerate(genomes):
#     #     genome.fitness = (genome_win_stats[i] / round_count_per_player[i]) * 100.0

#     for i, (genome_id, genome) in enumerate(genomes):
#         genome.fitness = 50.0 + (genome_accumulative_rewards[i] / round_count_per_player[i]) * 100.0



# def run_neat(train_data_path):
#     config_file = train_data_path + 'neat_config.txt'

#     print('========== RUN NEAT ALGORITHM ==========')
#     print(f'Initializing population from config file ...\n\tfilename: {config_file}')

#     # Load configuration.
#     config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
#                          neat.DefaultSpeciesSet, neat.DefaultStagnation,
#                          config_file)

    
#     # Create the population, which is the top-level object for a NEAT run.
#     p = neat.Population(config)
    
#     # # Restore population from checkpoint: 
#     # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-19')

#     # Add a stdout reporter to show progress in the terminal.
#     p.add_reporter(neat.StdOutReporter(True))
#     stats = neat.StatisticsReporter()
#     p.add_reporter(stats)

#     checkpoint_filename_prefix = 'checkpoint_gen_'
#     p.add_reporter(neat.Checkpointer(generation_interval=1, filename_prefix=train_data_path + checkpoint_filename_prefix))

#     winner = p.run(eval_genomes_vs_random, 100)

#     # # Display the winning genome.
#     # print('\nBest genome:\n{!s}'.format(winner))

#     # Save winning genome for using the neat player as a chess engine model:
#     winner_file = train_data_path + 'winner_gen_' + str(p.generation) + '.pkl'
#     print(f'\nsaving winner genome to {winner_file} ...', end='\t')
#     with open(winner_file, "wb") as f:
#         pickle.dump(winner, f)
#         f.close()
#     print('DONE')

#     # Save training vizualization:
#     visualize.plot_stats(stats, ylog=False, view=True, filename=train_data_path+'avg_fitness.svg')
#     visualize.plot_species(stats, view=True, filename=train_data_path+'speciation.svg')

#     # Run evaluation:
#     n_test_games_per_start_pos = 10
#     neat_player_win_count = 0
#     neat_player_win_count_white = 0
#     neat_player_win_count_black = 0
#     n_start_pos = 2
#     print(f'Evaluating the winner against a random player for {n_test_games_per_start_pos * n_start_pos}...')
#     for start_pos in range(n_start_pos):
#         for i in tqdm(range(n_test_games_per_start_pos)):
#             neat_players_reward, game_winner = play_chess_game(winner, config, start_pos)
#             if game_winner == start_pos:
#                 neat_player_win_count += 1
#                 if start_pos == 0:
#                     neat_player_win_count_white += 1
#                 elif start_pos == 1:
#                     neat_player_win_count_black += 1
    
#     print(f'NEAT PLAYER WON {neat_player_win_count} OUT OF {n_test_games_per_start_pos * n_start_pos} GAMES AGAINST A RANDOM PLAYER')
#     print(f'NEAT PLAYER WON {neat_player_win_count_white} GAMES OUT OF {n_test_games_per_start_pos} AS WHITE')
#     print(f'NEAT PLAYER WON {neat_player_win_count_black} GAMES OUT OF {n_test_games_per_start_pos} AS BLACK')
    
#     p.reporters.end_generation(p.config, p.population, p.species)

#     return True


##################################################################################
def play_chess_vs_random(engine_player_idx, engine_type):
    
    env = chess_v6.env(render_mode="ansi")
    env.reset(seed=42)

    players_in_game = {}
    player_rewards = {}

    game_log = {'players': None, 'uci_moves': [], 'game_over_info': None}
    player_types_in_game = {'player_0': None, 'player_1': None}

    for idx, agent_name in enumerate(env.agents):
        if idx==engine_player_idx:
            players_in_game[agent_name] = init_chess_engine_player(engine_type,engine_player_idx)
            player_types_in_game[agent_name] = engine_type
        else:
            players_in_game[agent_name] = RandomPlayer(name=agent_name)
            player_types_in_game[agent_name] = "random"
        
        player_rewards[agent_name] = []
    
    game_result = {'players_in_game': player_types_in_game, 'uci_moves': [], 'game_info': [], 'game_over_info': None, 'game_over_status': None}

    game_log['players'] = player_types_in_game

    # uci_move_hist_str = ""
    winner_idx = None
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        player_rewards[agent].append(env.env.rewards[agent])

        # print(f"uci_moves hist:\t{game_result['uci_moves']}")

        if termination or truncation:
            action = None

        else:
            obs = observation['observation']
            mask = observation["action_mask"]
            
            # print(f'fen_str: {env.env.board.fen()}')
            # print(env.env.board.legal_moves)

            # uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask)
            uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask, move_hist=game_result['uci_moves'])

            pz_legal_actions = chess_utils.legal_moves(env.env.board)
            uci_move_to_action = {"0000": None}
            for possible_action in pz_legal_actions:
                possible_uci_move = chess_utils.action_to_move(env.env.board,possible_action, int(str(agent[-1]))).uci()
                # print(f'possible uci moves: {possible_uci_move}')
                uci_move_to_action[possible_uci_move] = possible_action
                
            action = uci_move_to_action[uci_move]

            # if uci_move == "0000":
            #     chess_board = chess.Board(fen=env.env.board.fen())
            #     outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
            
            #     if outcome is not None:
            #         result = outcome.result()
            #         print(f'\tRESULT:\t{result}')
            #         # print(f'result int={chess_utils.result_to_int(result)}')
            #         # print(f'winner={outcome.winner}')
            #         if outcome.winner is not None:
            #             # color = 'white' if outcome.winner else 'black'
            #             winner_idx = 0 if outcome.winner else 0
            #             player_rewards['player_' + str(winner_idx)].append(1)
            #             # print(f'winner color={color}')
            #     else:
            #         if not chess_board.is_game_over(claim_draw=True):
            #             player_rewards[agent].append(-1)
                
            #     # print(type(outcome))
            #     game_result['game_over_info'] = {'outcome': outcome, 'last_board_fen': chess_board.fen()}
            #     break


            #     # TODO: Fix None action error with???: chess_v6.raw_env.set_game_result


            if action is None:
                chess_board = chess.Board(fen=env.env.board.fen())
                # if chess_board.is_game_over(claim_draw=True):

                # env.env.terminations[env.env.agent_selection] = False
                # env.env.truncations[env.env.agent_selection] = True
                
                outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
            
                if outcome is not None:
                    # result = outcome.result()
                    # # print(f'result={result}')
                    # # print(f'result int={chess_utils.result_to_int(result)}')
                    # # print(f'winner={outcome.winner}')
                    # if outcome.winner is not None:
                    #     # color = 'white' if outcome.winner else 'black'
                    #     winner_idx = 0 if outcome.winner else 0
                    #     player_rewards['player_' + str(winner_idx)].append(1)
                    #     # print(f'winner color={color}')


                    # game_result['game_over_status'] = {'termination': outcome.termination}
                    game_result['game_over_status'] = {'termination': outcome.termination, 'winner_idx': outcome.winner}
                    
                else:
                    if not chess_board.is_game_over(claim_draw=True):
                        player_rewards[agent].append(-1)
                    
                    if uci_move == "0000":
                        game_result['game_over_status'] = {'termination': agent + " made a NULL move", 'winner_idx': None}
                        


                        
                # print(type(outcome))
                game_result['game_over_info'] = {'outcome': outcome, 'last_board_fen': chess_board.fen()}
                break


                # TODO: Fix None action error with???: chess_v6.raw_env.set_game_result

            game_result['uci_moves'].append(uci_move)
            # uci_move_hist_str += " " + uci_move

        # game_result['game_over_info'] = {'agents': env.agents, 'rewards': env.rewards, 'termination': env.terminations, 'truncation': env.truncations, 'info': env.infos}

        env.step(action)

        
        # if outcome.winner is not None:
        #     # color = 'white' if outcome.winner else 'black'
        #     winner_idx = 0 if outcome.winner else 0
        #     player_rewards['player_' + str(winner_idx)].append(1)
        
        # ########## PRINTS OUTCOME OF GAME TO SCREEN ##########
        # if action is None:
        #     outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
            
        #     if outcome is not None:
        #         result = outcome.result()
        #         print(f'result={result}')
        #         print(f'result int={chess_utils.result_to_int(result)}')
        #         print(f'winner={outcome.winner}')
        #         if outcome.winner is not None:
        #             color = 'white' if outcome.winner else 'black'
        #             print(f'winner color={color}')
        #     # print(type(outcome))
        #     game_result['game_over_info'] = {'outcome': outcome}
        # ############################################################

        # game_result['game_over_info'] = {'outcome': chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)}
        outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
        game_result['game_over_info'] = {'outcome': outcome, 'last_board_fen': chess.Board(fen=env.env.board.fen()).fen()}
        if outcome is not None:
            if outcome.winner is not None:
                winner_idx = 0 if outcome.winner else 1
                player_rewards['player_' + str(winner_idx)].append(1)
            game_result['game_over_status'] = {'termination': outcome.termination, 'winner_idx': outcome.winner}
                
    game_result['game_info'] = {'player_rewards': player_rewards}


    color = 'white' if engine_player_idx==0 else 'black'
    print(f'\t{engine_type} playing as color:\t{color}')
    print(f'\tGAME OUTCOME:\t{outcome.termination if outcome is not None else None}')
    print(game_result["players_in_game"])
    print(game_result['game_over_info'])
    print(game_result['game_info'])
    print(game_result['uci_moves'])
    print(game_result['game_over_status'])
    print('\n\n')

    # print(game_result)

    # game_log['uci_moves'] = game_result['uci_moves']
    # game_log['game_over_info'] = game_result['game_over_info']


    env.close()

    # winner_idx = None
    # if player_rewards['player_0'][-1] > 0:
    #     winner_idx = 0
    # elif player_rewards['player_1'][-1] > 0:
    #     winner_idx = 1



    # game_log['uci_moves'] = game_result['uci_moves']
    # game_termination = game_result['game_over_status']
    # game_over_data = game_result['game_over_info']
    # game_outcome = game_over_data['outcome']
    # if game_over_data is not None:
    #     game_log['termination'] = game_outcome['termination']
    # game_log['game_over_info']

    


    # Shut down the engine player:
    players_in_game['player_' + str(engine_player_idx)].shut_down()

    return player_rewards['player_' + str(engine_player_idx)][-1], winner_idx, game_result


def init_chess_engine_player(engine_type, player_idx):
    if engine_type == "stockfish":
        return PlayerStockfish(name='player_' + str(player_idx))
    elif engine_type == "toga2":
        return PlayerToga2(name='player_' + str(player_idx))
    elif engine_type == "chessformer":
        cfg_dict = {'config': "chessformers/configs/default.yaml",
                    'load_model': "chessformers/model/chessformer_epoch_13.pth",
                    'tokenizer': "chessformers/vocabs/kaggle2_vocab.txt"}
        engine = ChessFormerPlayer(name='player_' + str(player_idx))
        engine.load_model(cfg_dict)
        return engine
    else:
        raise NotImplementedError



def eval_chess_engines_vs_random(n_eval_games:int=10):
    n_start_pos=2

    chess_engine_player_types = ["chessformer", "stockfish", "toga2"]
    # chess_engine_player_types = ["chessformer"]

    tournament_stats_engine_vs_random = {}


    engine_win_stats = np.zeros(len(chess_engine_player_types))
    round_count_per_player = np.zeros(len(chess_engine_player_types))
    engine_accumulative_rewards = np.zeros(len(chess_engine_player_types))
    engine_win_stats_per_color = np.zeros((2,len(chess_engine_player_types)))

    engine_rewards = np.zeros((len(chess_engine_player_types), n_start_pos, n_eval_games))

    # Loop through all engines:
    for engine_idx, chess_engine_player_type in enumerate(tqdm(chess_engine_player_types)):
        print(f'\n\n========== EVAL {chess_engine_player_type} ENGINE ==========')
        engine_game_date = []
        for engine_player_start_pos in range(n_start_pos):
            # print(f'\nNEAT PLAYER START POS:\t{neat_player_start_pos}')
            for game_i in range(n_eval_games):
                # winner_player_idx = play_chess_game(genome, config, neat_player_start_pos)
                neat_player_reward, winner_player_idx, game_result = play_chess_vs_random(engine_player_start_pos, chess_engine_player_type)
                engine_accumulative_rewards[engine_idx] += neat_player_reward

                engine_game_date.append(game_result)

                engine_rewards[engine_idx][engine_player_start_pos][game_i] = neat_player_reward
                # print(f'WINNING PLAYER:\t{winner_player_idx}')
                if winner_player_idx is not None:
                    if winner_player_idx == engine_player_start_pos:
                        engine_win_stats[engine_idx] += 1
                        engine_win_stats_per_color[engine_player_start_pos][engine_idx] += 1
                round_count_per_player[engine_idx] += 1
        
        tournament_stats_engine_vs_random[chess_engine_player_type] = engine_game_date
    
    print(f'ENGINE ACCUMULATIVE REWARDS:\t{engine_accumulative_rewards}')
    print(f'ENGINE WIN STATS:\t{engine_win_stats}')
    print(f'ROUND COUNT PER ENGINE:\t{set(round_count_per_player)}')
    print(f'\n\nTOTAL #ROUNDS IN THIS TURNAMENT:\t{sum(round_count_per_player)}')

    print(f'ENGINE WIN STATS per color [white, black]:\t{engine_win_stats_per_color}')

    print(f'========== ENGINE REWARDS ==========')
    for i, engine_type in enumerate(chess_engine_player_types):
        print(f'\tengine:\t{engine_type}')
        print(f'rewards as white: {engine_rewards[i][0]}')
        print(f'rewards as black: {engine_rewards[i][1]}')

    for key, value in tournament_stats_engine_vs_random.items():
        print(f'\n\n========== GAME LOGS ==========')
        print(f'\tCHESS ENGINE:\t{key}')

        engine_games = value

        game_outcome_as_white = {}
        game_outcome_as_black = {}


        for idx, game_result in enumerate(engine_games):
            players = game_result["players_in_game"]

            game_over_status = game_result['game_over_status']

            outcome = None
            if game_over_status is not None:
                outcome = game_over_status['termination']

            
            if players['player_1'] == 'random':
                game_outcome_as_white[idx] = outcome
            elif players['player_0'] == 'random':
                game_outcome_as_black[idx] = outcome

            # game_over_info = game_result['game_over_info']
            # # print(game_data["players_in_game"])
            # # print(game_data["game_over_info"])
            # # print(game_data['game_over_status'])
            # print(game_result["players_in_game"])
            # print(game_over_info['last_board_fen'])
            # print(game_result['game_info'])
            # print(game_result['uci_moves'])
            # # print(game_over_info['outcome'])
            # print(game_result['game_over_status'])
            # print('\n\n')
        
        print(f'outcome as white: {game_outcome_as_white}\n')
        print(f'outcome as black: {game_outcome_as_black}')
    
    return



######################################################################################
def play_chess_engine_vs_engine(engine_player_0_idx, engine_0_type, engine_player_1_idx, engine_1_type):
    
    env = chess_v6.env(render_mode="ansi")
    env.reset(seed=42)

    players_in_game = {}
    player_rewards = {}

    game_log = {'players': None, 'uci_moves': [], 'game_over_info': None}
    player_types_in_game = {'player_0': None, 'player_1': None}

    players_in_game['player_0'] = init_chess_engine_player(engine_0_type,engine_player_0_idx)
    player_types_in_game['player_0'] = engine_0_type

    players_in_game['player_1'] = init_chess_engine_player(engine_1_type,engine_player_1_idx)
    player_types_in_game['player_1'] = engine_1_type

    # for idx, agent_name in enumerate(env.agents):
    #     if idx==engine_player_0_idx:
    #         players_in_game[agent_name] = init_chess_engine_player(engine_0_type,engine_player_0_idx)
    #         player_types_in_game[agent_name] = engine_0_type
    #     else:
    #         players_in_game[agent_name] = RandomPlayer(name=agent_name)
    #         player_types_in_game[agent_name] = "random"
        
    player_rewards['player_0'] = []
    player_rewards['player_1'] = []
    
    game_result = {'players_in_game': player_types_in_game, 'uci_moves': [], 'game_info': [], 'game_over_info': None, 'game_over_status': None}

    game_log['players'] = player_types_in_game

    winner_idx = None
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        player_rewards[agent].append(env.env.rewards[agent])

        if termination or truncation:
            action = None

        else:
            obs = observation['observation']
            mask = observation["action_mask"]

            uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask, move_hist=game_result['uci_moves'])

            pz_legal_actions = chess_utils.legal_moves(env.env.board)
            uci_move_to_action = {"0000": None}
            for possible_action in pz_legal_actions:
                possible_uci_move = chess_utils.action_to_move(env.env.board,possible_action, int(str(agent[-1]))).uci()
                # print(f'possible uci moves: {possible_uci_move}')
                uci_move_to_action[possible_uci_move] = possible_action
                
            action = uci_move_to_action[uci_move]

            # if uci_move == "0000":
            #     chess_board = chess.Board(fen=env.env.board.fen())
            #     outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
            
            #     if outcome is not None:
            #         result = outcome.result()
            #         print(f'\tRESULT:\t{result}')
            #         # print(f'result int={chess_utils.result_to_int(result)}')
            #         # print(f'winner={outcome.winner}')
            #         if outcome.winner is not None:
            #             # color = 'white' if outcome.winner else 'black'
            #             winner_idx = 0 if outcome.winner else 0
            #             player_rewards['player_' + str(winner_idx)].append(1)
            #             # print(f'winner color={color}')
            #     else:
            #         if not chess_board.is_game_over(claim_draw=True):
            #             player_rewards[agent].append(-1)
                
            #     # print(type(outcome))
            #     game_result['game_over_info'] = {'outcome': outcome, 'last_board_fen': chess_board.fen()}
            #     break


            if action is None:
                chess_board = chess.Board(fen=env.env.board.fen())
                # if chess_board.is_game_over(claim_draw=True):

                # env.env.terminations[env.env.agent_selection] = False
                # env.env.truncations[env.env.agent_selection] = True
                
                outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
            
                if outcome is not None:
                    # result = outcome.result()
                    # # print(f'result={result}')
                    # # print(f'result int={chess_utils.result_to_int(result)}')
                    # # print(f'winner={outcome.winner}')
                    # if outcome.winner is not None:
                    #     # color = 'white' if outcome.winner else 'black'
                    #     winner_idx = 0 if outcome.winner else 0
                    #     player_rewards['player_' + str(winner_idx)].append(1)
                    #     # print(f'winner color={color}')


                    game_result['game_over_status'] = {'termination': outcome.termination, 'winner_idx': outcome.winner}
                    
                else:
                    if not chess_board.is_game_over(claim_draw=True):
                        player_rewards[agent].append(-1)
                    
                    if uci_move == "0000":
                        game_result['game_over_status'] = {'termination': agent + " made a NULL move", 'winner_idx': None}
                        
                game_result['game_over_info'] = {'outcome': outcome, 'last_board_fen': chess_board.fen()}
                break

            game_result['uci_moves'].append(uci_move)


        env.step(action)

        outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)
        game_result['game_over_info'] = {'outcome': outcome, 'last_board_fen': chess.Board(fen=env.env.board.fen()).fen()}
        if outcome is not None:
            if outcome.winner is not None:
                winner_idx = 0 if outcome.winner else 1
                player_rewards['player_' + str(winner_idx)].append(1)
            game_result['game_over_status'] = {'termination': outcome.termination, 'winner_idx': outcome.winner}
                
    game_result['game_info'] = {'player_rewards': player_rewards}


    # print(f'\t{engine_0_type} playing as color:\twhite')
    # print(f'\t{engine_1_type} playing as color:\tblack')
    # print(f'\tGAME OUTCOME:\t{outcome.termination if outcome is not None else None}')
    # print(game_result["players_in_game"])
    # print(game_result['game_over_info'])
    # print(game_result['game_info'])
    # print(game_result['uci_moves'])
    # print(game_result['game_over_status'])
    print(game_result)
    print('\n\n')


    env.close()


    # Shut down the engine player:
    players_in_game['player_0'].shut_down()
    players_in_game['player_1'].shut_down()

    return player_rewards['player_0'][-1], player_rewards['player_1'][-1], winner_idx, game_result


def eval_engine_vs_engine(n_eval_games:int=50):
    n_start_pos=2

    chess_engine_player_types = ["chessformer", "stockfish", "toga2"]
    # chess_engine_player_types = ["chessformer"]

    tournament_stats_engine_vs_random = {}


    engine_win_stats = np.zeros((len(chess_engine_player_types), len(chess_engine_player_types)))
    # round_count_per_player = np.zeros((len(chess_engine_player_types), len(chess_engine_player_types)))
    
    # engine_accumulative_rewards = np.zeros(len(chess_engine_player_types))

    # engine_win_stats_per_color = np.zeros((2,len(chess_engine_player_types),len(chess_engine_player_types)))

    engine_rewards = np.zeros((len(chess_engine_player_types), len(chess_engine_player_types), n_eval_games, n_start_pos))

    # Loop through all engines:
    for engine_0_idx, chess_engine_player_0_type in enumerate(tqdm(chess_engine_player_types)):
        print(f'\n\n========== EVAL {chess_engine_player_0_type} ENGINE ==========')
        engine_game_date = []
        for engine_1_idx, chess_engine_player_1_type in enumerate(chess_engine_player_types):
            # for engine_player_start_pos in range(n_start_pos):
            # print(f'\nNEAT PLAYER START POS:\t{neat_player_start_pos}')
                for game_i in range(n_eval_games):
                    # winner_player_idx = play_chess_game(genome, config, neat_player_start_pos)
                    engine_0_player_reward, engine_1_player_reward, winner_player_idx, game_result = play_chess_engine_vs_engine(0, chess_engine_player_0_type, 1, chess_engine_player_1_type)
                    # engine_accumulative_rewards[engine_0_idx, engine_1_idx] += engine_0_player_reward

                    engine_game_date.append(game_result)

                    engine_rewards[engine_0_idx][engine_1_idx][game_i][0] = engine_0_player_reward
                    engine_rewards[engine_0_idx][engine_1_idx][game_i][1] = engine_1_player_reward
                    # print(f'WINNING PLAYER:\t{winner_player_idx}')
                    if winner_player_idx is not None:
                        if winner_player_idx == 0:
                            engine_win_stats[engine_0_idx][engine_1_idx] += 1
                        elif winner_player_idx == 1:
                            engine_win_stats[engine_1_idx][engine_0_idx] += 1
                            # engine_win_stats_per_color[engine_player_start_pos][engine_0_idx] += 1
                    # round_count_per_player[engine_0_idx] += 1
        
        tournament_stats_engine_vs_random[chess_engine_player_0_type] = engine_game_date
        # tournament_stats_engine_vs_random[chess_engine_player_1_type] = engine_game_date
    
    # print(f'ENGINE ACCUMULATIVE REWARDS:\t{engine_accumulative_rewards}')
    print(f'ENGINE WIN STATS:\t{engine_win_stats}')
    # print(f'ROUND COUNT PER ENGINE:\t{set(round_count_per_player)}')
    # print(f'\n\nTOTAL #ROUNDS IN THIS TURNAMENT:\t{sum(round_count_per_player)}')

    # print(f'ENGINE WIN STATS per color [white, black]:\t{engine_win_stats_per_color}')

    print(f'========== SAVING ENGINE REWARDS ==========')
    output_engine_rewards_l = []
    for i, engine_type_white in enumerate(chess_engine_player_types):
        # print(f'\twhite engine:\t{engine_type_white}', end="\t")
        for j, engine_type in enumerate(chess_engine_player_types):
            output_engine_rewards_d = {'player_white': engine_type_white,
                                       'player_black': engine_type,
                                       'game_rewards': engine_rewards[i][j].tolist()
                                       }
            print(f'\twhite:\t{engine_type_white}\tVS\tblack:\t{engine_type}:')
            print(engine_rewards[i][j])

            output_engine_rewards_l.append(output_engine_rewards_d)
            # print(f'\n\t{engine_type}\tVS\t{engine_type_white}:')
            # print(engine_rewards[j][i])
            
            # print(f'\t\t{engine_type_white} rewards as white: {engine_rewards[i][j]}')
            # print(f'\t\t{engine_type_white} rewards as black: {engine_rewards[j][i]}')

    # game_result = {'players_in_game': player_types_in_game, 'uci_moves': [], 'game_info': [], 'game_over_info': None, 'game_over_status': None}
    filename_rewards = f'../engine_test_results/game_results_{time.time()}_rewards.json'
    with open(filename_rewards, 'w', encoding='UTF-8') as outfile:
        outfile.write(json.dumps(output_engine_rewards_l))

    print(f'\n\n========== SAVING GAME LOGS ==========')
    output_l = []
    for key, value in tournament_stats_engine_vs_random.items():
        print(f'\tCHESS ENGINE:\t{key}')

        engine_games = value

        # output_l = []
        for game_result in engine_games:
            output_d = {
                'player_white': '',
                'player_black': '',
                'termination': '',
                'winner': '',
                'fen': '',
                'uci_moves': [''],
                'player_white_reward': 0,
                'player_black_reward': 0
            }
            output_d['player_white'] = game_result['players_in_game']['player_0']
            output_d['player_black'] = game_result['players_in_game']['player_1']

            output_d['termination'] = game_result['game_over_status'].get('termination', 'none') if game_result['game_over_status'] is not None else 'none'
            if isinstance(output_d['termination'], chess.Termination):
                 output_d['termination'] = output_d['termination'].name
            if not isinstance(output_d['termination'], str):
                output_d['termination'] = str(output_d['termination'])

            output_d['winner'] = game_result['game_over_status'].get('winner_idx', 'none') if game_result['game_over_status'] is not None else 'none'
            if not isinstance(output_d['winner'], str):
                output_d['winner'] = str(output_d['winner'])
            
            output_d['fen'] = game_result['game_over_info']['last_board_fen']

            output_d['uci_moves'] = game_result['uci_moves']

            output_d['player_white_reward'] = game_result['game_info']['player_rewards']['player_0']
            output_d['player_black_reward'] = game_result['game_info']['player_rewards']['player_1']

            # output_d = {
            #     'player_white': game_result['players_in_game']['player_0'],
            #     'player_black': game_result['players_in_game']['player_1'],
            #     'termination': game_result['game_over_status']['termination'].name if 
            #         isinstance(game_result['game_over_status']['termination'], chess.Termination) else
            #         str(game_result['game_over_status'].get('termination', 'none')),
            #     'winner': str(game_result['game_over_status']['winner_idx']),
            #     'fen': game_result['game_over_info']['last_board_fen'],
            #     'uci_moves': game_result['uci_moves'],
            # }
            output_l.append(output_d)
    # filename = f'../engine_test_results/game_results_{key}_{time.time()}.json'
    filename = f'../engine_test_results/game_results_{time.time()}.json'
    with open(filename, 'w', encoding='UTF-8') as outfile:
        outfile.write(json.dumps(output_l))


    # for key, value in tournament_stats_engine_vs_random.items():
    #     print(f'\n\n========== GAME LOGS ==========')
    #     print(f'\tCHESS ENGINE:\t{key}')

    #     engine_games = value

    #     game_outcome_as_white = {}
    #     game_outcome_as_black = {}

    #     # game_winner = np.zeros((len(chess_engine_player_types), n_start_pos, n_eval_games))

    #     # game_winner_name = {}
    #     # game_winner_name[key] = {}

    #     for idx, game_result in enumerate(engine_games):
    #         players = game_result["players_in_game"]

    #         game_over_status = game_result['game_over_status']

    #         outcome = None
    #         if game_over_status is not None:
    #             outcome = game_over_status['termination']

            
    #         if players['player_1'] == 'random':
    #             game_outcome_as_white[idx] = outcome
    #         elif players['player_0'] == 'random':
    #             game_outcome_as_black[idx] = outcome
        
    #     print(f'outcome as white: {game_outcome_as_white}\n')
    #     print(f'outcome as black: {game_outcome_as_black}')
    
    return




if __name__ == '__main__':
    # # train_data_path_1 = 'neat_training_data_test_1/'
    # # run_neat(train_data_path_1)

    # train_data_path_2 = 'neat_training_data_test_2/'
    # run_neat(train_data_path_2)

    # eval_chess_engines_vs_random()

    eval_engine_vs_engine()
