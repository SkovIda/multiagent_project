import torch
import numpy as np


import chess
from pettingzoo.classic import chess_v6
from pettingzoo.classic.chess import chess_utils


from chessformers.chessformers.configuration import get_configuration
from chessformers.chessformers.model import Transformer
from chessformers.chessformers.tokenizer import Tokenizer


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

    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        chess_board = chess.Board(fen=chess_board_fen)
        legal_moves_uci_list = []
        for move in chess_board.legal_moves:
            legal_moves_uci_list.append(chess.Move.uci(move))
        # print(f'random_player.legal_moves: {legal_moves_uci_list}')
        return legal_moves_uci_list[np.random.randint(0, len(legal_moves_uci_list))]
    
# Player that utilizes the Stockfish chess engine for selecting its moves:
class PlayerStockfish(Player):
    def __init__(self, name: str='player_0'):
        self.player_name = name
        self.player_type='stockfish'
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
        
    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        chess_board = chess.Board(fen=chess_board_fen)
        engine_result = self.engine.play(chess_board, chess.engine.Limit(time=0.1))
        next_move_uci = engine_result.move.uci()
        return next_move_uci
    
    def shut_down(self):
        self.engine.close()
        return
    
# Player that utilizes the Toga2 chess engine for selecting its moves:
class PlayerToga2(Player):
    def __init__(self, name: str='player_0'):
        self.player_name = name
        self.player_type='toga2'
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/toga2")
        
    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        chess_board = chess.Board(fen=chess_board_fen)
        engine_result = self.engine.play(chess_board, chess.engine.Limit(time=0.1))
        next_move_uci = engine_result.move.uci()
        return next_move_uci
    
    def shut_down(self):
        self.engine.close()
        return


# Player that utilizes a Transformer Decoder-Only model trained for move selection:
class ChessFormerUCIPlayer(Player):
    def __init__(self, name='player_0'):
        self.player_name=name
        self.player_type='chessformer2400'

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

        # self.input_string = "<bos>"
        # self.prev_input_string = self.input_string
    
    def get_next_move(self, chess_board_fen, action_mask, move_hist):
        # Convert move history from UCI notation to SAN:
        input_string = "<bos>"
        if move_hist:
            for move in move_hist:
                input_string += " " + move

        prediction = self.model.predict_uci(
            input_string, 
            stop_at_next_move=True, 
            temperature=0.2,
            )
        
        pred_move_list = prediction.split(" ")
        print(f'Prediced move sequence:\t{pred_move_list}')

        next_move = "0000"
        next_move_pred = pred_move_list[-1]
        
        chess_board_from_fen = chess.Board(fen=chess_board_fen)
        if next_move_pred != self.tokenizer.eos_token:
            try:
                move = chess_board_from_fen.parse_uci(next_move_pred)
                if move in chess_board_from_fen.legal_moves:
                    next_move = chess_board_from_fen.uci(move=move)
            except chess.IllegalMoveError:
                print('illegal move error ...')
        else: #next_move_pred == self.tokenizer.eos_token:
            # if len(pred_move_list) > 2:
            try:
                move = chess_board_from_fen.parse_uci(pred_move_list[-2])
                next_move = pred_move_list[-2]
            except chess.IllegalMoveError:
                print('illegal move error ...')

        return next_move
    
    def shut_down(self):
        return


def init_chess_engine_player(engine_type, player_idx):
    if engine_type == "stockfish":
        return PlayerStockfish(name='player_' + str(player_idx))
    elif engine_type == "toga2":
        return PlayerToga2(name='player_' + str(player_idx))
    elif engine_type == "chessformer2400":
        cfg_dict = {'config': "chessformers/configs/config_uci.yaml",
                    'load_model': "chessformers/model/uci_test/chessformer_epoch_11.pth",
                    'tokenizer': "chessformers/vocabs/uci_vocab.txt"}
        engine = ChessFormerUCIPlayer(name='player_' + str(player_idx))
        engine.load_model(cfg_dict)
        return engine
    else:
        raise NotImplementedError
    

def play_chess_vs_random(engine_player_idx, engine_type):
    
    env = chess_v6.env(render_mode="ansi")
    env.reset(seed=42)

    players_in_game = {}

    game_log = {
        'player_0': None,
        'player_1': None,
        'uci_moves': [],
        'game_result': None,
        'termination': None
        }
    
    for idx, agent_name in enumerate(env.agents):
        if idx==engine_player_idx:
            players_in_game[agent_name] = init_chess_engine_player(engine_type,engine_player_idx)
            game_log[agent_name] = players_in_game[agent_name].player_type
        else:
            players_in_game[agent_name] = RandomPlayer(name=agent_name)
            game_log[agent_name] = players_in_game[agent_name].player_type

    winner_idx = None
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        print(game_log)

        if termination or truncation:
            action = None

        else:
            obs = observation['observation']
            mask = observation["action_mask"]
            
            uci_move = players_in_game[agent].get_next_move(chess_board_fen=env.env.board.fen(), action_mask=mask, move_hist=game_log['uci_moves'])

            pz_legal_actions = chess_utils.legal_moves(env.env.board)
            uci_move_to_action = {"0000": None}
            for possible_action in pz_legal_actions:
                possible_uci_move = chess_utils.action_to_move(env.env.board,possible_action, int(str(agent[-1]))).uci()
                uci_move_to_action[possible_uci_move] = possible_action
                
            action = uci_move_to_action[uci_move]

            if action is None:
                # chess_board = chess.Board(fen=env.env.board.fen())
                outcome = chess.Board(fen=env.env.board.fen()).outcome(claim_draw=True)

                if outcome is not None:
                    game_log['game_result'] = outcome.result()
                    game_log['termination'] = outcome.termination

                    # if outcome.winner is not None:
                    #     winner_idx = 0 if outcome.winner else 1
            

            game_log['uci_moves'].append(uci_move)
        
        env.step(action)
    

    color = 'white' if engine_player_idx==0 else 'black'
    print(f'\t{engine_type} playing as color:\t{color}')
    # print(f'\tGAME OUTCOME:\t{outcome.termination if outcome is not None else None}')
    # print(game_log["players_0"])
    # print(game_log["players_0"])
    # print(game_log['uci_moves'])
    # print(game_log['game_result'])
    # print(game_log['termination'])
    print(game_log)
    print('\n\n')

    env.close()

    players_in_game['player_' + str(engine_player_idx)].shut_down()

if __name__ == '__main__':
    play_chess_vs_random(engine_player_idx=0, engine_type='chessformer2400')