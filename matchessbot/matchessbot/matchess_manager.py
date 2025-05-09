import json

import rclpy
import rclpy.logging
import rclpy.qos

###############################################################################
# NOTE: Use this as inspiration for creating MatchessManager (which should be an interface between all the agents on the team and some chess game API/engine???):
# https://github.com/Shaswat2001/Multi_Agent_Path_Finding/blob/main/marl_planner/scripts/main.py
# from pettingzoo.classic import chess_v6
# from pettingzoo.classic.chess import chess_utils
from rclpy.node import Node

from std_msgs.msg import String
from matchess_interfaces.msg import ChessMove # type: ignore
from matchess_interfaces.msg import ChessMoveVote # type: ignore

from .game_status import GAME_STATUS
from matchess_interfaces.msg import GameStatus # type: ignore
from matchess_interfaces.msg import GameHist  # type: ignore


import chess
import chess.svg
from cairosvg import svg2png


class MatchessManager(Node):
    def __init__(self, ChessComm, config_filename: str='matches_manager_config.txt', use_pettingzoo_env: bool=True, train: bool=False):
        super().__init__('matchess_manager')

        # TODO: load config from file (config should contain args to train the MAS and run inference (and game history to init game from a later state of the game???)
        # TODO: Load config params from json or yaml file: player types for white and black, etc.
        # config = {
        #         'white_player': 'stockfish',
        #         'black_player': 'matchessbot',
        #         'game_history': [],
        #         }


        self.qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
            history=rclpy.qos.HistoryPolicy.KEEP_ALL, 
            depth=16,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            )
        
        # MATChess Game Manager input subscription:
        self.subscription = self.create_subscription(
            ChessMoveVote,
            'matchess/in',
            self.listener_callback,
            self.qos
        )
        self.subscription  # prevent unused variable warning


        # MATChess Game Manager state publisher:
        self.publisher_ = self.create_publisher(
            ChessMove,
            'matchess/out',
            self.qos
        )
        

        self.game_status_sub = self.create_subscription(
            GameStatus,
            'matchess/game_status',
            self.agent_status_callback,
            self.qos
        )
        self.game_status_sub  # prevent unused variable warning

        # Subscriber that works as a CLI for user input:
        self.cli_sub = self.create_subscription(
            GameStatus,
            'matchess/ui_cli',
            self.ui_cli_callback,
            self.qos
        )
        self.cli_sub  # prevent unused variable warning


        self.game_status_cmd_pub = self.create_publisher(
            GameStatus,
            'matchess/game_status_cmd',
            self.qos
        )

        self.game_status_hist_pub = self.create_publisher(
            GameHist,
            'matchess/game_hist',
            self.qos
        )
        
        
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.previous_move_uci = ''
        self.engine = None

        self.vote_count_during_turn = 0
        self.turn_agent_count = 0
        self.move_votes = {}
        self.recieved_vote_from_agent = {}
        self.prev_vote_from_agent = {}

        self.log_game_hist_png = ""
        self.game_half_move_count = 0
        self.log_game_hist = {}

        # Logging:
        self.global_board_state = chess.Board()
        
        svg_text = chess.svg.board(
                self.global_board_state,
                # fill=dict.fromkeys(self.global_board_state.attacks(chess.E4), "#cc0000cc"),
                # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
                # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
                size=350)
        self.game_hist_path_prefix = 'test_data/games/test_render_board_'

        hist_log_img_filename = self.game_hist_path_prefix + str(self.game_half_move_count)
        self.hist_log_moves_filename = 'test_data/games/game_hist.txt'

        self.log_game_vote_hist_filename = 'test_data/games/log_game_vote_hist.json'

        # # Save img of initial state as .svg:
        # with open(hist_log_img_filename + '.svg', 'w') as f:
        #     f.write(svg_text)
        
        # Save img of initial state as .png:
        svg2png(bytestring=svg_text, write_to=hist_log_img_filename + '.png')

        with open(self.hist_log_moves_filename, 'w') as f:
            f.write('Game log (half moves):')

    def listener_callback(self, msg_in):
        if msg_in.agentname in self.recieved_vote_from_agent.keys():
            return
        
        # if msg_in.agentname in self.prev_vote_from_agent.keys():
        #     self.get_logger().debug('Agent: %s has not been updated yet' % msg_in.agentname)
        #     return


        if not msg_in.uci in self.move_votes.keys():
            self.move_votes[msg_in.uci] = 0

        self.move_votes[msg_in.uci] += 1
        self.vote_count_during_turn += 1
        
        if self.turn_agent_count == 0:
            self.turn_agent_count = msg_in.agentcount
        elif msg_in.agentcount != self.turn_agent_count:
            self.get_logger().error(f"Recieved agentcount {msg_in.agentcount} conflicts with set {self.turn_agent_count}")


        # self.recieved_vote_from_agent.append(msg_in.agentname)
        self.recieved_vote_from_agent[msg_in.agentname] = msg_in.uci

        vote_count_string = msg_in.agentname + ' vote for ' + msg_in.uci + '\t(' + str(self.vote_count_during_turn) + ' votes of ' + str(msg_in.agentcount) + ' agents)'
        self.get_logger().debug(vote_count_string)

    def timer_callback(self):
        if self.turn_agent_count == 0:
            return
        
        if self.vote_count_during_turn != self.turn_agent_count:
            return

        # Store the move with the most votes:
        self.previous_move_uci = max(self.move_votes, key=self.move_votes.get)
        self.get_logger().debug('Move %s won the vote' % self.previous_move_uci)

        self.game_half_move_count += 1
        self.log_game_hist_png += str(self.previous_move_uci) + ' '
        self.log_game_hist[self.game_half_move_count] = {"Votes": self.move_votes, "agent Votes": self.recieved_vote_from_agent ,"Hist": self.log_game_hist_png}
        self.get_logger().debug('Game log: %s' % json.dumps(self.log_game_hist[self.game_half_move_count]))

        # Reset the variables used for collecting and counting votes:
        self.vote_count_during_turn = 0
        self.turn_agent_count = 0
        self.move_votes = {}
        self.prev_vote_from_agent = self.recieved_vote_from_agent
        self.recieved_vote_from_agent = {}


        # Update global state and save render of board as .png:
        self.global_board_state.push_uci(str(self.previous_move_uci))
        svg_text = chess.svg.board(
            self.global_board_state,
            size=350)

        hist_log_img_filename = self.game_hist_path_prefix + str(self.game_half_move_count)
        # # Save img of initial state as .svg:
        # with open(hist_log_img_filename + '.svg', 'w') as f:
        #     f.write(svg_text)
        
        # Save img of initial state as .png:
        svg2png(bytestring=svg_text, write_to=hist_log_img_filename + '.png')
        
        with open(self.hist_log_moves_filename, 'a') as f:
            f.write('\n' + str(self.game_half_move_count) + '.\t' + str(self.previous_move_uci))
            f.close()

        with open(self.log_game_vote_hist_filename, 'w') as log_game_vote_hist_file:
            json.dump(self.log_game_hist, log_game_vote_hist_file)
            log_game_vote_hist_file.close()

        msg_out = ChessMove()
        msg_out.uci = self.previous_move_uci
        self.publisher_.publish(msg_out)

    def agent_status_callback(self, msg_in):
        # TODO: Implement what should happen when the game is over!
        if msg_in.status_int == int(GAME_STATUS.GAME_OVER.value):
            raise Exception("Shutting down manager node, Game over")
        
    def ui_cli_callback(self, msg_in):
        msg_out = GameStatus()
        msg_out.status_str = msg_in.status_str
        msg_out.status_int = msg_in.status_int

        self.game_status_cmd_pub.publish(msg_out)
        self.get_logger().debug("pub command %s to all agents" % msg_out.status_str)
        
        if msg_in.status_int == GAME_STATUS.SET_GAME_STATE_FROM_HIST.value:
            msg_out_setup_hist = GameHist()
            # TODO: Load/input a GameHist to matchess_manager that will be used to set up the game state instead of the temporary hardcoded GAME_HIST below!
            msg_out_setup_hist.move_hist_uci = ["e2e4"]

            for uci_move in msg_out_setup_hist.move_hist_uci:
                self.game_half_move_count += 1
                self.previous_move_uci = uci_move
                self.log_game_hist_png += self.previous_move_uci + ' '
                self.log_game_hist[self.game_half_move_count] = {"Votes": {}, "agent Votes": {} ,"Hist": self.log_game_hist_png}
                self.get_logger().debug('Add Game log with move from input hist: %s' % json.dumps(self.log_game_hist[self.game_half_move_count]))

                # Update global state and save render of board as .png:
                self.global_board_state.push_uci(str(self.previous_move_uci))
                svg_text = chess.svg.board(
                    self.global_board_state,
                    size=350)

                hist_log_img_filename = self.game_hist_path_prefix + str(self.game_half_move_count)
                # # Save img of initial state as .svg:
                # with open(hist_log_img_filename + '.svg', 'w') as f:
                #     f.write(svg_text)
                
                # Save img of initial state as .png:
                svg2png(bytestring=svg_text, write_to=hist_log_img_filename + '.png')
                
                with open(self.hist_log_moves_filename, 'a') as f:
                    f.write('\n' + str(self.game_half_move_count) + '.\t' + str(self.previous_move_uci))
                    f.close()

                with open(self.log_game_vote_hist_filename, 'w') as log_game_vote_hist_file:
                    json.dump(self.log_game_hist, log_game_vote_hist_file)
                    log_game_vote_hist_file.close()

            self.game_status_hist_pub.publish(msg_out_setup_hist)
            
            self.get_logger().debug("pub GameHist to all agents. first move = %s" % msg_out_setup_hist.move_hist_uci[0])

        




def main(args=None):
    rclpy.init(args=None)

    # Init Matchess Game Manager Node
    matchess_man = MatchessManager(ChessComm=None, config_filename='matches_manager_config.txt', use_pettingzoo_env=True, train=False)

    try:
        rclpy.spin(matchess_man)
    except Exception as err:
        rclpy.logging.get_logger("node shutdown").info(f"{err.args}")
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    matchess_man.destroy_node()
    rclpy.try_shutdown()

if __name__ == '__main__':
    main()




# class MatchessManager(Node):
#     # TODO: Ask Stefan: "What is the ChessComm arg supposed to be?"
#     def __init__(self, ChessComm, config_filename: str='matches_manager_config.txt', use_pettingzoo_env: bool=True, train: bool=False):
#         super().__init__('matchess_manager')

#         # TODO: load config from file (config should contain args to train the MAS and run inference (and game history to init game from a later state of the game???)
#         # TODO: Load config params from json or yaml file: player types for white and black, etc.
#         # config = {
#         #         'white_player': 'stockfish',
#         #         'black_player': 'matchessbot',
#         #         'game_history': [],
#         #         }

#         # MATChess Game Manager input subscription:
#         self.qos = rclpy.qos.QoSProfile(
#             reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
#             history=rclpy.qos.HistoryPolicy.KEEP_ALL, 
#             depth=16)
#         self.subscription = self.create_subscription(
#             ChessMoveVote,
#             'matchess/in',
#             self.listener_callback,
#             self.qos)
#         self.subscription  # prevent unused variable warning


#         # MATChess Game Manager state publisher:
#         self.publisher_ = self.create_publisher(
#             ChessMove, # CHANGED FROM: String,
#             'matchess/out',
#             self.qos)
        
#         self.game_status_sub = self.create_subscription(
#             GameStatus,
#             'matchess/game_status',
#             self.agent_status_callback,
#             self.qos)
#         self.game_status_sub  # prevent unused variable warning
        
        
#         # TODO:
#         # Handle "lossy communication of moves to agents"
#         # Solution ideas:
#         # 1. Create timer callback that publish the move untill it has recieved an acknowledge from all pieces on the matchessbot team???
#         # 2. continually pub the previous move, and handle repeated moves in subscription callback of the chess piece agents???
#         timer_period = 0.5  # seconds
#         self.timer = self.create_timer(timer_period, self.timer_callback)

#         self.previous_move_uci = ''
#         self.engine = None
        
#         # # TODO:
#         # # 1. If white_player is not a matchessbot, then generate opponent's move (e.g. random move selection), apply the move, 
#         # # and publish the previous "chess move uci"
#         # # NOTE:
#         # # A "chess move uci" for a known "initial environment state" corresponds to an observed change in the environment from the perspective of the chess piece agents.
#         # # The input/output between MatchessManager and matchessbot is MOVES in UCI notation. Does the description above justify only communicating chess moves???
#         # if config['white_player']=='random':    # NOTE: This is just a temp test of manager output
#         #     # msg_out = ChessMove() # String()
            
#         #     # TODO: Temp test. Select a random move instead!!!
#         #     random_legal_move_uci = 'e2e4'
#         #     self.previous_move_uci = random_legal_move_uci

#         # # elif config['white_player'] == 'stockfish': # NOTE: This is just a temp test to use the stockfish engine in python
#         # #     # Implements a simple version of Stockfish opponent:
#         # #     # NOTE: SimpleEngine (synchronous version) is a blocking call!!!
#         # #     # TODO: Move this to a seperate file and make it into a ROS Node that subscribes/publishes to matchess output/input topics instead
#         # #     # TODO: Need to init stockfish player to a specific color from command line (just like the matchessbot player)
#         # #     self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
#         # #     self.engine_board = chess.Board()
#         # #     engine_result = self.engine.play(self.engine_board, chess.engine.Limit(time=0.1))
#         # #     self.previous_move_uci = engine_result.move.uci()
#         # #     self.engine_board.push(engine_result.move)

#         # #     self.engine.quit()  # temp needed to quit the engine!

#         self.vote_count_during_turn = 0
#         self.turn_agent_count = 0
#         self.move_votes = {}
#         # self.recieved_vote_from_agent = []
#         self.recieved_vote_from_agent = {}
#         self.prev_vote_from_agent = {}

#         self.log_game_hist_png = ""
#         self.game_half_move_count = 0
#         self.log_game_hist = {}

#         # Logging:
#         self.global_board_state = chess.Board()
        
#         svg_text = chess.svg.board(
#                 self.global_board_state,
#                 # fill=dict.fromkeys(self.global_board_state.attacks(chess.E4), "#cc0000cc"),
#                 # arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
#                 # squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
#                 size=350)
#         self.game_hist_path_prefix = 'test_data/games/test_render_board_'

#         hist_log_img_filename = self.game_hist_path_prefix + str(self.game_half_move_count)
#         self.hist_log_moves_filename = 'test_data/games/game_hist.txt'

#         self.log_game_vote_hist_filename = 'test_data/games/log_game_vote_hist.json'

#         # # Save img of initial state as .svg:
#         # with open(hist_log_img_filename + '.svg', 'w') as f:
#         #     f.write(svg_text)
        
#         # Save img of initial state as .png:
#         svg2png(bytestring=svg_text, write_to=hist_log_img_filename + '.png')

#         with open(self.hist_log_moves_filename, 'w') as f:
#             f.write('Game log (half moves):')

#     def listener_callback(self, msg_in):
#         ###### 1. Recieve the chosen move from the matchessbot:
        
#         if msg_in.agentname in self.recieved_vote_from_agent.keys():
#             return
        
#         if msg_in.agentname in self.prev_vote_from_agent.keys():
#             self.get_logger().debug('Agent: %s has not been updated yet' % msg_in.agentname)
#             return

#         # self.get_logger().info('I heard vote for: "%s"' % msg_in.uci)
#         # self.get_logger().info('From agent: "%s"' % msg_in.agentname)

#         if not msg_in.uci in self.move_votes.keys():
#             self.move_votes[msg_in.uci] = 0

#         self.move_votes[msg_in.uci] += 1
#         self.vote_count_during_turn += 1
        
#         if self.turn_agent_count == 0:
#             self.turn_agent_count = msg_in.agentcount
#         elif msg_in.agentcount != self.turn_agent_count:
#             self.get_logger().error(f"Recieved agentcount {msg_in.agentcount} conflicts with set {self.turn_agent_count}")


#         # self.recieved_vote_from_agent.append(msg_in.agentname)
#         self.recieved_vote_from_agent[msg_in.agentname] = msg_in.uci

#         vote_count_string = msg_in.agentname + ' vote for ' + msg_in.uci + '\t(' + str(self.vote_count_during_turn) + ' votes of ' + str(msg_in.agentcount) + ' agents)'
#         self.get_logger().debug(vote_count_string)

#         # if self.vote_count_during_turn >= msg_in.agentcount:
#         #     # Store the move with the most votes:
#         #     self.previous_move_uci = max(self.move_votes, key=self.move_votes.get)
#         #     self.get_logger().info('Move %s won the vote' % self.previous_move_uci)

#         #     self.game_half_move_count += 1
#         #     self.log_game_hist_png += str(self.previous_move_uci) + ' '
#         #     self.log_game_hist[self.game_half_move_count] = {"Votes": self.move_votes, "agent Votes": self.recieved_vote_from_agent ,"Hist": self.log_game_hist_png}
#         #     self.get_logger().info('Game log: %s' % json.dumps(self.log_game_hist))

#         #     # Reset the variables used for collecting and counting votes:
#         #     self.vote_count_during_turn = 0
#         #     self.move_votes = {}
#         #     # self.recieved_vote_from_agent = []
#         #     self.prev_vote_from_agent = self.recieved_vote_from_agent
#         #     self.recieved_vote_from_agent = {}


#         #     # Update global state and save render of board as .png:
#         #     self.global_board_state.push_uci(str(self.previous_move_uci))
#         #     svg_text = chess.svg.board(
#         #         self.global_board_state,
#         #         size=350)

#         #     hist_log_img_filename = self.game_hist_path_prefix + str(self.game_half_move_count)
#         #     # # Save img of initial state as .svg:
#         #     # with open(hist_log_img_filename + '.svg', 'w') as f:
#         #     #     f.write(svg_text)
            
#         #     # Save img of initial state as .png:
#         #     svg2png(bytestring=svg_text, write_to=hist_log_img_filename + '.png')
            
#         #     with open(self.hist_log_moves_filename, 'a') as f:
#         #         f.write('\n' + str(self.game_half_move_count) + '.\t' + str(self.previous_move_uci))
#         #         f.close()

#         #     with open(self.log_game_vote_hist_filename, 'w') as log_game_vote_hist_file:
#         #         json.dump(self.log_game_hist, log_game_vote_hist_file)
#         #         log_game_vote_hist_file.close()

#         #     # self.engine_board.push(self.previous_move_uci)

#         #     # # # Save game hist:
#         #     # # if self.game_half_move_count % 2 == 0:
#         #     # #     self.log_game_hist += str(self.game_half_move_count) + '. ' + str(self.previous_move_uci)
#         #     # # else:
#         #     # #     self.log_game_hist += ' ' + str(self.previous_move_uci) + '\t'
#         #     # self.log_game_hist_png += str(self.game_half_move_count) + '. ' + str(self.previous_move_uci) + ' '

#         #     # # self.get_logger().info('Game Hist: %s' % self.log_game_hist_png)
#         #     # # self.log_game_hist[self.game_half_move_count] = (self.move_votes, self.log_game_hist_png)

            


#         # ###### 3. publish the opponents move to the 'matchess/output'ø topic
#         # # msg_out = String()
#         # # msg_out.data = 'Hello World: %d' % msg_in.data
#         # # self.publisher_.publish(msg_out)
#         # # self.get_logger().info('Publishing: "%s"' % msg_out.data)
#         # # self.i += 1

#     def timer_callback(self):
#         if self.turn_agent_count == 0:
#             return
        
#         if self.vote_count_during_turn != self.turn_agent_count:
#             return

#         # Store the move with the most votes:
#         self.previous_move_uci = max(self.move_votes, key=self.move_votes.get)
#         self.get_logger().debug('Move %s won the vote' % self.previous_move_uci)

#         self.game_half_move_count += 1
#         self.log_game_hist_png += str(self.previous_move_uci) + ' '
#         self.log_game_hist[self.game_half_move_count] = {"Votes": self.move_votes, "agent Votes": self.recieved_vote_from_agent ,"Hist": self.log_game_hist_png}
#         self.get_logger().debug('Game log: %s' % json.dumps(self.log_game_hist))

#         # Reset the variables used for collecting and counting votes:
#         self.vote_count_during_turn = 0
#         self.turn_agent_count = 0
#         self.move_votes = {}
#         # self.recieved_vote_from_agent = []
#         self.prev_vote_from_agent = self.recieved_vote_from_agent
#         self.recieved_vote_from_agent = {}


#         # Update global state and save render of board as .png:
#         self.global_board_state.push_uci(str(self.previous_move_uci))
#         svg_text = chess.svg.board(
#             self.global_board_state,
#             size=350)

#         hist_log_img_filename = self.game_hist_path_prefix + str(self.game_half_move_count)
#         # # Save img of initial state as .svg:
#         # with open(hist_log_img_filename + '.svg', 'w') as f:
#         #     f.write(svg_text)
        
#         # Save img of initial state as .png:
#         svg2png(bytestring=svg_text, write_to=hist_log_img_filename + '.png')
        
#         with open(self.hist_log_moves_filename, 'a') as f:
#             f.write('\n' + str(self.game_half_move_count) + '.\t' + str(self.previous_move_uci))
#             f.close()

#         with open(self.log_game_vote_hist_filename, 'w') as log_game_vote_hist_file:
#             json.dump(self.log_game_hist, log_game_vote_hist_file)
#             log_game_vote_hist_file.close()

#         msg_out = ChessMove()
#         msg_out.uci = self.previous_move_uci
#         # msg_out.data = self.previous_move_uci
#         self.publisher_.publish(msg_out)

#     def agent_status_callback(self, msg_in):
#         if msg_in.status_int == int(GAME_STATUS.STOP.value):
#             raise Exception("Shutting down manager node, Game over")


# def main(args=None):
#     rclpy.init(args=None)

#     # Init Matchess Game Manager Node
#     # NOTE: this node manages input/output between the matchessbot and the chess game simulation?
#     matchess_man = MatchessManager(ChessComm=None, config_filename='matches_manager_config.txt', use_pettingzoo_env=True, train=False)

#     try:
#         rclpy.spin(matchess_man)
#     except Exception as err:
#         rclpy.logging.get_logger("node shutdown").info(f"{err.args}")
#     except KeyboardInterrupt:
#         pass

#     # Destroy the node explicitly
#     # (optional - otherwise it will be done automatically
#     # when the garbage collector destroys the node object)
#     matchess_man.destroy_node()
#     rclpy.try_shutdown()

# if __name__ == '__main__':
#     main()