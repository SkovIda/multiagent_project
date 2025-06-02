import json

import rclpy
import rclpy.logging
from rclpy.node import Node
import rclpy.qos

from rcl_interfaces.msg import ParameterType, ParameterDescriptor

import chess.engine

from matchess_interfaces.msg import ChessMove # type: ignore
from matchess_interfaces.msg import ChessMoveVote # type: ignore

from .game_status import GAME_STATUS
from matchess_interfaces.msg import GameStatus # type: ignore
from matchess_interfaces.msg import GameHist  # type: ignore

from . import chess_utils


class PlayerStockfish(Node):
    def __init__(self):
        super().__init__('player_stockfish')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('piece_color', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The color of the chess pieces controlled by this player')),
                ('engine_path', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The file path to the UCI chess engine that this player use to make decisions')),
            ])
        piece_color_param = self.get_parameter('piece_color').get_parameter_value().string_value
        self.piece_color = chess_utils.piece_color_from_str(piece_color_param)

        self.engine_path = self.get_parameter('engine_path').get_parameter_value().string_value
        self.engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        self.engine_board = chess.Board()

        self.agentname = 'stockfish'

        self.claim_draw_allowed = True

        # save previously recieved move:
        self.prev_move_uci = ''
        
        # Save the move that this agent selects:
        self.chosen_move_uci = ''

        # Init reliable Quality of Service (QoS) profile:
        self.qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
            history=rclpy.qos.HistoryPolicy.KEEP_LAST, 
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            )
        
        # Init agent input subscription:
        self.subscription = self.create_subscription(
            ChessMove,
            'matchess/out',
            self.listener_callback,
            self.qos)
        self.subscription  # prevent unused variable warning

        # Init agent output publisher:
        self.publisher_ = self.create_publisher(
            ChessMoveVote,
            'matchess/in',
            self.qos)

        # Init subscription to commands from the Matchess Game Manager:
        self.game_status_cmd_sub = self.create_subscription(
            GameStatus,
            'matchess/game_status_cmd',
            self.matchess_game_cmd_sub,
            self.qos)
        self.game_status_cmd_sub  # prevent unused variable warning

        # Init subscription to GameHist from the Matchess Game Manager (used for setting up the game state from a list of UCI moves):
        self.game_hist_sub = self.create_subscription(
            GameHist,
            'matchess/game_hist',
            self.matchess_game_hist_sub,
            self.qos
        )
        self.game_hist_sub  # prevent unused variable warning
        self.game_state_hist = []   # List for storing the list of uci moves recieved from the matchess manager

        # Publisher for sending agent's local game status to the manager:
        self.engine_status_pub = self.create_publisher(
            GameStatus,
            'matchess/game_status',
            self.qos)
        engine_timer_period = 0.5  # seconds
        self.timer = self.create_timer(engine_timer_period, self.main_logic)

        self.game_status = GAME_STATUS.IDLE
        self.game_status_cmd = GAME_STATUS.NONE
    
    def update_game_state(self, obs_move_uci):
        if chess.Move.from_uci(obs_move_uci) not in self.engine_board.legal_moves:
            return
        
        # Update internal state of the game:
        last_chess_move = self.engine_board.push_uci(obs_move_uci) # NOTE: board.push_uci() throws an error if the move is invalid

        # Verify state transistion is valid:
        if self.engine_board.is_valid():
            self.get_logger().debug('Accept valid move obs from manager: "%s"' % obs_move_uci)
            self.get_logger().debug('New state: "%s"' % self.engine_board.fen())
        else:
            self.get_logger().debug('Reject invalid move from manager: "%s"' % obs_move_uci)
    
    def choose_next_uci_move(self):
        if self.engine_board.turn == self.piece_color:
            self.chosen_move_uci = self.engine.play(self.engine_board, chess.engine.Limit(time=0.1)).move.uci()
        return

    def pub_agent_decision(self, state_transition_observed):
        if self.engine_board.turn == self.piece_color:
            if state_transition_observed:
                self.choose_next_uci_move()

                temp_debug_log = 'Agent: ' + self.agentname + '\tMoveVote: ' + self.chosen_move_uci
                self.get_logger().debug(temp_debug_log)

            msg_out = ChessMoveVote()
            msg_out.uci = self.chosen_move_uci
            msg_out.agentname = self.agentname
            msg_out.agentcount = 1

            self.publisher_.publish(msg_out)
            self.get_logger().debug('Pub MoveVote: "%s"' % msg_out.uci)
        return


    def listener_callback(self, msg):
        if self.game_status == GAME_STATUS.STOP:
            return
        
        if self.game_status == GAME_STATUS.KILL_NODE:
            return
        
        if self.game_status != GAME_STATUS.GAME_IN_PROGRESS:
            return
        
        state_transition_observed = False
        if self.prev_move_uci == msg.uci:
            # Do nothing if this is the first move in the game and the player is not white:
            if self.prev_move_uci != "" and (self.piece_color != chess.WHITE):
                return
        else:
            self.prev_move_uci = msg.uci
            state_transition_observed = True
            self.chosen_move_uci = ''
            self.get_logger().debug('Current state: "%s"' % self.engine_board.fen())

        # Log the incomming message:
        self.get_logger().info('Recieved Obs. Move: "%s"' % msg.uci)

        if msg.uci != "":
            self.update_game_state(msg.uci)

            # Check if game is over:
            if self.engine_board.is_game_over(claim_draw=self.claim_draw_allowed):
                self.get_logger().info('Game over with move: "%s"' % msg.uci)
                
                self.game_status = GAME_STATUS.GAME_OVER

                self.engine.quit()  # NOTE: quit the engine before shutting down the node
                return
        
        self.pub_agent_decision(state_transition_observed)

    def matchess_game_cmd_sub(self, game_status_cmd_msg):
        self.get_logger().debug('I heard: "%s"' % game_status_cmd_msg.status_str)
        self.game_status_cmd = GAME_STATUS(game_status_cmd_msg.status_int)

    def matchess_game_hist_sub(self, game_hist_uci_msg):
        self.game_state_hist = game_hist_uci_msg.move_hist_uci

    def set_game_state_from_hist(self):
        if len(self.game_state_hist) > 0:
            self.reset_game()
            full_game_hist_string = ""
            for uci_move_str in self.game_state_hist:
                self.update_game_state(uci_move_str)

                full_game_hist_string += uci_move_str
                full_game_hist_string += " "

                # Check if game is over:
                if self.engine_board.is_game_over(claim_draw=self.claim_draw_allowed):
                    self.get_logger().info('Game over with move: "%s"' % uci_move_str)

                    # Update game_status:
                    self.game_status = GAME_STATUS.GAME_OVER # This will destroy the node
                    return
            
            self.get_logger().info('Done setting game state from hist %s' %full_game_hist_string)
            self.game_state_hist = []
            self.game_status = GAME_STATUS.READY_TO_PLAY

    def reset_game(self):
        self.engine_board = chess.Board()
        self.engine_board.reset()
        # self.prev_move_uci = ''
        # self.chosen_move_uci = ''
        # self.game_state_hist = []


    def pub_game_status(self):
        msg_game_status = GameStatus()
        msg_game_status.status_str = self.game_status.name
        msg_game_status.status_int = self.game_status.value
        self.engine_status_pub.publish(msg_game_status)
        return
    
    def main_logic(self):
        execute_comand = False

        if self.game_status_cmd != GAME_STATUS.NONE:
            # Commands from matchess manager that should always be executed no matter what state the agent is in:
            if self.game_status_cmd == GAME_STATUS.KILL_NODE or self.game_status_cmd == GAME_STATUS.STOP:
                execute_comand = True
            elif self.game_status_cmd == GAME_STATUS.RESET_GAME_STATE:
                self.reset_game()
                self.game_status = GAME_STATUS.READY_TO_PLAY
                self.game_status_cmd = GAME_STATUS.NONE
            elif self.game_status_cmd == GAME_STATUS.SET_GAME_STATE_FROM_HIST:
                execute_comand = True
            elif self.game_status_cmd == GAME_STATUS.START_GAME:
                if self.game_status == GAME_STATUS.READY_TO_PLAY:
                    # self.choose_next_uci_move()
                    self.pub_agent_decision(state_transition_observed=True)
                    self.game_status = GAME_STATUS.GAME_IN_PROGRESS
                    self.game_status_cmd = GAME_STATUS.NONE
                    # execute_comand = True
            
            if execute_comand:
                self.game_status = self.game_status_cmd
                self.game_status_cmd = GAME_STATUS.NONE
                execute_comand = False

        # Change agent state to execute the last command recieved from the matchess manager:
        if self.game_status == GAME_STATUS.KILL_NODE:
            # Pub msg about game status before shutting down node:
            # msg_game_status = GameStatus()
            # msg_game_status.status_str = self.game_status.name
            # msg_game_status.status_int = self.game_status.value
            # self.engine_status_pub.publish(msg_game_status)
            self.pub_game_status()

            raise Exception("Shutting down node, Game over")
        elif self.game_status == GAME_STATUS.GAME_OVER:
            self.pub_game_status()
        elif self.game_status == GAME_STATUS.SET_GAME_STATE_FROM_HIST:
            self.set_game_state_from_hist()
        


def main(args=None):
    rclpy.init(args=args)

    # Init Matchess Game Manager Node
    # NOTE: this node manages input/output between the matchessbot and the chess game simulation?
    player_stockfish = PlayerStockfish()

    try:
        rclpy.spin(player_stockfish)
    except Exception as err:
        rclpy.logging.get_logger("node shutdown").info(f"{err.args}")
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    player_stockfish.destroy_node()
    rclpy.try_shutdown()

if __name__ == '__main__':
    main()