import rclpy
import rclpy.logging
from rclpy.node import Node
import rclpy.qos

from std_msgs.msg import String

from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType, ParameterDescriptor


# NOTE: Source for using parameters loaded from config to launch nodes in launch file: https://roboticsbackend.com/ros2-yaml-params/#Write_a_YAML_config_file_for_a_ROS2_node

import chess
from . import chess_utils


from matchess_interfaces.msg import ChessMove # type: ignore
from matchess_interfaces.msg import ChessMoveVote # type: ignore

from .game_status import GAME_STATUS
from matchess_interfaces.msg import GameStatus # type: ignore
from matchess_interfaces.msg import GameHist  # type: ignore

import random


class ChessPieceAgent(Node):
    def __init__(self):
        super().__init__('chess_piece_agent')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('piece_color', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The color of the chess piece')),
                ('piece_type', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The type of chess piece')),
                ('start_file_idx', None, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='The idx of the file on the board that the chess_piece_agent occupies at the start of the game'))
            ])

        # MAS input subscription:
        self.qos = rclpy.qos.QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
            history=rclpy.qos.HistoryPolicy.KEEP_ALL, 
            depth=1,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL
            )
        self.subscription = self.create_subscription(
            ChessMove,
            'matchess/out',
            self.listener_callback,
            self.qos)
        self.subscription  # prevent unused variable warning

        # MAS output publish:
        self.publisher_ = self.create_publisher(
            ChessMoveVote,
            'matchess/in',
            self.qos)

        # Test parameters:
        self.piece_color_param = self.get_parameter('piece_color').get_parameter_value().string_value
        self.piece_type_param = self.get_parameter('piece_type').get_parameter_value().string_value


        self.file_idx = self.get_parameter('start_file_idx').get_parameter_value().integer_value

        self.agentname = self.piece_type_param + str(self.file_idx)

        # Init state variables for chess_piece_agent:
        self.piece_color = chess_utils.piece_color_from_str(self.piece_color_param)
        self.piece_type = chess_utils.piece_type_from_str(self.piece_type_param)
        

        if self.piece_color == chess.WHITE:
            if self.piece_type == chess.PAWN:
                self.rank_idx = 1
            else:
                self.rank_idx = 0
        elif self.piece_color == chess.BLACK:
            if self.piece_type == chess.PAWN:
                self.rank_idx = 6
            else:
                self.rank_idx = 7

        self.start_file_idx = self.file_idx
        self.start_rank_idx = self.rank_idx


        self.board_state = chess.Board()
        self.current_pos_square = chess.square(file_index=self.file_idx, rank_index=self.rank_idx)

        # Log init params for agent:
        piece_params = chess_utils.color_name(self.piece_color) + " " + chess.piece_name(self.piece_type) + " at " + chess.square_name(self.current_pos_square)
        self.get_logger().info('Init %s' % piece_params)

        self.is_alive = True
        self.claim_draw_allowed = True

        # save previously recieved move:
        self.prev_move_uci = ''

        # Save the random move selected by this agent:
        self.chosen_move_uci = ''

        self._debug_pub_vote_count = 0

        ##########################################
        # if self.piece_color == chess.WHITE:
        #     legal_move_count = self.board_state.legal_moves.count()
        #     random_move_idx = random.sample(range(legal_move_count), 1)[0]
        #     chosen_move = list(self.board_state.legal_moves)[random_move_idx]
        #     self.chosen_move_uci = chess.Move.uci(chosen_move)

        self.game_status_cmd_sub = self.create_subscription(
            GameStatus,
            'matchess/game_status_cmd',
            self.matchess_game_cmd_sub,
            self.qos)
        self.game_status_cmd_sub  # prevent unused variable warning

        self.game_hist_sub = self.create_subscription(
            GameHist,
            'matchess/game_hist',
            self.matchess_game_hist_sub,
            self.qos
        )
        self.game_hist_sub  # prevent unused variable warning

        # self.game_status = GAME_STATUS.OK
        self.game_over = False
        self.robot_status_pub = self.create_publisher(
            GameStatus,
            'matchess/game_status',
            self.qos)
        robot_logic_timer_period = 0.5  # seconds
        self.timer = self.create_timer(robot_logic_timer_period, self.main_logic)

        self.game_status = GAME_STATUS.IDLE
        self.game_status_cmd = GAME_STATUS.NONE

    def update_game_state(self, obs_move_uci):
        if chess.Move.from_uci(obs_move_uci) not in self.board_state.legal_moves:
            return
        # Update internal state of the game:
        self.last_recieved_move = self.board_state.push_uci(obs_move_uci) # NOTE: board.push_uci() throws an error if the move is invalid

        # Verify state transistion is valid:
        if self.board_state.is_valid():
            self.get_logger().debug('Accept valid move obs from manager: "%s"' % obs_move_uci)
            
            # NOTE: Temp to ftest that state is updated:
            self.get_logger().debug('New state: "%s"' % self.board_state.fen())
        else:
            self.get_logger().debug('Reject invalid move from manager: "%s"' % obs_move_uci)
        

    def update_agent_state(self, obs_move_uci):
        if not self.is_alive:
            return
        
        # TODO: ensure that the pos of rooks are updated for castling moves:
        # update_rook_pos = False
        if (self.board_state.turn == self.piece_color) and (self.piece_type == chess.ROOK):
            chess_move = chess.Move.from_uci(obs_move_uci)
            if self.board_state.is_castling(chess_move):
                # update rook position for castling moves:
                self.rank_idx = chess.square_rank(chess_move.from_square)
                if self.board_state.is_kingside_castling(chess_move):
                    self.file_idx = 5   # f-file
                elif self.board_state.is_queenside_castling(chess_move):
                    self.file_idx = 3   # d-file
                self.current_pos_square = chess.square(file_index=self.file_idx, rank_index=self.rank_idx)

        # if len(self.board_state.move_stack) > 0:
        #     last_recieved_move = self.board_state.peek()

        # Check if this piece has been captured:
        if self.last_recieved_move.to_square == self.current_pos_square:
            self.is_alive = False
            self.get_logger().info('I was captured with move: "%s" ' % obs_move_uci)
            # TODO: Handle moving the physical robot off the board???
        elif self.last_recieved_move.from_square == self.current_pos_square:
            self.current_pos_square = self.last_recieved_move.to_square   # Update piece position
            self.file_idx = chess.square_file(self.current_pos_square)
            self.rank_idx = chess.square_rank(self.current_pos_square)

        # TODO: Handle en passant captures!!!
        return
    
    def choose_next_uci_move(self):
        # NOTE: Temp. random move selection until ML method is implemented
        if self.board_state.turn == self.piece_color and self.is_alive:
            # Choose a random move to vote for:
            legal_move_count = self.board_state.legal_moves.count()
            random_move_idx = random.sample(range(legal_move_count), 1)[0]
            chosen_move = list(self.board_state.legal_moves)[random_move_idx]
            self.chosen_move_uci = chess.Move.uci(chosen_move)
        return

    def pub_agent_decision(self, state_transition_observed):
        if self.board_state.turn == self.piece_color and self.is_alive:
            if state_transition_observed:
                # Choose a random move to vote for:
                self.choose_next_uci_move()
                # legal_move_count = self.board_state.legal_moves.count()
                # random_move_idx = random.sample(range(legal_move_count), 1)[0]
                # chosen_move = list(self.board_state.legal_moves)[random_move_idx]
                # self.chosen_move_uci = chess.Move.uci(chosen_move)

                temp_debug_log = 'Agent: ' + self.agentname + '\tMoveVote: ' + self.chosen_move_uci
                self.get_logger().debug(temp_debug_log)

            # NOTE: This is temporary, should be peer-to-peer communication (could be centralized so the king manages the coordination for the team???
            msg_out = ChessMoveVote()
            msg_out.uci = self.chosen_move_uci
            msg_out.agentname = self.agentname # chess.square_name(self.current_pos_square)
            
            alive_pieces_on_team = {key: value for key, value in self.board_state.piece_map().items() if value.color == self.piece_color}
            msg_out.agentcount = int(len(alive_pieces_on_team))
            
            self.publisher_.publish(msg_out)
            self.get_logger().debug('Pub MoveVote: "%s"' % msg_out.uci)
        return

    def listener_callback(self, msg):
        if self.game_status == GAME_STATUS.STOP:
            return
        
        if self.game_status == GAME_STATUS.KILL_NODE:
            return
        
        # if self.game_status == GAME_STATUS.START_GAME:
        #     self.choose_next_uci_move()
        #     self.game_status = GAME_STATUS.GAME_IN_PROGRESS
        
        # # Temp fix for dead agents not reacting to any messages:
        # if not self.is_alive:
        #     return
        if self.game_status != GAME_STATUS.GAME_IN_PROGRESS:
            return
        
        state_transition_observed = False
        # if self.prev_move_uci == msg.uci:
        #     return
        if (self.prev_move_uci == msg.uci):
            # Do nothing if this is the firt move in the game and the player is not white:
            if self.prev_move_uci != "" and (self.piece_color != chess.WHITE):
                return
        else:
            self.prev_move_uci = msg.uci
            state_transition_observed = True
            self.chosen_move_uci = ''
            self.get_logger().debug('Current state: "%s"' % self.board_state.fen())
            self._debug_pub_vote_count = 0

        # Log the incomming message:
        self.get_logger().debug('Recieved Obs. Move: "%s"' % msg.uci)

        if msg.uci != "":
            
            self.update_game_state(msg.uci)

            self.update_agent_state(msg.uci)

            # Check if game is over:
            if self.board_state.is_game_over(claim_draw=self.claim_draw_allowed):
                # TODO: handle rewards
                self.get_logger().debug('Game over with move: "%s"' % msg.uci)

                # Update game_status:
                # self.game_status = GAME_STATUS.STOP # This will destroy the node
                self.game_status = GAME_STATUS.GAME_OVER
                return
        
        self.pub_agent_decision(state_transition_observed)


    def matchess_game_cmd_sub(self, game_status_cmd_msg):
        self.get_logger().info('I heard: "%s"' % game_status_cmd_msg.status_str)
        self.game_status_cmd = GAME_STATUS(game_status_cmd_msg.status_int)

    def matchess_game_hist_sub(self, game_hist_uci_msg):
        if self.game_status == GAME_STATUS.SET_GAME_STATE_FROM_HIST:
            self.reset_game()
            for uci_move_str in game_hist_uci_msg.move_hist_uci:
                self.update_game_state(uci_move_str)

                self.update_agent_state(uci_move_str)

                # Check if game is over:
                if self.board_state.is_game_over(claim_draw=self.claim_draw_allowed):
                    # TODO: handle rewards
                    self.get_logger().info('Game over with move: "%s"' % uci_move_str)

                    # Update game_status:
                    self.game_status = GAME_STATUS.GAME_OVER # This will destroy the node
                    return
            
            self.game_status = GAME_STATUS.READY_TO_PLAY

    def reset_game(self):
        self.board_state = chess.Board()
        self.board_state.reset()
        self.file_idx = self.start_file_idx
        self.rank_idx = self.start_rank_idx
        self.current_pos_square = chess.square(file_index=self.file_idx, rank_index=self.rank_idx)
        self.is_alive = True

    def pub_game_status(self):
        msg_game_status = GameStatus()
        msg_game_status.status_str = self.game_status.name
        msg_game_status.status_int = self.game_status.value
        self.robot_status_pub.publish(msg_game_status)
        return
    
    def main_logic(self):
        '''
        main robot logic goes here
        '''
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
            # self.robot_status_pub.publish(msg_game_status)

            self.pub_game_status()

            raise Exception("Shutting down node, Game over")
        elif self.game_status == GAME_STATUS.GAME_OVER:
            self.pub_game_status()



def main(args=None):
    rclpy.init(args=args)

    chess_piece_agent_node = ChessPieceAgent()

    try:
        rclpy.spin(chess_piece_agent_node)
    except Exception as err:
        rclpy.logging.get_logger("node shutdown").info(f"{err.args}")
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    chess_piece_agent_node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()



# class ChessPieceAgent(Node):
#     def __init__(self):
#         super().__init__('chess_piece_agent')
#         self.declare_parameters(
#             namespace='',
#             parameters=[
#                 ('piece_color', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The color of the chess piece')),
#                 ('piece_type', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The type of chess piece')),
#                 ('start_file_idx', None, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='The idx of the file on the board that the chess_piece_agent occupies at the start of the game'))
#             ])

#         # MAS input subscription:
#         self.qos = rclpy.qos.QoSProfile(
#             reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, 
#             history=rclpy.qos.HistoryPolicy.KEEP_ALL, 
#             depth=1)
#         self.subscription = self.create_subscription(
#             ChessMove,
#             'matchess/out',
#             self.listener_callback,
#             self.qos)
#         self.subscription  # prevent unused variable warning

#         # MAS output publish:
#         self.publisher_ = self.create_publisher(
#             ChessMoveVote,
#             'matchess/in',
#             self.qos)

#         # Test parameters:
#         self.piece_color_param = self.get_parameter('piece_color').get_parameter_value().string_value
#         self.piece_type_param = self.get_parameter('piece_type').get_parameter_value().string_value


#         self.file_idx = self.get_parameter('start_file_idx').get_parameter_value().integer_value

#         self.agentname = self.piece_type_param + str(self.file_idx)

#         # Init state variables for chess_piece_agent:
#         self.piece_color = chess_utils.piece_color_from_str(self.piece_color_param)
#         self.piece_type = chess_utils.piece_type_from_str(self.piece_type_param)
        

#         if self.piece_color == chess.WHITE:
#             if self.piece_type == chess.PAWN:
#                 self.rank_idx = 1
#             else:
#                 self.rank_idx = 0
#         elif self.piece_color == chess.BLACK:
#             if self.piece_type == chess.PAWN:
#                 self.rank_idx = 6
#             else:
#                 self.rank_idx = 7


#         self.board_state = chess.Board()
#         self.current_pos_square = chess.square(file_index=self.file_idx, rank_index=self.rank_idx)

#         # Log init params for agent:
#         piece_params = chess_utils.color_name(self.piece_color) + " " + chess.piece_name(self.piece_type) + " at " + chess.square_name(self.current_pos_square)
#         self.get_logger().info('Init %s' % piece_params)

#         self.is_alive = True

#         # save previously recieved move:
#         self.prev_move_uci = ''

#         # Save the random move selected by this agent:
#         self.chosen_move_uci = ''

#         self._debug_pub_vote_count = 0

#         ##########################################
#         if self.piece_color == chess.WHITE:
#             legal_move_count = self.board_state.legal_moves.count()
#             random_move_idx = random.sample(range(legal_move_count), 1)[0]
#             chosen_move = list(self.board_state.legal_moves)[random_move_idx]
#             self.chosen_move_uci = chess.Move.uci(chosen_move)

#         self.game_status_cmd_sub = self.create_subscription(
#             GameStatus,
#             'matchess/game_status_cmd',
#             self.matchess_game_cmd_sub,
#             self.qos)
#         self.game_status_cmd_sub  # prevent unused variable warning

#         self.game_status = GAME_STATUS.OK
#         self.game_over = False
#         self.robot_loop_publisher_ = self.create_publisher(
#             GameStatus,
#             'matchess/game_status',
#             self.qos)
#         robot_logic_timer_period = 0.5  # seconds
#         self.timer = self.create_timer(robot_logic_timer_period, self.main_logic)

#     def update_game_state(self, obs_move_uci):
#         if chess.Move.from_uci(obs_move_uci) not in self.board_state.legal_moves:
#             return
#         # Update internal state of the game:
#         self.last_recieved_move = self.board_state.push_uci(obs_move_uci) # NOTE: board.push_uci() throws an error if the move is invalid

#         # Verify state transistion is valid:
#         if self.board_state.is_valid():
#             self.get_logger().debug('Accept valid move obs from manager: "%s"' % obs_move_uci)
            
#             # NOTE: Temp to ftest that state is updated:
#             self.get_logger().debug('New state: "%s"' % self.board_state.fen())
#         else:
#             self.get_logger().debug('Reject invalid move from manager: "%s"' % obs_move_uci)
        

#     def update_agent_state(self, obs_move_uci):
#         if not self.is_alive:
#             return
        
#         # TODO: ensure that the pos of rooks are updated for castling moves:
#         # update_rook_pos = False
#         if (self.board_state.turn == self.piece_color) and (self.piece_type == chess.ROOK):
#             chess_move = chess.Move.from_uci(obs_move_uci)
#             if self.board_state.is_castling(chess_move):
#                 # update rook position for castling moves:
#                 self.rank_idx = chess.square_rank(chess_move.from_square)
#                 if self.board_state.is_kingside_castling(chess_move):
#                     self.file_idx = 5   # f-file
#                 elif self.board_state.is_queenside_castling(chess_move):
#                     self.file_idx = 3   # d-file
#                 self.current_pos_square = chess.square(file_index=self.file_idx, rank_index=self.rank_idx)

#         # if len(self.board_state.move_stack) > 0:
#         #     last_recieved_move = self.board_state.peek()

#         # Check if this piece has been captured:
#         if self.last_recieved_move.to_square == self.current_pos_square:
#             self.is_alive = False
#             self.get_logger().info('I was captured with move: "%s" ' % obs_move_uci)
            
#             # TODO: Handle moving the physical robot off the board???
#         elif self.last_recieved_move.from_square == self.current_pos_square:
#             self.current_pos_square = self.last_recieved_move.to_square   # Update piece position

#         # TODO: Handle en passant captures!!!
#         return

#     def pub_agent_decision(self, state_transition_observed):
#         if self.board_state.turn == self.piece_color and self.is_alive:
#             if state_transition_observed:
#                 # Choose a random move to vote for:
#                 legal_move_count = self.board_state.legal_moves.count()
#                 random_move_idx = random.sample(range(legal_move_count), 1)[0]
#                 chosen_move = list(self.board_state.legal_moves)[random_move_idx]
#                 self.chosen_move_uci = chess.Move.uci(chosen_move)

#                 temp_debug_log = 'Agent: ' + self.agentname + '\tMoveVote: ' + self.chosen_move_uci
#                 self.get_logger().info(temp_debug_log)

#             # NOTE: This is temporary, should be peer-to-peer communication (could be centralized so the king manages the coordination for the team???
#             msg_out = ChessMoveVote()
#             msg_out.uci = self.chosen_move_uci
#             msg_out.agentname = self.agentname # chess.square_name(self.current_pos_square)
            
#             alive_pieces_on_team = {key: value for key, value in self.board_state.piece_map().items() if value.color == self.piece_color}
#             msg_out.agentcount = int(len(alive_pieces_on_team))
            
#             self.publisher_.publish(msg_out)
#         return




#     def listener_callback(self, msg):
#         if self.game_status == GAME_STATUS.STOP:
#             return
        
#         # # Temp fix for dead agents not reacting to any messages:
#         # if not self.is_alive:
#         #     return
        
#         state_transition_observed = False
#         # if self.prev_move_uci == msg.uci:
#         #     return
#         if (self.prev_move_uci == msg.uci):
#             # Do nothing if this is the firt move in the game and the player is not white:
#             if self.prev_move_uci != "" and (self.piece_color != chess.WHITE):
#                 return
#         else:
#             self.prev_move_uci = msg.uci
#             state_transition_observed = True
#             self.chosen_move_uci = ''
#             self.get_logger().info('Current state: "%s"' % self.board_state.fen())
#             self._debug_pub_vote_count = 0

#         # Log the incomming message:
#         self.get_logger().info('I heard: "%s"' % msg.uci)

#         if msg.uci != "":
            
#             self.update_game_state(msg.uci)

#             self.update_agent_state(msg.uci)

#             # Check if game is over:
#             if self.board_state.is_game_over():
#                 # TODO: handle rewards
#                 self.get_logger().info('Game over with move: "%s"' % msg.uci)

#                 # Update game_status:
#                 self.game_status = GAME_STATUS.STOP # This will destroy the node
#                 return
        
#         self.pub_agent_decision(state_transition_observed)


#     def matchess_game_cmd_sub(self, game_status_cmd_msg):
#         self.game_status = game_status_cmd_msg
    
#     def main_logic(self):
#         '''
#         main robot logic goes here
#         '''

#         if self.game_status == GAME_STATUS.STOP:
#             # Pub msg about game status before shutting down node:
#             msg_game_status = GameStatus()
#             msg_game_status.status_str = self.game_status.name
#             msg_game_status.status_int = self.game_status.value
#             self.robot_loop_publisher_.publish(msg_game_status)

#             raise Exception("Shutting down node, Game over")

# def main(args=None):
#     rclpy.init(args=args)

#     chess_piece_agent_node = ChessPieceAgent()

#     try:
#         rclpy.spin(chess_piece_agent_node)
#     except Exception as err:
#         rclpy.logging.get_logger("node shutdown").info(f"{err.args}")
#     except KeyboardInterrupt:
#         pass

#     # Destroy the node explicitly
#     # (optional - otherwise it will be done automatically
#     # when the garbage collector destroys the node object)
#     chess_piece_agent_node.destroy_node()
#     rclpy.try_shutdown()


# if __name__ == '__main__':
#     main()