import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType, ParameterDescriptor


# NOTE: Source for using parameters loaded from config to launch nodes in launch file: https://roboticsbackend.com/ros2-yaml-params/#Write_a_YAML_config_file_for_a_ROS2_node


# from pettingzoo.classic import chess_v6
import chess
from . import chess_utils


from matchess_interfaces.msg import ChessMove # type: ignore
from matchess_interfaces.msg import ChessMoveVote # type: ignore


import random



class ChessPieceAgent(Node):
    # def __init__(self, piece_color, piece_type, pos_square_idx):
    def __init__(self):
        super().__init__('chess_piece_agent')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('piece_color', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The color of the chess piece')),
                ('piece_type', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The type of chess piece')),
                # ('pos_square_idx', None, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='The idx of the square on the board that the chess_piece_agent is currently occupying'))
                ('start_file_idx', None, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='The idx of the file on the board that the chess_piece_agent occupies at the start of the game'))
            ])
        
        # self.piece_color = piece_color
        # self.piece_type = piece_type
        # self.pos_square_idx = pos_square_idx

        # MAS input subscription:
        self.subscription = self.create_subscription(
            ChessMove,
            'matchess/out',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # TODO: MAS output publish:
        self.publisher_ = self.create_publisher(
            ChessMoveVote,
            'matchess/in',
            10)

        # Test parameters:
        self.piece_color_param = self.get_parameter('piece_color').get_parameter_value().string_value
        self.piece_type_param = self.get_parameter('piece_type').get_parameter_value().string_value
        # piece_params = self.piece_color + " " + self.piece_type
        # self.get_logger().info('I am a %s!' % piece_params)
        self.file_idx = self.get_parameter('start_file_idx').get_parameter_value().integer_value

        self.agentname = self.piece_type_param + str(self.file_idx)

        # Init state variables for chess_piece_agent:
        # self.agent_name = self.piece_type_param + str(self.file_idx)
        self.piece_color = chess_utils.piece_color_from_str(self.piece_color_param)
        self.piece_type = chess_utils.piece_type_from_str(self.piece_type_param)
        
        # # Log init params for agent:
        # piece_params = chess.piece_name(self.piece_type) + " " + chess_utils.color_name(self.piece_color)
        # self.get_logger().info('I am a %s!' % piece_params)

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


        self.board_state = chess.Board()
        self.current_pos_square = chess.square(file_index=self.file_idx, rank_index=self.rank_idx)

        # Log init params for agent:
        piece_params = chess_utils.color_name(self.piece_color) + " " + chess.piece_name(self.piece_type) + " at " + chess.square_name(self.current_pos_square)
        self.get_logger().info('I am a %s!' % piece_params)

        self.is_alive = True

        # save previously recieved move:
        self.prev_move_uci = ''

        # Save the random move selected by this agent:
        self.chosen_move_uci = ''

        # TODO: Temp debugging ideas related to "lossy communication wrt. MoveVote msgs not being recieved by MATChessManager"
        # 1) Add counter for how many times the vote has been sent? (should be sent every time the manager sends the previous executed move by either player)
        self._debug_pub_vote_count = 0

        ##########################################
        if self.piece_color == chess.WHITE:
            legal_move_count = self.board_state.legal_moves.count()
            random_move_idx = random.sample(range(legal_move_count), 1)[0]
            chosen_move = list(self.board_state.legal_moves)[random_move_idx]
            self.chosen_move_uci = chess.Move.uci(chosen_move)

            # # NOTE: This is temporary, should be peer-to-peer communication (could be centralized so the king manages the coordination for the team???
            # msg_out = ChessMoveVote()
            # msg_out.uci = self.chosen_move_uci
            # msg_out.agentname = self.agentname # chess.square_name(self.current_pos_square)
            
            # alive_pieces_on_team = {key: value for key, value in self.board_state.piece_map().items() if value.color == self.piece_color}
            # msg_out.agentcount = int(len(alive_pieces_on_team))
            
            # self.publisher_.publish(msg_out)
            # self.get_logger().info('Publishing Vote by agent: "%s"' % msg_out.agentname)
            # self.get_logger().info('I vote for move: "%s"' % msg_out.uci)
        ##########################################

        
        # # TODO: Create common-agent to king-agent subscription:
        # self.peer2peer_subscription = self.create_subscription(
        #     ChessMove, # String,
        #     'matchess/team/',
        #     self.listener_callback,
        #     10)
        # self.peer2peer_subscription  # prevent unused variable warning

        
        # # TODO: Create commen-agent to king-agent publisher:
        # self.peer2peer_publisher_ = self.create_publisher(
        #     ChessMove,
        #     'matchess/king/in',
        #     10)
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.peer2peer_timer_callback)



    def listener_callback(self, msg):
        # Temp fix for dead agents not reacting to any messages:
        if not self.is_alive:
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
            self.get_logger().info('Current state: "%s"' % self.board_state.fen())
            self._debug_pub_vote_count = 0

        # Log the incomming message:
        self.get_logger().info('I heard: "%s"' % msg.uci)

        if msg.uci != "":
            # Verify that the observed move is legal:
            if chess.Move.from_uci(msg.uci) in self.board_state.legal_moves:
                # TODO: ensure that the pos of rooks are updated for castling moves:
                # update_rook_pos = False
                if (self.board_state.turn == self.piece_color) and (self.piece_type == chess.ROOK): # and self.board_state.is_castling(chess_move):
                    chess_move = chess.Move.from_uci(msg.uci)
                    if self.board_state.is_castling(chess_move):
                        # update_rook_pos = True
                        self.rank_idx = chess.square_rank(chess_move.from_square)
                        if self.board_state.is_kingside_castling(chess_move):
                            self.file_idx = 5   # f-file
                        elif self.board_state.is_queenside_castling(chess_move):
                            self.file_idx = 3   # d-file
                        self.current_pos_square = chess.square(file_index=self.file_idx, rank_index=self.rank_idx)
                
                # Update internal state of the game:
                recieved_move = self.board_state.push_uci(msg.uci) # NOTE: board.push_uci() throws an error if the move is invalid

                # Verify state transistion is valid:
                if self.board_state.is_valid():
                    self.get_logger().info('Accept valid move obs from manager: "%s"' % msg.uci)
                    
                    # NOTE: Temp to ftest that state is updated:
                    self.get_logger().info('New state: "%s"' % self.board_state.fen())
                else:
                    self.get_logger().info('Reject invalid move from manager: "%s"' % msg.uci)

                # Check if this piece has been captured:
                if recieved_move.to_square == self.current_pos_square:
                    self.is_alive = False
                    self.get_logger().info('I was captured with move: "%s" ' % msg.uci)
                    # TODO: Handle moving the physical robot off the board???
                    
                    # TODO: add following line to destroy the node???
                    # self.destroy_node()
                    return
                elif recieved_move.from_square == self.current_pos_square:
                    self.current_pos_square = recieved_move.to_square   # Update piece position

                # TODO: Handle en passant captures!!!

                # Check if game is over:
                if self.board_state.is_game_over():
                    # TODO: handle rewards (and closing down the game properly???)
                    self.get_logger().info('Game over with move: "%s"' % msg.uci)

                    # # TODO: add following line to destroy the node???
                    # self.destroy_node()
                    return

        if self.board_state.turn == self.piece_color and self.is_alive:
            if state_transition_observed:
                # Choose a random move to vote for:
                legal_move_count = self.board_state.legal_moves.count()
                random_move_idx = random.sample(range(legal_move_count), 1)[0]
                chosen_move = list(self.board_state.legal_moves)[random_move_idx]
                self.chosen_move_uci = chess.Move.uci(chosen_move)

                temp_debug_log = 'Agent: ' + self.agentname + '\twill vote for move: ' + self.chosen_move_uci
                self.get_logger().info(temp_debug_log)

            # NOTE: This is temporary, should be peer-to-peer communication (could be centralized so the king manages the coordination for the team???
            msg_out = ChessMoveVote()
            msg_out.uci = self.chosen_move_uci
            msg_out.agentname = self.agentname # chess.square_name(self.current_pos_square)
            
            alive_pieces_on_team = {key: value for key, value in self.board_state.piece_map().items() if value.color == self.piece_color}
            msg_out.agentcount = int(len(alive_pieces_on_team))
            
            self.publisher_.publish(msg_out)
            # # self.get_logger().info('Publishing Vote by agent: "%s"' % msg_out.agentpos)
            # self.get_logger().info('I vote for move: "%s"' % msg_out.uci)
            # # self.get_logger().info('#Agents on team still alive: %d' % msg_out.agentcount)

            # # NOTE: temp_debug_log:
            # self._debug_pub_vote_count += 1
            # temp_debug_log = 'Agent: ' + self.agentname + '\tvote for move: ' + msg_out.uci + '\tPub #' + str(self._debug_pub_vote_count)
            # self.get_logger().info(temp_debug_log)


        # # Test parameters:
        # piece_color = self.get_parameter('piece_color').get_parameter_value().string_value
        # piece_type = self.get_parameter('piece_type').get_parameter_value().string_value
        # piece_params = piece_color + piece_type
        # self.get_logger().info('I am a %s!' % piece_params)


def main(args=None):
    rclpy.init(args=args)

    chess_piece_agent_node = ChessPieceAgent()

    rclpy.spin(chess_piece_agent_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    chess_piece_agent_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()