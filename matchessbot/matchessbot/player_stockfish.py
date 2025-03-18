import json

import rclpy
from rclpy.node import Node
import rclpy.qos

from std_msgs.msg import String
from matchess_interfaces.msg import ChessMove # type: ignore
from matchess_interfaces.msg import ChessMoveVote # type: ignore


##### Used to implement the Stockfish opponent:
import chess.engine

class PlayerStockfish(Node):
    def __init__(self):
        super().__init__('player_stockfish')

        self.agentname = 'engine'

        # TODO: Make self.piece_color into a configurable variable/paramer.
        # self.piece_color = chess.WHITE
        self.piece_color = chess.BLACK


        self.qos = rclpy.qos.QoSProfile(reliability=rclpy.qos.ReliabilityPolicy.RELIABLE, history=rclpy.qos.HistoryPolicy.KEEP_LAST, depth=1)
        self.subscription = self.create_subscription(
            ChessMove,
            'matchess/out',
            self.listener_callback,
            self.qos)
        self.subscription  # prevent unused variable warning


        self.publisher_ = self.create_publisher(
            ChessMoveVote,
            'matchess/in',
            self.qos)
        
        self.engine = chess.engine.SimpleEngine.popen_uci(r"/usr/games/stockfish")
        self.engine_board = chess.Board()
        self.prev_move_uci = ''
        # Save the random move selected by this agent:
        self.chosen_move_uci = ''
        
        if self.piece_color == chess.WHITE:
            engine_result = self.engine.play(self.engine_board, chess.engine.Limit(time=0.1))
            
            self.chosen_move_uci = engine_result.move.uci()
            # self.engine_board.push(engine_result.move)

            # msg_out = ChessMoveVote()
            # msg_out.uci = self.chosen_move_uci
            # msg_out.agentname = self.agentname
            # msg_out.agentcount = 1

            # self.publisher_.publish(msg_out)
            # self.get_logger().info('Publishing Vote by agent: "%s"' % msg_out.agentname)
            # self.get_logger().info('I vote for move: "%s"' % msg_out.uci)
        
    def listener_callback(self, msg):
        state_transition_observed = False
        if self.prev_move_uci == msg.uci:
            if self.prev_move_uci != "" and (self.piece_color != chess.WHITE):
                return
            # return
        else:
            self.prev_move_uci = msg.uci
            state_transition_observed = True
            self.chosen_move_uci = ''
            self.get_logger().info('Current state: "%s"' % self.engine_board.fen())

        # Log the incomming message:
        self.get_logger().info('I heard: "%s"' % msg.uci)

        if msg.uci != "":
            # Verify that the opponent's move is legal:
            if chess.Move.from_uci(msg.uci) in self.engine_board.legal_moves:
                # Update internal state of the game:
                opponent_move = self.engine_board.push_uci(msg.uci) # NOTE: board.push_uci() throws an error if the move is invalid

                # Verify state transistion is valid:
                if self.engine_board.is_valid():
                    self.get_logger().info('Accept valid move obs from manager: "%s"' % msg.uci)
                    
                    # NOTE: Temp to ftest that state is updated:
                    # self.get_logger().info('New state: "%s"' % self.engine_board.board_fen())
                    self.get_logger().info('New state: "%s"' % self.engine_board.fen())
                else:
                    self.get_logger().info('Reject invalid move from manager: "%s"' % msg.uci)

                # Check if game is over:
                if self.engine_board.is_game_over():
                    # TODO: handle rewards (and closing down the game properly???)
                    self.get_logger().info('Game over with move: "%s"' % msg.uci)
                    self.engine.quit()  # NOTE: needed to quit the engine?

                    # # TODO: add following line to destroy the node???
                    # self.destroy_node()
                    return

        if self.engine_board.turn == self.piece_color:
            if state_transition_observed:
                # # Choose a random move to vote for:
                # legal_move_count = self.board_state.legal_moves.count()
                # random_move_idx = random.sample(range(legal_move_count), 1)[0]
                # chosen_move = list(self.board_state.legal_moves)[random_move_idx]
                self.chosen_move_uci = self.engine.play(self.engine_board, chess.engine.Limit(time=0.1)).move.uci()

            # NOTE: This is temporary, should be peer-to-peer communication (could be centralized so the king manages the coordination for the team???
            msg_out = ChessMoveVote()
            msg_out.uci = self.chosen_move_uci
            msg_out.agentname = self.agentname
            msg_out.agentcount = 1

            self.publisher_.publish(msg_out)
            # self.get_logger().info('Publishing Vote by agent: "%s"' % msg_out.agentpos)
            self.get_logger().info('I vote for move: "%s"' % msg_out.uci)


def main(args=None):
    rclpy.init(args=None)

    # Init Matchess Game Manager Node
    # NOTE: this node manages input/output between the matchessbot and the chess game simulation?
    player_stockfish = PlayerStockfish()

    rclpy.spin(player_stockfish)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    player_stockfish.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()