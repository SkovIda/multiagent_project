import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from enum import Enum

piece_type = Enum('piece_type', ['PAWN', 'KNIGHT', 'BISHOP', 'ROOK', 'QUEEN', 'KING'] )

# class ChessBoard():
#     self.board = array[8][8]


class ChessPiece(Node):
    def __init__(self):
        super().__init__('chess_piece')
        self.publisher_ = self.create_publisher(String, 'topic', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

        self.type = piece_type.PAWN

    def timer_callback(self):
        msg = String()
        # msg.data = 'Hello World! %d' % self.i
        msg.data = 'Hello World! I\'m a ' + self.type.name
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

def main(args=None):

    rclpy.init(args=args)

    chess_piece = ChessPiece()

    rclpy.spin(chess_piece)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    chess_piece.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
