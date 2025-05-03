import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from pettingzoo.classic import chess_v6
# from .pettingzoo_chess import env, raw_env
# import chess_utils


class MATChessEnv(Node):

    def __init__(self):
        super().__init__('minimal_publisher')

        self.publisher_ = self.create_publisher(
            String,
            'matchess/state',
            10
        )
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0
    
        self.subscribtion_ = self.create_subscription(
            String,
            'matchess/input',
            self.chess_piece_callback,
            10
        )
        # self.subscription  # prevent unused variable warning ???

        self.chess_env = chess_v6.env(render_mode="ansi")
        self.chess_env.reset(seed=42)


    def chess_piece_callback(self, msg_in: String):
        self.get_logger().info('I heard: "%s"' % msg_in.data)

        if msg_in.data == "ReqObs":
            msg_out = String()
            msg_out.data = self.chess_env.render()

            # msg = String()
            # msg.data = 'Hello World: %d' % self.i
            self.publisher_.publish(msg_out)
            self.get_logger().info('Publishing: "%s"' % msg_out.data)
            # self.i += 1
        elif msg_in.data == "Action":
            msg_out = String()
            msg_out.data = self.chess_env.render()

            # msg = String()
            # msg.data = 'Hello World: %d' % self.i
            self.publisher_.publish(msg_out)
            self.get_logger().info('Publishing: "%s"' % msg_out.data)
            # self.i += 1


def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = MATChessEnv()

    rclpy.spin(minimal_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



# class MATChessObsPublisher(Node):

#     def __init__(self):
#         super().__init__('minimal_publisher')
#         self.publisher_ = self.create_publisher(String, 'topic', 10)
#         timer_period = 0.5  # seconds
#         self.timer = self.create_timer(timer_period, self.timer_callback)
#         self.i = 0

#     def timer_callback(self):
#         msg = String()
#         msg.data = 'Hello World: %d' % self.i
#         self.publisher_.publish(msg)
#         self.get_logger().info('Publishing: "%s"' % msg.data)
#         self.i += 1


# def main(args=None):
#     rclpy.init(args=args)

#     minimal_publisher = MATChessObsPublisher()

#     rclpy.spin(minimal_publisher)

#     # Destroy the node explicitly
#     # (optional - otherwise it will be done automatically
#     # when the garbage collector destroys the node object)
#     minimal_publisher.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()