import rclpy
from rclpy.node import Node

from std_msgs.msg import String

from rclpy.exceptions import ParameterNotDeclaredException
from rcl_interfaces.msg import ParameterType, ParameterDescriptor


# NOTE: Source for using parameters loaded from config to launch nodes in launch file: https://roboticsbackend.com/ros2-yaml-params/#Write_a_YAML_config_file_for_a_ROS2_node

class ChessPieceAgent(Node):
    # def __init__(self, piece_color, piece_type, pos_square_idx):
    def __init__(self):
        super().__init__('chess_piece_agent')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('piece_color', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The color of the chess piece')),
                ('piece_type', None, ParameterDescriptor(type=ParameterType.PARAMETER_STRING, description='The type of chess piece')),
                ('pos_square_idx', None, ParameterDescriptor(type=ParameterType.PARAMETER_INTEGER, description='The idx of the square on the board that the chess_piece_agent is currently occupying'))
            ])
        
        # self.piece_color = piece_color
        # self.piece_type = piece_type
        # self.pos_square_idx = pos_square_idx

        self.subscription = self.create_subscription(
            String,
            'matchess/state',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Test parameters:
        piece_color = self.get_parameter('piece_color').get_parameter_value().string_value
        piece_type = self.get_parameter('piece_type').get_parameter_value().string_value
        piece_params = piece_color + " " + piece_type
        self.get_logger().info('I am a %s!' % piece_params)


    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)

        # Test parameters:
        piece_color = self.get_parameter('piece_color').get_parameter_value().string_value
        piece_type = self.get_parameter('piece_type').get_parameter_value().string_value
        piece_params = piece_color + piece_type
        self.get_logger().info('I am a %s!' % piece_params)


        # self.get_logger().info('I have params: "%s"' %self.)
        # self.get_logger().info('I am chess piece: "%s"' % self.piece_color)


def main(args=None):
    rclpy.init(args=args)

    # TODO: 
    # 1) Create launch file that inits all the chess piece agents for both black and white (NOTE: configure parameters for the chess_piece_agents from this launch file)
    # 2) init the environment and publish the state
    # 3) add game logic to agents - i.e., make them capable of computing the legal moves for the given state
    # 3.1) make agents subscribe to matchess/state topic
    # 3.2) make agents compute their own moves based on input state? or just publish legal moves as part of the state?
    # 3.3) add voting system to the agents and pub their chosen move to 'matchess/input/move_req' topic

    chess_piece_agent_node = ChessPieceAgent()

    rclpy.spin(chess_piece_agent_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    chess_piece_agent_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()