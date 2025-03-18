import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

from launch.substitutions import TextSubstitution
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('matchessbot'),
        'config',
        'chess_piece_agents_params.yaml'
    )

    piece_color_launch_arg = DeclareLaunchArgument(
        'piece_color', default_value=TextSubstitution(text='white')
    )
    piece_type_launch_arg = DeclareLaunchArgument(
        'piece_type', default_value=TextSubstitution(text='king')
    )

    start_file_idx_launch_arg = DeclareLaunchArgument(
        'start_file_idx', default_value=TextSubstitution(text='4')
    )

    ld = LaunchDescription()
    ld.add_action(piece_color_launch_arg)
    ld.add_action(piece_type_launch_arg)
    ld.add_action(start_file_idx_launch_arg)

    node = Node(
            package='matchessbot',
            # namespace='chess_piece_agent',
            executable='chess_piece_agent',
            name='white_king',
            parameters = [config],
        )
    ld.add_action(node)
    
    return ld