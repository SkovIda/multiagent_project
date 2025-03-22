import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

from launch.substitutions import TextSubstitution
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    arguments = []
    DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    if DEBUG_ENV:
        arguments = ['--ros-args', '--log-level', ['matchessbot', ':=', 'DEBUG']]

    piece_color_launch_arg = DeclareLaunchArgument(
        'piece_color', default_value=TextSubstitution(text='white')
    )
    engine_path_launch_arg = DeclareLaunchArgument(
        'engine_path', default_value=TextSubstitution(text="/usr/games/stockfish")
    )


    ld = LaunchDescription()
    ld.add_action(piece_color_launch_arg)
    ld.add_action(engine_path_launch_arg)

    config_filename = ''
    COLOUR = os.environ.get('SINGLE_AGENT_COLOR', 'WHITE').lower()
    if COLOUR == 'white':
        # pieces = white_peices
        config_filename = 'white_single_agent_params.yaml'
    elif COLOUR == 'black':
        # pieces = black_pieces
        config_filename = 'black_single_agent_params.yaml'
    else:
        print(f'Unknown color set [{COLOUR}]')
        exit(1)

    # Load config file for chess piece agents:
    config = os.path.join(
        get_package_share_directory('matchessbot'),
        'config',
        config_filename
    )

    node = Node(
            package='matchessbot',
            # namespace='chess_piece_agent',
            executable='player_stockfish',
            name='player_stockfish',
            parameters = [config],
            arguments=arguments
        )
    ld.add_action(node)
    
    return ld