import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

from launch.substitutions import TextSubstitution
from launch.substitutions import LaunchConfiguration

import argparse

from launch import LaunchDescriptionEntity

from launch.actions import DeclareLaunchArgument, SetLaunchConfiguration



from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    # config = os.path.join(
    #     get_package_share_directory('matchessbot'),
    #     'config',
    #     'chess_piece_agents_params.yaml'
    # )
    arguments = []
    DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    if DEBUG_ENV:
        arguments = ['--ros-args', '--log-level', ['matchessbot', ':=', 'DEBUG']]

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

    pieces = []
    pieces = ['king', 'queen', 'rook0', 'rook7', 'knight1', 'knight6', 'bishop2', 'bishop5',
                                  'pawn0', 'pawn1', 'pawn2', 'pawn3', 'pawn4', 'pawn5', 'pawn6', 'pawn7']
    
    config_filename = ''

    COLOUR = os.environ.get('MATCHESS_TEAM_COLOR', 'WHITE').lower()
    if COLOUR == 'white':
        # pieces = white_peices
        team_namespace = 'white'
        config_filename = 'white_pieces_params.yaml'
    elif COLOUR == 'black':
        # pieces = black_pieces
        team_namespace = 'black'
        config_filename = 'black_pieces_params.yaml'
    else:
        print(f'Unknown color set [{COLOUR}]')
        exit(1)

    # Load config file for chess piece agents:
    config = os.path.join(
        get_package_share_directory('matchessbot'),
        'config',
        config_filename
    )

    for agent_name in pieces:
        # node = Node(
        #         package='matchessbot',
        #         #namespace='white',
        #         executable='chess_piece_agent',
        #         # name=agent_name,
        #         name= agent_name,
        #         parameters = [config],
        #         arguments=arguments
        #     )
        node = Node(
                package='matchessbot',
                namespace=team_namespace,
                executable='chess_piece_agent',
                name= agent_name,
                parameters = [config],
                arguments=arguments,
                remappings=[
                    ('/' + team_namespace + '/matchess/in', '/matchess/in'),
                    ('/' + team_namespace + '/matchess/out', '/matchess/out'),
                    ('/' + team_namespace + '/matchess/game_status', '/matchess/game_status'),
                    ('/' + team_namespace + '/matchess/game_status_cmd', '/matchess/game_status_cmd'),
                    ('/' + team_namespace + '/matchess/game_hist' ,'/matchess/game_hist'),
                ]
            )
        ld.add_action(node)

    return ld
