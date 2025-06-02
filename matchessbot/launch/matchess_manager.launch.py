import os

import time

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node

from launch.substitutions import TextSubstitution
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    arguments = []
    DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    if DEBUG_ENV:
        arguments = ['--ros-args', '--log-level', ['matchess_manager', ':=', 'DEBUG']]

    # node = Node(
    #     package='matchessbot',
    #     executable='matchess_manager',
    #     name= 'matchess_manager',
    #     arguments=arguments
    # )
    
    ld = LaunchDescription()

    puzzle_mode_launch_arg = DeclareLaunchArgument(
        'puzzle_mode', default_value=TextSubstitution(text='True')
    )
    ld.add_action(puzzle_mode_launch_arg)

    test_data_dir_launch_arg = DeclareLaunchArgument(
        'test_save_dir', default_value=TextSubstitution(text='test_data/puzzles/XXX_time/')
    )
    ld.add_action(test_data_dir_launch_arg)


    GAME_ID_ENV = os.environ.get('GAME_ID', 'unknown')
    default_save_dir = 'test_data/games/'
    game_test_timestamp = str(int(time.time()))
    test_save_dir_param = default_save_dir + GAME_ID_ENV + '_' + game_test_timestamp + '/'

    manager = Node(
        package='matchessbot',
        executable='matchess_manager',
        name= 'matchess_manager',
        parameters=[
            {'puzzle_mode': str(False)},
            {'test_save_dir': str(test_save_dir_param)}
            ],
        arguments=arguments
    )
    ld.add_action(manager)

    return  ld # LaunchDescription([node])