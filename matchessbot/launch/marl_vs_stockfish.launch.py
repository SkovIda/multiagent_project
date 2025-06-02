import os
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
import launch
import launch.actions
import launch.event_handlers
from launch_ros.actions import Node

from launch.substitutions import TextSubstitution
from launch.actions import DeclareLaunchArgument

from ament_index_python.packages import get_package_share_directory

import time

def generate_launch_description():
    os.environ['RMW_IMPLEMENTATION'] = os.environ.get('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    # fast dds loses messages in reliable mode

    ld = LaunchDescription()

    puzzle_mode_launch_arg = DeclareLaunchArgument(
        'puzzle_mode', default_value=TextSubstitution(text='True')
    )
    ld.add_action(puzzle_mode_launch_arg)

    test_data_dir_launch_arg = DeclareLaunchArgument(
        'test_save_dir', default_value=TextSubstitution(text='test_data/puzzles/XXX_time/')
    )
    ld.add_action(test_data_dir_launch_arg)

    launches = []
    chess_team = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('matchessbot'), 'player_matchess.launch.py'
            ])
        ])
    )
    ld.add_action(chess_team)


    stockfish = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('matchessbot'), 'player_stockfish.launch.py'
            ])
        ])
    )
    ld.add_action(stockfish)

    arguments = []
    DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    if DEBUG_ENV:
        arguments = ['--ros-args', '--log-level', ['matchess_manager', ':=', 'DEBUG']]

    

    PUZZLE_ID_ENV = os.environ.get('PUZZLE_ID', 'unknown')
    
    default_save_dir = 'test_data/puzzles/'

    puzzle_test_timestamp = str(int(time.time()))
    TEST_ID_ENV = os.environ.get('TEST_ID', puzzle_test_timestamp)

    # test_save_dir_param = default_save_dir + PUZZLE_ID_ENV + '_' + puzzle_test_timestamp + '/'
    test_save_dir_param = default_save_dir + 'test_' + TEST_ID_ENV + '/' + PUZZLE_ID_ENV + '/'

    manager = Node(
        package='matchessbot',
        executable='matchess_manager',
        name= 'matchess_manager',
        parameters=[
            {'puzzle_mode': str(True)},
            {'test_save_dir': str(test_save_dir_param)}
            ],
        arguments=arguments
    )
    ld.add_action(manager)

    #shut down if manager dies
    manager_event_handler = launch.actions.RegisterEventHandler(
        event_handler=launch.event_handlers.OnProcessExit(
            target_action=manager,
            on_exit=[
                launch.actions.LogInfo(
                    msg='Manager shut down: shutting down everything'
                ),
                launch.actions.EmitEvent(
                    event=launch.events.Shutdown()
                )
            ]
        )
    )
    ld.add_action(manager_event_handler)

    return ld


    # launches = []
    # chess_team = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare('matchessbot'), 'player_matchess.launch.py'
    #         ])
    #     ])
    # )
    # launches.append(chess_team)


    # stockfish = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare('matchessbot'), 'player_stockfish.launch.py'
    #         ])
    #     ])
    # )
    # launches.append(stockfish)

    # arguments = []
    # DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    # if DEBUG_ENV:
    #     arguments = ['--ros-args', '--log-level', ['matchess_manager', ':=', 'DEBUG']]

    # manager = Node(
    #     package='matchessbot',
    #     executable='matchess_manager',
    #     name= 'matchess_manager',
    #     arguments=arguments
    # )
    # launches.append(manager)

    # #shut down if manager dies
    # manager_event_handler = launch.actions.RegisterEventHandler(
    #     event_handler=launch.event_handlers.OnProcessExit(
    #         target_action=manager,
    #         on_exit=[
    #             launch.actions.LogInfo(
    #                 msg='Manager shut down: shutting down everything'
    #             ),
    #             launch.actions.EmitEvent(
    #                 event=launch.events.Shutdown()
    #             )
    #         ]
    #     )
    # )
    # launches.append(manager_event_handler)

    # # ld = LaunchDescription(launches)

    # # puzzle_mode_launch_arg = DeclareLaunchArgument(
    # #     'puzzle_mode', default_value=TextSubstitution(text='True')
    # # )
    # # ld.add_action(puzzle_mode_launch_arg)

    # # test_data_dir_launch_arg = DeclareLaunchArgument(
    # #     'test_save_dir', default_value=TextSubstitution(text='test_data/puzzles/XXX_time/')
    # # )
    # # ld.add_action(test_data_dir_launch_arg)

    # return LaunchDescription(launches)