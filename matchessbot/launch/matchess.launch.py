import os
import time

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


    # launches = []
    chess_team = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('matchessbot'), 'chess_piece_agent_launch_all.launch.py'
            ])
        ])
    )
    # launches.append(chess_team)
    ld.add_action(chess_team)

    # manager = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource([
    #         PathJoinSubstitution([
    #             FindPackageShare('matchessbot'), 'matchess_manager.launch.py'
    #         ])
    #     ])
    # )
    # launches.append(manager)
    arguments = []
    DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    if DEBUG_ENV:
        arguments = ['--ros-args', '--log-level', ['matchess_manager', ':=', 'DEBUG']]


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


    # manager = Node(
    #     package='matchessbot',
    #     executable='matchess_manager',
    #     name= 'matchess_manager',
    #     arguments=arguments
    # )
    # ld.add_action(manager)

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

    return ld # LaunchDescription(launches)