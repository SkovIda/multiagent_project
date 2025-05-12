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


def generate_launch_description():
    os.environ['RMW_IMPLEMENTATION'] = os.environ.get('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    # fast dds loses messages in reliable mode

    launches = []
    chess_team = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('matchessbot'), 'player_matchess.launch.py'
            ])
        ])
    )
    launches.append(chess_team)


    stockfish = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('matchessbot'), 'player_stockfish.launch.py'
            ])
        ])
    )
    launches.append(stockfish)


    arguments = []
    DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    if DEBUG_ENV:
        arguments = ['--ros-args', '--log-level', ['matchess_manager', ':=', 'DEBUG']]

    manager = Node(
        package='matchessbot',
        executable='matchess_manager',
        name= 'matchess_manager',
        arguments=arguments
    )
    launches.append(manager)

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
    launches.append(manager_event_handler)

    return LaunchDescription(launches)