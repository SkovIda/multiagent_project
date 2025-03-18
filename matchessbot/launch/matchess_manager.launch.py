import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    os.environ['RMW_IMPLEMENTATION'] = 'rmw_cyclonedds_cpp' # fast dds loses messages in reliable mode
    arguments = []
    DEBUG_ENV = os.environ.get('MATCHESS_DEBUG', 'FALSE').lower() in ('true', '1', 't')
    if DEBUG_ENV:
        arguments = ['--ros-args', '--log-level', ['matchess_manager', ':=', 'DEBUG']]

    node = Node(
        package='matchessbot',
        executable='matchess_manager',
        name= 'matchess_manager',
        arguments=arguments
    )
    return LaunchDescription([node])