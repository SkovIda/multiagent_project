# # matchess/launch/matchess_game_sim.launch.py

# # import os

# # from ament_index_python import get_package_share_directory

# # from launch import LaunchDescription
# # # import launch_ros.actions

# # from launch.actions import DeclareLaunchArgument

# # from launch.launch_description_sources import PythonLaunchDescriptionSource
# # from launch.substitutions import LaunchConfiguration

# # from launch.substitutions import TextSubstitution

# # from launch_ros.actions import Node

# import os

# from ament_index_python import get_package_share_directory

# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument
# from launch.actions import IncludeLaunchDescription
# from launch.actions import GroupAction
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
# from launch.substitutions import TextSubstitution
# from launch_ros.actions import Node
# from launch_ros.actions import PushRosNamespace


# def generate_launch_description():
#     # args that can be set from the command line or a default will be used
#     game_start_fen_launch_arg = DeclareLaunchArgument(
#         "game_start_fen", default_value=TextSubstitution(text="'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'")
#     )
    
#     # start a turtlesim_node in the turtlesim1 namespace
#     matchess_env_node = Node(
#             package='matchess',
#             namespace='matchess1',
#             executable='matchess_game',
#             name='matchess_env',
#             parameters=[{
#                 "game_start_fen": LaunchConfiguration('game_start_fen'),
#             }]
#         )
    
#     return LaunchDescription([
#         game_start_fen_launch_arg,
#         matchess_env_node,
#     ])
