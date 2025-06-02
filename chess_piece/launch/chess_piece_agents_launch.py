# from launch import LaunchDescription
# from launch_ros.actions import Node

# import os

# from ament_index_python import get_package_share_directory

# from launch import LaunchDescription
# from launch.actions import DeclareLaunchArgument
# from launch.actions import IncludeLaunchDescription
# from launch.actions import GroupAction
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
# from launch_ros.actions import Node
# from launch_ros.actions import PushRosNamespace


# from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

# from launch_ros.actions import Node
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('chess_piece'),
        'config',
        'chess_piece_agents_params.yaml'
    )

    piece_color_launch_arg = DeclareLaunchArgument(
        'piece_color', default_value=TextSubstitution(text='white')
    )
    piece_type_launch_arg = DeclareLaunchArgument(
        'piece_type', default_value=TextSubstitution(text='king')
    )
    pos_square_idx_launch_arg = DeclareLaunchArgument(
        'pos_square_idx', default_value=TextSubstitution(text='4')
    )

    ld = LaunchDescription()
    ld.add_action(piece_color_launch_arg)
    ld.add_action(piece_type_launch_arg)
    ld.add_action(pos_square_idx_launch_arg)

    # pieces_on_team = {'king': 1, 'queen': 1, 'rook': 2, 'knight': 2, 'bishop': 2, 'pawn': 8}

    node = Node(
            package='chess_piece',
            # namespace='chess_piece_agent',
            executable='chess_piece',
            name='white_king',
            parameters = [config],
        )
    ld.add_action(node)
    
    node = Node(
            package='chess_piece',
            # namespace='chess_piece_agent',
            executable='chess_piece',
            name='white_queen',
            parameters = [config],
        )
    
    ld.add_action(node)

    return ld



###############################################################
##### WORKS:
# def generate_launch_description():
#     config = os.path.join(
#         get_package_share_directory('chess_piece'),
#         'config',
#         'chess_piece_agents_params.yaml'
#     )

#     piece_color_launch_arg = DeclareLaunchArgument(
#         'piece_color', default_value=TextSubstitution(text='black')
#     )
#     piece_type_launch_arg = DeclareLaunchArgument(
#         'piece_type', default_value=TextSubstitution(text='pawn')
#     )
#     pos_square_idx_launch_arg = DeclareLaunchArgument(
#         'pos_square_idx', default_value=TextSubstitution(text='8')
#     )

#     return LaunchDescription([
#         piece_color_launch_arg,
#         piece_type_launch_arg,
#         pos_square_idx_launch_arg,
#         Node(
#             package='chess_piece',
#             # namespace='chess_piece_agent',
#             executable='chess_piece',
#             name='chess_piece_agent',
#             parameters = [config],
#         ),
#         Node(
#             package='chess_piece',
#             # namespace='chess_piece_agent',
#             executable='chess_piece',
#             name='chess_piece_agent',
#             parameters = [config],
#         ),
#     ])
#############################################################



#     # args that can be set from the command line or a default will be used
#     piece_color_launch_arg = DeclareLaunchArgument(
#         "piece_color", default_value=TextSubstitution(text="1") # color.white=1 in python-chess
#     )

#    return LaunchDescription([
#       piece_color_launch_arg,
#    ])
#    return LaunchDescription([
#       DeclareLaunchArgument(
#          'piece_color', default_value='turtle1',
#          description='Target frame name.'
#       ),
#       Node(
#          package='turtle_tf2_py',
#          executable='turtle_tf2_broadcaster',
#          name='broadcaster1',
#          parameters=[
#             {'turtlename': 'turtle1'}
#          ]
#       ),
#       Node(
#          package='turtle_tf2_py',
#          executable='turtle_tf2_broadcaster',
#          name='broadcaster2',
#          parameters=[
#             {'turtlename': 'turtle2'}
#          ]
#       ),
#       Node(
#          package='turtle_tf2_py',
#          executable='turtle_tf2_listener',
#          name='listener',
#          parameters=[
#             {'target_frame': LaunchConfiguration('target_frame')}
#          ]
#       ),
#    ])
    
    # # args that can be set from the command line or a default will be used
    # piece_color_launch_arg = DeclareLaunchArgument(
    #     "piece_color", default_value=TextSubstitution(text="1") # color.white=1 in python-chess
    # )
    # piece_type_launch_arg = DeclareLaunchArgument(
    #     "piece_color", default_value=TextSubstitution(text="0")
    # )


    # return LaunchDescription([
        # Node(
        #     package='chess_piece',
        #     namespace='chess_piece1',
        #     executable='chess_piece_agent',
        #     name='agent_sim',
        #     arguments=['-_piece_color', 'robot_description',
        #        '-entity', 'ur5',
        #        '-z', '0.1']
        # ),
    #     # Node(
    #     #     package='chess_piece',
    #     #     namespace='chess_piece1',
    #     #     executable='chess_piece_agent',
    #     #     name='sim'
    #     # ),
    #     # Node(
    #     #     package='turtlesim',
    #     #     executable='mimic',
    #     #     name='mimic',
    #     #     remappings=[
    #     #         ('/input/pose', '/turtlesim1/turtle1/pose'),
    #     #         ('/output/cmd_vel', '/turtlesim2/turtle1/cmd_vel'),
    #     #     ]
    #     # )
    # ])