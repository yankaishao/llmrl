import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("hri_safety_core")
    default_world = os.path.join(pkg_share, "worlds", "tabletop_level1.sdf")

    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("ign_cmd", default_value="ign"),
            DeclareLaunchArgument("verbose", default_value="1"),
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("ign_cmd"),
                    "gazebo",
                    "-r",
                    "-v",
                    LaunchConfiguration("verbose"),
                    LaunchConfiguration("world"),
                ],
                output="screen",
            ),
        ]
    )
