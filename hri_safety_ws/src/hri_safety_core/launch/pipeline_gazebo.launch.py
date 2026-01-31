import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("hri_safety_core")
    default_world = os.path.join(pkg_share, "worlds", "tabletop.sdf")

    parser_mode = LaunchConfiguration("parser_mode")
    arbiter_mode = LaunchConfiguration("arbiter_mode")
    policy_path = LaunchConfiguration("policy_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument("parser_mode", default_value="mock"),
            DeclareLaunchArgument("arbiter_mode", default_value="rule"),
            DeclareLaunchArgument("policy_path", default_value="policies/ppo_policy.zip"),
            DeclareLaunchArgument("model", default_value="qwen3-max"),
            DeclareLaunchArgument(
                "base_url", default_value="https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            DeclareLaunchArgument("api_key_env", default_value="QWEN_API_KEY"),
            DeclareLaunchArgument("amb_high", default_value="0.4"),
            DeclareLaunchArgument("risk_high", default_value="0.7"),
            DeclareLaunchArgument("deterministic", default_value="true"),
            DeclareLaunchArgument("world", default_value=default_world),
            DeclareLaunchArgument("ign_cmd", default_value="ign"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("ign_topic", default_value="/world/default/pose/info"),
            DeclareLaunchArgument("publish_hz", default_value="2.0"),
            DeclareLaunchArgument("command_timeout", default_value="2.0"),
            DeclareLaunchArgument("max_summary_chars", default_value="1000"),
            DeclareLaunchArgument(
                "summary_context", default_value="context=[child_present=false], task_state=idle"
            ),
            DeclareLaunchArgument(
                "object_names", default_value="cup_red_1,cup_red_2,knife_1"
            ),
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("ign_cmd"),
                    "gazebo",
                    "-r",
                    "-v",
                    "1",
                    LaunchConfiguration("world"),
                ],
                condition=UnlessCondition(LaunchConfiguration("headless")),
                output="screen",
            ),
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("ign_cmd"),
                    "gazebo",
                    "-r",
                    "-v",
                    "1",
                    "-s",
                    LaunchConfiguration("world"),
                ],
                condition=IfCondition(LaunchConfiguration("headless")),
                output="screen",
            ),
            Node(
                package="hri_safety_core",
                executable="gazebo_world_state_node",
                name="gazebo_world_state_node",
                output="screen",
                parameters=[
                    {
                        "ign_cmd": LaunchConfiguration("ign_cmd"),
                        "ign_topic": LaunchConfiguration("ign_topic"),
                        "publish_hz": LaunchConfiguration("publish_hz"),
                        "command_timeout": LaunchConfiguration("command_timeout"),
                        "summary_context": LaunchConfiguration("summary_context"),
                        "max_summary_chars": LaunchConfiguration("max_summary_chars"),
                        "object_names": LaunchConfiguration("object_names"),
                    }
                ],
            ),
            Node(
                package="hri_safety_core",
                executable="parser_router",
                name="parser_router",
                output="screen",
                parameters=[
                    {
                        "parser_mode": parser_mode,
                        "model": LaunchConfiguration("model"),
                        "base_url": LaunchConfiguration("base_url"),
                        "api_key_env": LaunchConfiguration("api_key_env"),
                    }
                ],
            ),
            Node(
                package="hri_safety_core",
                executable="estimator_node",
                name="estimator_node",
                output="screen",
            ),
            Node(
                package="hri_safety_core",
                executable="arbiter_router",
                name="arbiter_router",
                output="screen",
                parameters=[
                    {
                        "arbiter_mode": arbiter_mode,
                        "policy_path": policy_path,
                        "amb_high": LaunchConfiguration("amb_high"),
                        "risk_high": LaunchConfiguration("risk_high"),
                        "deterministic": LaunchConfiguration("deterministic"),
                    }
                ],
            ),
        ]
    )
