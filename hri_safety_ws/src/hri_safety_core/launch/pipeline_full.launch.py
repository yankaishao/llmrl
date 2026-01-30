from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    parser_mode = LaunchConfiguration("parser_mode")
    arbiter_mode = LaunchConfiguration("arbiter_mode")
    policy_path = LaunchConfiguration("policy_path")
    start_instruction_source = LaunchConfiguration("start_instruction_source")

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
            DeclareLaunchArgument("start_instruction_source", default_value="false"),
            Node(
                package="hri_safety_core",
                executable="instruction_listener",
                name="instruction_listener",
                output="screen",
            ),
            Node(
                package="hri_safety_core",
                executable="scene_summary_stub",
                name="scene_summary_stub",
                output="screen",
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
            Node(
                package="hri_safety_core",
                executable="instruction_source",
                name="instruction_source",
                output="screen",
                condition=IfCondition(start_instruction_source),
            ),
        ]
    )
