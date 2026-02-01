from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    parser_mode = LaunchConfiguration("parser_mode")
    arbiter_mode = LaunchConfiguration("arbiter_mode")
    policy_path = LaunchConfiguration("policy_path")
    use_dialogue_manager = LaunchConfiguration("use_dialogue_manager")
    use_age_context = LaunchConfiguration("use_age_context")
    parser_input_topic = PythonExpression(
        ["'/dialogue/context_instruction' if ", use_dialogue_manager, " == 'true' else '/user/instruction'"]
    )

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
            DeclareLaunchArgument("max_turns", default_value="5"),
            DeclareLaunchArgument("max_repeat_action", default_value="2"),
            DeclareLaunchArgument("use_dialogue_manager", default_value="true"),
            DeclareLaunchArgument("obs_mode", default_value="legacy"),
            DeclareLaunchArgument("use_age_context", default_value="false"),
            DeclareLaunchArgument("age_context_mode", default_value="manual"),
            DeclareLaunchArgument("age_publish_hz", default_value="1.0"),
            DeclareLaunchArgument("age_seed", default_value="0"),
            DeclareLaunchArgument("age_p_minor", default_value="0.0"),
            DeclareLaunchArgument("age_p_adult", default_value="1.0"),
            DeclareLaunchArgument("age_p_older", default_value="0.0"),
            DeclareLaunchArgument("age_conf", default_value="1.0"),
            DeclareLaunchArgument("guardian_present", default_value="false"),
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
                        "input_topic": parser_input_topic,
                    }
                ],
            ),
            Node(
                package="hri_safety_core",
                executable="age_context_adapter",
                name="age_context_adapter",
                output="screen",
                condition=IfCondition(use_age_context),
                parameters=[
                    {
                        "mode": LaunchConfiguration("age_context_mode"),
                        "publish_hz": LaunchConfiguration("age_publish_hz"),
                        "seed": LaunchConfiguration("age_seed"),
                        "p_minor": LaunchConfiguration("age_p_minor"),
                        "p_adult": LaunchConfiguration("age_p_adult"),
                        "p_older": LaunchConfiguration("age_p_older"),
                        "age_conf": LaunchConfiguration("age_conf"),
                        "guardian_present": LaunchConfiguration("guardian_present"),
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
                executable="dialogue_manager",
                name="dialogue_manager",
                output="screen",
                condition=IfCondition(use_dialogue_manager),
                parameters=[
                    {
                        "policy_backend": arbiter_mode,
                        "policy_path": policy_path,
                        "amb_high": LaunchConfiguration("amb_high"),
                        "risk_high": LaunchConfiguration("risk_high"),
                        "deterministic": LaunchConfiguration("deterministic"),
                        "max_turns": LaunchConfiguration("max_turns"),
                        "max_repeat_action": LaunchConfiguration("max_repeat_action"),
                        "obs_mode": LaunchConfiguration("obs_mode"),
                        "context_topic": "/dialogue/context_instruction",
                        "use_age_context": use_age_context,
                    }
                ],
            ),
            Node(
                package="hri_safety_core",
                executable="arbiter_router",
                name="arbiter_router",
                output="screen",
                condition=UnlessCondition(use_dialogue_manager),
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
