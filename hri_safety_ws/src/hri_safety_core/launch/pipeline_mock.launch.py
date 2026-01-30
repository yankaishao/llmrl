from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
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
                executable="mock_llm_parser",
                name="mock_llm_parser",
                output="screen",
            ),
            Node(
                package="hri_safety_core",
                executable="estimator_node",
                name="estimator_node",
                output="screen",
            ),
            Node(
                package="hri_safety_core",
                executable="rule_based_arbiter",
                name="rule_based_arbiter",
                output="screen",
            ),
        ]
    )
