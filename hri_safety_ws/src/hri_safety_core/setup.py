from glob import glob

from setuptools import setup

package_name = "hri_safety_core"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/gazebo_only.launch.py",
                "launch/pipeline_mock.launch.py",
                "launch/pipeline_router.launch.py",
                "launch/pipeline_full.launch.py",
                "launch/pipeline_gazebo.launch.py",
            ],
        ),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf") + glob("worlds/*.world")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="TODO",
    maintainer_email="TODO",
    description="Minimal ROS2 nodes for instruction pipeline.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "instruction_source = hri_safety_core.instruction_source:main",
            "instruction_listener = hri_safety_core.instruction_listener:main",
            "scene_summary_stub = hri_safety_core.scene_summary_stub:main",
            "mock_llm_parser = hri_safety_core.mock_llm_parser:main",
            "qwen_api_parser = hri_safety_core.qwen_api_parser:main",
            "parser_router = hri_safety_core.parser_router:main",
            "estimator_node = hri_safety_core.estimator_node:main",
            "rule_based_arbiter = hri_safety_core.rule_based_arbiter:main",
            "rl_arbiter_node = hri_safety_core.rl_arbiter_node:main",
            "arbiter_router = hri_safety_core.arbiter_router:main",
            "gazebo_world_state_node = hri_safety_core.gazebo_world_state_node:main",
            "skill_executor_gazebo = hri_safety_core.skill_executor_gazebo:main",
            "action_to_skill_bridge = hri_safety_core.action_to_skill_bridge:main",
            "episode_manager_node = hri_safety_core.episode_manager_node:main",
        ],
    },
)
