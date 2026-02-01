import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hri_safety_core.parser_utils import DEFAULT_SCENE_SUMMARY, mock_parse


class MockLlmParser(Node):
    def __init__(self) -> None:
        super().__init__("mock_llm_parser")
        self.declare_parameter("input_topic", "/user/instruction")
        self.publisher_ = self.create_publisher(String, "/nl/parse_result", 10)
        self.last_summary = DEFAULT_SCENE_SUMMARY
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.create_subscription(String, "/scene/summary", self.on_summary, 10)
        self.create_subscription(String, self.input_topic, self.on_instruction, 10)
        self.get_logger().info(f"mock_llm_parser started input_topic={self.input_topic}.")

    def on_summary(self, msg: String) -> None:
        self.last_summary = msg.data

    def on_instruction(self, msg: String) -> None:
        instruction = msg.data.strip().lower()
        summary = self.last_summary
        parse_result = mock_parse(instruction, summary)
        out = String()
        out.data = json.dumps(parse_result, ensure_ascii=True)
        self.publisher_.publish(out)
        self.get_logger().info("published /nl/parse_result")


def main() -> None:
    rclpy.init()
    node = MockLlmParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
