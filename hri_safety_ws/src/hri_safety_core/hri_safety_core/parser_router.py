import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hri_safety_core.parser_utils import DEFAULT_SCENE_SUMMARY, mock_parse
from hri_safety_core.qwen_api_parser import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_CACHE_SIZE,
    DEFAULT_FALLBACK_MODE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SEC,
    QwenApiClient,
)


class ParserRouter(Node):
    def __init__(self) -> None:
        super().__init__("parser_router")
        self.declare_parameter("parser_mode", "mock")
        self.declare_parameter("model", DEFAULT_MODEL)
        self.declare_parameter("base_url", DEFAULT_BASE_URL)
        self.declare_parameter("api_key_env", DEFAULT_API_KEY_ENV)
        self.declare_parameter("timeout_sec", DEFAULT_TIMEOUT_SEC)
        self.declare_parameter("max_retries", DEFAULT_MAX_RETRIES)
        self.declare_parameter("temperature", DEFAULT_TEMPERATURE)
        self.declare_parameter("max_tokens", DEFAULT_MAX_TOKENS)
        self.declare_parameter("cache_size", DEFAULT_CACHE_SIZE)
        self.declare_parameter("fallback_mode", DEFAULT_FALLBACK_MODE)
        self.declare_parameter("input_topic", "/user/instruction")

        self.mode = str(self.get_parameter("parser_mode").value).lower()
        self.last_summary = DEFAULT_SCENE_SUMMARY

        self.qwen_client = None
        if self.mode == "qwen":
            self.qwen_client = QwenApiClient(
                model=str(self.get_parameter("model").value),
                base_url=str(self.get_parameter("base_url").value),
                api_key_env=str(self.get_parameter("api_key_env").value),
                timeout_sec=float(self.get_parameter("timeout_sec").value),
                max_retries=int(self.get_parameter("max_retries").value),
                temperature=float(self.get_parameter("temperature").value),
                max_tokens=int(self.get_parameter("max_tokens").value),
                cache_size=int(self.get_parameter("cache_size").value),
                fallback_mode=str(self.get_parameter("fallback_mode").value),
            )
        else:
            self.mode = "mock"

        self.input_topic = str(self.get_parameter("input_topic").value)

        self.publisher_ = self.create_publisher(String, "/nl/parse_result", 10)
        self.create_subscription(String, "/scene/summary", self.on_summary, 10)
        self.create_subscription(String, self.input_topic, self.on_instruction, 10)
        self.get_logger().info(f"parser_router started mode={self.mode} input_topic={self.input_topic}.")

    def on_summary(self, msg: String) -> None:
        self.last_summary = msg.data

    def on_instruction(self, msg: String) -> None:
        instruction = msg.data.strip().lower()
        summary = self.last_summary

        if self.mode == "qwen" and self.qwen_client is not None:
            result, meta = self.qwen_client.parse(instruction, summary)
            elapsed = meta.get("elapsed_sec", 0.0)
            fallback = meta.get("fallback", False)
            cache_hit = meta.get("cache_hit", False)
            reason = meta.get("reason", "")
            self.get_logger().info(
                f"qwen parse elapsed={elapsed:.2f}s cache_hit={cache_hit} fallback={fallback} reason={reason}"
            )
        else:
            result = mock_parse(instruction, summary)

        out = String()
        out.data = json.dumps(result, ensure_ascii=True)
        self.publisher_.publish(out)
        self.get_logger().info("published /nl/parse_result")


def main() -> None:
    rclpy.init()
    node = ParserRouter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
