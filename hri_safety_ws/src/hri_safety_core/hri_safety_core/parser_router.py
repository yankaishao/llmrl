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
from hri_safety_core.qwen_structured_parser import (
    DEFAULT_CACHE_SIZE as DEFAULT_STRUCTURED_CACHE_SIZE,
    DEFAULT_FALLBACK_MODE as DEFAULT_STRUCTURED_FALLBACK_MODE,
    DEFAULT_MAX_RETRIES as DEFAULT_STRUCTURED_MAX_RETRIES,
    DEFAULT_MAX_TOKENS as DEFAULT_STRUCTURED_MAX_TOKENS,
    DEFAULT_STRUCTURED_MODEL,
    DEFAULT_TEMPERATURE as DEFAULT_STRUCTURED_TEMPERATURE,
    DEFAULT_TIMEOUT_SEC as DEFAULT_STRUCTURED_TIMEOUT_SEC,
    QwenStructuredClient,
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
        self.declare_parameter("context_topic", "/dialogue/context_instruction")
        self.declare_parameter("scene_state_topic", "/scene/state_json")
        self.declare_parameter("structured_model", DEFAULT_STRUCTURED_MODEL)
        self.declare_parameter("structured_timeout_sec", DEFAULT_STRUCTURED_TIMEOUT_SEC)
        self.declare_parameter("structured_max_retries", DEFAULT_STRUCTURED_MAX_RETRIES)
        self.declare_parameter("structured_temperature", DEFAULT_STRUCTURED_TEMPERATURE)
        self.declare_parameter("structured_max_tokens", DEFAULT_STRUCTURED_MAX_TOKENS)
        self.declare_parameter("structured_cache_size", DEFAULT_STRUCTURED_CACHE_SIZE)
        self.declare_parameter("structured_fallback_mode", DEFAULT_STRUCTURED_FALLBACK_MODE)

        self.mode = str(self.get_parameter("parser_mode").value).lower()
        self.last_summary = DEFAULT_SCENE_SUMMARY
        self.last_state_json = ""
        self.last_context = ""

        self.qwen_client = None
        self.structured_client = None
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
        elif self.mode == "qwen_structured":
            self.structured_client = QwenStructuredClient(
                model=str(self.get_parameter("structured_model").value),
                base_url=str(self.get_parameter("base_url").value),
                api_key_env=str(self.get_parameter("api_key_env").value),
                timeout_sec=float(self.get_parameter("structured_timeout_sec").value),
                max_retries=int(self.get_parameter("structured_max_retries").value),
                temperature=float(self.get_parameter("structured_temperature").value),
                max_tokens=int(self.get_parameter("structured_max_tokens").value),
                cache_size=int(self.get_parameter("structured_cache_size").value),
                fallback_mode=str(self.get_parameter("structured_fallback_mode").value),
            )
        else:
            self.mode = "mock"

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.context_topic = str(self.get_parameter("context_topic").value)
        self.scene_state_topic = str(self.get_parameter("scene_state_topic").value)

        self.publisher_ = self.create_publisher(String, "/nl/parse_result", 10)
        self.create_subscription(String, "/scene/summary", self.on_summary, 10)
        self.create_subscription(String, self.scene_state_topic, self.on_state, 10)
        self.create_subscription(String, self.context_topic, self.on_context, 10)
        self.create_subscription(String, self.input_topic, self.on_instruction, 10)
        self.get_logger().info(f"parser_router started mode={self.mode} input_topic={self.input_topic}.")

    def on_summary(self, msg: String) -> None:
        self.last_summary = msg.data

    def on_state(self, msg: String) -> None:
        self.last_state_json = msg.data

    def on_context(self, msg: String) -> None:
        self.last_context = msg.data

    def _extract_last_user_text(self, context: str) -> str:
        lines = [line for line in context.splitlines() if line.strip()]
        for line in reversed(lines):
            lowered = line.lower()
            if lowered.startswith("user:"):
                return line.split(":", 1)[1].strip()
        return context.strip()

    def on_instruction(self, msg: String) -> None:
        instruction_raw = msg.data.strip()
        summary = self.last_summary
        state_json = self.last_state_json

        if self.mode == "qwen_structured" and self.structured_client is not None:
            context = self.last_context
            if self.input_topic == self.context_topic:
                context = instruction_raw
                instruction_text = self._extract_last_user_text(context)
            else:
                instruction_text = instruction_raw
            result, meta = self.structured_client.parse(
                instruction_text, summary, scene_state_json=state_json, conversation_context=context
            )
            elapsed = meta.get("elapsed_sec", 0.0)
            fallback = meta.get("fallback", False)
            cache_hit = meta.get("cache_hit", False)
            reason = meta.get("reason", "")
            self.get_logger().info(
                f"qwen_structured parse elapsed={elapsed:.2f}s cache_hit={cache_hit} "
                f"fallback={fallback} reason={reason}"
            )
        elif self.mode == "qwen" and self.qwen_client is not None:
            instruction = instruction_raw.lower()
            result, meta = self.qwen_client.parse(instruction, summary)
            elapsed = meta.get("elapsed_sec", 0.0)
            fallback = meta.get("fallback", False)
            cache_hit = meta.get("cache_hit", False)
            reason = meta.get("reason", "")
            self.get_logger().info(
                f"qwen parse elapsed={elapsed:.2f}s cache_hit={cache_hit} fallback={fallback} reason={reason}"
            )
        else:
            instruction = instruction_raw.lower()
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
