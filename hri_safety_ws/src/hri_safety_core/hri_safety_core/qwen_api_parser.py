import json
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hri_safety_core.parser_utils import (
    CLARIFY_TEMPLATES,
    DEFAULT_SCENE_SUMMARY,
    mock_parse,
    validate_parse_result,
)

DEFAULT_MODEL = "qwen3-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "QWEN_API_KEY"
DEFAULT_TIMEOUT_SEC = 10.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 256
DEFAULT_CACHE_SIZE = 128
DEFAULT_FALLBACK_MODE = "mock"

SYSTEM_PROMPT = (
    "You are a strict parser. Output ONLY a JSON object and nothing else. "
    "No markdown, no code fences, no extra text. "
    "Required keys: task_type, candidates, constraints, clarify_templates, notes. "
    "task_type must be one of: fetch, handover, place, unknown. "
    "candidates must be a list with at least 2 items, each item has id and score. "
    "Scores must sum to 1. "
    "constraints must include risk_hints (list) and fragile (bool). "
    "clarify_templates must match the provided templates exactly."
)


def _build_user_prompt(instruction: str, summary: str) -> str:
    templates_json = json.dumps(CLARIFY_TEMPLATES, ensure_ascii=True)
    schema = (
        "{"
        '"task_type":"fetch|handover|place|unknown",'
        '"candidates":[{"id":"...","score":0.5},{"id":"...","score":0.5}],'
        '"constraints":{"risk_hints":[],"fragile":false},'
        '"clarify_templates":' + templates_json + ","
        '"notes":"..."'
        "}"
    )
    return (
        "Instruction: " + instruction + "\n"
        "Scene summary: " + summary + "\n"
        "Return JSON only. Use candidate ids from the scene summary when possible; "
        "if unclear, use unknown_1 and unknown_2. "
        "Use these clarify_templates exactly: " + templates_json + "\n"
        "Schema: " + schema
    )


def _find_env_file(start: Path) -> Optional[Path]:
    for parent in [start, *start.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_env_file(env_path: Path) -> None:
    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_api_key(api_key_env: str) -> str:
    value = os.getenv(api_key_env, "")
    if not value and api_key_env != "OPENAI_API_KEY":
        value = os.getenv("OPENAI_API_KEY", "")
    if not value:
        return ""
    if os.path.isfile(value):
        try:
            return Path(value).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return value


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no_json_object")
    return text[start : end + 1]



class LruCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.data: OrderedDict[Tuple[str, str, str], Dict[str, object]] = OrderedDict()

    def get(self, key: Tuple[str, str, str]) -> Optional[Dict[str, object]]:
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def set(self, key: Tuple[str, str, str], value: Dict[str, object]) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.max_size:
            self.data.popitem(last=False)


class QwenApiClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str,
        timeout_sec: float,
        max_retries: int,
        temperature: float,
        max_tokens: int,
        cache_size: int,
        fallback_mode: str,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = LruCache(cache_size)
        self.fallback_mode = fallback_mode

        env_path = _find_env_file(Path(__file__).resolve())
        if env_path:
            _load_env_file(env_path)

    def parse(self, instruction: str, summary: str) -> Tuple[Dict[str, object], Dict[str, object]]:
        cache_key = (instruction, summary, self.model)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached, {"cache_hit": True, "fallback": False, "reason": ""}

        start = time.monotonic()
        try:
            result = self._query_model(instruction, summary)
            meta = {"cache_hit": False, "fallback": False, "reason": ""}
        except Exception as exc:
            reason = str(exc)
            result = self._fallback_parse(instruction, summary, reason)
            meta = {"cache_hit": False, "fallback": True, "reason": reason}

        meta["elapsed_sec"] = time.monotonic() - start
        self.cache.set(cache_key, result)
        return result, meta

    def _fallback_parse(self, instruction: str, summary: str, reason: str) -> Dict[str, object]:
        if self.fallback_mode == "clarify":
            return {
                "task_type": "unknown",
                "candidates": [
                    {"id": "unknown_1", "score": 0.5},
                    {"id": "unknown_2", "score": 0.5},
                ],
                "constraints": {"risk_hints": [], "fragile": False},
                "clarify_templates": CLARIFY_TEMPLATES,
                "notes": f"Fallback: {reason}",
            }
        fallback = mock_parse(instruction, summary)
        fallback["notes"] = f"Fallback: {reason}"
        return fallback

    def _query_model(self, instruction: str, summary: str) -> Dict[str, object]:
        api_key = _resolve_api_key(self.api_key_env)
        if not api_key:
            raise ValueError("missing_api_key")

        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(instruction, summary)},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        last_error: Optional[Exception] = None
        for _ in range(self.max_retries + 1):
            try:
                request = urllib.request.Request(url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    response_text = response.read().decode("utf-8")
                response_json = json.loads(response_text)
                content = self._extract_content(response_json)
                parsed_text = _extract_json(content)
                parsed = json.loads(parsed_text)
                return validate_parse_result(parsed)
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                continue

        if last_error:
            raise last_error
        raise ValueError("qwen_api_failure")

    def _extract_content(self, response_json: Dict[str, object]) -> str:
        choices = response_json.get("choices", []) if isinstance(response_json, dict) else []
        if not choices:
            raise ValueError("empty_choices")
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str) and content:
                    return content
            text = first.get("text", "")
            if isinstance(text, str) and text:
                return text
        raise ValueError("missing_content")


class QwenApiParser(Node):
    def __init__(self) -> None:
        super().__init__("qwen_api_parser")
        self.declare_parameter("model", DEFAULT_MODEL)
        self.declare_parameter("base_url", DEFAULT_BASE_URL)
        self.declare_parameter("api_key_env", DEFAULT_API_KEY_ENV)
        self.declare_parameter("timeout_sec", DEFAULT_TIMEOUT_SEC)
        self.declare_parameter("max_retries", DEFAULT_MAX_RETRIES)
        self.declare_parameter("temperature", DEFAULT_TEMPERATURE)
        self.declare_parameter("max_tokens", DEFAULT_MAX_TOKENS)
        self.declare_parameter("cache_size", DEFAULT_CACHE_SIZE)
        self.declare_parameter("fallback_mode", DEFAULT_FALLBACK_MODE)

        model = str(self.get_parameter("model").value)
        base_url = str(self.get_parameter("base_url").value)
        api_key_env = str(self.get_parameter("api_key_env").value)
        timeout_sec = float(self.get_parameter("timeout_sec").value)
        max_retries = int(self.get_parameter("max_retries").value)
        temperature = float(self.get_parameter("temperature").value)
        max_tokens = int(self.get_parameter("max_tokens").value)
        cache_size = int(self.get_parameter("cache_size").value)
        fallback_mode = str(self.get_parameter("fallback_mode").value)

        self.client = QwenApiClient(
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            temperature=temperature,
            max_tokens=max_tokens,
            cache_size=cache_size,
            fallback_mode=fallback_mode,
        )

        self.publisher_ = self.create_publisher(String, "/nl/parse_result", 10)
        self.last_summary = DEFAULT_SCENE_SUMMARY
        self.create_subscription(String, "/scene/summary", self.on_summary, 10)
        self.create_subscription(String, "/user/instruction", self.on_instruction, 10)
        self.get_logger().info("qwen_api_parser started.")

    def on_summary(self, msg: String) -> None:
        self.last_summary = msg.data

    def on_instruction(self, msg: String) -> None:
        instruction = msg.data.strip().lower()
        summary = self.last_summary
        result, meta = self.client.parse(instruction, summary)

        out = String()
        out.data = json.dumps(result, ensure_ascii=True)
        self.publisher_.publish(out)

        elapsed = meta.get("elapsed_sec", 0.0)
        fallback = meta.get("fallback", False)
        cache_hit = meta.get("cache_hit", False)
        reason = meta.get("reason", "")
        self.get_logger().info(
            f"published /nl/parse_result elapsed={elapsed:.2f}s cache_hit={cache_hit} "
            f"fallback={fallback} reason={reason}"
        )


def main() -> None:
    rclpy.init()
    node = QwenApiParser()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
