from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

from hri_safety_core.parse_result_utils import (
    PARSE_RESULT_VERSION,
    build_mock_parse_result_v1,
    ensure_scene_consistency,
    generate_request_id,
    validate_parse_result,
)
from hri_safety_core.scene_utils import build_scene_payload

DEFAULT_STRUCTURED_MODEL = "qwen3-max"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_API_KEY_ENV = "QWEN_API_KEY"
DEFAULT_TIMEOUT_SEC = 12.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 640
DEFAULT_CACHE_SIZE = 128
DEFAULT_FALLBACK_MODE = "mock"
DEFAULT_REPAIR_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You are a structured parser for human-robot interaction. "
    "You must return a tool call named emit_parse_result or, if tools are unavailable, "
    "output a JSON object only. No markdown or extra text. "
    "Return 1-5 candidates with confidence (0-1). "
    "Always fill hazards, contradictions, missing_slots, and clarifying_questions. "
    "Do not fabricate object_id; object_id must come from scene.objects. "
    "If object_id is not in scene.objects, include a contradiction and lower confidence. "
    "Do not infer age or demographics unless explicitly stated by the user; "
    "put only self-report clues in meta.debug." 
)

INTENT_GUIDE = (
    "Intent guide: execute_skill, handover, move_object, point_to, ask_user, refuse, fallback." 
)


def _build_user_prompt(
    text: str,
    scene_summary: Optional[str],
    objects: list,
    conversation_context: Optional[str],
) -> str:
    context = conversation_context.strip() if isinstance(conversation_context, str) else ""
    if context:
        lines = [line for line in context.splitlines() if line.strip()]
        if len(lines) > 10:
            context = "\n".join(lines[-10:])
    objects_json = json.dumps(objects, ensure_ascii=True)
    summary_text = scene_summary or ""
    return (
        f"Instruction: {text}\n"
        f"Conversation context (last turns): {context}\n"
        f"Scene summary: {summary_text}\n"
        f"Scene objects: {objects_json}\n"
        f"{INTENT_GUIDE}\n"
        "Return the parse_result JSON matching the schema. "
        "Use scene.objects ids only. Provide why (short reasons), risk_hints, hazards, "
        "contradictions, missing_slots, and clarifying_questions."
    )


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no_json_object")
    return text[start : end + 1]


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


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _build_tool_schema(schema: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(schema, dict):
        return {"type": "object"}
    props = schema.get("properties", {})
    required = schema.get("required", [])
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": required,
    }


class LruCache:
    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.data: OrderedDict[Tuple[str, str, str, str], Dict[str, object]] = OrderedDict()

    def get(self, key: Tuple[str, str, str, str]) -> Optional[Dict[str, object]]:
        if key not in self.data:
            return None
        self.data.move_to_end(key)
        return self.data[key]

    def set(self, key: Tuple[str, str, str, str], value: Dict[str, object]) -> None:
        self.data[key] = value
        self.data.move_to_end(key)
        if len(self.data) > self.max_size:
            self.data.popitem(last=False)


class QwenStructuredClient:
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
        repair_attempts: int = DEFAULT_REPAIR_ATTEMPTS,
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
        self.repair_attempts = max(0, int(repair_attempts))

        env_path = _find_env_file(Path(__file__).resolve())
        if env_path:
            _load_env_file(env_path)

    def parse(
        self,
        text: str,
        scene_summary: str,
        scene_state_json: Optional[str] = None,
        conversation_context: Optional[str] = None,
    ) -> Tuple[Dict[str, object], Dict[str, object]]:
        input_ts = time.time()
        request_id = generate_request_id(text, timestamp=input_ts)
        summary_text, objects = build_scene_payload(scene_summary, scene_state_json)

        text_hash = _hash_text(text)
        scene_hash = _hash_text((summary_text or "") + (scene_state_json or ""))
        context_hash = _hash_text(conversation_context or "")
        cache_key = (text_hash, scene_hash, context_hash, self.model)
        cached = self.cache.get(cache_key)
        if cached is not None:
            result = json.loads(json.dumps(cached))
            _apply_context_overrides(result, text, summary_text, objects, request_id, input_ts)
            _apply_meta_defaults(result, self.model, "qwen_structured", latency_ms=0.0, cache_hit=True)
            return result, {"cache_hit": True, "fallback": False, "reason": "", "elapsed_sec": 0.0}

        start = time.monotonic()
        try:
            result = self._query_model(text, summary_text, objects, conversation_context)
            _apply_context_overrides(result, text, summary_text, objects, request_id, input_ts)
            _apply_meta_defaults(
                result,
                self.model,
                "qwen_structured",
                latency_ms=(time.monotonic() - start) * 1000.0,
            )
            result = ensure_scene_consistency(result)
            ok, errors = validate_parse_result(result)
            if not ok:
                raise ValueError(";".join(errors))
            meta = {"cache_hit": False, "fallback": False, "reason": ""}
        except Exception as exc:
            reason = str(exc)
            result = self._fallback_parse(text, summary_text, scene_state_json, reason)
            _apply_context_overrides(result, text, summary_text, objects, request_id, input_ts)
            _apply_meta_defaults(
                result,
                None,
                "fallback",
                latency_ms=(time.monotonic() - start) * 1000.0,
                debug={"fallback_reason": reason},
            )
            meta = {"cache_hit": False, "fallback": True, "reason": reason}

        meta["elapsed_sec"] = time.monotonic() - start
        self.cache.set(cache_key, result)
        return result, meta

    def _fallback_parse(
        self, text: str, scene_summary: str, scene_state_json: Optional[str], reason: str
    ) -> Dict[str, object]:
        if self.fallback_mode == "mock":
            fallback = build_mock_parse_result_v1(text, scene_summary, scene_state_json, parse_mode="fallback")
            fallback_meta = fallback.get("meta")
            if isinstance(fallback_meta, dict):
                fallback_meta["debug"] = {"fallback_reason": reason}
            return fallback
        return build_mock_parse_result_v1(text, scene_summary, scene_state_json, parse_mode="fallback")

    def _query_model(
        self,
        text: str,
        scene_summary: Optional[str],
        objects: list,
        conversation_context: Optional[str],
    ) -> Dict[str, object]:
        api_key = _resolve_api_key(self.api_key_env)
        if not api_key:
            raise ValueError("missing_api_key")

        url = self.base_url
        if not url.endswith("/chat/completions"):
            url = url + "/chat/completions"

        schema = load_schema_safe()
        tool_schema = _build_tool_schema(schema)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _build_user_prompt(text, scene_summary, objects, conversation_context),
            },
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "emit_parse_result",
                    "description": "Emit a ParseResult object",
                    "parameters": tool_schema,
                },
            }
        ]

        last_error: Optional[Exception] = None
        repair_left = self.repair_attempts
        network_tries = 0
        while True:
            try:
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": "emit_parse_result"}},
                }
                response_json = _post_json(url, payload, api_key, self.timeout_sec)
                parsed = self._extract_parse_result(response_json)
                ok, errors = validate_parse_result(parsed)
                if ok:
                    return parsed
                if repair_left <= 0:
                    raise ValueError(";".join(errors))
                repair_left -= 1
                messages = _build_repair_messages(parsed, errors)
                continue
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if network_tries >= self.max_retries:
                    break
                network_tries += 1
                continue

        if last_error:
            raise last_error
        raise ValueError("qwen_structured_failure")

    def _extract_parse_result(self, response_json: Dict[str, object]) -> Dict[str, object]:
        choices = response_json.get("choices", []) if isinstance(response_json, dict) else []
        if not choices:
            raise ValueError("empty_choices")
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls")
                if isinstance(tool_calls, list):
                    for call in tool_calls:
                        if not isinstance(call, dict):
                            continue
                        function = call.get("function", {})
                        if not isinstance(function, dict):
                            continue
                        if function.get("name") != "emit_parse_result":
                            continue
                        args = function.get("arguments")
                        if isinstance(args, dict):
                            return args
                        if isinstance(args, str):
                            return json.loads(args)
                function_call = message.get("function_call")
                if isinstance(function_call, dict) and function_call.get("name") == "emit_parse_result":
                    args = function_call.get("arguments")
                    if isinstance(args, dict):
                        return args
                    if isinstance(args, str):
                        return json.loads(args)
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return json.loads(_extract_json(content))
            text = first.get("text", "")
            if isinstance(text, str) and text.strip():
                return json.loads(_extract_json(text))
        raise ValueError("missing_content")


def _build_repair_messages(parsed: Dict[str, object], errors: list) -> list:
    error_text = "\n".join([str(err) for err in errors])
    return [
        {
            "role": "system",
            "content": "Fix the JSON to match the schema. Output JSON only, no extra text.",
        },
        {
            "role": "user",
            "content": f"Errors:\n{error_text}\nJSON:\n{json.dumps(parsed, ensure_ascii=True)}",
        },
    ]


def _apply_context_overrides(
    result: Dict[str, object],
    text: str,
    scene_summary: Optional[str],
    objects: list,
    request_id: str,
    input_ts: float,
) -> None:
    result["version"] = PARSE_RESULT_VERSION
    result["request_id"] = request_id
    input_payload = result.get("input") if isinstance(result.get("input"), dict) else {}
    input_payload["text"] = text
    input_payload["lang"] = input_payload.get("lang") if "lang" in input_payload else None
    input_payload["timestamp"] = input_ts
    result["input"] = input_payload

    scene_payload = result.get("scene") if isinstance(result.get("scene"), dict) else {}
    scene_payload["summary"] = scene_summary
    scene_payload["objects"] = objects
    scene_payload["timestamp"] = scene_payload.get("timestamp") if "timestamp" in scene_payload else None
    result["scene"] = scene_payload


def _apply_meta_defaults(
    result: Dict[str, object],
    model: Optional[str],
    parse_mode: str,
    latency_ms: Optional[float],
    cache_hit: Optional[bool] = None,
    debug: Optional[Dict[str, object]] = None,
) -> None:
    meta = result.get("meta") if isinstance(result.get("meta"), dict) else {}
    meta["model"] = model
    meta["parse_mode"] = parse_mode
    meta["latency_ms"] = latency_ms
    if debug is not None:
        meta["debug"] = debug
    elif "debug" not in meta:
        meta["debug"] = {}
    if cache_hit is not None:
        if not isinstance(meta.get("debug"), dict):
            meta["debug"] = {}
        meta["debug"]["cache_hit"] = cache_hit
    result["meta"] = meta


def _post_json(url: str, payload: Dict[str, object], api_key: str, timeout_sec: float) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        response_text = response.read().decode("utf-8")
    return json.loads(response_text)


def load_schema_safe() -> Dict[str, object]:
    from hri_safety_core.parse_result_utils import load_parse_result_schema

    schema = load_parse_result_schema()
    return schema if isinstance(schema, dict) else {}
