#!/usr/bin/env python3
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "hri_safety_ws" / "src" / "hri_safety_core"))

from hri_safety_core.estimator.features_from_parse import features_from_parse_result  # noqa: E402
from hri_safety_core.parse_result_utils import build_mock_parse_result_v1  # noqa: E402
from hri_safety_core.parser_utils import DEFAULT_SCENE_SUMMARY  # noqa: E402
from hri_safety_core.qwen_structured_parser import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_CACHE_SIZE,
    DEFAULT_FALLBACK_MODE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_STRUCTURED_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SEC,
    QwenStructuredClient,
)

DEFAULT_TEXTS = [
    "hand me the left cup",
    "give me the knife",
    "把红色杯子递给我",
    "move it over there",
]


def _build_parser(parser_mode: str) -> QwenStructuredClient | None:
    if parser_mode != "qwen_structured":
        return None
    return QwenStructuredClient(
        model=DEFAULT_STRUCTURED_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key_env=DEFAULT_API_KEY_ENV,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        max_retries=DEFAULT_MAX_RETRIES,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        cache_size=DEFAULT_CACHE_SIZE,
        fallback_mode=DEFAULT_FALLBACK_MODE,
    )


def _format_row(row: dict, widths: dict) -> str:
    parts = []
    for key, width in widths.items():
        value = str(row.get(key, ""))
        if len(value) > width:
            value = value[: max(0, width - 3)] + "..."
        parts.append(value.ljust(width))
    return " | ".join(parts)


def _print_table(rows: list) -> None:
    widths = {
        "text": 26,
        "amb": 5,
        "risk": 5,
        "conflict": 8,
        "p_top": 5,
        "margin": 6,
        "hazard": 8,
        "missing": 7,
        "intent": 10,
        "q": 10,
    }
    header = _format_row(
        {
            "text": "text",
            "amb": "amb",
            "risk": "risk",
            "conflict": "conf",
            "p_top": "p_top",
            "margin": "margin",
            "hazard": "hazard",
            "missing": "missing",
            "intent": "intent",
            "q": "suggest_q",
        },
        widths,
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(_format_row(row, widths))


def run_demo(args: argparse.Namespace) -> None:
    texts = args.text or DEFAULT_TEXTS
    parser = _build_parser(args.parser)

    rows = []
    for text in texts:
        if args.parser == "mock":
            parse_result = build_mock_parse_result_v1(text, args.scene_summary, args.scene_state_json)
        elif parser is not None:
            parse_result, _meta = parser.parse(
                text,
                args.scene_summary,
                scene_state_json=args.scene_state_json,
                conversation_context=args.context,
            )
        else:
            raise ValueError("unknown_parser")

        features = features_from_parse_result(parse_result)
        rows.append(
            {
                "text": text,
                "amb": f"{features.get('amb', 0.0):.2f}",
                "risk": f"{features.get('risk', 0.0):.2f}",
                "conflict": f"{features.get('conflict', 0.0):.0f}",
                "p_top": f"{features.get('p_top', 0.0):.2f}",
                "margin": f"{features.get('margin', 0.0):.2f}",
                "hazard": str(features.get("hazard_tag_top", "")),
                "missing": str(features.get("missing_slots_count", 0)),
                "intent": str(features.get("top_intent", "")),
                "q": ",".join(features.get("suggested_question_types", [])[:2]),
            }
        )

    _print_table(rows)

    if not args.ros:
        return

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except Exception as exc:
        raise RuntimeError("ros2_not_available") from exc

    rclpy.init()
    node = Node("demo_nl_to_features")
    pub_instr = node.create_publisher(String, "/user/instruction", 10)
    pub_summary = node.create_publisher(String, "/scene/summary", 10)
    pub_state = node.create_publisher(String, "/scene/state_json", 10)

    for text in texts:
        msg_sum = String()
        msg_sum.data = args.scene_summary
        pub_summary.publish(msg_sum)

        if args.scene_state_json:
            msg_state = String()
            msg_state.data = args.scene_state_json
            pub_state.publish(msg_state)

        msg = String()
        msg.data = text
        pub_instr.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.1)
        time.sleep(args.ros_interval)

    node.destroy_node()
    rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parser", type=str, default="mock", choices=["mock", "qwen_structured"])
    parser.add_argument("--scene-summary", type=str, default=DEFAULT_SCENE_SUMMARY)
    parser.add_argument("--scene-state-json", type=str, default="")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--text", action="append", default=[])
    parser.add_argument("--ros", action="store_true")
    parser.add_argument("--ros-interval", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
