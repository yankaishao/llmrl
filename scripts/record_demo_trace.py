#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="results/demo_trace.jsonl")
    parser.add_argument("--duration", type=float, default=0.0)
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rclpy
        from rclpy.node import Node
        from std_msgs.msg import String
    except Exception as exc:
        raise RuntimeError("ros2_not_available") from exc

    rclpy.init()

    class TraceNode(Node):
        def __init__(self):
            super().__init__("record_demo_trace")
            self.file = out_path.open("a", encoding="utf-8")
            self.create_subscription(String, "/robot/utterance", self._on_utterance, 10)
            self.create_subscription(String, "/arbiter/action", self._on_action, 10)
            self.create_subscription(String, "/safety/features", self._on_features, 10)

        def _write(self, topic: str, payload: str) -> None:
            record = {"ts": time.time(), "topic": topic, "data": payload}
            self.file.write(json.dumps(record, ensure_ascii=True) + "\n")
            self.file.flush()

        def _on_utterance(self, msg: String) -> None:
            self._write("/robot/utterance", msg.data)

        def _on_action(self, msg: String) -> None:
            self._write("/arbiter/action", msg.data)

        def _on_features(self, msg: String) -> None:
            self._write("/safety/features", msg.data)

    node = TraceNode()
    start = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        node.file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    run()
