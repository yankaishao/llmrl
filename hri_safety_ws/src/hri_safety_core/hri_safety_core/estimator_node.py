import json
from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

RISK_RULES = [
    ("knife", 1.0),
    ("glass", 0.7),
    ("cup", 0.2),
]

DEFAULT_RISK = 0.1


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _extract_candidates(data: Dict[str, object]) -> List[Dict[str, float]]:
    raw = data.get("candidates", [])
    if not isinstance(raw, list):
        return []
    candidates: List[Dict[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id")
        score = item.get("score")
        if isinstance(obj_id, str) and isinstance(score, (int, float)):
            candidates.append({"id": obj_id, "score": float(score)})
    return candidates


def _select_top1(candidates: List[Dict[str, float]]) -> Optional[Dict[str, float]]:
    if not candidates:
        return None
    return max(candidates, key=lambda item: item["score"])


def _risk_from_id(obj_id: str) -> float:
    obj_id_lower = obj_id.lower()
    for key, value in RISK_RULES:
        if key in obj_id_lower:
            return value
    return DEFAULT_RISK


def _risk_from_constraints(constraints: Dict[str, object], base_risk: float) -> float:
    risk = base_risk
    risk_hints = constraints.get("risk_hints", []) if isinstance(constraints, dict) else []
    if isinstance(risk_hints, list):
        hints_lower = [str(hint).lower() for hint in risk_hints]
        if "sharp" in hints_lower:
            risk = max(risk, 1.0)
    return risk


def _build_features(parse_result: Dict[str, object]) -> Dict[str, object]:
    task_type = parse_result.get("task_type", "unknown")
    candidates = _extract_candidates(parse_result)

    conflict = 0
    reason = "none"
    if not candidates:
        conflict = 1
        reason = "no_candidate"
    elif task_type == "unknown":
        conflict = 1
        reason = "unknown_task"

    max_score = max((c["score"] for c in candidates), default=0.0)
    amb = _clamp(1.0 - max_score)

    top1 = _select_top1(candidates)
    selected_top1_id = top1["id"] if top1 else ""

    constraints = parse_result.get("constraints", {})
    risk = _risk_from_constraints(constraints, _risk_from_id(selected_top1_id)) if selected_top1_id else DEFAULT_RISK
    risk = _clamp(risk)

    return {
        "amb": amb,
        "risk": risk,
        "conflict": conflict,
        "conflict_reason": reason,
        "selected_top1_id": selected_top1_id,
    }


class EstimatorNode(Node):
    def __init__(self) -> None:
        super().__init__("estimator_node")
        self.publisher_ = self.create_publisher(String, "/safety/features", 10)
        self.last_summary = ""
        self.create_subscription(String, "/scene/summary", self.on_summary, 10)
        self.create_subscription(String, "/nl/parse_result", self.on_parse_result, 10)
        self.get_logger().info("estimator_node started.")

    def on_summary(self, msg: String) -> None:
        self.last_summary = msg.data

    def on_parse_result(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            features = {
                "amb": 1.0,
                "risk": DEFAULT_RISK,
                "conflict": 1,
                "conflict_reason": "invalid_json",
                "selected_top1_id": "",
            }
        else:
            features = _build_features(data)

        out = String()
        out.data = json.dumps(features, ensure_ascii=True)
        self.publisher_.publish(out)
        self.get_logger().info("published /safety/features")


def main() -> None:
    rclpy.init()
    node = EstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
