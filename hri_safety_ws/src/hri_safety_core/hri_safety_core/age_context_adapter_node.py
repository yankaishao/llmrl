import json
import time
from typing import Dict, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _normalize(p_minor: float, p_adult: float, p_older: float) -> Tuple[float, float, float]:
    total = p_minor + p_adult + p_older
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (p_minor / total, p_adult / total, p_older / total)


def _sample_age_context(rng: np.random.Generator) -> Dict[str, object]:
    true_group = rng.integers(0, 3)
    age_conf = float(rng.uniform(0.3, 1.0))
    misclass_prob = min(0.6, max(0.05, 1.0 - age_conf))

    probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    probs[true_group] = 0.85
    if rng.random() < misclass_prob:
        wrong = (true_group + rng.integers(1, 3)) % 3
        probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        probs[wrong] = 0.85

    noise = rng.normal(0.0, 0.05, size=3)
    probs = np.clip(probs + noise, 0.01, 0.98)
    p_minor, p_adult, p_older = _normalize(float(probs[0]), float(probs[1]), float(probs[2]))

    return {
        "p_minor": p_minor,
        "p_adult": p_adult,
        "p_older": p_older,
        "age_conf": age_conf,
        "guardian_present": None,
    }


class AgeContextAdapter(Node):
    def __init__(self) -> None:
        super().__init__("age_context_adapter")
        self.declare_parameter("mode", "manual")
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("seed", 0)
        self.declare_parameter("p_minor", 0.0)
        self.declare_parameter("p_adult", 1.0)
        self.declare_parameter("p_older", 0.0)
        self.declare_parameter("age_conf", 1.0)
        self.declare_parameter("guardian_present", None)

        self.mode = str(self.get_parameter("mode").value).lower()
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(seed)

        self.publisher = self.create_publisher(String, "/user/age_context", 10)
        timer_period = 1.0 / max(0.1, self.publish_hz)
        self.timer = self.create_timer(timer_period, self._on_timer)
        self.get_logger().info(f"age_context_adapter started mode={self.mode}")

    def _manual_payload(self) -> Dict[str, object]:
        p_minor = float(self.get_parameter("p_minor").value)
        p_adult = float(self.get_parameter("p_adult").value)
        p_older = float(self.get_parameter("p_older").value)
        p_minor, p_adult, p_older = _normalize(p_minor, p_adult, p_older)
        age_conf = float(self.get_parameter("age_conf").value)
        guardian_raw = self.get_parameter("guardian_present").value
        guardian = None
        if isinstance(guardian_raw, bool):
            guardian = guardian_raw
        elif isinstance(guardian_raw, str):
            lowered = guardian_raw.strip().lower()
            if lowered in {"true", "yes", "1"}:
                guardian = True
            elif lowered in {"false", "no", "0"}:
                guardian = False
        return {
            "p_minor": p_minor,
            "p_adult": p_adult,
            "p_older": p_older,
            "age_conf": age_conf,
            "guardian_present": guardian,
        }

    def _on_timer(self) -> None:
        if self.mode == "manual":
            payload = self._manual_payload()
            source = "manual"
        elif self.mode == "sim":
            payload = _sample_age_context(self.rng)
            source = "sim"
        else:
            payload = _sample_age_context(self.rng)
            source = "cv_stub"

        payload["source"] = source
        payload["ts"] = time.time()

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = AgeContextAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
