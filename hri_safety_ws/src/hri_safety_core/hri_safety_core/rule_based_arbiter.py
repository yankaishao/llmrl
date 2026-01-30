import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hri_safety_core.arbiter_utils import (
    A_HIGH_DEFAULT,
    R_HIGH_DEFAULT,
    clamp,
    extract_candidates,
    rule_based_action,
    safe_float,
    safe_int,
    top_two_candidates,
    utterance_for_action,
)


class RuleBasedArbiter(Node):
    def __init__(self) -> None:
        super().__init__("rule_based_arbiter")
        self.declare_parameter("amb_high", A_HIGH_DEFAULT)
        self.declare_parameter("risk_high", R_HIGH_DEFAULT)

        self.publisher_action = self.create_publisher(String, "/arbiter/action", 10)
        self.publisher_utterance = self.create_publisher(String, "/robot/utterance", 10)

        self.top2_candidates = []

        self.create_subscription(String, "/nl/parse_result", self.on_parse_result, 10)
        self.create_subscription(String, "/safety/features", self.on_features, 10)
        self.get_logger().info("rule_based_arbiter started.")

    def on_parse_result(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            self.top2_candidates = []
            return
        candidates = extract_candidates(data)
        self.top2_candidates = top_two_candidates(candidates)

    def on_features(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            data = {}

        amb = clamp(safe_float(data.get("amb", 0.0)))
        risk = clamp(safe_float(data.get("risk", 0.0)))
        conflict = safe_int(data.get("conflict", 0))
        conflict_reason = str(data.get("conflict_reason", "")) if data.get("conflict_reason") is not None else ""
        selected_id = str(data.get("selected_top1_id", "")) if data.get("selected_top1_id") is not None else ""

        a_high = safe_float(self.get_parameter("amb_high").value, A_HIGH_DEFAULT)
        r_high = safe_float(self.get_parameter("risk_high").value, R_HIGH_DEFAULT)

        has_choice = len(self.top2_candidates) >= 2
        action = rule_based_action(conflict, risk, amb, has_choice, a_high, r_high)

        action_msg = String()
        action_msg.data = action
        self.publisher_action.publish(action_msg)

        utterance = utterance_for_action(action, selected_id, conflict_reason, self.top2_candidates)
        utterance_msg = String()
        utterance_msg.data = utterance
        self.publisher_utterance.publish(utterance_msg)

        self.get_logger().info(f"action={action} amb={amb:.2f} risk={risk:.2f} conflict={conflict}")


def main() -> None:
    rclpy.init()
    node = RuleBasedArbiter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
