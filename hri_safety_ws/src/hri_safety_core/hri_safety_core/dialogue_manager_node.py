import json
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hri_safety_core.arbiter_utils import A_HIGH_DEFAULT, R_HIGH_DEFAULT
from hri_safety_core.belief_tracker import QUERY_ACTIONS
from hri_safety_core.dialogue_core import (
    STATE_NEED_DECISION,
    STATE_WAIT_USER,
    DialogueManagerCore,
)
from hri_safety_core.policy import RulePolicy, Sb3Policy
from hri_safety_core.policy.base import PolicyDecision

TERMINAL_ACTIONS = {"EXECUTE", "REFUSE_SAFE", "FALLBACK_HUMAN_HELP"}


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


class DialogueManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("dialogue_manager")
        self.declare_parameter("max_turns", 5)
        self.declare_parameter("max_repeat_action", 2)
        self.declare_parameter("policy_backend", "rule")
        self.declare_parameter("policy_path", "policies/ppo_policy.zip")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("amb_high", A_HIGH_DEFAULT)
        self.declare_parameter("risk_high", R_HIGH_DEFAULT)
        self.declare_parameter("context_topic", "/dialogue/context_instruction")
        self.declare_parameter("debug_topic", "/dialogue/debug_state")
        self.declare_parameter("obs_mode", "legacy")
        self.declare_parameter("use_age_context", False)

        max_turns = int(self.get_parameter("max_turns").value)
        max_repeat_action = int(self.get_parameter("max_repeat_action").value)
        use_age_context = _as_bool(self.get_parameter("use_age_context").value)
        self.core = DialogueManagerCore(
            max_turns=max_turns,
            max_repeat_action=max_repeat_action,
            use_age_context=use_age_context,
        )

        self.policy_backend = str(self.get_parameter("policy_backend").value).lower()
        self.policy = self._load_policy()

        self.context_topic = str(self.get_parameter("context_topic").value)
        self.debug_topic = str(self.get_parameter("debug_topic").value)

        self.publisher_context = self.create_publisher(String, self.context_topic, 10)
        self.publisher_action = self.create_publisher(String, "/arbiter/action", 10)
        self.publisher_utterance = self.create_publisher(String, "/robot/utterance", 10)
        self.publisher_debug = self.create_publisher(String, self.debug_topic, 10)

        self.create_subscription(String, "/user/instruction", self.on_user_instruction, 10)
        self.create_subscription(String, "/nl/parse_result", self.on_parse_result, 10)
        self.create_subscription(String, "/safety/features", self.on_features, 10)
        self.create_subscription(String, "/skill/result", self.on_skill_result, 10)
        if use_age_context:
            self.create_subscription(String, "/user/age_context", self.on_age_context, 10)

        self.get_logger().info(
            f"dialogue_manager started backend={self.policy_backend} max_turns={max_turns}"
        )

    def _load_policy(self):
        if self.policy_backend != "rl":
            return RulePolicy(
                amb_high=float(self.get_parameter("amb_high").value),
                risk_high=float(self.get_parameter("risk_high").value),
            )
        policy_path = str(self.get_parameter("policy_path").value)
        device = str(self.get_parameter("device").value)
        deterministic = _as_bool(self.get_parameter("deterministic").value)
        obs_mode = str(self.get_parameter("obs_mode").value)
        max_turns = int(self.get_parameter("max_turns").value)
        try:
            return Sb3Policy(
                policy_path=policy_path,
                device=device,
                deterministic=deterministic,
                obs_mode=obs_mode,
                max_turns=max_turns,
            )
        except Exception as exc:
            self.get_logger().error(f"Failed to load RL policy: {exc}. Falling back to rule policy.")
            self.policy_backend = "rule"
            return RulePolicy(
                amb_high=float(self.get_parameter("amb_high").value),
                risk_high=float(self.get_parameter("risk_high").value),
            )

    def on_user_instruction(self, msg: String) -> None:
        text = msg.data.strip()
        if not text:
            return

        if self.core.state == STATE_WAIT_USER:
            self.core.receive_user_reply(text)
        else:
            self.core.start_session(text)

        if self.core.should_force_terminal():
            self._force_terminal()
            return

        self._publish_context()

    def on_parse_result(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            self.core.update_parse_result(data)
        self._maybe_decide()

    def on_features(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            self.core.update_features(data)
        self._maybe_decide()

    def on_skill_result(self, msg: String) -> None:
        # Optional hook: could store success/failure for richer logs.
        _ = msg

    def on_age_context(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        self.core.update_age_context(data)
        p_minor = float(data.get("p_minor", 0.0))
        age_conf = float(data.get("age_conf", 0.0))
        source = str(data.get("source", ""))
        self.get_logger().info(
            f"age_context source={source} p_minor={p_minor:.2f} age_conf={age_conf:.2f}"
        )

    def _publish_context(self) -> None:
        if not self.core.context:
            return
        out = String()
        out.data = self.core.context
        self.publisher_context.publish(out)

    def _maybe_decide(self) -> None:
        if self.core.state != STATE_NEED_DECISION:
            return
        belief = self.core.build_belief()
        if belief is None:
            return

        if self.core.should_force_terminal():
            self._force_terminal()
            return

        action_mask = self.core.build_action_mask(belief)
        decision = self.policy.select_action(belief, action_mask)
        action = decision.action

        if action in QUERY_ACTIONS:
            if not self.core.can_ask() or self.core.query_repeat_exceeded(action):
                self._force_terminal()
                return
            self._publish_action(decision.action, decision.utterance)
            self.core.apply_decision(decision)
            self._publish_debug(decision, belief)
            return

        if action not in TERMINAL_ACTIONS:
            action = "FALLBACK_HUMAN_HELP"
            decision = self.core.choose_terminal_action(
                risk_high=float(self.get_parameter("risk_high").value)
            )

        self._publish_action(action, decision.utterance)
        self.core.apply_decision(decision)
        self._publish_debug(decision, belief)

    def _force_terminal(self) -> None:
        decision = self.core.choose_terminal_action(
            risk_high=float(self.get_parameter("risk_high").value)
        )
        self._publish_action(decision.action, decision.utterance)
        self.core.apply_decision(decision)
        self._publish_debug(decision, None)

    def _publish_action(self, action: str, utterance: str) -> None:
        action_msg = String()
        action_msg.data = action
        self.publisher_action.publish(action_msg)

        utterance_msg = String()
        utterance_msg.data = utterance
        self.publisher_utterance.publish(utterance_msg)

        self.get_logger().info(f"action={action} turn={self.core.turn_count}")

    def _publish_debug(self, decision: PolicyDecision, belief: Optional[object]) -> None:
        payload: Dict[str, object] = {
            "state": self.core.state,
            "turn_count": self.core.turn_count,
            "query_count": self.core.query_count,
            "last_action": decision.action,
            "policy_backend": self.policy_backend,
        }
        if decision.info:
            payload["policy_info"] = decision.info
        if belief is not None:
            payload.update(
                {
                    "amb": belief.amb,
                    "risk": belief.risk,
                    "conflict": belief.conflict,
                    "missing_slots": belief.missing_slots,
                    "margin": belief.margin,
                    "last_reply": belief.last_user_reply_type,
                }
            )
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.publisher_debug.publish(msg)


def main() -> None:
    rclpy.init()
    node = DialogueManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
