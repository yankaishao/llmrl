import json
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hri_safety_core.arbiter_utils import (
    A_HIGH_DEFAULT,
    R_HIGH_DEFAULT,
    QUERY_ACTIONS,
    action_from_index,
    build_rl_observation,
    extract_candidates,
    rule_based_action,
    safe_float,
    safe_int,
    top_two_candidates,
    utterance_for_action,
)


ACTION_COUNT = 5


class RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = True):
        action = self.rng.integers(0, ACTION_COUNT)
        return np.array([action], dtype=np.int64), None


def _load_policy(policy_path: str, device: str):
    if policy_path == "random":
        return RandomPolicy()
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable_baselines3_not_installed") from exc

    policy_file = Path(policy_path)
    if not policy_file.is_file():
        raise FileNotFoundError(f"policy_not_found:{policy_path}")

    return PPO.load(str(policy_file), device=device)


def _predict_action(policy, obs: np.ndarray, deterministic: bool) -> int:
    action, _ = policy.predict(obs, deterministic=deterministic)
    if isinstance(action, np.ndarray):
        action = int(action.item())
    elif isinstance(action, (int, np.integer)):
        action = int(action)
    else:
        raise ValueError("invalid_action_type")
    return action


class RlArbiterNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_arbiter_node")
        self.declare_parameter("policy_path", "policies/ppo_policy.zip")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("fallback_to_rule", True)
        self.declare_parameter("amb_high", A_HIGH_DEFAULT)
        self.declare_parameter("risk_high", R_HIGH_DEFAULT)

        self.publisher_action = self.create_publisher(String, "/arbiter/action", 10)
        self.publisher_utterance = self.create_publisher(String, "/robot/utterance", 10)

        self.top2_candidates = []
        self.query_count = 0
        self.last_outcome = 0.0

        self.policy = None
        self._load_policy_from_params()

        self.create_subscription(String, "/nl/parse_result", self.on_parse_result, 10)
        self.create_subscription(String, "/safety/features", self.on_features, 10)
        self.create_subscription(String, "/user/instruction", self.on_instruction, 10)
        self.get_logger().info("rl_arbiter_node started.")

    def _load_policy_from_params(self) -> None:
        policy_path = str(self.get_parameter("policy_path").value)
        device = str(self.get_parameter("device").value)
        if not policy_path:
            self.get_logger().warning("policy_path is empty; using fallback mode.")
            self.policy = None
            return
        try:
            self.policy = _load_policy(policy_path, device)
            self.get_logger().info(f"Loaded PPO policy from {policy_path}")
        except Exception as exc:
            self.get_logger().error(f"Failed to load policy: {exc}")
            self.policy = None

    def on_instruction(self, msg: String) -> None:
        self.query_count = 0
        self.last_outcome = 0.0

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

        conflict = safe_int(data.get("conflict", 0))
        conflict_reason = str(data.get("conflict_reason", "")) if data.get("conflict_reason") is not None else ""
        selected_id = str(data.get("selected_top1_id", "")) if data.get("selected_top1_id") is not None else ""

        obs = build_rl_observation(data, query_count=self.query_count, last_outcome=self.last_outcome)

        action = None
        if self.policy is not None:
            deterministic = bool(self.get_parameter("deterministic").value)
            try:
                action_idx = _predict_action(self.policy, obs, deterministic)
                if action_idx < 0 or action_idx >= ACTION_COUNT:
                    raise ValueError("action_out_of_range")
                action = action_from_index(action_idx)
            except Exception as exc:
                self.get_logger().warning(f"Policy inference failed: {exc}")
                action = None

        if action is None:
            fallback_to_rule = bool(self.get_parameter("fallback_to_rule").value)
            if fallback_to_rule:
                amb = float(obs[0])
                risk = float(obs[1])
                a_high = safe_float(self.get_parameter("amb_high").value, A_HIGH_DEFAULT)
                r_high = safe_float(self.get_parameter("risk_high").value, R_HIGH_DEFAULT)
                has_choice = len(self.top2_candidates) >= 2
                action = rule_based_action(conflict, risk, amb, has_choice, a_high, r_high)
            else:
                action = "REFUSE_SAFE"

        if action in QUERY_ACTIONS:
            self.query_count += 1

        if action == "EXECUTE":
            self.last_outcome = 1.0 if conflict == 0 and float(obs[1]) < 0.7 else 0.0
        elif action == "REFUSE_SAFE":
            self.last_outcome = 1.0 if conflict == 1 or float(obs[1]) >= 0.7 else 0.0
        else:
            self.last_outcome = 0.0

        action_msg = String()
        action_msg.data = action
        self.publisher_action.publish(action_msg)

        utterance = utterance_for_action(action, selected_id, conflict_reason, self.top2_candidates)
        utterance_msg = String()
        utterance_msg.data = utterance
        self.publisher_utterance.publish(utterance_msg)

        self.get_logger().info(
            f"action={action} amb={float(obs[0]):.2f} risk={float(obs[1]):.2f} conflict={conflict}"
        )


def main() -> None:
    rclpy.init()
    node = RlArbiterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
