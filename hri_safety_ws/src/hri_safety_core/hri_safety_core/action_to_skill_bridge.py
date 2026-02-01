import json
from typing import Tuple

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # pragma: no cover - optional ROS dependency for non-ROS tests
    rclpy = None

    class Node:  # type: ignore[override]
        pass

    class String:  # type: ignore[override]
        pass


DEFAULT_PLACE_POSITION = "0.4,-0.3,0.8"
DEFAULT_HANDOVER_POSITION = "0.6,0.0,1.0"


def should_dispatch_skill(action: str) -> bool:
    return action.strip().upper() == "EXECUTE"


def _parse_vector3(value: object, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return float(value[0]), float(value[1]), float(value[2])
        except (TypeError, ValueError):
            return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if len(parts) >= 3:
            try:
                return float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                return default
    return default


def _pose_from_vector3(vec: Tuple[float, float, float]) -> dict:
    return {
        "position": {"x": vec[0], "y": vec[1], "z": vec[2]},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


class ActionToSkillBridge(Node):
    def __init__(self) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy_not_available")
        super().__init__("action_to_skill_bridge")
        self.declare_parameter("place_position", DEFAULT_PLACE_POSITION)
        self.declare_parameter("handover_position", DEFAULT_HANDOVER_POSITION)

        place_vec = _parse_vector3(self.get_parameter("place_position").value, (0.4, -0.3, 0.8))
        hand_vec = _parse_vector3(self.get_parameter("handover_position").value, (0.6, 0.0, 1.0))
        self.place_pose = _pose_from_vector3(place_vec)
        self.handover_pose = _pose_from_vector3(hand_vec)

        self.publisher = self.create_publisher(String, "/skill/command", 10)
        self.create_subscription(String, "/arbiter/action", self.on_action, 10)
        self.create_subscription(String, "/safety/features", self.on_features, 10)
        self.create_subscription(String, "/nl/parse_result", self.on_parse_result, 10)
        self.create_subscription(String, "/skill/result", self.on_skill_result, 10)
        self.create_subscription(String, "/user/instruction", self.on_instruction, 10)

        self.selected_target_id = ""
        self.task_type = "unknown"
        self.instruction_seq = 0
        self.last_execute_seq = -1
        self.pending_stage = ""
        self.pending_next_skill = ""
        self.pending_target_id = ""

        self.get_logger().info("action_to_skill_bridge started.")

    def on_instruction(self, msg: String) -> None:
        self.instruction_seq += 1
        self.last_execute_seq = -1
        self.pending_stage = ""
        self.pending_next_skill = ""
        self.pending_target_id = ""

    def on_features(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            target_id = data.get("selected_top1_id")
            if isinstance(target_id, str):
                self.selected_target_id = target_id

    def on_parse_result(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            task_type = data.get("task_type")
            if isinstance(task_type, str):
                self.task_type = task_type

    def on_action(self, msg: String) -> None:
        action = msg.data.strip().upper()
        if not should_dispatch_skill(action):
            return
        if self.pending_stage:
            return
        if self.last_execute_seq == self.instruction_seq:
            return
        if not self.selected_target_id:
            self.get_logger().warning("No selected target id; skipping skill.")
            return

        next_skill = "handover" if self.task_type == "handover" else "place"
        self.pending_stage = "pick"
        self.pending_next_skill = next_skill
        self.pending_target_id = self.selected_target_id
        self.last_execute_seq = self.instruction_seq

        self._publish_skill("pick", self.selected_target_id, None)

    def on_skill_result(self, msg: String) -> None:
        if not self.pending_stage:
            return
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not isinstance(data, dict):
            return
        success = bool(data.get("success", False))
        if not success:
            self.pending_stage = ""
            self.pending_next_skill = ""
            self.pending_target_id = ""
            return

        if self.pending_stage == "pick":
            next_skill = self.pending_next_skill
            target_pose = self.handover_pose if next_skill == "handover" else self.place_pose
            self.pending_stage = next_skill
            self._publish_skill(next_skill, self.pending_target_id, target_pose)
            return

        self.pending_stage = ""
        self.pending_next_skill = ""
        self.pending_target_id = ""

    def _publish_skill(self, skill: str, target_id: str, target_pose: object) -> None:
        payload = {"skill": skill, "target_id": target_id}
        if target_pose is not None:
            payload["target_pose"] = target_pose
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=True)
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = ActionToSkillBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
