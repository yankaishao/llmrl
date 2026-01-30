import json
from typing import Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from hri_safety_core.gazebo_utils import (
    DEFAULT_IGN_CMD,
    DEFAULT_MAX_RETRIES,
    DEFAULT_SET_POSE_REPTYPE,
    DEFAULT_SET_POSE_REQTYPE,
    DEFAULT_SET_POSE_SERVICE,
    DEFAULT_TIMEOUT_SEC,
    build_pose_request,
    call_set_pose,
)


DEFAULT_GRIPPER_POSITION = "0.5,0.0,0.8"
DEFAULT_PLACE_POSITION = "0.4,-0.3,0.8"
DEFAULT_HANDOVER_POSITION = "0.6,0.0,1.0"


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


def _pose_from_vector3(vec: Tuple[float, float, float]) -> Dict[str, Dict[str, float]]:
    return {
        "position": {"x": vec[0], "y": vec[1], "z": vec[2]},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def _extract_pose(pose: object, fallback: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    if not isinstance(pose, dict):
        return fallback
    position = pose.get("position")
    orientation = pose.get("orientation")
    if isinstance(position, dict):
        pos = {
            "x": float(position.get("x", fallback["position"]["x"])),
            "y": float(position.get("y", fallback["position"]["y"])),
            "z": float(position.get("z", fallback["position"]["z"])),
        }
    else:
        pos = {
            "x": float(pose.get("x", fallback["position"]["x"])),
            "y": float(pose.get("y", fallback["position"]["y"])),
            "z": float(pose.get("z", fallback["position"]["z"])),
        }
    if isinstance(orientation, dict):
        ori = {
            "x": float(orientation.get("x", fallback["orientation"]["x"])),
            "y": float(orientation.get("y", fallback["orientation"]["y"])),
            "z": float(orientation.get("z", fallback["orientation"]["z"])),
            "w": float(orientation.get("w", fallback["orientation"]["w"])),
        }
    else:
        ori = dict(fallback["orientation"])
    return {"position": pos, "orientation": ori}


class SkillExecutorGazebo(Node):
    def __init__(self) -> None:
        super().__init__("skill_executor_gazebo")
        self.declare_parameter("ign_cmd", DEFAULT_IGN_CMD)
        self.declare_parameter("pose_service", DEFAULT_SET_POSE_SERVICE)
        self.declare_parameter("pose_reqtype", DEFAULT_SET_POSE_REQTYPE)
        self.declare_parameter("pose_reptype", DEFAULT_SET_POSE_REPTYPE)
        self.declare_parameter("command_timeout", DEFAULT_TIMEOUT_SEC)
        self.declare_parameter("max_retries", DEFAULT_MAX_RETRIES)
        self.declare_parameter("gripper_position", DEFAULT_GRIPPER_POSITION)
        self.declare_parameter("place_position", DEFAULT_PLACE_POSITION)
        self.declare_parameter("handover_position", DEFAULT_HANDOVER_POSITION)

        self.ign_cmd = str(self.get_parameter("ign_cmd").value)
        self.pose_service = str(self.get_parameter("pose_service").value)
        self.pose_reqtype = str(self.get_parameter("pose_reqtype").value)
        self.pose_reptype = str(self.get_parameter("pose_reptype").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.max_retries = int(self.get_parameter("max_retries").value)

        grip_vec = _parse_vector3(self.get_parameter("gripper_position").value, (0.5, 0.0, 0.8))
        place_vec = _parse_vector3(self.get_parameter("place_position").value, (0.4, -0.3, 0.8))
        hand_vec = _parse_vector3(self.get_parameter("handover_position").value, (0.6, 0.0, 1.0))
        self.gripper_pose = _pose_from_vector3(grip_vec)
        self.place_pose = _pose_from_vector3(place_vec)
        self.handover_pose = _pose_from_vector3(hand_vec)

        self.attached_id = ""
        self.known_entities = set()

        self.publisher = self.create_publisher(String, "/skill/result", 10)
        self.create_subscription(String, "/skill/command", self.on_command, 10)
        self.create_subscription(String, "/scene/state_json", self.on_state, 10)
        self.get_logger().info("skill_executor_gazebo started.")

    def on_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objects = payload.get("objects", []) if isinstance(payload, dict) else []
        entities = set()
        for obj in objects:
            if isinstance(obj, dict):
                obj_id = obj.get("id")
                if isinstance(obj_id, str):
                    entities.add(obj_id)
        self.known_entities = entities

    def on_command(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            self._publish_result(False, "invalid_json", "", 1.0)
            return
        if not isinstance(payload, dict):
            self._publish_result(False, "invalid_payload", "", 1.0)
            return

        skill = str(payload.get("skill", "")).lower()
        target_id = str(payload.get("target_id", "")).strip()
        target_pose = payload.get("target_pose")

        if skill == "pick":
            self._handle_pick(target_id, target_pose)
        elif skill == "place":
            self._handle_place(target_id, target_pose)
        elif skill == "handover":
            self._handle_handover(target_id, target_pose)
        else:
            self._publish_result(False, "unsupported_skill", target_id, 1.0)

    def _handle_pick(self, target_id: str, target_pose: object) -> None:
        if not target_id:
            self._publish_result(False, "missing_target_id", "", 1.0)
            return
        if self.known_entities and target_id not in self.known_entities:
            self._publish_result(False, "entity_not_found", target_id, 1.0)
            return
        pose = _extract_pose(target_pose, self.gripper_pose)
        ok, reason = self._set_pose(target_id, pose)
        if ok:
            self.attached_id = target_id
            self._publish_result(True, "", target_id, 0.0)
        else:
            self._publish_result(False, reason, target_id, 1.0)

    def _handle_place(self, target_id: str, target_pose: object) -> None:
        if not self.attached_id:
            self._publish_result(False, "no_attached_object", "", 1.0)
            return
        if target_id and target_id != self.attached_id:
            self._publish_result(False, "attached_mismatch", self.attached_id, 1.0)
            return
        pose = _extract_pose(target_pose, self.place_pose)
        ok, reason = self._set_pose(self.attached_id, pose)
        if ok:
            executed_id = self.attached_id
            self.attached_id = ""
            self._publish_result(True, "", executed_id, 0.0)
        else:
            self._publish_result(False, reason, self.attached_id, 1.0)

    def _handle_handover(self, target_id: str, target_pose: object) -> None:
        if not self.attached_id:
            self._publish_result(False, "no_attached_object", "", 1.0)
            return
        if target_id and target_id != self.attached_id:
            self._publish_result(False, "attached_mismatch", self.attached_id, 1.0)
            return
        pose = _extract_pose(target_pose, self.handover_pose)
        ok, reason = self._set_pose(self.attached_id, pose)
        if ok:
            executed_id = self.attached_id
            self.attached_id = ""
            self._publish_result(True, "", executed_id, 0.0)
        else:
            self._publish_result(False, reason, self.attached_id, 1.0)

    def _set_pose(self, target_id: str, pose: Dict[str, Dict[str, float]]) -> Tuple[bool, str]:
        req = build_pose_request(target_id, pose["position"], pose["orientation"])
        return call_set_pose(
            self.ign_cmd,
            self.pose_service,
            self.pose_reqtype,
            self.pose_reptype,
            req,
            self.command_timeout,
            self.max_retries,
        )

    def _publish_result(self, success: bool, reason: str, target_id: str, cost: float) -> None:
        result = {
            "success": bool(success),
            "reason": reason,
            "cost": float(cost),
            "executed_target_id": target_id,
        }
        msg = String()
        msg.data = json.dumps(result, ensure_ascii=True)
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = SkillExecutorGazebo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
