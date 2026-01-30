import json
import random
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

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


DEFAULT_RISK_KEYWORDS = ["knife", "scissor", "blade"]
DEFAULT_DISTRACTOR_POSITION = "0.0,0.5,0.8"


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


def _group_key(name: str) -> str:
    parts = name.split("_")
    if parts and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return name


def _is_risky(name: str) -> bool:
    lower = name.lower()
    return any(keyword in lower for keyword in DEFAULT_RISK_KEYWORDS)


class EpisodeManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("episode_manager_node")
        self.declare_parameter("ign_cmd", DEFAULT_IGN_CMD)
        self.declare_parameter("pose_service", DEFAULT_SET_POSE_SERVICE)
        self.declare_parameter("pose_reqtype", DEFAULT_SET_POSE_REQTYPE)
        self.declare_parameter("pose_reptype", DEFAULT_SET_POSE_REPTYPE)
        self.declare_parameter("command_timeout", DEFAULT_TIMEOUT_SEC)
        self.declare_parameter("max_retries", DEFAULT_MAX_RETRIES)
        self.declare_parameter("random_seed", 0)
        self.declare_parameter("swap_similar", True)
        self.declare_parameter("risk_offset_range", 0.2)
        self.declare_parameter("generic_offset_range", 0.05)
        self.declare_parameter("distractor_enabled", False)
        self.declare_parameter("distractor_position", DEFAULT_DISTRACTOR_POSITION)

        self.ign_cmd = str(self.get_parameter("ign_cmd").value)
        self.pose_service = str(self.get_parameter("pose_service").value)
        self.pose_reqtype = str(self.get_parameter("pose_reqtype").value)
        self.pose_reptype = str(self.get_parameter("pose_reptype").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.max_retries = int(self.get_parameter("max_retries").value)

        self.initial_state: List[Dict[str, object]] | None = None
        self.latest_state: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.last_seed = 0

        self.publisher = self.create_publisher(String, "/episode/metadata", 10)
        self.create_subscription(String, "/scene/state_json", self.on_state, 10)
        self.create_service(Trigger, "/episode/reset", self.on_reset)
        self.create_service(Trigger, "/episode/step_end", self.on_step_end)
        self.get_logger().info("episode_manager_node started.")

    def on_state(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        objects = payload.get("objects", []) if isinstance(payload, dict) else []
        state: Dict[str, Dict[str, Dict[str, float]]] = {}
        parsed_objects = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            obj_id = obj.get("id")
            if not isinstance(obj_id, str):
                continue
            position = obj.get("position") if isinstance(obj.get("position"), dict) else {}
            orientation = obj.get("orientation") if isinstance(obj.get("orientation"), dict) else {}
            pos = {
                "x": float(position.get("x", 0.0)),
                "y": float(position.get("y", 0.0)),
                "z": float(position.get("z", 0.0)),
            }
            ori = {
                "x": float(orientation.get("x", 0.0)),
                "y": float(orientation.get("y", 0.0)),
                "z": float(orientation.get("z", 0.0)),
                "w": float(orientation.get("w", 1.0)),
            }
            state[obj_id] = {"position": pos, "orientation": ori}
            parsed_objects.append({"id": obj_id, "position": pos, "orientation": ori})
        if parsed_objects:
            self.latest_state = state
            if self.initial_state is None:
                self.initial_state = deepcopy(parsed_objects)

    def on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self.initial_state:
            response.success = False
            response.message = "no_initial_state"
            return response

        seed_param = int(self.get_parameter("random_seed").value)
        seed = seed_param if seed_param > 0 else int(time.time())
        rng = random.Random(seed)
        self.last_seed = seed

        pose_map = {obj["id"]: deepcopy(obj) for obj in self.initial_state}

        if bool(self.get_parameter("swap_similar").value):
            groups: Dict[str, List[str]] = {}
            for obj_id in pose_map:
                groups.setdefault(_group_key(obj_id), []).append(obj_id)
            for group_ids in groups.values():
                if len(group_ids) < 2:
                    continue
                positions = [pose_map[obj_id]["position"] for obj_id in group_ids]
                rng.shuffle(positions)
                for obj_id, pos in zip(group_ids, positions):
                    pose_map[obj_id]["position"] = pos

        risk_range = float(self.get_parameter("risk_offset_range").value)
        generic_range = float(self.get_parameter("generic_offset_range").value)
        for obj_id, entry in pose_map.items():
            offset = risk_range if _is_risky(obj_id) else generic_range
            if offset <= 0:
                continue
            entry["position"]["x"] += rng.uniform(-offset, offset)
            entry["position"]["y"] += rng.uniform(-offset, offset)

        if bool(self.get_parameter("distractor_enabled").value):
            distractor_pos = _parse_vector3(
                self.get_parameter("distractor_position").value,
                (0.0, 0.5, 0.8),
            )
            candidates = [obj_id for obj_id in pose_map if not _is_risky(obj_id)]
            if candidates:
                chosen = rng.choice(candidates)
                pose_map[chosen]["position"] = {
                    "x": distractor_pos[0],
                    "y": distractor_pos[1],
                    "z": distractor_pos[2],
                }

        failures = []
        for obj_id, entry in pose_map.items():
            req = build_pose_request(obj_id, entry["position"], entry["orientation"])
            ok, reason = call_set_pose(
                self.ign_cmd,
                self.pose_service,
                self.pose_reqtype,
                self.pose_reptype,
                req,
                self.command_timeout,
                self.max_retries,
            )
            if not ok:
                failures.append(f"{obj_id}:{reason}")

        self._publish_metadata(pose_map, seed)

        if failures:
            response.success = False
            response.message = "reset_failed:" + ",".join(failures[:3])
        else:
            response.success = True
            response.message = "reset_ok"
        return response

    def on_step_end(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = True
        response.message = "step_end"
        return response

    def _publish_metadata(self, pose_map: Dict[str, Dict[str, object]], seed: int) -> None:
        metadata = {
            "seed": seed,
            "objects": [{"id": obj_id, "risk": _is_risky(obj_id)} for obj_id in pose_map],
        }
        msg = String()
        msg.data = json.dumps(metadata, ensure_ascii=True)
        self.publisher.publish(msg)


def main() -> None:
    rclpy.init()
    node = EpisodeManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
