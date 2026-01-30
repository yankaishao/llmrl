import json
import re
import shlex
import subprocess
from typing import Dict, List, Optional, Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


DEFAULT_IGN_CMD = "ign"
DEFAULT_IGN_TOPIC = "/world/default/pose/info"
DEFAULT_PUBLISH_HZ = 2.0
DEFAULT_COMMAND_TIMEOUT = 2.0
DEFAULT_SUMMARY_CONTEXT = "context=[child_present=false], task_state=idle"
DEFAULT_MAX_SUMMARY_CHARS = 1000
DEFAULT_OBJECT_NAMES = ["cup_red_1", "cup_red_2", "knife_1"]

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _extract_blocks(text: str, keyword: str) -> List[str]:
    blocks = []
    idx = 0
    while True:
        idx = text.find(keyword, idx)
        if idx == -1:
            break
        brace_idx = text.find("{", idx)
        if brace_idx == -1:
            break
        depth = 0
        j = brace_idx
        while j < len(text):
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                blocks.append(text[brace_idx + 1 : j])
                idx = j + 1
                break
            j += 1
        else:
            break
    return blocks


def _find_string(block: str, key: str) -> Optional[str]:
    match = re.search(rf"\\b{key}\\b\\s*:\\s*\"([^\"]+)\"", block)
    if match:
        return match.group(1)
    return None


def _find_float(block: str, key: str) -> Optional[float]:
    match = re.search(rf"\\b{key}\\b\\s*:\\s*({FLOAT_RE})", block)
    if match:
        return float(match.group(1))
    return None


def _find_block(block: str, key: str) -> Optional[str]:
    match = re.search(rf"{key}\\s*\\{{(.*?)\\}}", block, re.DOTALL)
    if match:
        return match.group(1)
    return None


def _parse_vector(block: str, key: str, fields: List[str]) -> Dict[str, float]:
    inner = _find_block(block, key)
    if not inner:
        return {}
    values: Dict[str, float] = {}
    for field in fields:
        value = _find_float(inner, field)
        if value is not None:
            values[field] = value
    return values


def parse_pose_info(text: str) -> List[Dict[str, object]]:
    blocks = _extract_blocks(text, "poses")
    if not blocks:
        blocks = _extract_blocks(text, "pose")
    objects: List[Dict[str, object]] = []
    for block in blocks:
        name = _find_string(block, "name")
        if not name:
            continue
        obj_id = _find_float(block, "id")
        position = _parse_vector(block, "position", ["x", "y", "z"])
        orientation = _parse_vector(block, "orientation", ["x", "y", "z", "w"])
        obj = {"id": name}
        if obj_id is not None:
            obj["entity_id"] = int(obj_id)
        if position:
            obj["position"] = position
        if orientation:
            obj["orientation"] = orientation
        objects.append(obj)
    return objects


def _parse_object_names(value: object, fallback: Sequence[str]) -> List[str]:
    if isinstance(value, (list, tuple)):
        names = [str(item).strip() for item in value if str(item).strip()]
        return names or list(fallback)
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",") if part.strip()]
        return names or list(fallback)
    return list(fallback)


def _extract_pose_xyz(obj: Dict[str, object]) -> Optional[Dict[str, float]]:
    position = obj.get("position")
    if not isinstance(position, dict):
        return None
    try:
        return {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def build_scene_summary(
    objects_by_name: Dict[str, Dict[str, object]],
    object_names: Sequence[str],
    context: str,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        max_chars = DEFAULT_MAX_SUMMARY_CHARS
    parts = []
    for name in object_names:
        obj = objects_by_name.get(name)
        if not obj:
            parts.append(f"{name}(missing)")
            continue
        pose = _extract_pose_xyz(obj)
        if not pose:
            parts.append(f"{name}(missing)")
            continue
        parts.append(
            f"{name}(x={pose['x']:.2f},y={pose['y']:.2f},z={pose['z']:.2f})"
        )
    summary = f"objects=[{', '.join(parts)}], {context}"
    if len(summary) > max_chars:
        truncated = summary[: max_chars - 3].rstrip()
        summary = truncated + "..."
    return summary


class GazeboWorldStateNode(Node):
    def __init__(self) -> None:
        super().__init__("gazebo_world_state_node")
        self.declare_parameter("ign_cmd", DEFAULT_IGN_CMD)
        self.declare_parameter("ign_topic", DEFAULT_IGN_TOPIC)
        self.declare_parameter("publish_hz", DEFAULT_PUBLISH_HZ)
        self.declare_parameter("command_timeout", DEFAULT_COMMAND_TIMEOUT)
        self.declare_parameter("summary_context", DEFAULT_SUMMARY_CONTEXT)
        self.declare_parameter("max_summary_chars", DEFAULT_MAX_SUMMARY_CHARS)
        self.declare_parameter("object_names", ",".join(DEFAULT_OBJECT_NAMES))

        self.publisher_state = self.create_publisher(String, "/scene/state_json", 10)
        self.publisher_summary = self.create_publisher(String, "/scene/summary", 10)

        self.ign_cmd = str(self.get_parameter("ign_cmd").value)
        self.ign_topic = str(self.get_parameter("ign_topic").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.summary_context = str(self.get_parameter("summary_context").value)
        self.max_summary_chars = int(self.get_parameter("max_summary_chars").value)
        self.object_names = _parse_object_names(
            self.get_parameter("object_names").value,
            DEFAULT_OBJECT_NAMES,
        )

        if not self.ign_topic or self.ign_topic.lower() == "auto":
            self.ign_topic = self._auto_detect_pose_topic() or DEFAULT_IGN_TOPIC

        publish_hz = float(self.get_parameter("publish_hz").value)
        publish_hz = publish_hz if publish_hz > 0 else DEFAULT_PUBLISH_HZ
        self.timer = self.create_timer(1.0 / publish_hz, self._tick)

        self.get_logger().info(
            f"gazebo_world_state_node started ign_cmd={self.ign_cmd} topic={self.ign_topic}"
        )

    def _run_ign_echo(self) -> str:
        cmd = shlex.split(self.ign_cmd)
        if not cmd:
            cmd = [DEFAULT_IGN_CMD]
        cmd = cmd + ["topic", "-e", "-t", self.ign_topic, "-n", "1"]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.command_timeout,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise RuntimeError(stderr or "ign_topic_failed")
        return result.stdout

    def _auto_detect_pose_topic(self) -> Optional[str]:
        cmd = shlex.split(self.ign_cmd)
        if not cmd:
            cmd = [DEFAULT_IGN_CMD]
        cmd = cmd + ["topic", "-l"]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.command_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        topics = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        pose_topics = [topic for topic in topics if "pose" in topic]
        for topic in pose_topics:
            if "/world/" in topic and "pose" in topic and "info" in topic:
                return topic
        if pose_topics:
            return pose_topics[0]
        return None

    def _tick(self) -> None:
        try:
            text = self._run_ign_echo()
            objects = parse_pose_info(text)
            objects_by_name = {}
            for obj in objects:
                obj_id = str(obj.get("id", "")).strip()
                if obj_id:
                    objects_by_name[obj_id] = obj

            payload_objects = []
            for name in self.object_names:
                obj = objects_by_name.get(name)
                if not obj:
                    continue
                pose = _extract_pose_xyz(obj)
                if not pose:
                    continue
                payload_objects.append({"id": name, "pose": pose})

            state = {"objects": payload_objects}
            state_msg = String()
            state_msg.data = json.dumps(state, ensure_ascii=True)
            self.publisher_state.publish(state_msg)

            summary = build_scene_summary(
                objects_by_name,
                self.object_names,
                self.summary_context,
                max_chars=self.max_summary_chars,
            )
            summary_msg = String()
            summary_msg.data = summary
            self.publisher_summary.publish(summary_msg)
        except Exception as exc:
            self.get_logger().warning(f"Failed to read ign topic: {exc}")


def main() -> None:
    rclpy.init()
    node = GazeboWorldStateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
