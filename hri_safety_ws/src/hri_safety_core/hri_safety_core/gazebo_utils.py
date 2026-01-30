import shlex
import subprocess
import time
from typing import Dict, List, Optional, Tuple

DEFAULT_IGN_CMD = "ign"
DEFAULT_SET_POSE_SERVICE = "/world/default/set_pose"
DEFAULT_SET_POSE_REQTYPE = "gz.msgs.Pose"
DEFAULT_SET_POSE_REPTYPE = "gz.msgs.Boolean"
DEFAULT_TIMEOUT_SEC = 1.5
DEFAULT_MAX_RETRIES = 1


def list_ign_topics(ign_cmd: str, timeout_sec: float) -> List[str]:
    cmd = shlex.split(ign_cmd) if ign_cmd else [DEFAULT_IGN_CMD]
    cmd = cmd + ["topic", "-l"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def build_pose_request(name: str, position: Dict[str, object], orientation: Dict[str, object]) -> str:
    pos_x = _safe_float(position.get("x", 0.0))
    pos_y = _safe_float(position.get("y", 0.0))
    pos_z = _safe_float(position.get("z", 0.0))
    ori_x = _safe_float(orientation.get("x", 0.0))
    ori_y = _safe_float(orientation.get("y", 0.0))
    ori_z = _safe_float(orientation.get("z", 0.0))
    ori_w = _safe_float(orientation.get("w", 1.0))
    return (
        f'name: "{name}" '
        f"position {{ x: {pos_x:.3f} y: {pos_y:.3f} z: {pos_z:.3f} }} "
        f"orientation {{ x: {ori_x:.3f} y: {ori_y:.3f} z: {ori_z:.3f} w: {ori_w:.3f} }}"
    )


def call_set_pose(
    ign_cmd: str,
    service: str,
    reqtype: str,
    reptype: str,
    req: str,
    timeout_sec: float,
    max_retries: int,
) -> Tuple[bool, str]:
    cmd = shlex.split(ign_cmd) if ign_cmd else [DEFAULT_IGN_CMD]
    cmd = cmd + [
        "service",
        "-s",
        service,
        "--reqtype",
        reqtype,
        "--reptype",
        reptype,
        "--timeout",
        str(int(timeout_sec * 1000)),
        "--req",
        req,
    ]
    last_reason = ""
    for attempt in range(max_retries + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec + 0.5,
                check=False,
            )
            stdout = (result.stdout or "").lower()
            stderr = (result.stderr or "").strip()
            if result.returncode == 0 and ("true" in stdout or "success" in stdout):
                return True, ""
            last_reason = stderr or stdout or f"service_failed_attempt_{attempt}"
        except subprocess.TimeoutExpired:
            last_reason = "service_timeout"
        time.sleep(0.05)
    return False, last_reason
