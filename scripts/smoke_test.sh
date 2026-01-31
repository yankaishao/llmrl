#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="$REPO_ROOT/hri_safety_ws"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "Missing /opt/ros/humble/setup.bash. Install ROS 2 Humble first." >&2
  exit 1
fi

set +u
source /opt/ros/humble/setup.bash
set -u

(cd "$ROS_WS" && colcon build --symlink-install)

set +u
source "$ROS_WS/install/setup.bash"
set -u
if [[ -z "${AMENT_PREFIX_PATH:-}" ]] || [[ ":$AMENT_PREFIX_PATH:" != *":$ROS_WS/install/hri_safety_core:"* ]]; then
  export AMENT_PREFIX_PATH="$ROS_WS/install/hri_safety_core:${AMENT_PREFIX_PATH:-}"
fi

execs="$(ros2 pkg executables hri_safety_core)"

match_exec() {
  local needle="$1"
  if command -v rg >/dev/null 2>&1; then
    rg -q "(^|\\s)${needle}(\\s|$)" <<<"$execs"
  else
    grep -qw "$needle" <<<"$execs"
  fi
}

required_nodes=(
  "instruction_source"
  "parser_router"
  "arbiter_router"
  "estimator_node"
  "gazebo_world_state_node"
  "episode_manager_node"
)

for node in "${required_nodes[@]}"; do
  if ! match_exec "$node"; then
    echo "Missing executable: $node" >&2
    exit 1
  fi
done

"$PYTHON_BIN" "$REPO_ROOT/evaluation/ros_eval_runner.py" --help >/dev/null
"$PYTHON_BIN" "$REPO_ROOT/evaluation/make_report.py" --help >/dev/null
