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

if [[ ! -f "$ROS_WS/install/setup.bash" ]]; then
  (cd "$ROS_WS" && colcon build --symlink-install)
fi

set +u
source "$ROS_WS/install/local_setup.bash"
set -u
if [[ -z "${AMENT_PREFIX_PATH:-}" ]] || [[ ":$AMENT_PREFIX_PATH:" != *":$ROS_WS/install/hri_safety_core:"* ]]; then
  export AMENT_PREFIX_PATH="$ROS_WS/install/hri_safety_core:${AMENT_PREFIX_PATH:-}"
fi

timestamp="$(date +%Y%m%d_%H%M%S)"
policy_path="$ROS_WS/policies/ppo_policy_sanity_${timestamp}.zip"
results_dir="$REPO_ROOT/results/sanity_${timestamp}"

echo "Training quick PPO policy..."
"$PYTHON_BIN" "$REPO_ROOT/training/train_ppo.py" \
  --total-timesteps 5000 \
  --seed 123 \
  --save-path "$policy_path" \
  --eval-episodes 10 \
  --eval-freq 500

echo "Running ROS integration sanity check (mock parser + RL arbiter)..."
"$PYTHON_BIN" "$REPO_ROOT/evaluation/ros_eval_runner.py" \
  --matrix \
  --parser-modes mock \
  --arbiter-modes rl \
  --launch-file pipeline_full.launch.py \
  --episodes 4 \
  --results-dir "$results_dir" \
  --policy-path "$policy_path" \
  --no-reset

echo "Sanity check results: $results_dir"
