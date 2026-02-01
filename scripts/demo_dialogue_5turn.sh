#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source /opt/ros/humble/setup.bash

cd "$ROOT_DIR/hri_safety_ws"
colcon build
source install/local_setup.bash
cd "$ROOT_DIR"

cleanup() {
  if [[ -n "${LAUNCH_PID:-}" ]]; then
    kill "$LAUNCH_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

ros2 launch hri_safety_core pipeline_full.launch.py \
  use_dialogue_manager:=true use_age_context:=true parser_mode:=mock arbiter_mode:=rule &
LAUNCH_PID=$!

sleep 3
ros2 topic pub -1 /user/instruction std_msgs/msg/String "{data: 'bring the cup'}" >/dev/null
sleep 2
ros2 topic pub -1 /user/instruction std_msgs/msg/String "{data: 'left one'}" >/dev/null
sleep 2

ros2 topic pub -1 /user/age_context std_msgs/msg/String "{data: '{\"p_minor\":0.9,\"p_adult\":0.1,\"p_older\":0.0,\"age_conf\":0.9,\"guardian_present\":false,\"source\":\"manual\"}'}" >/dev/null
sleep 1
ros2 topic pub -1 /user/instruction std_msgs/msg/String "{data: 'pick up the knife'}" >/dev/null
sleep 2
ros2 topic echo -n 1 /arbiter/action
