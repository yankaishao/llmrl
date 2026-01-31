#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  :
}
trap cleanup INT TERM EXIT

# Full 2x2 matrix (mock/qwen x rule/rl) with multi-seed support.
# Usage examples:
#   ./evaluation/run_matrix_full.sh --episodes 12 --results-dir results/full --seeds 0,1,2
#   ./evaluation/run_matrix_full.sh --episodes 12 --results-dir results/full --seeds 0,1,2 --policy-path hri_safety_ws/policies/ppo_policy.zip

usage() {
  cat <<'USAGE'
Usage: ./evaluation/run_matrix_full.sh [options]

Options:
  --episodes N           Episodes per cell (default: 12).
  --results-dir PATH     Output root (default: results/full_matrix).
  --instructions PATH    Instruction list file (default: instructions/base.txt).
  --seeds CSV            Comma-separated seeds (default: 0).
  --policy-path PATH     Use existing policy (skip training). If omitted, train per seed.
  --train-steps N        PPO timesteps when training (default: 10000).
  --noise-scale X        Training noise scale (default: 0.0).
  --noise-amb X          Training amb noise (default: 0.0).
  --noise-risk X         Training risk noise (default: 0.0).
  --noise-conflict X     Training conflict flip prob (default: 0.0).
  --max-turns-per-episode N  Max interactive turns per episode (default: 10).
  --max-repeat-action N      Max repeats of same query action (default: 3).
  --point-response TEXT      Simulated response for ASK_POINT.
  --clarify-left TEXT        Simulated response for left disambiguation.
  --clarify-right TEXT       Simulated response for right disambiguation.
  --clarify-default TEXT     Simulated response when unclear.
  --model NAME           Qwen model name.
  --base-url URL         Qwen base URL.
  --api-key-env VAR      Env var holding Qwen API key.
  -h, --help             Show this help text.
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="$REPO_ROOT/hri_safety_ws"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

EPISODES=12
RESULTS_DIR="$REPO_ROOT/results/full_matrix"
INSTRUCTIONS="$REPO_ROOT/instructions/base.txt"
SEEDS="0"
POLICY_PATH=""
TRAIN_STEPS=10000
NOISE_SCALE=0.0
NOISE_AMB=0.0
NOISE_RISK=0.0
NOISE_CONFLICT=0.0
MAX_TURNS_PER_EPISODE=10
MAX_REPEAT_ACTION=3
POINT_RESPONSE="the left cup"
CLARIFY_LEFT="the left cup"
CLARIFY_RIGHT="the right cup"
CLARIFY_DEFAULT="the left cup"
MODEL="qwen3-max"
BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV="QWEN_API_KEY"
PARSER_TIMEOUT_SEC=10.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --episodes)
      EPISODES="$2"
      shift 2
      ;;
    --results-dir)
      RESULTS_DIR="$2"
      shift 2
      ;;
    --instructions)
      INSTRUCTIONS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --policy-path)
      POLICY_PATH="$2"
      shift 2
      ;;
    --train-steps)
      TRAIN_STEPS="$2"
      shift 2
      ;;
    --noise-scale)
      NOISE_SCALE="$2"
      shift 2
      ;;
    --noise-amb)
      NOISE_AMB="$2"
      shift 2
      ;;
    --noise-risk)
      NOISE_RISK="$2"
      shift 2
      ;;
    --noise-conflict)
      NOISE_CONFLICT="$2"
      shift 2
      ;;
    --max-turns-per-episode)
      MAX_TURNS_PER_EPISODE="$2"
      shift 2
      ;;
    --max-repeat-action)
      MAX_REPEAT_ACTION="$2"
      shift 2
      ;;
    --point-response)
      POINT_RESPONSE="$2"
      shift 2
      ;;
    --clarify-left)
      CLARIFY_LEFT="$2"
      shift 2
      ;;
    --clarify-right)
      CLARIFY_RIGHT="$2"
      shift 2
      ;;
    --clarify-default)
      CLARIFY_DEFAULT="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --api-key-env)
      API_KEY_ENV="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR"
fi
if [[ "$INSTRUCTIONS" != /* ]]; then
  INSTRUCTIONS="$REPO_ROOT/$INSTRUCTIONS"
fi
if [[ -n "$POLICY_PATH" ]] && [[ "$POLICY_PATH" != /* ]]; then
  POLICY_PATH="$REPO_ROOT/$POLICY_PATH"
fi
if [[ -n "$POLICY_PATH" ]] && [[ "$POLICY_PATH" != "random" ]] && [[ ! -f "$POLICY_PATH" ]]; then
  echo "policy-path not found: $POLICY_PATH" >&2
  exit 1
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

mkdir -p "$RESULTS_DIR"

git_hash="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")"
instructions_hash="$(sha256sum "$INSTRUCTIONS" | awk '{print $1}')"
reward_config_json="$("$PYTHON_BIN" - <<PY
import json
import sys
sys.path.append(r"$REPO_ROOT/training")
from symbolic_env import REWARD_CONFIG
print(json.dumps(REWARD_CONFIG, sort_keys=True))
PY
)"

index_path="$RESULTS_DIR/index.json"

IFS=',' read -r -a seed_list <<< "$SEEDS"

cells=("mock_rule" "qwen_rule" "mock_rl" "qwen_rl")

for seed in "${seed_list[@]}"; do
  seed_dir="$RESULTS_DIR/seed_${seed}"
  mkdir -p "$seed_dir"

  policy_path="$POLICY_PATH"
  if [[ -z "$policy_path" ]]; then
    policy_dir="$seed_dir/policy"
    mkdir -p "$policy_dir"
    policy_path="$policy_dir/ppo_policy_seed_${seed}.zip"
    "$PYTHON_BIN" "$REPO_ROOT/training/train_ppo.py" \
      --total-timesteps "$TRAIN_STEPS" \
      --seed "$seed" \
      --save-path "$policy_path" \
      --noise-scale "$NOISE_SCALE" \
      --noise-amb "$NOISE_AMB" \
      --noise-risk "$NOISE_RISK" \
      --noise-conflict "$NOISE_CONFLICT"
  fi

  policy_sha=""
  if [[ -f "$policy_path" ]]; then
    policy_sha="$(sha256sum "$policy_path" | awk '{print $1}')"
  fi

  run_meta="$seed_dir/run_meta.json"
  cat > "$run_meta" <<JSON
{
  "git_commit": "${git_hash}",
  "seed": ${seed},
  "episodes": ${EPISODES},
  "max_turns_per_episode": ${MAX_TURNS_PER_EPISODE},
  "max_repeat_action": ${MAX_REPEAT_ACTION},
  "noise": {
    "noise_scale": ${NOISE_SCALE},
    "noise_amb": ${NOISE_AMB},
    "noise_risk": ${NOISE_RISK},
    "noise_conflict": ${NOISE_CONFLICT}
  },
  "reward_config": ${reward_config_json},
  "instructions_path": "${INSTRUCTIONS}",
  "instructions_sha256": "${instructions_hash}",
  "policy_path": "${policy_path}",
  "policy_sha256": "${policy_sha}",
  "parser_config": {
    "model": "${MODEL}",
    "base_url": "${BASE_URL}",
    "api_key_env": "${API_KEY_ENV}",
    "timeout_sec": ${PARSER_TIMEOUT_SEC}
  }
}
JSON

  for cell in "${cells[@]}"; do
    cell_dir="$seed_dir/$cell"
    mkdir -p "$cell_dir"
    parser_mode="${cell%%_*}"
    arbiter_mode="${cell##*_}"

    "$PYTHON_BIN" "$REPO_ROOT/evaluation/ros_eval_runner.py" \
      --matrix \
      --parser-modes "$parser_mode" \
      --arbiter-modes "$arbiter_mode" \
      --episodes "$EPISODES" \
      --results-dir "$cell_dir" \
      --instructions "$INSTRUCTIONS" \
      --launch-file pipeline_full.launch.py \
      --no-reset \
      --policy-path "$policy_path" \
      --max-turns-per-episode "$MAX_TURNS_PER_EPISODE" \
      --max-repeat-action "$MAX_REPEAT_ACTION" \
      --point-response "$POINT_RESPONSE" \
      --clarify-left "$CLARIFY_LEFT" \
      --clarify-right "$CLARIFY_RIGHT" \
      --clarify-default "$CLARIFY_DEFAULT" \
      --model "$MODEL" \
      --base-url "$BASE_URL" \
      --api-key-env "$API_KEY_ENV"
  done
done

cat > "$index_path" <<JSON
{
  "seeds": [$(printf '"%s",' "${seed_list[@]}" | sed 's/,$//')],
  "cells": ["mock_rule","qwen_rule","mock_rl","qwen_rl"],
  "root": "${RESULTS_DIR}"
}
JSON
