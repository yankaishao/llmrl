#!/usr/bin/env bash
set -euo pipefail

# Run the 2x2 (mock/qwen x rule/rl) matrix evaluation in Gazebo.
# Usage examples:
#   ./evaluation/run_matrix_gazebo.sh --episodes 12 --results-dir results
#   QWEN_API_KEY=... ./evaluation/run_matrix_gazebo.sh --episodes 12 --results-dir results
#   ./evaluation/run_matrix_gazebo.sh --episodes 12 --results-dir results --policy-path hri_safety_ws/policies/ppo_policy.zip

usage() {
  cat <<'USAGE'
Usage: ./evaluation/run_matrix_gazebo.sh [options]

Options:
  --episodes N           Repeat instructions to reach N episodes (default: 0 = use base list).
  --results-dir PATH     Output directory for CSVs (default: results).
  --instructions PATH    Instruction list file (default: instructions/base.txt).
  --policy-path PATH     PPO policy path (default: hri_safety_ws/policies/ppo_policy.zip).
  --launch-file NAME     Launch file to use (default: pipeline_gazebo.launch.py).
  --launch-wait SEC      Seconds to wait for launch (default: 6.0).
  --headless             Launch Gazebo Sim without GUI.
  --timeout SEC          Per-episode timeout (default: 2.0).
  --reset-timeout SEC    Reset service timeout (default: 2.0).
  --no-reset             Skip /episode/reset calls.
  --max-turns-per-episode N  Max interactive turns per episode (default: 10).
  --max-repeat-action N      Max repeats of same query action (default: 3).
  --point-response TEXT      Simulated response for ASK_POINT.
  --clarify-left TEXT        Simulated response for left disambiguation.
  --clarify-right TEXT       Simulated response for right disambiguation.
  --clarify-default TEXT     Simulated response when unclear.
  --matrix-parser-only   Run only mock+rule and qwen+rule combinations.
  --parser-modes CSV     Override parser modes (default: mock,qwen).
  --arbiter-modes CSV    Override arbiter modes (default: rule,rl).
  --model NAME           Qwen model name.
  --base-url URL         Qwen base URL.
  --api-key-env VAR      Env var holding Qwen API key.
  -h, --help             Show this help text.
USAGE
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_WS="$REPO_ROOT/hri_safety_ws"

EPISODES=0
RESULTS_DIR="$REPO_ROOT/results"
INSTRUCTIONS="$REPO_ROOT/instructions/base.txt"
POLICY_PATH="$ROS_WS/policies/ppo_policy.zip"
LAUNCH_FILE="pipeline_gazebo.launch.py"
LAUNCH_WAIT="6.0"
HEADLESS=0
TIMEOUT="2.0"
RESET_TIMEOUT="2.0"
NO_RESET=0
MATRIX_PARSER_ONLY=0
MAX_TURNS_PER_EPISODE=10
MAX_REPEAT_ACTION=3
POINT_RESPONSE="the left cup"
CLARIFY_LEFT="the left cup"
CLARIFY_RIGHT="the right cup"
CLARIFY_DEFAULT="the left cup"
PARSER_MODES=""
ARBITER_MODES=""
MODEL="qwen3-max"
BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY_ENV="QWEN_API_KEY"
EXTRA_ARGS=()
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
    --policy-path)
      POLICY_PATH="$2"
      shift 2
      ;;
    --launch-file)
      LAUNCH_FILE="$2"
      shift 2
      ;;
    --launch-wait)
      LAUNCH_WAIT="$2"
      shift 2
      ;;
    --headless)
      HEADLESS=1
      shift
      ;;
    --timeout)
      TIMEOUT="$2"
      shift 2
      ;;
    --reset-timeout)
      RESET_TIMEOUT="$2"
      shift 2
      ;;
    --no-reset)
      NO_RESET=1
      shift
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
    --matrix-parser-only)
      MATRIX_PARSER_ONLY=1
      shift
      ;;
    --parser-modes)
      PARSER_MODES="$2"
      shift 2
      ;;
    --arbiter-modes)
      ARBITER_MODES="$2"
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
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$RESULTS_DIR" != /* ]]; then
  RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR"
fi
if [[ "$INSTRUCTIONS" != /* ]]; then
  INSTRUCTIONS="$REPO_ROOT/$INSTRUCTIONS"
fi
if [[ "$POLICY_PATH" != /* ]]; then
  POLICY_PATH="$REPO_ROOT/$POLICY_PATH"
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
source "$ROS_WS/install/setup.bash"
set -u
if [[ -z "${AMENT_PREFIX_PATH:-}" ]] || [[ ":$AMENT_PREFIX_PATH:" != *":$ROS_WS/install/hri_safety_core:"* ]]; then
  export AMENT_PREFIX_PATH="$ROS_WS/install/hri_safety_core:${AMENT_PREFIX_PATH:-}"
fi

mkdir -p "$RESULTS_DIR"

cmd=(
  "$PYTHON_BIN" "$REPO_ROOT/evaluation/ros_eval_runner.py"
  --matrix
  --episodes "$EPISODES"
  --results-dir "$RESULTS_DIR"
  --instructions "$INSTRUCTIONS"
  --policy-path "$POLICY_PATH"
  --launch-file "$LAUNCH_FILE"
  --launch-wait "$LAUNCH_WAIT"
  --timeout "$TIMEOUT"
  --reset-timeout "$RESET_TIMEOUT"
  --max-turns-per-episode "$MAX_TURNS_PER_EPISODE"
  --max-repeat-action "$MAX_REPEAT_ACTION"
  --point-response "$POINT_RESPONSE"
  --clarify-left "$CLARIFY_LEFT"
  --clarify-right "$CLARIFY_RIGHT"
  --clarify-default "$CLARIFY_DEFAULT"
  --model "$MODEL"
  --base-url "$BASE_URL"
  --api-key-env "$API_KEY_ENV"
)

if [[ "$NO_RESET" -eq 1 ]]; then
  cmd+=(--no-reset)
fi
if [[ "$HEADLESS" -eq 1 ]]; then
  cmd+=(--headless)
fi
if [[ "$MATRIX_PARSER_ONLY" -eq 1 ]]; then
  cmd+=(--matrix-parser-only)
fi
if [[ -n "$PARSER_MODES" ]]; then
  cmd+=(--parser-modes "$PARSER_MODES")
fi
if [[ -n "$ARBITER_MODES" ]]; then
  cmd+=(--arbiter-modes "$ARBITER_MODES")
fi
cmd+=("${EXTRA_ARGS[@]}")

EVAL_PID=""
USE_PG=0

cleanup() {
  if [[ -n "${EVAL_PID}" ]] && kill -0 "$EVAL_PID" 2>/dev/null; then
    if [[ "$USE_PG" -eq 1 ]]; then
      kill -TERM "-$EVAL_PID" 2>/dev/null || true
    else
      kill -TERM "$EVAL_PID" 2>/dev/null || true
    fi
    wait "$EVAL_PID" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

if command -v setsid >/dev/null 2>&1; then
  setsid "${cmd[@]}" &
  EVAL_PID=$!
  USE_PG=1
else
  "${cmd[@]}" &
  EVAL_PID=$!
fi

wait "$EVAL_PID"
