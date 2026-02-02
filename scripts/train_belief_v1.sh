#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY_DIR="$ROOT_DIR/hri_safety_ws/policies"
LOG_DIR="$ROOT_DIR/results/belief_v1_train"

TIMESTEPS=2000000
MAX_STEPS=5
N_ENVS=8
SEEDS="0,1,2"
EVAL_EPISODES=200
EVAL_FREQ=20000

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timesteps)
      TIMESTEPS="$2"
      shift 2
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --n-envs)
      N_ENVS="$2"
      shift 2
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --eval-episodes)
      EVAL_EPISODES="$2"
      shift 2
      ;;
    --eval-freq)
      EVAL_FREQ="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 1
      ;;
  esac
 done

mkdir -p "$POLICY_DIR" "$LOG_DIR"
SUMMARY_CSV="$LOG_DIR/summary.csv"
echo "seed,mean_reward,violation_rate,query_rate,policy_path,best_policy_path" > "$SUMMARY_CSV"

IFS=',' read -r -a SEED_LIST <<< "$SEEDS"

for seed in "${SEED_LIST[@]}"; do
  seed_trimmed="${seed// /}"
  [[ -z "$seed_trimmed" ]] && continue
  save_path="$POLICY_DIR/ppo_policy_belief_v1_seed${seed_trimmed}.zip"
  log_path="$LOG_DIR/train_seed_${seed_trimmed}.log"

  echo "=== Training seed ${seed_trimmed} ==="
  python "$ROOT_DIR/training/train_ppo.py" \
    --env safety \
    --obs-mode belief_v1 \
    --total-timesteps "$TIMESTEPS" \
    --max-steps "$MAX_STEPS" \
    --n-envs "$N_ENVS" \
    --seed "$seed_trimmed" \
    --save-path "$save_path" \
    --eval-episodes "$EVAL_EPISODES" \
    --eval-freq "$EVAL_FREQ" \
    | tee "$log_path"

  python - <<PY
import re, csv
from pathlib import Path
log_path = Path("$log_path")
text = log_path.read_text(encoding="utf-8", errors="ignore")
pattern = re.compile(r"Eval reward: ([\-\d\.]+) .*?\| query_rate=([\-\d\.]+) \| violation_rate=([\-\d\.]+)")
matches = pattern.findall(text)
if not matches:
    raise SystemExit("eval_stats_not_found")
mean_reward, query_rate, violation_rate = matches[-1]
summary_path = Path("$SUMMARY_CSV")
with summary_path.open("a", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow([
        "$seed_trimmed",
        float(mean_reward),
        float(violation_rate),
        float(query_rate),
        "$save_path",
        str(Path("$save_path").with_name(f"best_{Path('$save_path').name}")),
    ])
PY

done

python - <<'PY'
import csv
from pathlib import Path
summary_path = Path("$SUMMARY_CSV")
rows = []
with summary_path.open("r", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        rows.append(row)
if not rows:
    raise SystemExit("no_rows")

def key(row):
    return (-float(row["mean_reward"]), float(row["violation_rate"]))

best = sorted(rows, key=key)[0]
policy_path = Path(best["policy_path"])
best_policy_path = Path(best["best_policy_path"])
if best_policy_path.is_file():
    chosen = best_policy_path
else:
    chosen = policy_path

policy_dir = policy_path.parent
best_out = policy_dir / "best_ppo_policy.zip"
meta_src = policy_path.with_suffix(".meta.json")
config_src = policy_path.with_suffix(".config.json")
meta_out = policy_dir / "best_ppo_policy.meta.json"
config_out = policy_dir / "best_ppo_policy.config.json"

best_out.write_bytes(chosen.read_bytes())
if meta_src.is_file():
    meta_out.write_bytes(meta_src.read_bytes())
if config_src.is_file():
    config_out.write_bytes(config_src.read_bytes())

print("Selected seed:", best["seed"])
print("Mean reward:", best["mean_reward"], "Violation rate:", best["violation_rate"], "Query rate:", best["query_rate"])
print("Copied policy:", chosen)
print("Output ->", best_out)
PY
