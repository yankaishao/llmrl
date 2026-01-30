import argparse
import csv
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from symbolic_env import REWARD_CONFIG, SymbolicSafetyEnv


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "hri_safety_ws" / "policies" / "ppo_policy.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--out-csv", type=str, default="eval_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = SymbolicSafetyEnv(max_steps=args.max_steps)
    model = PPO.load(args.policy_path)

    rows = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        total_cost = 0.0
        steps = 0
        success = 0.0
        violation = 0.0
        refused = 0.0
        queries = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(int(action))
            total_reward += float(reward)
            total_cost += float(info.get("cost", 0.0))
            steps += 1
            if done:
                success = float(info.get("success", 0.0))
                violation = float(info.get("violation", 0.0))
                refused = float(info.get("refused", 0.0))
                queries = float(info.get("queries", 0.0))
        rows.append({"reward": total_reward, "cost": total_cost, "steps": steps})
        rows[-1].update(
            {
                "success": success,
                "violation": violation,
                "refused": refused,
                "queries": queries,
            }
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reward", "cost", "steps", "success", "violation", "refused", "queries"],
        )
        writer.writeheader()
        writer.writerows(rows)

    avg_reward = np.mean([row["reward"] for row in rows])
    avg_cost = np.mean([row["cost"] for row in rows])
    avg_steps = np.mean([row["steps"] for row in rows])
    success_rate = np.mean([row["success"] for row in rows])
    violation_rate = np.mean([row["violation"] for row in rows])
    refusal_rate = np.mean([row["refused"] for row in rows])
    avg_queries = np.mean([row["queries"] for row in rows])
    print(
        "Avg reward={:.2f}, cost={:.2f}, steps={:.2f}, success_rate={:.2f}, "
        "violation_rate={:.2f}, refusal_rate={:.2f}, avg_queries={:.2f}".format(
            avg_reward,
            avg_cost,
            avg_steps,
            success_rate,
            violation_rate,
            refusal_rate,
            avg_queries,
        )
    )


if __name__ == "__main__":
    main()
