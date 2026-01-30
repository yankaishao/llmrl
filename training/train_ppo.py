import argparse
from pathlib import Path

import json
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from symbolic_env import ACTION_MAP, REWARD_CONFIG, SymbolicSafetyEnv


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "hri_safety_ws" / "policies" / "ppo_policy.zip"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--eval-episodes", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    def make_env():
        return SymbolicSafetyEnv(max_steps=args.max_steps, seed=args.seed)

    env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    model.learn(total_timesteps=args.total_timesteps)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    meta_path = save_path.with_suffix(".meta.json")
    meta = {
        "obs_dim": int(env.observation_space.shape[0]),
        "action_map": ACTION_MAP,
        "training_steps": args.total_timesteps,
        "seed": args.seed,
        "reward_config": REWARD_CONFIG,
        "env": {"max_steps": args.max_steps},
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes)
    print(f"Eval reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
