import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from symbolic_env import ACTION_MAP, REWARD_CONFIG, FeatureNoiseWrapper, SymbolicSafetyEnv


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
    parser.add_argument("--eval-freq", type=int, default=1000)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--noise-amb", type=float, default=0.0)
    parser.add_argument("--noise-risk", type=float, default=0.0)
    parser.add_argument("--noise-conflict", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    set_random_seed(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    noise_amb = args.noise_amb
    noise_risk = args.noise_risk
    noise_conflict = args.noise_conflict
    if args.noise_scale > 0.0 and noise_amb == 0.0 and noise_risk == 0.0 and noise_conflict == 0.0:
        noise_amb = args.noise_scale
        noise_risk = args.noise_scale
        noise_conflict = min(0.3, args.noise_scale)

    def make_env():
        base_env = SymbolicSafetyEnv(max_steps=args.max_steps, seed=args.seed)
        if noise_amb > 0.0 or noise_risk > 0.0 or noise_conflict > 0.0:
            return FeatureNoiseWrapper(
                base_env,
                noise_amb=noise_amb,
                noise_risk=noise_risk,
                noise_conflict=noise_conflict,
                seed=args.seed,
            )
        return base_env

    env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    best_dir = Path(args.save_path).resolve().parent
    best_dir.mkdir(parents=True, exist_ok=True)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=args.eval_episodes,
        eval_freq=max(1, args.eval_freq),
        best_model_save_path=str(best_dir),
        deterministic=True,
        verbose=0,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))

    best_model_path = best_dir / "best_model.zip"
    best_policy_path = save_path.with_name(f"best_{save_path.name}")
    if best_model_path.is_file():
        best_policy_path.write_bytes(best_model_path.read_bytes())

    config_path = save_path.with_suffix(".config.json")
    config = {
        "timestamp": timestamp,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
        "reward_config": REWARD_CONFIG,
        "env": {"max_steps": args.max_steps},
        "noise": {
            "noise_scale": args.noise_scale,
            "noise_amb": noise_amb,
            "noise_risk": noise_risk,
            "noise_conflict": noise_conflict,
        },
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    meta_path = save_path.with_suffix(".meta.json")
    meta = {
        "obs_dim": int(env.observation_space.shape[0]),
        "action_map": ACTION_MAP,
        "training_steps": args.total_timesteps,
        "seed": args.seed,
        "reward_config": REWARD_CONFIG,
        "env": {"max_steps": args.max_steps},
        "timestamp": timestamp,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=args.eval_episodes)
    print(f"Eval reward: {mean_reward:.2f} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
