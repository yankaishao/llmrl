import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from symbolic_env import (
    ACTION_CONFIRM,
    ACTION_CHOICE,
    ACTION_MAP,
    ACTION_POINT,
    DIALOGUE_CONFIG,
    FeatureNoiseWrapper,
    REWARD_CONFIG,
    SymbolicBeliefV1Env,
    SymbolicDialogueEnv,
    SymbolicSafetyEnv,
)


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "hri_safety_ws" / "policies" / "ppo_policy.zip"
)
OBS_FIELDS = {
    "legacy": ["amb", "risk", "conflict", "query_count", "last_outcome"],
    "belief_v1": [
        "amb",
        "risk",
        "conflict",
        "p_top",
        "margin",
        "entropy",
        "missing_slots_count",
        "query_count",
        "last_outcome",
        "p_minor",
        "age_conf",
        "guardian_code",
        "hazard_top",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="safety", choices=["safety", "dialogue"])
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--save-path", type=str, default=str(DEFAULT_POLICY_PATH))
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--noise-amb", type=float, default=0.0)
    parser.add_argument("--noise-risk", type=float, default=0.0)
    parser.add_argument("--noise-conflict", type=float, default=0.0)
    parser.add_argument("--obs-mode", type=str, default="legacy")
    parser.add_argument("--n-envs", type=int, default=8)
    return parser.parse_args()


def _evaluate_with_stats(model, env, n_eval_episodes: int) -> dict:
    query_actions = {ACTION_CONFIRM, ACTION_CHOICE, ACTION_POINT}
    episode_rewards = []
    episode_lengths = []
    total_steps = 0
    query_steps = 0
    violations = 0
    high_amb_steps = 0
    high_amb_queries = 0
    low_amb_steps = 0
    low_amb_queries = 0

    for _ in range(n_eval_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs, _info = obs
        if isinstance(obs, np.ndarray) and obs.ndim == 2:
            initial_amb = float(obs[0][0])
        else:
            initial_amb = float(obs[0])
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            act = int(action[0]) if isinstance(action, np.ndarray) else int(action)
            obs, reward, done, info = env.step(action)

            if isinstance(done, np.ndarray):
                done_flag = bool(done[0])
                info0 = info[0] if isinstance(info, list) else info
            else:
                done_flag = bool(done)
                info0 = info

            ep_reward += float(reward[0]) if isinstance(reward, np.ndarray) else float(reward)
            ep_len += 1
            total_steps += 1

            if act in query_actions:
                query_steps += 1
                if initial_amb >= 0.6:
                    high_amb_queries += 1
                if initial_amb <= 0.3:
                    low_amb_queries += 1

            if initial_amb >= 0.6:
                high_amb_steps += 1
            if initial_amb <= 0.3:
                low_amb_steps += 1

            if done_flag:
                violations += int(info0.get("violation", 0)) if isinstance(info0, dict) else 0
            done = done_flag

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_len)

    mean_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0
    std_reward = float(np.std(episode_rewards)) if episode_rewards else 0.0
    mean_len = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    query_rate = float(query_steps / total_steps) if total_steps > 0 else 0.0
    violation_rate = float(violations / len(episode_rewards)) if episode_rewards else 0.0
    high_query_rate = float(high_amb_queries / high_amb_steps) if high_amb_steps > 0 else 0.0
    low_query_rate = float(low_amb_queries / low_amb_steps) if low_amb_steps > 0 else 0.0
    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_ep_length": mean_len,
        "query_rate": query_rate,
        "violation_rate": violation_rate,
        "high_amb_query_rate": high_query_rate,
        "low_amb_query_rate": low_query_rate,
    }


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

    def make_env_fn(rank: int):
        def _init():
            seed = args.seed + rank
            if args.obs_mode == "belief_v1":
                return SymbolicBeliefV1Env(max_steps=args.max_steps, seed=seed)
            if args.env == "dialogue":
                return SymbolicDialogueEnv(max_steps=args.max_steps, seed=seed)

            base_env = SymbolicSafetyEnv(max_steps=args.max_steps, seed=seed)
            if noise_amb > 0.0 or noise_risk > 0.0 or noise_conflict > 0.0:
                return FeatureNoiseWrapper(
                    base_env,
                    noise_amb=noise_amb,
                    noise_risk=noise_risk,
                    noise_conflict=noise_conflict,
                    seed=seed,
                )
            return base_env

        return _init

    n_envs = max(1, int(args.n_envs))
    if n_envs > 1:
        try:
            env = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
        except Exception:
            env = DummyVecEnv([make_env_fn(i) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env_fn(0)])

    eval_env = DummyVecEnv([make_env_fn(10_000)])

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
    env_name = "belief_v1" if args.obs_mode == "belief_v1" else args.env
    config = {
        "timestamp": timestamp,
        "env_name": env_name,
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
        "obs_mode": args.obs_mode,
        "obs_fields": OBS_FIELDS.get(args.obs_mode, []),
        "reward_config": REWARD_CONFIG,
        "env": {"max_steps": args.max_steps, "n_envs": n_envs},
        "noise": {
            "noise_scale": args.noise_scale,
            "noise_amb": noise_amb,
            "noise_risk": noise_risk,
            "noise_conflict": noise_conflict,
        },
    }
    if args.env == "dialogue":
        config["env_config"] = DIALOGUE_CONFIG
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    meta_path = save_path.with_suffix(".meta.json")
    meta = {
        "obs_dim": int(env.observation_space.shape[0]),
        "obs_mode": args.obs_mode,
        "obs_fields": OBS_FIELDS.get(args.obs_mode, []),
        "action_map": ACTION_MAP,
        "training_steps": args.total_timesteps,
        "seed": args.seed,
        "reward_config": REWARD_CONFIG,
        "env_name": env_name,
        "env": {"max_steps": args.max_steps, "n_envs": n_envs},
        "timestamp": timestamp,
    }
    if args.env == "dialogue":
        meta["env_config"] = DIALOGUE_CONFIG
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    stats = _evaluate_with_stats(model, eval_env, n_eval_episodes=args.eval_episodes)
    print(
        "Eval reward: {mean:.2f} +/- {std:.2f} | mean_ep_length={length:.2f} "
        "| query_rate={qr:.2f} | violation_rate={vr:.2f} "
        "| high_amb_query_rate={hqr:.2f} | low_amb_query_rate={lqr:.2f}".format(
            mean=stats["mean_reward"],
            std=stats["std_reward"],
            length=stats["mean_ep_length"],
            qr=stats["query_rate"],
            vr=stats["violation_rate"],
            hqr=stats["high_amb_query_rate"],
            lqr=stats["low_amb_query_rate"],
        )
    )


if __name__ == "__main__":
    main()
