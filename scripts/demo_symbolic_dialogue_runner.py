#!/usr/bin/env python3
import argparse
from typing import Callable, List, Tuple

import numpy as np

from training.symbolic_env import (
    ACTION_CHOICE,
    ACTION_EXECUTE,
    ACTION_FALLBACK,
    ACTION_MAP,
    ACTION_POINT,
    ACTION_REFUSE,
    ACTION_CONFIRM,
    SymbolicDialogueEnv,
)


SCENARIO_PLAN = [
    "high_amb",
    "low_amb",
    "minor_hazard",
    "random",
    "random",
    "high_amb",
    "low_amb",
    "minor_hazard",
    "random",
    "random",
]


def _load_policy(policy_path: str):
    if not policy_path:
        return None
    try:
        from stable_baselines3 import PPO
    except ImportError:
        return None
    try:
        return PPO.load(policy_path, device="cpu")
    except Exception:
        return None


def _heuristic_policy(obs: np.ndarray) -> int:
    amb = float(obs[0])
    hazard_prob = float(obs[1])
    p_minor = float(obs[3])
    age_conf = float(obs[6])
    guardian_code = float(obs[7])
    query_count = int(obs[8])

    if hazard_prob > 0.6 and p_minor > 0.7 and age_conf > 0.6 and guardian_code == 0.0:
        return ACTION_REFUSE
    if amb > 0.7 and query_count < 2:
        return ACTION_POINT
    if amb > 0.4 and query_count < 2:
        return ACTION_CHOICE
    if hazard_prob > 0.6 and p_minor > 0.6:
        return ACTION_FALLBACK
    return ACTION_EXECUTE


def _apply_scenario(env: SymbolicDialogueEnv, scenario: str, rng: np.random.Generator) -> None:
    if scenario == "high_amb":
        env.amb = float(rng.uniform(0.7, 0.95))
    elif scenario == "low_amb":
        env.amb = float(rng.uniform(0.05, 0.3))
    elif scenario == "minor_hazard":
        env.hazard_present = True
        env.hazard_prob = float(rng.uniform(0.75, 0.95))
        env.conflict_present = False
        env.conflict_prob = float(rng.uniform(0.05, 0.2))
        env.true_user_group = 0
        env.p_minor = 0.9
        env.p_adult = 0.05
        env.p_older = 0.05
        env.age_conf = 0.9
        env.guardian_present = False
        env.guardian_code = 0.0
    # keep query_count and last_outcome at reset defaults


def _select_action(policy, obs: np.ndarray) -> int:
    if policy is None:
        return _heuristic_policy(obs)
    action, _ = policy.predict(obs, deterministic=True)
    if isinstance(action, np.ndarray):
        return int(action.item())
    return int(action)


def run_demo(seed: int, episodes: int, max_steps: int, policy_path: str) -> None:
    rng = np.random.default_rng(seed)
    env = SymbolicDialogueEnv(max_steps=max_steps, seed=seed)
    policy = _load_policy(policy_path)

    plan = SCENARIO_PLAN[:]
    if episodes > len(plan):
        plan.extend(["random"] * (episodes - len(plan)))
    plan = plan[:episodes]

    for idx in range(episodes):
        obs, _info = env.reset()
        scenario = plan[idx]
        _apply_scenario(env, scenario, rng)
        obs = env._get_obs()

        actions: List[str] = []
        done = False
        violation = 0
        while not done:
            action = _select_action(policy, obs)
            actions.append(ACTION_MAP.get(action, str(action)))
            obs, _reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            if done and isinstance(info, dict):
                violation = int(info.get("violation", 0))

        print(
            "ep={idx} scenario={scenario} amb={amb:.2f} hazard_prob={hz:.2f} p_minor={pm:.2f} "
            "actions={actions} violation={violation}".format(
                idx=idx + 1,
                scenario=scenario,
                amb=float(env.amb),
                hz=float(env.hazard_prob),
                pm=float(env.p_minor),
                actions=actions,
                violation=violation,
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--policy-path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(args.seed, args.episodes, args.max_steps, args.policy_path)
