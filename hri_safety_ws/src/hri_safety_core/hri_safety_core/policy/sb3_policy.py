from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from hri_safety_core.arbiter_utils import action_from_index, build_rl_observation, utterance_for_action
from hri_safety_core.belief_tracker import BeliefState, QUERY_ACTIONS
from hri_safety_core.policy.base import PolicyBase, PolicyDecision


class Sb3Policy(PolicyBase):
    def __init__(
        self,
        policy_path: str,
        device: str = "cpu",
        deterministic: bool = True,
        obs_mode: str = "legacy",
        max_turns: int = 5,
    ) -> None:
        self.policy_path = policy_path
        self.device = device
        self.deterministic = deterministic
        self.obs_mode = obs_mode
        self.max_turns = max(1, int(max_turns))
        self.model = self._load_policy(policy_path, device)

    def _load_policy(self, policy_path: str, device: str):
        try:
            from stable_baselines3 import PPO
        except ImportError as exc:
            raise RuntimeError("stable_baselines3_not_installed") from exc
        policy_file = Path(policy_path)
        if not policy_file.is_file():
            raise FileNotFoundError(f"policy_not_found:{policy_path}")
        return PPO.load(str(policy_file), device=device)

    def _build_obs(self, belief: BeliefState) -> np.ndarray:
        if self.obs_mode == "belief":
            return np.array(belief.to_vector(max_turns=self.max_turns), dtype=np.float32)
        if self.obs_mode in {"belief_v1", "extended"}:
            return np.array(belief.to_vector_v1(), dtype=np.float32)
        features = {"amb": belief.amb, "risk": belief.risk, "conflict": belief.conflict}
        return build_rl_observation(features, query_count=belief.query_count, last_outcome=belief.last_outcome)

    def _adapt_obs(self, obs: np.ndarray) -> np.ndarray:
        try:
            target_dim = int(self.model.observation_space.shape[0])
        except Exception:
            return obs
        if obs.shape[0] == target_dim:
            return obs
        if obs.shape[0] > target_dim:
            return obs[:target_dim]
        pad = np.zeros(target_dim - obs.shape[0], dtype=np.float32)
        return np.concatenate([obs, pad], axis=0)

    def select_action(self, belief: BeliefState, action_mask: Dict[str, bool]) -> PolicyDecision:
        obs = self._adapt_obs(self._build_obs(belief))
        action_idx, _ = self.model.predict(obs, deterministic=self.deterministic)
        if isinstance(action_idx, np.ndarray):
            action_idx = int(action_idx.item())
        action = action_from_index(int(action_idx))

        if not action_mask.get(action, True):
            action = "FALLBACK_HUMAN_HELP"

        utterance = utterance_for_action(action, belief.selected_top1_id, belief.conflict_reason, belief.top_candidates)
        info = {"policy": "sb3", "raw_action": action_idx}
        if action in QUERY_ACTIONS:
            info["query"] = True
        return PolicyDecision(action=action, utterance=utterance, info=info)
