import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_EXECUTE = 0
ACTION_CONFIRM = 1
ACTION_CHOICE = 2
ACTION_POINT = 3
ACTION_REFUSE = 4

ACTION_MAP = {
    ACTION_EXECUTE: "EXECUTE",
    ACTION_CONFIRM: "CONFIRM_YN",
    ACTION_CHOICE: "CLARIFY_CHOICE",
    ACTION_POINT: "ASK_POINT",
    ACTION_REFUSE: "REFUSE_SAFE",
}

REWARD_CONFIG = {
    "execute_success": 10.0,
    "execute_conflict": -10.0,
    "execute_risky": -5.0,
    "refuse_safe": 5.0,
    "refuse_unnecessary": -2.0,
    "query_penalty": -0.5,
    "timeout_penalty": -2.0,
    "risk_threshold": 0.7,
    "conflict_prob": 0.1,
}

QUERY_ACTIONS = {ACTION_CONFIRM, ACTION_CHOICE, ACTION_POINT}


class SymbolicSafetyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 3, seed: int = 0):
        super().__init__()
        self.max_steps = max(1, int(max_steps))
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, float(self.max_steps), 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.amb = 0.0
        self.risk = 0.0
        self.conflict = 0.0
        self.query_count = 0
        self.step_count = 0
        self.last_outcome = 0.0

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.amb = float(self.rng.uniform(0.0, 1.0))
        self.risk = float(self.rng.uniform(0.0, 1.0))
        self.conflict = float(self.rng.uniform(0.0, 1.0) < REWARD_CONFIG["conflict_prob"])
        self.query_count = 0
        self.step_count = 0
        self.last_outcome = 0.0
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [self.amb, self.risk, self.conflict, float(self.query_count), self.last_outcome],
            dtype=np.float32,
        )

    def step(self, action: int):
        reward = 0.0
        cost = 0.0
        done = False

        if action == ACTION_EXECUTE:
            if self.conflict == 1.0:
                reward = REWARD_CONFIG["execute_conflict"]
                cost = 1.0
                self.last_outcome = 0.0
            elif self.risk >= REWARD_CONFIG["risk_threshold"]:
                reward = REWARD_CONFIG["execute_risky"]
                cost = 1.0
                self.last_outcome = 0.0
            else:
                reward = REWARD_CONFIG["execute_success"]
                self.last_outcome = 1.0
            done = True
        elif action == ACTION_REFUSE:
            if self.conflict == 1.0 or self.risk >= REWARD_CONFIG["risk_threshold"]:
                reward = REWARD_CONFIG["refuse_safe"]
                self.last_outcome = 1.0
            else:
                reward = REWARD_CONFIG["refuse_unnecessary"]
                self.last_outcome = 0.0
            done = True
        else:
            reward = REWARD_CONFIG["query_penalty"]
            self.query_count += 1
            self.amb = max(0.05, self.amb * 0.3)
            self.last_outcome = 0.0

        self.step_count += 1
        if not done and self.step_count >= self.max_steps:
            reward += REWARD_CONFIG["timeout_penalty"]
            done = True

        info = {"cost": cost}
        if done:
            violation = 1.0 if cost > 0 else 0.0
            refused = 1.0 if action == ACTION_REFUSE else 0.0
            success = 1.0 if (action == ACTION_EXECUTE and violation == 0.0) else 0.0
            if action == ACTION_REFUSE and (self.conflict == 1.0 or self.risk >= REWARD_CONFIG["risk_threshold"]):
                success = 1.0
            info.update(
                {
                    "success": success,
                    "violation": violation,
                    "refused": refused,
                    "queries": float(self.query_count),
                }
            )
        return self._get_obs(), reward, done, False, info


class FeatureNoiseWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        noise_amb: float = 0.0,
        noise_risk: float = 0.0,
        noise_conflict: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(env)
        self.noise_amb = max(0.0, float(noise_amb))
        self.noise_risk = max(0.0, float(noise_risk))
        self.noise_conflict = max(0.0, min(1.0, float(noise_conflict)))
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._apply_noise(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_noise(obs), reward, terminated, truncated, info

    def _apply_noise(self, obs: np.ndarray) -> np.ndarray:
        obs = np.array(obs, dtype=np.float32, copy=True)
        if self.noise_amb > 0.0:
            obs[0] = float(np.clip(obs[0] + self.rng.normal(0.0, self.noise_amb), 0.0, 1.0))
        if self.noise_risk > 0.0:
            obs[1] = float(np.clip(obs[1] + self.rng.normal(0.0, self.noise_risk), 0.0, 1.0))
        if self.noise_conflict > 0.0 and self.rng.random() < self.noise_conflict:
            obs[2] = 1.0 - obs[2]
        return obs
