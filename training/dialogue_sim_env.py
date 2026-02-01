import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_EXECUTE = 0
ACTION_CONFIRM = 1
ACTION_CHOICE = 2
ACTION_POINT = 3
ACTION_REFUSE = 4
ACTION_FALLBACK = 5

QUERY_ACTIONS = {ACTION_CONFIRM, ACTION_CHOICE, ACTION_POINT}

REWARD_CONFIG = {
    "execute_success": 10.0,
    "execute_ambiguous": -2.0,
    "execute_conflict": -10.0,
    "execute_risky": -5.0,
    "refuse_safe": 5.0,
    "refuse_unnecessary": -2.0,
    "fallback_good": 0.2,
    "fallback_bad": -0.2,
    "query_penalty": -0.5,
    "timeout_penalty": -2.0,
    "risk_threshold": 0.7,
    "conflict_prob": 0.1,
    "amb_high": 0.7,
}


class DialogueSimEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_turns: int = 5, seed: int = 0):
        super().__init__()
        self.max_turns = max(1, int(max_turns))
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    float(self.max_turns),
                    3.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.amb = 0.0
        self.risk = 0.0
        self.conflict = 0.0
        self.turn_count = 0
        self.query_count = 0
        self.last_reply_code = 0.0
        self.last_outcome = 0.0
        self.user_group_true = 1
        self.age_context = {
            "p_minor": 0.0,
            "p_adult": 1.0,
            "p_older": 0.0,
            "age_conf": 1.0,
            "guardian_present": None,
        }
        self.hazard = False

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.amb = float(self.rng.uniform(0.0, 1.0))
        self.hazard = bool(self.rng.random() < 0.3)
        self.risk = float(self.rng.uniform(0.0, 1.0))
        if self.hazard:
            self.risk = max(self.risk, 0.8)
        self.conflict = float(self.rng.uniform(0.0, 1.0) < REWARD_CONFIG["conflict_prob"])
        self.turn_count = 0
        self.query_count = 0
        self.last_reply_code = 0.0
        self.last_outcome = 0.0
        self.user_group_true = int(self.rng.integers(0, 3))
        self.age_context = self._sample_age_context(self.user_group_true)
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        guardian_code = -1.0
        guardian = self.age_context.get("guardian_present")
        if isinstance(guardian, bool):
            guardian_code = 1.0 if guardian else 0.0
        return np.array(
            [
                self.amb,
                self.risk,
                self.conflict,
                float(self.turn_count),
                self.last_reply_code,
                self.last_outcome,
                float(self.age_context["p_minor"]),
                float(self.age_context["p_adult"]),
                float(self.age_context["p_older"]),
                float(self.age_context["age_conf"]),
                guardian_code,
            ],
            dtype=np.float32,
        )

    def _sample_age_context(self, true_group: int) -> Dict[str, object]:
        age_conf = float(self.rng.uniform(0.3, 1.0))
        misclass_prob = min(0.6, max(0.05, 1.0 - age_conf))
        probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        probs[true_group] = 0.85
        if self.rng.random() < misclass_prob:
            wrong = (true_group + self.rng.integers(1, 3)) % 3
            probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
            probs[wrong] = 0.85
        noise = self.rng.normal(0.0, 0.05, size=3)
        probs = np.clip(probs + noise, 0.01, 0.98)
        total = float(np.sum(probs))
        probs = probs / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3])
        return {
            "p_minor": float(probs[0]),
            "p_adult": float(probs[1]),
            "p_older": float(probs[2]),
            "age_conf": age_conf,
            "guardian_present": None,
        }

    def step(self, action: int):
        reward = 0.0
        cost = 0.0
        terminated = False
        truncated = False

        if action == ACTION_EXECUTE:
            if self.conflict == 1.0:
                reward = REWARD_CONFIG["execute_conflict"]
                cost = 1.0
            elif self.risk >= REWARD_CONFIG["risk_threshold"]:
                reward = REWARD_CONFIG["execute_risky"]
                cost = 1.0
                if self.hazard and self.user_group_true == 0:
                    reward += -5.0
            elif self.amb >= 0.4:
                reward = REWARD_CONFIG["execute_ambiguous"]
            else:
                reward = REWARD_CONFIG["execute_success"]
                self.last_outcome = 1.0
            terminated = True
        elif action == ACTION_REFUSE:
            if self.conflict == 1.0 or self.risk >= REWARD_CONFIG["risk_threshold"]:
                reward = REWARD_CONFIG["refuse_safe"]
                self.last_outcome = 1.0
            else:
                reward = REWARD_CONFIG["refuse_unnecessary"]
            terminated = True
        elif action == ACTION_FALLBACK:
            stuck = self.turn_count >= max(1, self.max_turns - 1)
            high_amb = self.amb >= REWARD_CONFIG["amb_high"]
            high_risk = self.risk >= REWARD_CONFIG["risk_threshold"]
            conflict = self.conflict == 1.0
            if stuck or high_amb or high_risk or conflict:
                reward = REWARD_CONFIG["fallback_good"]
            else:
                reward = REWARD_CONFIG["fallback_bad"]
            terminated = True
        else:
            reward = REWARD_CONFIG["query_penalty"]
            self.query_count += 1
            self.amb = max(0.05, self.amb * 0.3)
            if action == ACTION_CONFIRM:
                self.last_reply_code = 1.0 if self.rng.random() < 0.7 else 2.0
            elif action == ACTION_CHOICE:
                self.last_reply_code = 3.0
            else:
                self.last_reply_code = 3.0

        self.turn_count += 1
        if not terminated and self.turn_count >= self.max_turns:
            reward += REWARD_CONFIG["timeout_penalty"]
            truncated = True

        info = {"cost": cost}
        if terminated or truncated:
            violation = 1.0 if cost > 0 else 0.0
            refused = 1.0 if action == ACTION_REFUSE else 0.0
            fallback = 1.0 if action == ACTION_FALLBACK else 0.0
            success = 1.0 if (action == ACTION_EXECUTE and violation == 0.0) else 0.0
            if action == ACTION_REFUSE and (self.conflict == 1.0 or self.risk >= REWARD_CONFIG["risk_threshold"]):
                success = 1.0
            info.update(
                {
                    "success": success,
                    "violation": violation,
                    "refused": refused,
                    "fallback": fallback,
                    "queries": float(self.query_count),
                }
            )

        return self._get_obs(), reward, terminated, truncated, info
