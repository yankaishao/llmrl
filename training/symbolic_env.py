import numpy as np
import gymnasium as gym
from gymnasium import spaces

ACTION_EXECUTE = 0
ACTION_CONFIRM = 1
ACTION_CHOICE = 2
ACTION_POINT = 3
ACTION_REFUSE = 4
ACTION_FALLBACK = 5

ACTION_MAP = {
    ACTION_EXECUTE: "EXECUTE",
    ACTION_CONFIRM: "CONFIRM_YN",
    ACTION_CHOICE: "CLARIFY_CHOICE",
    ACTION_POINT: "ASK_POINT",
    ACTION_REFUSE: "REFUSE_SAFE",
    ACTION_FALLBACK: "FALLBACK_HUMAN_HELP",
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
    "fallback_good": 0.2,
    "fallback_bad": -0.2,
    "fallback_amb_high": 0.7,
}

QUERY_ACTIONS = {ACTION_CONFIRM, ACTION_CHOICE, ACTION_POINT}
HAZARD_KEYWORDS = {"knife", "scissors", "blade", "cutter", "sharp"}

DIALOGUE_CONFIG = {
    "hazard_prob": 0.2,
    "conflict_prob": 0.1,
    "minor_prob": 0.3,
    "older_prob": 0.2,
    "guardian_prob_minor": 0.4,
    "guardian_prob_other": 0.1,
    "guardian_known_prob": 0.5,
    "age_conf_min": 0.2,
    "age_conf_max": 1.0,
    "hazard_noise": 0.2,
    "conflict_noise": 0.2,
    "age_noise": 0.15,
    "misclass_base": 0.6,
    "query_base_penalty": -0.2,
    "query_growth_penalty": -0.05,
    "step_penalty": -0.05,
    "timeout_penalty": -2.0,
    "execute_success": 10.0,
    "execute_wrong": -4.0,
    "execute_violation": -12.0,
    "execute_conflict": -8.0,
    "execute_ambiguous": -2.0,
    "refuse_safe": 5.0,
    "refuse_unnecessary": -2.0,
    "refuse_ambiguous": -1.0,
    "fallback_good": 0.5,
    "fallback_bad": -0.5,
}


class SymbolicSafetyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 3, seed: int = 0):
        super().__init__()
        self.max_steps = max(1, int(max_steps))
        self.rng = np.random.default_rng(seed)

        self.action_space = spaces.Discrete(6)
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
        elif action == ACTION_FALLBACK:
            stuck = self.query_count >= max(1, self.max_steps - 1)
            high_amb = self.amb >= REWARD_CONFIG["fallback_amb_high"]
            high_risk = self.risk >= REWARD_CONFIG["risk_threshold"]
            conflict = self.conflict == 1.0
            if stuck or high_amb or high_risk or conflict:
                reward = REWARD_CONFIG["fallback_good"]
            else:
                reward = REWARD_CONFIG["fallback_bad"]
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
                    "outcome": "fallback" if fallback else "",
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


class SymbolicBeliefV1Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 5, seed: int = 0, config: dict | None = None):
        super().__init__()
        self.max_steps = max(1, int(max_steps))
        self.rng = np.random.default_rng(seed)
        self.config = dict(DIALOGUE_CONFIG)
        if isinstance(config, dict):
            self.config.update(config)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                    float(self.max_steps),
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
        self.p_top = 0.0
        self.margin = 0.0
        self.entropy = 0.0
        self.missing_slots = 0
        self.query_count = 0
        self.last_outcome = 0.0
        self.p_minor = 0.0
        self.p_adult = 0.0
        self.p_older = 0.0
        self.age_conf = 0.0
        self.guardian_code = -1.0
        self.hazard_top = 0.0
        self.step_count = 0

        self.hazard_present = False
        self.guardian_present = False
        self.true_user_group = 1

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.query_count = 0
        self.last_outcome = 0.0

        self.amb = float(self.rng.uniform(0.0, 1.0))
        self.conflict = float(self.rng.random() < self.config["conflict_prob"])

        self.hazard_present = bool(self.rng.random() < self.config["hazard_prob"])
        if self.hazard_present:
            self.hazard_top = float(self.rng.uniform(0.6, 1.0))
        else:
            self.hazard_top = float(self.rng.uniform(0.0, 0.4))

        self.true_user_group = self._sample_user_group()
        self.guardian_present = self._sample_guardian_present()

        self.age_conf = float(self.rng.uniform(self.config["age_conf_min"], self.config["age_conf_max"]))
        self.p_minor, self.p_adult, self.p_older = self._sample_age_probs(
            self.true_user_group, self.age_conf
        )
        if self.rng.random() < self.config["guardian_known_prob"] * self.age_conf:
            self.guardian_code = 1.0 if self.guardian_present else 0.0
        else:
            self.guardian_code = -1.0

        self._sample_candidate_distribution()
        self.missing_slots = self._sample_missing_slots()
        self.risk = self._compute_risk()

        return self._get_obs(), {}

    def _sample_user_group(self) -> int:
        minor_prob = self.config["minor_prob"]
        older_prob = self.config["older_prob"]
        adult_prob = max(0.0, 1.0 - minor_prob - older_prob)
        r = self.rng.random()
        if r < minor_prob:
            return 0
        if r < minor_prob + adult_prob:
            return 1
        return 2

    def _sample_guardian_present(self) -> bool:
        if self.true_user_group == 0:
            return bool(self.rng.random() < self.config["guardian_prob_minor"])
        return bool(self.rng.random() < self.config["guardian_prob_other"])

    def _sample_age_probs(self, true_group: int, age_conf: float) -> tuple[float, float, float]:
        misclass_prob = self.config["misclass_base"] * (1.0 - age_conf)
        probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        probs[true_group] = 0.85
        if self.rng.random() < misclass_prob:
            wrong = (true_group + int(self.rng.integers(1, 3))) % 3
            probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
            probs[wrong] = 0.85
        noise = self.rng.normal(0.0, self.config["age_noise"] * (1.0 - age_conf), size=3)
        probs = np.clip(probs + noise, 0.01, 0.98)
        total = float(np.sum(probs))
        probs = probs / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        return float(probs[0]), float(probs[1]), float(probs[2])

    def _sample_candidate_distribution(self) -> None:
        n = 3
        base = self.rng.random(n) + 1e-3
        exponent = 1.0 + (1.0 - self.amb) * 4.0
        scaled = base**exponent
        probs = scaled / float(np.sum(scaled))
        probs = np.clip(probs, 1e-4, 1.0)
        probs = probs / float(np.sum(probs))

        self.p_top = float(np.max(probs))
        sorted_probs = np.sort(probs)[::-1]
        self.margin = float(sorted_probs[0] - (sorted_probs[1] if len(sorted_probs) > 1 else 0.0))
        entropy = -float(np.sum(probs * np.log(probs)))
        self.entropy = float(entropy / max(1e-6, np.log(float(n))))

    def _sample_missing_slots(self) -> int:
        if self.amb > 0.7:
            count = int(self.rng.integers(2, 4))
        elif self.amb > 0.4:
            count = int(self.rng.integers(1, 3))
        else:
            count = int(self.rng.integers(0, 2))
        return min(3, max(0, count))

    def _compute_risk(self) -> float:
        intent_risk = float(self.rng.uniform(0.0, 0.2))
        conflict_extra = 0.3 if self.conflict >= 1.0 else 0.0
        minor_high = self.p_minor >= 0.7 and self.age_conf >= 0.6
        age_extra = 0.2 if minor_high and self.hazard_top >= 0.3 else 0.0
        risk = intent_risk + 0.6 * self.hazard_top + age_extra + conflict_extra
        return float(np.clip(risk, 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        missing_norm = float(self.missing_slots) / 3.0
        return np.array(
            [
                self.amb,
                self.risk,
                self.conflict,
                self.p_top,
                self.margin,
                self.entropy,
                missing_norm,
                float(self.query_count),
                self.last_outcome,
                self.p_minor,
                self.age_conf,
                self.guardian_code,
                self.hazard_top,
            ],
            dtype=np.float32,
        )

    def step(self, action: int):
        reward = 0.0
        cost = 0.0
        terminated = False
        truncated = False

        if action in QUERY_ACTIONS:
            reward += self.config["query_base_penalty"] + self.config["query_growth_penalty"] * self.query_count
            self.query_count += 1
            self.amb = max(0.05, self.amb * 0.4)
            self.missing_slots = max(0, self.missing_slots - 1)
            self._sample_candidate_distribution()
            self.risk = max(0.0, self.risk - 0.05)
            self.last_outcome = 0.0

        elif action == ACTION_EXECUTE:
            if self.conflict >= 1.0:
                reward = REWARD_CONFIG["execute_conflict"]
                cost = 1.0
            elif self.risk >= REWARD_CONFIG["risk_threshold"]:
                reward = REWARD_CONFIG["execute_risky"]
                cost = 1.0
            elif self.amb >= 0.6 or self.missing_slots >= 2:
                reward = DIALOGUE_CONFIG["execute_ambiguous"]
            else:
                reward = REWARD_CONFIG["execute_success"]
                self.last_outcome = 1.0
            terminated = True

        elif action == ACTION_REFUSE:
            if self.conflict >= 1.0 or self.risk >= REWARD_CONFIG["risk_threshold"]:
                reward = REWARD_CONFIG["refuse_safe"]
                self.last_outcome = 1.0
            else:
                reward = REWARD_CONFIG["refuse_unnecessary"]
            terminated = True

        elif action == ACTION_FALLBACK:
            good = (
                self.query_count >= max(1, self.max_steps - 1)
                or self.amb >= 0.6
                or self.risk >= REWARD_CONFIG["risk_threshold"]
                or self.conflict >= 1.0
            )
            reward = REWARD_CONFIG["fallback_good"] if good else REWARD_CONFIG["fallback_bad"]
            terminated = True

        self.step_count += 1
        if not terminated and self.step_count >= self.max_steps:
            reward += REWARD_CONFIG["timeout_penalty"]
            truncated = True

        info = {"cost": cost}
        if terminated or truncated:
            violation = 1.0 if cost > 0 else 0.0
            refused = 1.0 if action == ACTION_REFUSE else 0.0
            fallback = 1.0 if action == ACTION_FALLBACK else 0.0
            success = 1.0 if (action == ACTION_EXECUTE and violation == 0.0) else 0.0
            if action == ACTION_REFUSE and (self.conflict >= 1.0 or self.risk >= REWARD_CONFIG["risk_threshold"]):
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


class SymbolicDialogueEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 5,
        seed: int = 0,
        debug_mode: bool = False,
        config: dict | None = None,
    ):
        super().__init__()
        self.max_steps = max(1, int(max_steps))
        self.rng = np.random.default_rng(seed)
        self.debug_mode = bool(debug_mode)
        self.config = dict(DIALOGUE_CONFIG)
        if isinstance(config, dict):
            self.config.update(config)

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            high=np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, float(self.max_steps), 1.0],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.amb = 0.0
        self.hazard_prob = 0.0
        self.conflict_prob = 0.0
        self.p_minor = 0.0
        self.p_adult = 0.0
        self.p_older = 0.0
        self.age_conf = 0.0
        self.guardian_code = -1.0
        self.query_count = 0
        self.step_count = 0
        self.last_outcome = 0.0

        self.true_target = 0
        self.hazard_present = False
        self.conflict_present = False
        self.true_user_group = 1
        self.guardian_present = False
        self.user_patience = 0.5

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.step_count = 0
        self.query_count = 0
        self.last_outcome = 0.0

        self.true_target = int(self.rng.integers(0, 2))
        self.hazard_present = bool(self.rng.random() < self.config["hazard_prob"])
        conflict_bias = 0.15 if self.hazard_present else 0.0
        self.conflict_present = bool(self.rng.random() < min(1.0, self.config["conflict_prob"] + conflict_bias))

        self.true_user_group = self._sample_user_group()
        self.guardian_present = self._sample_guardian_present()
        self.user_patience = float(self.rng.uniform(0.3, 1.0))

        self.amb = float(self.rng.uniform(0.0, 1.0))
        self.hazard_prob = self._noisy_prob(float(self.hazard_present), self.config["hazard_noise"])
        self.conflict_prob = self._noisy_prob(float(self.conflict_present), self.config["conflict_noise"])

        self.age_conf = float(self.rng.uniform(self.config["age_conf_min"], self.config["age_conf_max"]))
        self.p_minor, self.p_adult, self.p_older = self._sample_age_probs(
            self.true_user_group, self.age_conf
        )

        if self.rng.random() < self.config["guardian_known_prob"] * self.age_conf:
            self.guardian_code = 1.0 if self.guardian_present else 0.0
        else:
            self.guardian_code = -1.0

        return self._get_obs(), {}

    def _sample_user_group(self) -> int:
        minor_prob = self.config["minor_prob"]
        older_prob = self.config["older_prob"]
        adult_prob = max(0.0, 1.0 - minor_prob - older_prob)
        r = self.rng.random()
        if r < minor_prob:
            return 0
        if r < minor_prob + adult_prob:
            return 1
        return 2

    def _sample_guardian_present(self) -> bool:
        if self.true_user_group == 0:
            return bool(self.rng.random() < self.config["guardian_prob_minor"])
        return bool(self.rng.random() < self.config["guardian_prob_other"])

    def _sample_age_probs(self, true_group: int, age_conf: float) -> tuple[float, float, float]:
        misclass_prob = self.config["misclass_base"] * (1.0 - age_conf)
        probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        probs[true_group] = 0.85
        if self.rng.random() < misclass_prob:
            wrong = (true_group + int(self.rng.integers(1, 3))) % 3
            probs = np.array([0.05, 0.05, 0.05], dtype=np.float32)
            probs[wrong] = 0.85
        noise = self.rng.normal(0.0, self.config["age_noise"] * (1.0 - age_conf), size=3)
        probs = np.clip(probs + noise, 0.01, 0.98)
        total = float(np.sum(probs))
        probs = probs / total if total > 0 else np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        return float(probs[0]), float(probs[1]), float(probs[2])

    def _noisy_prob(self, base: float, sigma: float) -> float:
        return float(np.clip(base + self.rng.normal(0.0, sigma), 0.0, 1.0))

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.amb,
                self.hazard_prob,
                self.conflict_prob,
                self.p_minor,
                self.p_adult,
                self.p_older,
                self.age_conf,
                self.guardian_code,
                float(self.query_count),
                self.last_outcome,
            ],
            dtype=np.float32,
        )

    def step(self, action: int):
        reward = float(self.config["step_penalty"])
        cost = 0.0
        terminated = False
        truncated = False
        execution_failed = 0
        wrong_target = 0

        if action in QUERY_ACTIONS:
            query_penalty = self.config["query_base_penalty"] + self.config["query_growth_penalty"] * self.query_count
            reward += query_penalty
            self.query_count += 1

            if action == ACTION_CONFIRM:
                self.hazard_prob = float(
                    np.clip(
                        self.hazard_prob
                        + self.user_patience * (float(self.hazard_present) - self.hazard_prob) * 0.6,
                        0.0,
                        1.0,
                    )
                )
                self.conflict_prob = float(
                    np.clip(
                        self.conflict_prob
                        + self.user_patience * (float(self.conflict_present) - self.conflict_prob) * 0.4,
                        0.0,
                        1.0,
                    )
                )
                self.age_conf = float(min(1.0, self.age_conf + 0.2 * self.user_patience))
                if self.guardian_code < 0 and self.rng.random() < 0.3 * self.user_patience:
                    self.guardian_code = 1.0 if self.guardian_present else 0.0
            elif action == ACTION_CHOICE:
                factor = 0.7 - 0.2 * self.user_patience
                self.amb = max(0.05, self.amb * factor)
            elif action == ACTION_POINT:
                factor = 0.5 - 0.2 * self.user_patience
                self.amb = max(0.05, self.amb * factor)
                reward += -0.05

        elif action == ACTION_EXECUTE:
            p_wrong = float(np.clip(self.amb, 0.0, 0.95))
            if self.rng.random() < p_wrong:
                reward += self.config["execute_wrong"]
                execution_failed = 1
                wrong_target = 1
                terminated = True
            else:
                minor_high = self.p_minor >= 0.7 and self.age_conf >= 0.6
                guardian_ok = self.guardian_present if self.guardian_code >= 0 else self.guardian_present
                if self.hazard_present and minor_high and not guardian_ok:
                    reward += self.config["execute_violation"]
                    cost = 1.0
                    terminated = True
                elif self.conflict_present:
                    reward += self.config["execute_conflict"]
                    cost = 1.0
                    terminated = True
                elif self.amb >= 0.6:
                    reward += self.config["execute_ambiguous"]
                    terminated = True
                else:
                    reward += self.config["execute_success"]
                    self.last_outcome = 1.0
                    terminated = True

        elif action == ACTION_REFUSE:
            if self.hazard_present or self.conflict_present:
                reward += self.config["refuse_safe"]
                self.last_outcome = 1.0
            elif self.amb < 0.3:
                reward += self.config["refuse_unnecessary"]
            else:
                reward += self.config["refuse_ambiguous"]
            terminated = True

        elif action == ACTION_FALLBACK:
            good = (
                self.query_count >= max(1, self.max_steps - 1)
                or self.amb >= 0.6
                or self.hazard_prob >= 0.6
                or self.conflict_prob >= 0.6
            )
            reward += self.config["fallback_good"] if good else self.config["fallback_bad"]
            terminated = True

        self.step_count += 1
        if not terminated and self.step_count >= self.max_steps:
            reward += self.config["timeout_penalty"]
            truncated = True

        info = {"cost": cost}
        if terminated or truncated:
            violation = 1.0 if cost > 0 else 0.0
            refused = 1.0 if action == ACTION_REFUSE else 0.0
            fallback = 1.0 if action == ACTION_FALLBACK else 0.0
            success = 1.0 if (action == ACTION_EXECUTE and violation == 0.0 and wrong_target == 0) else 0.0
            if action == ACTION_REFUSE and (self.hazard_present or self.conflict_present):
                success = 1.0
            info.update(
                {
                    "success": success,
                    "violation": violation,
                    "refused": refused,
                    "fallback": fallback,
                    "queries": float(self.query_count),
                    "execution_failed": execution_failed,
                    "wrong_target": wrong_target,
                }
            )
            if self.debug_mode:
                info.update(
                    {
                        "hazard_present": int(self.hazard_present),
                        "true_user_group": int(self.true_user_group),
                        "guardian_present": int(self.guardian_present),
                    }
                )

        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        mask = np.ones(self.action_space.n, dtype=np.int8)
        if (
            self.hazard_prob > 0.6
            and self.p_minor > 0.7
            and self.age_conf > 0.6
            and self.guardian_code == 0.0
        ):
            mask[ACTION_EXECUTE] = 0
        return mask
