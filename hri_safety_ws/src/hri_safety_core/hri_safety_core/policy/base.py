from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from hri_safety_core.belief_tracker import BeliefState


@dataclass
class PolicyDecision:
    action: str
    utterance: str
    info: Dict[str, object]


class PolicyBase:
    def select_action(self, belief: BeliefState, action_mask: Dict[str, bool]) -> PolicyDecision:
        raise NotImplementedError
