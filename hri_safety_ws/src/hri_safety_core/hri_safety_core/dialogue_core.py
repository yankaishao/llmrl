from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from hri_safety_core.arbiter_utils import R_HIGH_DEFAULT, extract_candidates, top_two_candidates, utterance_for_action
from hri_safety_core.belief_tracker import (
    QUERY_ACTIONS,
    build_action_mask,
    build_belief_state,
    classify_reply,
)
from hri_safety_core.policy.base import PolicyDecision

STATE_IDLE = "IDLE"
STATE_NEED_DECISION = "ACTIVE:NEED_DECISION"
STATE_WAIT_USER = "ACTIVE:WAIT_USER"
STATE_TERMINAL = "TERMINAL"


@dataclass
class DialogueSnapshot:
    state: str
    turn_count: int
    query_count: int
    sys_ask_count: int
    last_user_reply_type: str
    last_action: str
    repeat_query_count: int
    context: str


class DialogueManagerCore:
    def __init__(self, max_turns: int = 5, max_repeat_action: int = 2, use_age_context: bool = False) -> None:
        self.max_turns = max(1, int(max_turns))
        self.max_repeat_action = max(1, int(max_repeat_action))
        self.use_age_context = bool(use_age_context)
        self.reset()

    def reset(self) -> None:
        self.state = STATE_IDLE
        self.turn_count = 0
        self.query_count = 0
        self.sys_ask_count = 0
        self.last_user_reply_type = "other"
        self.last_action = ""
        self.last_query_action = ""
        self.repeat_query_count = 0
        self.context = ""
        self.last_system_utterance = ""
        self.parse_result: Optional[Dict[str, object]] = None
        self.features: Optional[Dict[str, object]] = None
        self.age_context: Optional[Dict[str, object]] = None
        self.last_outcome = 0.0

    def snapshot(self) -> DialogueSnapshot:
        return DialogueSnapshot(
            state=self.state,
            turn_count=self.turn_count,
            query_count=self.query_count,
            sys_ask_count=self.sys_ask_count,
            last_user_reply_type=self.last_user_reply_type,
            last_action=self.last_action,
            repeat_query_count=self.repeat_query_count,
            context=self.context,
        )

    def start_session(self, text: str) -> str:
        self.reset()
        self.state = STATE_NEED_DECISION
        self.context = text.strip()
        self.last_user_reply_type = classify_reply(text)
        self.parse_result = None
        self.features = None
        return self.context

    def receive_user_reply(self, text: str) -> str:
        self.turn_count += 1
        self._append_context("User", text)
        self.last_user_reply_type = classify_reply(text)
        self.state = STATE_NEED_DECISION
        self.parse_result = None
        self.features = None
        return self.context

    def record_system_question(self, action: str, utterance: str) -> None:
        self.turn_count += 1
        self.query_count += 1
        self.sys_ask_count += 1
        if action == self.last_query_action:
            self.repeat_query_count += 1
        else:
            self.last_query_action = action
            self.repeat_query_count = 1
        self.last_action = action
        self.last_system_utterance = utterance
        self._append_context("System", utterance)
        self.state = STATE_WAIT_USER

    def end_session(self) -> None:
        self.state = STATE_TERMINAL
        self.state = STATE_IDLE
        self.context = ""
        self.parse_result = None
        self.features = None
        self.last_action = ""
        self.last_query_action = ""
        self.repeat_query_count = 0

    def update_parse_result(self, parse_result: Dict[str, object]) -> None:
        self.parse_result = parse_result

    def update_features(self, features: Dict[str, object]) -> None:
        self.features = features

    def update_age_context(self, age_context: Dict[str, object]) -> None:
        self.age_context = age_context

    def should_force_terminal(self) -> bool:
        return self.turn_count >= self.max_turns

    def can_ask(self) -> bool:
        return self.turn_count <= self.max_turns - 2

    def query_repeat_exceeded(self, action: str) -> bool:
        if action != self.last_query_action:
            return False
        return self.repeat_query_count >= self.max_repeat_action

    def choose_terminal_action(self, risk_high: float = R_HIGH_DEFAULT) -> PolicyDecision:
        features = self.features or {}
        risk = float(features.get("risk", 0.0))
        conflict = int(features.get("conflict", 0))
        conflict_reason = str(features.get("conflict_reason", ""))
        selected_id = str(features.get("selected_top1_id", ""))
        top_candidates = []
        if self.parse_result:
            candidates = extract_candidates(self.parse_result)
            top_candidates = top_two_candidates(candidates)

        if conflict == 1 or risk >= risk_high:
            action = "REFUSE_SAFE"
        else:
            action = "FALLBACK_HUMAN_HELP"
        utterance = utterance_for_action(action, selected_id, conflict_reason, top_candidates)
        return PolicyDecision(action=action, utterance=utterance, info={"forced": True})

    def build_belief(self) -> Optional[object]:
        if self.parse_result is None or self.features is None:
            return None
        return build_belief_state(
            parse_result=self.parse_result,
            features=self.features,
            turn_sys_asks=self.sys_ask_count,
            last_user_reply_type=self.last_user_reply_type,
            history_summary=self.context,
            query_count=self.query_count,
            last_outcome=self.last_outcome,
            age_context=self.age_context,
            use_age_context=self.use_age_context,
        )

    def build_action_mask(self, belief: Optional[object]) -> Dict[str, bool]:
        return build_action_mask(self.turn_count, self.max_turns, belief=belief)

    def _append_context(self, role: str, text: str) -> None:
        cleaned = text.strip()
        if not cleaned:
            return
        line = f"{role}: {cleaned}"
        if self.context:
            self.context = self.context + "\n" + line
        else:
            self.context = line

    def apply_decision(self, decision: PolicyDecision) -> None:
        action = decision.action
        if action in QUERY_ACTIONS:
            self.record_system_question(action, decision.utterance)
            return
        self.last_action = action
        self.end_session()
