from __future__ import annotations

from typing import Dict, List

from hri_safety_core.arbiter_utils import A_HIGH_DEFAULT, R_HIGH_DEFAULT, utterance_for_action
from hri_safety_core.belief_tracker import BeliefState
from hri_safety_core.policy.base import PolicyBase, PolicyDecision

MARGIN_LOW_DEFAULT = 0.2


def _format_choice(templates: Dict[str, str], candidates: List[str]) -> str:
    template = templates.get("choice", "Which one do you mean: {obj_a} or {obj_b}?")
    obj_a = candidates[0] if len(candidates) > 0 else "object A"
    obj_b = candidates[1] if len(candidates) > 1 else "object B"
    return template.format(obj_a=obj_a, obj_b=obj_b)


def _format_confirm(templates: Dict[str, str], target_id: str) -> str:
    template = templates.get("yesno", "Do you mean {obj_a}?")
    obj_a = target_id or "this object"
    return template.format(obj_a=obj_a, obj_b="")


def _format_point(templates: Dict[str, str]) -> str:
    return templates.get("point", "Could you indicate which object you mean?")


class RulePolicy(PolicyBase):
    def __init__(self, amb_high: float = A_HIGH_DEFAULT, risk_high: float = R_HIGH_DEFAULT, margin_low: float = MARGIN_LOW_DEFAULT):
        self.amb_high = float(amb_high)
        self.risk_high = float(risk_high)
        self.margin_low = float(margin_low)
        self.minor_high = 0.6
        self.age_conf_high = 0.7
        self.age_conf_low = 0.3

    def select_action(self, belief: BeliefState, action_mask: Dict[str, bool]) -> PolicyDecision:
        info: Dict[str, object] = {}
        templates = belief.clarify_templates or {}

        high_risk = belief.risk >= self.risk_high or belief.conflict == 1
        ambiguous = belief.missing_slots > 0 or belief.margin < self.margin_low or belief.amb >= self.amb_high
        hazard = belief.hazard
        minor_high_conf = belief.p_minor >= self.minor_high and belief.age_conf >= self.age_conf_high
        low_age_conf = belief.age_conf <= self.age_conf_low

        if hazard and low_age_conf and action_mask.get("CONFIRM_YN", True):
            action = "CONFIRM_YN"
            utterance = (
                "For safety, are you under 18, or is a guardian/teacher present to confirm this request?"
            )
            info["reason"] = "age_conf_low"
            return PolicyDecision(action=action, utterance=utterance, info=info)

        if hazard and minor_high_conf:
            if belief.guardian_present and action_mask.get("CONFIRM_YN", True):
                action = "CONFIRM_YN"
                utterance = (
                    "This involves a potentially dangerous item. Can a guardian confirm this request?"
                )
                info["reason"] = "minor_high_confirm_guardian"
                return PolicyDecision(action=action, utterance=utterance, info=info)
            action = "REFUSE_SAFE"
            utterance = utterance_for_action(
                action,
                belief.selected_top1_id,
                "minor_hazard",
                belief.top_candidates,
            )
            info["reason"] = "minor_high_refuse"
            return PolicyDecision(action=action, utterance=utterance, info=info)

        if high_risk:
            if belief.last_user_reply_type in {"yes", "no"} or not action_mask.get("CONFIRM_YN", True):
                action = "REFUSE_SAFE"
                utterance = utterance_for_action(
                    action,
                    belief.selected_top1_id,
                    belief.conflict_reason,
                    belief.top_candidates,
                )
                info["reason"] = "risk_or_conflict"
                return PolicyDecision(action=action, utterance=utterance, info=info)

            action = "CONFIRM_YN"
            utterance = _format_confirm(templates, belief.selected_top1_id)
            info["reason"] = "confirm_risk"
            return PolicyDecision(action=action, utterance=utterance, info=info)

        if ambiguous:
            if action_mask.get("CLARIFY_CHOICE", True) and len(belief.top_candidates) >= 2:
                action = "CLARIFY_CHOICE"
                utterance = _format_choice(templates, belief.top_candidates)
                info["reason"] = "clarify_choice"
                return PolicyDecision(action=action, utterance=utterance, info=info)

            if action_mask.get("ASK_POINT", True):
                action = "ASK_POINT"
                utterance = _format_point(templates)
                info["reason"] = "ask_point"
                return PolicyDecision(action=action, utterance=utterance, info=info)

            action = "FALLBACK_HUMAN_HELP"
            utterance = utterance_for_action(action, belief.selected_top1_id, belief.conflict_reason, belief.top_candidates)
            info["reason"] = "no_query_budget"
            return PolicyDecision(action=action, utterance=utterance, info=info)

        action = "EXECUTE"
        utterance = utterance_for_action(action, belief.selected_top1_id, belief.conflict_reason, belief.top_candidates)
        info["reason"] = "execute"
        return PolicyDecision(action=action, utterance=utterance, info=info)
