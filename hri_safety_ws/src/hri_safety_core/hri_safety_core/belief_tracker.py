from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from hri_safety_core.arbiter_utils import clamp, safe_float, safe_int

REPLY_YES = {"yes", "yeah", "yep", "sure", "ok", "okay", "affirmative", "correct"}
REPLY_NO = {"no", "nope", "nah", "negative", "incorrect"}
POINT_HINTS = {"left", "right", "this", "that", "here", "there", "point"}

ACTION_NAMES = [
    "EXECUTE",
    "CONFIRM_YN",
    "CLARIFY_CHOICE",
    "ASK_POINT",
    "REFUSE_SAFE",
    "FALLBACK_HUMAN_HELP",
]
QUERY_ACTIONS = {"CONFIRM_YN", "CLARIFY_CHOICE", "ASK_POINT"}
HAZARD_KEYWORDS = {"knife", "scissors", "blade", "cutter", "sharp"}

AGE_CONTEXT_DEFAULT = {
    "p_minor": 0.0,
    "p_adult": 1.0,
    "p_older": 0.0,
    "age_conf": 1.0,
    "guardian_present": None,
}


@dataclass
class BeliefState:
    p_top: float
    margin: float
    missing_slots: int
    amb: float
    risk: float
    conflict: int
    turn_sys_asks: int
    last_user_reply_type: str
    history_summary: str
    task_type: str
    top_candidates: List[str]
    selected_top1_id: str
    clarify_templates: Dict[str, str]
    conflict_reason: str
    query_count: int
    last_outcome: float
    p_minor: float
    p_adult: float
    p_older: float
    age_conf: float
    guardian_present: Optional[bool]
    hazard: bool
    hazard_reason: str

    def to_vector(self, max_turns: int) -> List[float]:
        missing_norm = clamp(float(self.missing_slots) / 3.0)
        turn_norm = clamp(float(self.turn_sys_asks) / max(1.0, float(max_turns)))
        reply_code = reply_type_code(self.last_user_reply_type)
        guardian_code = guardian_present_code(self.guardian_present)
        return [
            clamp(self.p_top),
            clamp(self.margin),
            missing_norm,
            clamp(self.amb),
            clamp(self.risk),
            float(self.conflict),
            turn_norm,
            float(reply_code),
            clamp(self.p_minor),
            clamp(self.p_adult),
            clamp(self.p_older),
            clamp(self.age_conf),
            float(guardian_code),
        ]


def reply_type_code(reply_type: str) -> int:
    reply_type = reply_type.lower().strip()
    if reply_type == "yes":
        return 1
    if reply_type == "no":
        return 2
    if reply_type == "point":
        return 3
    return 0


def guardian_present_code(value: Optional[bool]) -> int:
    if value is None:
        return -1
    return 1 if value else 0


def classify_reply(text: str) -> str:
    lowered = text.strip().lower()
    if not lowered:
        return "other"
    tokens = {token.strip(".,!? ") for token in lowered.split()}
    if tokens & REPLY_YES:
        return "yes"
    if tokens & REPLY_NO:
        return "no"
    if tokens & POINT_HINTS:
        return "point"
    if any(char.isdigit() for char in lowered) and "," in lowered:
        return "point"
    return "other"


def extract_candidates(parse_result: Dict[str, object]) -> List[Tuple[str, float]]:
    raw = parse_result.get("candidates", []) if isinstance(parse_result, dict) else []
    if not isinstance(raw, list):
        return []
    candidates: List[Tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        obj_id = item.get("id")
        score = item.get("score")
        if isinstance(obj_id, str) and isinstance(score, (int, float)):
            candidates.append((obj_id, float(score)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _normalize_probs(p_minor: float, p_adult: float, p_older: float) -> Tuple[float, float, float]:
    total = p_minor + p_adult + p_older
    if total <= 0:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    return (p_minor / total, p_adult / total, p_older / total)


def parse_age_context(age_context: Optional[Dict[str, object]], use_age_context: bool) -> Dict[str, object]:
    if not use_age_context:
        return dict(AGE_CONTEXT_DEFAULT)

    if not isinstance(age_context, dict):
        return {
            "p_minor": 1.0 / 3.0,
            "p_adult": 1.0 / 3.0,
            "p_older": 1.0 / 3.0,
            "age_conf": 0.0,
            "guardian_present": None,
        }

    p_minor = clamp(safe_float(age_context.get("p_minor", 0.0)))
    p_adult = clamp(safe_float(age_context.get("p_adult", 0.0)))
    p_older = clamp(safe_float(age_context.get("p_older", 0.0)))
    p_minor, p_adult, p_older = _normalize_probs(p_minor, p_adult, p_older)
    age_conf = clamp(safe_float(age_context.get("age_conf", 0.0)))
    guardian = age_context.get("guardian_present")
    guardian_present = None
    if isinstance(guardian, bool):
        guardian_present = guardian
    elif isinstance(guardian, str):
        lowered = guardian.strip().lower()
        if lowered in {"true", "yes", "1"}:
            guardian_present = True
        elif lowered in {"false", "no", "0"}:
            guardian_present = False
    return {
        "p_minor": p_minor,
        "p_adult": p_adult,
        "p_older": p_older,
        "age_conf": age_conf,
        "guardian_present": guardian_present,
    }


def detect_hazard(parse_result: Dict[str, object], features: Dict[str, object]) -> Tuple[bool, str]:
    reason = ""
    selected = str(features.get("selected_top1_id", "")) if isinstance(features, dict) else ""
    selected_lower = selected.lower()
    if any(keyword in selected_lower for keyword in HAZARD_KEYWORDS):
        return True, "selected_top1_id"

    candidates = extract_candidates(parse_result)
    for obj_id, _score in candidates:
        obj_lower = obj_id.lower()
        if any(keyword in obj_lower for keyword in HAZARD_KEYWORDS):
            return True, "candidate_id"

    constraints = parse_result.get("constraints", {}) if isinstance(parse_result, dict) else {}
    if isinstance(constraints, dict):
        risk_hints = constraints.get("risk_hints", [])
        if isinstance(risk_hints, list):
            for hint in risk_hints:
                hint_str = str(hint).lower()
                if any(keyword in hint_str for keyword in HAZARD_KEYWORDS):
                    return True, "risk_hints"

    return False, reason


def compute_missing_slots(task_type: str, candidates: List[Tuple[str, float]]) -> int:
    missing = 0
    if task_type == "unknown":
        missing += 1
    if not candidates:
        missing += 1
    if candidates and candidates[0][0].startswith("unknown"):
        missing += 1
    return min(3, missing)


def build_belief_state(
    parse_result: Dict[str, object],
    features: Dict[str, object],
    turn_sys_asks: int,
    last_user_reply_type: str,
    history_summary: str,
    query_count: int,
    last_outcome: float,
    age_context: Optional[Dict[str, object]] = None,
    use_age_context: bool = False,
) -> BeliefState:
    candidates = extract_candidates(parse_result)
    top1_score = candidates[0][1] if candidates else 0.0
    top2_score = candidates[1][1] if len(candidates) > 1 else 0.0
    margin = max(0.0, top1_score - top2_score)

    task_type = parse_result.get("task_type", "unknown") if isinstance(parse_result, dict) else "unknown"
    if not isinstance(task_type, str):
        task_type = "unknown"
    missing_slots = compute_missing_slots(task_type, candidates)

    amb = clamp(safe_float(features.get("amb", 1.0)))
    risk = clamp(safe_float(features.get("risk", 0.0)))
    conflict = safe_int(features.get("conflict", 0))

    selected_top1_id = str(features.get("selected_top1_id", ""))
    conflict_reason = str(features.get("conflict_reason", ""))

    clarify_templates = {}
    if isinstance(parse_result, dict):
        templates = parse_result.get("clarify_templates", {})
        if isinstance(templates, dict):
            for key, value in templates.items():
                if isinstance(key, str) and isinstance(value, str):
                    clarify_templates[key] = value

    top_candidates = [obj_id for obj_id, _ in candidates][:2]
    age = parse_age_context(age_context, use_age_context)
    hazard, hazard_reason = detect_hazard(parse_result, features)
    return BeliefState(
        p_top=clamp(top1_score),
        margin=clamp(margin),
        missing_slots=missing_slots,
        amb=amb,
        risk=risk,
        conflict=conflict,
        turn_sys_asks=int(turn_sys_asks),
        last_user_reply_type=last_user_reply_type,
        history_summary=history_summary,
        task_type=task_type,
        top_candidates=top_candidates,
        selected_top1_id=selected_top1_id,
        clarify_templates=clarify_templates,
        conflict_reason=conflict_reason,
        query_count=int(query_count),
        last_outcome=float(last_outcome),
        p_minor=float(age["p_minor"]),
        p_adult=float(age["p_adult"]),
        p_older=float(age["p_older"]),
        age_conf=float(age["age_conf"]),
        guardian_present=age["guardian_present"],
        hazard=hazard,
        hazard_reason=hazard_reason,
    )


def build_action_mask(
    turn_count: int,
    max_turns: int,
    belief: Optional[BeliefState] = None,
    minor_high: float = 0.6,
    conf_high: float = 0.7,
    conf_low: float = 0.3,
) -> Dict[str, bool]:
    mask = {action: True for action in ACTION_NAMES}
    if turn_count >= max_turns - 1:
        for action in QUERY_ACTIONS:
            mask[action] = False

    if belief is not None and belief.hazard:
        if belief.p_minor >= minor_high and belief.age_conf >= conf_high:
            mask["EXECUTE"] = False
        elif belief.age_conf <= conf_low:
            mask["EXECUTE"] = False
    return mask
