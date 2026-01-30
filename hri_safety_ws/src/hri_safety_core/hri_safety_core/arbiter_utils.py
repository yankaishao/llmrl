from typing import Dict, List, Tuple

import numpy as np

A_HIGH_DEFAULT = 0.4
R_HIGH_DEFAULT = 0.7

ACTION_MAP = {
    0: "EXECUTE",
    1: "CONFIRM_YN",
    2: "CLARIFY_CHOICE",
    3: "ASK_POINT",
    4: "REFUSE_SAFE",
}

QUERY_ACTIONS = {"CONFIRM_YN", "CLARIFY_CHOICE", "ASK_POINT"}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def extract_candidates(parse_result: Dict[str, object]) -> List[Tuple[str, float]]:
    raw = parse_result.get("candidates", [])
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
    return candidates


def top_two_candidates(candidates: List[Tuple[str, float]]) -> List[str]:
    if not candidates:
        return []
    sorted_candidates = sorted(candidates, key=lambda item: item[1], reverse=True)
    ids: List[str] = []
    for obj_id, _ in sorted_candidates:
        if obj_id not in ids:
            ids.append(obj_id)
        if len(ids) == 2:
            break
    return ids


def action_from_index(index: int) -> str:
    return ACTION_MAP.get(index, "EXECUTE")


def rule_based_action(conflict: int, risk: float, amb: float, has_choice: bool, a_high: float, r_high: float) -> str:
    if conflict == 1:
        return "REFUSE_SAFE"
    if risk >= r_high:
        return "CONFIRM_YN"
    if amb >= a_high:
        return "CLARIFY_CHOICE" if has_choice else "ASK_POINT"
    return "EXECUTE"


def utterance_for_action(action: str, selected_id: str, conflict_reason: str, top2: List[str]) -> str:
    if action == "EXECUTE":
        return f"Executing: {selected_id}" if selected_id else "Executing."
    if action == "CONFIRM_YN":
        return f"Do you mean {selected_id}?" if selected_id else "Do you mean this object?"
    if action == "CLARIFY_CHOICE":
        if len(top2) >= 2:
            return f"Which one do you mean: {top2[0]} or {top2[1]}?"
        return "Please indicate which object you mean."
    if action == "ASK_POINT":
        return "Please indicate which object you mean."
    if action == "REFUSE_SAFE":
        reason = conflict_reason or "unspecified"
        return f"I cannot do that because it may be unsafe or not feasible. Reason: {reason}"
    return ""


def build_rl_observation(
    features: Dict[str, object],
    query_count: float = 0.0,
    last_outcome: float = 0.0,
) -> np.ndarray:
    amb = clamp(safe_float(features.get("amb", 0.0)))
    risk = clamp(safe_float(features.get("risk", 0.0)))
    conflict = float(safe_int(features.get("conflict", 0)))
    query_value = safe_float(query_count, 0.0)
    outcome_value = safe_float(last_outcome, 0.0)
    return np.array([amb, risk, conflict, query_value, outcome_value], dtype=np.float32)
