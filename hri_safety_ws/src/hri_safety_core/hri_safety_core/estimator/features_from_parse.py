from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from hri_safety_core.arbiter_utils import clamp
from hri_safety_core.belief_tracker import parse_age_context
from hri_safety_core.parse_result_utils import extract_candidate_object_id, extract_scene_object_ids

HAZARD_WEIGHTS = {
    "knife": 1.0,
    "blade": 1.0,
    "sharp": 0.9,
    "fire": 0.9,
    "chemical": 0.85,
    "force": 0.9,
    "privacy": 0.6,
    "glass": 0.6,
}

INTENT_RISK = {
    "handover": 0.2,
    "move_object": 0.1,
    "point_to": 0.05,
    "execute_skill": 0.1,
    "ask_user": 0.0,
    "refuse": 0.0,
    "fallback": 0.0,
}

CRITICAL_SLOTS = {"object_id", "target_location", "recipient", "tool"}


def _normalize_probs(values: List[float]) -> List[float]:
    total = sum(values)
    if total <= 0:
        return [1.0 / len(values)] * len(values) if values else []
    return [val / total for val in values]


def _entropy(values: List[float]) -> float:
    if not values:
        return 0.0
    probs = _normalize_probs(values)
    if len(probs) <= 1:
        return 0.0
    entropy = 0.0
    for p in probs:
        if p <= 0:
            continue
        entropy -= p * math.log(p + 1e-12)
    return clamp(entropy / math.log(len(probs)))


def _is_v1_parse_result(parse_result: Dict[str, object]) -> bool:
    if not isinstance(parse_result, dict):
        return False
    if "version" not in parse_result or "input" not in parse_result:
        return False
    candidates = parse_result.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return False
    first = candidates[0]
    return isinstance(first, dict) and "intent" in first and "confidence" in first


def _extract_candidates_v1(parse_result: Dict[str, object]) -> List[Tuple[str, float]]:
    candidates_raw = parse_result.get("candidates", [])
    candidates: List[Tuple[str, float]] = []
    if not isinstance(candidates_raw, list):
        return candidates
    for cand in candidates_raw:
        if not isinstance(cand, dict):
            continue
        obj_id = extract_candidate_object_id(cand)
        confidence = cand.get("confidence", 0.0)
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            candidates.append((obj_id, float(confidence)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _extract_candidates_legacy(parse_result: Dict[str, object]) -> List[Tuple[str, float]]:
    candidates_raw = parse_result.get("candidates", [])
    candidates: List[Tuple[str, float]] = []
    if not isinstance(candidates_raw, list):
        return candidates
    for cand in candidates_raw:
        if not isinstance(cand, dict):
            continue
        obj_id = cand.get("id")
        score = cand.get("score")
        if isinstance(obj_id, str) and isinstance(score, (int, float)) and not isinstance(score, bool):
            candidates.append((obj_id, float(score)))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return candidates


def _hazard_from_candidates(candidates: List[Tuple[str, float]]) -> Tuple[str, float]:
    if not candidates:
        return "", 0.0
    top_id = candidates[0][0].lower() if candidates[0][0] else ""
    for tag, weight in HAZARD_WEIGHTS.items():
        if tag in top_id:
            return tag, weight
    return "", 0.0


def _hazard_from_parse_v1(parse_result: Dict[str, object]) -> Tuple[str, float]:
    hazards = parse_result.get("hazards", []) if isinstance(parse_result, dict) else []
    if not isinstance(hazards, list) or not hazards:
        return "", 0.0
    best_tag = ""
    best_score = 0.0
    for hazard in hazards:
        if not isinstance(hazard, dict):
            continue
        tag = hazard.get("tag")
        severity = hazard.get("severity")
        if not isinstance(tag, str) or not isinstance(severity, (int, float)):
            continue
        weight = HAZARD_WEIGHTS.get(tag.lower(), 0.4)
        score = float(severity) * weight
        if score > best_score:
            best_score = score
            best_tag = tag
    return best_tag, clamp(best_score)


def _intent_from_v1(parse_result: Dict[str, object]) -> str:
    candidates = parse_result.get("candidates", []) if isinstance(parse_result, dict) else []
    if not isinstance(candidates, list):
        return ""
    top = None
    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        confidence = cand.get("confidence")
        if not isinstance(confidence, (int, float)):
            continue
        if top is None or float(confidence) > float(top.get("confidence", 0.0)):
            top = cand
    if not isinstance(top, dict):
        return ""
    intent = top.get("intent")
    return intent if isinstance(intent, str) else ""


def _risk_from_intent(intent: str) -> float:
    return INTENT_RISK.get(intent, 0.05)


def _risk_from_age_context(age_context: Optional[Dict[str, object]], hazard_tag: str) -> float:
    if not age_context:
        return 0.0
    age = parse_age_context(age_context, use_age_context=True)
    p_minor = float(age.get("p_minor", 0.0))
    age_conf = float(age.get("age_conf", 0.0))
    guardian_present = age.get("guardian_present")
    if p_minor < 0.6 or age_conf < 0.7:
        return 0.0
    if hazard_tag.lower() not in {"knife", "blade", "sharp", "fire", "chemical", "force"}:
        return 0.0
    extra = 0.2
    if guardian_present is False:
        extra += 0.1
    return extra


def _amb_score(p_top: float, margin: float, entropy: float, missing_slots_count: int) -> float:
    missing_norm = clamp(float(missing_slots_count) / 3.0)
    return clamp(0.35 * (1.0 - p_top) + 0.25 * (1.0 - margin) + 0.25 * entropy + 0.15 * missing_norm)


def _missing_slots_legacy(parse_result: Dict[str, object], candidates: List[Tuple[str, float]]) -> List[str]:
    missing = []
    task_type = parse_result.get("task_type", "unknown")
    if not isinstance(task_type, str) or task_type == "unknown":
        missing.append("task_type")
    if not candidates:
        missing.append("object_id")
    elif candidates[0][0].startswith("unknown"):
        missing.append("object_id")
    return missing


def _contradictions_from_scene(scene_ids: List[str], candidates: List[Tuple[str, float]]) -> List[Dict[str, object]]:
    contradictions = []
    if not scene_ids:
        return contradictions
    for obj_id, _score in candidates:
        if obj_id and obj_id not in scene_ids:
            contradictions.append(
                {
                    "type": "object_not_in_scene",
                    "detail": f"object_id {obj_id} not in scene.objects",
                    "severity": 0.6,
                }
            )
            break
    return contradictions


def _conflict_score(contradictions: List[Dict[str, object]]) -> float:
    if not contradictions:
        return 0.0
    total = 0.0
    max_sev = 0.0
    for item in contradictions:
        if not isinstance(item, dict):
            continue
        severity = item.get("severity")
        if isinstance(severity, (int, float)) and not isinstance(severity, bool):
            total += float(severity)
            max_sev = max(max_sev, float(severity))
    return clamp(max(max_sev, total / max(1.0, len(contradictions))))


def _suggested_q_types(parse_result: Dict[str, object]) -> List[str]:
    questions = parse_result.get("clarifying_questions", [])
    if not isinstance(questions, list):
        return []
    scored = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        q_type = item.get("q_type")
        priority = item.get("priority", 0.0)
        if not isinstance(q_type, str):
            continue
        if isinstance(priority, (int, float)) and not isinstance(priority, bool):
            scored.append((q_type, float(priority)))
        else:
            scored.append((q_type, 0.0))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [q for q, _ in scored[:3]]


def _build_action_mask(
    hazard_tag: str,
    hazard_severity: float,
    age_context: Optional[Dict[str, object]],
) -> Dict[str, int]:
    mask = {
        "EXECUTE": 1,
        "CONFIRM_YN": 1,
        "CLARIFY_CHOICE": 1,
        "ASK_POINT": 1,
        "REFUSE_SAFE": 1,
        "FALLBACK_HUMAN_HELP": 1,
    }
    if not age_context:
        return mask
    age = parse_age_context(age_context, use_age_context=True)
    p_minor = float(age.get("p_minor", 0.0))
    age_conf = float(age.get("age_conf", 0.0))
    guardian = age.get("guardian_present")
    if hazard_severity >= 0.7 and p_minor >= 0.7 and age_conf >= 0.7 and guardian is False:
        mask["EXECUTE"] = 0
    return mask


def features_from_parse_result(
    parse_result: Dict[str, object],
    age_context: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    if _is_v1_parse_result(parse_result):
        candidates = _extract_candidates_v1(parse_result)
        confidences = [score for _obj_id, score in candidates]
        p_top = max(confidences) if confidences else 0.0
        margin = p_top - (confidences[1] if len(confidences) > 1 else 0.0)
        entropy = _entropy(confidences)
        missing_slots = parse_result.get("missing_slots", [])
        if not isinstance(missing_slots, list):
            missing_slots = []
        missing_slots_count = len([slot for slot in missing_slots if isinstance(slot, str)])
        missing_slots_critical = len(
            [slot for slot in missing_slots if isinstance(slot, str) and slot in CRITICAL_SLOTS]
        )

        hazard_tag, hazard_severity = _hazard_from_parse_v1(parse_result)
        intent = _intent_from_v1(parse_result)
        base_hazard_risk = hazard_severity
        risk = clamp(base_hazard_risk + _risk_from_intent(intent) + _risk_from_age_context(age_context, hazard_tag))

        contradictions = parse_result.get("contradictions", [])
        if not isinstance(contradictions, list):
            contradictions = []
        scene_ids = extract_scene_object_ids(parse_result.get("scene", {}))
        contradictions += _contradictions_from_scene(scene_ids, candidates)
        conflict_score = _conflict_score(contradictions)
        conflict = 1.0 if conflict_score >= 0.5 else 0.0
        conflict_reason = "contradiction" if conflict_score > 0 else ""

        amb = _amb_score(p_top, margin, entropy, missing_slots_count)
        selected_top1_id = candidates[0][0] if candidates else ""
        suggested_q = _suggested_q_types(parse_result)
        action_mask = _build_action_mask(hazard_tag, hazard_severity, age_context)

        return {
            "p_top": clamp(p_top),
            "margin": clamp(margin),
            "entropy": clamp(entropy),
            "amb": clamp(amb),
            "risk": clamp(risk),
            "conflict": conflict,
            "conflict_score": clamp(conflict_score),
            "conflict_reason": conflict_reason,
            "missing_slots_count": missing_slots_count,
            "missing_slots_critical": missing_slots_critical,
            "hazard_tag_top": hazard_tag,
            "hazard_severity_top": clamp(hazard_severity),
            "selected_top1_id": selected_top1_id,
            "suggested_question_types": suggested_q,
            "action_mask": action_mask,
            "top_intent": intent,
        }

    candidates = _extract_candidates_legacy(parse_result)
    confidences = [score for _obj_id, score in candidates]
    p_top = max(confidences) if confidences else 0.0
    margin = p_top - (confidences[1] if len(confidences) > 1 else 0.0)
    entropy = _entropy(confidences)
    missing_slots = _missing_slots_legacy(parse_result, candidates)
    missing_slots_count = len(missing_slots)
    missing_slots_critical = len([slot for slot in missing_slots if slot in CRITICAL_SLOTS])

    hazard_tag, hazard_severity = _hazard_from_candidates(candidates)
    constraints = parse_result.get("constraints", {}) if isinstance(parse_result, dict) else {}
    if isinstance(constraints, dict):
        risk_hints = constraints.get("risk_hints", [])
        if isinstance(risk_hints, list):
            for hint in risk_hints:
                if isinstance(hint, str) and "sharp" in hint.lower():
                    hazard_tag = hazard_tag or "sharp"
                    hazard_severity = max(hazard_severity, 1.0)

    intent = str(parse_result.get("task_type", "unknown"))
    risk = clamp(hazard_severity + _risk_from_intent(intent) + _risk_from_age_context(age_context, hazard_tag))

    scene_ids = extract_scene_object_ids(parse_result.get("scene", {}))
    contradictions = _contradictions_from_scene(scene_ids, candidates)
    conflict_score = _conflict_score(contradictions)
    conflict = 1.0 if conflict_score >= 0.5 else 0.0
    conflict_reason = "unknown_task" if intent == "unknown" else ""
    if conflict_score > 0 and not conflict_reason:
        conflict_reason = "object_not_in_scene"

    amb = _amb_score(p_top, margin, entropy, missing_slots_count)
    selected_top1_id = candidates[0][0] if candidates else ""

    return {
        "p_top": clamp(p_top),
        "margin": clamp(margin),
        "entropy": clamp(entropy),
        "amb": clamp(amb),
        "risk": clamp(risk),
        "conflict": conflict,
        "conflict_score": clamp(conflict_score),
        "conflict_reason": conflict_reason,
        "missing_slots_count": missing_slots_count,
        "missing_slots_critical": missing_slots_critical,
        "hazard_tag_top": hazard_tag,
        "hazard_severity_top": clamp(hazard_severity),
        "selected_top1_id": selected_top1_id,
        "suggested_question_types": [],
        "action_mask": _build_action_mask(hazard_tag, hazard_severity, age_context),
        "top_intent": intent,
    }
