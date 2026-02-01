from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PARSE_RESULT_VERSION = "1.0"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "parse_result.schema.json"


class ParseResultValidationError(Exception):
    def __init__(self, errors: List[str]):
        super().__init__("parse_result_invalid")
        self.errors = errors


def _load_schema_text() -> str:
    try:
        return SCHEMA_PATH.read_text(encoding="utf-8")
    except OSError:
        return ""


_SCHEMA_CACHE: Optional[Dict[str, object]] = None


def load_parse_result_schema() -> Dict[str, object]:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    raw = _load_schema_text()
    if raw:
        try:
            _SCHEMA_CACHE = json.loads(raw)
        except json.JSONDecodeError:
            _SCHEMA_CACHE = {}
    else:
        _SCHEMA_CACHE = {}
    return _SCHEMA_CACHE


def generate_request_id(text: str, timestamp: Optional[float] = None) -> str:
    ts = time.time() if timestamp is None else float(timestamp)
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"req_{int(ts * 1000)}_{digest}"


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def migrate_parse_result(obj: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(obj, dict):
        return {}
    version = str(obj.get("version", ""))
    if version == PARSE_RESULT_VERSION:
        return obj
    # Future migrations can be added here.
    return obj


def validate_parse_result(obj: Dict[str, object]) -> Tuple[bool, List[str]]:
    schema = load_parse_result_schema()
    if schema:
        try:
            import jsonschema

            validator = jsonschema.Draft202012Validator(schema)
            errors = []
            for err in sorted(validator.iter_errors(obj), key=lambda e: list(e.path)):
                path = ".".join([str(p) for p in err.path]) or "root"
                errors.append(f"{path}: {err.message}")
            return (len(errors) == 0, errors)
        except Exception:
            pass
    errors: List[str] = []
    _manual_validate_parse_result(obj, errors)
    return (len(errors) == 0, errors)


def _manual_validate_parse_result(obj: Dict[str, object], errors: List[str]) -> None:
    if not isinstance(obj, dict):
        errors.append("root: not an object")
        return

    required = {
        "version",
        "request_id",
        "input",
        "scene",
        "candidates",
        "missing_slots",
        "contradictions",
        "hazards",
        "clarifying_questions",
        "meta",
    }
    _check_required(obj, required, errors, "root")
    _check_no_extra(obj, required, errors, "root")

    _check_string(obj.get("version"), errors, "version")
    _check_string(obj.get("request_id"), errors, "request_id")
    _check_input(obj.get("input"), errors, "input")
    _check_scene(obj.get("scene"), errors, "scene")
    _check_candidates(obj.get("candidates"), errors, "candidates")
    _check_string_list(obj.get("missing_slots"), errors, "missing_slots")
    _check_contradictions(obj.get("contradictions"), errors, "contradictions")
    _check_hazards(obj.get("hazards"), errors, "hazards")
    _check_questions(obj.get("clarifying_questions"), errors, "clarifying_questions")
    _check_meta(obj.get("meta"), errors, "meta")


def _check_required(obj: Dict[str, object], required: set, errors: List[str], path: str) -> None:
    for key in required:
        if key not in obj:
            errors.append(f"{path}: missing {key}")


def _check_no_extra(obj: Dict[str, object], allowed: set, errors: List[str], path: str) -> None:
    for key in obj.keys():
        if key not in allowed:
            errors.append(f"{path}: unexpected field {key}")


def _check_string(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, str):
        errors.append(f"{path}: expected string")


def _check_nullable_string(value: object, errors: List[str], path: str) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        errors.append(f"{path}: expected string or null")


def _check_nullable_number(value: object, errors: List[str], path: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        errors.append(f"{path}: expected number or null")
        return
    if not isinstance(value, (int, float)):
        errors.append(f"{path}: expected number or null")


def _check_number_range(value: object, errors: List[str], path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        errors.append(f"{path}: expected number")
        return
    if value < 0.0 or value > 1.0:
        errors.append(f"{path}: out of range")


def _check_input(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected object")
        return
    required = {"text", "lang", "timestamp"}
    _check_required(value, required, errors, path)
    _check_no_extra(value, required, errors, path)
    _check_string(value.get("text"), errors, f"{path}.text")
    _check_nullable_string(value.get("lang"), errors, f"{path}.lang")
    _check_nullable_number(value.get("timestamp"), errors, f"{path}.timestamp")


def _check_scene(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected object")
        return
    required = {"summary", "objects", "timestamp"}
    _check_required(value, required, errors, path)
    _check_no_extra(value, required, errors, path)
    summary = value.get("summary")
    if summary is not None and not isinstance(summary, str):
        errors.append(f"{path}.summary: expected string or null")
    _check_nullable_number(value.get("timestamp"), errors, f"{path}.timestamp")
    objects = value.get("objects")
    if not isinstance(objects, list):
        errors.append(f"{path}.objects: expected array")
        return
    for idx, obj in enumerate(objects):
        obj_path = f"{path}.objects[{idx}]"
        if not isinstance(obj, dict):
            errors.append(f"{obj_path}: expected object")
            continue
        required_obj = {"id", "name", "tags"}
        _check_required(obj, required_obj, errors, obj_path)
        _check_no_extra(obj, required_obj, errors, obj_path)
        _check_string(obj.get("id"), errors, f"{obj_path}.id")
        _check_string(obj.get("name"), errors, f"{obj_path}.name")
        tags = obj.get("tags")
        if not isinstance(tags, list):
            errors.append(f"{obj_path}.tags: expected array")
            continue
        for jdx, tag in enumerate(tags):
            if not isinstance(tag, str):
                errors.append(f"{obj_path}.tags[{jdx}]: expected string")


def _check_candidates(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: expected array")
        return
    if len(value) < 1 or len(value) > 5:
        errors.append(f"{path}: size out of range")
    for idx, candidate in enumerate(value):
        cpath = f"{path}[{idx}]"
        if not isinstance(candidate, dict):
            errors.append(f"{cpath}: expected object")
            continue
        required = {"intent", "slots", "skill", "confidence", "why", "risk_hints"}
        _check_required(candidate, required, errors, cpath)
        _check_no_extra(candidate, required, errors, cpath)
        _check_string(candidate.get("intent"), errors, f"{cpath}.intent")
        slots = candidate.get("slots")
        if not isinstance(slots, dict):
            errors.append(f"{cpath}.slots: expected object")
        skill = candidate.get("skill")
        _check_skill(skill, errors, f"{cpath}.skill")
        _check_number_range(candidate.get("confidence"), errors, f"{cpath}.confidence")
        why = candidate.get("why")
        if not isinstance(why, list):
            errors.append(f"{cpath}.why: expected array")
        else:
            if len(why) > 5:
                errors.append(f"{cpath}.why: too many items")
            for jdx, reason in enumerate(why):
                if not isinstance(reason, str):
                    errors.append(f"{cpath}.why[{jdx}]: expected string")
                elif len(reason) > 120:
                    errors.append(f"{cpath}.why[{jdx}]: too long")
        risk_hints = candidate.get("risk_hints")
        if not isinstance(risk_hints, list):
            errors.append(f"{cpath}.risk_hints: expected array")
        else:
            for jdx, hint in enumerate(risk_hints):
                if not isinstance(hint, str):
                    errors.append(f"{cpath}.risk_hints[{jdx}]: expected string")


def _check_skill(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected object")
        return
    required = {"name", "args"}
    _check_required(value, required, errors, path)
    _check_no_extra(value, required, errors, path)
    _check_nullable_string(value.get("name"), errors, f"{path}.name")
    args = value.get("args")
    if args is not None and not isinstance(args, dict):
        errors.append(f"{path}.args: expected object or null")


def _check_string_list(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: expected array")
        return
    for idx, item in enumerate(value):
        if not isinstance(item, str):
            errors.append(f"{path}[{idx}]: expected string")


def _check_contradictions(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: expected array")
        return
    for idx, item in enumerate(value):
        ipath = f"{path}[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{ipath}: expected object")
            continue
        required = {"type", "detail", "severity"}
        _check_required(item, required, errors, ipath)
        _check_no_extra(item, required, errors, ipath)
        _check_string(item.get("type"), errors, f"{ipath}.type")
        _check_string(item.get("detail"), errors, f"{ipath}.detail")
        _check_number_range(item.get("severity"), errors, f"{ipath}.severity")


def _check_hazards(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: expected array")
        return
    for idx, item in enumerate(value):
        ipath = f"{path}[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{ipath}: expected object")
            continue
        required = {"tag", "severity", "evidence"}
        _check_required(item, required, errors, ipath)
        _check_no_extra(item, required, errors, ipath)
        _check_string(item.get("tag"), errors, f"{ipath}.tag")
        _check_number_range(item.get("severity"), errors, f"{ipath}.severity")
        evidence = item.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{ipath}.evidence: expected array")
        else:
            for jdx, ev in enumerate(evidence):
                if not isinstance(ev, str):
                    errors.append(f"{ipath}.evidence[{jdx}]: expected string")


def _check_questions(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{path}: expected array")
        return
    if len(value) > 5:
        errors.append(f"{path}: too many items")
    for idx, item in enumerate(value):
        ipath = f"{path}[{idx}]"
        if not isinstance(item, dict):
            errors.append(f"{ipath}: expected object")
            continue
        required = {"q_type", "text", "target_slots", "priority"}
        _check_required(item, required, errors, ipath)
        _check_no_extra(item, required, errors, ipath)
        _check_string(item.get("q_type"), errors, f"{ipath}.q_type")
        _check_string(item.get("text"), errors, f"{ipath}.text")
        target_slots = item.get("target_slots")
        if not isinstance(target_slots, list):
            errors.append(f"{ipath}.target_slots: expected array")
        else:
            for jdx, slot in enumerate(target_slots):
                if not isinstance(slot, str):
                    errors.append(f"{ipath}.target_slots[{jdx}]: expected string")
        _check_number_range(item.get("priority"), errors, f"{ipath}.priority")


def _check_meta(value: object, errors: List[str], path: str) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path}: expected object")
        return
    required = {"model", "latency_ms", "parse_mode", "debug"}
    _check_required(value, required, errors, path)
    _check_no_extra(value, required, errors, path)
    _check_nullable_string(value.get("model"), errors, f"{path}.model")
    latency = value.get("latency_ms")
    if latency is not None:
        if isinstance(latency, bool) or not isinstance(latency, (int, float)):
            errors.append(f"{path}.latency_ms: expected number or null")
        elif latency < 0:
            errors.append(f"{path}.latency_ms: negative")
    _check_string(value.get("parse_mode"), errors, f"{path}.parse_mode")
    debug = value.get("debug")
    if debug is not None and not isinstance(debug, dict):
        errors.append(f"{path}.debug: expected object or null")


def extract_scene_object_ids(scene: Dict[str, object]) -> List[str]:
    objects = scene.get("objects", []) if isinstance(scene, dict) else []
    ids = []
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            obj_id = item.get("id")
            if isinstance(obj_id, str) and obj_id:
                ids.append(obj_id)
    return ids


def extract_candidate_object_id(candidate: Dict[str, object]) -> str:
    slots = candidate.get("slots", {}) if isinstance(candidate, dict) else {}
    if not isinstance(slots, dict):
        return ""
    for key in ["object_id", "object", "target_object", "target_object_id", "item", "item_id"]:
        value = slots.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, dict):
            inner = value.get("id")
            if isinstance(inner, str) and inner:
                return inner
    return ""


def ensure_scene_consistency(parse_result: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(parse_result, dict):
        return parse_result
    scene = parse_result.get("scene", {})
    scene_ids = set(extract_scene_object_ids(scene if isinstance(scene, dict) else {}))
    candidates = parse_result.get("candidates", [])
    if not isinstance(candidates, list):
        return parse_result
    contradictions = parse_result.get("contradictions", [])
    if not isinstance(contradictions, list):
        contradictions = []
    updated = False
    for idx, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            continue
        obj_id = extract_candidate_object_id(candidate)
        if not obj_id:
            continue
        if scene_ids and obj_id not in scene_ids:
            contradictions.append(
                {
                    "type": "object_not_in_scene",
                    "detail": f"object_id {obj_id} not in scene.objects",
                    "severity": 0.6,
                }
            )
            confidence = candidate.get("confidence")
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
                candidate["confidence"] = clamp01(min(float(confidence), 0.25))
            updated = True
    if updated:
        parse_result["contradictions"] = contradictions
    return parse_result


def build_mock_parse_result_v1(
    text: str,
    scene_summary: str,
    scene_state_json: Optional[str] = None,
    parse_mode: str = "mock_structured",
) -> Dict[str, object]:
    from hri_safety_core.parser_utils import infer_constraints, infer_task_type, parse_objects, select_candidates
    from hri_safety_core.scene_utils import build_scene_payload

    summary_text, objects = build_scene_payload(scene_summary, scene_state_json)
    objects_for_scoring = parse_objects(scene_summary or "")
    if not objects_for_scoring and objects:
        objects_for_scoring = [{"id": obj["id"], "attrs": obj.get("tags", [])} for obj in objects]

    candidates_basic = select_candidates(objects_for_scoring, text.lower())
    task_type = infer_task_type(text.lower())
    constraints = infer_constraints(candidates_basic, objects_for_scoring, text.lower())

    intent_map = {
        "handover": "handover",
        "fetch": "move_object",
        "place": "move_object",
        "unknown": "ask_user",
    }
    intent = intent_map.get(task_type, "ask_user")
    risk_hints = constraints.get("risk_hints", []) if isinstance(constraints, dict) else []
    if not isinstance(risk_hints, list):
        risk_hints = []

    candidates = []
    for cand in candidates_basic:
        obj_id = cand.get("id", "")
        confidence = cand.get("score", 0.0)
        slots = {}
        if isinstance(obj_id, str) and obj_id and not obj_id.startswith("unknown"):
            slots = {"object_id": obj_id}
        why = ["matched scene object"] if slots else ["object not specified"]
        candidates.append(
            {
                "intent": intent,
                "slots": slots,
                "skill": {"name": None, "args": None},
                "confidence": clamp01(float(confidence)),
                "why": why,
                "risk_hints": [str(hint) for hint in risk_hints if isinstance(hint, str)],
            }
        )

    if not candidates:
        candidates = [
            {
                "intent": "ask_user",
                "slots": {},
                "skill": {"name": None, "args": None},
                "confidence": 0.2,
                "why": ["no candidates"],
                "risk_hints": [],
            }
        ]

    missing_slots = []
    top_obj = extract_candidate_object_id(candidates[0]) if candidates else ""
    if not top_obj:
        missing_slots.append("object_id")

    hazards = []
    if any("sharp" in str(h).lower() for h in risk_hints):
        hazards.append({"tag": "sharp", "severity": 0.8, "evidence": ["risk_hint"]})
    if "knife" in text.lower() or "blade" in text.lower():
        hazards.append({"tag": "knife", "severity": 0.9, "evidence": ["instruction"]})

    clarifying_questions = []
    if missing_slots:
        clarifying_questions.append(
            {
                "q_type": "object_id",
                "text": "Which object do you mean?",
                "target_slots": ["object_id"],
                "priority": 0.7,
            }
        )

    parse_result = {
        "version": PARSE_RESULT_VERSION,
        "request_id": generate_request_id(text),
        "input": {"text": text, "lang": None, "timestamp": time.time()},
        "scene": {"summary": summary_text, "objects": objects, "timestamp": None},
        "candidates": candidates[:5],
        "missing_slots": missing_slots,
        "contradictions": [],
        "hazards": hazards,
        "clarifying_questions": clarifying_questions,
        "meta": {"model": None, "latency_ms": None, "parse_mode": parse_mode, "debug": None},
    }
    return ensure_scene_consistency(parse_result)
