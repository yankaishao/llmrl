import re
from typing import Dict, List, Tuple

DEFAULT_SCENE_SUMMARY = (
    "objects=[cup_red_1(left), cup_red_2(right), knife_1(center,risky)], "
    "context=[child_present=false], task_state=idle"
)

CLARIFY_TEMPLATES = {
    "yesno": "Do you mean {obj_a}?",
    "choice": "Which one do you mean: {obj_a} or {obj_b}?",
    "point": "Could you indicate which object you mean?",
}

ALLOWED_TASK_TYPES = {"fetch", "handover", "place", "unknown"}
DEFAULT_CONSTRAINTS = {"risk_hints": [], "fragile": False}


def split_objects_list(objects_str: str) -> List[str]:
    items = []
    buf = []
    depth = 0
    for ch in objects_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            item = "".join(buf).strip()
            if item:
                items.append(item)
            buf = []
            continue
        buf.append(ch)
    last = "".join(buf).strip()
    if last:
        items.append(last)
    return items


def parse_objects(summary: str) -> List[Dict[str, object]]:
    match = re.search(r"objects=\[(.*?)]", summary)
    if not match:
        return []
    objects_raw = match.group(1).strip()
    if not objects_raw:
        return []
    objects = []
    for item in split_objects_list(objects_raw):
        obj_match = re.match(r"([A-Za-z0-9_]+)(?:\(([^)]*)\))?", item)
        if not obj_match:
            continue
        obj_id = obj_match.group(1)
        attrs_str = obj_match.group(2) or ""
        attrs = [a.strip() for a in attrs_str.split(",") if a.strip()]
        objects.append({"id": obj_id, "attrs": attrs})
    return objects


def parse_id_parts(obj_id: str) -> Tuple[str, str]:
    parts = obj_id.split("_")
    obj_type = parts[0] if parts else obj_id
    color = ""
    for part in parts[1:]:
        if not part.isdigit():
            color = part
            break
    return obj_type, color


def score_object(obj: Dict[str, object], instruction: str) -> float:
    score = 0.1
    obj_id = obj["id"]
    obj_type, color = parse_id_parts(obj_id)
    if obj_type and obj_type in instruction:
        score += 0.6
    if color and color in instruction:
        score += 0.2
    for attr in obj.get("attrs", []):
        if attr in instruction:
            score += 0.1
    return score


def _normalize_score_values(scores: List[float]) -> List[float]:
    total = sum(scores)
    if total <= 0:
        if not scores:
            return []
        return [1.0 / len(scores)] * len(scores)
    return [s / total for s in scores]


def normalize_scores(values: List[object]) -> List[object]:
    if not values:
        return []
    if isinstance(values[0], dict):
        scores = [float(item.get("score", 0.0)) for item in values]
        normalized = _normalize_score_values(scores)
        output: List[object] = []
        for item, score in zip(values, normalized):
            new_item = dict(item)
            new_item["score"] = score
            output.append(new_item)
        return output
    scores = [float(value) for value in values]
    return _normalize_score_values(scores)


def select_candidates(objects: List[Dict[str, object]], instruction: str) -> List[Dict[str, object]]:
    if not objects:
        return [
            {"id": "unknown_1", "score": 0.5},
            {"id": "unknown_2", "score": 0.5},
        ]

    scored = [(obj, score_object(obj, instruction)) for obj in objects]
    scored.sort(key=lambda item: item[1], reverse=True)

    top = scored[:2]
    if len(top) == 1:
        top.append(({"id": "unknown_1", "attrs": []}, 0.4))

    scores = normalize_scores([score for _, score in top])
    candidates = []
    for (obj, _), score in zip(top, scores):
        candidates.append({"id": obj["id"], "score": score})
    return candidates


def infer_task_type(instruction: str) -> str:
    if any(word in instruction for word in ["hand", "give", "pass", "deliver"]):
        return "handover"
    if any(word in instruction for word in ["fetch", "get", "grab", "pick", "bring", "take"]):
        return "fetch"
    return "unknown"


def infer_constraints(
    candidates: List[Dict[str, object]],
    objects: List[Dict[str, object]],
    instruction: str,
) -> Dict[str, object]:
    risky = False
    fragile = False
    candidate_ids = {c["id"] for c in candidates}
    for obj in objects:
        if obj["id"] not in candidate_ids:
            continue
        attrs = obj.get("attrs", [])
        if "risky" in attrs or "sharp" in attrs:
            risky = True
        obj_type, _ = parse_id_parts(obj["id"])
        if obj_type in {"knife", "scissors"}:
            risky = True
        if obj_type in {"glass", "cup"}:
            fragile = True
    if any(word in instruction for word in ["knife", "sharp", "blade"]):
        risky = True
    risk_hints = ["sharp"] if risky else []
    return {"risk_hints": risk_hints, "fragile": fragile}


def build_mock_parse_result(instruction: str, summary: str) -> Dict[str, object]:
    objects = parse_objects(summary)
    candidates = select_candidates(objects, instruction)
    task_type = infer_task_type(instruction)
    constraints = infer_constraints(candidates, objects, instruction)
    return build_parse_result(
        task_type=task_type,
        candidates=candidates,
        constraints=constraints,
        notes="Mock parser output",
    )


def build_parse_result(
    task_type: str,
    candidates: List[Dict[str, object]],
    constraints: Dict[str, object] | None,
    notes: str,
    clarify_templates: Dict[str, str] | None = None,
) -> Dict[str, object]:
    if task_type not in ALLOWED_TASK_TYPES:
        task_type = "unknown"

    if not isinstance(candidates, list) or len(candidates) < 2:
        candidates = [
            {"id": "unknown_1", "score": 0.5},
            {"id": "unknown_2", "score": 0.5},
        ]

    candidates = normalize_scores(candidates)

    if not isinstance(constraints, dict):
        constraints = DEFAULT_CONSTRAINTS.copy()
    risk_hints = constraints.get("risk_hints", [])
    if not isinstance(risk_hints, list):
        risk_hints = []
    fragile = bool(constraints.get("fragile", False))
    constraints = {"risk_hints": risk_hints, "fragile": fragile}

    templates = CLARIFY_TEMPLATES if clarify_templates is None else dict(clarify_templates)
    for key, value in CLARIFY_TEMPLATES.items():
        if key not in templates or not isinstance(templates.get(key), str):
            templates[key] = value

    notes = notes if isinstance(notes, str) and notes else "Parse result"

    return {
        "task_type": task_type,
        "candidates": candidates,
        "constraints": constraints,
        "clarify_templates": templates,
        "notes": notes,
    }


def validate_parse_result(obj: Dict[str, object]) -> Dict[str, object]:
    if not isinstance(obj, dict):
        raise ValueError("not_a_dict")

    for key in ["task_type", "candidates", "constraints", "clarify_templates", "notes"]:
        if key not in obj:
            raise ValueError(f"missing_{key}")

    task_type = obj.get("task_type")
    if not isinstance(task_type, str) or task_type not in ALLOWED_TASK_TYPES:
        raise ValueError("invalid_task_type")

    candidates_raw = obj.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) < 2:
        raise ValueError("invalid_candidates")

    candidates: List[Dict[str, object]] = []
    scores: List[float] = []
    for item in candidates_raw:
        if not isinstance(item, dict):
            raise ValueError("invalid_candidate_type")
        obj_id = item.get("id")
        score = item.get("score")
        if not isinstance(obj_id, str):
            raise ValueError("invalid_candidate_id")
        if not isinstance(score, (int, float)):
            raise ValueError("invalid_candidate_score")
        candidates.append({"id": obj_id, "score": float(score)})
        scores.append(float(score))

    if sum(scores) <= 0:
        raise ValueError("invalid_score_sum")

    constraints = obj.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("invalid_constraints")
    risk_hints = constraints.get("risk_hints")
    fragile = constraints.get("fragile")
    if not isinstance(risk_hints, list):
        raise ValueError("invalid_risk_hints")
    if not isinstance(fragile, bool):
        raise ValueError("invalid_fragile")

    templates = obj.get("clarify_templates")
    if not isinstance(templates, dict):
        raise ValueError("invalid_clarify_templates")
    for key in CLARIFY_TEMPLATES:
        value = templates.get(key)
        if not isinstance(value, str):
            raise ValueError(f"invalid_template_{key}")

    notes = obj.get("notes")
    if not isinstance(notes, str):
        raise ValueError("invalid_notes")

    return build_parse_result(
        task_type=task_type,
        candidates=candidates,
        constraints={"risk_hints": risk_hints, "fragile": fragile},
        notes=notes,
        clarify_templates=templates,
    )


def mock_parse(instruction: str, summary: str) -> Dict[str, object]:
    return build_mock_parse_result(instruction, summary)
