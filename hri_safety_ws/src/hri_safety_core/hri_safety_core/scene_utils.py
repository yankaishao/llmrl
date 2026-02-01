from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from hri_safety_core.parser_utils import parse_objects


def extract_objects_from_summary(summary: str) -> List[Dict[str, object]]:
    objects = []
    for obj in parse_objects(summary):
        obj_id = str(obj.get("id", "")).strip()
        if not obj_id:
            continue
        attrs = obj.get("attrs", [])
        tags = [str(tag) for tag in attrs if isinstance(tag, str)]
        objects.append({"id": obj_id, "name": obj_id, "tags": tags})
    return objects


def extract_objects_from_state_json(state_json: str) -> List[Dict[str, object]]:
    if not state_json:
        return []
    try:
        payload = json.loads(state_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    raw_objects = payload.get("objects", [])
    if not isinstance(raw_objects, list):
        return []
    objects = []
    for item in raw_objects:
        if not isinstance(item, dict):
            continue
        obj_id = str(item.get("id", "")).strip()
        if not obj_id:
            continue
        tags = []
        if isinstance(item.get("tags"), list):
            tags = [str(tag) for tag in item.get("tags") if isinstance(tag, str)]
        objects.append({"id": obj_id, "name": obj_id, "tags": tags})
    return objects


def merge_scene_objects(
    summary_objects: List[Dict[str, object]],
    state_objects: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    merged: Dict[str, Dict[str, object]] = {}
    for obj in summary_objects + state_objects:
        obj_id = str(obj.get("id", "")).strip()
        if not obj_id:
            continue
        existing = merged.get(obj_id, {"id": obj_id, "name": obj_id, "tags": []})
        tags = set(existing.get("tags", []))
        for tag in obj.get("tags", []):
            if isinstance(tag, str):
                tags.add(tag)
        merged[obj_id] = {"id": obj_id, "name": str(obj.get("name", obj_id)), "tags": sorted(tags)}
    return list(merged.values())


def build_scene_payload(summary: str, state_json: Optional[str]) -> Tuple[Optional[str], List[Dict[str, object]]]:
    summary = summary or ""
    summary_objects = extract_objects_from_summary(summary)
    state_objects = extract_objects_from_state_json(state_json or "")
    objects = merge_scene_objects(summary_objects, state_objects)
    return summary if summary else None, objects
