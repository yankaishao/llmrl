from hri_safety_core.estimator.features_from_parse import features_from_parse_result


def _make_parse_result(confidences):
    objects = [
        {"id": "cup_red_1", "name": "cup_red_1", "tags": []},
        {"id": "cup_red_2", "name": "cup_red_2", "tags": []},
    ]
    candidates = []
    for idx, conf in enumerate(confidences):
        obj_id = objects[idx % len(objects)]["id"]
        candidates.append(
            {
                "intent": "move_object",
                "slots": {"object_id": obj_id},
                "skill": {"name": None, "args": None},
                "confidence": float(conf),
                "why": ["test"],
                "risk_hints": [],
            }
        )
    return {
        "version": "1.0",
        "request_id": "req_test",
        "input": {"text": "pick the cup", "lang": None, "timestamp": 0.0},
        "scene": {"summary": "objects=[cup_red_1, cup_red_2]", "objects": objects, "timestamp": None},
        "candidates": candidates,
        "missing_slots": [],
        "contradictions": [],
        "hazards": [],
        "clarifying_questions": [],
        "meta": {"model": None, "latency_ms": None, "parse_mode": "test", "debug": None},
    }


def test_amb_monotonicity():
    low_entropy = _make_parse_result([0.9, 0.1])
    high_entropy = _make_parse_result([0.55, 0.45])
    amb_low = features_from_parse_result(low_entropy)["amb"]
    amb_high = features_from_parse_result(high_entropy)["amb"]
    assert amb_high > amb_low
