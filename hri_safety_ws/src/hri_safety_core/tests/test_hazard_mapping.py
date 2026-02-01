from hri_safety_core.estimator.features_from_parse import features_from_parse_result


def test_hazard_mapping_knife():
    parse_result = {
        "version": "1.0",
        "request_id": "req_test",
        "input": {"text": "hand me the knife", "lang": None, "timestamp": 0.0},
        "scene": {
            "summary": "objects=[knife_1]",
            "objects": [{"id": "knife_1", "name": "knife_1", "tags": ["sharp"]}],
            "timestamp": None,
        },
        "candidates": [
            {
                "intent": "handover",
                "slots": {"object_id": "knife_1"},
                "skill": {"name": None, "args": None},
                "confidence": 0.9,
                "why": ["user asked for knife"],
                "risk_hints": ["sharp"],
            }
        ],
        "missing_slots": [],
        "contradictions": [],
        "hazards": [{"tag": "knife", "severity": 0.9, "evidence": ["instruction"]}],
        "clarifying_questions": [],
        "meta": {"model": None, "latency_ms": None, "parse_mode": "test", "debug": None},
    }
    features = features_from_parse_result(parse_result)
    assert features["hazard_tag_top"] == "knife"
    assert features["risk"] >= 0.5
