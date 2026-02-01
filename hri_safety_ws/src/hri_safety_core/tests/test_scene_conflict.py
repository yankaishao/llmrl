from hri_safety_core.estimator.features_from_parse import features_from_parse_result
from hri_safety_core.parse_result_utils import ensure_scene_consistency


def test_scene_conflict_missing_object():
    parse_result = {
        "version": "1.0",
        "request_id": "req_test",
        "input": {"text": "pick the apple", "lang": None, "timestamp": 0.0},
        "scene": {
            "summary": "objects=[cup_red_1]",
            "objects": [{"id": "cup_red_1", "name": "cup_red_1", "tags": []}],
            "timestamp": None,
        },
        "candidates": [
            {
                "intent": "move_object",
                "slots": {"object_id": "apple_1"},
                "skill": {"name": None, "args": None},
                "confidence": 0.8,
                "why": ["test"],
                "risk_hints": [],
            }
        ],
        "missing_slots": [],
        "contradictions": [],
        "hazards": [],
        "clarifying_questions": [],
        "meta": {"model": None, "latency_ms": None, "parse_mode": "test", "debug": None},
    }
    parse_result = ensure_scene_consistency(parse_result)
    assert parse_result["contradictions"]
    features = features_from_parse_result(parse_result)
    assert features["conflict"] >= 0.5
