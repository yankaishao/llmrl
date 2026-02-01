from hri_safety_core.belief_tracker import build_action_mask, build_belief_state
from hri_safety_core.policy.rule_policy import RulePolicy


def _mock_parse_result(obj_id: str = "knife_1"):
    return {
        "task_type": "fetch",
        "candidates": [
            {"id": obj_id, "score": 0.7},
            {"id": "cup_red_1", "score": 0.3},
        ],
        "constraints": {"risk_hints": ["sharp"], "fragile": False},
        "clarify_templates": {
            "yesno": "Do you mean {obj_a}?",
            "choice": "Which one do you mean: {obj_a} or {obj_b}?",
            "point": "Could you indicate which object you mean?",
        },
        "notes": "test",
    }


def _mock_features(obj_id: str = "knife_1"):
    return {
        "amb": 0.2,
        "risk": 0.9,
        "conflict": 0,
        "conflict_reason": "",
        "selected_top1_id": obj_id,
    }


def test_age_context_in_belief_vector():
    belief = build_belief_state(
        parse_result=_mock_parse_result(),
        features=_mock_features(),
        turn_sys_asks=1,
        last_user_reply_type="other",
        history_summary="User: pick the knife",
        query_count=0,
        last_outcome=0.0,
        age_context={
            "p_minor": 0.2,
            "p_adult": 0.7,
            "p_older": 0.1,
            "age_conf": 0.8,
            "guardian_present": False,
        },
        use_age_context=True,
    )
    vector = belief.to_vector(max_turns=5)
    assert len(vector) == 13
    assert abs(vector[-5] - 0.2) < 1e-6


def test_mask_execute_for_minor_hazard():
    belief = build_belief_state(
        parse_result=_mock_parse_result(),
        features=_mock_features(),
        turn_sys_asks=0,
        last_user_reply_type="other",
        history_summary="User: pick the knife",
        query_count=0,
        last_outcome=0.0,
        age_context={
            "p_minor": 0.95,
            "p_adult": 0.03,
            "p_older": 0.02,
            "age_conf": 0.9,
            "guardian_present": False,
        },
        use_age_context=True,
    )
    mask = build_action_mask(turn_count=0, max_turns=5, belief=belief)
    assert mask["EXECUTE"] is False


def test_low_conf_prompts_confirm():
    belief = build_belief_state(
        parse_result=_mock_parse_result(),
        features=_mock_features(),
        turn_sys_asks=0,
        last_user_reply_type="other",
        history_summary="User: pick the knife",
        query_count=0,
        last_outcome=0.0,
        age_context={
            "p_minor": 0.4,
            "p_adult": 0.3,
            "p_older": 0.3,
            "age_conf": 0.1,
            "guardian_present": None,
        },
        use_age_context=True,
    )
    policy = RulePolicy()
    mask = build_action_mask(turn_count=0, max_turns=5, belief=belief)
    decision = policy.select_action(belief, mask)
    assert decision.action == "CONFIRM_YN"
