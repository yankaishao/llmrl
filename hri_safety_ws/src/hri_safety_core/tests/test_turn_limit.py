from hri_safety_core.dialogue_core import DialogueManagerCore, STATE_IDLE
from hri_safety_core.policy.rule_policy import RulePolicy
from hri_safety_core.belief_tracker import QUERY_ACTIONS


def _mock_parse_result():
    return {
        "task_type": "unknown",
        "candidates": [
            {"id": "unknown_1", "score": 0.5},
            {"id": "unknown_2", "score": 0.5},
        ],
        "constraints": {"risk_hints": [], "fragile": False},
        "clarify_templates": {
            "yesno": "Do you mean {obj_a}?",
            "choice": "Which one do you mean: {obj_a} or {obj_b}?",
            "point": "Could you indicate which object you mean?",
        },
        "notes": "test",
    }


def _mock_features():
    return {
        "amb": 1.0,
        "risk": 0.1,
        "conflict": 0,
        "conflict_reason": "",
        "selected_top1_id": "unknown_1",
    }


def test_turn_limit_forces_terminal():
    core = DialogueManagerCore(max_turns=5, max_repeat_action=2)
    policy = RulePolicy()

    core.start_session("get the cup")
    core.update_parse_result(_mock_parse_result())
    core.update_features(_mock_features())

    for _ in range(10):
        belief = core.build_belief()
        decision = policy.select_action(belief, core.build_action_mask(belief))

        if core.should_force_terminal() or not core.can_ask() or core.query_repeat_exceeded(decision.action):
            decision = core.choose_terminal_action()
            core.apply_decision(decision)
            break

        core.apply_decision(decision)
        if decision.action in QUERY_ACTIONS:
            core.receive_user_reply("left")
            core.update_parse_result(_mock_parse_result())
            core.update_features(_mock_features())

    assert core.state == STATE_IDLE
    assert core.turn_count <= 5
