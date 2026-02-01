from hri_safety_core.dialogue_core import DialogueManagerCore


def test_context_append():
    core = DialogueManagerCore(max_turns=5, max_repeat_action=2)
    core.start_session("pick the left cup")
    core.record_system_question("CLARIFY_CHOICE", "Which one do you mean: cup_red_1 or cup_red_2?")
    core.receive_user_reply("left one")

    context = core.context
    assert "System:" in context
    assert "User:" in context
    assert "left one" in context
