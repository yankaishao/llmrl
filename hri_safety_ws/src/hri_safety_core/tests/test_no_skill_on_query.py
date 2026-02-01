from hri_safety_core.action_to_skill_bridge import should_dispatch_skill


def test_no_skill_on_query():
    assert should_dispatch_skill("EXECUTE")
    assert not should_dispatch_skill("CONFIRM_YN")
    assert not should_dispatch_skill("CLARIFY_CHOICE")
    assert not should_dispatch_skill("ASK_POINT")
