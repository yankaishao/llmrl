from hri_safety_core.parse_result_utils import build_mock_parse_result_v1, validate_parse_result
from hri_safety_core.parser_utils import DEFAULT_SCENE_SUMMARY


def test_schema_validation_passes():
    result = build_mock_parse_result_v1("pick up the red cup", DEFAULT_SCENE_SUMMARY)
    ok, errors = validate_parse_result(result)
    assert ok, errors


def test_schema_validation_missing_field():
    result = build_mock_parse_result_v1("pick up the red cup", DEFAULT_SCENE_SUMMARY)
    result.pop("candidates", None)
    ok, _errors = validate_parse_result(result)
    assert ok is False


def test_schema_validation_range():
    result = build_mock_parse_result_v1("pick up the red cup", DEFAULT_SCENE_SUMMARY)
    result["candidates"][0]["confidence"] = 1.5
    ok, _errors = validate_parse_result(result)
    assert ok is False
