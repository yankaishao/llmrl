from hri_safety_core.qwen_structured_parser import QwenStructuredClient
from hri_safety_core.parser_utils import DEFAULT_SCENE_SUMMARY


def test_fallback_behavior(monkeypatch):
    client = QwenStructuredClient(
        model="test",
        base_url="http://localhost",
        api_key_env="QWEN_API_KEY",
        timeout_sec=1.0,
        max_retries=0,
        temperature=0.0,
        max_tokens=64,
        cache_size=1,
        fallback_mode="mock",
    )

    def _bad_query(*_args, **_kwargs):
        return {"bad": "data"}

    monkeypatch.setattr(client, "_query_model", _bad_query)
    result, meta = client.parse("pick the cup", DEFAULT_SCENE_SUMMARY)
    assert meta.get("fallback") is True
    assert result.get("meta", {}).get("parse_mode") == "fallback"
