#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "hri_safety_ws" / "src" / "hri_safety_core"))

from hri_safety_core.arbiter_utils import build_rl_observation  # noqa: E402
from hri_safety_core.belief_tracker import build_action_mask, build_belief_state  # noqa: E402
from hri_safety_core.estimator.features_from_parse import features_from_parse_result  # noqa: E402
from hri_safety_core.parse_result_utils import build_mock_parse_result_v1  # noqa: E402
from hri_safety_core.parser_utils import DEFAULT_SCENE_SUMMARY  # noqa: E402
from hri_safety_core.policy import RulePolicy, Sb3Policy  # noqa: E402
from hri_safety_core.qwen_structured_parser import (  # noqa: E402
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_CACHE_SIZE,
    DEFAULT_FALLBACK_MODE,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_STRUCTURED_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT_SEC,
    QwenStructuredClient,
)


def _load_age_context(json_text: str) -> Optional[dict]:
    if not json_text:
        return None
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _build_parser(parser_mode: str) -> Optional[QwenStructuredClient]:
    if parser_mode != "qwen_structured":
        return None
    return QwenStructuredClient(
        model=DEFAULT_STRUCTURED_MODEL,
        base_url=DEFAULT_BASE_URL,
        api_key_env=DEFAULT_API_KEY_ENV,
        timeout_sec=DEFAULT_TIMEOUT_SEC,
        max_retries=DEFAULT_MAX_RETRIES,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=DEFAULT_MAX_TOKENS,
        cache_size=DEFAULT_CACHE_SIZE,
        fallback_mode=DEFAULT_FALLBACK_MODE,
    )


def _select_policy(policy_path: str, obs_mode: str, max_turns: int):
    if policy_path:
        try:
            return Sb3Policy(policy_path=policy_path, obs_mode=obs_mode, max_turns=max_turns)
        except Exception:
            pass
    return RulePolicy()


def _build_obs_vector(belief, obs_mode: str, max_turns: int):
    if obs_mode == "belief":
        return belief.to_vector(max_turns=max_turns)
    if obs_mode in {"belief_v1", "extended"}:
        return belief.to_vector_v1()
    features = {"amb": belief.amb, "risk": belief.risk, "conflict": belief.conflict}
    obs = build_rl_observation(features, query_count=belief.query_count, last_outcome=belief.last_outcome)
    return obs.tolist()


def run(args: argparse.Namespace) -> None:
    text = args.text.strip()
    if not text:
        raise SystemExit("empty_text")

    parser = _build_parser(args.parser)
    if args.parser == "mock":
        parse_result = build_mock_parse_result_v1(text, args.scene_summary, args.scene_state_json)
    elif parser is not None:
        parse_result, _meta = parser.parse(
            text,
            args.scene_summary,
            scene_state_json=args.scene_state_json,
            conversation_context=args.context,
        )
    else:
        raise SystemExit("unknown_parser")

    age_context = _load_age_context(args.age_context_json)
    features = features_from_parse_result(parse_result, age_context=age_context)

    belief = build_belief_state(
        parse_result=parse_result,
        features=features,
        turn_sys_asks=0,
        last_user_reply_type="other",
        history_summary=f"User: {text}",
        query_count=0,
        last_outcome=0.0,
        age_context=age_context,
        use_age_context=bool(age_context),
    )

    action_mask = build_action_mask(turn_count=0, max_turns=args.max_turns, belief=belief)
    policy = _select_policy(args.policy_path, args.obs_mode, args.max_turns)
    decision = policy.select_action(belief, action_mask)
    obs_vector = _build_obs_vector(belief, args.obs_mode, args.max_turns)

    print("text:", text)
    print("action:", decision.action)
    print("utterance:", decision.utterance)
    print("obs_mode:", args.obs_mode)
    print("obs_dim:", len(obs_vector))
    print("obs:", obs_vector)
    print("features:", json.dumps(features, ensure_ascii=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--parser", type=str, default="mock", choices=["mock", "qwen_structured"])
    parser.add_argument("--scene-summary", type=str, default=DEFAULT_SCENE_SUMMARY)
    parser.add_argument("--scene-state-json", type=str, default="")
    parser.add_argument("--context", type=str, default="")
    parser.add_argument("--policy-path", type=str, default="")
    parser.add_argument("--obs-mode", type=str, default="legacy")
    parser.add_argument("--max-turns", type=int, default=5)
    parser.add_argument("--age-context-json", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
