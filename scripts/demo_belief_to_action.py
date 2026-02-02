#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "hri_safety_ws" / "src" / "hri_safety_core"))

from hri_safety_core.arbiter_utils import action_from_index  # noqa: E402


OBS_FIELDS = {
    "legacy": ["amb", "risk", "conflict", "query_count", "last_outcome"],
    "belief_v1": [
        "amb",
        "risk",
        "conflict",
        "p_top",
        "margin",
        "entropy",
        "missing_slots_count",
        "query_count",
        "last_outcome",
        "p_minor",
        "age_conf",
        "guardian_code",
        "hazard_top",
    ],
    "belief": [
        "p_top",
        "margin",
        "missing_norm",
        "amb",
        "risk",
        "conflict",
        "turn_norm",
        "reply_code",
        "p_minor",
        "p_adult",
        "p_older",
        "age_conf",
        "guardian_code",
    ],
}


def _find_latest_policy(policies_dir: Path) -> Optional[Path]:
    if not policies_dir.is_dir():
        return None
    candidates = [path for path in policies_dir.glob("*.zip") if path.is_file()]
    if not candidates:
        return None

    def newest(paths):
        return max(paths, key=lambda path: path.stat().st_mtime)

    preferred = [path for path in candidates if path.name.startswith("best_ppo_policy")]
    if preferred:
        return newest(preferred)
    preferred = [path for path in candidates if path.name.startswith("ppo_policy")]
    if preferred:
        return newest(preferred)
    return newest(candidates)


def _find_meta_for_policy(policy_path: Path) -> Optional[Path]:
    direct = policy_path.with_suffix(".meta.json")
    if direct.is_file():
        return direct
    name = policy_path.name
    if name.startswith("best_"):
        alt = policy_path.with_name(name[len("best_") :]).with_suffix(".meta.json")
        if alt.is_file():
            return alt
    return None


def _resolve_obs_mode(args: argparse.Namespace, policy_path: Path) -> Tuple[str, str]:
    if args.obs_mode and args.obs_mode != "auto":
        return args.obs_mode, "cli"
    meta_path = _find_meta_for_policy(policy_path)
    if meta_path is not None:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
        meta_mode = meta.get("obs_mode")
        if isinstance(meta_mode, str) and meta_mode:
            return meta_mode, "meta"
    return "legacy", "default"


def _parse_obs_list(text: str) -> List[float]:
    raw = text.strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("obs_json_not_list")
        return [float(v) for v in data]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [float(p) for p in parts]


def _parse_obs_dict(text: str) -> Dict[str, float]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("obs_json_not_object")
    output: Dict[str, float] = {}
    for key, value in data.items():
        output[str(key)] = float(value)
    return output


def _build_obs_from_dict(obs_dict: Dict[str, float], fields: List[str]) -> List[float]:
    values: List[float] = []
    missing = [name for name in fields if name not in obs_dict]
    if missing:
        raise ValueError(f"missing_fields:{','.join(missing)}")
    for name in fields:
        values.append(float(obs_dict[name]))
    return values


def _load_policy(policy_path: Path):
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise RuntimeError("stable_baselines3_not_installed") from exc
    return PPO.load(str(policy_path))


def run(args: argparse.Namespace) -> None:
    policy_path = args.policy_path.strip() if args.policy_path else ""
    if not policy_path:
        latest = _find_latest_policy(ROOT / "hri_safety_ws" / "policies")
        if latest is None:
            raise SystemExit("policy_not_found")
        policy_path = str(latest)
    policy_path = Path(policy_path).resolve()

    obs_mode, obs_source = _resolve_obs_mode(args, policy_path)
    fields = OBS_FIELDS.get(obs_mode)
    if fields is None:
        raise SystemExit(f"unsupported_obs_mode:{obs_mode}")

    if args.obs_json:
        obs_dict = _parse_obs_dict(args.obs_json)
        obs_values = _build_obs_from_dict(obs_dict, fields)
    elif args.obs:
        obs_values = _parse_obs_list(args.obs)
    else:
        try:
            raw = input("Enter obs values (comma-separated or JSON list): ").strip()
        except EOFError:
            raw = ""
        obs_values = _parse_obs_list(raw)

    if len(obs_values) != len(fields):
        raise SystemExit(f"obs_dim_mismatch: expected {len(fields)}, got {len(obs_values)}")

    obs = np.array(obs_values, dtype=np.float32)
    model = _load_policy(policy_path)
    action_idx, _ = model.predict(obs, deterministic=args.deterministic)
    if isinstance(action_idx, np.ndarray):
        action_idx = int(action_idx.item())

    print("policy_path:", str(policy_path))
    print("obs_mode:", obs_mode)
    print("obs_mode_source:", obs_source)
    print("obs_dim:", len(obs_values))
    print("obs_fields:", ",".join(fields))
    print("obs:", obs_values)
    print("action_index:", int(action_idx))
    print("action_name:", action_from_index(int(action_idx)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", type=str, default="")
    parser.add_argument("--obs-mode", type=str, default="auto", choices=["auto", "legacy", "belief", "belief_v1"])
    parser.add_argument("--obs", type=str, default="")
    parser.add_argument("--obs-json", type=str, default="")
    parser.add_argument("--deterministic", action="store_true", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
