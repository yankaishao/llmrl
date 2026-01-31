import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PARSER_ORDER = ["mock", "qwen"]
ARBITER_ORDER = ["rule", "rl"]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _collect_run_rows(results_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in results_dir.rglob("*.csv"):
        if "report" in path.parts:
            continue
        if path.name in {"summary.csv", "metrics.csv"}:
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            if "success" not in reader.fieldnames or "violation" not in reader.fieldnames:
                continue
            for row in reader:
                rows.append(row)
    return rows


def _aggregate_runs(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    buckets: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        parser_mode = str(row.get("parser_mode", "")).strip()
        arbiter_mode = str(row.get("arbiter_mode", "")).strip()
        if not parser_mode or not arbiter_mode:
            continue
        key = (parser_mode, arbiter_mode)
        bucket = buckets.setdefault(
            key,
            {"episodes": 0, "success": 0.0, "violation": 0.0, "queries": 0.0, "refuse": 0.0},
        )
        bucket["episodes"] += 1
        bucket["success"] += _safe_int(row.get("success", 0))
        bucket["violation"] += _safe_int(row.get("violation", 0))
        bucket["queries"] += _safe_float(row.get("queries", 0.0))
        bucket["refuse"] += 1 if _safe_int(row.get("refuse", 0)) > 0 else 0

    metrics: List[Dict[str, object]] = []
    for (parser_mode, arbiter_mode), bucket in buckets.items():
        episodes = int(bucket["episodes"])
        if episodes <= 0:
            continue
        metrics.append(
            {
                "parser_mode": parser_mode,
                "arbiter_mode": arbiter_mode,
                "episodes": episodes,
                "success_rate": bucket["success"] / episodes,
                "safety_violation_rate": bucket["violation"] / episodes,
                "avg_queries_per_episode": bucket["queries"] / episodes,
                "refusal_rate": bucket["refuse"] / episodes,
            }
        )
    return metrics


def _read_summary(summary_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not summary_path.is_file():
        return rows
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return rows
        for row in reader:
            parser_mode = str(row.get("parser_mode", "")).strip()
            arbiter_mode = str(row.get("arbiter_mode", "")).strip()
            if not parser_mode or not arbiter_mode:
                continue
            rows.append(
                {
                    "parser_mode": parser_mode,
                    "arbiter_mode": arbiter_mode,
                    "episodes": 0,
                    "success_rate": _safe_float(row.get("success_rate", 0.0)),
                    "safety_violation_rate": _safe_float(row.get("safety_violation_rate", 0.0)),
                    "avg_queries_per_episode": _safe_float(row.get("avg_queries_per_episode", 0.0)),
                    "refusal_rate": _safe_float(row.get("refusal_rate", 0.0)),
                }
            )
    return rows


def _sort_key(row: Dict[str, object]) -> Tuple[int, int, str, str]:
    parser = str(row.get("parser_mode", ""))
    arbiter = str(row.get("arbiter_mode", ""))
    p_idx = PARSER_ORDER.index(parser) if parser in PARSER_ORDER else len(PARSER_ORDER)
    a_idx = ARBITER_ORDER.index(arbiter) if arbiter in ARBITER_ORDER else len(ARBITER_ORDER)
    return (p_idx, a_idx, parser, arbiter)


def _write_metrics_csv(metrics: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "parser_mode",
        "arbiter_mode",
        "episodes",
        "success_rate",
        "safety_violation_rate",
        "avg_queries_per_episode",
        "refusal_rate",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(metrics, key=_sort_key):
            writer.writerow(row)


def _format_cell(metric: Dict[str, object] | None) -> str:
    if not metric:
        return "n/a"
    return (
        "SR="
        f"{metric['success_rate']:.2f} "
        "SV="
        f"{metric['safety_violation_rate']:.2f} "
        "Q="
        f"{metric['avg_queries_per_episode']:.2f} "
        f"n={metric['episodes']}"
    )


def _write_summary_md(metrics: List[Dict[str, object]], out_path: Path, results_dir: Path) -> None:
    metrics_map = {
        (row["parser_mode"], row["arbiter_mode"]): row for row in metrics
    }
    lines = [
        "# Matrix Summary",
        "",
        f"Results: {results_dir}",
        "",
        "| parser \\ arbiter | rule | rl |",
        "| --- | --- | --- |",
    ]
    for parser in PARSER_ORDER:
        rule_cell = _format_cell(metrics_map.get((parser, "rule")))
        rl_cell = _format_cell(metrics_map.get((parser, "rl")))
        lines.append(f"| {parser} | {rule_cell} | {rl_cell} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric(
    metrics: List[Dict[str, object]],
    key: str,
    ylabel: str,
    out_path: Path,
    clamp_01: bool = False,
) -> None:
    metrics_map = {
        (row["parser_mode"], row["arbiter_mode"]): row for row in metrics
    }
    labels: List[str] = []
    values: List[float] = []
    for parser in PARSER_ORDER:
        for arbiter in ARBITER_ORDER:
            metric = metrics_map.get((parser, arbiter))
            if not metric:
                continue
            labels.append(f"{parser}-{arbiter}")
            values.append(float(metric[key]))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#4c78a8")
    ax.set_ylabel(ylabel)
    if clamp_01:
        ax.set_ylim(0.0, 1.0)
    else:
        max_val = max(values)
        ax.set_ylim(0.0, max(1.0, max_val * 1.1))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()

    run_rows = _collect_run_rows(results_dir)
    summary_rows = _read_summary(results_dir / "summary.csv")

    if run_rows:
        metrics = _aggregate_runs(run_rows)
    else:
        metrics = summary_rows

    if not metrics:
        raise SystemExit(f"No evaluation CSVs found under {results_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / "report"

    _write_metrics_csv(metrics, report_dir / "metrics.csv")
    _write_summary_md(metrics, report_dir / "summary.md", results_dir)

    _plot_metric(
        metrics,
        "success_rate",
        "success rate",
        report_dir / "success_rate.png",
        clamp_01=True,
    )
    _plot_metric(
        metrics,
        "safety_violation_rate",
        "safety violation rate",
        report_dir / "safety_violation_rate.png",
        clamp_01=True,
    )
    _plot_metric(
        metrics,
        "avg_queries_per_episode",
        "avg queries",
        report_dir / "avg_queries.png",
        clamp_01=False,
    )


if __name__ == "__main__":
    main()
