import argparse
import csv
import math
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
        if path.name in {"summary.csv", "metrics.csv", "metrics_aggregate.csv"}:
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
            {
                "episodes": 0,
                "success": 0.0,
                "task_success": 0.0,
                "safe_refusal": 0.0,
                "violation": 0.0,
                "queries": 0.0,
                "refuse": 0.0,
                "fallback": 0.0,
            },
        )
        bucket["episodes"] += 1
        bucket["success"] += _safe_int(row.get("success", 0))
        bucket["task_success"] += _safe_int(row.get("task_success", 0))
        bucket["safe_refusal"] += _safe_int(row.get("safe_refusal", 0))
        bucket["violation"] += _safe_int(row.get("violation", 0))
        bucket["queries"] += _safe_float(row.get("queries", 0.0))
        bucket["refuse"] += 1 if _safe_int(row.get("refuse", 0)) > 0 else 0
        bucket["fallback"] += 1 if _safe_int(row.get("fallback", 0)) > 0 else 0

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
                "task_success_rate": bucket["task_success"] / episodes,
                "safe_refusal_rate": bucket["safe_refusal"] / episodes,
                "safety_violation_rate": bucket["violation"] / episodes,
                "avg_queries_per_episode": bucket["queries"] / episodes,
                "refusal_rate": bucket["refuse"] / episodes,
                "fallback_rate": bucket["fallback"] / episodes,
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
                    "task_success_rate": _safe_float(row.get("task_success_rate", 0.0)),
                    "safe_refusal_rate": _safe_float(row.get("safe_refusal_rate", 0.0)),
                    "safety_violation_rate": _safe_float(row.get("safety_violation_rate", 0.0)),
                    "avg_queries_per_episode": _safe_float(row.get("avg_queries_per_episode", 0.0)),
                    "refusal_rate": _safe_float(row.get("refusal_rate", 0.0)),
                    "fallback_rate": _safe_float(row.get("fallback_rate", 0.0)),
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
        "task_success_rate",
        "safe_refusal_rate",
        "safety_violation_rate",
        "avg_queries_per_episode",
        "refusal_rate",
        "fallback_rate",
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
        "FB="
        f"{metric['fallback_rate']:.2f} "
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


def _seed_dirs(results_dir: Path) -> List[Path]:
    return sorted([path for path in results_dir.glob("seed_*") if path.is_dir()])


def _load_seed_cell_summaries(results_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for seed_dir in _seed_dirs(results_dir):
        for cell_dir in seed_dir.iterdir():
            if not cell_dir.is_dir():
                continue
            summary_path = cell_dir / "summary.csv"
            if not summary_path.is_file():
                continue
            for row in _read_summary(summary_path):
                row["seed"] = seed_dir.name
                rows.append(row)
    return rows


def _compute_stats(values: List[float]) -> Dict[str, float]:
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0, "stderr": 0.0, "ci95": 0.0, "n": 0}
    mean = sum(values) / n
    if n == 1:
        return {"mean": mean, "std": 0.0, "stderr": 0.0, "ci95": 0.0, "n": 1}
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(variance)
    stderr = std / math.sqrt(n)
    ci95 = 1.96 * stderr
    return {"mean": mean, "std": std, "stderr": stderr, "ci95": ci95, "n": n}


def _aggregate_seed_metrics(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    metric_keys = [
        "success_rate",
        "task_success_rate",
        "safe_refusal_rate",
        "safety_violation_rate",
        "avg_queries_per_episode",
        "refusal_rate",
        "fallback_rate",
    ]
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for row in rows:
        parser_mode = str(row.get("parser_mode", "")).strip()
        arbiter_mode = str(row.get("arbiter_mode", "")).strip()
        if not parser_mode or not arbiter_mode:
            continue
        key = (parser_mode, arbiter_mode)
        bucket = buckets.setdefault(key, {k: [] for k in metric_keys})
        for key_name in metric_keys:
            bucket[key_name].append(_safe_float(row.get(key_name, 0.0)))

    aggregated: List[Dict[str, object]] = []
    for (parser_mode, arbiter_mode), bucket in buckets.items():
        record: Dict[str, object] = {
            "parser_mode": parser_mode,
            "arbiter_mode": arbiter_mode,
        }
        for key_name in metric_keys:
            stats = _compute_stats(bucket[key_name])
            record[f"{key_name}_mean"] = stats["mean"]
            record[f"{key_name}_std"] = stats["std"]
            record[f"{key_name}_stderr"] = stats["stderr"]
            record[f"{key_name}_ci95"] = stats["ci95"]
            record["n"] = stats["n"]
        aggregated.append(record)
    return aggregated


def _write_metrics_aggregate_csv(metrics: List[Dict[str, object]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    metric_keys = [
        "success_rate",
        "task_success_rate",
        "safe_refusal_rate",
        "safety_violation_rate",
        "avg_queries_per_episode",
        "refusal_rate",
        "fallback_rate",
    ]
    fields = ["parser_mode", "arbiter_mode", "n"]
    for key_name in metric_keys:
        fields.extend(
            [
                f"{key_name}_mean",
                f"{key_name}_std",
                f"{key_name}_stderr",
                f"{key_name}_ci95",
            ]
        )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in sorted(metrics, key=_sort_key):
            writer.writerow(row)


def _format_cell_aggregate(metric: Dict[str, object] | None) -> str:
    if not metric:
        return "n/a"
    sr = metric.get("success_rate_mean", 0.0)
    sr_ci = metric.get("success_rate_ci95", 0.0)
    sv = metric.get("safety_violation_rate_mean", 0.0)
    fb = metric.get("fallback_rate_mean", 0.0)
    q = metric.get("avg_queries_per_episode_mean", 0.0)
    n = metric.get("n", 0)
    return f"SR={sr:.2f}±{sr_ci:.2f} SV={sv:.2f} FB={fb:.2f} Q={q:.2f} n={n}"


def _write_summary_md_aggregate(
    metrics: List[Dict[str, object]], out_path: Path, results_dir: Path
) -> None:
    metrics_map = {(row["parser_mode"], row["arbiter_mode"]): row for row in metrics}
    lines = [
        "# Matrix Summary (Aggregate)",
        "",
        f"Results: {results_dir}",
        "",
        "| parser \\ arbiter | rule | rl |",
        "| --- | --- | --- |",
    ]
    for parser in PARSER_ORDER:
        rule_cell = _format_cell_aggregate(metrics_map.get((parser, "rule")))
        rl_cell = _format_cell_aggregate(metrics_map.get((parser, "rl")))
        lines.append(f"| {parser} | {rule_cell} | {rl_cell} |")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_metric_aggregate(
    metrics: List[Dict[str, object]],
    key: str,
    ylabel: str,
    out_path: Path,
    clamp_01: bool = False,
) -> None:
    labels: List[str] = []
    values: List[float] = []
    for parser in PARSER_ORDER:
        for arbiter in ARBITER_ORDER:
            metric = next(
                (row for row in metrics if row["parser_mode"] == parser and row["arbiter_mode"] == arbiter),
                None,
            )
            if not metric:
                continue
            labels.append(f"{parser}-{arbiter}")
            values.append(float(metric.get(f"{key}_mean", 0.0)))

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


def _collect_reason_counts(results_dir: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, int]], Dict[Tuple[str, str], Dict[str, int]]]:
    failure_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    fallback_counts: Dict[Tuple[str, str], Dict[str, int]] = {}
    for path in results_dir.rglob("*.csv"):
        if path.name in {"summary.csv", "metrics.csv", "metrics_aggregate.csv"}:
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            if "failure_reason" not in reader.fieldnames and "fallback_reason" not in reader.fieldnames:
                continue
            for row in reader:
                parser_mode = str(row.get("parser_mode", "")).strip()
                arbiter_mode = str(row.get("arbiter_mode", "")).strip()
                if not parser_mode or not arbiter_mode:
                    continue
                key = (parser_mode, arbiter_mode)
                failure_reason = str(row.get("failure_reason", "")).strip()
                if failure_reason:
                    failure_counts.setdefault(key, {})
                    failure_counts[key][failure_reason] = failure_counts[key].get(failure_reason, 0) + 1
                fallback_reason = str(row.get("fallback_reason", "")).strip()
                if fallback_reason:
                    fallback_counts.setdefault(key, {})
                    fallback_counts[key][fallback_reason] = fallback_counts[key].get(fallback_reason, 0) + 1
    return failure_counts, fallback_counts


def _write_reasons_outputs(
    failure_counts: Dict[Tuple[str, str], Dict[str, int]],
    fallback_counts: Dict[Tuple[str, str], Dict[str, int]],
    out_csv: Path,
    out_md: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for (parser_mode, arbiter_mode), counts in failure_counts.items():
        total = sum(counts.values()) or 1
        for reason, count in counts.items():
            rows.append(
                {
                    "parser_mode": parser_mode,
                    "arbiter_mode": arbiter_mode,
                    "reason_type": "failure_reason",
                    "reason": reason,
                    "count": count,
                    "fraction": count / total,
                }
            )
    for (parser_mode, arbiter_mode), counts in fallback_counts.items():
        total = sum(counts.values()) or 1
        for reason, count in counts.items():
            rows.append(
                {
                    "parser_mode": parser_mode,
                    "arbiter_mode": arbiter_mode,
                    "reason_type": "fallback_reason",
                    "reason": reason,
                    "count": count,
                    "fraction": count / total,
                }
            )
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["parser_mode", "arbiter_mode", "reason_type", "reason", "count", "fraction"],
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Reason Breakdown", ""]
    for parser in PARSER_ORDER:
        for arbiter in ARBITER_ORDER:
            lines.append(f"## {parser}-{arbiter}")
            fail = failure_counts.get((parser, arbiter), {})
            fb = fallback_counts.get((parser, arbiter), {})
            lines.append("")
            lines.append("Failure reasons:")
            if fail:
                for reason, count in sorted(fail.items(), key=lambda item: item[1], reverse=True)[:3]:
                    lines.append(f"- {reason}: {count}")
            else:
                lines.append("- n/a")
            lines.append("")
            lines.append("Fallback reasons:")
            if fb:
                for reason, count in sorted(fb.items(), key=lambda item: item[1], reverse=True)[:3]:
                    lines.append(f"- {reason}: {count}")
            else:
                lines.append("- n/a")
            lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[1]
    report_dir = repo_root / "report"

    seed_dirs = _seed_dirs(results_dir)
    if seed_dirs:
        seed_rows = _load_seed_cell_summaries(results_dir)
        if not seed_rows:
            raise SystemExit(f"No per-cell summary.csv found under {results_dir}")
        metrics_agg = _aggregate_seed_metrics(seed_rows)
        _write_metrics_aggregate_csv(metrics_agg, report_dir / "metrics_aggregate.csv")
        _write_summary_md_aggregate(metrics_agg, report_dir / "summary.md", results_dir)

        _plot_metric_aggregate(
            metrics_agg,
            "success_rate",
            "success rate",
            report_dir / "success_rate.png",
            clamp_01=True,
        )
        _plot_metric_aggregate(
            metrics_agg,
            "safety_violation_rate",
            "safety violation rate",
            report_dir / "safety_violation_rate.png",
            clamp_01=True,
        )
        _plot_metric_aggregate(
            metrics_agg,
            "avg_queries_per_episode",
            "avg queries",
            report_dir / "avg_queries.png",
            clamp_01=False,
        )
        _plot_metric_aggregate(
            metrics_agg,
            "fallback_rate",
            "fallback rate",
            report_dir / "fallback_rate.png",
            clamp_01=True,
        )
    else:
        run_rows = _collect_run_rows(results_dir)
        summary_rows = _read_summary(results_dir / "summary.csv")

        if run_rows:
            metrics = _aggregate_runs(run_rows)
        else:
            metrics = summary_rows

        if not metrics:
            raise SystemExit(f"No evaluation CSVs found under {results_dir}")

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
        _plot_metric(
            metrics,
            "fallback_rate",
            "fallback rate",
            report_dir / "fallback_rate.png",
            clamp_01=True,
        )

    failure_counts, fallback_counts = _collect_reason_counts(results_dir)
    _write_reasons_outputs(
        failure_counts,
        fallback_counts,
        report_dir / "reasons.csv",
        report_dir / "reasons.md",
    )


if __name__ == "__main__":
    main()
