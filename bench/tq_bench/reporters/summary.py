from __future__ import annotations

from typing import Any, Iterable


def _fmt(val: Any, decimals: int = 1) -> str:
    """Format a numeric value for table display."""
    if val is None or val == 0 or val == 0.0:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _fmt_score(val: Any) -> str:
    if val is None:
        return "-"
    return f"{val:.4f}" if isinstance(val, float) else str(val)


def _get_latency_mean(row: dict[str, Any], key: str) -> str:
    stats = row.get(key)
    if isinstance(stats, dict):
        return _fmt(stats.get("mean"))
    return "-"


def _get_latency_p95(row: dict[str, Any], key: str) -> str:
    stats = row.get(key)
    if isinstance(stats, dict):
        return _fmt(stats.get("p95"))
    return "-"


def _get_throughput_mean(row: dict[str, Any]) -> str:
    stats = row.get("decode_throughput_stats")
    if isinstance(stats, dict):
        return _fmt(stats.get("mean"))
    return "-"


def render_markdown_summary(records: Iterable[dict[str, Any]]) -> str:
    rows = list(records)
    if not rows:
        return "| runtime | benchmark | status | score |\n| --- | --- | --- | --- |\n"

    has_model_id = any(row.get("model_id") for row in rows)
    has_timings = any(
        row.get("ttft_stats") and isinstance(row["ttft_stats"], dict) and row["ttft_stats"].get("mean")
        for row in rows
    )

    if has_timings:
        cols = ["runtime", "benchmark", "status", "score", "score_std",
                "TTFT_ms", "latency_p95_ms", "tok/s", "gpu_MiB"]
        if has_model_id:
            cols = ["model"] + cols
        header = "| " + " | ".join(cols) + " |"
        rule = "| " + " | ".join("---" for _ in cols) + " |"
        body = []
        for row in rows:
            cells = [
                row.get("runtime_id", ""),
                row.get("benchmark_id", ""),
                row.get("status", ""),
                _fmt_score(row.get("score")),
                _fmt(row.get("score_std"), 4),
                _get_latency_mean(row, "ttft_stats"),
                _get_latency_p95(row, "total_latency_stats"),
                _get_throughput_mean(row),
                _fmt(row.get("gpu_memory_bytes", 0) / (1024 * 1024), 0) if row.get("gpu_memory_bytes") else "-",
            ]
            if has_model_id:
                cells = [row.get("model_id", "")] + cells
            body.append("| " + " | ".join(str(c) for c in cells) + " |")
    else:
        if has_model_id:
            header = "| model | runtime | benchmark | status | score |"
            rule = "| --- | --- | --- | --- | --- |"
            body = [
                f"| {row.get('model_id', '')} | {row.get('runtime_id', '')} | "
                f"{row.get('benchmark_id', '')} | {row.get('status', '')} | "
                f"{_fmt_score(row.get('score'))} |"
                for row in rows
            ]
        else:
            header = "| runtime | benchmark | status | score |"
            rule = "| --- | --- | --- | --- |"
            body = [
                f"| {row.get('runtime_id', '')} | {row.get('benchmark_id', '')} | "
                f"{row.get('status', '')} | {_fmt_score(row.get('score'))} |"
                for row in rows
            ]
    return "\n".join([header, rule, *body]) + "\n"
