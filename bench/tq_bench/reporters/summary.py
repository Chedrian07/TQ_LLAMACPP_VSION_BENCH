from __future__ import annotations

from typing import Any, Iterable


def render_markdown_summary(records: Iterable[dict[str, Any]]) -> str:
    rows = list(records)
    if not rows:
        return "| runtime | benchmark | status | score |\n| --- | --- | --- | --- |\n"

    has_model_id = any(row.get("model_id") for row in rows)
    if has_model_id:
        header = "| model | runtime | benchmark | status | score |"
        rule = "| --- | --- | --- | --- | --- |"
        body = [
            f"| {row.get('model_id', '')} | {row.get('runtime_id', '')} | "
            f"{row.get('benchmark_id', '')} | {row.get('status', '')} | "
            f"{row.get('score', '')} |"
            for row in rows
        ]
    else:
        header = "| runtime | benchmark | status | score |"
        rule = "| --- | --- | --- | --- |"
        body = [
            f"| {row.get('runtime_id', '')} | {row.get('benchmark_id', '')} | "
            f"{row.get('status', '')} | {row.get('score', '')} |"
            for row in rows
        ]
    return "\n".join([header, rule, *body]) + "\n"
