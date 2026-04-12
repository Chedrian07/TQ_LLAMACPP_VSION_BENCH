from __future__ import annotations

from typing import Any, Iterable


def render_markdown_summary(records: Iterable[dict[str, Any]]) -> str:
    rows = list(records)
    if not rows:
        return "| runtime | benchmark | status | score |\n| --- | --- | --- | --- |\n"

    header = "| runtime | benchmark | status | score |"
    rule = "| --- | --- | --- | --- |"
    body = [
        f"| {row.get('runtime_id', '')} | {row.get('benchmark_id', '')} | {row.get('status', '')} | {row.get('score', '')} |"
        for row in rows
    ]
    return "\n".join([header, rule, *body]) + "\n"

