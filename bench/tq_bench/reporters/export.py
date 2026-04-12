from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


def _normalize_record(record: Any) -> dict[str, Any]:
    if is_dataclass(record):
        return asdict(record)
    if isinstance(record, dict):
        return record
    raise TypeError(f"Unsupported record type: {type(record).__name__}")


def export_json(records: Iterable[Any], path: str | Path) -> None:
    normalized = [_normalize_record(record) for record in records]
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def export_csv(records: Iterable[Any], path: str | Path) -> None:
    normalized = [_normalize_record(record) for record in records]
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not normalized:
        resolved.write_text("", encoding="utf-8")
        return

    with resolved.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(normalized[0].keys()))
        writer.writeheader()
        writer.writerows(normalized)

