"""Shared test fixtures for evaluator parity tests."""

from __future__ import annotations

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list[dict]:
    """Load a JSON golden fixture file."""
    path = FIXTURES_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))
