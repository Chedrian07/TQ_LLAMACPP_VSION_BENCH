"""Tests for :mod:`tq_bench.kv_analysis.report`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tq_bench.kv_analysis.report import generate_full_report
from tq_bench.kv_analysis.tests.conftest import (
    SyntheticDumpSpec,
    build_pair_of_dumps,
)


def test_generate_full_report_writes_all_artifacts(tmp_path: Path) -> None:
    spec = SyntheticDumpSpec(
        n_tokens=16,
        n_layers=2,
        n_kv_head=2,
        head_dim=16,
        vision_token_count=8,
    )
    baseline, quant = build_pair_of_dumps(tmp_path, spec, noise=0.05)

    out = generate_full_report(
        baseline_dump=baseline,
        quant_dumps={"q": quant},
        output_dir=tmp_path / "report",
        make_plots=True,
        bits_by_run={"q": 3},
    )

    for key in (
        "distribution_stats",
        "outliers_per_layer",
        "quant_errors",
        "quant_theoretical_comparison",
        "rotation_analysis",
        "report_md",
    ):
        assert key in out
        assert out[key].exists(), f"missing artifact {key}"

    # CSVs should be loadable.
    dist = pd.read_csv(out["distribution_stats"])
    assert "run" in dist.columns
    assert {"baseline", "q"}.issubset(dist["run"].unique())

    quant_df = pd.read_csv(out["quant_errors"])
    assert "per_coord_mse" in quant_df.columns

    md_text = out["report_md"].read_text(encoding="utf-8")
    assert "# KV cache analysis report" in md_text
    assert "Distribution" in md_text
