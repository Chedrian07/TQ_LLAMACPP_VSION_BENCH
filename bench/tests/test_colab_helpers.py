from __future__ import annotations

import os
import time
from pathlib import Path

from tq_bench.colab import (
    _repo_slug_from_url,
    build_run_bench_command,
    find_latest_run_file,
)


def test_build_run_bench_command_includes_colab_overrides(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    (repo_root / "bench").mkdir(parents=True)
    cmd = build_run_bench_command(
        repo_root,
        num=10,
        model_id="qwen3_vl_2b_instruct",
        model_quant="q4_k_m",
        runtimes=["baseline", "tq-4"],
        benchmarks=["ai2d"],
        profile="colab",
        results_dir=tmp_path / "results",
        output_path=tmp_path / "results" / "run.json",
        resume_path=tmp_path / "results" / "resume.json",
        slot_save_path=tmp_path / "kvcache",
    )

    joined = " ".join(cmd)
    assert str(repo_root / "bench" / "run_bench.py") in joined
    assert "--profile colab" in joined
    assert "--model qwen3_vl_2b_instruct" in joined
    assert "--model-quant q4_k_m" in joined
    assert "--output" in cmd
    assert "--resume" in cmd
    assert "--slot-save-path" in cmd


def test_find_latest_run_file_prefers_latest_matching_model(tmp_path: Path) -> None:
    older = tmp_path / "bench_model_a_old.json"
    newer = tmp_path / "bench_model_a_new.json"
    other = tmp_path / "bench_model_b.json"
    for idx, path in enumerate((older, other, newer)):
        path.write_text("{}", encoding="utf-8")
        ts = time.time() + idx
        os.utime(path, (ts, ts))

    assert find_latest_run_file(tmp_path, model_id="model_a") == newer
    assert find_latest_run_file(tmp_path) == newer


def test_repo_slug_from_url_is_stable() -> None:
    assert _repo_slug_from_url("https://github.com/Chedrian07/llama.cpp.git") == "Chedrian07__llama.cpp"
