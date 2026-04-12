from __future__ import annotations

from pathlib import Path

from tq_bench.config import BenchmarkConfig, ExperimentCell, ModelConfig, RuntimeConfig
from tq_bench.orchestrator import Orchestrator, OrchestratorConfig
from tq_bench.runner import RunRecord


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        checkpoint_path=tmp_path / "checkpoint.json",
        results_dir=tmp_path / "results",
        runtimes_yaml=tmp_path / "runtimes.yaml",
        benchmarks_yaml=tmp_path / "benchmarks.yaml",
        models_yaml=tmp_path / "models.yaml",
        server_binary=tmp_path / "llama-server",
        model_ids=("thinking", "instruct"),
        num_gpus=2,
    )


def test_build_parallel_lanes_respects_model_gpu_affinity(tmp_path: Path) -> None:
    orch = Orchestrator(_make_config(tmp_path))
    thinking = ModelConfig(
        id="thinking",
        family="qwen3-vl",
        model_path=tmp_path / "thinking.gguf",
        gpu_id=0,
        port=8080,
        parallel_requests=2,
    )
    instruct = ModelConfig(
        id="instruct",
        family="qwen3-vl",
        model_path=tmp_path / "instruct.gguf",
        gpu_id=1,
        port=8081,
        parallel_requests=4,
    )

    lanes = orch._build_parallel_lanes([thinking, instruct])

    assert [(lane.model_config.id, lane.gpu_id, lane.port, lane.parallel_requests) for lane in lanes] == [
        ("thinking", 0, 8080, 2),
        ("instruct", 1, 8081, 4),
    ]


def test_assign_cells_to_matching_model_lanes(tmp_path: Path) -> None:
    orch = Orchestrator(_make_config(tmp_path))
    runtime = RuntimeConfig(
        id="baseline",
        method="none",
        cache_type_k="f16",
        cache_type_v="f16",
        bits="16",
    )
    benchmark = BenchmarkConfig(
        id="ai2d",
        task_type="vlm",
        sample_count=5,
        metric="option_match",
    )
    thinking = ModelConfig(
        id="thinking",
        family="qwen3-vl",
        model_path=tmp_path / "thinking.gguf",
        gpu_id=0,
    )
    instruct = ModelConfig(
        id="instruct",
        family="qwen3-vl",
        model_path=tmp_path / "instruct.gguf",
        gpu_id=1,
    )

    lanes = orch._build_parallel_lanes([thinking, instruct])
    pending = [
        ExperimentCell(runtime=runtime, benchmark=benchmark, model_id="thinking"),
        ExperimentCell(runtime=runtime, benchmark=benchmark, model_id="instruct"),
        ExperimentCell(runtime=runtime, benchmark=benchmark, model_id="thinking"),
    ]

    queues = orch._assign_cells_to_lanes(pending, lanes)

    assert [cell.model_id for cell in queues["thinking"]] == ["thinking", "thinking"]
    assert [cell.model_id for cell in queues["instruct"]] == ["instruct"]


def test_generate_report_splits_outputs_per_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    chart_calls: list[tuple[str, tuple[str, ...]]] = []

    def _fake_generate_all_charts(results_df, output_dir):  # type: ignore[no-untyped-def]
        chart_calls.append(
            (
                str(output_dir),
                tuple(sorted(results_df["model_id"].dropna().unique().tolist())),
            )
        )

    from tq_bench.reporters import charts

    monkeypatch.setattr(charts, "generate_all_charts", _fake_generate_all_charts)

    orch = Orchestrator(_make_config(tmp_path))
    records = [
        RunRecord(
            runtime_id="baseline",
            benchmark_id="ai2d",
            status="ok",
            model_id="thinking",
            score=1.0,
        ),
        RunRecord(
            runtime_id="baseline",
            benchmark_id="ai2d",
            status="ok",
            model_id="instruct",
            score=0.5,
        ),
    ]

    orch._generate_report(records)

    reports_dir = tmp_path / "results" / "reports"
    assert (reports_dir / "results_summary.json").exists()
    assert (reports_dir / "results_summary.csv").exists()
    assert (reports_dir / "results_summary.md").exists()
    assert (reports_dir / "thinking" / "results_summary.json").exists()
    assert (reports_dir / "instruct" / "results_summary.json").exists()
    assert len(chart_calls) == 2
    assert all(len(model_ids) == 1 for _, model_ids in chart_calls)
