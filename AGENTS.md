# AGENTS.md

## Purpose

This repository benchmarks TurboQuant KV cache quantization on vision-language models with two coordinated codebases:

1. `llama.cpp/` for the GGML / CUDA implementation
2. `bench/` for experiment orchestration, evaluation, and reporting

`CLAUDE.md` is the detailed project state document. This file is the short operational guide.

## Retained Docs

Keep these project-authored docs aligned when behavior changes:

- `README.md` — operator entry point and quick-start
- `AGENTS.md` — this file
- `CLAUDE.md` — detailed architecture and status
- `docs/OFFICIAL_PARITY_AUDIT.md` — benchmark interpretation caveats

The old standalone phase-plan doc was intentionally removed during cleanup.

## Repository Layout

```text
.
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── docs/
│   └── OFFICIAL_PARITY_AUDIT.md
├── llama.cpp/
├── bench/
├── models/
├── results/
├── logs/
└── pdfs/
```

### `llama.cpp/`

Primary patch areas:

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-common.h`
- `ggml/src/ggml-common-turbo.h`
- `ggml/src/ggml-quants.h`
- `ggml/src/ggml-quants.c`
- `ggml/src/ggml-cpu/quants.c`
- `ggml/src/ggml-cuda/`
- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `common/arg.cpp`

### `bench/`

Active framework pieces:

- `configs/` — runtimes, benchmarks, models, profiles
- `tq_bench/` — runner, orchestrator, client, server, datasets, evaluators, KV analysis
- `tests/` — evaluator / orchestration regression tests
- `run_bench.py` — main CLI
- `run_parity_smoke.py` — focused P0 parity smoke

## Current Codebase Status

As of 2026-04-13:

- TurboQuant MSE and prod GGML types are implemented in `llama.cpp`, including CUDA flash-attention integration and KV dump tooling.
- The Python benchmark framework is feature-complete for planned experiments: dual-GPU execution lanes, model-specific sampling controls, checkpoint/resume, per-sample timings, aggregate latency stats, and KV analysis.
- The test suite currently contains 224 collected tests (85 evaluator/parity, 139 KV-analysis).
- Local exploratory outputs already include parity smoke, partial P0/core/TQ runs, and a Thinking-model smoke run.
- Canonical frozen results and the final report are still behind the latest code/config state.

## Immediate Priorities

1. Freeze a canonical P0 baseline/core result set with the current code and configs.
2. Consolidate corrected prod/QJL results and replace stale historical interpretations.
3. Run controlled Thinking experiments that separate short-answer vs long-reasoning behavior.
4. Expand to the full runtime × benchmark matrix only after the above are stable.

## Working Rules

- Preserve upstream `llama.cpp` style in C/C++ changes.
- Use 4-space indentation in new code.
- Keep GGML type naming consistent with existing quant type conventions.
- Do not commit model files, results, logs, caches, or notebook checkpoints.
- Keep notebook logic thin; put reusable logic in `bench/tq_bench/`.
- Keep prod runtime failures explicit in result records rather than silently skipping them.
- Treat `parallel_requests` as an experimental control, not a tuning knob to “fix” failing runs.
- Treat model sampling settings (`temperature`, `top_p`, `top_k`, `min_p`, penalties, seed) as fixed controls unless the experiment explicitly changes them.
- If benchmark semantics change, update both `README.md` and `docs/OFFICIAL_PARITY_AUDIT.md`.

## Bench Framework Expectations

Expected execution flow:

1. Load YAML configs.
2. Build the runtime × benchmark × model matrix.
3. For each runtime/model lane:
   - launch `llama-server`
   - run benchmark samples through the OpenAI-compatible API
   - record per-sample timings/tokens and aggregate stats
   - checkpoint results
   - stop the server
   - optionally run `llama-kv-dump` and start background CPU analysis
4. Wait for background KV analysis jobs.
5. Write JSON / CSV / markdown summaries.

## Validation Commands

```bash
python3 -m compileall bench/tq_bench
cd bench && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -q
cd bench && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tq_bench/kv_analysis/tests/ -q
```

## Notes For Agents

- Read `CLAUDE.md` before making architectural changes.
- Treat local `results/` artifacts as exploratory unless explicitly promoted in docs.
- If you touch `llama.cpp` quantization or flash-attention paths, verify both build integrity and CLI wiring.
- If you touch `SampleResult`, `RunRecord`, or `CompletionTimings`, check all constructors and failure paths.
- The default runner is subset-based; do not claim full official parity without checking the audit doc and the invocation path.
