# AGENTS.md

## Purpose

This repository benchmarks the effect of TurboQuant KV cache quantization on vision-language models using a patched `llama.cpp` fork.

The work splits into two tracks:

1. `llama.cpp/` implementation of new TurboQuant GGML KV cache types
2. `bench/` framework for running VLM and text benchmarks against `llama-server`

`CLAUDE.md` is the detailed design document.
This file is the short operational guide for coding agents working in this repository.

## Repository Layout

```text
.
├── AGENTS.md
├── CLAUDE.md
├── llama.cpp/         # upstream fork to patch
├── bench/             # Python benchmark framework
├── models/            # local GGUF/mmproj files, ignored by git
├── results/           # experiment outputs, ignored by git
└── pdfs/              # reference papers
```

### `llama.cpp/`

Primary patch targets:

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-common.h`
- `ggml/src/ggml-quants.h`
- `ggml/src/ggml-quants.c`
- `ggml/src/ggml-cpu/quants.c`
- `ggml/src/ggml-cuda/`
- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `common/arg.cpp`

### `bench/`

The benchmark framework is fully implemented:

- `configs/` for runtime (15), benchmark (11), and model (2) YAML
- `tq_bench/` Python package: runner, orchestrator, server, client, datasets×11, evaluators×10
- `tq_bench/evaluators/` includes 4 **official parity evaluators** (`mmmu_official`, `mathvista_official`, `textvqa_official`, `chartqapro_official`) alongside 6 existing approximate evaluators
- `tq_bench/kv_analysis/` for KV dump statistics and visualization (134 tests)
- `notebooks/` for thin execution wrappers
- `reporters/` for CSV/JSON/markdown/chart export
- `tests/` for golden parity evaluator tests (62 tests)
- `run_bench.py` — unified CLI runner (`--num`, `--runtimes`, `--benchmarks`, `--model`, `--resume`)
- `docs/OFFICIAL_PARITY_AUDIT.md` — per-benchmark parity gap analysis

## Project Goal

Implement and benchmark:

- TurboQuant Algorithm 1 (MSE): `turbo2`, `turbo2h`, `turbo3`, `turbo3h`, `turbo4`
- TurboQuant Algorithm 2 (prod/QJL): `turbop3`, `turbop4`, `turbop5`
- Baselines: `f16`, `q8_0`, `q4_0`, `q2_K`

Main target model family:

- `Qwen3-VL-2B-Instruct`
- `Qwen3-VL-2B-Thinking`

Main research claim to validate:

- MSE-only TurboQuant can be benchmarked systematically on VLMs
- prod/QJL variants likely fail in real generation despite decent cosine similarity

## Current Status

As of 2026-04-12, phases 1-8.5 are complete:

- `llama.cpp/` contains TurboQuant MSE and prod GGML types, CUDA KV write/dequant
  paths, flash-attention integration, KV dump tooling, and CLI wiring.
- `bench/` contains a production benchmark framework with official parity evaluators
  for P0 benchmarks (AI2D, MMMU, MathVista), parallel sample requests, dual-GPU
  orchestration, checkpoint/resume, `<think>` strip for Thinking models,
  and a unified CLI runner (`run_bench.py`).
- 196 tests passing (62 parity golden + 134 KV analysis).
- Parity smoke test completed: baseline × 3 VLM × n=10.

## Immediate Priorities

1. Phase 9: run `run_bench.py --num 100` to reproduce official VLM baseline scores
   and compare TurboQuant variants
2. Phase 10: consolidate prod/QJL corrected CUDA results
3. Phase 11: run `Qwen3-VL-2B-Thinking` KV-length experiments
4. Phase 12+: expand to the full runtime × benchmark matrix

## Working Rules

- Preserve upstream `llama.cpp` coding style in C/C++ changes.
- Use 4-space indentation in new code.
- Keep new GGML type naming consistent with existing quant types.
- Do not commit model files, benchmark outputs, caches, or notebook checkpoints.
- Treat `models/` and `results/` as local-only storage.
- Keep notebook logic thin; put real logic in `bench/tq_bench/`.
- When adding benchmark functionality, prefer reusable package code over notebook-only code.
- When implementing evaluation logic, keep prod runtime failures representable as explicit failed records rather than silent skips.

## Bench Framework Expectations

The intended flow is:

1. Load YAML configs
2. Build runtime × benchmark matrix
3. Launch `llama-server` with the requested KV cache types
4. Query the OpenAI-compatible API
5. Evaluate outputs
6. Persist checkpoints and result artifacts
7. Generate CSV, JSON, and markdown summaries

Dual-GPU execution is a requirement for the orchestrator design.

## Build And Run

### `llama.cpp`

```bash
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### `bench`

```bash
cd bench
uv sync
uv run jupyter lab
```

### Quick Python sanity check

```bash
python3 -m compileall bench/tq_bench
cd bench && uv run pytest tests/ -q          # 62 parity evaluator tests
cd bench && uv run pytest tq_bench/kv_analysis/tests/ -q  # 134 KV analysis tests
```

### Running benchmarks

```bash
cd bench
# Quick smoke (10 samples, 5 core runtimes, P0 benchmarks)
uv run python run_bench.py --num 10

# Full P0 run (100 samples)
uv run python run_bench.py --num 100

# All runtimes × all benchmarks
uv run python run_bench.py --num 100 --runtimes all --benchmarks all

# Thinking model
uv run python run_bench.py --num 30 --model qwen3_vl_2b_thinking
```

## Notes For Agents

- Read `CLAUDE.md` before making major architectural decisions.
- Prefer small, verifiable increments because both `llama.cpp` kernel work and benchmark orchestration are easy to break.
- If you touch `llama.cpp` quantization paths, verify both build integrity and the affected runtime CLI wiring.
- If you touch `bench/`, verify config loading and package importability before moving on.
- Prefer validating `bench/` changes with `python3 -m compileall bench/tq_bench` and targeted `pytest` where practical.
- If the user asks only for scaffolding or layout work, do not jump ahead into algorithm implementation without instruction.
