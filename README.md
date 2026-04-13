# TQ-VLM-Bench

Benchmarking TurboQuant KV cache quantization on vision-language models with a patched `llama.cpp` fork and a Python benchmark framework around `llama-server`.

## Current Snapshot

- `llama.cpp/` implements 8 TurboQuant GGML KV-cache types: `turbo2`, `turbo2h`, `turbo3`, `turbo3h`, `turbo4`, `turbop3`, `turbop4`, `turbop5`.
- `bench/` contains the active experiment framework: 15 runtime configs, 11 benchmark configs, 2 model configs, mixed-model dual-GPU orchestration, resume/checkpointing, per-sample timing/token instrumentation, and KV-dump analysis.
- Official-style evaluators are implemented for `mmmu`, `mathvista`, `textvqa`, and `chartqapro`; AI2D uses the existing MCQ scorer. The default runner still evaluates sampled subsets driven by `--num`.
- The current suite contains 224 collected tests: 85 evaluator/parity tests and 139 KV-analysis tests.
- Local run artifacts beyond the parity smoke test already exist, but the canonical frozen result set and final report are not yet synchronized with the latest code and configs.

## Retained Documentation

These are the project-authored source documents intentionally kept after cleanup:

- `README.md`: top-level operator entry point and quick-start.
- `AGENTS.md`: short operational guide for coding agents.
- `CLAUDE.md`: detailed project state, architecture, and experiment status.
- `docs/OFFICIAL_PARITY_AUDIT.md`: scoring/protocol caveats for benchmark interpretation.

Generated artifacts under `results/` and `logs/` are not the source of truth for project state.

## Repository Layout

```text
.
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── docs/
│   └── OFFICIAL_PARITY_AUDIT.md
├── llama.cpp/         # patched upstream fork (TurboQuant implementation)
├── bench/             # Python benchmark framework
├── models/            # local GGUF/mmproj assets, ignored by git
├── results/           # local run outputs, ignored by git
├── logs/              # KV-dump analysis outputs, ignored by git
└── pdfs/              # reference papers
```

## What Is Implemented

### `llama.cpp/`

- GGML type registration for all TurboQuant MSE/prod variants.
- CPU quantize/dequantize support and CPU vec-dot support.
- CUDA KV write, dequant, and flash-attention integration.
- KV-cache debug dump APIs and the `llama-kv-dump` tool.
- CLI wiring for `--cache-type-k` / `--cache-type-v`.

### `bench/`

- Runtime, benchmark, model, and execution-profile YAML configs.
- Dataset loaders for 8 VLM benchmarks and 3 text benchmarks.
- Evaluator registry with both approximate and official-style scorers.
- `run_bench.py` for matrix execution over `llama-server`.
- Mixed-model / dual-GPU orchestration support.
- KV-dump extraction plus Python-side KV analysis and reporting.

## Quick Start

### Build `llama.cpp`

```bash
cd llama.cpp
cmake -B build   -DGGML_CUDA=ON   -DCMAKE_CUDA_ARCHITECTURES="89;120"   -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

### Install the benchmark framework

```bash
cd bench
uv sync
```

### Sanity checks

```bash
python3 -m compileall bench/tq_bench
cd bench && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -q
cd bench && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tq_bench/kv_analysis/tests/ -q
```

### Run representative experiments

```bash
cd bench

# Parity-oriented P0 run on the core runtime group
uv run python run_bench.py --num 100 --runtimes core --benchmarks p0

# Add inline KV dump / analysis
uv run python run_bench.py --num 100 --runtimes core --benchmarks p0 --kv-dump

# Explore all MSE TurboQuant variants on P0
uv run python run_bench.py --num 100 --runtimes tq-all --benchmarks p0

# Thinking model smoke / comparison run
uv run python run_bench.py --num 20 --model qwen3_vl_2b_thinking   --runtimes baseline tq-4 tq-K4V3 --benchmarks p0
```

## Interpretation Notes

- `run_bench.py` overwrites benchmark sample counts with `--num`; the default runner does not automatically switch to full-split parity evaluation even though the config schema supports `parity_sample_count = -1`.
- Scores from text benchmarks (`mmlu`, `commonsenseqa`, `hellaswag`) are project-controlled generation scores, not official leaderboard scores.
- Use `docs/OFFICIAL_PARITY_AUDIT.md` before making any “official reproduction” claim.
- Treat `results/` and `logs/` as local experiment storage, not as maintained documentation.
