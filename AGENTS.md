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
├── logs/              # KV analysis outputs (kv_analysis/<model>_<ts>/), ignored by git
└── pdfs/              # reference papers
```

### `llama.cpp/`

Primary patch targets:

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

The benchmark framework is fully implemented (~13,000 lines):

- `configs/` for runtime (14), benchmark (11), model (2+), and profile YAML
- `tq_bench/` Python package: runner, orchestrator, server, client, datasets×11, evaluators×12
- `tq_bench/evaluators/` includes 4 **official parity evaluators** (`mmmu_official`, `mathvista_official`, `textvqa_official`, `chartqapro_official`) alongside 6 existing approximate evaluators + 2 aliases
- `tq_bench/kv_analysis/` for KV dump statistics, attention comparison (K-as-Q-probe), and visualization (139 tests)
- `tq_bench/kv_dump_runner.py` wraps C++ `llama-kv-dump` + Python `kv_analysis` into a single pipeline
- `notebooks/` for thin execution wrappers
- `reporters/` for CSV/JSON/markdown/chart export (auto-detects timing data for extended tables)
- `tests/` for golden parity evaluator tests (85 tests)
- `run_bench.py` — unified CLI runner (`--num`, `--runtimes`, `--benchmarks`, `--model`, `--resume`)
- `docs/OFFICIAL_PARITY_AUDIT.md` — per-benchmark parity gap analysis

**Observability infrastructure (per-sample instrumentation):**
- `CompletionTimings` in `client.py`: parses llama-server `usage` + `timings` fields, measures wall-clock
- `SampleResult` records: ttft_ms, total_latency_ms, prefill_ms, decode_ms, decode_throughput_tps, prompt_tokens, completion_tokens, n_images
- `RunRecord` aggregates: score_std, score_median, LatencyStats (p50/p95/p99), ThroughputStats (mean/min/max), gpu_memory_bytes, kv_cache_bytes
- `LlamaServer`: nvidia-smi GPU memory query, /slots KV cache query (best-effort)

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

As of 2026-04-13, phases 1-8.5 are complete:

- `llama.cpp/` contains TurboQuant MSE and prod GGML types, CUDA KV write/dequant
  paths, flash-attention integration, KV dump tooling, and CLI wiring.
- `bench/` contains a production benchmark framework with official parity evaluators
  for P0 benchmarks (AI2D, MMMU, MathVista), parallel sample requests, dual-GPU
  orchestration, checkpoint/resume, `<think>` strip for Thinking models,
  per-sample timing/token instrumentation (TTFT, latency, tok/s, GPU memory),
  aggregate stats (score std/median, latency p50/p95/p99), and a unified CLI
  runner (`run_bench.py`).
- 224 tests passing (85 parity golden + 139 KV analysis).
- Parity smoke test completed: baseline × 3 VLM × n=10.
- Inline KV dump pipeline: `--kv-dump` flag interleaves dump extraction between
  runtime switches (GPU, ~10s per runtime), Python analysis runs in background (CPU).

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
- Treat model-level request parallelism as a required benchmark setting. The
  benchmark runner must keep explicit per-model `parallel_requests` behavior,
  and benchmark changes must not silently collapse model-side concurrency.
- Treat benchmark decoding/sampling settings as fixed experimental controls.
  Benchmark code and configs must keep the project's current generation
  settings (`temperature`, `top_p`, `top_k`, `min_p`, penalties, seed,
  `max_tokens` rules) stable unless the user explicitly asks to run a different
  experiment. Do not casually "improve" or "tune" these values during
  debugging.

## Bench Framework Expectations

The intended flow is:

1. Load YAML configs
2. Build runtime × benchmark matrix
3. For each runtime:
   a. Launch `llama-server` with the requested KV cache types
   b. For each benchmark: query OpenAI-compatible API, extract per-sample timing/token data, evaluate, compute aggregate stats
   c. Snapshot GPU memory, persist checkpoints
   d. Stop `llama-server`
   e. **If `--kv-dump`**: run `llama-kv-dump` on same GPU (~10s), kick off Python analysis in background thread (CPU)
4. Wait for all background KV analysis threads
5. Generate CSV, JSON, markdown summaries, and KV analysis reports

Dual-GPU execution is a requirement for the orchestrator design.

### Observable Data Collected

**Per sample** (every inference):
- Generated text (raw), ground truth, metric score, pass/fail, error message
- TTFT (ms), total latency (ms), prefill time (ms), decode time (ms)
- Decode throughput (tok/s)
- Prompt tokens, completion tokens, image count

**Per cell** (runtime × benchmark):
- Score: mean, std, median
- TTFT: mean, p50, p95, p99
- Total latency: mean, p50, p95, p99
- Decode throughput: mean, min, max
- GPU memory (bytes), KV cache tokens

**KV cache analysis** (inline via `--kv-dump` or offline via kv_analysis/):
- Per-layer K/V distribution: mean, std, min, max, quantiles (q01-q99)
- L2 norm distribution, K/V norm ratio, outlier channel ratio (10x median)
- Vision vs text token separation for all above
- Quant error: per-coordinate MSE, cosine similarity, inner product bias, vs theoretical
- Attention comparison (K-as-Q-probe): KL divergence, JS divergence, top-1 match rate, top-5 Jaccard overlap, entropy baseline/quantized/delta — per layer × vision/text
- Rotation: Beta KS test, coordinate independence, vision vs text

## Build And Run

### `llama.cpp`

```bash
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4    # -j4 recommended (nvcc OOM risk at -j8)
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
cd bench && uv run pytest tests/ -q          # 85 parity evaluator tests
cd bench && uv run pytest tq_bench/kv_analysis/tests/ -q  # 139 KV analysis tests
```

### Running benchmarks

```bash
cd bench
# Quick smoke (10 samples, 5 core runtimes, P0 benchmarks)
uv run python run_bench.py --num 10

# Full P0 run (100 samples) + KV dump/analysis
uv run python run_bench.py --num 100 --kv-dump

# All runtimes × all benchmarks
uv run python run_bench.py --num 100 --runtimes all --benchmarks all --kv-dump

# Thinking model
uv run python run_bench.py --num 30 --model qwen3_vl_2b_thinking

# Custom KV dump probe image
uv run python run_bench.py --num 50 --kv-dump --kv-dump-image path/to/chart.png
```

## Notes For Agents

- Read `CLAUDE.md` before making major architectural decisions.
- Prefer small, verifiable increments because both `llama.cpp` kernel work and benchmark orchestration are easy to break.
- If you touch `llama.cpp` quantization paths, verify both build integrity and the affected runtime CLI wiring.
- If you touch `bench/`, verify config loading and package importability before moving on.
- Prefer validating `bench/` changes with `python3 -m compileall bench/tq_bench` and targeted `pytest` where practical.
- If the user asks only for scaffolding or layout work, do not jump ahead into algorithm implementation without instruction.
- When modifying `SampleResult` or `RunRecord`, ensure all constructors (including error/crash paths) populate the new fields.
- The `CompletionTimings` dataclass is the single source of truth for server timing data — don't add parallel timing extraction elsewhere.
