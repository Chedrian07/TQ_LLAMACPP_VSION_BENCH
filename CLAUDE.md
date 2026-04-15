# CLAUDE.md

## Project

TQ-VLM-Bench studies TurboQuant KV cache quantization on vision-language models with a patched `llama.cpp` fork and a Python benchmark framework around `llama-server`.

The project has two code tracks:

1. `llama.cpp/` — implement new TurboQuant GGML KV cache types and runtime integration
2. `bench/` — run VLM/text benchmarks, collect metrics, and analyze KV dumps

## Current Status (2026-04-15)

### Codebase status

- TurboQuant Algorithm 1 (MSE) and Algorithm 2 (prod/QJL) GGML types are implemented in `llama.cpp`.
- CUDA KV write/dequant/flash-attention paths and KV-dump tooling are present.
- The benchmark framework supports 15 runtime configs, 11 benchmark configs, 2 Qwen3-VL model configs, dual-GPU execution lanes, checkpoint/resume, per-sample timing/token instrumentation, and KV analysis.
- Official-style evaluators are implemented for `mmmu`, `mathvista`, `textvqa`, and `chartqapro`; AI2D uses the existing MCQ scorer.
- The active research scope is frozen to 5 benchmarks: `ai2d`, `mmmu`, `mathvista`, `textvqa`, and `docvqa`. The other benchmark configs remain available in the framework but are out of scope for current experiments and reporting unless explicitly reopened.
- The current test suite contains 224 collected tests: 85 evaluator/parity tests and 139 KV-analysis tests.

### Blackwell flash-attention status

CUDA 12.8 targeting sm_120 (Blackwell) has a compiler bug where generic-pointer shared-memory indexing in the cooperative FWHT butterfly generates byte-stride loads instead of float-stride loads. This produces `Invalid __shared__ read of size 4 bytes` misaligned-address crashes.

**Fix applied (turbo-common.cuh):** All shared-memory float accesses use `volatile float *` to force correct scalar load/store. Global-memory turbo block reads use `memcpy` + byte offsets (no struct-pointer casts). Confirmed working on RTX 5070 Ti (sm_120, CUDA 13.1); pending final verification on Colab B200 (sm_120, CUDA 12.8).

**Important:** Never disable flash attention for turbo types — V quantization requires FA to be ON.

### Known issue: QJL scale factor

`TURBO_QJL_SCALE = sqrt(pi/2) ≈ 1.2533` is defined in `ggml-common-turbo.h` but not applied in the QJL correction formula (`fattn-common.cuh`). The correction overestimates by ~25%. Needs paper cross-reference before the canonical P0 freeze.

### Experiment status

- Parity smoke (`baseline × {ai2d, mmmu, mathvista} × n=10`) has been run and kept as a small sanity check.
- Additional local artifacts exist for P0/core/TurboQuant sweeps and a Thinking-model smoke run.
- Current experiment planning is limited to `ai2d`, `mmmu`, `mathvista`, `textvqa`, and `docvqa`.
- Those artifacts are exploratory, not the canonical frozen result set.
- The final report and historical result interpretation are not yet synchronized with the latest code/config state.

### Practical meaning

The project is past the feature-construction phase. The main unfinished work is experimental consolidation:

1. freeze canonical P0 results with the current code path
2. consolidate corrected prod/QJL interpretation (including the sqrt(2/pi) scale question)
3. run controlled Thinking experiments on the active 5-benchmark scope
4. keep reporting and reruns scoped to `ai2d`, `mmmu`, `mathvista`, `textvqa`, and `docvqa`

## Retained Documentation

These are the maintained source documents after cleanup:

- `README.md` — high-level entry point
- `AGENTS.md` — short operational guide
- `CLAUDE.md` — this detailed project state doc
- `docs/OFFICIAL_PARITY_AUDIT.md` — benchmark scoring/protocol caveats

Do not treat generated files under `results/` or `logs/` as maintained project documentation.

## Repository Layout

```text
.
├── README.md
├── AGENTS.md
├── CLAUDE.md
├── docs/
│   └── OFFICIAL_PARITY_AUDIT.md
├── llama.cpp/         # patched upstream fork
├── bench/             # Python benchmark framework
├── models/            # local-only GGUF/mmproj assets
├── results/           # local-only run outputs
├── logs/              # local-only KV analysis outputs
└── pdfs/              # reference papers
```

## `llama.cpp/` Implementation Summary

### Registered TurboQuant GGML types

- Algorithm 1 (MSE): `turbo2`, `turbo2h`, `turbo3`, `turbo3h`, `turbo4`
- Algorithm 2 (prod/QJL): `turbop3`, `turbop4`, `turbop5`
- Baselines used for comparison: `f16`, `q8_0`, `q4_0`, `q2_K`

The current enum assignments are:

- `GGML_TYPE_TURBO2_0 = 42`
- `GGML_TYPE_TURBO2H_0 = 43`
- `GGML_TYPE_TURBO3_0 = 44`
- `GGML_TYPE_TURBO3H_0 = 45`
- `GGML_TYPE_TURBO4_0 = 46`
- `GGML_TYPE_TURBOP3_0 = 47`
- `GGML_TYPE_TURBOP4_0 = 48`
- `GGML_TYPE_TURBOP5_0 = 49`

### Main implementation areas

- `ggml/include/ggml.h` — type enum registration
- `ggml/src/ggml-common-turbo.h` — block structs, codebooks, prod-specific constants
- `ggml/src/ggml-quants.c` / `.h` — CPU quantize/dequantize reference paths
- `ggml/src/ggml-cpu/quants.c` — CPU vec-dot support
- `ggml/src/ggml-cuda/` — CUDA set-rows, convert, and flash-attention support
- `src/llama-kv-cache.*` and `include/llama.h` — KV debug dump APIs
- `tools/kv-dump/kv-dump.cpp` — offline KV extraction tool
- `common/arg.cpp` — CLI exposure of `turbo*` / `turbop*`

### What is implemented today

- MSE types cover symmetric and fractional bitwidth variants.
- Asymmetric K/V runtime combinations are supported through config, not new GGML enums.
- prod/QJL support includes CUDA flash-attention vec-dot handling in the current codebase.
- KV dump tooling can export layer-wise K/V tensors for offline Python analysis.

### Blackwell (sm_120) portability notes

The cooperative FWHT in `turbo-common.cuh` uses `volatile float *` for all shared-memory buffer accesses. This works around a CUDA 12.8 compiler bug where generic-pointer shared-memory indexing produces byte-stride instead of float-stride loads. Do NOT remove `volatile` from these functions without verifying on CUDA 12.8 + sm_120. The turbo dequantize functions use raw `const char *` + `memcpy` for global-memory block field access (no struct-pointer casts) for the same reason.

## `bench/` Framework Summary

### Config surface

- `bench/configs/runtimes.yaml` — 15 runtime definitions
- `bench/configs/benchmarks.yaml` — 11 benchmark definitions
- `bench/configs/models.yaml` — model paths, per-model execution lanes, sampling defaults
- `bench/configs/profiles.yaml` — environment-level overrides such as `local` and `colab`

### Runtime groups in `run_bench.py`

- `core` = `baseline`, `lcpp-kv-8`, `lcpp-kv-4`, `tq-4`, `tq-K4V3`
- `tq-all` = core + `tq-2`, `tq-2h`, `tq-3`, `tq-3h`, `tq-K4V2`, `tq-K3V2`
- `prod` = `tqp-3`, `tqp-4`, `tqp-5`
- `all` = `tq-all + prod`

### Benchmarks

Active experiment scope:

- `ai2d`
- `docvqa`
- `mathvista`
- `mmmu`
- `textvqa`

Configured but out of scope unless explicitly reopened:

VLM:

- `chartqa`
- `chartqapro`
- `ocrbench_v2`

Text:

- `mmlu`
- `commonsenseqa`
- `hellaswag`

### Model targets

Primary models:

- `qwen3_vl_2b_instruct`
- `qwen3_vl_2b_thinking`

Current model config also stores:

- GPU/port affinity for mixed-model dual-GPU runs
- `parallel_requests`
- model-specific sampling defaults
- optional quantized model paths (`q8_0`, `q4_k_m`)
- Thinking-model `max_tokens_override`

### Active framework features

- `BenchmarkRunner` launches `llama-server`, issues requests, captures timings/tokens, and evaluates samples.
- `LlamaApiClient` parses `usage` + `timings` into `CompletionTimings`.
- `RunRecord` stores score, failure counts, latency stats, throughput stats, and resource snapshots.
- `run_bench.py` supports matrix execution, resume, output merging, and inline KV dump scheduling.
- Colab helpers can build into `llama.cpp/build-colab/` and pass explicit binary overrides into `run_bench.py`.
- `Orchestrator` supports mixed-model / dual-GPU lane assignment.
- `tq_bench/kv_analysis/` provides distribution, outlier, quant-error, rotation, and attention analyses.

## Important Caveats

### 1. Official-style scorer integration is not exact official parity

The codebase now includes official-style scorers, but the default runner remains subset-based:

- `benchmarks.yaml` can express `parity_sample_count = -1`
- `BaseBenchmarkDataset._deterministic_sample()` supports `-1` as “full split”
- `run_bench.py` currently overwrites benchmark `sample_count` with `--num`

So the default `run_bench.py --num N ...` flow is still a sampled experiment path. Use `docs/OFFICIAL_PARITY_AUDIT.md` before making any leaderboard-comparison claim.

### 2. Local result artifacts may lag the code

Local files under `results/` and `logs/` were produced at different points in the project timeline. They are useful as exploratory records, but not all of them reflect the latest implementation state.

### 3. Historical prod/QJL interpretations need consolidation

The current codebase and the historical writeups are no longer perfectly aligned. Treat corrected prod/QJL behavior as an active consolidation task rather than a fully frozen conclusion.

## Phase View

### Completed implementation phases

The codebase is effectively complete for the original implementation work:

- TurboQuant GGML type integration
- CUDA flash-attention support
- KV-dump tooling
- benchmark framework build-out
- parity evaluator integration
- mixed-model execution support
- KV analysis pipeline

### Open consolidation phases

#### Phase 9 — canonical P0 freeze

Needed:

- rerun `baseline/core × p0` with the current code path
- freeze the canonical result set used for comparisons
- make sure the maintained docs refer to that result set, not older exploratory runs

#### Phase 10 — corrected prod/QJL consolidation

Needed:

- settle which prod/QJL outputs are considered current
- replace stale historical interpretations that predate the current code path
- regenerate any summary tables that must represent prod behavior

#### Phase 11 — controlled Thinking experiments

Needed:

- compare `baseline`, `tq-4`, and `tq-K4V3` on the Thinking model across `ai2d`, `mmmu`, `mathvista`, `textvqa`, and `docvqa`
- separate short-answer and long-reasoning conditions
- analyze whether longer reasoning degrades vision-token usefulness under KV quantization

#### Phase 12 — optional full matrix

This is no longer part of the active plan. Keep the current experiment scope limited to `ai2d`, `mmmu`, `mathvista`, `textvqa`, and `docvqa` unless the user explicitly reopens broader coverage.

## Build And Run

### Build `llama.cpp`

```bash
cd llama.cpp
cmake -B build   -DGGML_CUDA=ON   -DCMAKE_CUDA_ARCHITECTURES="89;120"   -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

### Install `bench/`

```bash
cd bench
uv sync
```

### Basic validation

```bash
python3 -m compileall bench/tq_bench
cd bench && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/ -q
cd bench && UV_CACHE_DIR=/tmp/uv-cache uv run pytest tq_bench/kv_analysis/tests/ -q
```

### Representative commands

```bash
cd bench

# P0/core sampled comparison
uv run python run_bench.py --num 100 --runtimes core --benchmarks p0

# P0/core with inline KV dump
uv run python run_bench.py --num 100 --runtimes core --benchmarks p0 --kv-dump

# Explore all MSE TurboQuant variants on P0
uv run python run_bench.py --num 100 --runtimes tq-all --benchmarks p0

# Thinking smoke
uv run python run_bench.py --num 20 --model qwen3_vl_2b_thinking   --runtimes baseline tq-4 tq-K4V3 --benchmarks p0
```

## Guidance For Future Edits

- Preserve upstream `llama.cpp` style when editing C/C++.
- Keep benchmark logic in `bench/tq_bench/`, not notebooks.
- Treat `parallel_requests` and model sampling defaults as controlled experiment settings.
- When changing scoring semantics, update both the README and the parity audit.
- When changing `SampleResult`, `RunRecord`, or `CompletionTimings`, audit success and failure paths.
- Do not commit models, results, logs, or notebook outputs.
