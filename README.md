# TQ-VLM-Bench: TurboQuant KV Cache Benchmark on Vision-Language Models

> Systematic evaluation of TurboQuant (ICLR 2026) KV cache quantization on Vision-Language Models, with a patched `llama.cpp` fork implementing both Algorithm 1 (MSE) and Algorithm 2 (prod/QJL).

## Overview

[TurboQuant](https://arxiv.org/abs/2504.19874) is a KV cache quantization method based on random orthogonal rotation (Fast Walsh-Hadamard Transform + deterministic sign flips) followed by Lloyd-Max scalar quantization on a Beta distribution. The paper proposes two variants:

- **Algorithm 1 (MSE)** — minimizes reconstruction MSE. Stable in practice.
- **Algorithm 2 (prod)** — `(b-1)`-bit MSE + 1-bit QJL projection on the residual. Unbiased inner-product estimator in theory; community reports consistent generation failure due to softmax amplifying QJL variance.

The original paper evaluates **text-only LLMs** (Llama-3.1-8B, Ministral-7B). **No systematic VLM evaluation exists.** This project closes that gap.

TQ-VLM-Bench delivers:

1. A patched `llama.cpp` fork with 8 new GGML KV cache types covering both algorithms, including fractional bitwidths and asymmetric K/V.
2. A Python benchmark framework driving `llama-server` over the OpenAI-compatible API across a 15-runtime × 11-benchmark matrix (165 cells).
3. Dual-GPU parallel orchestration with resume/checkpointing.
4. Reproducible results on Qwen3-VL-2B over 8 VLM benchmarks and 3 text benchmarks.

## Key Contributions

1. **First systematic VLM benchmark of TurboQuant** — 8 VLM tasks, 3 text tasks, 5 bitwidths, 2 algorithms.
2. **Direct implementation of Algorithm 2 (prod/QJL)** — no existing community fork implements it. We quantify its failure on real generation.
3. **Fractional bitwidths (2.5-bit, 3.5-bit)** — paper Section 4.3 technique, not available in any prior fork.
4. **Asymmetric K/V strategies** — `K4V2`, `K4V3`, `K3V2` at matched average bits.
5. **Same-bitwidth method comparison** — `q4_0`/`q2_K` (llama.cpp native) vs `turbo4`/`turbo2` (TurboQuant MSE) on identical cells.

## Architecture

```
.
├── llama.cpp/          # Forked llama.cpp with TurboQuant patches
│   ├── ggml/src/       # New GGML types, CPU + CUDA kernels
│   ├── src/            # KV cache integration
│   └── tools/          # llama-server, llama-mtmd-cli
│
├── bench/              # Python benchmark framework (UV managed)
│   ├── tq_bench/       # Core package: server, client, datasets, evaluators, orchestrator
│   ├── configs/        # runtimes.yaml, benchmarks.yaml, models.yaml
│   └── notebooks/      # Thin execution/analysis notebooks
│
├── models/             # GGUF model files (gitignored)
├── results/            # Run artifacts (gitignored)
└── pdfs/               # Reference papers
```

Design principles:

- **`llama-server` as inference engine.** Runtime conditions are injected via `--cache-type-k` / `--cache-type-v` CLI arguments at server launch. No in-process engine.
- **Notebooks are thin wrappers.** All logic lives in the `tq_bench` package.
- **Resume by default.** The orchestrator checkpoints each completed cell; reruns skip finished work.
- **Dual GPU parallelism.** Two `llama-server` instances on GPU 0 / GPU 1 run different runtimes concurrently.

See [CLAUDE.md](CLAUDE.md) for implementation details, block structures, and the full rationale.

## Benchmark Matrix

### Runtimes (15)

| # | Runtime ID | Method | cache-type-k | cache-type-v | Bits |
|---|---|---|---|---|---|
| 1 | `baseline` | FP16 | `f16` | `f16` | 16 |
| 2 | `lcpp-kv-8` | llama.cpp native | `q8_0` | `q8_0` | 8 |
| 3 | `lcpp-kv-4` | llama.cpp native | `q4_0` | `q4_0` | 4 |
| 4 | `lcpp-kv-2` | llama.cpp native | `q2_K` | `q2_K` | 2 |
| 5 | `tq-2` | TQ MSE (Alg 1) | `turbo2` | `turbo2` | 2 |
| 6 | `tq-2h` | TQ MSE fractional | `turbo2h` | `turbo2h` | 2.5 |
| 7 | `tq-3` | TQ MSE | `turbo3` | `turbo3` | 3 |
| 8 | `tq-3h` | TQ MSE fractional | `turbo3h` | `turbo3h` | 3.5 |
| 9 | `tq-4` | TQ MSE | `turbo4` | `turbo4` | 4 |
| 10 | `tq-K4V2` | TQ MSE asymmetric | `turbo4` | `turbo2` | avg 3 |
| 11 | `tq-K4V3` | TQ MSE asymmetric | `turbo4` | `turbo3` | avg 3.5 |
| 12 | `tq-K3V2` | TQ MSE asymmetric | `turbo3` | `turbo2` | avg 2.5 |
| 13 | `tqp-3` | TQ prod (Alg 2) | `turbop3` | `turbop3` | 3 |
| 14 | `tqp-4` | TQ prod (Alg 2) | `turbop4` | `turbop4` | 4 |
| 15 | `tqp-5` | TQ prod (Alg 2) | `turbop5` | `turbop5` | 5 |

### Benchmarks (11)

| # | ID | Type | N | Metric |
|---|---|---|---|---|
| 1 | `ai2d` | VLM | 500 | option_match |
| 2 | `chartqa` | VLM | 500 | relaxed_accuracy |
| 3 | `chartqapro` | VLM | 500 | relaxed_accuracy |
| 4 | `docvqa` | VLM | 500 | ANLS |
| 5 | `mathvista` | VLM | 500 | mathvista_match |
| 6 | `mmmu` | VLM | 500 | option_match |
| 7 | `ocrbench_v2` | VLM | 500 | exact_match |
| 8 | `textvqa` | VLM | 500 | normalized_exact_match |
| 9 | `mmlu` | Text | 1000 | option_match |
| 10 | `commonsenseqa` | Text | 3000 | option_match |
| 11 | `hellaswag` | Text | 3000 | option_match |

**Total: 15 × 11 = 165 cells.**

Primary comparison axes:

- Same bitwidth, different method: `lcpp-kv-4` vs `tq-4`, `lcpp-kv-2` vs `tq-2`.
- Algorithm 1 vs Algorithm 2: `tq-3` vs `tqp-3`, `tq-4` vs `tqp-4`.
- Same average bits, symmetric vs asymmetric: `tq-3` vs `tq-K4V2`.
- Bitwidth degradation curve: 2 → 2.5 → 3 → 3.5 → 4.
- VLM vs text gap at matched runtime.

## Setup

### Prerequisites

- NVIDIA GPU with CUDA compute capability sm_89 or sm_120 (tested on RTX 4060 Ti + RTX 5070 Ti)
- CUDA Toolkit 12.x
- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- 64GB RAM recommended (large text datasets and parallel runs)

### Build the patched llama.cpp

```bash
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

Binaries of interest: `build/bin/llama-server`, `build/bin/llama-mtmd-cli`.

### Install the Python framework

```bash
cd bench
uv sync
```

### Download models

```bash
# Qwen3-VL-2B-Instruct BF16 GGUF + mmproj
huggingface-cli download unsloth/Qwen3-VL-2B-Instruct-GGUF \
  Qwen3-VL-2B-Instruct-BF16.gguf mmproj-BF16.gguf \
  --local-dir models/Qwen3-VL-2B-Instruct

# Optional: Thinking variant
huggingface-cli download unsloth/Qwen3-VL-2B-Thinking-GGUF \
  Qwen3-VL-2B-Thinking-BF16.gguf mmproj-BF16.gguf \
  --local-dir models/Qwen3-VL-2B-Thinking
```

Paths are referenced by `bench/configs/models.yaml`.

### Preload datasets (optional)

```bash
cd bench
uv run python preload_datasets.py
uv run python verify_datasets.py
```

## Usage

### Google Colab

For a single-GPU Colab workflow, open [`bench/notebooks/00_colab_bench.ipynb`](bench/notebooks/00_colab_bench.ipynb) in Colab and run all cells.

- The notebook clones the repo into `/content`
- The notebook separately clones your custom `llama.cpp` fork and checks out the pinned commit before building
- `bench/` is installed with `pip install -e`
- `llama-server` is built once per custom `llama.cpp` remote + commit / CUDA arch and cached in Drive
- GGUF/mmproj artifacts are cached in Drive and copied into the repo-local `models/` layout before each run
- Results are written to Drive with resume support

### Smoke test

Quick correctness check with a few samples across a small subset of runtimes:

```bash
cd bench
uv run python smoke_test.py \
  --runtime baseline --runtime tq-3 --runtime tqp-3 \
  --benchmark commonsenseqa --benchmark docvqa \
  --n 3
```

`prod` runtimes are expected to FAIL generation; this is recorded rather than errored.

### Full run

```bash
cd bench
uv run jupyter lab notebooks/02_full_run.ipynb
```

Or programmatically:

```python
from tq_bench.orchestrator import Orchestrator, OrchestratorConfig

cfg = OrchestratorConfig.load("configs/runtimes.yaml",
                              "configs/benchmarks.yaml",
                              "configs/models.yaml")
orch = Orchestrator(cfg)
orch.run(parallel=True, resume=True)
```

### Analysis

```bash
uv run jupyter lab notebooks/03_analysis.ipynb  # metrics, heatmaps, curves
uv run jupyter lab notebooks/04_kv_analysis.ipynb  # KV distributions, VLM vs text
```

## GGML Type Reference

### Algorithm 1 (MSE)

| GGML Type | CLI name | Bits | Block | Bytes | Notes |
|---|---|---|---|---|---|
| `GGML_TYPE_TURBO2_0` | `turbo2` | 2 | 128 | 36 | 4B norm + 32B packed 2-bit |
| `GGML_TYPE_TURBO2H_0` | `turbo2h` | 2.5 | 128 | 40 | 32ch×3bit + 96ch×2bit |
| `GGML_TYPE_TURBO3_0` | `turbo3` | 3 | 128 | 52 | 4B norm + 48B packed 3-bit |
| `GGML_TYPE_TURBO3H_0` | `turbo3h` | 3.5 | 128 | 60 | 64ch×4bit + 64ch×3bit |
| `GGML_TYPE_TURBO4_0` | `turbo4` | 4 | 128 | 68 | 4B norm + 64B packed 4-bit |

### Algorithm 2 (prod = MSE + QJL)

| GGML Type | CLI name | Total Bits | Block | Bytes | Layout |
|---|---|---|---|---|---|
| `GGML_TYPE_TURBOP3_0` | `turbop3` | 3 | 128 | 52 | 2-bit MSE + 1-bit QJL sign + norm |
| `GGML_TYPE_TURBOP4_0` | `turbop4` | 4 | 128 | 68 | 3-bit MSE + 1-bit QJL sign + norm |
| `GGML_TYPE_TURBOP5_0` | `turbop5` | 5 | 128 | 84 | 4-bit MSE + 1-bit QJL sign + norm |

Types are enabled across CPU quantize/dequantize, CPU flash-attention `vec_dot`, and CUDA paths (`set-rows`, `convert`, `fattn-vec`). Block size is locked to Qwen3-VL's `head_dim = 128`.

## Algorithm Details

### Algorithm 1 — MSE

1. Apply random orthogonal rotation to each 128-dim head vector via Fast Walsh-Hadamard Transform + deterministic per-index sign flips (seeded).
2. Post-rotation, each coordinate follows a Beta distribution.
3. Scalar-quantize with a Lloyd-Max codebook optimized for that Beta (constants embedded in `ggml-common.h`).
4. Store `(norm, packed_indices)` per block. Dequantization reverses sign flip and FWHT.

### Algorithm 2 — prod (MSE + QJL)

1. Apply FWHT + sign flip.
2. Quantize with `(b-1)`-bit MSE codebook → `mse_indices`.
3. Compute residual `r = rotated - dequant(mse_indices)`.
4. Encode residual with 1-bit QJL: random ±1 projection + sign extraction → `qjl_signs` (16 B for 128 dims).
5. At attention time, the inner product is `⟨q, k_mse⟩ + ‖r‖ · (1/√d) · Σᵢ sign(rᵢ) · qᵢ · projᵢ`. Unbiased, but high variance — softmax amplifies this exponentially, breaking generation in practice.

Community reports across 6+ independent teams converge on the same failure mode. This project reproduces it quantitatively.

## Results

After a run, artifacts land under `results/`:

```
results/
├── runs/<timestamp>/
│   ├── checkpoint.json           # cell completion state
│   ├── cells/<runtime>__<bench>.json
│   └── server_logs/
├── reports/<timestamp>/
│   ├── summary.md                # markdown tables
│   ├── matrix.csv                # full 165-cell grid
│   ├── heatmap_vlm.png
│   └── degradation_curves.png
└── kv_dumps/                     # optional KV tensor dumps for analysis
```

`prod` runtime cells record generation pass/fail rate in addition to whatever partial metric is produced (typically 0 on breakdown).

## References

- TurboQuant — [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- QJL — [arXiv:2406.03482](https://arxiv.org/abs/2406.03482)
- PolarQuant — [arXiv:2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- KIVI — [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- HIGGS — [arXiv:2411.17525](https://arxiv.org/abs/2411.17525)
- Upstream [`llama.cpp`](https://github.com/ggerganov/llama.cpp)

Community prior art referenced: `spiritbuun/llama-cpp-turboquant-cuda`, `Pascal-SAPUI5/llama.cpp-turboquant`, `domvox/llama.cpp-turboquant-hip`, llama.cpp Discussion #20969 (TheTom), ik_llama.cpp Issue #1509. None implement Algorithm 2, fractional bitwidths, or asymmetric K/V.

## License

The `llama.cpp/` subtree inherits the upstream MIT license. The `bench/` framework is released under the same terms. See `llama.cpp/LICENSE`.
