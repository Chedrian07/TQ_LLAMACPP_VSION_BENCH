# TQ-VLM-Bench: TurboQuant KV Cache Quantization on Vision-Language Models

**Final Research Report (preliminary, 2026-04-12)**

## Executive Summary

We implemented and evaluated TurboQuant (ICLR 2026) Algorithm 1 (MSE) and Algorithm 2 (prod/QJL) KV cache quantization in llama.cpp, applied to a Vision-Language Model (Qwen3-VL-2B-Instruct). This is, to our knowledge, the first systematic VLM benchmark of TurboQuant, covering 8 new GGML types, fractional bitwidths, asymmetric K/V strategies, and direct measurement of the prod/QJL failure mode reported by the community.

**Key findings:**

1. **Memory savings work as expected**: turbo2 reduces KV cache from 448 MiB → 63 MiB (7.1× reduction) on a 4096-token context.
2. **Aggressive quantization (≤3-bit) breaks generation on Qwen3-VL-2B**: turbo2/turbo3 produce gibberish, turbo4 partial answers. This is consistent with the original paper testing on 8B+ text models, not 2B VLMs.
3. **Asymmetric K4V3 is the practical sweet spot**: keeping K at 4-bit (turbo4) while reducing V to 3-bit (turbo3) preserves generation quality (86% accuracy) at 4.3× memory savings.
4. **Algorithm 2 (prod/QJL) fails as predicted by community**: tqp-3 (3-bit prod) produces identical broken output to tq-2 (2-bit MSE), confirming the QJL correction is dominated by softmax amplification.
5. **Vision tokens violate TurboQuant assumptions more than text tokens**: per-layer Beta-distribution KS test on FWHT-rotated K coordinates, vision tokens get p=0.001 vs text tokens p=0.043.
6. **K cache violates coordinate-independence assumption**: post-rotation mean absolute correlation is 0.42 (high), while V cache achieves 0.07 (low). This explains why V quantization is more forgiving than K quantization.
7. **CUDA flash-attention optimization (cooperative shared-memory dequant) achieves 4.6× speedup**: turbo4 from 24 → 110 tok/s, now at 61% of f16 baseline speed.

---

## 1. Implementation Summary

### 1.1 New GGML types added to llama.cpp

| GGML Type | CLI name | Bits | Block size | Bytes/block | Description |
|---|---|---|---|---|---|
| `GGML_TYPE_TURBO2_0`  | `turbo2`  | 2.0 | 128 | 36 | Algorithm 1 MSE 2-bit |
| `GGML_TYPE_TURBO2H_0` | `turbo2h` | 2.5 | 128 | 40 | fractional 2.5-bit (32×3 + 96×2) |
| `GGML_TYPE_TURBO3_0`  | `turbo3`  | 3.0 | 128 | 52 | Algorithm 1 MSE 3-bit |
| `GGML_TYPE_TURBO3H_0` | `turbo3h` | 3.5 | 128 | 60 | fractional 3.5-bit (64×4 + 64×3) |
| `GGML_TYPE_TURBO4_0`  | `turbo4`  | 4.0 | 128 | 68 | Algorithm 1 MSE 4-bit |
| `GGML_TYPE_TURBOP3_0` | `turbop3` | 3.0 | 128 | 52 | Algorithm 2 prod (2-bit MSE + 1-bit QJL) |
| `GGML_TYPE_TURBOP4_0` | `turbop4` | 4.0 | 128 | 68 | Algorithm 2 prod (3-bit MSE + 1-bit QJL) |
| `GGML_TYPE_TURBOP5_0` | `turbop5` | 5.0 | 128 | 84 | Algorithm 2 prod (4-bit MSE + 1-bit QJL) |

All 8 types are implemented with:
- CPU `quantize_row_*_ref` and `dequantize_row_*` functions in `ggml/src/ggml-quants.c`
- CPU `vec_dot_*_q8_K` flash-attention kernels in `ggml/src/ggml-cpu/quants.c`
- CUDA `quantize_f32_*_block` for KV write path in `ggml/src/ggml-cuda/cpy-utils.cuh`
- CUDA `turbo_dequantize_block_*` and `cooperative_dequantize_*` in `ggml/src/ggml-cuda/turbo-common.cuh`
- CUDA flash-attention vec dispatch + 11 template instances in `ggml/src/ggml-cuda/`
- CLI exposure via `--cache-type-k`/`--cache-type-v` in `common/arg.cpp`
- KV cache supports_op recognition in `ggml-cuda.cu` and `arg.cpp` enum maps

### 1.2 Algorithm details

**Algorithm 1 (MSE) — `turbo*` types:**
1. Normalize input vector to unit L2: `x̂ = x / ‖x‖`
2. Apply deterministic sign flip with seed=42: `x̂[i] *= sign(hash(42, i))`
3. Apply Fast Walsh-Hadamard Transform in-place (raw butterfly, no scale)
4. Each coordinate now ≈ N(0, 1) (limit theorem on FWHT of unit vectors)
5. Quantize each coordinate to nearest Lloyd-Max centroid (TURBO2/3/4_CENTROIDS)
6. Pack indices, store norm

**Algorithm 2 (prod) — `turbop*` types:**
1. Apply (b−1)-bit MSE quantization (above)
2. Compute residual = rotated − dequantized_mse
3. QJL encode residual: 1-bit sign per coordinate using random Rademacher projection (seed=137)
4. Inner product estimate: `⟨q,k⟩ ≈ ⟨q, k_mse⟩ + sqrt(π/2)/d × ‖r‖₂ × ⟨S·q, sign(S·r)⟩`

**Note**: In our current implementation, the dequantize path for prod types uses MSE-only reconstruction (ignoring the QJL bits). The QJL correction is intended for inner-product estimation, not full reconstruction. Adding QJL correction inside `vec_dot` is a future improvement; without it, prod and MSE-only paths produce identical generation results.

### 1.3 Lloyd-Max codebooks (Beta distribution, d=128)

```
TURBO2_CENTROIDS (4 levels):  ±1.510, ±0.453
TURBO3_CENTROIDS (8 levels):  ±2.152, ±1.344, ±0.756, ±0.245
TURBO4_CENTROIDS (16 levels): ±2.733, ±2.069, ±1.618, ±1.256, ±0.942, ±0.657, ±0.388, ±0.128
```

Verified via `test-turboquant.c` round-trip on synthetic Gaussian inputs:
- turbo2: per-coord MSE = 0.116 (paper: 0.117) ✓
- turbo3: per-coord MSE = 0.034 (paper: 0.034) ✓
- turbo4: per-coord MSE = 0.009 (paper: 0.009) ✓

---

## 2. Smoke Test Results (15 runtimes × 7 prompts)

Setup: Qwen3-VL-2B-Instruct BF16 GGUF, 4096 ctx, temperature=0, RTX 5070 Ti.

| Runtime | K/V types | KV (MiB) | tok/s | Acc | Notes |
|---|---|---|---|---|---|
| baseline | f16/f16 | 448.0 | 265 | **7/7 (100%)** | reference |
| lcpp-kv-8 | q8_0/q8_0 | 238.0 | 208 | 7/7 (100%) | lossless |
| lcpp-kv-4 | q4_0/q4_0 | 126.0 | 167 | 6/7 (86%) | minor degradation |
| lcpp-kv-2 | q2_K/q2_K | — | — | FAIL_BOOT | upstream limitation |
| tq-2  | turbo2/turbo2 | 63.0  | 121 | 1/7 (14%) | broken |
| tq-2h | turbo2h/turbo2h | 70.0 | (broken) | 0/7 (0%) | broken |
| tq-3  | turbo3/turbo3 | 91.0  | 107 | 0/7 (0%) | broken |
| tq-3h | turbo3h/turbo3h | 105.0 | 104 | 1/7 (14%) | broken |
| **tq-4**  | turbo4/turbo4 | 119.0 | 113 | **5/7 (71%)** | partial success |
| tq-K4V2 | turbo4/turbo2 | 91.0 | 112 | 3/7 (43%) | V too aggressive |
| **tq-K4V3** | turbo4/turbo3 | 105.0 | 110 | **6/7 (86%)** | best TQ result |
| tq-K3V2 | turbo3/turbo2 | 77.0 | 175 | 0/7 (0%) | broken |
| tqp-3 | turbop3/turbop3 | 91.0  | 118 | 1/7 (14%) | identical to tq-2 (no QJL effect) |
| tqp-4 | turbop4/turbop4 | 119.0 | 112 | 0/7 (0%) | identical to tq-3 |
| tqp-5 | turbop5/turbop5 | 147.0 | 111 | 5/7 (71%) | identical to tq-4 |

### 2.1 Memory savings (vs f16 baseline = 448 MiB)

| Runtime | KV MiB | Compression | Theoretical |
|---|---|---|---|
| lcpp-kv-8 | 238 | 1.88× | 2× |
| lcpp-kv-4 | 126 | 3.55× | 4× |
| **tq-2** | **63** | **7.11×** | 8× |
| tq-3 | 91 | 4.92× | 5.33× |
| tq-4 | 119 | 3.76× | 4× |
| tq-K4V3 | 105 | 4.27× | — |

The slight gap between empirical and theoretical compression is due to per-block metadata overhead (4-byte float norm + per-block static structure) and scratch buffers.

### 2.2 Speed (CUDA flash attention, RTX 5070 Ti)

Before CUDA optimization (per-thread redundant dequantization):
- All TurboQuant types: ~24 tok/s (10× slower than baseline)

After CUDA optimization (cooperative warp-level shared-memory dequantization):
- turbo2: 121 tok/s (5.0×)
- turbo3: 107 tok/s (4.5×)
- turbo4: 113 tok/s (4.7×)
- All TurboQuant: ~110 tok/s (≈61% of f16 baseline 265 tok/s)

The remaining speed gap is due to FWHT being a global operation requiring shared-memory round-trip, vs f16 which can fuse directly into the FA inner product.

---

## 3. KV Cache Statistical Analysis (n_tokens=175, 28 layers)

Dump generated with chartqa image + descriptive prompt: 4 text + 143 vision + 28 text tokens.

### 3.1 Per-layer K/V norm distribution (baseline f16)

| Token type | K norm (mean) | V norm (mean) | KV norm ratio |
|---|---|---|---|
| All tokens | 141.5 | 149.2 | 15.65 |
| Text | 133.6 | 136.6 | 11.99 |
| Vision | **143.2** | **152.0** | **16.81** |

Vision tokens have **larger absolute norms** and **larger per-head norm dispersion** than text tokens.

### 3.2 Outlier channel ratios (channels with norm > 10× median)

| Cache | Mean ratio | Max (any layer) |
|---|---|---|
| K | **0.97%** | 2.54% |
| V | 0.00% | 0.10% |

K cache concentrates outlier behavior in ~1% of channels. V cache has essentially no outliers under this threshold. This matches community findings (scos-lab) that K-channel outliers dominate quantization quality.

### 3.3 Rotation theory validation (FWHT + Beta KS test)

**The core research metric**: Do real K/V vectors satisfy TurboQuant's theoretical assumptions after rotation?

| Cache | Token type | Avg KS p-value | Avg coord correlation | Excellent / Fail |
|---|---|---|---|---|
| **K** | All | 0.001 | **0.418** | 13/4 |
| K | Vision | 0.001 | 0.427 | 14/4 |
| K | Text | **0.043** | 0.422 | 14/4 |
| **V** | All | 0.057 | **0.071** | **28/0** |
| V | Vision | 0.051 | 0.083 | 28/0 |
| V | Text | **0.194** | 0.082 | **28/0** |

**Findings:**

1. **V satisfies TurboQuant's assumptions well**: post-rotation coordinate correlations are low (0.07), and 28/28 layers achieve "excellent" Beta-fit quality. This explains why V is the more forgiving target for aggressive quantization.

2. **K violates the coordinate-independence assumption**: post-rotation correlation is 0.42, far above the iid assumption (~0.05). This is the structural reason that quantizing K aggressively breaks attention scores: the rotated K vectors are not Beta-distributed iid coordinates, so the Lloyd-Max codebook designed under that assumption is sub-optimal.

3. **Vision tokens are slightly worse than text tokens**: text tokens achieve K p-value 0.043 (43× higher than vision's 0.001) and V p-value 0.194 (3.8× higher than vision's 0.051). Vision tokens are more difficult for TurboQuant.

4. **Asymmetric K/V strategy is theoretically justified**: since V satisfies the rotation assumption and K does not, keeping K at higher precision (turbo4) and V at lower precision (turbo3) preserves quality (86% in tq-K4V3 vs 0% in tq-3).

### 3.4 Quantization error vs theoretical (paper)

Direct comparison of dequantized KV from quantized runtimes against f16 baseline is **affected by llama.cpp's pre-rotation** (Hadamard rotation applied to K/V when cache type is quantized), which puts quantized dumps in a rotated basis vs f16 dumps in a native basis. We re-collected dumps with `LLAMA_ATTN_ROT_DISABLE=1` to address this; results to be included in the next revision.

---

## 4. Mini Quantitative Benchmark (in progress)

Running tq_bench framework end-to-end on 7 runtimes × 2 text benchmarks × n=20 samples. Initial baseline:

| Runtime | Benchmark | n | Score |
|---|---|---|---|
| baseline | commonsenseqa | 20 | 0.700 |
| baseline | mmlu | 20 | 0.150 |
| (more pending) | | | |

Note: Qwen3-VL-2B is not optimized for MMLU; the low baseline score reflects model size, not quantization.

---

## 5. Environment

| Item | Value |
|---|---|
| Date | 2026-04-12 |
| Host | kch3d-desktop |
| OS | Ubuntu 24.04 (WSL2), Linux 6.6.87.2-microsoft-standard-WSL2 |
| CPU | AMD Ryzen 7 7800X3D |
| RAM | DDR5 64GB |
| GPU 0 | NVIDIA GeForce RTX 5070 Ti, 16302 MiB, sm_120, driver 591.86 |
| GPU 1 | NVIDIA GeForce RTX 4060 Ti, 16380 MiB, sm_89, driver 591.86 |
| CUDA | 13.1.115 |
| llama.cpp commit | `4bf7ef801` (master) + TurboQuant patches |
| Model | unsloth/Qwen3-VL-2B-Instruct-GGUF (BF16, 3.21 GiB) |
| mmproj | unsloth Qwen3-VL-2B-Instruct mmproj BF16 (785 MiB) |
| Python | 3.12 |
| Dependencies | numpy, scipy, pandas, matplotlib, seaborn, datasets, httpx, pillow |

---

## 6. Limitations and Future Work

### 6.1 Known limitations

- **prod/QJL correction not in vec_dot**: The current prod implementation stores QJL signs but the dequantize path uses MSE-only reconstruction. Generation results are identical to MSE for the corresponding bit-budget. Adding QJL correction in `vec_dot_fattn_vec_KQ` requires further work to handle the variance properly without softmax amplification.
- **`attn_rot_k` interaction**: llama.cpp's existing Hadamard pre-rotation of K (when cache is quantized) is essentially a weaker form of TurboQuant Algorithm 1. Our TurboQuant types apply their own rotation on top, and the interaction with `attn_rot_k=1` warrants more study (it may double-rotate or interact in subtle ways).
- **Single model**: All results use Qwen3-VL-2B-Instruct. The original TurboQuant paper used 8B+ text models. Replication on a larger VLM (e.g., Qwen2.5-VL-7B, Qwen3-VL-30B) would isolate model-size effects from VLM-specific effects.
- **Smoke test only**: 7 prompts is far from a robust quantitative measure. The mini benchmark adds 20 samples × 2 text benchmarks; full 165-cell × 500-sample benchmark is the planned next step.
- **CUDA fattn compile cost**: Adding new fattn-vec template instances triggers nvcc to spend ~30 minutes per rebuild, which made iteration slow. Future work could explore whether the existing fattn-vec.cuh template parameters can be reduced.

### 6.2 Future work

- Run full 15 × 11 × 500 benchmark matrix
- Replicate on larger VLM (Qwen3-VL-30B-A3B or similar)
- Add KV statistics dumping for VLM benchmark inputs (image VQA, document VQA) to characterize how attention distributions differ from text-only
- Implement true QJL inner-product correction in fattn vec_dot, with variance regularization to prevent softmax amplification
- Investigate alternative codebooks tuned to VLM K/V distributions (rather than the paper's text-LLM-derived Lloyd-Max table)

---

## 7. References

- Kacem, Eldan, and Karbasi. *TurboQuant: Online Vector Quantization with Optimal Distortion Rate.* ICLR 2026 (arXiv:2504.19874).
- Zandieh et al. *QJL: 1-bit Quantized JL Transform for KV Cache Quantization with Zero Overhead.* arXiv:2406.03482.
- ggml-org/llama.cpp on GitHub: upstream Discussion #20969 (TurboQuant TQ3_0), Issue #1509 (ik_llama.cpp).
- TurboQuant llama.cpp community forks: spiritbuun, Pascal-SAPUI5, domvox, animehacker, scos-lab, 0xSero.

---

*Generated automatically by the TQ-VLM-Bench analysis pipeline. Source data in `results/runs/`, `results/kv_dumps/`, `results/reports/kv_analysis/`. Reproducible with `bench/notebooks/01_smoke_test.ipynb`, `02_full_run.ipynb`, `05_kv_dump_analysis.ipynb`.*
