# Phase 10-A: prod vec_dot QJL Correction Implementation Plan

## Overview
Implement TurboQuant Algorithm 2 inner product correction in the flash attention vec_dot path. Currently, `turbop3/4/5` types store QJL signs but the dequantize path ignores them, making prod identical to MSE-only. This phase adds proper QJL correction to measure whether Algorithm 2 actually fails due to softmax amplification as the community reports.

## Root Problem

Current `block_turbop3_0` / `_4_0` / `_5_0` structures only store:
- `float norm` (original vector norm before normalization)
- `uint8_t qs[...]` (MSE indices)
- `uint8_t qjl[16]` (QJL signs)

**Missing**: `float r_norm` (residual norm after MSE quantization on unit-sphere). Required for the correction formula:
```
<q, k> ≈ ||k|| * [<q_rot, k_mse_hat> + sqrt(π/2)/d * ||r_hat|| * <S·q_rot, sign(S·r_hat)>]
```

## Implementation Steps

### Step 1: Extend block_turbop structures (ggml-common.h)

Add `r_norm` after `norm`:

```c
typedef struct {
    float norm;                    // 4 bytes: vector norm
    float r_norm;                  // 4 bytes: residual norm after MSE (NEW)
    uint8_t qs[QK_TURBO / 4];     // 32 bytes: 2-bit MSE indices
    uint8_t qjl[QK_TURBO / 8];    // 16 bytes: 1-bit QJL signs
} block_turbop3_0;  // 56 bytes (was 52)
```

Same for turbop4_0 (was 68 → 72) and turbop5_0 (was 84 → 88).

Update `static_assert` sizes.

### Step 2: Update quantize functions (ggml-quants.c)

In `quantize_row_turbop{3,4,5}_0_ref`, after computing residual:
```c
// Compute residual norm BEFORE QJL encoding
float r_norm_sq = 0.0f;
for (int j = 0; j < TURBO_DIM; j++) {
    r_norm_sq += residual[j] * residual[j];
}
y[i].r_norm = sqrtf(r_norm_sq);
```

### Step 3: Add new CPU vec_dot functions for prod

`ggml/src/ggml-cpu/quants.c` — create separate vec_dot path that includes QJL correction:

```c
void ggml_vec_dot_turbop3_0_q8_K(int n, float * s, size_t bs,
        const void * vx, size_t bx,
        const void * vy, size_t by, int nrc) {
    // Same as turbo2_0_q8_K but with QJL correction added
    // ...
    // After MSE dot accumulation:
    for (int i = 0; i < nb; i++) {
        // MSE dequant + dot
        float mse_dot = /* existing MSE computation */;
        
        // QJL correction
        float qjl_corr = 0.0f;
        const float r_norm = x[i].r_norm;
        for (int j = 0; j < QK_TURBO; j++) {
            int qjl_bit = (x[i].qjl[j/8] >> (j%8)) & 1;
            uint32_t h = TURBO_QJL_SEED * 2654435761u + j * 2246822519u;
            float proj_sign = (h & 0x80000000u) ? -1.0f : 1.0f;
            // Rotated query (pre-FWHT applied outside or here)
            float q_val = /* y[i] equivalent rotated */;
            qjl_corr += q_val * proj_sign * (qjl_bit ? 1.0f : -1.0f);
        }
        qjl_corr *= r_norm * TURBO_QJL_SCALE / sqrtf(QK_TURBO);
        
        sumf += (mse_dot + qjl_corr) * x[i].norm * y[i].d;
    }
}
```

**Note**: CPU path will be used as vec_dot when FP8 K8 vec_dot is not available. The main path is CUDA.

### Step 4: CUDA turbo-common.cuh — Add cooperative prod dequant helper

Create `cooperative_dequantize_prod_turbo{3,4,5}` that extracts both MSE values AND QJL signs for use in vec_dot.

Alternative: store dequantized MSE in shared memory AND precompute per-coordinate QJL contribution, then vec_dot accumulates both.

### Step 5: CUDA fattn-common.cuh — New vec_dot for prod types

Create **separate** function `vec_dot_fattn_vec_KQ_turbop`:

```cuda
template <ggml_type type_K, int D, int nthreads>
__device__ static float vec_dot_fattn_vec_KQ_turbop(
    const char * __restrict__ K_c, const void * __restrict__ Q_v,
    const int * __restrict__ Q_q8, const float2 * __restrict__ Q_ds_v
) {
    // Get block
    const block_turbop3_0 * block = /* ... */;
    
    __shared__ float K_f_shared[8][128];  // MSE-reconstructed K
    __shared__ float K_qjl_shared[8][128]; // QJL sign contribution (pre-multiplied)
    
    const int warp = threadIdx.y;
    
    // Step 1: Cooperative MSE dequant (existing logic)
    cooperative_dequantize_mse_part(block, K_f_shared[warp]);
    
    // Step 2: Cooperative QJL contribution computation
    // For each coordinate, pre-multiply with deterministic projection sign
    const int tid = threadIdx.x;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int j = tid * 4 + k;
        int qjl_bit = (block->qjl[j/8] >> (j%8)) & 1;
        uint32_t h = 137u * 2654435761u + (uint32_t)j * 2246822519u;
        float proj_sign = (h & 0x80000000u) ? -1.0f : 1.0f;
        // Combined with apply_sign_flip (seed=42) and inverse FWHT
        K_qjl_shared[warp][j] = (qjl_bit ? 1.0f : -1.0f) * proj_sign;
    }
    __syncwarp();
    
    // Apply same sign flip + FWHT to QJL shared buffer (it's on unit sphere)
    // Scale by r_norm * TURBO_QJL_SCALE / D
    
    // Step 3: Dot product — MSE part + QJL correction
    float sum_mse = 0.0f;
    float sum_qjl = 0.0f;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        int j = tid * 4 + k;
        float q_val = /* dequantized query */;
        sum_mse += q_val * K_f_shared[warp][j];
        sum_qjl += q_val * K_qjl_shared[warp][j];
    }
    
    // Warp reduction
    // ...
    
    float final_dot = sum_mse + (block->r_norm * TURBO_QJL_SCALE / sqrtf((float)D)) * sum_qjl;
    return final_dot;
}
```

### Step 6: Update CUDA dispatcher in fattn-common.cuh

```cuda
// Around line 746
} else if constexpr (type_K == GGML_TYPE_TURBOP3_0) {
    return vec_dot_fattn_vec_KQ_turbop<GGML_TYPE_TURBOP3_0, D, nthreads>;  // NEW func
} else if constexpr (type_K == GGML_TYPE_TURBOP4_0) {
    return vec_dot_fattn_vec_KQ_turbop<GGML_TYPE_TURBOP4_0, D, nthreads>;
} else if constexpr (type_K == GGML_TYPE_TURBOP5_0) {
    return vec_dot_fattn_vec_KQ_turbop<GGML_TYPE_TURBOP5_0, D, nthreads>;
```

### Step 7: Rebuild & test

1. `cmake --build build -j4 --target test-turboquant`
   - Round-trip MSE should be UNCHANGED (prod dequant still returns MSE-only)
2. `cmake --build build -j4 --target ggml-cuda` (slow, ~30-40min for template instances)
3. `cmake --build build -j4 --target llama-server`
4. Smoke test: turbop3/4/5 with the same 7 prompts
   - **Expected outcome A (variance amplification)**: QJL correction added noise, generation even worse → community report confirmed
   - **Expected outcome B (partial recovery)**: QJL correction actually helps, some prompts fixed → new finding
   - **Expected outcome C (full recovery)**: prod works well, MSE-only was dominant → community report was wrong

### Step 8: Variance regularization (if outcome A)

If QJL amplifies noise as expected, add optional regularization:
- Clip correction contribution: `qjl_corr = sign(qjl_corr) * min(|qjl_corr|, α * |mse_dot|)` for some α
- Or: blend with confidence-weighted: `final = mse_dot + λ * qjl_corr` with λ < 1
- Measure the blend coefficient that minimizes generation degradation

## Files to Modify

| File | Change | Risk |
|---|---|---|
| `ggml/src/ggml-common.h` | Add `r_norm` to block_turbop*_0 | Struct layout change affects all CUDA/CPU code |
| `ggml/src/ggml-quants.c` | Store residual norm in quantize functions | Low |
| `ggml/src/ggml-cpu/quants.c` | Optional: separate vec_dot for prod | Medium (CPU fallback) |
| `ggml/src/ggml-cuda/turbo-common.cuh` | Add prod-specific dequant helper | Medium |
| `ggml/src/ggml-cuda/fattn-common.cuh` | New `vec_dot_fattn_vec_KQ_turbop` + dispatch | High (shared memory layout) |
| `ggml/src/ggml-cuda/cpy-utils.cuh` | Update CUDA quantize to store r_norm | Low (mirror CPU logic) |
| `ggml/src/ggml-cpu/ggml-cpu.c` | Verify type_traits block_size reflects new struct | Low |
| `llama.cpp/tests/test-turboquant.c` | Add prod round-trip test with QJL correction | Low |

## Block Size Impact

- turbop3_0: 52 → 56 bytes (+4B, +7.7% overhead)
- turbop4_0: 68 → 72 bytes (+4B, +5.9%)
- turbop5_0: 84 → 88 bytes (+4B, +4.8%)

Still smaller than corresponding lcpp types for the same bit budget.

## Testing Strategy

1. **Unit test**: Round-trip quantize → compute inner product with random query, compare:
   - MSE-only inner product estimate
   - MSE + QJL corrected estimate  
   - Ground truth inner product
   - Expected: corrected should have lower bias, higher variance (per paper)

2. **Smoke test**: Same 7 prompts, turbop3/4/5 with QJL correction on

3. **Mini benchmark**: 7 text × n=50, compare tqp-3/4/5 before and after

4. **Attention analysis**: Dump attention weights with TQ baseline vs prod corrected, compute KL divergence. If correction causes attention distribution to become more uniform (higher entropy), that's the softmax amplification effect in action.

## Expected Timeline

- Step 1-2 (struct + quantize): 20 min
- Step 3-5 (CUDA implementation): 1-2 hours
- Step 7 (rebuild): 30-40 min
- Step 7 (testing): 30 min
- Total: 2.5-4 hours

## Deliverable

1. Modified source files (8 listed above)
2. Rebuild success
3. Smoke test comparison (before/after)
4. Brief report of which outcome (A/B/C) occurred
5. If A: attention entropy analysis showing amplification
