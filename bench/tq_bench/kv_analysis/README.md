# tq_bench.kv_analysis

Python analysis pipeline for KV cache dumps produced by the
`llama-kv-dump` C++ tool that lives in the `llama.cpp/` fork.  Used to
validate the TurboQuant theoretical assumptions (Beta marginal,
coordinate independence, per-coord MSE) against real Qwen3-VL KV cache
data.

## Input format

The C++ tool is expected to write one directory per run:

```
results/kv_dumps/<run_name>/
├── meta.json
├── K_layer_0.bin          # float32, shape (n_tokens, n_kv_head, head_dim)
├── V_layer_0.bin
├── K_layer_1.bin
├── ...
└── V_layer_{L-1}.bin
```

`meta.json` schema:

```json
{
  "n_tokens": 87,
  "n_layers": 28,
  "n_kv_head": 8,
  "head_dim": 128,
  "vision_token_mask": [false, ..., true, ..., false],
  "model_path": "...",
  "cache_type_k": "f16",
  "cache_type_v": "f16",
  "prompt_text": "Describe this image.",
  "image_path": "/path/to/img.jpg",
  "timestamp": "2026-04-12T15:23:45"
}
```

For Qwen3-VL-2B the expected shape per layer is
`(n_tokens, 8, 128)` with 28 layers.

## Modules

| Module | Purpose |
| --- | --- |
| `loader.py` | `KVDump` / `KVDumpWriter` — read and write dump directories |
| `distribution.py` | per-layer K/V value + norm statistics |
| `outliers.py` | outlier channel detection (> 10x median norm) |
| `quant_error.py` | per-layer diff metrics + theoretical MSE check |
| `rotation_analysis.py` | FWHT + sign-flip, Beta fit, coordinate independence |
| `attention_analysis.py` | softmax attention, KL / JS / top-k overlap |
| `report.py` | end-to-end report generator (CSVs + markdown + plots) |

## Usage

```python
from pathlib import Path
from tq_bench.kv_analysis import (
    KVDump,
    compute_per_layer_stats,
    outlier_ratio_vision_vs_text,
    compare_dumps,
    vision_vs_text_rotation_analysis,
    generate_full_report,
)

baseline = KVDump(Path("results/kv_dumps/baseline"))
tq3 = KVDump(Path("results/kv_dumps/tq-3"))

# Per-layer stats split by vision vs text.
dist_df = compute_per_layer_stats(baseline, separate_vision_text=True)

# Outlier channels per layer.
outlier_df = outlier_ratio_vision_vs_text(baseline, threshold=10.0)

# Quantization error vs baseline.
quant_df = compare_dumps(baseline, tq3)

# TurboQuant rotation diagnostics (Beta fit, independence).
rot_df = vision_vs_text_rotation_analysis(baseline)

# Full report (CSVs + markdown + plots).
generate_full_report(
    baseline_dump=baseline,
    quant_dumps={"tq-3": tq3},
    output_dir="results/kv_analysis/run_001",
    bits_by_run={"tq-3": 3},
)
```

A ready-to-run notebook wrapper is in
[`bench/notebooks/05_kv_dump_analysis.ipynb`](../../notebooks/05_kv_dump_analysis.ipynb).

## Rotation parity with the C++ reference

`rotation_analysis.apply_fwht` matches `turbo_forward_rotate` in
`ggml/src/ggml-quants.c` exactly: the same deterministic sign-flip
(`seed=42`, golden-ratio hash, top-bit test) followed by the same
unscaled in-place Walsh-Hadamard transform.  Numerical parity is
verified bitwise for `n in {8, 16, 32, 64, 128, 256}`.

The theoretical Lloyd-Max MSE values used by `compare_with_theoretical`
come from the TurboQuant paper (arXiv 2504.19874):

| bits | theoretical MSE / coord |
| --- | --- |
| 2 | 0.117 |
| 3 | 0.034 |
| 4 | 0.009 |

## Tests

```bash
cd bench
uv run pytest tq_bench/kv_analysis/tests/
```

All tests run with synthetic dumps — no dependency on the C++ tool —
so the pipeline can be developed and verified in parallel with the
C++ side.
