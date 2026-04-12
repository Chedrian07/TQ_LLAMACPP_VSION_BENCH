"""KV cache analysis pipeline for TQ-VLM-Bench.

This package reads raw FP16/FP32 KV cache dumps produced by the C++
``llama-kv-dump`` tool and computes a suite of statistics that feed
into the TurboQuant VLM benchmark analysis:

* :mod:`tq_bench.kv_analysis.loader` — read binary dumps + metadata
* :mod:`tq_bench.kv_analysis.distribution` — per-layer K/V value
  and norm statistics
* :mod:`tq_bench.kv_analysis.outliers` — outlier channel detection
* :mod:`tq_bench.kv_analysis.quant_error` — quantization error metrics
  measured on real KV tensors
* :mod:`tq_bench.kv_analysis.rotation_analysis` — FWHT + sign-flip
  rotation diagnostics (Beta fit + coordinate independence)
* :mod:`tq_bench.kv_analysis.attention_analysis` — attention weight
  comparison helpers
* :mod:`tq_bench.kv_analysis.position_analysis` — position-aware
  per-token norms, outlier ratios, and quantization error
* :mod:`tq_bench.kv_analysis.codebook_analysis` — Lloyd-Max codebook
  bucket usage, vision/text skew, and bit-exact reproducibility
* :mod:`tq_bench.kv_analysis.visualization` — histogram plots
  (value, pre/post rotation, vision vs text)
* :mod:`tq_bench.kv_analysis.layer_plots` — layer-wise curve plots
  (norm ratio, distortion, cosine similarity, relative error)
* :mod:`tq_bench.kv_analysis.report` — end-to-end report generation

The analysis contract assumes the C++ tool writes:

``results/kv_dumps/<run_name>/``

::

    meta.json
    K_layer_0.bin, V_layer_0.bin
    ...
    K_layer_{L-1}.bin, V_layer_{L-1}.bin

where each ``.bin`` file is a little-endian float32 array of shape
``(n_tokens, n_kv_head, head_dim)``.  Tests in this package use
synthetic inputs so the Python side can be developed ahead of the
C++ tool.
"""

from .loader import KVDump, KVDumpWriter, load_dump
from .distribution import (
    compute_value_stats,
    compute_norm_stats,
    compute_per_layer_stats,
)
from .outliers import (
    find_outlier_channels,
    outlier_statistics,
    outlier_ratio_per_layer,
    outlier_ratio_vision_vs_text,
)
from .quant_error import (
    compare_dumps,
    compare_tensors,
    compare_with_theoretical,
    summarize_against_theoretical,
    THEORETICAL_MSE_PER_COORD,
)
from .rotation_analysis import (
    apply_fwht,
    analyze_rotation_per_layer,
    beta_distribution_fit_test,
    coordinate_independence_test,
    fwht_round_trip,
    sign_flip_mask,
    vision_vs_text_rotation_analysis,
    TURBO_DIM,
    TURBO_SEED,
)
from .attention_analysis import (
    attention_entropy,
    compute_attention_weights_from_kv,
    js_divergence_attention,
    kl_divergence_attention,
    top1_position_match_rate,
    topk_overlap_rate,
)
from .position_analysis import (
    compute_per_token_norms,
    per_position_outlier_ratio,
    per_position_quant_error,
    plot_token_norm_vs_position,
    plot_position_outlier_heatmap,
    plot_position_quant_error,
    token_index_vs_norm_df,
)
from .codebook_analysis import (
    TURBO_CENTROIDS,
    bit_exact_reproducibility,
    codebook_bucket_usage,
    codebook_vision_vs_text_skew,
    plot_codebook_skew,
    plot_codebook_usage,
    quantize_to_codebook,
)
from .report import generate_full_report

# -- visualization (histogram plots) ----------------------------------------
from .visualization import (
    plot_value_histogram,
    plot_prerot_vs_postrot,
    plot_vision_vs_text_histogram,
    generate_all_histograms,
)

# -- layer-wise curve plots --------------------------------------------------
from .layer_plots import (
    plot_kv_norm_ratio_curve,
    plot_layer_distortion_curves,
    plot_layer_cosine_curves,
    plot_layer_relative_error_curves,
    plot_all_layer_curves,
)

__all__ = [
    # loader
    "KVDump",
    "KVDumpWriter",
    "load_dump",
    # distribution
    "compute_value_stats",
    "compute_norm_stats",
    "compute_per_layer_stats",
    # outliers
    "find_outlier_channels",
    "outlier_statistics",
    "outlier_ratio_per_layer",
    "outlier_ratio_vision_vs_text",
    # quant error
    "compare_dumps",
    "compare_tensors",
    "compare_with_theoretical",
    "summarize_against_theoretical",
    "THEORETICAL_MSE_PER_COORD",
    # rotation
    "apply_fwht",
    "analyze_rotation_per_layer",
    "beta_distribution_fit_test",
    "coordinate_independence_test",
    "fwht_round_trip",
    "sign_flip_mask",
    "vision_vs_text_rotation_analysis",
    "TURBO_DIM",
    "TURBO_SEED",
    # attention
    "attention_entropy",
    "compute_attention_weights_from_kv",
    "js_divergence_attention",
    "kl_divergence_attention",
    "top1_position_match_rate",
    "topk_overlap_rate",
    # position analysis
    "compute_per_token_norms",
    "per_position_outlier_ratio",
    "per_position_quant_error",
    "plot_token_norm_vs_position",
    "plot_position_outlier_heatmap",
    "plot_position_quant_error",
    "token_index_vs_norm_df",
    # codebook analysis
    "TURBO_CENTROIDS",
    "bit_exact_reproducibility",
    "codebook_bucket_usage",
    "codebook_vision_vs_text_skew",
    "plot_codebook_skew",
    "plot_codebook_usage",
    "quantize_to_codebook",
    # report
    "generate_full_report",
    # visualization
    "plot_value_histogram",
    "plot_prerot_vs_postrot",
    "plot_vision_vs_text_histogram",
    "generate_all_histograms",
    # layer plots
    "plot_kv_norm_ratio_curve",
    "plot_layer_distortion_curves",
    "plot_layer_cosine_curves",
    "plot_layer_relative_error_curves",
    "plot_all_layer_curves",
]
