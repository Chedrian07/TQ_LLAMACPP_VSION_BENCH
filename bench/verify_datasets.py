#!/usr/bin/env python3
"""Verify all dataset loaders and evaluators work end-to-end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PIL import Image

from tq_bench.datasets import get_dataset, list_benchmarks
from tq_bench.evaluators import get_evaluator

METRICS = {
    "ai2d": "option_match",
    "chartqa": "relaxed_accuracy",
    "chartqapro": "relaxed_accuracy",
    "docvqa": "anls",
    "mathvista": "mathvista_match",
    "mmmu": "option_match",
    "ocrbench_v2": "exact_match",
    "textvqa": "normalized_exact_match",
    "mmlu": "option_match",
    "commonsenseqa": "option_match",
    "hellaswag": "option_match",
}

VLM_BENCHMARKS = {"ai2d", "chartqa", "chartqapro", "docvqa", "mathvista", "mmmu", "ocrbench_v2", "textvqa"}
TEXT_BENCHMARKS = {"mmlu", "commonsenseqa", "hellaswag"}

# Mock predictions for evaluator testing
MOCK_PREDICTIONS = {
    "option_match": ("B", "B"),          # (prediction, reference)
    "relaxed_accuracy": ("42.5", "42"),
    "anls": ("hello world", ["hello world", "hello"]),
    "exact_match": ("Yes", "yes"),
    "normalized_exact_match": ("the cat", ["cat", "a cat"]),
    "mathvista_match": ("The answer is 3.14", "3.14"),
}

passed = 0
failed = 0

for bid in list_benchmarks():
    print(f"\n{'='*60}")
    print(f"Testing: {bid}")
    print(f"{'='*60}")
    try:
        # 1. Load 2 samples
        ds = get_dataset(bid)
        ds.load(n_samples=2, seed=42)
        samples = list(ds.iter_samples())
        print(f"  Loaded {len(samples)} samples")

        if len(samples) == 0:
            print(f"  [FAIL] No samples loaded!")
            failed += 1
            continue

        # 2. Print first sample keys and truncated values
        s = samples[0]
        print(f"  Keys: {list(s.keys())}")
        for k, v in s.items():
            if isinstance(v, Image.Image):
                print(f"    {k}: PIL.Image mode={v.mode} size={v.size}")
            elif isinstance(v, str):
                print(f"    {k}: {v[:80]!r}{'...' if len(v) > 80 else ''}")
            elif isinstance(v, list) and len(v) > 0:
                print(f"    {k}: list[{len(v)}] first={str(v[0])[:60]!r}")
            else:
                print(f"    {k}: {v!r}")

        # 3. VLM: verify image field is PIL Image
        if bid in VLM_BENCHMARKS:
            img = s.get("image")
            if isinstance(img, Image.Image):
                print(f"  [OK] 'image' is PIL.Image ({img.mode}, {img.size})")
            else:
                print(f"  [FAIL] 'image' is {type(img)}, expected PIL.Image")
                failed += 1
                continue

        # 4. Text: verify options field exists
        if bid in TEXT_BENCHMARKS:
            if "options" in s:
                print(f"  [OK] 'options' present: {len(s['options'])} choices")
            else:
                print(f"  [FAIL] 'options' field missing for text benchmark")
                failed += 1
                continue

        # 5. Test evaluator with mock prediction
        metric = METRICS[bid]
        evaluator = get_evaluator(metric)
        pred, ref = MOCK_PREDICTIONS[metric]
        score = evaluator.score(pred, ref)
        print(f"  Evaluator '{metric}': mock score = {score:.4f}")

        passed += 1
        print(f"  [PASS]")

    except Exception as e:
        failed += 1
        print(f"  [FAIL] {type(e).__name__}: {e}")

print(f"\n{'='*60}")
print(f"SUMMARY: {passed} passed, {failed} failed out of {passed + failed}")
print(f"{'='*60}")
sys.exit(1 if failed > 0 else 0)
