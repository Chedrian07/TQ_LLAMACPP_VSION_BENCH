#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from tq_bench.datasets import get_dataset, list_benchmarks

for bid in list_benchmarks():
    print(f"\n>>> Loading {bid}...")
    try:
        ds = get_dataset(bid)
        ds.load(n_samples=5, seed=42)
        samples = list(ds.iter_samples())
        print(f"  OK {bid}: {len(samples)} samples loaded")
        if samples:
            s0 = samples[0]
            print(f"    keys: {list(s0.keys())}")
    except Exception as e:
        print(f"  FAIL {bid}: {type(e).__name__}: {e}")
