#!/usr/bin/env python3
"""Re-score an existing MMMU run with raw and eval-only official scorers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tq_bench.datasets.vlm import MMMUDataset
from tq_bench.evaluators.mmmu_official import (
    MMMUEvalOnlyEvaluator,
    MMMUOfficialRawEvaluator,
)


def _load_record(run_path: Path, model_id: str, runtime_id: str) -> dict[str, Any]:
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    matches = [
        rec for rec in records
        if rec.get("benchmark_id") == "mmmu"
        and rec.get("model_id") == model_id
        and rec.get("runtime_id") == runtime_id
    ]

    if not matches:
        raise SystemExit(
            f"No MMMU record found in {run_path} for model={model_id!r}, "
            f"runtime={runtime_id!r}"
        )
    if len(matches) > 1:
        raise SystemExit(
            f"Multiple MMMU records found in {run_path} for model={model_id!r}, "
            f"runtime={runtime_id!r}; narrow the selection first."
        )

    return payload, matches[0]


def _load_samples(n_samples: int, seed: int) -> dict[str, dict[str, Any]]:
    ds = MMMUDataset()
    ds.load(n_samples, seed=seed)
    return {sample["id"]: dict(sample) for sample in ds.iter_samples()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_json", type=Path, help="Existing benchmark JSON file")
    parser.add_argument("--model", required=True, help="Model id inside the run file")
    parser.add_argument("--runtime", required=True, help="Runtime id inside the run file")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed override")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a JSON rescore report",
    )
    args = parser.parse_args()

    payload, record = _load_record(args.run_json, args.model, args.runtime)
    seed = args.seed
    if seed is None:
        seed = int(payload.get("meta", {}).get("seed", 42))

    samples = _load_samples(int(record["n_samples"]), seed)
    raw_ev = MMMUOfficialRawEvaluator(seed=seed)
    eval_only_ev = MMMUEvalOnlyEvaluator()

    rows: list[dict[str, Any]] = []
    stored_sum = 0.0
    raw_sum = 0.0
    eval_only_sum = 0.0

    for sr in record["sample_results"]:
        sample_id = sr["sample_id"]
        sample = samples[sample_id]
        prediction = sr["prediction"]
        stored_score = float(sr["score"])
        raw_score = raw_ev.score(prediction, sample["answer"], metadata=sample)
        eval_only_score = eval_only_ev.score(prediction, sample["answer"], metadata=sample)
        extracted = eval_only_ev.extract_prediction(prediction, metadata=sample)

        stored_sum += stored_score
        raw_sum += raw_score
        eval_only_sum += eval_only_score

        rows.append(
            {
                "sample_id": sample_id,
                "question_type": sample["question_type"],
                "stored_score": stored_score,
                "raw_score": raw_score,
                "eval_only_score": eval_only_score,
                "extracted_prediction": extracted,
                "reference": sample["answer"],
                "prediction": prediction,
            }
        )

    n_samples = len(rows)
    summary = {
        "run_json": str(args.run_json),
        "model_id": args.model,
        "runtime_id": args.runtime,
        "seed": seed,
        "n_samples": n_samples,
        "stored_score": round(stored_sum / n_samples, 4) if n_samples else 0.0,
        "raw_score": round(raw_sum / n_samples, 4) if n_samples else 0.0,
        "eval_only_score": round(eval_only_sum / n_samples, 4) if n_samples else 0.0,
        "stored_metric": record.get("metric_used"),
        "diff_counts": {
            "stored_vs_raw": sum(1 for row in rows if row["stored_score"] != row["raw_score"]),
            "stored_vs_eval_only": sum(
                1 for row in rows if row["stored_score"] != row["eval_only_score"]
            ),
            "raw_vs_eval_only": sum(
                1 for row in rows if row["raw_score"] != row["eval_only_score"]
            ),
        },
    }

    print(
        f"MMMU re-score: model={args.model} runtime={args.runtime} "
        f"n={n_samples} seed={seed}"
    )
    print(
        f"  stored={summary['stored_score']:.4f} "
        f"raw={summary['raw_score']:.4f} "
        f"eval_only={summary['eval_only_score']:.4f}"
    )
    print(
        "  diffs: "
        f"stored/raw={summary['diff_counts']['stored_vs_raw']} "
        f"stored/eval_only={summary['diff_counts']['stored_vs_eval_only']} "
        f"raw/eval_only={summary['diff_counts']['raw_vs_eval_only']}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"  wrote report: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
