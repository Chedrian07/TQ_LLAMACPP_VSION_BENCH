# Official Parity Audit

Last updated: 2026-04-13

## Purpose

This document describes how close the current benchmark framework is to each benchmark's official evaluation protocol and how to interpret scores produced by the default runner.

## Executive Summary

The current codebase contains several official-style scorers, but the default `run_bench.py` flow is still a **sampled project benchmark**, not an exact official evaluation pipeline.

Why:

1. `run_bench.py` overwrites benchmark sample counts with `--num`.
2. The dataset layer supports `parity_sample_count = -1`, but the default runner does not automatically switch to it.
3. Some benchmarks are still evaluated through project-specific generation-based approximations.

When `--benchmarks` is omitted, the runner now defaults to the active 5-benchmark research scope:
`ai2d`, `mmmu`, `mathvista`, `textvqa`, and `docvqa`.
When `--runtimes` is omitted, it defaults to the active runtime set:
`lcpp-kv-8`, `lcpp-kv-4`, all `tq-*`, and all `tqp-*` variants, excluding `baseline`.

This means the framework is strong for **controlled relative comparison across runtimes**, but any “official reproduction” claim must be phrased carefully.

## Code Facts Behind That Verdict

- `bench/configs/benchmarks.yaml` carries `parity_metric` and `parity_sample_count` fields.
- `bench/tq_bench/datasets/base.py` supports `n = -1` as “use the full split”.
- `bench/run_bench.py` currently rebuilds benchmark configs with `sample_count = args.num`.
- `bench/tq_bench/evaluators/__init__.py` registers official-style evaluators for `mmmu`, `mathvista`, `textvqa`, and `chartqapro`.

## Current Benchmark Classification

| Benchmark | Current status | Notes |
| --- | --- | --- |
| `ai2d` | Near parity under project controls | MCQ scoring is aligned enough for controlled comparison, but default runs still use sampled subsets. |
| `mmmu` | Partial parity | Official-style scorer exists, but default runs remain subset-based and generation-driven. |
| `mathvista` | Partial parity | Official-style scorer exists, but default runs remain subset-based. |
| `chartqapro` | Partial parity | Official-style scorer exists, but it is not part of the default P0 path and still depends on sampled invocation. |
| `textvqa` | Partial parity | Official-style scorer exists, but default runs remain subset-based. |
| `docvqa` | Near parity under project controls | ANLS is appropriate, but the runner is still subset-based and not an official challenge submission path. |
| `chartqa` | Partial parity | Relaxed-accuracy scoring is useful, but it is still a project approximation. |
| `ocrbench_v2` | No exact parity | Official evaluation requires task-specific metrics not reproduced in the default runner. |
| `mmlu` | No exact parity | Official evaluations are ranking/logprob-based, not free-form generation MCQ. |
| `commonsenseqa` | No exact parity | Project score is useful for runtime comparison, not leaderboard comparison. |
| `hellaswag` | No exact parity | Project score is useful for runtime comparison, not leaderboard comparison. |

## How To Interpret Results

### Safe claims

- Runtime A is better or worse than runtime B under the same project setup.
- A scorer is more faithful than a previous local approximation.
- A sampled P0 run moved closer to or farther from published reference numbers.

### Unsafe claims without extra work

- “This exactly reproduces the official leaderboard score.”
- “This benchmark now has full official parity.”
- “A sampled `--num` run is directly comparable to a full official split.”

## Recommended Wording

When reporting current results, prefer wording like:

- “sampled P0 comparison with official-style scorers”
- “project-controlled reproduction check”
- “relative comparison under the repository's benchmark protocol”

Avoid wording like:

- “official reproduced score”
- “exact leaderboard parity”

unless the invocation path has been updated to use the full split and benchmark-specific protocol end to end.

## What Would Be Needed For Exactness

1. Stop overriding sample counts with `--num` for exact-parity runs.
2. Wire benchmark-specific full-split invocation paths.
3. Keep scorer, answer extraction, and aggregation aligned per benchmark.
4. Separate “sampled comparison mode” from “exact parity mode” in the CLI and documentation.
