# Official Parity Audit

Last updated: 2026-04-12  
Repo commit audited: `5e3bbe3`

## Goal

This document audits whether the current benchmark implementation in
`bench/tq_bench/` matches the *official* evaluation protocol of each
benchmark used by this repository.

Here, "official parity" means matching the benchmark's intended behavior
across all of the following dimensions:

1. dataset split
2. sample coverage / aggregation
3. prompt or answer protocol
4. answer extraction / response parsing
5. scoring function

If any of the above differs materially, the benchmark is **not** considered
to have exact official parity.

## Scope

Local code audited:

- `bench/configs/benchmarks.yaml`
- `bench/tq_bench/datasets/base.py`
- `bench/tq_bench/datasets/vlm.py`
- `bench/tq_bench/datasets/text.py`
- `bench/tq_bench/evaluators/mcq.py`
- `bench/tq_bench/evaluators/vqa.py`
- `bench/tq_bench/runner.py`

Primary external sources audited:

- MMLU official eval code:
  `https://raw.githubusercontent.com/hendrycks/test/master/evaluate.py`
- MMMU official README + eval utils:
  `https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/main/mmmu/README.md`
  `https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/main/mmmu/utils/eval_utils.py`
- MathVista official answer extraction + scoring:
  `https://raw.githubusercontent.com/lupantech/MathVista/main/evaluation/extract_answer.py`
  `https://raw.githubusercontent.com/lupantech/MathVista/main/evaluation/calculate_score.py`
- TextVQA challenge docs + official MMF evaluator:
  `https://raw.githubusercontent.com/facebookresearch/mmf/main/website/docs/challenges/textvqa_challenge.md`
  `https://raw.githubusercontent.com/facebookresearch/mmf/main/mmf/utils/m4c_evaluators.py`
- ChartQA repo / paper hub:
  `https://github.com/vis-nlp/ChartQA`
  `https://aclanthology.org/2022.findings-acl.177/`
- ChartQAPro official evaluator:
  `https://raw.githubusercontent.com/vis-nlp/ChartQAPro/main/evaluate_predictions.py`
- OCRBench-v2 official README + evaluator:
  `https://raw.githubusercontent.com/Yuliang-Liu/MultimodalOCR/main/OCRBench_v2/README.md`
  `https://raw.githubusercontent.com/Yuliang-Liu/MultimodalOCR/main/OCRBench_v2/eval_scripts/eval.py`
- CommonsenseQA README:
  `https://raw.githubusercontent.com/jonathanherzig/commonsenseqa/master/README.md`
- HellaSwag README:
  `https://raw.githubusercontent.com/nyu-mll/hellaswag/master/README.md`
- AI2D dataset card:
  `https://huggingface.co/datasets/lmms-lab/ai2d/raw/main/README.md`

## Executive Summary

The current framework does **not** have exact official parity for any of the
11 configured benchmarks.

Main reasons:

1. The framework evaluates deterministic sampled subsets rather than the full
   benchmark split.
2. The framework uses a unified **chat generation + string scoring** pipeline,
   while several official benchmarks assume different evaluation protocols:
   logprob ranking, benchmark-specific answer extraction, benchmark-specific
   normalization, or task-type-dependent scorers.
3. Some benchmark implementations use the correct high-level metric family but
   a simplified local approximation rather than the benchmark's official scorer.

This does **not** make the framework useless. It is still valid for
*controlled relative comparison* across KV cache runtimes inside this project.
But it means:

- score numbers are not always directly comparable to official leaderboard
  numbers
- only a subset of benchmarks are close enough for "near-official" claims
- VLM baseline reproduction claims should be restricted to benchmarks whose
  scorer and protocol are brought into parity first

## Status Labels

- `No exact parity`: local implementation materially differs from official
  split/protocol/parser/scorer
- `Partial parity`: local implementation tracks the official task reasonably
  well, but at least one material difference remains
- `Near parity`: scorer is effectively aligned, but subset coverage and/or
  protocol details still differ

## Summary Table

| Benchmark | Local metric | Official parity | Main issue |
|---|---|---|---|
| AI2D | `option_match` | Partial parity | subset evaluation; no proof of full official protocol match |
| ChartQA | `relaxed_accuracy` | Partial parity | likely close metric, but subset and protocol differences remain |
| ChartQAPro | `relaxed_accuracy` | No exact parity | official scorer is question-type dependent, local scorer is too simple |
| DocVQA | `anls` | Near parity | scorer is aligned, but uses validation subset rather than official full eval flow |
| MathVista | `mathvista_match` | No exact parity | official extraction/normalization pipeline is more complex |
| MMMU | `option_match` | No exact parity | official parse/eval flow differs for MCQ and open questions |
| OCRBench-v2 | `exact_match` | No exact parity | official benchmark is multi-task with many specialized metrics |
| TextVQA | `normalized_exact_match` | No exact parity | official VQA consensus scoring is approximated, not replicated exactly |
| MMLU | `option_match` | No exact parity | official protocol is few-shot + choice logprob ranking, not free generation |
| CommonsenseQA | `option_match` | No exact parity | validation subset + generation scoring, not official leaderboard protocol |
| HellaSwag | `option_match` | No exact parity | validation subset + generation scoring, not standard ranking-style protocol |

## Global Mismatches

### 1. Subset sampling instead of full benchmark coverage

`BaseBenchmarkDataset._deterministic_sample()` shuffles and selects `n` examples.
This affects every benchmark in the project.

Implication:

- scores are useful for *relative runtime comparison*
- scores are not equivalent to official full-split benchmark scores

Relevant local code:

- `bench/tq_bench/datasets/base.py`
- `bench/configs/benchmarks.yaml`

### 2. Unified generation-based evaluation

All benchmarks run through the same `BenchmarkRunner` flow:

1. build prompt
2. send chat completion request
3. score returned text with a local evaluator

This differs from official evaluation for benchmarks that expect:

- logprob ranking over fixed choices
- benchmark-specific answer extraction
- task-type-specific evaluation branches

Relevant local code:

- `bench/tq_bench/runner.py`

### 3. Uniform aggregation by simple mean over sampled items

The runner computes aggregate score as mean of per-sample scores. This is fine
for many benchmarks, but it still differs from official setups when:

- the official benchmark uses hidden test servers
- the official benchmark reports separate category scores
- the official benchmark uses special aggregation over multiple annotations

## Benchmark-by-Benchmark Audit

### AI2D

Local:

- dataset: `lmms-lab/ai2d`, `test`
- answer form: multiple choice
- scorer: letter extraction + exact option match

Parity assessment: `Partial parity`

Why:

- The task formulation is MCQ, and the local scorer is conceptually aligned.
- But the project still evaluates only a deterministic subset and does not
  reproduce an official end-to-end leaderboard protocol beyond plain MCQ
  accuracy.

Risk level: low to medium

### ChartQA

Local:

- dataset: `HuggingFaceM4/ChartQA`, `test`
- scorer: exact normalized string match or numeric match within 5%

Parity assessment: `Partial parity`

Why:

- The local metric is close to standard relaxed chart QA scoring.
- However, the framework still uses subset sampling and a generic generation
  protocol rather than the original benchmark pipeline.
- No official repository scorer was wired directly into the project.

Risk level: medium

### ChartQAPro

Local:

- dataset: `ahmed-masry/ChartQAPro`, `test`
- scorer: plain `relaxed_accuracy`

Official:

- question-type dependent evaluation
- year flags
- exact match branches for fact checking / multi choice
- ANLS fallback for text-like answers
- list-aware handling

Parity assessment: `No exact parity`

Why:

- The local evaluator is too coarse compared with the official
  `evaluate_predictions.py`.
- This benchmark should not be presented as officially reproduced until the
  official scorer is integrated.

Risk level: high

### DocVQA

Local:

- dataset: `lmms-lab/DocVQA`, `validation`
- scorer: ANLS, max across references

Official:

- ANLS is the right metric family for DocVQA

Parity assessment: `Near parity`

Why:

- The local ANLS implementation is aligned with the standard definition:
  normalized Levenshtein similarity with thresholding at 0.5.
- The main gap is not the scorer; it is the use of a sampled validation subset
  instead of the official full challenge flow.

Risk level: low

### MathVista

Local:

- dataset: `AI4Math/MathVista`, `testmini`
- uses official `query` field if present
- local heuristic extraction from response
- local numeric / text / letter matching

Official:

- separate answer extraction stage
- answer-type-aware normalization
- multi-choice maps letter to choice text
- float precision handling is benchmark-specific

Parity assessment: `No exact parity`

Why:

- The local implementation is thoughtfully adapted, but still heuristic.
- The official pipeline explicitly separates extraction and scoring.
- The local `MathVistaMatchEvaluator` is good enough for project comparison,
  but not identical to the official evaluator.

Risk level: high

### MMMU

Local:

- dataset: `MMMU/MMMU`, merged validation subjects
- MCQ and open questions share local `option_match`
- open questions use list/string heuristics via `OptionMatchEvaluator`

Official:

- two modes: evaluation-only with final predictions, or parse-and-eval
- MCQ parser picks option letters or content answers using benchmark logic
- open questions use dedicated normalization / number extraction logic

Parity assessment: `No exact parity`

Why:

- The official MMMU parser/evaluator is benchmark-specific.
- The local implementation is simpler and cannot be called exact parity.

Risk level: high

### OCRBench-v2

Local:

- dataset loader treats the benchmark as plain VQA
- scorer: `exact_match`

Official:

- benchmark contains many task types
- scoring uses task-dependent metrics including OCR, VQA, table, spotting,
  grounding, TEDS, IoU-like and other specialized evaluators

Parity assessment: `No exact parity`

Why:

- This is the largest parity gap in the repo.
- A single exact-match scorer does not represent OCRBench-v2 faithfully.

Risk level: very high

### TextVQA

Local:

- scorer normalizes text and applies `min(matches / 3, 1)` on all references
- includes a convenience short-answer extraction path

Official:

- uses EvalAI/MMF answer normalization
- uses the 10-human-answer consensus score computed by leave-one-out averaging

Parity assessment: `No exact parity`

Why:

- The local scorer is a reasonable approximation but not the exact official
  VQA consensus computation.
- The local short-answer fallback also makes the behavior more permissive than
  the official evaluator.

Risk level: high

### MMLU

Local:

- dataset: `cais/mmlu`, `all`, `test`
- prompt: zero-shot generation with answer letter requested
- scorer: letter extraction + exact letter match

Official:

- few-shot prompt construction
- evaluates by ranking fixed answer options through token logprobs

Parity assessment: `No exact parity`

Why:

- This is a protocol mismatch, not just a scorer mismatch.
- Current MMLU numbers are project-internal sanity checks, not official MMLU
  reproduction numbers.

Risk level: high

### CommonsenseQA

Local:

- dataset: `tau/commonsense_qa`, `validation`
- prompt: generation of answer letter
- scorer: local MCQ option match

Official:

- leaderboard is tied to benchmark splits and evaluation scripts
- current local path does not replicate official leaderboard protocol

Parity assessment: `No exact parity`

Why:

- The benchmark is useful as an internal MCQ stress test, but not an official
  CommonsenseQA reproduction.

Risk level: medium to high

### HellaSwag

Local:

- dataset: `Rowan/hellaswag`, `validation`
- prompt: generation of answer letter
- scorer: local MCQ option match

Official:

- typical evaluation is ranking endings rather than free-form generation

Parity assessment: `No exact parity`

Why:

- The task is MCQ-like, but the evaluation protocol is still different enough
  that leaderboard comparability should not be claimed.

Risk level: medium to high

## Recommended Priority Order

### P0: Required for Phase 9 official VLM reproduction claims

1. `MMMU`
2. `MathVista`
3. `AI2D`

Reason:

These are the three benchmarks explicitly targeted for official
Qwen3-VL-2B comparison in this repo.

### P1: High-value VLM parity fixes

4. `TextVQA`
5. `ChartQA`
6. `ChartQAPro`
7. `DocVQA`

### P2: Lowest parity today, but also highest implementation cost

8. `OCRBench-v2`

### P3: Text sanity-check benchmarks

9. `MMLU`
10. `CommonsenseQA`
11. `HellaSwag`

These are valuable for internal relative comparison, but not the first place
to spend effort if the goal is official VLM-score parity.

## Concrete Fixes Needed

### Must-do for official parity claims

- remove subset sampling for parity runs
- record full benchmark coverage separately from fast debug runs
- integrate official scorer code where it exists instead of local approximation
- split "debug / internal comparison" mode from "official parity" mode

### Benchmark-specific fixes

- `MMMU`: use official parse-and-eval or eval-only final-answer flow
- `MathVista`: use official extraction and calculation scripts
- `TextVQA`: port official MMF `TextVQAAccuracyEvaluator` exactly
- `ChartQAPro`: replace local relaxed scorer with official evaluator
- `OCRBench-v2`: redesign around task-type-specific official metrics
- `MMLU`: add a ranking-based evaluator instead of generation-only scoring

## Practical Interpretation for This Repo

Current scores are best interpreted as:

- **good for comparing KV cache runtime variants under one controlled harness**
- **not uniformly valid as official benchmark reproduction numbers**

Therefore:

- claiming "TurboQuant variant A beats variant B" inside this harness is fine
- claiming "our score matches official benchmark X" is only safe after per-task
  parity is explicitly closed

## Recommended Follow-up Deliverables

1. `docs/OFFICIAL_PARITY_ACTION_PLAN.md`
   - one checklist per benchmark
2. `parity_mode` in config
   - full split
   - official scorer
   - no debug subsampling
3. benchmark-specific adapters
   - `mathvista_official.py`
   - `mmmu_official.py`
   - `textvqa_official.py`
   - `ocrbench_v2_official.py`

## Bottom Line

The current benchmark framework is strong as a **research harness for relative
KV-cache comparisons**, but it is not yet a **benchmark-exact reproduction
framework**.

The best immediate target is not "fix everything at once". It is:

1. close official parity for `AI2D`, `MMMU`, and `MathVista`
2. clearly label every other benchmark as internal / approximate until proven
   otherwise
