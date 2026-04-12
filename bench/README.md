# `bench/` — TQ-VLM-Bench Python Framework

Python framework that drives the patched `llama-server` across the full runtime × benchmark matrix. See the project [README](../README.md) and [CLAUDE.md](../CLAUDE.md) for context.

## Layout

```
bench/
├── pyproject.toml            # UV project
├── configs/
│   ├── runtimes.yaml         # 15 runtime conditions (baseline, lcpp-kv-*, tq-*, tqp-*)
│   ├── benchmarks.yaml       # 11 benchmarks (8 VLM + 3 text)
│   └── models.yaml           # GGUF + mmproj paths
├── tq_bench/                 # Core package
│   ├── config.py             # YAML loading, matrix expansion
│   ├── server.py             # llama-server subprocess manager
│   ├── client.py             # OpenAI-compatible HTTP client (VLM base64)
│   ├── datasets/             # HF-backed loaders (VLM + text)
│   ├── evaluators/           # VQA (ANLS, relaxed), MCQ (option match)
│   ├── runner.py             # single cell: start → generate → evaluate → stop
│   ├── orchestrator.py       # matrix traversal, dual-GPU parallel, resume
│   └── reporters/            # CSV/JSON export, charts, markdown summary
├── notebooks/                # thin execution/analysis wrappers
├── preload_datasets.py       # cache HF datasets locally
├── verify_datasets.py        # sanity-check local caches
└── smoke_test.py             # fast CLI entrypoint
```

## Install

```bash
uv sync
```

## Quick start

```bash
# Smoke test a few cells
uv run python smoke_test.py --runtime baseline --runtime tq-3 \
  --benchmark commonsenseqa --n 3

# Full matrix via notebook
uv run jupyter lab notebooks/02_full_run.ipynb
```

## Design notes

- `llama-server` is launched per cell with `--cache-type-k` / `--cache-type-v` derived from the runtime config (`turbo*`, `turbop*`, `q4_0`, `f16`, ...).
- The orchestrator checkpoints each completed cell to `results/runs/<ts>/checkpoint.json` for resume.
- Dual-GPU mode launches two servers on ports 8080/8081 (GPU 0/1) and dispatches cells through a 2-worker pool.
- `prod` (`tqp-*`) runtimes are expected to fail generation; the runner records this as a FAIL cell rather than raising.
- All seeds default to 42. Temperature is pinned to 0.0.

## Configuration

Edit `configs/runtimes.yaml` to add or disable runtimes, `configs/benchmarks.yaml` to adjust sample counts `N`, and `configs/models.yaml` to point at local GGUF files. Matrix expansion is driven entirely from these files — no code changes needed for new combinations.
