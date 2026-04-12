# AGENTS.md

## Purpose

This repository benchmarks the effect of TurboQuant KV cache quantization on vision-language models using a patched `llama.cpp` fork.

The work splits into two tracks:

1. `llama.cpp/` implementation of new TurboQuant GGML KV cache types
2. `bench/` framework for running VLM and text benchmarks against `llama-server`

`CLAUDE.md` is the detailed design document.
This file is the short operational guide for coding agents working in this repository.

## Repository Layout

```text
.
├── AGENTS.md
├── CLAUDE.md
├── llama.cpp/         # upstream fork to patch
├── bench/             # Python benchmark framework
├── models/            # local GGUF/mmproj files, ignored by git
├── results/           # experiment outputs, ignored by git
└── pdfs/              # reference papers
```

### `llama.cpp/`

Primary patch targets:

- `ggml/include/ggml.h`
- `ggml/src/ggml.c`
- `ggml/src/ggml-common.h`
- `ggml/src/ggml-quants.h`
- `ggml/src/ggml-quants.c`
- `ggml/src/ggml-cpu/quants.c`
- `ggml/src/ggml-cuda/`
- `src/llama-kv-cache.cpp`
- `src/llama-kv-cache.h`
- `common/arg.cpp`

### `bench/`

Current scaffold already exists:

- `configs/` for runtime, benchmark, and model YAML
- `tq_bench/` for the Python package
- `notebooks/` for thin execution wrappers

Most Python modules are scaffold-only and intentionally still contain `NotImplementedError`.

## Project Goal

Implement and benchmark:

- TurboQuant Algorithm 1 (MSE): `turbo2`, `turbo2h`, `turbo3`, `turbo3h`, `turbo4`
- TurboQuant Algorithm 2 (prod/QJL): `turbop3`, `turbop4`, `turbop5`
- Baselines: `f16`, `q8_0`, `q4_0`, `q2_K`

Main target model family:

- `Qwen3-VL-2B-Instruct`
- `Qwen3-VL-2B-Thinking`

Main research claim to validate:

- MSE-only TurboQuant can be benchmarked systematically on VLMs
- prod/QJL variants likely fail in real generation despite decent cosine similarity

## Immediate Priorities

Follow this order unless the user asks otherwise:

1. Phase 1: add MSE TurboQuant GGML types to `llama.cpp`
2. Phase 2: add prod/QJL GGML types
3. Phase 3: add CUDA kernels
4. Phase 4: wire KV cache CLI and runtime integration
5. Phase 5: implement the Python benchmark framework in `bench/`
6. Phase 6: run smoke tests, then full benchmarks
7. Phase 7: export analysis and reports

## Working Rules

- Preserve upstream `llama.cpp` coding style in C/C++ changes.
- Use 4-space indentation in new code.
- Keep new GGML type naming consistent with existing quant types.
- Do not commit model files, benchmark outputs, caches, or notebook checkpoints.
- Treat `models/` and `results/` as local-only storage.
- Keep notebook logic thin; put real logic in `bench/tq_bench/`.
- When adding benchmark functionality, prefer reusable package code over notebook-only code.
- When implementing evaluation logic, keep prod runtime failures representable as explicit failed records rather than silent skips.

## Bench Framework Expectations

The intended flow is:

1. Load YAML configs
2. Build runtime × benchmark matrix
3. Launch `llama-server` with the requested KV cache types
4. Query the OpenAI-compatible API
5. Evaluate outputs
6. Persist checkpoints and result artifacts
7. Generate CSV, JSON, and markdown summaries

Dual-GPU execution is a requirement for the orchestrator design.

## Build And Run

### `llama.cpp`

```bash
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### `bench`

```bash
cd bench
uv sync
uv run jupyter lab
```

### Quick Python sanity check

```bash
python3 -m compileall bench/tq_bench
```

## Notes For Agents

- Read `CLAUDE.md` before making major architectural decisions.
- Prefer small, verifiable increments because both `llama.cpp` kernel work and benchmark orchestration are easy to break.
- If you touch `llama.cpp` quantization paths, verify both build integrity and the affected runtime CLI wiring.
- If you touch `bench/`, verify config loading and package importability before moving on.
- If the user asks only for scaffolding or layout work, do not jump ahead into algorithm implementation without instruction.

