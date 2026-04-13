#!/bin/bash
# Phase 9-E is already running externally. This script runs 9-C, 9-D, 10-A, 10-B, 11-A sequentially.
# Started by Claude while user sleeps.

set -e
cd /home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH/bench

LOG="/home/kch3dri4n/lab/TQ_LLAMACPP_VSION_BENCH/results/runs/all_phases_log.txt"
echo "=== Phase 9-E ~ 11-A sequential run ===" | tee "$LOG"
echo "Started: $(date)" | tee -a "$LOG"

# Find the latest 9-E output (tq-all result) for resume
find_latest() {
    ls -t ../results/runs/bench_qwen3_vl_2b_instruct_*tq-all*p0*n100*.json 2>/dev/null | head -1
}

# Find the latest vlm result for resume
find_latest_vlm() {
    ls -t ../results/runs/bench_qwen3_vl_2b_instruct_*core*vlm*n100*.json 2>/dev/null | head -1
}

###############################################################################
# Phase 9-C: vlm × core × n=100 (resume from 9-E which includes core P0)
###############################################################################
echo "" | tee -a "$LOG"
echo "=== Phase 9-C: vlm × core × n=100 ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

RESUME_9E=$(find_latest)
if [ -z "$RESUME_9E" ]; then
    echo "WARNING: No 9-E output found, running without resume" | tee -a "$LOG"
    uv run python run_bench.py --num 100 --runtimes core --benchmarks vlm \
        --model qwen3_vl_2b_instruct --model-quant bf16 2>&1 | tee -a "$LOG"
else
    echo "Resuming from: $RESUME_9E" | tee -a "$LOG"
    uv run python run_bench.py --num 100 --runtimes core --benchmarks vlm \
        --model qwen3_vl_2b_instruct --model-quant bf16 \
        --resume "$RESUME_9E" 2>&1 | tee -a "$LOG"
fi
echo "9-C done: $(date)" | tee -a "$LOG"

###############################################################################
# Phase 9-D: text × core × n=100
###############################################################################
echo "" | tee -a "$LOG"
echo "=== Phase 9-D: text × core × n=100 ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

uv run python run_bench.py --num 100 --runtimes core --benchmarks text \
    --model qwen3_vl_2b_instruct --model-quant bf16 2>&1 | tee -a "$LOG"
echo "9-D done: $(date)" | tee -a "$LOG"

###############################################################################
# Phase 10-A: prod smoke n=10
###############################################################################
echo "" | tee -a "$LOG"
echo "=== Phase 10-A: prod smoke × P0 × n=10 ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

uv run python run_bench.py --num 10 --runtimes prod --benchmarks p0 \
    --model qwen3_vl_2b_instruct --model-quant bf16 2>&1 | tee -a "$LOG"
echo "10-A done: $(date)" | tee -a "$LOG"

###############################################################################
# Phase 10-B: prod × P0 × n=50 (run regardless — crashes are recorded)
###############################################################################
echo "" | tee -a "$LOG"
echo "=== Phase 10-B: prod × P0 × n=50 ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

uv run python run_bench.py --num 50 --runtimes prod --benchmarks p0 \
    --model qwen3_vl_2b_instruct --model-quant bf16 2>&1 | tee -a "$LOG"
echo "10-B done: $(date)" | tee -a "$LOG"

###############################################################################
# Phase 11-A: Thinking model smoke n=20
###############################################################################
echo "" | tee -a "$LOG"
echo "=== Phase 11-A: Thinking smoke × P0 × n=20 ===" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"

uv run python run_bench.py --num 20 --runtimes baseline tq-4 tq-K4V3 --benchmarks p0 \
    --model qwen3_vl_2b_thinking --model-quant bf16 2>&1 | tee -a "$LOG"
echo "11-A done: $(date)" | tee -a "$LOG"

###############################################################################
echo "" | tee -a "$LOG"
echo "=== ALL PHASES COMPLETE ===" | tee -a "$LOG"
echo "Finished: $(date)" | tee -a "$LOG"
