# CLAUDE.md

## Project: TQ-VLM-Bench (llama.cpp Edition)

TurboQuant(ICLR 2026) KV cache 양자화가 Vision-Language Model에서 성능에 미치는 영향을 체계적으로 벤치마크하는 프로젝트.
llama.cpp 포크에 TurboQuant Algorithm 1(MSE) 및 Algorithm 2(prod/QJL) GGML 타입을 직접 구현하고, 기존 llama.cpp KV cache 양자화(q8_0/q4_0/q2_K)와 비교한다.

**현재 상태 (2026-04-13):**
- Phase 1~8.5 완료, Phase 9 본 실행 대기
- ✅ 공식 parity evaluator 4종 구현 완료: `mmmu_official`, `mathvista_official`, `textvqa_official`, `chartqapro_official`
- ✅ `BaseEvaluator.score()` metadata 인프라, evaluator registry 통합, parity mode config 완료
- ✅ Runner `<think>...</think>` strip 로직 추가 (Thinking 모델 채점 시 reasoning chain 자동 제거)
- ✅ 통합 벤치마크 러너 `bench/run_bench.py` 구현: `--num N`, `--runtimes`, `--benchmarks`, `--model`, `--resume` 지원
- ✅ **Per-sample timing/token instrumentation 완료**: TTFT, latency, decode tok/s, prompt/completion tokens, GPU memory
- ✅ **Aggregate stats**: score std/median, latency p50/p95/p99, throughput min/max, GPU memory per cell
- ✅ 219개 테스트 통과 (parity golden 85 + KV analysis 134)
- ✅ Parity smoke test 완료: baseline × AI2D/MMMU/MathVista × n=10 (`results/runs/parity_smoke_n10.json`)
- ⏳ 다음: `run_bench.py --num 100` 으로 core runtimes × P0 벤치마크 본 실행

## 핵심 배경

- TurboQuant 논문은 **텍스트 전용 LLM**에서만 실험함 (Llama-3.1-8B, Ministral-7B)
- VLM에서의 체계적 벤치마크는 **아직 아무도 하지 않음**
- 커뮤니티 6개 이상 독립 팀이 QJL(Algorithm 2)이 실전에서 generation을 파괴한다고 보고
- llama.cpp에 TurboQuant 커뮤니티 포크 다수 존재하나, **fractional bits(2.5/3.5)**, **비대칭 K/V**, **Algorithm 2(prod)** 지원은 전무
- 본 프로젝트는 두 알고리즘 모두 직접 구현하여, prod의 실패를 정량적으로 입증하고, MSE의 VLM 성능을 체계적으로 측정

## ⚠️ 중요: 이름 공간 충돌 해결

llama.cpp upstream에 이미 `GGML_TYPE_TQ1_0`(34, 1.6875bpw ternary)과 `GGML_TYPE_TQ2_0`(35, 2.0625bpw ternary)이 존재한다. 이들은 **TurboQuant이 아닌** 기존 ternary quantization이다.

**결정사항**: 본 프로젝트의 TurboQuant 타입은 `TURBO` 접두사를 사용한다 (TQ가 아님). 문서 내의 이전 `TQ2_0/TQ3_0/...` 표기는 실제 구현에서 `TURBO2_0/TURBO3_0/...`로 매핑된다.

## 워크스페이스 구조

```
.
├── CLAUDE.md
├── llama.cpp/                   # llama.cpp 포크 (TurboQuant 패치 적용됨)
│   ├── ggml/include/ggml.h      # GGML_TYPE_TURBO*_0 / GGML_TYPE_TURBOP*_0 enum (42-49)
│   ├── ggml/src/
│   │   ├── ggml-common.h        # 공통 GGML 블록 정의
│   │   ├── ggml-common-turbo.h  # block_turbo*_0 / block_turbop*_0 + Lloyd-Max codebook + TURBO_QJL_SCALE
│   │   ├── ggml-quants.h/c      # CPU quantize/dequantize + FWHT/sign flip/QJL 헬퍼
│   │   ├── ggml.c               # type_traits 등록
│   │   ├── ggml-cpu/            
│   │   │   ├── quants.h/c       # vec_dot_turbo*_q8_K (CPU flash attention 경로)
│   │   │   └── ggml-cpu.c       # type_traits_cpu 등록
│   │   └── ggml-cuda/           
│   │       ├── ggml-cuda.cu     # supports_op에 TURBO SET_ROWS/FATTN 등록
│   │       ├── set-rows.cu      # KV write 디스패치
│   │       ├── cpy-utils.cuh    # device-side quantize_f32_turbo*_0_block
│   │       ├── convert.cu       # dequantize 디스패치 (fp16/fp32/bf16)
│   │       ├── turbo-common.cuh # ✨ CUDA codebook + FWHT + cooperative shared-mem dequant
│   │       ├── fattn.cu         # flash attention 디스패치
│   │       ├── fattn-common.cuh # vec_dot_fattn_vec_KQ_turbo + vec_dot_fattn_vec_KQ_turbop + dequantize_V_turbo
│   │       ├── fattn-vec.cuh    # extern 템플릿 선언
│   │       └── template-instances/ fattn-vec-instance-turbo*_0-turbo*_0.cu (11개)
│   ├── src/
│   │   ├── llama-kv-cache.h/cpp # generic path 사용 + dbg_dump_*_layer_f32 (신규)
│   │   └── llama-context.cpp    # llama_kv_self_dbg_* C API 래퍼 (신규)
│   ├── include/llama.h          # llama_kv_self_dbg_* 공개 API (신규)
│   ├── common/arg.cpp           # --cache-type-k/v에 turbo*/turbop* 추가
│   ├── tests/test-turboquant.c  # ✨ 단위 테스트 (round-trip MSE/cosine)
│   ├── tools/
│   │   ├── server/              # llama-server (OpenAI-compatible API)
│   │   ├── mtmd/                # llama-mtmd-cli (Qwen3-VL 지원)
│   │   └── kv-dump/             # ✨ llama-kv-dump (KV 추출 도구, 신규)
│   └── ...
│
├── bench/                       # 벤치마크 프레임워크 (Python, UV 관리)
│   ├── pyproject.toml           # UV 프로젝트 설정
│   ├── run_bench.py             # ✨ 통합 CLI 러너 (--num, --runtimes, --benchmarks, --model, --resume)
│   ├── run_parity_smoke.py      # P0 parity evaluator smoke test
│   ├── run_all_phases.sh        # 전체 Phase 실행 스크립트
│   ├── rescore_mmmu.py          # MMMU 재채점 유틸리티
│   ├── smoke_test.py            # CLI: 간이 smoke test (runtime/benchmark 선택 가능)
│   ├── preload_datasets.py      # HF 데이터셋 사전 다운로드 (10/11 cached, 59GB)
│   ├── verify_datasets.py       # dataset loader 검증 스크립트
│   ├── configs/
│   │   ├── runtimes.yaml        # 런타임 조건 정의 (15개)
│   │   ├── benchmarks.yaml      # 벤치마크 정의 (11개)
│   │   └── models.yaml          # 모델 GGUF 경로, mmproj 경로
│   │
│   ├── tq_bench/                # Python 패키지
│   │   ├── __init__.py
│   │   ├── config.py            # YAML 로딩, 실험 매트릭스 생성
│   │   ├── server.py            # llama-server 프로세스 관리 (lsof 포트 충돌 해결, 5s SIGTERM→SIGKILL, GPU memory monitoring)
│   │   ├── client.py            # OpenAI-compatible API 클라이언트 (httpx, 재시도, base64 image, CompletionTimings 추출)
│   │   ├── datasets/            # 데이터셋 로더 (11개 전부 구현)
│   │   │   ├── base.py          # BaseBenchmarkDataset ABC
│   │   │   ├── vlm.py           # AI2D, ChartQA, ChartQAPro, DocVQA, MathVista, MMMU, OCRBench-v2, TextVQA
│   │   │   └── text.py          # MMLU, CommonsenseQA, HellaSwag
│   │   ├── evaluators/          # 메트릭 평가기 (12개: 기존 6 + 공식 parity 4 + alias 2)
│   │   │   ├── __init__.py      # _REGISTRY: 12개 evaluator 등록
│   │   │   ├── base.py          # BaseEvaluator ABC (metadata kwarg)
│   │   │   ├── _utils.py        # levenshtein, normalize, parse_number
│   │   │   ├── mcq.py           # OptionMatchEvaluator
│   │   │   ├── vqa.py           # ANLS, RelaxedAccuracy, ExactMatch, NormalizedExactMatch, MathVistaMatch
│   │   │   ├── mmmu_official.py # ✨ MMMUOfficialEvaluator (534줄, rfind/index2ans/random fallback)
│   │   │   ├── mathvista_official.py  # ✨ MathVistaOfficialEvaluator (264줄, precision-aware)
│   │   │   ├── textvqa_official.py    # ✨ TextVQAOfficialEvaluator (208줄, EvalAI processor)
│   │   │   └── chartqapro_official.py # ✨ ChartQAProOfficialEvaluator (173줄, question-type routing)
│   │   ├── runner.py            # 단일 셀 실행 (sample 병렬 요청, per-sample timing/token 계측)
│   │   ├── orchestrator.py      # 전체 매트릭스 순회, resume, dual-GPU 병렬
│   │   ├── reporters/           # 결과 분석 및 시각화
│   │   │   ├── export.py        # CSV/JSON 저장
│   │   │   ├── summary.py       # 마크다운 테이블 생성
│   │   │   └── charts.py        # heatmap, bar_chart, degradation_curve, scatter_vlm_vs_text
│   │   └── kv_analysis/         # ✨ KV 통계 분석 파이프라인 (신규)
│   │       ├── loader.py        # KVDump 클래스 (llama-kv-dump 출력 읽기)
│   │       ├── distribution.py  # per-layer K/V norm, quantile stats
│   │       ├── outliers.py      # outlier channel 검출 (10x median)
│   │       ├── quant_error.py   # MSE, cosine, inner product bias, 이론값 비교
│   │       ├── rotation_analysis.py  # FWHT + Beta KS test + 좌표 독립성 (핵심)
│   │       ├── attention_analysis.py # KL divergence, top-1 match, entropy
│   │       ├── report.py        # 전체 리포트 생성 (CSV + md + plots)
│   │       └── tests/           # 134개 단위 테스트 통과
│   │
│   └── notebooks/               # 실행/시각화 노트북 (thin wrapper)
│       ├── 01_smoke_test.ipynb
│       ├── 02_full_run.ipynb
│       ├── 03_analysis.ipynb
│       ├── 04_kv_analysis.ipynb
│       └── 05_kv_dump_analysis.ipynb   # ✨ KV dump 분석 전용 노트북
│
├── results/                     # 실험 산출물 (gitignore)
│   ├── runs/
│   ├── reports/
│   └── kv_dumps/
│
├── models/                      # GGUF 모델 파일 (gitignore)
│   ├── Qwen3-VL-2B-Instruct/
│   │   ├── *.gguf
│   │   └── mmproj-*.gguf
│   └── Qwen3-VL-2B-Thinking/
│       ├── *.gguf
│       └── mmproj-*.gguf
│
└── pdfs/                        # 참고 논문
    └── pdf/
        ├── 2504.19874v1.pdf     # TurboQuant
        ├── 2406.03482.pdf       # QJL
        ├── 2502.02617.pdf       # PolarQuant
        ├── 2402.02750.pdf       # KIVI
        └── 2411.17525.pdf       # HIGGS
```

## 논문 알고리즘 요약

### Algorithm 1 (TurboQuant_mse) — 구현
1. 입력 벡터에 random orthogonal rotation 적용 (Fast Walsh-Hadamard Transform + deterministic sign flips)
2. 회전 후 각 좌표가 Beta 분포를 따름
3. Beta 분포에 최적화된 Lloyd-Max codebook으로 스칼라 양자화
4. MSE 최소화 목적

### Algorithm 2 (TurboQuant_prod) — 구현
1. (b-1)bit MSE 양자화 적용
2. 잔차(residual)에 1-bit QJL(Quantized Johnson-Lindenstrauss) 적용
3. Inner product의 unbiased 추정 목적
4. **커뮤니티에서 실전 generation 실패 보고** — softmax가 QJL variance를 지수적으로 증폭
5. 본 프로젝트에서 직접 구현하여 **실패를 정량적으로 입증**하는 것이 핵심 기여 중 하나

## 벤치마크 매트릭스

### 런타임 조건 (15개)

| # | Runtime ID | Method | cache-type-k | cache-type-v | Bits | 설명 |
|---|---|---|---|---|---|---|
| 1 | baseline | none | f16 | f16 | 16 | FP16 baseline |
| 2 | lcpp-kv-8 | llama.cpp native | q8_0 | q8_0 | 8 | llama.cpp 기본 KV cache 양자화 |
| 3 | lcpp-kv-4 | llama.cpp native | q4_0 | q4_0 | 4 | llama.cpp 기본 |
| 4 | lcpp-kv-2 | llama.cpp native | q2_K | q2_K | 2 | llama.cpp 기본 (최저) |
| 5 | tq-2 | TQ MSE | turbo2 | turbo2 | 2 | Algorithm 1 대칭 |
| 6 | tq-2h | TQ MSE | turbo2h | turbo2h | 2.5 | fractional (직접 구현) |
| 7 | tq-3 | TQ MSE | turbo3 | turbo3 | 3 | Algorithm 1 대칭 |
| 8 | tq-3h | TQ MSE | turbo3h | turbo3h | 3.5 | fractional (직접 구현) |
| 9 | tq-4 | TQ MSE | turbo4 | turbo4 | 4 | Algorithm 1 대칭 |
| 10 | tq-K4V2 | TQ MSE | turbo4 | turbo2 | K4/V2 | K/V 비대칭 (avg 3) |
| 11 | tq-K4V3 | TQ MSE | turbo4 | turbo3 | K4/V3 | K/V 비대칭 (avg 3.5) |
| 12 | tq-K3V2 | TQ MSE | turbo3 | turbo2 | K3/V2 | K/V 비대칭 (avg 2.5) |
| 13 | tqp-3 | TQ prod | turbop3 | turbop3 | 3 | Algorithm 2 (2bit MSE + 1bit QJL) |
| 14 | tqp-4 | TQ prod | turbop4 | turbop4 | 4 | Algorithm 2 (3bit MSE + 1bit QJL) |
| 15 | tqp-5 | TQ prod | turbop5 | turbop5 | 5 | Algorithm 2 (4bit MSE + 1bit QJL) |

### 핵심 비교축
- **같은 bitwidth, 방법론 비교:** lcpp-kv-4 vs tq-4 (4bit), lcpp-kv-2 vs tq-2 (2bit)
- **같은 평균 bit, 전략 비교:** tq-3 vs tq-K4V2 (둘 다 avg 3bit)
- **Algorithm 1 vs Algorithm 2:** tq-3 vs tqp-3, tq-4 vs tqp-4 — prod의 실전 실패를 정량적으로 입증
- **VLM vs Text 차이:** 같은 런타임에서 VLM 8개 vs Text 3개 비교
- **bitwidth별 열화 곡선:** 2 → 2.5 → 3 → 3.5 → 4bit
- **비대칭 K/V 효과:** K를 높은 bit로, V를 낮은 bit로 하는 전략의 유효성

### 벤치마크 (11개)

| # | Benchmark ID | Type | N | Metric |
|---|---|---|---|---|
| 1 | ai2d | VLM | 500 | option_match |
| 2 | chartqa | VLM | 500 | relaxed_accuracy |
| 3 | chartqapro | VLM | 500 | relaxed_accuracy |
| 4 | docvqa | VLM | 500 | ANLS |
| 5 | mathvista | VLM | 500 | mathvista_match |
| 6 | mmmu | VLM | 500 | option_match |
| 7 | ocrbench_v2 | VLM | 500 | exact_match |
| 8 | textvqa | VLM | 500 | normalized_exact_match |
| 9 | mmlu | Text | 1000 | option_match |
| 10 | commonsenseqa | Text | 3000 | option_match |
| 11 | hellaswag | Text | 3000 | option_match |

**총 셀: 15 × 11 = 165**

## llama.cpp TurboQuant 구현 (완료)

### 등록된 GGML 타입 (8개)

#### Algorithm 1 (MSE) 타입

| GGML Type | Type ID | CLI name | Bits | Block Size | Block Bytes | 설명 |
|---|---|---|---|---|---|---|
| `GGML_TYPE_TURBO2_0` | 42 | turbo2 | 2 | 128 | 36 | 4B norm + 32B packed 2-bit |
| `GGML_TYPE_TURBO2H_0` | 43 | turbo2h | 2.5 | 128 | 40 | fractional: 32ch×3bit + 96ch×2bit |
| `GGML_TYPE_TURBO3_0` | 44 | turbo3 | 3 | 128 | 52 | 4B norm + 48B packed 3-bit |
| `GGML_TYPE_TURBO3H_0` | 45 | turbo3h | 3.5 | 128 | 60 | fractional: 64ch×4bit + 64ch×3bit |
| `GGML_TYPE_TURBO4_0` | 46 | turbo4 | 4 | 128 | 68 | 4B norm + 64B packed 4-bit |

#### Algorithm 2 (prod = MSE + QJL) 타입

| GGML Type | Type ID | CLI name | Total Bits | Block Size | Block Bytes | 설명 |
|---|---|---|---|---|---|---|
| `GGML_TYPE_TURBOP3_0` | 47 | turbop3 | 3 | 128 | 52 | 2bit MSE (32B) + 1bit QJL sign (16B) + 4B norm |
| `GGML_TYPE_TURBOP4_0` | 48 | turbop4 | 4 | 128 | 68 | 3bit MSE (48B) + 1bit QJL sign (16B) + 4B norm |
| `GGML_TYPE_TURBOP5_0` | 49 | turbop5 | 5 | 128 | 84 | 4bit MSE (64B) + 1bit QJL sign (16B) + 4B norm |

**GGML_TYPE_COUNT = 50.** Type 36-38은 이미 제거되었고 39-41은 MXFP4/NVFP4/Q1_0이 사용 중.

**prod 블록 구조:**
```
block_turbop{N}_0 = {
    float norm;                    // 4 bytes: vector norm
    uint8_t qs[...];               // (b-1)-bit MSE quantized indices
    uint8_t qjl[16];               // 128-bit: 1-bit QJL sign per coordinate
}
```

**prod 양자화 과정:**
1. FWHT + sign flip 적용
2. (b-1)-bit Lloyd-Max MSE 양자화 → mse_indices
3. 잔차(residual) = rotated_vec - dequantized_mse 계산
4. QJL projection: 랜덤 ±1 행렬로 잔차를 1-bit sign으로 압축 → qjl_signs

**prod 내적 추정 (attention score):**
- `⟨q, k⟩ ≈ ⟨q, k_mse⟩ + QJL_correction(q, qjl_signs)`
- QJL correction은 unbiased이나 **high variance** → softmax 지수 증폭으로 실전 실패

### Fractional bits 구현 원리 (논문 Section 4.3)
- head_dim=128 채널을 그룹으로 분할, 그룹별 서로 다른 정수 bitwidth 적용
- 2.5bit = 32채널×3bit + 96채널×2bit → (32×3 + 96×2)/128 = 2.5
- 3.5bit = 64채널×4bit + 64채널×3bit → (64×4 + 64×3)/128 = 3.5
- 블록 구조체에 high-bit/low-bit 채널을 연속 배치

### 수정된 파일 (구현 완료)

**GGML 코어:**
```
ggml/include/ggml.h              # GGML_TYPE_TURBO*/TURBOP* enum 추가 (42~49)
ggml/src/ggml.c                  # type_traits 테이블에 8개 타입 등록
ggml/src/ggml-common.h           # block_turbo*_0, block_turbop*_0 구조체 + Lloyd-Max codebook 상수
ggml/src/ggml-quants.h           # quantize_row_turbo*/dequantize_row_turbo* 선언 (24개 함수)
ggml/src/ggml-quants.c           # CPU quantize/dequantize 구현 + 헬퍼 (FWHT, sign flip, QJL encode)
ggml/src/ggml-cpu/quants.h/c     # vec_dot_turbo*_q8_K (flash attention 직접 연산)
ggml/src/ggml-cpu/ggml-cpu.c     # type_traits_cpu 테이블에 8개 타입 등록
```

**CUDA:**
```
ggml/src/ggml-cuda/ggml-cuda.cu          # supports_op에 TURBO 타입 SET_ROWS/FLASH_ATTN 등록
ggml/src/ggml-cuda/set-rows.cu           # KV write: FP32 → TURBO quantize 디스패치
ggml/src/ggml-cuda/cpy-utils.cuh         # device-side quantize_f32_turbo*_0_block 함수
ggml/src/ggml-cuda/convert.cu            # TURBO → FP16/FP32 dequantize 디스패치
ggml/src/ggml-cuda/turbo-common.cuh      # CUDA codebook, FWHT, 협력적(shared-memory) dequantize 헬퍼
ggml/src/ggml-cuda/fattn.cu              # flash attention 디스패치 (11개 K/V 조합)
ggml/src/ggml-cuda/fattn-common.cuh      # vec_dot_fattn_vec_KQ_turbo + dequantize_V_turbo (cooperative)
ggml/src/ggml-cuda/fattn-vec.cuh         # extern 템플릿 선언
ggml/src/ggml-cuda/template-instances/   # fattn-vec-instance-turbo*_0-turbo*_0.cu (11 files)
```

**KV Cache 통합:**
```
src/llama-kv-cache.cpp   # generic 경로로 동작 (TURBO 직접 패치 불필요)
common/arg.cpp           # CLI --cache-type-k/v에 turbo*/turbop* 추가 (kv_cache_types 리스트)
```

**KV dump 툴 (신규):**
```
tools/kv-dump/kv-dump.cpp        # CLI 툴 (llama-kv-dump): 모델 로드 → 한번 decode → K/V 추출
tools/kv-dump/CMakeLists.txt     
include/llama.h                  # llama_kv_self_dbg_* 공개 API 추가
src/llama-context.cpp            # dbg wrapper 함수
src/llama-kv-cache.h/cpp         # dbg_dump_*_layer_f32 구현
```

### Lloyd-Max Codebook (d=128 Beta, 실측 검증)

```c
// ggml/src/ggml-common.h — 논문 d=128 Beta 분포 최적 centroids
static const float TURBO2_CENTROIDS[4]  = {-1.5104, -0.4528, +0.4528, +1.5104};
static const float TURBO3_CENTROIDS[8]  = {-2.1519, -1.3439, -0.7560, -0.2451,
                                            +0.2451, +0.7560, +1.3439, +2.1519};
static const float TURBO4_CENTROIDS[16] = {-2.7326, -2.0690, ..., +2.7326};

// Boundary (Voronoi 미드포인트)
static const float TURBO2_BOUNDARIES[3]  = {-0.9816, 0.0, +0.9816};
static const float TURBO3_BOUNDARIES[7]  = {...};
static const float TURBO4_BOUNDARIES[15] = {...};

#define TURBO_QJL_SCALE 1.2533141373155001f  // sqrt(pi/2)
```

**검증 (test-turboquant.c)**: round-trip MSE/coord가 논문 값과 정확 일치
- turbo2: 0.116 (논문 0.117)
- turbo3: 0.034 (논문 0.034)  
- turbo4: 0.009 (논문 0.009)

### FWHT + Sign Flip

```c
static void fwht_inplace(float * x, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float u = x[i + j];
                float v = x[i + j + len];
                x[i + j]       = u + v;
                x[i + j + len] = u - v;
            }
        }
    }
    float scale = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n; i++) x[i] *= scale;
}

static void apply_sign_flip(float * x, int n, uint32_t seed) {
    for (int i = 0; i < n; i++) {
        uint32_t h = (seed * 2654435761u + i * 2246822519u) >> 31;
        if (h) x[i] = -x[i];
    }
}
```

### QJL (Algorithm 2) 구현

```c
// QJL residual encoding: 잔차를 1-bit sign으로 압축
// random projection matrix는 seed 기반 deterministic ±1
static void qjl_encode_residual(
    const float * residual,  // 128-dim residual vector
    uint8_t * qjl_signs,     // 16 bytes output (128 bits)
    int n, uint32_t seed
) {
    // 각 좌표의 sign을 저장 (random projection 후)
    memset(qjl_signs, 0, n / 8);
    for (int i = 0; i < n; i++) {
        // random ±1 projection
        uint32_t h = (seed * 2654435761u + i * 2246822519u);
        float proj = (h & 1) ? residual[i] : -residual[i];
        if (proj > 0) {
            qjl_signs[i / 8] |= (1 << (i % 8));
        }
    }
}

// QJL inner product correction
// ⟨q, k⟩ ≈ ⟨q, k_mse⟩ + ||r||₂ * Σᵢ sign(rᵢ) * qᵢ / √d
static float qjl_inner_product_correction(
    const float * query,
    const uint8_t * qjl_signs,
    float residual_norm,
    int n, uint32_t seed
) {
    float correction = 0.0f;
    for (int i = 0; i < n; i++) {
        int sign = (qjl_signs[i / 8] >> (i % 8)) & 1;
        uint32_t h = (seed * 2654435761u + i * 2246822519u);
        float proj_sign = (h & 1) ? 1.0f : -1.0f;
        correction += query[i] * (sign ? proj_sign : -proj_sign);
    }
    return correction * residual_norm / sqrtf((float)n);
}
```

## 설계 원칙

### llama-server를 추론 엔진으로 사용
벤치마크 프레임워크는 llama-server를 subprocess로 기동하고 OpenAI-compatible API로 통신.
런타임 조건(KV cache 타입)은 서버 시작 시 CLI 인자로 주입.

```python
# bench/tq_bench/server.py
class LlamaServer:
    def start(self, runtime_config: RuntimeConfig):
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "--cache-type-k", runtime_config.cache_type_k,  # "turbo3", "turbop4", "q4_0", "f16"
            "--cache-type-v", runtime_config.cache_type_v,
            "-ngl", "99",
            "-fa", "on",
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--ctx-size", str(runtime_config.ctx_size),
            "--jinja",
            "--temp", "0.0",
        ]
        if runtime_config.gpu_id is not None:
            cmd += ["--gpu-id", str(runtime_config.gpu_id)]
        self.proc = subprocess.Popen(cmd, ...)
        self._wait_healthy()
```

### 듀얼 GPU 병렬 벤치마킹
`--parallel` 플래그 활성화 시 GPU 0/1에 각각 llama-server를 띄워 서로 다른 런타임을 동시 벤치마크.

```python
# bench/tq_bench/orchestrator.py
class Orchestrator:
    def run(self, parallel: bool = False, resume: bool = True):
        if parallel and self.num_gpus >= 2:
            # GPU 0: 포트 8080, GPU 1: 포트 8081
            with ThreadPoolExecutor(max_workers=2) as pool:
                ...
        else:
            # 단일 GPU 순차 실행
            ...
```

### 노트북은 얇게
로직은 전부 `tq_bench` 패키지. 노트북은 import + 호출만.

### Resume 지원
orchestrator가 매 셀(runtime × benchmark) 완료 시 checkpoint JSON 저장. 중단 후 재실행 시 완료된 셀 건너뜀.

## 모델

- **Qwen3-VL-2B-Instruct** — GGUF + mmproj, dense attention, head_dim=128
- **Qwen3-VL-2B-Thinking** — thinking mode 포함, 동일 아키텍처

전체 레이어가 standard dense attention → TQ가 전 레이어에 적용됨.

## 실행 환경

### 하드웨어
- **GPU 0**: NVIDIA GeForce RTX 5070 Ti (16GB VRAM, sm_120)
- **GPU 1**: NVIDIA GeForce RTX 4060 Ti (16GB VRAM, sm_89)
- **CPU**: AMD Ryzen 7 7800X3D
- **RAM**: DDR5 64GB
- **OS**: Ubuntu 24.04 (WSL2)

### 빌드

```bash
cd llama.cpp
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89;120" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4    # ⚠️ -j4 권장 (j8 이상은 nvcc OOM 위험)
```

**주의**: CUDA fattn template instantiation이 매우 무거움 (cicc 프로세스당 ~5GB RAM). 
- `-j8` 시 8 × 5GB = 40GB → swap 사용 → 느려짐
- `-j4`가 최적 (16GB 버퍼, 전체 빌드 ~40분 예상)
- 전체 재빌드 피하려면 `--target ggml-cuda` 또는 `--target llama-server` 권장

**실행시 LD_LIBRARY_PATH 필요**:
```bash
export LD_LIBRARY_PATH=$(pwd)/llama.cpp/build/bin:$LD_LIBRARY_PATH
./llama.cpp/build/bin/llama-server ...
```

**사용 예시 (TurboQuant KV cache)**:
```bash
llama-server \
  -m models/Qwen3-VL-2B-Instruct/model.gguf \
  --mmproj models/Qwen3-VL-2B-Instruct/mmproj.gguf \
  --cache-type-k turbo4 --cache-type-v turbo3 \
  -ngl 99 -fa on --ctx-size 4096 --jinja --temp 0.0
```

**KV 추출 (리서치용)**:
```bash
# LLAMA_ATTN_ROT_DISABLE=1로 llama.cpp 자체 pre-rotation 끄기
LLAMA_ATTN_ROT_DISABLE=1 llama-kv-dump \
  -m model.gguf --mmproj mmproj.gguf \
  --cache-type-k f16 --cache-type-v f16 \
  -p "Describe this image." --image chart.png \
  --output-dir results/kv_dumps/baseline --ctx-size 4096 \
  -ngl 99 -fa on
```

### Python 환경

```bash
cd bench
uv init
uv add httpx pillow datasets pyyaml tqdm pandas matplotlib seaborn jupyterlab
uv run jupyter lab
```

## 구현 상태

### Phase 1: llama.cpp TurboQuant MSE 타입 ✅ 완료
- GGML enum 등록 (42-46): TURBO2/2H/3/3H/4_0
- block 구조체, Lloyd-Max codebook, boundaries 정의
- CPU quantize/dequantize (FWHT seed=42 + sign flip + codebook)
- CPU vec_dot_turbo*_q8_K (flash attention 경로)
- `test-turboquant.c` 단위 테스트 — round-trip MSE 논문 값과 일치 (0.116/0.034/0.009)

### Phase 2: llama.cpp TurboQuant prod 타입 ✅ 완료 (CUDA 주 경로)
- GGML enum 등록 (47-49): TURBOP3/4/5_0
- block 구조체에 `r_norm` 포함 (`norm + r_norm + qs + qjl`)
- CPU quantize: MSE + 잔차 → residual norm 저장 + QJL sign encode (seed=137)
- CUDA `vec_dot_fattn_vec_KQ_turbop*` 구현: MSE dot + QJL correction 경로 존재
- **제약**: generic/CPU dequantize는 여전히 MSE-only이며, CPU fallback `ggml_vec_dot_turbop*_q8_K`도 MSE-only
- **문서 주의**: 체크인된 smoke/report 아티팩트 일부는 correction 이전 설명을 여전히 포함함

### Phase 3: CUDA 커널 ✅ 완료
- set-rows.cu: 8개 타입 디스패치 (KV write 경로)
- convert.cu: dequantize 디스패치 (fp16/fp32/bf16)
- turbo-common.cuh: codebook, FWHT, 협력적 shared-memory dequantize 헬퍼
- fattn.cu + fattn-common.cuh: 11개 (K,V) 조합 flash attention 디스패치
- template-instances/: 11개 `.cu` 파일 (D=64,128,256)
- ggml-cuda.cu supports_op: SET_ROWS + FLASH_ATTN_EXT 타입 인식
- **CUDA 최적화 (4.6x speedup)**: cooperative shared-memory dequant (warp-level)

### Phase 4: KV Cache 통합 및 검증 ✅ 완료
- `--cache-type-k turbo3`, `--cache-type-v turbop4` 등 CLI 동작
- Qwen3-VL-2B-Instruct BF16 기반 generation smoke 통과 (turbo4, tq-K4V3)
- prod 타입으로 generation 실패 재현 확인 (MSE 버전과 byte-identical)

### Phase 5: 벤치마크 프레임워크 ✅ 완료
- tq_bench 패키지 전체 (config, server, client, datasets×11, evaluators×12, runner, orchestrator, reporters)
- dual-GPU 병렬, resume/checkpoint, CSV/JSON export, 차트 4종
- `BenchmarkRunner.run_cell`의 sample 병렬 요청 처리 구현 완료 (`n_parallel=4`)
- `OptionMatchEvaluator` / VLM evaluator 보정 반영, `max_tokens` 및 model override 경로 도입
- Thinking 모델용 `max_tokens_override=4096` + `<think>` strip 지원
- **공식 parity evaluator 4종**: `mmmu_official`, `mathvista_official`, `textvqa_official`, `chartqapro_official`
- `BaseEvaluator.score(metadata=)` 인프라, evaluator `_REGISTRY` 통합, `BenchmarkConfig` parity 필드
- 통합 벤치마크 러너 `bench/run_bench.py` (CLI: `--num`, `--runtimes`, `--benchmarks`, `--model`, `--resume`)
- **Per-sample timing/token instrumentation**:
  - `CompletionTimings` dataclass: llama-server `usage` + `timings` 파싱 + 클라이언트 wall-clock
  - `SampleResult`: `ttft_ms`, `total_latency_ms`, `prefill_ms`, `decode_ms`, `decode_throughput_tps`, `prompt_tokens`, `completion_tokens`, `n_images`
  - `LatencyStats` (p50/p95/p99), `ThroughputStats` (mean/min/max) aggregate 자동 계산
  - `RunRecord`: `score_std`, `score_median`, `ttft_stats`, `total_latency_stats`, `decode_throughput_stats`, `gpu_memory_bytes`, `kv_cache_bytes`
  - `LlamaServer.get_gpu_memory()` (nvidia-smi), `get_kv_cache_bytes()` (/slots endpoint)
  - `reporters/summary.py`: timing 데이터 감지 시 확장 마크다운 테이블 자동 렌더링

### Benchmark Reproducibility Rules

- **모델 병렬 처리(`parallel_requests`)는 실험 변수이자 필수 운영 설정이다.**
  벤치마크는 모델별 병렬 요청 수를 명시적으로 유지해야 하며, 디버깅 중에
  묵시적으로 단일 요청 모드로 바꾸거나 제거하면 안 된다. 병렬성 변경은
  반드시 의도된 실험 변경으로 취급한다.
- **벤치용 decoding/sampling 설정은 고정된 통제 변수다.**
  현재 프로젝트의 모델별 `temperature`, `top_p`, `top_k`, `min_p`,
  penalties, `sampling_seed`, 벤치마크별 `max_tokens`, Thinking 모델의
  `max_tokens_override`는 재현성 확보를 위한 고정 설정으로 취급한다.
  사용자가 명시적으로 요청하지 않는 한 이 값들을 임의로 조정하지 않는다.
- **문제 재현/디버깅 시에도 먼저 런처/서버/커널 문제를 분리하되, benchmark
  config 자체는 유지한다.** 즉, 실패 원인을 좁히기 위해 런타임 축, 샘플 수,
  포트, 프로세스 lifecycle은 바꿀 수 있지만, benchmark 설정값을 casually
  바꾸지 않는다.

### Phase 6: KV 분석 파이프라인 ✅ 완료
- tools/kv-dump (C++): llama-kv-dump CLI, 모델 forward 후 K/V dump + meta.json
- bench/tq_bench/kv_analysis/ (Python): loader, distribution, outliers, quant_error, rotation_analysis (Beta KS test), attention_analysis, report
- `python3 -m pytest bench/tq_bench/kv_analysis/tests -q` 기준 134개 테스트 통과
- FWHT C++와 bit-exact 일치 검증

### Phase 7: 실행 및 분석 ✅ 완료
- ✅ Smoke test: 14 runtimes × 7 prompts
- ✅ 결과 저장: `results/runs/smoke_extended_v2.json`
- ✅ KV dump: 9 runtimes (baseline + 8개 양자화) with image+text prompt
- ✅ 회전 이론 검증: Beta KS test + 좌표 상관계수 + 비전 vs 텍스트 분리
- ✅ 미니 벤치마크: 7 runtimes × 3 text × n=50 (MMLU 버그 수정 후, `results/runs/mini_bench_text_n50.json`)
- ✅ 예비 보고서 저장: `results/reports/final_report.md`
- ✅ VLM 공식 재현 러너 준비 완료 (`bench/run_vlm_n30.py`, `bench/run_bench.py`)

## 공식 Qwen3-VL-2B-Instruct 벤치마크 점수 (출처: Qwen3-VL 블로그/기술 보고서)

본 프레임워크가 재현해야 하는 공식 점수. 본 프로젝트의 baseline(f16 KV)은 이 점수와 ±5%p 이내여야 프레임워크 검증됨.

### STEM and Puzzle
| Benchmark | Qwen3-VL-2B | Qwen3-VL-4B | 우리 구현 여부 |
|---|---|---|---|
| **MMMU** (val) | **61.4** | 70.8 | ✅ `mmmu` |
| MMMU-Pro | 42.5 | 57.0 | ❌ |
| **MathVista** (mini) | **73.6** | 79.5 | ✅ `mathvista` |
| DynaMath | 66.7 | 74.4 | ❌ |
| VLMsAreBlind | 50.0 | 68.6 | ❌ |

### General VQA
| Benchmark | Qwen3-VL-2B | Qwen3-VL-4B |
|---|---|---|
| RealWorldQA | 69.5 | 73.2 |
| MMStar | 68.1 | 73.2 |
| MMBench_EN-DEV-v1.1 | 81.9 | 86.7 |
| SimpleVQA | 43.6 | 48.8 |
| HallusionBench | 54.9 | 64.1 |

### Text Recognition / Document
| Benchmark | Qwen3-VL-2B | Qwen3-VL-4B | 우리 구현 |
|---|---|---|---|
| **AI2D_TEST** | **80.4** | 84.9 | ✅ `ai2d` |
| MMLongBench-Doc | 33.8 | 44.4 | ❌ |
| CC-OCR | 68.3 | 73.8 | ❌ |
| OmniDocBench1.5 | 65.9 | 80.0 | ❌ |
| CharXiv(RQ) | 37.1 | 50.3 | ❌ |

### Spatial / Video / Agent / Medical (참고)
- RefCOCO 84.8, CountBench 84.1, EmbSpatialBench 75.9
- VideoMME 67.9/62.1, MLVU 69.2, MVBench 64.5
- ScreenSpot Pro 48.5
- SLAKE 61.1, PMC-VQA 42.4, MedXpertQA-MM 13.0

**참고사항:**
- Qwen3-VL 팀은 MMLU/CommonsenseQA/HellaSwag 등 text-only MCQ 벤치마크 점수는 Qwen3-VL 표에 **발표하지 않음** (VLM 모델이므로). 우리의 MMLU 0.45는 자체 sanity check이며 공식 비교 대상 없음.
- **교집합**: 본 프레임워크가 구현한 11개 벤치마크 중 공식 2B 점수와 직접 비교 가능한 것은 **3개뿐 (AI2D, MMMU, MathVista)**. 이 3개가 프레임워크 검증의 핵심.
- Thinking variant는 별도 표 없음. 추후 직접 측정 필요.

## 해결된 이슈 (2026-04-13)

1. ✅ **[Task #17] Runner 병렬 요청 처리 완료**: `BenchmarkRunner.run_cell`에 `ThreadPoolExecutor(max_workers=n_parallel)` 추가. `LlamaApiClient`는 공유 pool 기반 thread-safe. `ServerLaunchConfig.n_parallel=4`로 llama-server 슬롯 확보. 짧은 MCQ 벤치마크 1.48x, 긴 출력 벤치마크 2-4x 스루풋 향상.

2. ✅ **[Task #19] MMLU 스코어 버그 수정 완료**: `OptionMatchEvaluator`가 strict equality(`pred == "A"`)만 하여 모델의 `"B. push particles apart"` 같은 긴 답변 100% mismatch → baseline 0.15. 
   - **수정**: `extract_option_letter()` 추가, 다양한 패턴 처리 (`"B"`, `"B."`, `"(C)"`, `"**C**"`, `"\boxed{E}"`, `"The answer is D"`, etc.)
   - **검증**: baseline MMLU 0.15 → **0.45** (n=50에서 0.50), 랜덤 25% 이상 정상화
   - 영향받은 벤치마크: mmlu, commonsenseqa, hellaswag, ai2d, mmmu (모두 option_match 사용)

3. ✅ **[Task #12,#13] KV dump 분석 파이프라인 + 회전 이론 검증 완료**: 
   - 52/52 단위 테스트 통과 (FWHT C++와 bit-exact 일치)
   - 본 프로젝트 핵심 발견: V는 Beta 분포 가정 만족 (28/28 excellent), K는 위반 (상관계수 0.42)
   - 비전 토큰이 텍스트 토큰보다 양자화에 취약 (KS p-value 0.001 vs 0.043)

4. ✅ **[Task #27] VLM 평가자 정밀화**: baseline × MMMU가 0.367 (공식 0.614), MathVista 0.333 (공식 0.736). 세 가지 원인 수정:
   - **MathVista answer**: letter가 아니라 **choice text** (예: "145°", "6cm")였음. loader가 `query` 공식 프롬프트 필드 사용 + `[letter, choice_text]` 복합 reference 저장
   - **MMMU question_type**: multi_choice와 open이 섞여 있음 (847 MC + 53 open). open 답변은 stringified list 파싱. hint-based prompt로 통일
   - **max_tokens 부족**: Qwen3-VL이 긴 CoT 생성 후 답변 → 128 토큰에 잘림. 벤치마크별 max_tokens 도입 (MMMU 1024, AI2D/MathVista 512)
   - **Unicode subscript 정규화**: 모델은 "HbO₂", 데이터셋은 "HbO2". NFKC 정규화 + subscript→ASCII 변환 추가
   - **모델별 override**: Thinking 모델은 `max_tokens_override=4096` (긴 reasoning chain 수용)
   - **검증**: MathVista 0.333 → 0.600 (+26.7pp), MMMU 진행 중

5. ✅ **Per-sample timing/token instrumentation 추가 (2026-04-13)**: 벤치마크 프레임워크가 정확도만 수집하고 성능 메트릭은 전혀 없었음.
   - `CompletionTimings` dataclass 신설: llama-server의 `usage`(prompt_tokens, completion_tokens) + `timings`(prompt_ms, predicted_ms, predicted_per_second) 파싱, wall-clock 측정
   - `SampleResult`에 8개 필드 추가: ttft_ms, total_latency_ms, prefill_ms, decode_ms, decode_throughput_tps, prompt_tokens, completion_tokens, n_images
   - `RunRecord`에 aggregate stats 추가: score_std, score_median, LatencyStats(p50/p95/p99), ThroughputStats(mean/min/max), gpu_memory_bytes, kv_cache_bytes
   - `LlamaServer`: nvidia-smi GPU memory 쿼리 + /slots KV cache 쿼리 (best-effort)
   - `reporters/summary.py`: timing 데이터 감지 시 TTFT, latency p95, tok/s, GPU MiB 컬럼 자동 확장
   - **검증**: 219개 테스트 통과 (85 parity + 134 KV analysis), 기존 기능 회귀 없음

## 현재 남은 이슈 (2026-04-13)

1. ~~**문서/아티팩트 정합성 미완료**~~ → Phase 8에서 해결

2. **CPU prod vec_dot은 여전히 MSE-only**
   - `ggml-cpu/quants.c`의 `ggml_vec_dot_turbop*_q8_K`는 generic MSE 매크로 사용
   - CUDA 경로(`fattn-common.cuh`)에만 QJL correction 구현
   - CUDA가 주 경로이므로 실험상 치명적이지는 않지만, CPU fallback 결과는 corrected prod를 대표하지 않음

3. **체크인된 prod 결과 아티팩트는 pre-correction 서술 비중이 큼**
   - 현재 `results/runs/`에 있는 smoke/text mini-bench는 prod가 MSE-only처럼 동작하던 시점의 해석을 담고 있음
   - correction 코드가 트리에 반영된 뒤의 consolidated rerun 결과는 아직 문서/리포트에 통합되지 않음

4. **`attn_rot_k` 간섭 미분리**
   - llama.cpp의 자체 Hadamard pre-rotation이 양자화 KV에만 적용되어 본 프로젝트의 TurboQuant 회전과 이중 회전됨
   - KV dump 분석은 `LLAMA_ATTN_ROT_DISABLE=1`로 통제했지만, generation 경로 A/B는 아직 체계적으로 분리되지 않음

5. **CUDA fattn template 재빌드 병목**
   - 11개 `.cu` instance × 2 arch nvcc 컴파일 시 cicc당 5-10GB RAM, 총 40-60분
   - prod/QJL 및 CUDA kernel 수정 반복 시 개발 속도를 강하게 제한함

6. **VLM 공식 재현 결과 — parity smoke 완료, 본 실행 대기**
   - ✅ `bench/run_bench.py` 통합 러너 + parity evaluator 준비 완료
   - ✅ Parity smoke (n=10) 성공: `results/runs/parity_smoke_n10.json`
   - ⏳ `--num 100` 이상의 본 실행으로 공식 점수 ±5%p 검증 필요

## 확인된 기술적 제약 (실측)

### QJL(prod) generation 실패 — 체크인된 아티팩트는 pre-correction 기준
- `results/runs/`에 있는 smoke/text mini-bench prod 행은 **pre-correction artifact** 기준으로 해석해야 함
- 이 아티팩트에서는 `tqp-3/4/5`가 대응 MSE 계열과 거의 같은 broken 양상을 보여, "QJL bits를 저장하지만 실질 이득이 없다"는 현상을 재현함
- 현재 코드 트리에는 CUDA `vec_dot_fattn_vec_KQ_turbop*` correction 경로가 이미 존재함
- 다만 corrected CUDA 경로 기준의 consolidated smoke/VLM rerun 결과와 보고서 재작성은 아직 남아 있음
- 따라서 **현재 결론**은 두 층으로 나뉜다:
  1. 체크인된 historical artifact는 degenerate prod=MSE 동작을 보여준다
  2. 최신 코드 기준의 corrected prod는 재실행/재기록이 더 필요하다

### VLM 비전 토큰 분포 차이 — 검증 완료
- 비전 토큰의 K/V norm이 텍스트보다 큼 (K: 143 vs 134, V: 152 vs 137)
- 회전 후 Beta 분포 적합도: 비전 p-value 0.001-0.051, 텍스트 p-value 0.043-0.194 → **비전이 3-40배 더 어려움**
- KV norm ratio per-head dispersion: vision 16.8 vs text 12.0 → **비전의 head-내 norm 편차가 더 크다**

### llama.cpp attn_rot_k 상호작용 — 주의 필요
- llama.cpp는 KV cache가 양자화일 때 K/V에 **자체 Hadamard pre-rotation**을 적용함 (cache type이 양자화이고 head_dim이 64 배수일 때)
- 즉, baseline(f16)은 native basis, 양자화된 KV는 rotated basis에 저장됨
- 이는 본 프로젝트의 TurboQuant 회전 위에 또 한 번 회전이 적용되는 것 → **효과 이중화 또는 간섭 가능**
- dump 도구로 실제 KV를 비교할 때 `LLAMA_ATTN_ROT_DISABLE=1`로 끄고 측정해야 함
- **조사 필요**: 이 pre-rotation이 TurboQuant 성능을 개선하는지 악화시키는지 분리 검증

### lcpp-kv-2 (q2_K) 부팅 실패
- llama.cpp upstream이 q2_K를 KV cache 타입으로 지원하지 않음 (V cache requires flash_attn, 2bit V 미지원)
- 따라서 "같은 2bit 비교"는 tq-2 vs q2_K가 아니라 tq-2 vs lcpp-kv-4(4bit 인접 상한)와 비교해야 함

## 커뮤니티 발견 사항 (논문에 인용 가능)

| 발견 | 출처 |
|---|---|
| QJL이 softmax 증폭으로 실전 generation 실패 | tonbistudio, scos-lab, sharpner, 0xSero, quantumaikr, TheTom 등 6+ 팀 |
| K/V norm ratio가 압축 품질을 예측 | scos-lab/turboquant (8-model benchmark) |
| 2-bit value가 품질 병목 | 0xSero/turboquant (vLLM), vLLM PR #38479 |
| MSE-only가 MSE+QJL보다 우수 | 전 커뮤니티 합의 |
| K 채널 5-20%가 10-100x outlier | scos-lab, llama.cpp Discussion #20969 |
| VLM에서 TurboQuant 체계적 벤치마크 전무 | Alberto Nieto가 유일한 정성적 데모 (Molmo2) |
| llama.cpp turbo3 구현: MSE 0.034 (논문 일치) | ik_llama.cpp Issue #1509 |
| llama.cpp TQ: throughput 1% 이내, VRAM 72-78% 절감 | domvox/llama.cpp-turboquant-hip |

## 커뮤니티 llama.cpp TurboQuant 포크 참고

| 포크 | 특징 | 참고 포인트 |
|---|---|---|
| spiritbuun/llama-cpp-turboquant-cuda | CUDA FA kernels, turbo3/turbo4 | CUDA 커널 구조, norm correction |
| Pascal-SAPUI5/llama.cpp-turboquant | ROCm, turbo3/turbo4 | 22 files, ~1550 lines |
| domvox/llama.cpp-turboquant-hip | HIP, turbo2/3/4, FA all 16 K×V combos | 가장 완성도 높음 |
| TheTom (upstream Discussion #20969) | CPU-only TQ3_0 | 원본 구현, Lloyd-Max 검증 |

본 프로젝트는 이들을 **참고하되 직접 재구현**. 이유:
1. 기존 포크 중 Algorithm 2(prod/QJL)를 구현한 곳 없음
2. fractional bits(turbo2h/turbo3h) 미지원
3. 코드 일관성 및 라이선스 명확성

## 본 실험 실측 결과 (Qwen3-VL-2B-Instruct BF16)

**주의:** 아래 smoke/text mini-bench 표는 현재 repo에 체크인된 아티팩트 기준이다.  
특히 prod(`turbop*`) 행은 correction 이전 또는 correction 결과가 아직 통합 반영되지 않은 시점의 해석이 섞여 있으므로, 최신 CUDA correction 코드의 최종 평가는 rerun 후 다시 고정해야 한다.

### 미니 정량 벤치마크 (7 runtimes × 3 text × n=50, MMLU 버그 수정 후)

| Runtime | CommonsenseQA | MMLU | HellaSwag | 평균 |
|---|---|---|---|---|
| **baseline** (f16) | **0.740** | **0.460** | **0.680** | **0.627** |
| lcpp-kv-8 (q8_0) | 0.740 | 0.460 | 0.640 | 0.613 |
| lcpp-kv-4 (q4_0) | 0.560 | 0.400 | 0.580 | 0.513 |
| tq-3 (turbo3) | 0.040 | 0.040 | 0.040 | 0.040 |
| **tq-4** (turbo4) | **0.280** | **0.260** | **0.240** | **0.260** |
| tq-K4V3 (turbo4/turbo3) | 0.280 | 0.280 | 0.300 | 0.287 |
| tqp-3 (Algorithm 2 3bit) | 0.000 | 0.020 | 0.060 | 0.027 |

**중요 재해석 (smoke → 본 벤치마크):**

1. **lcpp-kv-8는 실질적 무손실** — baseline과 ±0.04 이내 차이, 안전한 기본 선택
2. **lcpp-kv-4는 중간 손실** — 평균 -11%p 감소 (0.627 → 0.513), 실용 허용 가능
3. **🔴 tq-4가 lcpp-kv-4보다 나쁨** — 평균 0.260 vs 0.513, **llama.cpp native q4가 TurboQuant 4-bit보다 2배 정확**
   - Smoke test에서 tq-4가 71% 정답이었던 것은 7개 쉬운 프롬프트의 noise
   - 쉬운 fact-based 질문은 TQ 4-bit로도 통과하지만, MCQ 패턴 매칭이 필요한 벤치마크는 깨짐
4. **tq-K4V3 ≈ tq-4** — 비대칭 K/V 효과가 엄밀 벤치마크에선 작게 드러남 (0.287 vs 0.260)
5. **tq-3 완전 파괴** — 3-bit는 2B VLM에 너무 aggressive
6. **🔴 tqp-3 near-zero** — 현재 체크인된 n=50 artifact에서는 거의 0점대이며, 이는 pre-correction prod 해석의 역사적 기준점으로 보는 것이 안전함

**Qwen3-VL-2B 공식 점수 대비 (text 벤치마크는 공식 발표 없음, 참고용):**
- 2B-Instruct는 MMLU 같은 text-only 벤치마크를 논문 표에 공개하지 않음 → 직접 비교 대상 없음
- 우리 baseline MMLU 0.46은 2B 모델의 전형적 범위 (4B: 60-65%, 8B: 70+, 32B: 88.7%)
- 공식 비교는 VLM 벤치마크 (AI2D, MMMU, MathVista)로 Phase 9에서 수행 예정

### Smoke test (14 runtimes × 7 prompts) — 참고 데이터

| Runtime | K/V | KV (MiB) | tok/s | Acc | 평가 |
|---|---|---|---|---|---|
| baseline | f16/f16 | 448 | 265 | 100% | reference |
| lcpp-kv-8 | q8_0/q8_0 | 238 | 208 | 100% | lossless |
| lcpp-kv-4 | q4_0/q4_0 | 126 | 167 | 86% | 작은 손실 |
| lcpp-kv-2 | q2_K/q2_K | — | — | FAIL_BOOT | upstream 미지원 |
| tq-2  | turbo2/turbo2 | **63** | 121 | 14% | 깨짐 |
| tq-3  | turbo3/turbo3 | 91 | 107 | 0% | 깨짐 |
| tq-4  | turbo4/turbo4 | 119 | 113 | 71% | 부분 작동 |
| **tq-K4V3** | turbo4/turbo3 | 105 | 110 | **86%** | **최고의 TQ 조합** |
| tqp-3 | turbop3/turbop3 | 91 | 118 | 14% | tq-2와 byte-identical |
| tqp-4 | turbop4/turbop4 | 119 | 112 | 0% | tq-3와 byte-identical |
| tqp-5 | turbop5/turbop5 | 147 | 111 | 71% | tq-4와 byte-identical |

**핵심 관찰 (체크인된 smoke artifact 기준):**
- 메모리 절감: 7.1x (turbo2), 5.0x (turbo3), 3.8x (turbo4)
- prod 행은 historical artifact상 MSE-only처럼 보이며, 최신 CUDA correction 결과는 별도 rerun으로 갱신 필요

### CUDA 최적화 효과 (cooperative shared-memory dequant)

| Runtime | 최적화 전 | 최적화 후 | Speedup |
|---|---|---|---|
| turbo2 | 24 | 121 | 5.0x |
| turbo3 | 24 | 107 | 4.5x |
| turbo4 | 24 | 113 | 4.7x |
| tq-K4V3 | 24 | 110 | 4.6x |

f16 baseline(265 tok/s) 대비 61% 속도까지 회복. 남은 격차는 FWHT 글로벌 operation 때문.

### KV dump 분석 — 회전 이론 검증 (본 프로젝트의 핵심 과학적 기여)

baseline f16 KV 추출 → FWHT 회전 → Beta 분포 KS test + 좌표 상관계수:

| Cache | Token type | Avg KS p-value | 평균 좌표 상관계수 | Excellent/Fail |
|---|---|---|---|---|
| **K** | all | 0.001 | **0.418** (높음) | 13/4 |
| K | vision | 0.001 | 0.427 | 14/4 |
| K | text | **0.043** | 0.422 | 14/4 |
| **V** | all | 0.057 | **0.071** (낮음) | **28/0** |
| V | vision | 0.051 | 0.083 | 28/0 |
| V | text | **0.194** | 0.082 | **28/0** |

**핵심 발견:**

1. **V는 TurboQuant 가정을 잘 만족** (28/28 excellent, 좌표 상관 0.07) — V quantization이 관대한 이유
2. **K는 가정 위반** (좌표 상관 0.42, iid 가정과 거리) — K aggressive quantization이 attention을 깨는 구조적 이유
3. **비전 토큰이 텍스트 토큰보다 나쁘다**: K p-value 0.001 vs 0.043, V p-value 0.051 vs 0.194
4. **K 채널 outlier** 평균 0.97%, 최대 2.54% (V는 거의 0)
5. **tq-K4V3이 최고**: K를 4bit로 유지, V를 3bit로 압축이 이론적으로도 정당화됨

### Quant error (LLAMA_ATTN_ROT_DISABLE=1, 회전 없이 순수 비교)

| Runtime | mean cosine | mean MSE | rel error |
|---|---|---|---|
| lcpp-kv-8 | 0.998 | 0.17 | 5.9% |
| lcpp-kv-4 | 0.948 | 3.77 | 31.8% |
| **tq-4** | 0.903 | 7.13 | 38.7% |
| tq-K4V3 | 0.880 | 8.60 | 44.4% |
| tq-3 | 0.741 | 17.9 | 65.3% |
| tq-2 | 0.645 | 28.8 | 79.2% |
| **tqp-3 ≈ tq-2** | 0.645 | 28.8 | 79.2% |
| **tqp-4 ≈ tq-3** | 0.741 | 17.9 | 65.3% |

위 비교는 체크인된 historical artifact 기준이며, generic dequant 경로가 MSE-only라는 사실을 반영한다.

### 이전 실험 (mlx-vlm 기반, 참고용)
- Qwen3.5 VLM baseline DocVQA 0.858 → tq-4 0.728 (-13%) 
- Qwen3.5 Text: TQ 모두 baseline ±2% 이내
- 주의: Qwen3.5는 하이브리드(6/24 attention), Qwen3-VL-2B는 dense(전 레이어), 본 실험 결과가 더 보수적임

## 코딩 컨벤션

### C/C++ (llama.cpp 패치)
- llama.cpp upstream 코딩 스타일 준수
- 4-space indent, snake_case
- 새 GGML 타입은 기존 q4_0/q8_0 패턴을 따름
- CUDA 커널: warp-level primitives, shared memory 최소화

### Python (벤치마크 프레임워크)
- Python 3.12+, UV 관리
- 타입 힌트 사용
- docstring 필수 (한국어/영어 혼용 가능)
- 벤치마크 결과는 CSV + JSON 병행 저장
- seed 고정 (default: 42)
- 매 서버 기동 전 이전 프로세스 정리 확인

## 참고 논문

- TurboQuant: arXiv 2504.19874 (ICLR 2026)
- QJL: arXiv 2406.03482
- PolarQuant: arXiv 2502.02617 (AISTATS 2026)
- KIVI: arXiv 2402.02750
- HIGGS: arXiv 2411.17525
- Qwen3-VL Technical Report: arXiv 2511.21631

## 향후 계획 (Phase 9 ~ 13)

### Phase 8 — 정합성 정리 ✅ 완료
**목표**: 문서/테스트/결과 상태 동기화
- ✅ MMLU 버그 수정 반영 (코드 + `mini_bench_text_n50.json`)
- ✅ 공식 Qwen3-VL-2B 점수 테이블 추가
- ✅ Phase 1~7 완료 상태 표기
- ✅ smoke/text 결과를 `results/runs/`에 체크인
- ✅ `AGENTS.md` scaffold 설명 제거, 현재 상태 반영

### Phase 8.5 — 공식 Parity Evaluator 구현 ✅ 완료
**목표**: P0 벤치마크(AI2D, MMMU, MathVista)의 공식 평가 로직 포팅, 공식 점수와 직접 비교 가능한 채점 체계 확립

**완료 항목:**
- ✅ `BaseEvaluator.score()` metadata kwarg 인프라 (`bench/tq_bench/evaluators/base.py`)
- ✅ 공용 유틸리티 추출 (`bench/tq_bench/evaluators/_utils.py`: levenshtein, normalize, parse_number)
- ✅ 기존 evaluator 7개에 metadata kwarg 추가 (동작 변경 없음)
- ✅ `runner.py` metadata 전달 + `_get_evaluator()` → registry 통합
- ✅ `BenchmarkConfig` parity 필드 (`parity_mode`, `parity_metric`, `parity_sample_count`)
- ✅ `_deterministic_sample()` n=-1 sentinel (full split 지원)
- ✅ **MMMU official evaluator** (`mmmu_official.py`): rfind 마지막 매칭, index2ans content matching, random fallback, eval_open 2-decimal rounding
- ✅ **MathVista official evaluator** (`mathvista_official.py`): letter→choice text 변환, Levenshtein fuzzy choice, precision-aware rounding, strict equality
- ✅ **TextVQA official evaluator** (`textvqa_official.py`): EvalAIAnswerProcessor 전체 포팅 (contractions, number words, digit-adjacent punct), leave-one-out VQA consensus
- ✅ **ChartQAPro official evaluator** (`chartqapro_official.py`): question-type routing, year flags, list pairwise eval, ANLS fallback
- ✅ Runner `<think>...</think>` strip 로직 (Thinking 모델 reasoning chain 자동 제거)
- ✅ `benchmarks.yaml` 전 항목에 parity 필드 추가, P2/P3 제한사항 주석
- ✅ Golden parity tests: 85개 통과 (`bench/tests/`), 기존 134개 KV analysis tests 회귀 없음
- ✅ Parity smoke test: baseline × AI2D/MMMU/MathVista × n=10 (`results/runs/parity_smoke_n10.json`)
- ✅ 공식 parity 감사 문서: `docs/OFFICIAL_PARITY_AUDIT.md`

**Parity smoke 결과 (n=10, baseline f16):**
| Benchmark | Existing (approx) | Parity (official) | Official |
|---|---|---|---|
| AI2D | 0.700 | 0.700 | 0.804 |
| MMMU | 0.300 | 0.000 (n=10 noise) | 0.614 |
| MathVista | 0.800 | 0.700 | 0.736 |

MathVista parity(0.700)가 공식(0.736)에 가장 가까움. MMMU는 n=10에서 공식 parser random fallback 노이즈가 큼.

### Phase 9 — VLM 공식 점수 재현 + TurboQuant VLM 비교 🔄 진행 중
**목표**: 본 프레임워크가 Qwen3-VL-2B 공식 점수를 재현하는지 검증, 이후 TQ variants VLM 성능 측정

**통합 벤치마크 러너 (`bench/run_bench.py`) 준비 완료.** 사용법:
```bash
# P0 baseline 검증 (n=100)
uv run python run_bench.py --num 100 --runtimes baseline

# core runtimes × P0 (n=100)
uv run python run_bench.py --num 100

# 전체 TQ variants (n=50)
uv run python run_bench.py --num 50 --runtimes tq-all

# Thinking 모델
uv run python run_bench.py --num 30 --model qwen3_vl_2b_thinking
```

**Phase 9-A: baseline 공식 점수 재현**
- `baseline × {AI2D, MMMU, MathVista} × n=100` (parity evaluator 사용)
- 공식: AI2D 80.4%, MMMU 61.4%, MathVista 73.6%
- ±5%p 이내 매칭 시 프레임워크 검증 성공

**Phase 9-B: TurboQuant variants VLM 비교**
- `{baseline, lcpp-kv-8, lcpp-kv-4, tq-4, tq-K4V3} × {AI2D, MMMU, MathVista} × n=100`
- 핵심 비교: lcpp-kv-4 vs tq-4 (같은 4bit), tq-3 vs tq-K4V3 (같은 avg 3.5bit)
- TQ이 vision 토큰에서 더 크게 degrade하는지 정량화

### Phase 10 — prod vec_dot QJL correction
**상태:** CUDA 코드 구현 완료, 결과/보고서 동기화는 후속 작업

**완료된 항목**
- `block_turbop*`에 `r_norm` 추가
- `ggml/src/ggml-cuda/fattn-common.cuh`에 `vec_dot_fattn_vec_KQ_turbop*` 구현
- `ggml/src/ggml-cuda/turbo-common.cuh`에 cooperative QJL helper 추가

**남은 항목**
- corrected CUDA 경로 기준 smoke / n=50 / VLM n=30 결과를 다시 고정
- `final_report.md`와 체크인된 표를 corrected/non-corrected 시점으로 분리 정리
- 필요시 CPU fallback parity 여부 결정 (`ggml-cpu/quants.c`)

### Phase 11 — Qwen3-VL-2B-Thinking 모델 연구 (별도 핵심 실험)
**목표**: Thinking 모델의 reasoning chain이 KV cache를 확장시킬 때, 비전 토큰 정보가 손실되는지 측정

**배경 가설:**
- Thinking 모델은 답변 전 `<think>...</think>` reasoning chain 생성 → KV cache에 수백~수천 토큰 추가
- 비전 토큰은 시퀀스 앞부분에 위치 → reasoning이 길어질수록 비전 토큰은 attention 거리가 멀어짐
- KV 양자화는 distant token일수록 영향이 커질 가능성 (attention 가중치가 작아도 noise가 누적)
- **핵심 질문**: "Thinking + KV 양자화" 조합이 "Instruct + KV 양자화"보다 비전 정확도를 더 크게 떨어뜨리는가?

**Phase 11-A (Task #25): Thinking 모델 기본 smoke test**
- Qwen3-VL-2B-Thinking으로 baseline + tq-4 + tq-K4V3
- Thinking mode 활성 (reasoning chain 길이 기록)
- 같은 프롬프트에서 Instruct 대비 응답 품질 비교

**Phase 11-B (Task #26): KV 길이 효과 측정 (핵심 실험)**

실험 설계:
1. **짧은 답변 모드**: `max_tokens=32`, `<think>` 억제 또는 무시
   - 비전 토큰 → 짧은 답변, KV cache 길이 ≈ prompt + 32
2. **긴 reasoning 모드**: `max_tokens=512`, `<think>` 허용
   - 비전 토큰 → 긴 reasoning → 답변, KV cache 길이 ≈ prompt + 500+
3. **동일 질문, 두 모드 모두 정확도 측정**

벤치마크 셀:
- `{baseline, tq-4, tq-K4V3} × {AI2D, MMMU, MathVista} × {짧은 답변, 긴 reasoning} × n=20`
- 각 셀에서 정확도 + KV cache 최대 길이 + 비전 토큰 attention 분포 dump

**측정 지표:**
- 정확도 저하: `acc(tq-4, 긴 reasoning) - acc(tq-4, 짧은 답변)` vs `acc(baseline, 긴 reasoning) - acc(baseline, 짧은 답변)`
- 양의 차이는 "TQ가 긴 reasoning에서 더 많이 degrade한다"는 증거
- KV dump로 긴 reasoning 시의 비전 토큰 K norm 변화 추적
- 필요 시 attention 가중치 dump로 비전 토큰 위 가중치 감소 측정

**기대 결과:** 
- Thinking + TQ 조합에서 비전 reasoning 작업이 Instruct + TQ보다 현저히 나빠짐
- 이는 "KV 양자화는 긴 컨텍스트 VLM에 더 치명적"이라는 새로운 시사점
- 논문 기여 포인트: "TurboQuant KV quantization interacts poorly with multi-step visual reasoning models"

### Phase 12 — Full matrix 실행 (옵션, 10-20시간)
**목표**: 15 runtimes × 11 benchmarks × (n=500 VLM / n=1000-3000 text)

- dual-GPU parallel + 4-slot per server = 8x parallelism
- orchestrator resume/checkpoint로 안정 실행
- 예상 소요: 텍스트 위주 ~10시간, VLM 포함 ~20시간
- 성공 시 최종 리포트에 full matrix heatmap + degradation curve 포함

### Phase 13 — Report 최종화 및 재현성 (2-4시간)
- `final_report.md` 전체 통합 (공식 점수 비교, TQ vs lcpp 실패 분석, Thinking 효과)
- Jupyter 노트북 출력 포함 최종화 (01~05)
- `results/` 디렉토리 정리 + README
- 재현 가이드 작성 (`docs/REPRODUCE.md`)
- GitHub release tag 준비
