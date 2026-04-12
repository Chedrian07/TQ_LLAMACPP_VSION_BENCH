from __future__ import annotations

import logging
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .config import BenchmarkConfig, ExperimentCell, ModelConfig, RuntimeConfig
from .server import LlamaServer, ServerLaunchConfig
from .client import ChatMessage, LlamaApiClient
from PIL import Image

from .datasets.base import BaseBenchmarkDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SampleResult:
    """Result for a single sample within a benchmark run."""

    sample_id: str
    prediction: str
    reference: str | list[str]
    score: float
    error: str | None = None


@dataclass
class RunRecord:
    """Complete record for a single experiment cell (model x runtime x benchmark)."""

    runtime_id: str
    benchmark_id: str
    status: str  # "ok", "fail", "server_crash", "error"
    model_id: str = ""
    score: float | None = None
    n_samples: int = 0
    n_succeeded: int = 0
    n_failed: int = 0
    wall_time_seconds: float = 0.0
    sample_results: list[SampleResult] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @property
    def cell_id(self) -> str:
        if self.model_id:
            return f"{self.model_id}_{self.runtime_id}_{self.benchmark_id}"
        return f"{self.runtime_id}_{self.benchmark_id}"


# ---------------------------------------------------------------------------
# Dataset / evaluator registry helpers
# ---------------------------------------------------------------------------

def _get_dataset(benchmark_id: str) -> BaseBenchmarkDataset:
    """Resolve a benchmark ID to its dataset loader instance.

    The VLM and text dataset modules are expected to register concrete
    subclasses.  This function lazily imports them and looks up by
    ``benchmark_id``.
    """
    # Lazy import to avoid circular deps and to let other teams land their
    # dataset implementations independently.
    from .datasets import vlm as _vlm_mod  # noqa: F811
    from .datasets import text as _text_mod  # noqa: F811

    # Scan both modules for subclasses that declare a matching benchmark_id.
    registry: dict[str, type[BaseBenchmarkDataset]] = {}
    for mod in (_vlm_mod, _text_mod):
        for attr_name in dir(mod):
            cls = getattr(mod, attr_name)
            if (
                isinstance(cls, type)
                and issubclass(cls, BaseBenchmarkDataset)
                and cls is not BaseBenchmarkDataset
                and hasattr(cls, "benchmark_id")
            ):
                registry[cls.benchmark_id] = cls

    if benchmark_id in registry:
        return registry[benchmark_id]()

    raise ValueError(
        f"No dataset loader registered for benchmark '{benchmark_id}'. "
        f"Available: {sorted(registry.keys())}"
    )


def _get_evaluator(metric_name: str) -> Any:
    """Resolve a metric name to an evaluator instance via the central registry."""
    from .evaluators import get_evaluator
    return get_evaluator(metric_name)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_VLM = (
    "You are a helpful assistant. Carefully look at the image and answer "
    "the user's question. Follow any instructions in the question (such as "
    "'Answer with the letter ...' or 'Provide the final answer at the end') "
    "and put your final answer at the very end of your response."
)

_SYSTEM_PROMPT_TEXT = (
    "You are a helpful assistant. Answer the user's question. "
    "Follow any instructions in the question (such as 'Answer with the letter "
    "...' or 'Provide the final answer at the end') and put your final answer "
    "at the very end of your response."
)


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Single-cell runner: start server, generate outputs, evaluate, persist.

    Args:
        server_binary: Path to the llama-server binary.
        default_host: Host address for the server.
        default_port: Port for the server.
        request_timeout: Per-request timeout in seconds for API calls.
        max_retries: Number of retries for transient API errors.
        max_tokens: Maximum tokens for generation.
        parallel_requests: Number of concurrent HTTP requests submitted to
            llama-server per cell. Matches the server's ``--parallel`` slot
            count (default 4).
        batch_size: ``-b`` value passed to llama-server.
        ubatch_size: ``-ub`` value passed to llama-server.
        n_gpu_layers: ``-ngl`` value passed to llama-server.
        cache_ram: ``--cache-ram`` value passed to llama-server.
        slot_save_path: Optional ``--slot-save-path`` directory.
        no_warmup: Whether to pass ``--no-warmup``.
        no_mmap: Whether to pass ``--no-mmap``.
    """

    def __init__(
        self,
        server_binary: Path,
        *,
        default_host: str = "127.0.0.1",
        default_port: int = 8080,
        request_timeout: float = 120.0,
        max_retries: int = 2,
        max_tokens: int = 256,
        parallel_requests: int = 4,
        batch_size: int = 512,
        ubatch_size: int = 512,
        n_gpu_layers: int = 99,
        cache_ram: int = 16384,
        slot_save_path: Path | None = Path("./kvcache"),
        no_warmup: bool = True,
        no_mmap: bool = True,
    ) -> None:
        self.server_binary = server_binary
        self.default_host = default_host
        self.default_port = default_port
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.parallel_requests = max(1, parallel_requests)
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.n_gpu_layers = n_gpu_layers
        self.cache_ram = cache_ram
        self.slot_save_path = slot_save_path
        self.no_warmup = no_warmup
        self.no_mmap = no_mmap

    def run_cell(
        self,
        cell: ExperimentCell,
        model_config: ModelConfig,
        *,
        gpu_id: int | None = None,
        port: int | None = None,
        seed: int = 42,
        parallel_requests: int | None = None,
        progress_position: int = 0,
        progress_log_interval: int = 5,
        progress_heartbeat_seconds: float = 15.0,
    ) -> RunRecord:
        """Execute a single experiment cell.

        1. Start llama-server with the runtime's cache-type-k/v.
        2. Load the dataset (n_samples from cell.benchmark).
        3. For each sample, send a prompt (with image for VLM) and collect the response.
        4. Evaluate all predictions with the appropriate metric.
        5. Stop the server.
        6. Return a RunRecord with scores, timing, and metadata.

        If the server crashes mid-run (common with prod/QJL types), the cell
        is marked as ``server_crash`` with score=0.
        """
        runtime = cell.runtime
        benchmark = cell.benchmark
        port = port or self.default_port
        n_parallel = max(1, parallel_requests or self.parallel_requests)

        logger.info(
            "Running cell: %s x %s (bits=%s, method=%s, parallel=%d)",
            runtime.id,
            benchmark.id,
            runtime.bits,
            runtime.method,
            n_parallel,
        )

        t_start = time.monotonic()

        # -- 1. Start server -------------------------------------------------
        launch_config = ServerLaunchConfig(
            binary_path=self.server_binary,
            model_path=model_config.model_path,
            mmproj_path=model_config.mmproj_path,
            host=self.default_host,
            port=port,
            n_parallel=n_parallel,
            batch_size=self.batch_size,
            ubatch_size=self.ubatch_size,
            n_gpu_layers=self.n_gpu_layers,
            cache_ram=self.cache_ram,
            slot_save_path=self.slot_save_path,
            no_warmup=self.no_warmup,
            no_mmap=self.no_mmap,
        )
        server = LlamaServer(launch_config)

        try:
            server.start(runtime, gpu_id=gpu_id)
        except (FileNotFoundError, RuntimeError, TimeoutError) as exc:
            wall_time = time.monotonic() - t_start
            logger.error(
                "Server failed to start for %s x %s: %s",
                runtime.id,
                benchmark.id,
                exc,
            )
            return RunRecord(
                runtime_id=runtime.id,
                benchmark_id=benchmark.id,
                status="error",
                model_id=model_config.id,
                score=0.0,
                wall_time_seconds=wall_time,
                notes=f"Server start failed: {exc}",
            )

        # -- 2. Load dataset --------------------------------------------------
        try:
            dataset = _get_dataset(benchmark.id)
            dataset.load(benchmark.sample_count, seed=seed)
        except Exception as exc:
            server.stop()
            wall_time = time.monotonic() - t_start
            logger.error("Dataset load failed for %s: %s", benchmark.id, exc)
            return RunRecord(
                runtime_id=runtime.id,
                benchmark_id=benchmark.id,
                status="error",
                model_id=model_config.id,
                score=0.0,
                wall_time_seconds=wall_time,
                notes=f"Dataset load failed: {exc}",
            )

        # -- 3. Iterate samples and generate ----------------------------------
        client = LlamaApiClient(
            base_url=server.base_url,
            timeout_seconds=self.request_timeout,
            pool_max_connections=max(16, n_parallel * 4),
            pool_max_keepalive=max(8, n_parallel * 2),
        )
        evaluator = _get_evaluator(benchmark.metric)
        is_vlm = benchmark.task_type == "vlm"
        system_prompt = _SYSTEM_PROMPT_VLM if is_vlm else _SYSTEM_PROMPT_TEXT

        samples = list(dataset.iter_samples())
        n_samples = len(samples)
        # Pre-allocated slots keep per-index ordering even though worker
        # threads complete out of order.
        sample_results: list[SampleResult | None] = [None] * n_samples
        server_crashed_flag = threading.Event()
        desc = f"{model_config.id}:{runtime.id} x {benchmark.id}"

        # Effective max_tokens: benchmark-defined value, bumped by the
        # model's override floor if set (e.g. Thinking models need 4096).
        effective_max_tokens = max(
            getattr(benchmark, "max_tokens", self.max_tokens) or self.max_tokens,
            model_config.max_tokens_override or 0,
        )
        logger.info(
            "  max_tokens=%d (benchmark=%s, model_override=%s)",
            effective_max_tokens,
            getattr(benchmark, "max_tokens", None),
            model_config.max_tokens_override,
        )

        try:
            with ThreadPoolExecutor(
                max_workers=n_parallel,
                thread_name_prefix=f"tqbench-{model_config.id}-{runtime.id}",
            ) as pool:
                futures = {
                    pool.submit(
                        self._run_sample,
                        idx=idx,
                        sample=sample,
                        client=client,
                        evaluator=evaluator,
                        model_config=model_config,
                        model_id=model_config.id,
                        system_prompt=system_prompt,
                        is_vlm=is_vlm,
                        server=server,
                        crash_flag=server_crashed_flag,
                        max_tokens=effective_max_tokens,
                    ): idx
                    for idx, sample in enumerate(samples)
                }

                log_every = max(1, progress_log_interval)
                next_progress_log = log_every
                n_failed_so_far = 0

                heartbeat_every = max(1.0, float(progress_heartbeat_seconds))

                with tqdm(
                    total=n_samples,
                    desc=desc,
                    unit="sample",
                    position=progress_position,
                    dynamic_ncols=True,
                ) as pbar:
                    pending = set(futures)
                    while pending:
                        done, pending = wait(
                            pending,
                            timeout=heartbeat_every,
                            return_when=FIRST_COMPLETED,
                        )

                        if not done:
                            running_count = sum(1 for fut in pending if fut.running())
                            queued_count = len(pending) - running_count
                            logger.info(
                                "  heartbeat[%s] %s x %s: done=%d/%d, in_flight=%d, queued=%d, ok=%d, fail=%d, elapsed=%.0fs",
                                model_config.id,
                                runtime.id,
                                benchmark.id,
                                pbar.n,
                                n_samples,
                                running_count,
                                queued_count,
                                pbar.n - n_failed_so_far,
                                n_failed_so_far,
                                time.monotonic() - t_start,
                            )
                            if not server_crashed_flag.is_set() and server.poll() is not None:
                                server_crashed_flag.set()
                                logger.warning(
                                    "Server crashed during %s x %s (detected during heartbeat, after %d completions, exit_code=%s)",
                                    runtime.id,
                                    benchmark.id,
                                    pbar.n,
                                    server.poll(),
                                )
                            if server_crashed_flag.is_set():
                                for pending_fut in pending:
                                    pending_fut.cancel()
                                break
                            continue

                        for fut in done:
                            if fut.cancelled():
                                continue

                            idx, result = fut.result()
                            sample_results[idx] = result
                            pbar.update(1)
                            if result.error:
                                n_failed_so_far += 1

                            if pbar.n == 1 or pbar.n >= next_progress_log or pbar.n == n_samples:
                                logger.info(
                                    "  progress[%s] %s x %s: %d/%d done, ok=%d, fail=%d, elapsed=%.0fs",
                                    model_config.id,
                                    runtime.id,
                                    benchmark.id,
                                    pbar.n,
                                    n_samples,
                                    pbar.n - n_failed_so_far,
                                    n_failed_so_far,
                                    time.monotonic() - t_start,
                                )
                                while next_progress_log <= pbar.n:
                                    next_progress_log += log_every

                        # Every few completions, double-check the server is
                        # still alive. If it has died, cancel the rest so
                        # remaining slots fail fast rather than waiting for
                        # the full request timeout.
                        if (
                            not server_crashed_flag.is_set()
                            and pbar.n % 8 == 0
                            and server.poll() is not None
                        ):
                            server_crashed_flag.set()
                            logger.warning(
                                "Server crashed during %s x %s (detected after "
                                "%d completions, exit_code=%s)",
                                runtime.id,
                                benchmark.id,
                                pbar.n,
                                server.poll(),
                            )

                        if server_crashed_flag.is_set():
                            # Try to cancel not-yet-started futures. Already
                            # running workers will see the flag and fail
                            # their next server.is_healthy() check, returning
                            # a crash error.
                            for pending_fut in pending:
                                pending_fut.cancel()
                            break
        finally:
            client.close()
            try:
                server.stop(dump_output=server_crashed_flag.is_set())
            except Exception as exc:
                logger.warning("Error stopping server: %s", exc)

        server_crashed = server_crashed_flag.is_set()

        # Fill any missing slots (cancelled futures) with crash errors so the
        # result list is dense.
        for idx, result in enumerate(sample_results):
            if result is not None:
                continue
            sample = samples[idx]
            sample_results[idx] = SampleResult(
                sample_id=str(sample.get("id", idx)),
                prediction="",
                reference=sample.get("answer", ""),
                score=0.0,
                error="Server crashed before this sample was reached",
            )
            server_crashed = True

        # All slots are filled now; convert to a concrete list[SampleResult]
        sample_results_final: list[SampleResult] = [
            sr for sr in sample_results if sr is not None
        ]
        sample_results = sample_results_final  # type: ignore[assignment]

        # -- 4. Compute aggregate score ----------------------------------------
        n_succeeded = sum(1 for sr in sample_results if sr.error is None)
        n_failed = len(sample_results) - n_succeeded

        if sample_results:
            aggregate_score = sum(sr.score for sr in sample_results) / len(
                sample_results
            )
        else:
            aggregate_score = 0.0

        wall_time = time.monotonic() - t_start

        # -- 6. Build and return RunRecord -------------------------------------
        if server_crashed:
            status = "server_crash"
            notes = (
                f"Server crashed after {n_succeeded}/{len(sample_results)} "
                f"samples. Method={runtime.method}, bits={runtime.bits}."
            )
        elif n_failed > 0 and n_succeeded == 0:
            status = "fail"
            notes = f"All {n_failed} samples failed."
        elif n_failed > 0:
            status = "ok"
            notes = f"{n_failed}/{len(sample_results)} samples had errors."
        else:
            status = "ok"
            notes = ""

        record = RunRecord(
            runtime_id=runtime.id,
            benchmark_id=benchmark.id,
            status=status,
            model_id=model_config.id,
            score=aggregate_score,
            n_samples=len(sample_results),
            n_succeeded=n_succeeded,
            n_failed=n_failed,
            wall_time_seconds=wall_time,
            sample_results=sample_results,
            notes=notes,
        )

        logger.info(
            "Cell %s x %s finished: status=%s, score=%.4f, "
            "time=%.1fs, succeeded=%d/%d, model=%s",
            runtime.id,
            benchmark.id,
            status,
            aggregate_score,
            wall_time,
            n_succeeded,
            len(sample_results),
            model_config.id,
        )

        return record

    # ------------------------------------------------------------------
    # Per-sample worker (runs in ThreadPoolExecutor)
    # ------------------------------------------------------------------

    def _run_sample(
        self,
        *,
        idx: int,
        sample: dict[str, Any],
        client: LlamaApiClient,
        evaluator: Any,
        model_config: ModelConfig,
        model_id: str,
        system_prompt: str,
        is_vlm: bool,
        server: LlamaServer,
        crash_flag: threading.Event,
        max_tokens: int | None = None,
    ) -> tuple[int, SampleResult]:
        """Run one sample end-to-end. Called concurrently from worker threads.

        Returns a ``(idx, SampleResult)`` tuple so the caller can place the
        result in the correct position in the pre-allocated results list.

        Honours ``crash_flag``: if another worker has detected a server
        crash, this method returns a crash error immediately without
        issuing another HTTP request.
        """
        sample_id = str(sample.get("id", idx))
        question = sample.get("question", "")
        reference = sample.get("answer", "")
        image_pil: Image.Image | None = sample.get("image")

        # Short-circuit if a crash has already been detected.
        if crash_flag.is_set():
            return idx, SampleResult(
                sample_id=sample_id,
                prediction="",
                reference=reference,
                score=0.0,
                error="Server crashed before this sample was reached",
            )

        # Encode image to base64 if present (CPU-bound, held by GIL but
        # cheap; done per-thread so threads aren't serialised on it).
        image_b64: str | None = None
        if is_vlm and image_pil is not None:
            try:
                image_b64 = self._encode_image_base64(image_pil)
            except Exception as exc:  # noqa: BLE001
                return idx, SampleResult(
                    sample_id=sample_id,
                    prediction="",
                    reference=reference,
                    score=0.0,
                    error=f"Image encoding error: {exc}",
                )

        messages = self._build_messages(
            system_prompt=system_prompt,
            question=question,
            image_base64=image_b64,
        )

        payload = client.build_chat_payload(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            temperature=(
                model_config.temperature
                if model_config.temperature is not None
                else 0.0
            ),
            seed=model_config.sampling_seed,
            top_k=model_config.top_k,
            top_p=model_config.top_p,
            min_p=model_config.min_p,
            repeat_penalty=model_config.repeat_penalty,
            presence_penalty=model_config.presence_penalty,
            frequency_penalty=model_config.frequency_penalty,
        )

        prediction = ""
        error_msg: str | None = None

        for attempt in range(1, self.max_retries + 1):
            if crash_flag.is_set():
                error_msg = "Server crashed before this sample was reached"
                break

            try:
                response = client.chat_completions(payload)
                prediction = self._extract_content(response)
                error_msg = None
                break
            except Exception as exc:  # noqa: BLE001
                error_msg = f"Attempt {attempt}/{self.max_retries}: {exc}"
                logger.warning(
                    "Request error for sample %s (attempt %d/%d): %s",
                    sample_id,
                    attempt,
                    self.max_retries,
                    exc,
                )
                # Only classify this as a server crash when the child process
                # has actually exited. The health endpoint can fail
                # transiently under heavy VLM load even while the server keeps
                # processing in-flight requests.
                exit_code = server.poll()
                if exit_code is not None:
                    crash_flag.set()
                    error_msg = f"Server crashed (exit_code={exit_code}): {exc}"
                    break

        if error_msg is not None and prediction == "":
            return idx, SampleResult(
                sample_id=sample_id,
                prediction="",
                reference=reference,
                score=0.0,
                error=error_msg,
            )

        try:
            sample_score = evaluator.score(prediction, reference, metadata=sample)
        except Exception as eval_exc:  # noqa: BLE001
            logger.warning(
                "Evaluation error for sample %s: %s", sample_id, eval_exc
            )
            return idx, SampleResult(
                sample_id=sample_id,
                prediction=prediction,
                reference=reference,
                score=0.0,
                error=f"Evaluation error: {eval_exc}",
            )

        return idx, SampleResult(
            sample_id=sample_id,
            prediction=prediction,
            reference=reference,
            score=sample_score,
            error=None,
        )

    # ------------------------------------------------------------------
    # Message construction helpers
    # ------------------------------------------------------------------

    # Qwen3-VL vision encoder limits.  Images beyond MAX_PIXELS are
    # resized by the model anyway, so sending oversized PNGs just wastes
    # bandwidth and can hit llama-server's HTTP body-size limit (→ 400).
    _VLM_MAX_PIXELS = 1280 * 28 * 28   # 1,003,520
    _VLM_MIN_PIXELS = 256 * 28 * 28    # 200,704

    @classmethod
    def _encode_image_base64(cls, image: Image.Image, fmt: str = "JPEG") -> str:
        """Encode a PIL Image to a base64 string.

        Large images are downscaled to fit within the VLM's max_pixels
        budget *before* encoding, and JPEG is used by default to keep
        payload size manageable (~100-300 KB instead of 5-10 MB PNG).
        """
        import base64
        import io
        import math

        w, h = image.size
        pixels = w * h

        if pixels > cls._VLM_MAX_PIXELS:
            scale = math.sqrt(cls._VLM_MAX_PIXELS / pixels)
            new_w = max(28, int(w * scale))
            new_h = max(28, int(h * scale))
            image = image.resize((new_w, new_h), Image.LANCZOS)
        elif pixels < cls._VLM_MIN_PIXELS:
            scale = math.sqrt(cls._VLM_MIN_PIXELS / pixels)
            new_w = max(28, int(w * scale))
            new_h = max(28, int(h * scale))
            image = image.resize((new_w, new_h), Image.LANCZOS)

        if image.mode != "RGB":
            image = image.convert("RGB")

        buf = io.BytesIO()
        if fmt.upper() == "JPEG":
            image.save(buf, format="JPEG", quality=90)
        else:
            image.save(buf, format=fmt)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @staticmethod
    def _build_messages(
        system_prompt: str,
        question: str,
        image_base64: str | None = None,
    ) -> list[ChatMessage]:
        """Build the chat message list for a single sample.

        For VLM benchmarks with images, the user message contains both
        an image_url content part and a text content part, following the
        OpenAI vision API format that llama-server supports.
        """
        messages: list[ChatMessage] = [
            ChatMessage(role="system", content=system_prompt),
        ]

        if image_base64 is not None:
            # Multi-modal message: image + text
            user_content: list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}",
                    },
                },
                {
                    "type": "text",
                    "text": question,
                },
            ]
            messages.append(ChatMessage(role="user", content=user_content))
        else:
            # Text-only message
            messages.append(ChatMessage(role="user", content=question))

        return messages

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        """Extract the assistant's text content from a chat completion response.

        If the response contains a ``<think>...</think>`` reasoning block
        (Qwen3-VL-Thinking and similar models), only the text AFTER the
        closing ``</think>`` tag is returned.  This prevents evaluators
        from accidentally matching letters or values mentioned during
        intermediate reasoning steps.
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return ""
            message = choices[0].get("message", {})
            content = message.get("content", "")
            text = content.strip() if isinstance(content, str) else str(content).strip()
        except (KeyError, IndexError, TypeError):
            return ""

        # Strip <think>...</think> blocks — keep only final answer.
        # Only applies to Thinking models that emit reasoning chains.
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()

        return text
