from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen

import httpx

from .config import RuntimeConfig

logger = logging.getLogger(__name__)


def _query_gpu_compute_cap(gpu_id: int | None) -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0 and result.stdout:
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            index = gpu_id if gpu_id is not None and 0 <= gpu_id < len(lines) else 0
            if lines:
                major_minor = lines[index].split(".", 1)
                if len(major_minor) == 2 and all(part.isdigit() for part in major_minor):
                    return int(major_minor[0]) * 10 + int(major_minor[1])
    except Exception:
        pass

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = gpu_id if gpu_id is not None else 0
        major, minor = torch.cuda.get_device_capability(index)
        return major * 10 + minor
    except Exception:
        return None


def _should_disable_flash_attn(runtime_config: RuntimeConfig, gpu_id: int | None) -> bool:
    is_turbo = (
        runtime_config.cache_type_k.startswith("turbo")
        or runtime_config.cache_type_v.startswith("turbo")
    )
    if not is_turbo:
        return False

    compute_cap = _query_gpu_compute_cap(gpu_id)
    if compute_cap is None:
        return False

    # Temporary Colab workaround: Blackwell turbo runtimes currently hit a
    # misaligned-address fault in the CUDA flash-attn vec path, while A100/H100
    # should continue to use flash attention.
    return compute_cap >= 120


@dataclass(frozen=True)
class ServerLaunchConfig:
    binary_path: Path
    model_path: Path
    mmproj_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    n_parallel: int = 4
    batch_size: int = 512
    ubatch_size: int = 512
    n_gpu_layers: int = 99
    cache_ram: int = 16384
    slot_save_path: Path | None = Path("./kvcache")
    no_warmup: bool = True
    no_mmap: bool = True


class LlamaServer:
    """Builds the `llama-server` command line and owns its process handle."""

    def __init__(self, launch_config: ServerLaunchConfig) -> None:
        self.launch_config = launch_config
        self.proc: Popen[str] | None = None

    def build_command(self, runtime_config: RuntimeConfig, *, gpu_id: int | None = None) -> list[str]:
        n_parallel = max(1, self.launch_config.n_parallel)
        disable_flash_attn = _should_disable_flash_attn(runtime_config, gpu_id)
        flash_attn_mode = "off" if disable_flash_attn else "on"

        cmd = [
            str(self.launch_config.binary_path),
            "-m",
            str(self.launch_config.model_path),
            "--cache-type-k",
            runtime_config.cache_type_k,
            "--cache-type-v",
            runtime_config.cache_type_v,
            "--host",
            self.launch_config.host,
            "--port",
            str(self.launch_config.port),
            "--ctx-size",
            str(runtime_config.ctx_size * n_parallel),
            "-b",
            str(self.launch_config.batch_size),
            "-ub",
            str(self.launch_config.ubatch_size),
            "--jinja",
            "--temp",
            "0.0",
            "-ngl",
            str(self.launch_config.n_gpu_layers),
            "-fa",
            flash_attn_mode,
            "--parallel",
            str(n_parallel),
        ]
        if self.launch_config.mmproj_path is not None:
            cmd.extend(["--mmproj", str(self.launch_config.mmproj_path)])
        cmd.extend(["--cache-ram", str(self.launch_config.cache_ram)])
        if self.launch_config.slot_save_path is not None:
            cmd.extend(["--slot-save-path", str(self.launch_config.slot_save_path)])
        # Benchmark scores should not depend on a synthetic empty warmup pass.
        # Disabling warmup also avoids transient VRAM spikes on smaller GPUs.
        if self.launch_config.no_warmup:
            cmd.append("--no-warmup")
        if self.launch_config.no_mmap:
            cmd.append("--no-mmap")
        if disable_flash_attn:
            logger.info(
                "Disabling flash attention for %s on Blackwell TurboQuant runtime",
                runtime_config.id,
            )
        return cmd

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    def start(
        self,
        runtime_config: RuntimeConfig,
        *,
        gpu_id: int | None = None,
        startup_timeout: float = 120.0,
        health_poll_interval: float = 2.0,
    ) -> None:
        """Launch llama-server and block until it is healthy.

        Args:
            runtime_config: Runtime configuration (cache types, ctx size, etc.).
            gpu_id: Optional GPU device id to pin the server to.
            startup_timeout: Maximum seconds to wait for the server to become
                healthy before raising ``TimeoutError``.
            health_poll_interval: Seconds between health-check polls.

        Raises:
            FileNotFoundError: If the server binary or model file does not exist.
            RuntimeError: If the server process exits during startup or the port
                is already occupied.
            TimeoutError: If the server does not become healthy within
                *startup_timeout* seconds.
        """
        # Validate paths --------------------------------------------------
        if not self.launch_config.binary_path.exists():
            raise FileNotFoundError(
                f"llama-server binary not found: {self.launch_config.binary_path}"
            )
        if not self.launch_config.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.launch_config.model_path}"
            )
        if (
            self.launch_config.mmproj_path is not None
            and not self.launch_config.mmproj_path.exists()
        ):
            raise FileNotFoundError(
                f"mmproj file not found: {self.launch_config.mmproj_path}"
            )
        if self.launch_config.slot_save_path is not None:
            self.launch_config.slot_save_path.mkdir(parents=True, exist_ok=True)

        # Make sure no stale server is occupying our port ------------------
        self._kill_existing_on_port()
        self._ensure_port_available()

        # Stop any previously managed process ------------------------------
        self.stop()

        cmd = self.build_command(runtime_config, gpu_id=gpu_id)
        logger.info(
            "Starting llama-server on %s:%d  (runtime=%s, gpu=%s)",
            self.launch_config.host,
            self.launch_config.port,
            runtime_config.id,
            gpu_id,
        )
        logger.debug("Command: %s", " ".join(cmd))

        # Pin to a specific GPU via CUDA_VISIBLE_DEVICES.
        # On WSL2 with mixed GPUs (e.g. RTX 5070 Ti + RTX 4060 Ti):
        #   - --main-gpu fails: WSL2 duplicate PCI bus IDs cause llama.cpp
        #     to deduplicate GPUs, so only device 0 is available.
        #   - --device CUDAX fails: NCCL ncclCommInitAll() crashes during
        #     ggml_cuda_init() before device selection is even reached.
        #   - CUDA_VISIBLE_DEVICES=N works: prevents multi-GPU NCCL init
        #     entirely, making only the selected GPU visible to CUDA.
        proc_env = os.environ.copy()
        if gpu_id is not None:
            proc_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            logger.info("Pinning to GPU %d via CUDA_VISIBLE_DEVICES=%d", gpu_id, gpu_id)

        # Keep attention rotation disabled across benchmark runs so KV-cache
        # comparisons are not confounded by llama.cpp's built-in Hadamard
        # pre-rotation path.
        proc_env["LLAMA_ATTN_ROT_DISABLE"] = "1"
        logger.info("Setting LLAMA_ATTN_ROT_DISABLE=1")

        self.proc = Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=proc_env,
        )

        # Wait for health ---------------------------------------------------
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            # Check that the process has not exited.
            retcode = self.proc.poll()
            if retcode is not None:
                # Grab whatever output is available for diagnostics.
                stdout_tail = ""
                if self.proc.stdout is not None:
                    stdout_tail = self.proc.stdout.read()
                self.proc = None
                raise RuntimeError(
                    f"llama-server exited during startup with code {retcode}.\n"
                    f"--- stdout/stderr tail ---\n{stdout_tail[-2000:]}"
                )

            if self.is_healthy():
                logger.info(
                    "llama-server healthy on port %d (pid=%d)",
                    self.launch_config.port,
                    self.proc.pid,
                )
                return

            time.sleep(health_poll_interval)

        # Timed out -- kill the stalled process and raise.
        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy within {startup_timeout}s "
            f"on port {self.launch_config.port}"
        )

    def stop(self, *, dump_output: bool = False) -> None:
        """Gracefully terminate the managed llama-server process.

        Sends SIGTERM and waits up to 5 seconds, then escalates to SIGKILL.
        If *dump_output* is True, reads and logs the server's stdout/stderr
        before closing — useful for diagnosing crashes.
        """
        if self.proc is None:
            return

        pid = self.proc.pid
        logger.info("Stopping llama-server (pid=%d) ...", pid)

        try:
            self.proc.terminate()  # SIGTERM
            self.proc.wait(timeout=5)
            logger.info("llama-server (pid=%d) terminated gracefully.", pid)
        except subprocess.TimeoutExpired:
            logger.warning(
                "llama-server (pid=%d) did not exit after SIGTERM; sending SIGKILL.",
                pid,
            )
            self.proc.kill()  # SIGKILL
            self.proc.wait(timeout=5)
        finally:
            if dump_output and self.proc is not None and self.proc.stdout is not None:
                try:
                    tail = self.proc.stdout.read()
                    if tail and tail.strip():
                        # Show last 3000 chars of server output
                        snippet = tail.strip()[-3000:]
                        logger.error(
                            "llama-server (pid=%d) output before exit:\n%s",
                            pid, snippet,
                        )
                except Exception:
                    pass
            # Ensure stdout pipe is closed.
            if self.proc is not None and self.proc.stdout is not None:
                self.proc.stdout.close()
            self.proc = None
            self._wait_until_port_free(timeout=10.0)

    def is_healthy(self) -> bool:
        """Return True if the server responds 200 to ``GET /health``."""
        try:
            resp = httpx.get(self.healthcheck_url, timeout=3.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, OSError):
            return False

    def poll(self) -> int | None:
        """Return the child process exit code if it has exited, else None."""
        if self.proc is None:
            return None
        return self.proc.poll()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def healthcheck_url(self) -> str:
        return f"http://{self.launch_config.host}:{self.launch_config.port}/health"

    @property
    def base_url(self) -> str:
        return f"http://{self.launch_config.host}:{self.launch_config.port}"

    def _kill_existing_on_port(self) -> None:
        """Kill any process listening on our configured port.

        Uses ``lsof`` on Linux/macOS to discover the PID.  If ``lsof`` is not
        available the step is silently skipped.
        """
        port = self.launch_config.port
        try:
            result = subprocess.run(
                ["lsof", "-t", "-i", f"TCP:{port}", "-sTCP:LISTEN"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            pids = [
                int(p)
                for p in result.stdout.strip().split()
                if p.strip().isdigit()
            ]
            for pid in pids:
                if pid == os.getpid():
                    continue
                logger.warning(
                    "Killing existing process pid=%d on port %d", pid, port
                )
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            # Small grace period for the OS to free the port.
            if pids:
                self._wait_until_port_free(timeout=10.0)
        except FileNotFoundError:
            # lsof not available -- try ss as fallback.
            try:
                result = subprocess.run(
                    ["ss", "-tlnp", f"sport = :{port}"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                for line in result.stdout.splitlines():
                    if f":{port}" in line and "pid=" in line:
                        # Extract pid from e.g. "pid=12345,"
                        for token in line.split(","):
                            if token.startswith("pid="):
                                pid = int(token.split("=")[1])
                                if pid != os.getpid():
                                    logger.warning(
                                        "Killing existing process pid=%d on port %d",
                                        pid,
                                        port,
                                    )
                                    try:
                                        os.kill(pid, signal.SIGKILL)
                                    except OSError:
                                        pass
                if result.stdout.strip():
                    self._wait_until_port_free(timeout=10.0)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.debug(
                    "Neither lsof nor ss available; skipping port cleanup."
                )
        except subprocess.TimeoutExpired:
            logger.debug("lsof timed out; skipping port cleanup.")

    def _wait_until_port_free(self, timeout: float = 10.0) -> None:
        """Wait until the configured TCP port is no longer accepting connections."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            in_use = not self._can_bind_port()

            if not in_use:
                return

            time.sleep(0.1)

        logger.warning(
            "Port %d still appears busy after waiting %.1fs",
            self.launch_config.port,
            timeout,
        )

    def _can_bind_port(self) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.launch_config.host, self.launch_config.port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _ensure_port_available(self) -> None:
        if self._can_bind_port():
            return
        raise RuntimeError(
            f"Port {self.launch_config.port} is already in use before starting llama-server. "
            f"Choose another --port or stop the conflicting process."
        )

    # ------------------------------------------------------------------
    # GPU / KV cache memory monitoring
    # ------------------------------------------------------------------

    def get_gpu_memory(self, *, gpu_id: int | None = None) -> int:
        """Return GPU memory used (bytes) by querying nvidia-smi.

        Best-effort: returns 0 if nvidia-smi is unavailable or fails.
        """
        if self.proc is None:
            return 0
        try:
            # Query memory used by our server process
            device = str(gpu_id) if gpu_id is not None else "0"
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device}",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # nvidia-smi reports in MiB
                mib = float(result.stdout.strip().split("\n")[0])
                return int(mib * 1024 * 1024)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return 0

    def get_kv_cache_bytes(self) -> int:
        """Return KV cache size in bytes by querying the /slots endpoint.

        Best-effort: returns 0 if the endpoint is unavailable or fails.
        """
        if self.proc is None:
            return 0
        try:
            resp = httpx.get(
                f"{self.base_url}/slots",
                timeout=3.0,
            )
            if resp.status_code == 200:
                slots = resp.json()
                # Sum n_past * 2 (K+V) * type_size across slots
                # The /slots endpoint returns per-slot info; extract cache usage
                total = 0
                for slot in slots:
                    cache_tokens = slot.get("n_past", 0)
                    # Approximate: each cached token uses (n_embd * 2 * type_bytes)
                    # but we don't know type_bytes here. Use raw n_past as a proxy.
                    total += cache_tokens
                return total
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, OSError, ValueError):
            pass
        return 0
