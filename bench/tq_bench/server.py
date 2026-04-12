from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import Popen

import httpx

from .config import RuntimeConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServerLaunchConfig:
    binary_path: Path
    model_path: Path
    mmproj_path: Path | None = None
    host: str = "127.0.0.1"
    port: int = 8080
    n_parallel: int = 4


class LlamaServer:
    """Builds the `llama-server` command line and owns its process handle."""

    def __init__(self, launch_config: ServerLaunchConfig) -> None:
        self.launch_config = launch_config
        self.proc: Popen[str] | None = None

    def build_command(self, runtime_config: RuntimeConfig, *, gpu_id: int | None = None) -> list[str]:
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
            str(runtime_config.ctx_size),
            "--jinja",
            "--temp",
            "0.0",
            "-ngl",
            "99",
            "-fa",
            "on",
            "--parallel",
            str(max(1, self.launch_config.n_parallel)),
        ]
        if self.launch_config.mmproj_path is not None:
            cmd.extend(["--mmproj", str(self.launch_config.mmproj_path)])
        if gpu_id is not None:
            cmd.extend(["--gpu-id", str(gpu_id)])
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

        # Make sure no stale server is occupying our port ------------------
        self._kill_existing_on_port()

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

        self.proc = Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
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

    def stop(self) -> None:
        """Gracefully terminate the managed llama-server process.

        Sends SIGTERM and waits up to 5 seconds, then escalates to SIGKILL.
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
            # Ensure stdout pipe is closed.
            if self.proc is not None and self.proc.stdout is not None:
                self.proc.stdout.close()
            self.proc = None

    def is_healthy(self) -> bool:
        """Return True if the server responds 200 to ``GET /health``."""
        try:
            resp = httpx.get(self.healthcheck_url, timeout=3.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError, OSError):
            return False

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
                time.sleep(0.5)
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
                    time.sleep(0.5)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.debug(
                    "Neither lsof nor ss available; skipping port cleanup."
                )
        except subprocess.TimeoutExpired:
            logger.debug("lsof timed out; skipping port cleanup.")
