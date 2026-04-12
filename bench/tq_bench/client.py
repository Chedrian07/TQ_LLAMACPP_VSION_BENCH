from __future__ import annotations

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str | list[dict[str, Any]]


class LlamaApiClient:
    """OpenAI-compatible client for llama-server.

    Handles chat completions with optional VLM image payloads, automatic
    retries on transient errors, and base64 image encoding helpers.

    Thread safety: a single ``httpx.Client`` instance is created per
    ``LlamaApiClient`` and shared across calling threads. ``httpx.Client``
    uses a connection pool that is safe to use concurrently, so multiple
    threads can invoke :meth:`chat_completions` in parallel without any
    additional locking. The pool size scales with ``pool_max_connections``.
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 3,
        pool_max_connections: int = 16,
        pool_max_keepalive: int = 8,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        # Shared, thread-safe httpx.Client with a connection pool sized for
        # concurrent requests (llama-server default n_parallel=4, so 16 is
        # plenty of headroom for retries/keep-alive).
        limits = httpx.Limits(
            max_connections=pool_max_connections,
            max_keepalive_connections=pool_max_keepalive,
        )
        self._http = httpx.Client(
            timeout=self.timeout_seconds,
            limits=limits,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying HTTP connection pool."""
        try:
            self._http.close()
        except Exception:
            pass

    def __enter__(self) -> "LlamaApiClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Payload builders
    # ------------------------------------------------------------------

    def build_chat_payload(
        self,
        model: str,
        messages: list[ChatMessage],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        return {
            "model": model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    @staticmethod
    def encode_image_to_data_uri(
        image: Image.Image,
        fmt: str = "PNG",
    ) -> str:
        """Encode a PIL Image to a ``data:image/...;base64,...`` URI.

        Args:
            image: The PIL image to encode.
            fmt: Image format (``"PNG"``, ``"JPEG"``, etc.).

        Returns:
            A data URI string suitable for the ``image_url.url`` field in the
            OpenAI chat message content array.
        """
        buf = io.BytesIO()
        image.save(buf, format=fmt)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        mime = f"image/{fmt.lower()}"
        return f"data:{mime};base64,{b64}"

    # ------------------------------------------------------------------
    # HTTP transport
    # ------------------------------------------------------------------

    def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a chat completion request and return the parsed JSON response.

        Retries up to ``self.max_retries`` times on 5xx / transport errors
        using exponential back-off (2^attempt seconds).

        Thread-safety: uses the shared ``httpx.Client`` connection pool, so
        multiple threads may call this method concurrently.

        Args:
            payload: The full JSON request body (as produced by
                :meth:`build_chat_payload`).

        Returns:
            The parsed JSON response dict from the server.

        Raises:
            httpx.HTTPStatusError: On non-retryable HTTP errors (4xx).
            httpx.ConnectError: If the server cannot be reached after retries.
        """
        url = f"{self.base_url}/v1/chat/completions"
        last_exc: BaseException | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._http.post(url, json=payload)

                if resp.status_code < 500:
                    resp.raise_for_status()
                    return resp.json()

                # 5xx -- retryable
                last_exc = httpx.HTTPStatusError(
                    f"Server error {resp.status_code}",
                    request=resp.request,
                    response=resp,
                )
                logger.warning(
                    "chat_completions attempt %d/%d failed: HTTP %d",
                    attempt,
                    self.max_retries,
                    resp.status_code,
                )
            except (httpx.ConnectError, httpx.TimeoutException, httpx.ReadError) as exc:
                last_exc = exc
                logger.warning(
                    "chat_completions attempt %d/%d failed: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                backoff = 2 ** attempt
                logger.debug("Retrying in %ds ...", backoff)
                time.sleep(backoff)

        # All retries exhausted.
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # High-level helpers
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        image: Image.Image | None = None,
        model: str = "default",
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Send a prompt (optionally with an image) and return the reply text.

        This is the main entry point used by the benchmark runner.

        Args:
            prompt: The user text prompt.
            image: Optional PIL image for VLM benchmarks.
            model: Model name for the payload (llama-server typically ignores
                this but it must be present).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The assistant's reply as a plain string.
        """
        if image is not None:
            data_uri = self.encode_image_to_data_uri(image)
            content: str | list[dict[str, Any]] = [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ]
        else:
            content = prompt

        messages = [ChatMessage(role="user", content=content)]
        payload = self.build_chat_payload(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        resp = self.chat_completions(payload)
        return self.extract_reply(resp)

    @staticmethod
    def extract_reply(response: dict[str, Any]) -> str:
        """Extract the assistant text from a chat completions response.

        Args:
            response: The parsed JSON response from ``/v1/chat/completions``.

        Returns:
            The text content of the first choice's message.

        Raises:
            KeyError: If the response structure is unexpected.
        """
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(
                f"Unexpected response structure: {response!r}"
            ) from exc
