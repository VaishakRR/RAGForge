from __future__ import annotations

import asyncio
import json
import socket
import struct
from collections.abc import AsyncIterator
from typing import Any

import httpx


class OllamaError(RuntimeError):
    """Raised when Ollama cannot be reached or returns an error."""


def default_gateway_ip() -> str | None:
    """Return the default gateway IP (the Windows host when running in WSL2)."""
    try:
        with open("/proc/net/route", encoding="ascii") as fh:
            next(fh)
            for line in fh:
                fields = line.split()
                if len(fields) >= 3 and fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except (OSError, ValueError, StopIteration):
        return None
    return None


def candidate_base_urls(configured: str = "") -> list[str]:
    configured = configured.strip().rstrip("/")
    if configured and configured.lower() != "auto":
        return [configured]

    urls = ["http://localhost:11434", "http://127.0.0.1:11434"]
    gateway = default_gateway_ip()
    if gateway:
        urls.append(f"http://{gateway}:11434")
    return urls


class OllamaClient:
    """Thin async client for the Ollama HTTP API."""

    def __init__(self, base_url: str = "", timeout: float = 180.0) -> None:
        self._configured = base_url
        self.timeout = timeout
        self._resolved: str | None = None
        self._resolve_lock = asyncio.Lock()
        self._http: httpx.AsyncClient | None = None

    async def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=5.0),
                headers={"User-Agent": "ragforge/0.1"},
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def resolve_base_url(self) -> str:
        if self._resolved is not None:
            return self._resolved

        async with self._resolve_lock:
            if self._resolved is not None:
                return self._resolved

            client = await self._client()
            tried: list[str] = []
            for url in candidate_base_urls(self._configured):
                tried.append(url)
                try:
                    response = await client.get(f"{url}/api/version", timeout=3.0)
                except httpx.HTTPError:
                    continue
                if response.status_code == 200:
                    self._resolved = url
                    return url

            raise OllamaError(
                "Could not reach Ollama. Tried: "
                + ", ".join(tried)
                + ". Start Ollama and, if it runs on Windows, run "
                "scripts/setup_ollama.ps1 so it listens on 0.0.0.0:11434."
            )

    async def _url(self, path: str) -> str:
        return f"{await self.resolve_base_url()}{path}"

    async def _request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        client = await self._client()
        url = await self._url(path)
        try:
            response = await client.request(method, url, json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama request to {path} failed: {exc}") from exc

        if response.status_code >= 400:
            detail = response.text[:500]
            raise OllamaError(f"Ollama returned {response.status_code} for {path}: {detail}")

        try:
            return response.json()
        except ValueError as exc:
            raise OllamaError(f"Ollama returned invalid JSON for {path}") from exc

    async def version(self) -> str:
        data = await self._request("GET", "/api/version")
        return str(data.get("version", "unknown"))

    async def list_models(self) -> list[dict[str, Any]]:
        data = await self._request("GET", "/api/tags")
        return list(data.get("models", []))

    async def model_names(self) -> list[str]:
        return sorted(model["name"] for model in await self.list_models() if model.get("name"))

    async def embed(self, model: str, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            data = await self._request(
                "POST", "/api/embed", {"model": model, "input": batch, "truncate": True}
            )
            embeddings = data.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise OllamaError(
                    f"Embedding model '{model}' did not return embeddings. "
                    "Pull an embedding model such as nomic-embed-text."
                )
            vectors.extend(embeddings)
        return vectors

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ) -> AsyncIterator[dict[str, Any]]:
        client = await self._client()
        url = await self._url("/api/chat")
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature, "num_ctx": num_ctx},
        }

        try:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", "replace")[:500]
                    raise OllamaError(
                        f"Ollama chat failed with {response.status_code}: {body}"
                    )
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        raise OllamaError(str(chunk["error"]))
                    yield chunk
        except httpx.HTTPError as exc:
            raise OllamaError(f"Ollama chat request failed: {exc}") from exc

    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        num_ctx: int = 8192,
    ) -> str:
        data = await self._request(
            "POST",
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": temperature, "num_ctx": num_ctx},
            },
        )
        return str(data.get("message", {}).get("content", ""))
