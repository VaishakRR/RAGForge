from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .ollama_client import OllamaClient, OllamaError

TOKEN_RE = re.compile(r"[a-z0-9_]+")


class EmbedderMismatch(RuntimeError):
    """Raised when the configured embedder differs from the indexed one."""


@dataclass(frozen=True)
class EmbedderSpec:
    kind: str
    model: str
    dim: int

    @property
    def id(self) -> str:
        return f"{self.kind}:{self.model}:{self.dim}"

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "model": self.model, "dim": self.dim, "id": self.id}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> EmbedderSpec:
        return cls(
            kind=str(data["kind"]),
            model=str(data.get("model", "")),
            dim=int(data.get("dim", 0)),
        )

    @classmethod
    def from_json(cls, raw: str) -> EmbedderSpec:
        return cls.from_dict(json.loads(raw))


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


class HashingEmbedder:
    """Deterministic offline fallback embedder (hashed words + char trigrams)."""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        words = TOKEN_RE.findall(text.lower())

        for word in words:
            self._add(vector, word, 1.0)
            if len(word) >= 3:
                for i in range(len(word) - 2):
                    self._add(vector, word[i : i + 3], 0.35)

        return normalize_vector(vector)

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed_one(text) for text in texts]

    def _add(self, vector: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % self.dim
        sign = 1.0 if (value >> 63) & 1 else -1.0
        vector[index] += sign * weight


class EmbeddingService:
    def __init__(self, client: OllamaClient, settings: Settings) -> None:
        self.client = client
        self.settings = settings

    async def embed_with_spec(self, spec: EmbedderSpec, texts: list[str]) -> list[np.ndarray]:
        if not texts:
            return []

        if spec.kind == "ollama":
            raw = await self.client.embed(spec.model, texts)
            return [normalize_vector(np.asarray(vector, dtype=np.float32)) for vector in raw]

        if spec.kind == "local":
            return HashingEmbedder(spec.dim).embed(texts)

        raise ValueError(f"Unknown embedder kind: {spec.kind}")

    async def embed_for_ingest(
        self, existing: EmbedderSpec | None, texts: list[str]
    ) -> tuple[list[np.ndarray], EmbedderSpec]:
        if existing is not None:
            if existing.kind == "ollama" and existing.model != self.settings.embed_model:
                raise EmbedderMismatch(
                    f"The index was built with embedding model '{existing.model}', but "
                    f"RAGFORGE_EMBED_MODEL is '{self.settings.embed_model}'. Reset the index "
                    "(ragforge reset) or set RAGFORGE_EMBED_MODEL to the indexed model."
                )
            vectors = await self.embed_with_spec(existing, texts)
            return vectors, existing

        preferred = EmbedderSpec(kind="ollama", model=self.settings.embed_model, dim=0)
        try:
            vectors = await self.embed_with_spec(preferred, texts)
        except OllamaError:
            if not self.settings.allow_fallback_embeddings:
                raise
            spec = EmbedderSpec(kind="local", model="hashing", dim=512)
            vectors = await self.embed_with_spec(spec, texts)
            return vectors, spec

        spec = EmbedderSpec(
            kind="ollama", model=self.settings.embed_model, dim=len(vectors[0])
        )
        return vectors, spec

    async def embed_query(self, spec: EmbedderSpec, text: str) -> np.ndarray:
        vectors = await self.embed_with_spec(spec, [text])
        return vectors[0]
