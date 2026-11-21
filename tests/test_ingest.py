from __future__ import annotations

import numpy as np
import pytest

from ragforge.config import Settings
from ragforge.embeddings import EmbedderSpec
from ragforge.ingest import DuplicateDocument, IngestError, Ingestor
from ragforge.store import VectorStore


class FakeEmbeddings:
    async def embed_for_ingest(self, existing, texts):
        spec = existing or EmbedderSpec("local", "fake", 8)
        vectors = [np.ones(8, dtype=np.float32) / np.sqrt(8) for _ in texts]
        return vectors, spec


def make_settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        chunk_size=200,
        chunk_overlap=40,
        min_chunk_chars=20,
    )


def make_ingestor(tmp_path) -> tuple[Ingestor, VectorStore]:
    settings = make_settings(tmp_path)
    settings.ensure_dirs()
    store = VectorStore(settings.database_path)
    store.init()
    return Ingestor(store, FakeEmbeddings(), settings), store


@pytest.mark.asyncio
async def test_ingest_text_file(tmp_path) -> None:
    ingestor, store = make_ingestor(tmp_path)
    path = tmp_path / "notes.md"
    path.write_text(
        "# Retrieval\n\n" + "Hybrid retrieval combines dense and sparse signals. " * 20,
        encoding="utf-8",
    )

    document = await ingestor.ingest_file(path)

    assert document.name == "notes.md"
    assert document.num_chunks > 1
    assert document.dim == 8
    assert store.counts()["documents"] == 1
    assert store.get_embedder() == EmbedderSpec("local", "fake", 8)


@pytest.mark.asyncio
async def test_duplicate_document_is_rejected(tmp_path) -> None:
    ingestor, _ = make_ingestor(tmp_path)
    path = tmp_path / "duplicate.txt"
    path.write_text("The same content twice. " * 20, encoding="utf-8")

    await ingestor.ingest_file(path)
    with pytest.raises(DuplicateDocument):
        await ingestor.ingest_file(path)


@pytest.mark.asyncio
async def test_unsupported_file_type_is_rejected(tmp_path) -> None:
    ingestor, _ = make_ingestor(tmp_path)
    path = tmp_path / "image.bmp"
    path.write_bytes(b"not an image")

    with pytest.raises(IngestError):
        await ingestor.ingest_file(path)


@pytest.mark.asyncio
async def test_empty_file_is_rejected(tmp_path) -> None:
    ingestor, _ = make_ingestor(tmp_path)
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(IngestError):
        await ingestor.ingest_file(path)
