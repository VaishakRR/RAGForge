from __future__ import annotations

import numpy as np

from ragforge.embeddings import EmbedderSpec
from ragforge.store import ChunkRecord, DocumentRecord, VectorStore


def make_document(document_id: str = "doc1", sha: str = "abc") -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        name="example.txt",
        sha256=sha,
        size_bytes=100,
        num_chunks=2,
        num_chars=50,
        created_at="2026-01-01T00:00:00+00:00",
        embedder="local:test:2",
        dim=2,
    )


def make_chunks(document_id: str = "doc1") -> list[ChunkRecord]:
    return [
        ChunkRecord(
            id=f"{document_id}:0",
            document_id=document_id,
            document_name="example.txt",
            chunk_index=0,
            page=1,
            text="first chunk",
            start_offset=0,
            end_offset=11,
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        ChunkRecord(
            id=f"{document_id}:1",
            document_id=document_id,
            document_name="example.txt",
            chunk_index=1,
            page=1,
            text="second chunk",
            start_offset=12,
            end_offset=24,
            embedding=np.array([0.0, 1.0], dtype=np.float32),
        ),
    ]


def test_store_lifecycle(tmp_path) -> None:
    store = VectorStore(tmp_path / "test.db")
    store.init()
    store.set_embedder(EmbedderSpec("local", "test", 2))

    store.add_document(make_document(), make_chunks())
    assert store.counts() == {"documents": 1, "chunks": 2, "characters": 50}
    assert store.find_by_sha("abc") is not None
    assert store.get_embedder() == EmbedderSpec("local", "test", 2)

    chunks, matrix = store.load_index()
    assert len(chunks) == 2
    assert matrix.shape == (2, 2)
    assert chunks[0].embedding.tolist() == [1.0, 0.0]

    assert store.delete_document("doc1") is True
    assert store.counts()["documents"] == 0
    assert store.load_index()[0] == []


def test_cache_is_invalidated_on_write(tmp_path) -> None:
    store = VectorStore(tmp_path / "cache.db")
    store.init()
    store.add_document(make_document(), make_chunks())
    assert len(store.load_index()[0]) == 2

    store.add_document(make_document("doc2", "def"), make_chunks("doc2"))
    assert len(store.load_index()[0]) == 4


def test_filtered_index_does_not_use_unfiltered_cache(tmp_path) -> None:
    store = VectorStore(tmp_path / "filter.db")
    store.init()
    store.add_document(make_document("doc1", "a"), make_chunks("doc1"))
    store.add_document(make_document("doc2", "b"), make_chunks("doc2"))

    assert len(store.load_index()[0]) == 4
    filtered, _ = store.load_index(["doc2"])
    assert len(filtered) == 2
    assert all(chunk.document_id == "doc2" for chunk in filtered)


def test_reset_clears_everything(tmp_path) -> None:
    store = VectorStore(tmp_path / "reset.db")
    store.init()
    store.add_document(make_document(), make_chunks())
    store.set_embedder(EmbedderSpec("local", "test", 2))

    store.reset()
    assert store.counts() == {"documents": 0, "chunks": 0, "characters": 0}
    assert store.get_embedder() is None
