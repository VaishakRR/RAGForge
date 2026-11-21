from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embeddings import EmbedderSpec

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    num_chunks INTEGER NOT NULL,
    num_chars INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    embedder TEXT NOT NULL,
    dim INTEGER NOT NULL,
    source_path TEXT,
    mime TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_sha ON documents(sha256);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass
class DocumentRecord:
    id: str
    name: str
    sha256: str
    size_bytes: int
    num_chunks: int
    num_chars: int
    created_at: str
    embedder: str
    dim: int
    source_path: str | None = None
    mime: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "num_chunks": self.num_chunks,
            "num_chars": self.num_chars,
            "created_at": self.created_at,
            "embedder": self.embedder,
            "dim": self.dim,
            "source_path": self.source_path,
            "mime": self.mime,
        }


@dataclass
class ChunkRecord:
    id: str
    document_id: str
    document_name: str
    chunk_index: int
    page: int
    text: str
    start_offset: int
    end_offset: int
    embedding: np.ndarray


class VectorStore:
    """SQLite-backed document and vector store."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._version = 0
        self._cache: tuple[int, list[ChunkRecord], np.ndarray] | None = None

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(SCHEMA)

    def _invalidate(self) -> None:
        with self._lock:
            self._version += 1
            self._cache = None

    def add_document(self, document: DocumentRecord, chunks: list[ChunkRecord]) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO documents
                    (id, name, sha256, size_bytes, num_chunks, num_chars, created_at,
                     embedder, dim, source_path, mime)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document.id,
                    document.name,
                    document.sha256,
                    document.size_bytes,
                    document.num_chunks,
                    document.num_chars,
                    document.created_at,
                    document.embedder,
                    document.dim,
                    document.source_path,
                    document.mime,
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks
                    (id, document_id, chunk_index, page, text, start_offset, end_offset, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.document_id,
                        chunk.chunk_index,
                        chunk.page,
                        chunk.text,
                        chunk.start_offset,
                        chunk.end_offset,
                        sqlite3.Binary(
                            np.asarray(chunk.embedding, dtype=np.float32).tobytes()
                        ),
                    )
                    for chunk in chunks
                ],
            )
        self._invalidate()

    def _row_to_document(self, row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            name=row["name"],
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            num_chunks=row["num_chunks"],
            num_chars=row["num_chars"],
            created_at=row["created_at"],
            embedder=row["embedder"],
            dim=row["dim"],
            source_path=row["source_path"],
            mime=row["mime"],
        )

    def list_documents(self) -> list[DocumentRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC, name ASC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def get_document(self, document_id: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def find_by_sha(self, sha256: str) -> DocumentRecord | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def delete_document(self, document_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
            deleted = cursor.rowcount > 0
        if deleted:
            self._invalidate()
        return deleted

    def counts(self) -> dict[str, int]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM documents) AS documents,
                    (SELECT COUNT(*) FROM chunks) AS chunks,
                    (SELECT COALESCE(SUM(num_chars), 0) FROM documents) AS characters
                """
            ).fetchone()
        return {
            "documents": int(row["documents"]),
            "chunks": int(row["chunks"]),
            "characters": int(row["characters"]),
        }

    def load_chunks(self, document_ids: list[str] | None = None) -> list[ChunkRecord]:
        query = """
            SELECT c.id, c.document_id, d.name AS document_name, c.chunk_index, c.page,
                   c.text, c.start_offset, c.end_offset, c.embedding
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
        """
        params: list[str] = []
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            query += f" WHERE c.document_id IN ({placeholders})"
            params.extend(document_ids)
        query += " ORDER BY d.created_at, c.chunk_index"

        with self._connection() as connection:
            rows = connection.execute(query, params).fetchall()

        return [
            ChunkRecord(
                id=row["id"],
                document_id=row["document_id"],
                document_name=row["document_name"],
                chunk_index=row["chunk_index"],
                page=row["page"],
                text=row["text"],
                start_offset=row["start_offset"],
                end_offset=row["end_offset"],
                embedding=np.frombuffer(row["embedding"], dtype=np.float32).copy(),
            )
            for row in rows
        ]

    def load_index(
        self, document_ids: list[str] | None = None
    ) -> tuple[list[ChunkRecord], np.ndarray]:
        if document_ids is None:
            with self._lock:
                if self._cache is not None and self._cache[0] == self._version:
                    return self._cache[1], self._cache[2]

        chunks = self.load_chunks(document_ids)
        if chunks:
            matrix = np.vstack([chunk.embedding for chunk in chunks]).astype(np.float32)
        else:
            matrix = np.zeros((0, 0), dtype=np.float32)

        if document_ids is None:
            with self._lock:
                self._cache = (self._version, chunks, matrix)
        return chunks, matrix

    def get_meta(self, key: str) -> str | None:
        with self._connection() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_embedder(self) -> EmbedderSpec | None:
        raw = self.get_meta("embedder")
        if not raw:
            return None
        try:
            return EmbedderSpec.from_json(raw)
        except (ValueError, KeyError, TypeError):
            return None

    def set_embedder(self, spec: EmbedderSpec) -> None:
        self.set_meta("embedder", spec.to_json())

    def reset(self) -> None:
        with self._connection() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM documents")
            connection.execute("DELETE FROM meta")
        self._invalidate()
