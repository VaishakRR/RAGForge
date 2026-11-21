from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .chunking import chunk_pages
from .config import Settings
from .embeddings import EmbeddingService
from .parsing import SUPPORTED_EXTENSIONS, parse_file
from .store import ChunkRecord, DocumentRecord, VectorStore


class IngestError(RuntimeError):
    """Raised when a document cannot be ingested."""


class DuplicateDocument(IngestError):
    def __init__(self, existing: DocumentRecord) -> None:
        self.existing = existing
        super().__init__(f"'{existing.name}' is already indexed.")


class Ingestor:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingService,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.settings = settings

    async def ingest_file(
        self,
        path: Path,
        original_name: str | None = None,
        mime: str | None = None,
    ) -> DocumentRecord:
        path = Path(path)
        name = Path(original_name or path.name).name
        suffix = Path(name).suffix.lower()

        if suffix not in SUPPORTED_EXTENSIONS:
            raise IngestError(
                f"Unsupported file type '{suffix or '(none)'}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        data = path.read_bytes()
        if not data:
            raise IngestError("File is empty.")

        sha256 = hashlib.sha256(data).hexdigest()
        existing = self.store.find_by_sha(sha256)
        if existing is not None:
            raise DuplicateDocument(existing)

        parsed = parse_file(path, name=name)
        chunks = chunk_pages(
            parsed.pages,
            size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            min_chars=self.settings.min_chunk_chars,
        )
        if not chunks:
            raise IngestError("No indexable text was found in the document.")

        texts = [chunk.text for chunk in chunks]
        spec = self.store.get_embedder()
        vectors, spec = await self.embeddings.embed_for_ingest(spec, texts)

        if len(vectors) != len(chunks):
            raise IngestError("Embedding count did not match chunk count.")

        if self.store.get_embedder() is None:
            self.store.set_embedder(spec)

        document_id = uuid4().hex
        created_at = datetime.now(UTC).isoformat()
        document = DocumentRecord(
            id=document_id,
            name=name,
            sha256=sha256,
            size_bytes=len(data),
            num_chunks=len(chunks),
            num_chars=parsed.characters,
            created_at=created_at,
            embedder=spec.id,
            dim=spec.dim,
            source_path=str(path),
            mime=mime,
        )

        stored_chunks = [
            ChunkRecord(
                id=f"{document_id}:{chunk.index}",
                document_id=document_id,
                document_name=name,
                chunk_index=chunk.index,
                page=chunk.page,
                text=chunk.text,
                start_offset=chunk.start,
                end_offset=chunk.end,
                embedding=vector,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.store.add_document(document, stored_chunks)
        return document
