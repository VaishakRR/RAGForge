from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .config import settings as default_settings
from .embeddings import EmbeddingService
from .ingest import DuplicateDocument, IngestError, Ingestor
from .ollama_client import OllamaClient, OllamaError
from .parsing import SUPPORTED_EXTENSIONS
from .rag import RAGPipeline
from .retrieval import Retriever
from .store import VectorStore

STATIC_DIR = Path(__file__).parent / "static"


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    history: list[ChatMessage] = Field(default_factory=list)
    model: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    document_ids: list[str] | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] | None = None


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _state(request: Request) -> object:
    return request.app.state


def create_app(app_settings: Settings | None = None) -> FastAPI:
    config = app_settings or default_settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config.ensure_dirs()
        store = VectorStore(config.database_path)
        store.init()
        client = OllamaClient(config.ollama_base_url, timeout=config.request_timeout)
        embeddings = EmbeddingService(client, config)

        app.state.settings = config
        app.state.store = store
        app.state.client = client
        app.state.embeddings = embeddings
        app.state.ingestor = Ingestor(store, embeddings, config)
        app.state.retriever = Retriever(store, embeddings, config)
        app.state.pipeline = RAGPipeline(
            app.state.retriever, client, config
        )

        yield
        await client.close()

    app = FastAPI(
        title="RAGForge",
        description="A fully local, from-scratch RAG studio powered by Ollama.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon() -> FileResponse:
        return FileResponse(STATIC_DIR / "favicon.svg")

    @app.get("/api/health")
    async def health(request: Request) -> dict[str, object]:
        state = _state(request)
        store: VectorStore = state.store
        client: OllamaClient = state.client
        settings: Settings = state.settings

        counts = store.counts()
        embedder = store.get_embedder()
        result: dict[str, object] = {
            "ok": True,
            "ollama": {"connected": False},
            "config": {
                "chat_model": settings.chat_model,
                "embed_model": settings.embed_model,
                "top_k": settings.top_k,
                "temperature": settings.temperature,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "mmr_lambda": settings.mmr_lambda,
            },
            "index": {
                **counts,
                "embedder": embedder.to_dict() if embedder else None,
            },
        }

        try:
            version = await client.version()
            models = await client.model_names()
            base_url = await client.resolve_base_url()
            result["ollama"] = {
                "connected": True,
                "version": version,
                "base_url": base_url,
                "models": models,
            }
        except OllamaError as exc:
            result["ok"] = False
            result["ollama"] = {"connected": False, "error": str(exc)}

        return result

    @app.get("/api/models")
    async def models(request: Request) -> dict[str, object]:
        client: OllamaClient = _state(request).client
        try:
            installed = await client.list_models()
        except OllamaError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "models": [
                {
                    "name": model.get("name"),
                    "size": model.get("size"),
                    "family": model.get("details", {}).get("family"),
                    "parameters": model.get("details", {}).get("parameter_size"),
                    "quantization": model.get("details", {}).get("quantization_level"),
                }
                for model in installed
            ]
        }

    @app.get("/api/documents")
    async def list_documents(request: Request) -> dict[str, object]:
        store: VectorStore = _state(request).store
        return {"documents": [document.to_dict() for document in store.list_documents()]}

    @app.post("/api/documents")
    async def upload_documents(
        request: Request,
        files: Annotated[list[UploadFile], File()],
    ) -> dict[str, object]:
        state = _state(request)
        settings: Settings = state.settings
        ingestor: Ingestor = state.ingestor

        documents: list[dict[str, object]] = []
        errors: list[dict[str, object]] = []
        max_bytes = settings.max_upload_mb * 1024 * 1024

        for upload in files:
            name = Path(upload.filename or "upload").name
            suffix = Path(name).suffix.lower()

            if suffix not in SUPPORTED_EXTENSIONS:
                errors.append({"file": name, "error": f"Unsupported file type '{suffix}'."})
                continue

            data = await upload.read()
            if not data:
                errors.append({"file": name, "error": "File is empty."})
                continue
            if len(data) > max_bytes:
                errors.append(
                    {
                        "file": name,
                        "error": f"File exceeds the {settings.max_upload_mb} MB limit.",
                    }
                )
                continue

            destination = settings.uploads_dir / f"{uuid4().hex}{suffix}"
            destination.write_bytes(data)

            try:
                document = await ingestor.ingest_file(
                    destination, original_name=name, mime=upload.content_type
                )
            except DuplicateDocument as exc:
                destination.unlink(missing_ok=True)
                errors.append(
                    {
                        "file": name,
                        "error": f"Already indexed as '{exc.existing.name}'.",
                        "document_id": exc.existing.id,
                    }
                )
            except (IngestError, OllamaError, ValueError) as exc:
                destination.unlink(missing_ok=True)
                errors.append({"file": name, "error": str(exc)})
            except Exception as exc:  # keep the UI resilient to bad files
                destination.unlink(missing_ok=True)
                errors.append({"file": name, "error": f"Unexpected error: {exc}"})
            else:
                documents.append(document.to_dict())

        return {"documents": documents, "errors": errors}

    @app.delete("/api/documents/{document_id}")
    async def delete_document(request: Request, document_id: str) -> dict[str, object]:
        state = _state(request)
        store: VectorStore = state.store
        settings: Settings = state.settings

        document = store.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found.")

        deleted = store.delete_document(document_id)
        if deleted and document.source_path:
            source = Path(document.source_path)
            try:
                if source.resolve().is_relative_to(settings.uploads_dir.resolve()):
                    source.unlink(missing_ok=True)
            except (OSError, ValueError):
                pass

        return {"deleted": deleted, "id": document_id}

    @app.post("/api/search")
    async def search(request: Request, payload: SearchRequest) -> dict[str, object]:
        retriever: Retriever = _state(request).retriever
        try:
            results = await retriever.search(
                payload.query, top_k=payload.top_k, document_ids=payload.document_ids
            )
        except (OllamaError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "query": payload.query,
            "results": [result.to_source(i) for i, result in enumerate(results, start=1)],
        }

    @app.post("/api/chat")
    async def chat(request: Request, payload: ChatRequest) -> StreamingResponse:
        state = _state(request)
        pipeline: RAGPipeline = state.pipeline
        history = [message.model_dump() for message in payload.history]

        async def event_stream():
            try:
                async for event in pipeline.stream(
                    payload.query,
                    history=history,
                    model=payload.model,
                    top_k=payload.top_k,
                    temperature=payload.temperature,
                    document_ids=payload.document_ids,
                ):
                    yield _sse(event)
            except asyncio.CancelledError:
                raise
            except (OllamaError, RuntimeError, ValueError) as exc:
                yield _sse({"type": "error", "message": str(exc)})
                yield _sse({"type": "done"})
            except Exception as exc:
                yield _sse({"type": "error", "message": f"Unexpected error: {exc}"})
                yield _sse({"type": "done"})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/api/stats")
    async def stats(request: Request) -> dict[str, object]:
        state = _state(request)
        store: VectorStore = state.store
        settings: Settings = state.settings
        embedder = store.get_embedder()
        return {
            **store.counts(),
            "embedder": embedder.to_dict() if embedder else None,
            "chat_model": settings.chat_model,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }

    @app.post("/api/admin/reset")
    async def reset(request: Request) -> dict[str, object]:
        state = _state(request)
        store: VectorStore = state.store
        settings: Settings = state.settings
        store.reset()
        if settings.uploads_dir.exists():
            for file in settings.uploads_dir.iterdir():
                if file.is_file():
                    file.unlink(missing_ok=True)
        return {"ok": True, **store.counts()}

    return app


app = create_app()
