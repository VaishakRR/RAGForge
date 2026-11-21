from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from .config import Settings
from .config import settings as default_settings
from .embeddings import EmbeddingService
from .ingest import DuplicateDocument, IngestError, Ingestor
from .ollama_client import OllamaClient, OllamaError
from .parsing import SUPPORTED_EXTENSIONS
from .rag import RAGPipeline
from .retrieval import Retriever
from .store import VectorStore


def _build(
    settings: Settings,
) -> tuple[VectorStore, OllamaClient, EmbeddingService, Ingestor, Retriever, RAGPipeline]:
    settings.ensure_dirs()
    store = VectorStore(settings.database_path)
    store.init()
    client = OllamaClient(settings.ollama_base_url, timeout=settings.request_timeout)
    embeddings = EmbeddingService(client, settings)
    ingestor = Ingestor(store, embeddings, settings)
    retriever = Retriever(store, embeddings, settings)
    pipeline = RAGPipeline(retriever, client, settings)
    return store, client, embeddings, ingestor, retriever, pipeline


def _expand_paths(paths: list[str]) -> list[Path]:
    expanded: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            expanded.extend(
                sorted(
                    child
                    for child in path.rglob("*")
                    if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS
                )
            )
        elif path.is_file():
            expanded.append(path)
        else:
            print(f"[skip] {path}: not found", file=sys.stderr)
    return expanded


async def _check(settings: Settings) -> int:
    store, client, _, _, _, _ = _build(settings)
    try:
        version = await client.version()
        base_url = await client.resolve_base_url()
        models = await client.model_names()
        counts = store.counts()
        embedder = store.get_embedder()

        print("RAGForge status")
        print(f"  Ollama:    connected ({base_url}, v{version})")
        print(f"  Models:    {', '.join(models) if models else '(none)'}")
        print(f"  Chat:      {settings.chat_model}")
        embedder_label = (
            f"{embedder.model} ({embedder.dim}-d)"
            if embedder
            else f"{settings.embed_model} (not indexed yet)"
        )
        print(f"  Embedder:  {embedder_label}")
        print(f"  Documents: {counts['documents']}")
        print(f"  Chunks:    {counts['chunks']}")
        return 0
    except OllamaError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


async def _ingest(settings: Settings, paths: list[str]) -> int:
    files = _expand_paths(paths)
    if not files:
        print("No supported files found.", file=sys.stderr)
        return 1

    store, client, _, ingestor, _, _ = _build(settings)
    failures = 0
    try:
        for path in files:
            try:
                document = await ingestor.ingest_file(path)
            except DuplicateDocument as exc:
                print(f"[skip] {path.name}: already indexed as '{exc.existing.name}'")
            except (IngestError, OllamaError, ValueError) as exc:
                failures += 1
                print(f"[fail] {path.name}: {exc}", file=sys.stderr)
            else:
                print(
                    f"[ok]   {document.name}: {document.num_chunks} chunks, "
                    f"{document.dim}-d embeddings"
                )
    finally:
        await client.close()

    counts = store.counts()
    print(f"\nIndex: {counts['documents']} documents, {counts['chunks']} chunks")
    return 1 if failures else 0


async def _ask(
    settings: Settings,
    query: str,
    model: str | None,
    top_k: int | None,
    show_sources: bool,
) -> int:
    _, client, _, _, _, pipeline = _build(settings)
    sources: list[dict[str, object]] = []
    try:
        async for event in pipeline.stream(query, model=model, top_k=top_k):
            if event["type"] == "sources":
                sources = event["sources"]  # type: ignore[assignment]
            elif event["type"] == "token":
                sys.stdout.write(str(event["content"]))
                sys.stdout.flush()
            elif event["type"] == "error":
                print(f"\n[error] {event['message']}", file=sys.stderr)
                return 1
        print()

        if show_sources and sources:
            print("\nSources:")
            for source in sources:
                scores = source.get("scores", {})
                print(
                    f"  [{source['n']}] {source['document']} "
                    f"(page {source['page']}, chunk {source['chunk_index']}, "
                    f"dense={scores.get('dense')}, bm25={scores.get('bm25')})"
                )
        return 0
    except OllamaError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


async def _docs(settings: Settings) -> int:
    store, client, _, _, _, _ = _build(settings)
    try:
        documents = store.list_documents()
        if not documents:
            print("No documents indexed.")
            return 0
        for document in documents:
            print(
                f"{document.name}  "
                f"({document.num_chunks} chunks, {document.num_chars} chars, "
                f"{document.embedder})"
            )
        return 0
    finally:
        await client.close()


async def _models(settings: Settings) -> int:
    _, client, _, _, _, _ = _build(settings)
    try:
        for model in await client.list_models():
            details = model.get("details", {})
            print(
                f"{model.get('name')}  "
                f"{details.get('parameter_size', '?')} "
                f"{details.get('quantization_level', '')}"
            )
        return 0
    except OllamaError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        await client.close()


async def _reset(settings: Settings, assume_yes: bool) -> int:
    store, client, _, _, _, _ = _build(settings)
    try:
        counts = store.counts()
        if not assume_yes:
            answer = input(
                f"Delete {counts['documents']} documents and {counts['chunks']} chunks? [y/N] "
            )
            if answer.strip().lower() not in {"y", "yes"}:
                print("Cancelled.")
                return 0
        store.reset()
        if settings.uploads_dir.exists():
            for file in settings.uploads_dir.iterdir():
                if file.is_file():
                    file.unlink(missing_ok=True)
        print("Index reset.")
        return 0
    finally:
        await client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragforge",
        description="A fully local RAG studio powered by Ollama.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the web UI and API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    subparsers.add_parser("check", help="Check Ollama connectivity and index status.")

    ingest = subparsers.add_parser("ingest", help="Ingest files or directories.")
    ingest.add_argument("paths", nargs="+")

    ask = subparsers.add_parser("ask", help="Ask a question from the terminal.")
    ask.add_argument("query")
    ask.add_argument("--model")
    ask.add_argument("--top-k", type=int)
    ask.add_argument("--sources", action="store_true", help="Print retrieved sources.")

    subparsers.add_parser("docs", help="List indexed documents.")
    subparsers.add_parser("models", help="List installed Ollama models.")

    reset = subparsers.add_parser("reset", help="Delete the index and uploaded files.")
    reset.add_argument("--yes", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "ragforge.api:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
        return 0

    settings = default_settings
    if args.command == "check":
        return asyncio.run(_check(settings))
    if args.command == "ingest":
        return asyncio.run(_ingest(settings, args.paths))
    if args.command == "ask":
        return asyncio.run(_ask(settings, args.query, args.model, args.top_k, args.sources))
    if args.command == "docs":
        return asyncio.run(_docs(settings))
    if args.command == "models":
        return asyncio.run(_models(settings))
    if args.command == "reset":
        return asyncio.run(_reset(settings, args.yes))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
