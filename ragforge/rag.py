from __future__ import annotations

from collections.abc import AsyncIterator

from .config import Settings
from .ollama_client import OllamaClient
from .retrieval import RetrievedChunk, Retriever

SYSTEM_PROMPT = """You are RAGForge, a precise retrieval-augmented assistant.
Answer ONLY using the numbered sources provided by the user.
If the sources do not contain the answer, say that the indexed documents do not
contain enough information. Do not use outside knowledge.

Each source block begins with its citation number in brackets, for example
"[2] document: notes.md, page 3". When a sentence uses information from a source,
append that exact block number in brackets, such as [2]. If several sources
support the same claim, cite them together as [1][3]. Never cite a source that
does not contain the claim, and never invent citations.

Citation rules:
- Every factual sentence MUST end with at least one citation, for example [1].
- Every bullet list item MUST end with at least one citation.
- If a claim cannot be supported by a source, omit the claim.
- Do not write a factual statement without a citation.

Write formulas in plain text instead of LaTeX. Be concise, direct, and well
structured."""

NO_CONTEXT_MESSAGE = (
    "I could not find relevant content in the indexed documents. "
    "Try uploading more documents or rephrasing your question."
)


def select_context(
    results: list[RetrievedChunk], max_chars: int
) -> list[tuple[int, RetrievedChunk, str]]:
    entries: list[tuple[int, RetrievedChunk, str]] = []
    used = 0
    number = 0

    for result in results:
        text = result.chunk.text.strip()
        if not text:
            continue
        number += 1
        block = (
            f"[{number}] document: {result.chunk.document_name}, "
            f"page {result.chunk.page}\n{text}"
        )
        if entries and used + len(block) > max_chars:
            break
        entries.append((number, result, block))
        used += len(block) + 2

    return entries


def build_messages(
    query: str,
    entries: list[tuple[int, RetrievedChunk, str]],
    history: list[dict[str, str]] | None,
    max_history: int = 6,
) -> list[dict[str, str]]:
    context = "\n\n".join(block for _, _, block in entries)
    user_prompt = (
        f"SOURCES\n{context}\n\n"
        f"QUESTION\n{query}\n\n"
        "Answer the question using only the sources above. "
        "Cite sources inline as [n]."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for message in history[-max_history:]:
            role = message.get("role", "user")
            content = (message.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content[:4000]})
    messages.append({"role": "user", "content": user_prompt})
    return messages


class RAGPipeline:
    def __init__(self, retriever: Retriever, client: OllamaClient, settings: Settings) -> None:
        self.retriever = retriever
        self.client = client
        self.settings = settings

    async def stream(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
        model: str | None = None,
        top_k: int | None = None,
        temperature: float | None = None,
        document_ids: list[str] | None = None,
    ) -> AsyncIterator[dict[str, object]]:
        model = model or self.settings.chat_model
        temperature = self.settings.temperature if temperature is None else temperature

        results = await self.retriever.search(
            query, top_k=top_k, document_ids=document_ids
        )
        entries = select_context(results, self.settings.max_context_chars)
        sources = [result.to_source(number) for number, result, _ in entries]
        yield {"type": "sources", "sources": sources}

        if not entries:
            yield {"type": "token", "content": NO_CONTEXT_MESSAGE}
            yield {"type": "done", "model": model, "retrieved": 0}
            return

        messages = build_messages(
            query, entries, history, max_history=self.settings.max_history_messages
        )
        async for chunk in self.client.chat_stream(
            model,
            messages,
            temperature=temperature,
            num_ctx=self.settings.num_ctx,
        ):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield {"type": "token", "content": content}
            if chunk.get("done"):
                yield {
                    "type": "done",
                    "model": model,
                    "retrieved": len(entries),
                    "eval_count": chunk.get("eval_count"),
                    "eval_duration": chunk.get("eval_duration"),
                    "total_duration": chunk.get("total_duration"),
                }
