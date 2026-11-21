from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .config import Settings
from .embeddings import EmbeddingService
from .store import ChunkRecord, VectorStore

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def cosine_similarities(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    return (matrix @ query).astype(np.float32)


def bm25_scores(
    query: str, documents: list[str], k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    """Rank documents with Okapi BM25. Returns a score per document."""
    count = len(documents)
    if count == 0:
        return np.zeros((0,), dtype=np.float32)

    tokenized = [tokenize(document) for document in documents]
    lengths = np.array([len(tokens) for tokens in tokenized], dtype=np.float32)
    average_length = float(lengths.mean()) if count else 0.0
    if average_length == 0:
        return np.zeros((count,), dtype=np.float32)

    document_frequency: Counter[str] = Counter()
    term_counts = [Counter(tokens) for tokens in tokenized]
    for counts in term_counts:
        document_frequency.update(counts.keys())

    scores = np.zeros((count,), dtype=np.float32)
    for term in tokenize(query):
        frequency = document_frequency.get(term, 0)
        if frequency == 0:
            continue
        idf = math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
        for index, counts in enumerate(term_counts):
            term_frequency = counts.get(term, 0)
            if term_frequency == 0:
                continue
            denominator = term_frequency + k1 * (
                1.0 - b + b * float(lengths[index]) / average_length
            )
            scores[index] += idf * (term_frequency * (k1 + 1.0)) / denominator

    return scores


def reciprocal_rank_fusion(
    rankings: list[list[int]], k: int = 60
) -> dict[int, float]:
    """Fuse ranked lists with reciprocal rank fusion."""
    scores: defaultdict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] += 1.0 / (k + rank)
    return dict(scores)


def maximal_marginal_relevance(
    query: np.ndarray,
    candidates: np.ndarray,
    k: int,
    lambda_: float = 0.6,
) -> list[int]:
    """Select k candidates balancing query relevance and diversity."""
    count = len(candidates)
    if count == 0:
        return []
    k = min(k, count)

    query_scores = candidates @ query
    selected: list[int] = []
    remaining = set(range(count))

    while remaining and len(selected) < k:
        if not selected:
            best = max(remaining, key=lambda index: float(query_scores[index]))
        else:
            selected_vectors = candidates[selected]
            best = max(
                remaining,
                key=lambda index: lambda_ * float(query_scores[index])
                - (1.0 - lambda_)
                * float(np.max(selected_vectors @ candidates[index])),
            )
        selected.append(best)
        remaining.remove(best)

    return selected


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0

    def to_source(self, number: int) -> dict[str, object]:
        return {
            "n": number,
            "chunk_id": self.chunk.id,
            "document_id": self.chunk.document_id,
            "document": self.chunk.document_name,
            "page": self.chunk.page,
            "chunk_index": self.chunk.chunk_index,
            "text": self.chunk.text,
            "scores": {
                "dense": round(self.dense_score, 4),
                "bm25": round(self.bm25_score, 4),
                "rrf": round(self.rrf_score, 6),
            },
        }


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingService,
        settings: Settings,
    ) -> None:
        self.store = store
        self.embeddings = embeddings
        self.settings = settings

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        dense_k: int | None = None,
        bm25_k: int | None = None,
        mmr_lambda: float | None = None,
        document_ids: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []

        spec = self.store.get_embedder()
        if spec is None:
            return []

        top_k = top_k or self.settings.top_k
        dense_k = dense_k or self.settings.dense_k
        bm25_k = bm25_k or self.settings.bm25_k
        mmr_lambda = self.settings.mmr_lambda if mmr_lambda is None else mmr_lambda

        chunks, matrix = self.store.load_index(document_ids)
        if not chunks:
            return []

        query_vector = await self.embeddings.embed_query(spec, query)
        if query_vector.shape[0] != matrix.shape[1]:
            raise RuntimeError(
                "Query embedding dimension does not match the index. "
                "The embedding model may have changed; reset the index and re-ingest."
            )

        dense = cosine_similarities(query_vector, matrix)
        dense_order = np.argsort(-dense, kind="stable")[:dense_k].tolist()

        sparse = bm25_scores(query, [chunk.text for chunk in chunks])
        sparse_order = [
            int(index) for index in np.argsort(-sparse, kind="stable") if sparse[index] > 0
        ][:bm25_k]

        rankings = [dense_order]
        if sparse_order:
            rankings.append(sparse_order)
        fused = reciprocal_rank_fusion(rankings)
        if not fused:
            return []

        pool_size = max(top_k * 4, top_k)
        pool = sorted(fused, key=lambda index: fused[index], reverse=True)[:pool_size]
        pool_matrix = matrix[pool]

        chosen = maximal_marginal_relevance(
            query_vector, pool_matrix, k=top_k, lambda_=mmr_lambda
        )
        # MMR picks a diverse set; present it in relevance order so the most
        # useful evidence is source [1], [2], ... in the prompt.
        chosen.sort(key=lambda local_index: fused[pool[local_index]], reverse=True)

        results: list[RetrievedChunk] = []
        for local_index in chosen:
            global_index = pool[local_index]
            results.append(
                RetrievedChunk(
                    chunk=chunks[global_index],
                    dense_score=float(dense[global_index]),
                    bm25_score=float(sparse[global_index]),
                    rrf_score=float(fused[global_index]),
                )
            )
        return results
