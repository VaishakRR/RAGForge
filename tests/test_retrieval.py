from __future__ import annotations

import numpy as np

from ragforge.retrieval import (
    bm25_scores,
    cosine_similarities,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
    tokenize,
)


def test_tokenize_lowercases_and_keeps_alphanumerics() -> None:
    assert tokenize("RAG, BM25 and [citations]!") == ["rag", "bm25", "and", "citations"]


def test_bm25_ranks_relevant_document_first() -> None:
    documents = [
        "The cat sat on the warm mat.",
        "Quantum physics describes particles and fields.",
        "A cat and dogs are common household pets with many extra words here.",
    ]
    scores = bm25_scores("cat", documents)
    assert scores[0] > scores[2] > scores[1]
    assert scores[1] == 0


def test_bm25_handles_empty_corpus() -> None:
    assert bm25_scores("anything", []).shape == (0,)


def test_cosine_similarities() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    scores = cosine_similarities(query, matrix)
    np.testing.assert_allclose(scores, [1.0, 0.0, -1.0])


def test_rrf_prefers_documents_in_multiple_rankings() -> None:
    fused = reciprocal_rank_fusion([[0, 1], [1, 0], [1]])
    assert fused[1] > fused[0]


def test_mmr_selects_diverse_results() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    chosen = maximal_marginal_relevance(query, candidates, k=2, lambda_=0.3)
    assert chosen[0] == 0
    assert chosen[1] == 2


def test_mmr_handles_k_larger_than_candidates() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    candidates = np.array([[1.0, 0.0]], dtype=np.float32)
    assert maximal_marginal_relevance(query, candidates, k=5) == [0]
