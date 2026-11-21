from __future__ import annotations

from ragforge.chunking import chunk_pages, chunk_text
from ragforge.parsing import Page


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_single_chunk() -> None:
    text = "RAG combines retrieval with generation to ground answers in sources."
    chunks = chunk_text(text, size=900)
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].page == 1


def test_long_text_splits_and_respects_size() -> None:
    sentence = "Retrieval augmented generation grounds model answers in documents. "
    text = sentence * 80
    chunks = chunk_text(text, size=300, overlap=60)

    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.text
        assert len(chunk.text) <= 300


def test_overlap_creates_more_chunks_than_no_overlap() -> None:
    text = " ".join(f"Sentence number {i} explains a retrieval concept." for i in range(80))
    with_overlap = chunk_text(text, size=240, overlap=80)
    without_overlap = chunk_text(text, size=240, overlap=0)
    assert len(with_overlap) > len(without_overlap)


def test_long_unbroken_text_is_hard_split() -> None:
    text = "x" * 2000
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) >= 6
    assert all(len(chunk.text) <= 300 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks).count("x") >= 2000


def test_chunk_pages_keeps_page_numbers_and_contiguous_indexes() -> None:
    pages = [
        Page(number=1, text="Page one talks about embeddings and vectors. " * 20),
        Page(number=2, text="Page two talks about retrieval and ranking. " * 20),
    ]
    chunks = chunk_pages(pages, size=300, overlap=60)

    assert {chunk.page for chunk in chunks} == {1, 2}
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.text for chunk in chunks)


def test_tiny_tail_is_merged_into_previous_chunk() -> None:
    text = "A" * 290 + ". Short tail"
    chunks = chunk_text(text, size=300, overlap=10, min_chars=40)
    assert chunks
    assert "Short tail" in chunks[-1].text
