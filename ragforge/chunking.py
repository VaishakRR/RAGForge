from __future__ import annotations

from dataclasses import dataclass

from .parsing import Page, clean_text


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    start: int
    end: int
    page: int


def chunk_text(
    text: str,
    size: int = 900,
    overlap: int = 150,
    page: int = 1,
    start_index: int = 0,
    min_chars: int = 40,
) -> list[TextChunk]:
    """Split text into overlapping, boundary-aware character windows.

    The window is cut at a sentence or paragraph boundary when one exists in
    the latter half of the window, otherwise at the nearest whitespace.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0:
        raise ValueError("overlap must not be negative")
    if overlap >= size:
        overlap = size // 4

    text = clean_text(text)
    if not text:
        return []

    length = len(text)
    chunks: list[TextChunk] = []
    start = 0
    index = start_index

    while start < length:
        end = min(start + size, length)

        if end < length:
            window = text[start:end]
            boundary = max(
                window.rfind("\n\n"),
                window.rfind(". "),
                window.rfind("! "),
                window.rfind("? "),
                window.rfind("\n"),
            )
            if boundary >= int(size * 0.55):
                end = start + boundary + 1
            else:
                whitespace = text.rfind(" ", start, end)
                if whitespace > start + size // 2:
                    end = whitespace

        piece = text[start:end].strip()
        is_last = end >= length

        if piece and len(piece) >= min_chars:
            chunks.append(TextChunk(index=index, text=piece, start=start, end=end, page=page))
            index += 1
        elif piece and is_last:
            if chunks and chunks[-1].page == page:
                previous = chunks[-1]
                merged = f"{previous.text} {piece}".strip()
                chunks[-1] = TextChunk(
                    index=previous.index,
                    text=merged,
                    start=previous.start,
                    end=end,
                    page=page,
                )
            else:
                chunks.append(TextChunk(index=index, text=piece, start=start, end=end, page=page))
                index += 1

        if is_last:
            break

        next_start = max(end - overlap, start + 1)
        while next_start < end and text[next_start] not in " \n\t":
            next_start += 1
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def chunk_pages(
    pages: list[Page],
    size: int = 900,
    overlap: int = 150,
    min_chars: int = 40,
) -> list[TextChunk]:
    """Chunk every page, keeping page numbers and a contiguous global index."""
    chunks: list[TextChunk] = []
    for page in pages:
        page_chunks = chunk_text(
            page.text,
            size=size,
            overlap=overlap,
            page=page.number,
            start_index=len(chunks),
            min_chars=min_chars,
        )
        chunks.extend(page_chunks)

    if not chunks:
        return []

    return [
        TextChunk(index=i, text=chunk.text, start=chunk.start, end=chunk.end, page=chunk.page)
        for i, chunk in enumerate(chunks)
    ]
