from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".markdown",
    ".rst",
    ".csv",
    ".json",
    ".log",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cpp",
    ".h",
    ".sql",
    ".sh",
    ".ps1",
    ".css",
}

SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | {".pdf", ".docx", ".html", ".htm"}


class ParseError(RuntimeError):
    """Raised when a document cannot be parsed."""


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    pages: list[Page]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)

    @property
    def characters(self) -> int:
        return sum(len(page.text) for page in self.pages)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _parse_pdf(path: Path) -> list[Page]:
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pypdf raises several exception types
        raise ParseError(f"Could not read PDF: {exc}") from exc

    if reader.is_encrypted:
        try:
            if reader.decrypt("") == 0:
                raise ParseError("PDF is password protected.")
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError("PDF is password protected.") from exc

    pages: list[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        cleaned = clean_text(text)
        if cleaned:
            pages.append(Page(number=number, text=cleaned))

    if not pages:
        raise ParseError(
            "No extractable text found in PDF. Scanned PDFs need OCR, which is not supported yet."
        )
    return pages


def _parse_docx(path: Path) -> list[Page]:
    try:
        document = DocxDocument(str(path))
    except Exception as exc:
        raise ParseError(f"Could not read DOCX: {exc}") from exc

    parts: list[str] = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = clean_text("\n".join(parts))
    if not text:
        raise ParseError("DOCX contains no extractable text.")
    return [Page(number=1, text=text)]


def _parse_html(path: Path) -> list[Page]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    text = clean_text(soup.get_text("\n"))
    if not text:
        raise ParseError("HTML contains no extractable text.")
    return [Page(number=1, text=text)]


def _parse_text(path: Path) -> list[Page]:
    text = clean_text(path.read_text(encoding="utf-8", errors="replace"))
    if not text:
        raise ParseError("File contains no text.")
    return [Page(number=1, text=text)]


def parse_file(path: Path, name: str | None = None) -> ParsedDocument:
    """Parse a supported file into page-like sections."""
    suffix = Path(name or path.name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ParseError(
            f"Unsupported file type '{suffix or '(none)'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        pages = _parse_pdf(path)
    elif suffix == ".docx":
        pages = _parse_docx(path)
    elif suffix in {".html", ".htm"}:
        pages = _parse_html(path)
    else:
        pages = _parse_text(path)

    return ParsedDocument(pages=pages)
