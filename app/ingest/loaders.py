"""Load unstructured documents (PDF, Markdown, text) into plain-text pages."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DOC_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}
TABLE_EXTENSIONS = {".csv"}


@dataclass
class Document:
    source: str
    text: str
    metadata: dict = field(default_factory=dict)


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in {".md", ".markdown", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [Document(source=path.name, text=text, metadata={"type": ext.lstrip(".")})] if text.strip() else []
    raise ValueError(f"Unsupported document type: {path.name}")


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    docs = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(source=path.name, text=text, metadata={"type": "pdf", "page": i}))
    return docs


def discover(directory: Path) -> tuple[list[Path], list[Path]]:
    """Return (document files, table files) found under a directory."""
    docs, tables = [], []
    for p in sorted(Path(directory).rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in DOC_EXTENSIONS:
            docs.append(p)
        elif ext in TABLE_EXTENSIONS:
            tables.append(p)
    return docs, tables
