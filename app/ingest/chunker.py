"""Structure-aware chunking.

Markdown is split on headings first so a chunk never straddles two sections,
then paragraphs are packed into chunks of roughly `chunk_size` characters with
`overlap` characters carried over for continuity. Each chunk keeps its section
heading so the model sees where a passage came from.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .loaders import Document

_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class Chunk:
    id: str
    source: str
    text: str
    section: str = ""
    page: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def citation(self) -> str:
        loc = f" p.{self.page}" if self.page else ""
        sec = f" > {self.section}" if self.section else ""
        return f"{self.source}{loc}{sec}"


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [("", text)]
    out = []
    if matches[0].start() > 0 and text[: matches[0].start()].strip():
        out.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end() : end].strip()
        if body:
            out.append((m.group(2).strip(), body))
    return out


def _pack(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for para in paragraphs:
        # Hard-split paragraphs that are bigger than a whole chunk.
        while len(para) > size:
            cut = para.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            pieces, para = para[:cut], para[cut:].lstrip()
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.append(pieces)
        if buf and len(buf) + len(para) + 2 > size:
            chunks.append(buf)
            tail = buf[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            buf = f"{tail}\n\n{para}" if tail else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        chunks.append(buf)
    return chunks


def chunk_documents(docs: list[Document], size: int = 800, overlap: int = 120) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    for doc in docs:
        page = doc.metadata.get("page")
        for section, body in _sections(doc.text):
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
            for i, text in enumerate(_pack(paragraphs, size, overlap)):
                digest = hashlib.sha1(f"{doc.source}|{page}|{section}|{i}|{text}".encode()).hexdigest()[:16]
                chunks.append(Chunk(id=digest, source=doc.source, text=text.strip(), section=section, page=page))
    return chunks
