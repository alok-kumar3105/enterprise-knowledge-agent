from app.ingest.chunker import chunk_documents
from app.ingest.loaders import Document, discover

from .conftest import SAMPLE


def test_chunks_respect_markdown_sections():
    doc = Document("p.md", "# Title\n\nintro\n\n## Alpha\n\nalpha body\n\n## Beta\n\nbeta body")
    chunks = chunk_documents([doc], size=200, overlap=20)
    assert [c.section for c in chunks] == ["Title", "Alpha", "Beta"]
    assert all(c.source == "p.md" for c in chunks)
    assert chunks[1].citation == "p.md > Alpha"


def test_long_sections_are_split_with_overlap_and_bounded_size():
    para = " ".join(f"word{i}" for i in range(400))
    chunks = chunk_documents([Document("x.txt", para + "\n\n" + para)], size=300, overlap=50)
    assert len(chunks) > 5
    assert max(len(c.text) for c in chunks) <= 300 + 50 + 2


def test_chunk_ids_are_stable_and_unique():
    doc = Document("a.md", "## A\n\none\n\n## B\n\ntwo")
    first = [c.id for c in chunk_documents([doc])]
    second = [c.id for c in chunk_documents([doc])]
    assert first == second and len(set(first)) == len(first)


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_documents([Document("a", "b")], size=100, overlap=100)


def test_discover_splits_documents_and_tables():
    docs, tables = discover(SAMPLE)
    assert {p.name for p in tables} == {"sales.csv", "support_tickets.csv"}
    assert "leave_policy.md" in {p.name for p in docs}
