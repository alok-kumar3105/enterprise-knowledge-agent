"""Facade that wires storage, retrieval, agents and observability together."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agents.graph import build_graph, cited_indices
from .config import Settings
from .data.sql_store import SQLStore
from .ingest.chunker import chunk_documents
from .ingest.loaders import DOC_EXTENSIONS, TABLE_EXTENSIONS, discover, load_file
from .llm import TracedLLM, build_llm
from .retrieval.hybrid import HybridRetriever
from .retrieval.vector_store import LocalVectorStore, VectorStore
from .tracing import MetricsRegistry, TraceSink, span, start_trace


@dataclass
class IngestReport:
    documents: int = 0
    chunks: int = 0
    tables: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"documents": self.documents, "chunks": self.chunks, "tables": self.tables, "skipped": self.skipped}


def _make_store(settings: Settings) -> VectorStore:
    if settings.vector_backend == "chroma":
        from .retrieval.vector_store import ChromaVectorStore

        return ChromaVectorStore(settings.data_dir / "chroma")
    return LocalVectorStore(settings.data_dir / "index")


class KnowledgeAgent:
    def __init__(self, settings: Settings, llm: TracedLLM | None = None,
                 store: VectorStore | None = None, sql_store: SQLStore | None = None,
                 persist_traces: bool = True):
        self.settings = settings
        self.llm = llm or build_llm(settings)
        self.store = store or _make_store(settings)
        self.sql = sql_store or SQLStore(settings.data_dir / "warehouse.sqlite", row_limit=settings.sql_row_limit)
        self.retriever = HybridRetriever(self.store, self.llm)
        self.graph = build_graph(self.llm, self.retriever, self.sql, settings)
        self.metrics = MetricsRegistry()
        self.sink = TraceSink(settings.trace_dir) if persist_traces else None

    # ---- ingestion pipeline ----------------------------------------------
    def ingest_file(self, path: Path, report: IngestReport | None = None) -> IngestReport:
        report = report or IngestReport()
        path = Path(path)
        ext = path.suffix.lower()
        try:
            if ext in TABLE_EXTENSIONS:
                table, rows = self.sql.load_csv(path)
                report.tables[table] = rows
            elif ext in DOC_EXTENSIONS:
                docs = load_file(path)
                chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
                # Re-ingesting a file replaces its old chunks instead of duplicating them.
                self.store.delete_source(path.name)
                if chunks:
                    vectors = self.llm.embed([f"{c.section}\n{c.text}" for c in chunks], kind="document")
                    self.store.upsert(chunks, vectors)
                report.documents += 1
                report.chunks += len(chunks)
            else:
                report.skipped.append(f"{path.name}: unsupported type")
        except Exception as exc:
            report.skipped.append(f"{path.name}: {exc}")
        return report

    def ingest_directory(self, directory: Path) -> IngestReport:
        report = IngestReport()
        docs, tables = discover(directory)
        for p in [*docs, *tables]:
            self.ingest_file(p, report)
        return report

    def reset(self) -> None:
        self.store.clear()
        self.sql.drop_all()

    # ---- question answering ----------------------------------------------
    def ask(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        with start_trace(question) as trace, span("graph.invoke"):
            state = self.graph.invoke({"question": question, "revisions": 0, "steps": []})
        summary = trace.summary()
        self.metrics.observe(summary)
        if self.sink:
            self.sink.write(trace)

        answer = state.get("answer", "")
        contexts = state.get("contexts", [])
        cited = set(cited_indices(answer))
        sources = [
            {"ref": i, "citation": r.chunk.citation, "source": r.chunk.source, "score": r.score,
             "cited": i in cited, "snippet": r.chunk.text[:240]}
            for i, r in enumerate(contexts, start=1)
        ]
        return {
            "answer": answer,
            "route": state.get("route"),
            "sources": sources,
            "sql": state.get("sql"),
            "sql_error": state.get("sql_error"),
            "critique": state.get("critique"),
            "revisions": state.get("revisions", 0),
            "steps": state.get("steps", []),
            "metrics": summary,
            "trace": trace.to_dict()["spans"],
        }

    def stats(self) -> dict[str, Any]:
        return {"chunks": self.store.count(), "tables": self.sql.tables(),
                "provider": self.settings.llm_provider, "model": self.llm.model}
