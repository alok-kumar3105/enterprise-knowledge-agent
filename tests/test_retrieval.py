from app.ingest.chunker import Chunk
from app.llm import MockLLM, TracedLLM
from app.retrieval.hybrid import BM25, HybridRetriever, rrf
from app.retrieval.vector_store import LocalVectorStore


def _chunks():
    return [
        Chunk(id="a", source="leave.md", text="Employees get 24 days of annual leave.", section="Annual Leave"),
        Chunk(id="b", source="expense.md", text="Hotel limit is 260 USD in high-cost cities.", section="Hotels"),
        Chunk(id="c", source="security.md", text="Passwords need 14 characters and MFA.", section="Passwords"),
    ]


def test_local_store_persists_and_reloads(tmp_path):
    llm = MockLLM(embedding_dim=64)
    chunks = _chunks()
    store = LocalVectorStore(tmp_path)
    store.upsert(chunks, llm.embed([c.text for c in chunks]))
    reloaded = LocalVectorStore(tmp_path)
    assert reloaded.count() == 3
    top, score = reloaded.search(llm.embed(["annual leave days"], kind="query")[0], k=1)[0]
    assert top.id == "a" and score > 0


def test_upsert_replaces_same_id_and_delete_source(tmp_path):
    llm = MockLLM(embedding_dim=64)
    store = LocalVectorStore(tmp_path)
    chunks = _chunks()
    store.upsert(chunks, llm.embed([c.text for c in chunks]))
    store.upsert(chunks[:1], llm.embed([chunks[0].text]))
    assert store.count() == 3
    assert store.delete_source("expense.md") == 1
    assert {c.source for c in store.all_chunks()} == {"leave.md", "security.md"}


def test_dimension_change_is_rejected(tmp_path):
    import pytest

    store = LocalVectorStore(tmp_path)
    store.upsert(_chunks()[:1], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        store.upsert(_chunks()[1:2], [[1.0, 0.0, 0.0]])


def test_bm25_ranks_exact_terms():
    bm = BM25([["hotel", "limit", "260"], ["annual", "leave"], ["password", "mfa"]])
    scores = bm.scores(["260"])
    assert scores.index(max(scores)) == 0 and scores[1] == 0


def test_rrf_rewards_agreement():
    fused = rrf([["x", "y", "z"], ["y", "x"]])
    assert set(fused) == {"x", "y", "z"}
    assert fused["z"] < fused["x"] and fused["z"] < fused["y"]


def test_hybrid_search_returns_ranked_results_with_both_signals():
    llm = TracedLLM(MockLLM(embedding_dim=64), 0, 0)
    store = LocalVectorStore()
    chunks = _chunks()
    store.upsert(chunks, llm.embed([c.text for c in chunks]))
    retriever = HybridRetriever(store, llm)
    hits = retriever.search("what is the hotel limit in high-cost cities", k=2)
    assert hits[0].chunk.id == "b"
    assert hits[0].dense_rank is not None and hits[0].sparse_rank is not None


def test_bm25_index_refreshes_after_ingest():
    llm = TracedLLM(MockLLM(embedding_dim=64), 0, 0)
    store = LocalVectorStore()
    retriever = HybridRetriever(store, llm)
    assert retriever.search("anything") == []
    c = _chunks()
    store.upsert(c, llm.embed([x.text for x in c]))
    assert retriever.search("passwords MFA", k=1)[0].chunk.id == "c"
