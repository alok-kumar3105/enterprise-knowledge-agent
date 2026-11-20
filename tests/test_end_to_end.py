"""End-to-end run on the sample data with the offline mock provider."""
from app.evals import run_eval, write_report
from app.service import KnowledgeAgent

from .conftest import ROOT, SAMPLE


def test_full_pipeline_with_mock_provider(settings, tmp_path):
    agent = KnowledgeAgent(settings)
    report = agent.ingest_directory(SAMPLE)
    assert report.documents == 5 and report.chunks > 10 and not report.skipped
    assert set(report.tables) == {"sales", "support_tickets"}

    out = agent.ask("What is the hotel limit per night in high-cost cities?")
    assert out["sources"][0]["source"] == "expense_policy.md"
    assert "260" in out["answer"]
    assert (settings.trace_dir / "traces.jsonl").exists()

    # Re-ingesting the same folder must not duplicate chunks.
    before = agent.store.count()
    agent.ingest_directory(SAMPLE)
    assert agent.store.count() == before


def test_eval_harness_produces_report(settings, tmp_path):
    agent = KnowledgeAgent(settings, persist_traces=False)
    agent.ingest_directory(SAMPLE)
    results, summary = run_eval(agent, ROOT / "evals" / "dataset.jsonl")
    assert summary["cases"] == len(results) == 16
    assert summary["retrieval_hit_rate"] >= 0.8
    path = write_report(results, summary, tmp_path / "results")
    assert path.exists() and "Evaluation report" in path.read_text(encoding="utf-8")
