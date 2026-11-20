"""Offline evaluation harness.

Each case in the dataset says what a good run looks like:
    {"question": ..., "expected_route": "docs|data|both|direct",
     "expected_sources": ["file.md"], "must_include": ["14 days"]}

Metrics:
* route_accuracy: did the supervisor delegate to the right specialists?
* retrieval_hit_rate: was at least one expected source in the retrieved context?
* answer_keyword_recall: share of required facts present in the answer.
* groundedness: LLM-as-judge score (opt-in with --judge; costs tokens).
* latency p50/p95, tokens and cost per request.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .agents import prompts as P
from .agents.graph import format_contexts
from .tracing import aggregate


@dataclass
class CaseResult:
    question: str
    route: str | None
    route_ok: bool | None
    retrieval_hit: bool | None
    keyword_recall: float | None
    groundedness: float | None
    latency_ms: float
    total_tokens: int
    cost_usd: float
    answer: str
    metrics: dict[str, Any] = field(default_factory=dict)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.lstrip().startswith("//"):
            cases.append(json.loads(line))
    return cases


def _judge(agent, case: dict, result: dict) -> float | None:
    # Rebuild the context the synthesizer saw from the returned sources.
    context = "\n\n".join(f"[{s['ref']}] ({s['citation']})\n{s['snippet']}" for s in result["sources"])
    sql_block = f"SQL: {result['sql']}" if result.get("sql") else ""
    try:
        out = agent.llm.generate_json(
            P.JUDGE_PROMPT.format(context=context or format_contexts([]), sql_block=sql_block,
                                  answer=result["answer"], question=case["question"]),
            system=P.JUDGE_SYSTEM, task="judge")
        return max(0.0, min(1.0, float(out.get("score", 0.0))))
    except (ValueError, TypeError):
        return None


def run_eval(agent, dataset: Path, judge: bool = False) -> tuple[list[CaseResult], dict[str, Any]]:
    results: list[CaseResult] = []
    for case in load_dataset(dataset):
        out = agent.ask(case["question"])
        # Ignore thousands separators so "9,527,800" matches "9527800".
        answer_l = out["answer"].lower().replace(",", "")
        expected_route = case.get("expected_route")
        route_ok = (out["route"] == expected_route) if expected_route else None
        exp_src = case.get("expected_sources") or []
        hit = any(s["source"] in exp_src for s in out["sources"]) if exp_src else None
        must = case.get("must_include") or []
        recall = (sum(1 for k in must if k.lower().replace(",", "") in answer_l) / len(must)) if must else None
        ground = _judge(agent, case, out) if judge else None
        m = out["metrics"]
        results.append(CaseResult(case["question"], out["route"], route_ok, hit, recall, ground,
                                  m["latency_ms"], m["total_tokens"], m["cost_usd"], out["answer"], m))

    def mean(vals):
        vals = [float(v) for v in vals if v is not None]
        return round(statistics.mean(vals), 3) if vals else None

    summary = {
        "cases": len(results),
        "provider": agent.settings.llm_provider,
        "model": agent.llm.model,
        "route_accuracy": mean(r.route_ok for r in results),
        "retrieval_hit_rate": mean(r.retrieval_hit for r in results),
        "answer_keyword_recall": mean(r.keyword_recall for r in results),
        "groundedness": mean(r.groundedness for r in results),
        **aggregate([r.metrics for r in results]),
    }
    return results, summary


def write_report(results: list[CaseResult], summary: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"eval-{summary['provider']}-{stamp}.md"

    def fmt(v):
        if v is None:
            return "–"
        if isinstance(v, bool):
            return "✅" if v else "❌"
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    lines = [f"# Evaluation report ({summary['provider']} / {summary['model']})", "",
             f"Generated {stamp} UTC over {summary['cases']} cases.", "", "## Summary", "",
             "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {fmt(v)} |" for k, v in summary.items() if k not in {"provider", "model"}]
    lines += ["", "## Cases", "",
              "| Question | Route | Route ok | Retrieval hit | Keyword recall | Grounded | Latency ms |",
              "|---|---|---|---|---|---|---|"]
    for r in results:
        q = r.question.replace("|", "\\|")
        lines.append(f"| {q} | {r.route} | {fmt(r.route_ok)} | {fmt(r.retrieval_hit)} | {fmt(r.keyword_recall)} "
                     f"| {fmt(r.groundedness)} | {r.latency_ms:.0f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "latest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path
