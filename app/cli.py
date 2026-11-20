"""Command-line entry point.

    python -m app.cli ingest sample_data --reset
    python -m app.cli ask "How many days of parental leave do employees get?"
    python -m app.cli eval --judge
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import get_settings
from .evals import run_eval, write_report
from .service import KnowledgeAgent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="knowledge-agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ing = sub.add_parser("ingest", help="ingest a folder of documents (.pdf/.md/.txt) and tables (.csv)")
    p_ing.add_argument("path", type=Path)
    p_ing.add_argument("--reset", action="store_true", help="clear the existing index first")

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--json", action="store_true", help="print the full JSON response")

    p_eval = sub.add_parser("eval", help="run the evaluation dataset")
    p_eval.add_argument("--dataset", type=Path, default=Path("evals/dataset.jsonl"))
    p_eval.add_argument("--judge", action="store_true", help="also score groundedness with an LLM judge")
    p_eval.add_argument("--out", type=Path, default=Path("evals/results"))

    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    agent = KnowledgeAgent(get_settings())

    if args.cmd == "ingest":
        if args.reset:
            agent.reset()
        report = agent.ingest_directory(args.path)
        print(json.dumps(report.to_dict(), indent=2))
        return 1 if report.skipped and not (report.documents or report.tables) else 0

    if args.cmd == "ask":
        out = agent.ask(args.question)
        if args.json:
            print(json.dumps(out, indent=2, default=str))
            return 0
        print(out["answer"], "\n")
        for s in out["sources"]:
            print(f"  [{s['ref']}] {s['citation']}{'  (cited)' if s['cited'] else ''}")
        if out["sql"]:
            print(f"\n  SQL: {out['sql']}")
        m = out["metrics"]
        print(f"\n  route={out['route']}  steps={' -> '.join(out['steps'])}")
        print(f"  latency={m['latency_ms']}ms  tokens={m['total_tokens']}  "
              f"tok/s={m['tokens_per_sec']}  cost=${m['cost_usd']}")
        return 0

    if args.cmd == "eval":
        results, summary = run_eval(agent, args.dataset, judge=args.judge)
        path = write_report(results, summary, args.out)
        print(json.dumps(summary, indent=2))
        print(f"\nReport written to {path}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
