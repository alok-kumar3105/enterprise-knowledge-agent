"""Lightweight request tracing and LLM-native metrics.

Every request gets a Trace. Spans record timing for each agent step, and LLM
calls record token usage so we can report tokens/sec and cost-per-request.
The active trace is held in a ContextVar so any code path can attach spans
without threading the object through every function.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_current: ContextVar[Trace | None] = ContextVar("current_trace", default=None)


@dataclass
class Span:
    name: str
    start_ms: float
    duration_ms: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMCall:
    task: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass
class Trace:
    question: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    started: float = field(default_factory=time.perf_counter)
    spans: list[Span] = field(default_factory=list)
    llm_calls: list[LLMCall] = field(default_factory=list)
    total_ms: float = 0.0

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        start = time.perf_counter()
        s = Span(name=name, start_ms=round((start - self.started) * 1000, 2), attrs=dict(attrs))
        try:
            yield s
        except Exception as exc:
            s.attrs["error"] = repr(exc)
            raise
        finally:
            s.duration_ms = round((time.perf_counter() - start) * 1000, 2)
            self.spans.append(s)

    def record_llm(self, call: LLMCall) -> None:
        self.llm_calls.append(call)

    def finish(self) -> None:
        self.total_ms = round((time.perf_counter() - self.started) * 1000, 2)

    # ---- aggregates -------------------------------------------------------
    def summary(self) -> dict[str, Any]:
        inp = sum(c.input_tokens for c in self.llm_calls)
        out = sum(c.output_tokens for c in self.llm_calls)
        llm_ms = sum(c.latency_ms for c in self.llm_calls)
        return {
            "trace_id": self.trace_id,
            "latency_ms": self.total_ms,
            "llm_calls": len(self.llm_calls),
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": inp + out,
            "tokens_per_sec": round(out / (llm_ms / 1000), 1) if llm_ms > 0 else 0.0,
            "cost_usd": round(sum(c.cost_usd for c in self.llm_calls), 6),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            **self.summary(),
            "spans": [asdict(s) for s in self.spans],
            "llm": [asdict(c) for c in self.llm_calls],
        }


def current_trace() -> Trace | None:
    return _current.get()


@contextmanager
def start_trace(question: str) -> Iterator[Trace]:
    trace = Trace(question=question)
    token = _current.set(trace)
    try:
        yield trace
    finally:
        trace.finish()
        _current.reset(token)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span | None]:
    """Open a span on the active trace, or do nothing if there is none."""
    trace = _current.get()
    if trace is None:
        yield None
        return
    with trace.span(name, **attrs) as s:
        yield s


class TraceSink:
    """Appends finished traces to a JSONL file for offline inspection."""

    def __init__(self, directory: Path):
        self.path = Path(directory) / "traces.jsonl"
        self._lock = threading.Lock()

    def write(self, trace: Trace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(trace.to_dict(), ensure_ascii=False)
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class MetricsRegistry:
    """Rolling in-process aggregates exposed on /metrics."""

    def __init__(self, window: int = 1000):
        self.window = window
        self._rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def observe(self, summary: dict[str, Any]) -> None:
        with self._lock:
            self._rows.append(summary)
            if len(self._rows) > self.window:
                self._rows = self._rows[-self.window :]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            rows = list(self._rows)
        return aggregate(rows)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo), 2)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"requests": 0}
    lat = [r["latency_ms"] for r in rows]
    tps = [r["tokens_per_sec"] for r in rows if r["tokens_per_sec"] > 0]
    return {
        "requests": len(rows),
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "avg_tokens_per_request": round(statistics.mean(r["total_tokens"] for r in rows), 1),
        "avg_tokens_per_sec": round(statistics.mean(tps), 1) if tps else 0.0,
        "avg_cost_per_request_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 6),
    }
