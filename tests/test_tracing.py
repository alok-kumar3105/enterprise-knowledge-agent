from app.llm import LLMResponse, TracedLLM, parse_json
from app.tracing import aggregate, percentile, span, start_trace


class _Fixed:
    model = "fixed"

    def generate(self, prompt, *, system=None, json_mode=False, task="generate"):
        return LLMResponse(text='```json\n{"ok": true}\n```', input_tokens=1_000, output_tokens=500)

    def embed(self, texts, *, kind="document"):
        return [[1.0] for _ in texts]


def test_llm_calls_record_tokens_and_cost_on_trace():
    llm = TracedLLM(_Fixed(), price_in_per_m=0.30, price_out_per_m=2.50)
    with start_trace("q") as trace, span("step", x=1):
        assert llm.generate_json("hi", task="router") == {"ok": True}
    s = trace.summary()
    assert s["llm_calls"] == 1 and s["input_tokens"] == 1000 and s["output_tokens"] == 500
    assert abs(s["cost_usd"] - (1000 * 0.30 + 500 * 2.50) / 1e6) < 1e-9
    assert trace.spans[0].name == "step" and trace.spans[0].attrs == {"x": 1}
    assert trace.llm_calls[0].task == "router"


def test_calls_outside_a_trace_are_not_recorded():
    llm = TracedLLM(_Fixed(), 1, 1)
    llm.generate("hi")  # must not raise


def test_span_records_errors():
    import pytest

    with start_trace("q") as trace, pytest.raises(RuntimeError), span("boom"):
        raise RuntimeError("x")
    assert "RuntimeError" in trace.spans[0].attrs["error"]


def test_percentiles_and_aggregate():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10, 20], 95) == 19.5
    rows = [{"latency_ms": v, "tokens_per_sec": 10.0, "total_tokens": 100, "cost_usd": 0.001} for v in (100, 200)]
    agg = aggregate(rows)
    assert agg["requests"] == 2 and agg["latency_p50_ms"] == 150 and agg["total_cost_usd"] == 0.002


def test_parse_json_tolerates_prose():
    assert parse_json('Sure! {"route": "docs"} hope this helps') == {"route": "docs"}
