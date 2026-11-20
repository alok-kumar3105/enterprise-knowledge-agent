"""Behavioural tests for the multi-agent graph, driven by a scripted LLM."""

ROUTE_DOCS = {"route": "docs", "search_query": "parental leave secondary caregiver", "reason": "policy"}
ROUTE_DATA = {"route": "data", "search_query": "open tickets", "reason": "numbers"}
ROUTE_BOTH = {"route": "both", "search_query": "Orion-D warranty", "reason": "policy + numbers"}
PASS = {"verdict": "pass", "issues": [], "followup_query": None}


def test_docs_route_retrieves_answers_and_cites(make_agent):
    agent = make_agent({"router": [ROUTE_DOCS], "answer": ["Secondary caregivers get 6 weeks [1]."],
                        "critic": [PASS]})
    out = agent.ask("How much parental leave does a secondary caregiver get?")
    assert out["route"] == "docs"
    assert out["steps"] == ["router:docs", "retriever", "synthesizer", "critic:pass", "finalize"]
    assert out["sources"][0]["source"] == "leave_policy.md" and out["sources"][0]["cited"]
    assert "sql" not in agent.llm.inner.tasks()


def test_data_route_runs_sql_and_passes_result_to_synthesizer(make_agent):
    sql = "SELECT COUNT(*) AS open_tickets FROM support_tickets WHERE product='Orion-D' AND status='open'"
    agent = make_agent({"router": [ROUTE_DATA], "sql": [{"sql": sql, "explanation": ""}],
                        "answer": ["There are 7 open Orion-D tickets [SQL]."], "critic": [PASS]})
    out = agent.ask("How many Orion-D tickets are open?")
    assert out["route"] == "data" and out["sql"] == sql and out["sql_error"] is None
    answer_prompt = next(p for t, p in agent.llm.inner.calls if t == "answer")
    assert "open_tickets\n7" in answer_prompt
    assert "retriever" not in out["steps"]


def test_both_route_uses_retriever_then_sql(make_agent):
    agent = make_agent({"router": [ROUTE_BOTH],
                        "sql": [{"sql": "SELECT COUNT(*) FROM support_tickets WHERE product='Orion-D'"}],
                        "answer": ["3-year warranty [1]; 40 tickets [SQL]."], "critic": [PASS]})
    out = agent.ask("Orion-D warranty and ticket count?")
    assert out["steps"][:3] == ["router:both", "retriever", "sql_analyst:ok"]


def test_sql_analyst_retries_with_error_feedback(make_agent):
    agent = make_agent({"router": [ROUTE_DATA],
                        "sql": [{"sql": "SELECT nope FROM missing_table"},
                                {"sql": "DROP TABLE sales"},
                                {"sql": "SELECT COUNT(*) FROM sales"}],
                        "answer": ["36 rows [SQL]."], "critic": [PASS]})
    out = agent.ask("How many sales rows?")
    assert out["sql"] == "SELECT COUNT(*) FROM sales" and out["sql_error"] is None
    sql_prompts = [p for t, p in agent.llm.inner.calls if t == "sql"]
    assert "no such table" in sql_prompts[1]
    assert "rejected by safety guard" in sql_prompts[2]
    assert agent.sql.query("SELECT COUNT(*) FROM sales").rows == [(36,)]


def test_sql_failure_is_reported_not_raised(make_agent):
    bad = {"sql": "SELECT nope FROM missing"}
    agent = make_agent({"router": [ROUTE_DATA], "sql": [bad, bad, bad],
                        "answer": ["I could not compute this."], "critic": [PASS]})
    out = agent.ask("Impossible metric?")
    assert out["sql_error"] and "sql_analyst:failed" in out["steps"]


def test_critic_followup_triggers_corrective_retrieval(make_agent):
    revise = {"verdict": "revise", "issues": ["missing hotel limit"], "followup_query": "hotel nightly limit"}
    agent = make_agent({"router": [ROUTE_DOCS], "answer": ["Draft one.", "260 USD per night [2]."],
                        "critic": [revise, PASS]})
    out = agent.ask("What is the hotel limit?")
    assert out["steps"] == ["router:docs", "retriever", "synthesizer", "critic:revise",
                            "retriever", "synthesizer", "critic:pass", "finalize"]
    assert any(s["source"] == "expense_policy.md" for s in out["sources"])
    second_answer_prompt = [p for t, p in agent.llm.inner.calls if t == "answer"][1]
    assert "missing hotel limit" in second_answer_prompt and "Draft one." in second_answer_prompt


def test_revision_budget_is_enforced(make_agent, settings):
    revise = {"verdict": "revise", "issues": ["still wrong"], "followup_query": None}
    agent = make_agent({"router": [ROUTE_DOCS], "critic": [revise] * 10})
    out = agent.ask("Anything about leave?")
    assert out["revisions"] == settings.max_revisions + 1
    assert out["steps"].count("synthesizer") == settings.max_revisions + 1
    assert out["steps"][-1] == "finalize"


def test_router_garbage_falls_back_to_docs(make_agent):
    agent = make_agent({"router": ["not json at all", ], "critic": [PASS]})
    out = agent.ask("What is the password length?")
    assert out["route"] == "docs"


def test_data_route_without_tables_falls_back_to_docs(make_agent):
    agent = make_agent({"router": [ROUTE_DATA], "critic": [PASS]}, ingest=False)
    out = agent.ask("How many tickets?")
    assert out["route"] == "docs"


def test_direct_route_skips_retrieval_and_critic_llm(make_agent):
    agent = make_agent({"router": [{"route": "direct", "search_query": "hi"}], "direct": ["Hello!"]})
    out = agent.ask("hi")
    assert out["answer"] == "Hello!" and out["sources"] == []
    assert "critic" not in agent.llm.inner.tasks()


def test_metrics_capture_every_llm_call(make_agent):
    agent = make_agent({"router": [ROUTE_DOCS], "answer": ["x [1]"], "critic": [PASS]})
    out = agent.ask("leave?")
    m = out["metrics"]
    assert m["llm_calls"] == 3 and m["input_tokens"] == 300 and m["output_tokens"] == 60
    assert m["cost_usd"] > 0 and m["latency_ms"] > 0
    assert {s["name"] for s in out["trace"]} >= {"agent.router", "agent.retriever", "agent.critic"}
    assert agent.metrics.snapshot()["requests"] == 1
