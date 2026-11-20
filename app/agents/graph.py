"""The multi-agent workflow, built as a LangGraph state machine.

    START -> router --docs/both--> retriever --both--> sql_analyst --> synthesizer
                    --data-------------------------->  sql_analyst
                    --direct-------------------------------------->   synthesizer
    synthesizer -> critic --pass / budget spent--> finalize -> END
                          --revise + better query--> retriever (corrective RAG)
                          --revise-----------------> synthesizer (self-reflection)

* router: a supervisor that delegates to specialist agents (hierarchical delegation).
* retriever: hybrid dense + BM25 search over the document index.
* sql_analyst: writes SQL, executes it, and retries using the error message
  (an act -> observe -> retry loop) under read-only guards.
* synthesizer: writes a grounded, cited answer.
* critic: self-reflection step that checks grounding and citations and can
  send the draft back for revision or trigger a new retrieval.
"""
from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..data.sql_store import SQLStore, UnsafeQueryError
from ..llm import TracedLLM
from ..retrieval.hybrid import HybridRetriever, Retrieved
from ..tracing import span
from . import prompts as P

ROUTES = {"docs", "data", "both", "direct"}


class AgentState(TypedDict, total=False):
    question: str
    route: str
    search_query: str
    followup_query: str | None
    contexts: list[Retrieved]
    sql_attempted: bool
    sql: str | None
    sql_result: str | None
    sql_error: str | None
    draft: str
    critique: dict[str, Any]
    revisions: int
    answer: str
    steps: list[str]


# --------------------------------------------------------------------------
# Prompt helpers
# --------------------------------------------------------------------------
def format_contexts(contexts: list[Retrieved]) -> str:
    if not contexts:
        return "CONTEXT: (no document context)"
    return "\n\n".join(f"[{i}] ({r.chunk.citation})\n{r.chunk.text}" for i, r in enumerate(contexts, start=1))


def format_sql(state: AgentState) -> str:
    if state.get("sql_result"):
        return f"\nSQL RESULT (query: {state.get('sql')}):\n{state['sql_result']}\n"
    if state.get("sql_error"):
        return f"\nSQL RESULT: unavailable ({state['sql_error']})\n"
    return ""


def cited_indices(answer: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_graph(llm: TracedLLM, retriever: HybridRetriever, sql_store: SQLStore, settings: Settings):
    def step(state: AgentState, name: str) -> list[str]:
        return [*state.get("steps", []), name]

    # ---- router (supervisor) ---------------------------------------------
    def router(state: AgentState) -> AgentState:
        with span("agent.router") as s:
            tables = ", ".join(sql_store.tables()) or "(none)"
            try:
                out = llm.generate_json(P.ROUTER_PROMPT.format(tables=tables, question=state["question"]),
                                        system=P.ROUTER_SYSTEM, task="router")
                route = str(out.get("route", "docs")).lower()
                query = str(out.get("search_query") or state["question"])
            except ValueError:
                route, query = "docs", state["question"]
            if route not in ROUTES:
                route = "docs"
            if route in {"data", "both"} and not sql_store.tables():
                route = "docs"  # no structured data loaded; fall back to documents
            if s:
                s.attrs.update(route=route, search_query=query)
        return {"route": route, "search_query": query, "steps": step(state, f"router:{route}")}

    # ---- retriever ---------------------------------------------------------
    def retrieve(state: AgentState) -> AgentState:
        query = state.get("followup_query") or state.get("search_query") or state["question"]
        with span("agent.retriever", query=query) as s:
            results = retriever.search(query, k=settings.top_k)
            merged = list(state.get("contexts", []))
            seen = {r.chunk.id for r in merged}
            merged += [r for r in results if r.chunk.id not in seen]
            merged = merged[: settings.top_k * 2]
            if s:
                s.attrs.update(hits=len(results), sources=sorted({r.chunk.source for r in results}))
        return {"contexts": merged, "followup_query": None, "steps": step(state, "retriever")}

    # ---- SQL analyst (act -> observe -> retry) -----------------------------
    def sql_analyst(state: AgentState) -> AgentState:
        with span("agent.sql_analyst") as s:
            schema = sql_store.schema_description()
            sql, error, result = None, None, None
            for attempt in range(1, settings.max_sql_attempts + 1):  # noqa: B007 (read after loop)
                feedback = P.SQL_FEEDBACK.format(sql=sql, error=error) if error else ""
                try:
                    out = llm.generate_json(P.SQL_PROMPT.format(schema=schema, feedback=feedback,
                                                                question=state["question"]),
                                            system=P.SQL_SYSTEM, task="sql")
                    sql = str(out.get("sql", "")).strip()
                    result = sql_store.query(sql).to_text()
                    error = None
                    break
                except UnsafeQueryError as exc:
                    error = f"rejected by safety guard: {exc}"
                except Exception as exc:  # SQL errors, bad JSON: feed back and retry
                    error = str(exc)
            if s:
                s.attrs.update(attempts=attempt, sql=sql, ok=error is None)
        return {"sql_attempted": True, "sql": sql, "sql_result": result, "sql_error": error,
                "steps": step(state, f"sql_analyst:{'ok' if error is None else 'failed'}")}

    # ---- synthesizer -------------------------------------------------------
    def synthesize(state: AgentState) -> AgentState:
        with span("agent.synthesizer", revision=state.get("revisions", 0)):
            if state.get("route") == "direct":
                resp = llm.generate(f"Question: {state['question']}", system=P.DIRECT_SYSTEM, task="direct")
                return {"draft": resp.text.strip(), "steps": step(state, "synthesizer")}
            critique = state.get("critique") or {}
            feedback = ""
            if critique.get("verdict") == "revise":
                issues = "\n".join(f"- {i}" for i in critique.get("issues", [])) or "- unspecified"
                feedback = P.ANSWER_FEEDBACK.format(issues=issues, draft=state.get("draft", ""))
            prompt = P.ANSWER_PROMPT.format(context=format_contexts(state.get("contexts", [])),
                                            sql_block=format_sql(state), feedback=feedback,
                                            question=state["question"])
            resp = llm.generate(prompt, system=P.ANSWER_SYSTEM, task="answer")
        return {"draft": resp.text.strip(), "steps": step(state, "synthesizer")}

    # ---- critic (self-reflection) -----------------------------------------
    def critic(state: AgentState) -> AgentState:
        if state.get("route") == "direct":
            return {"critique": {"verdict": "pass", "issues": []}}
        with span("agent.critic") as s:
            prompt = P.CRITIC_PROMPT.format(context=format_contexts(state.get("contexts", [])),
                                            sql_block=format_sql(state), draft=state.get("draft", ""),
                                            question=state["question"])
            try:
                out = llm.generate_json(prompt, system=P.CRITIC_SYSTEM, task="critic")
            except ValueError:
                out = {"verdict": "pass", "issues": ["critic output unparseable; accepted draft"]}
            verdict = "revise" if str(out.get("verdict", "pass")).lower() == "revise" else "pass"
            followup = out.get("followup_query") or None
            critique = {"verdict": verdict, "issues": list(out.get("issues") or []),
                        "followup_query": followup}
            if s:
                s.attrs.update(verdict=verdict, issues=len(critique["issues"]))
        revisions = state.get("revisions", 0) + (1 if verdict == "revise" else 0)
        update: AgentState = {"critique": critique, "revisions": revisions, "steps": step(state, f"critic:{verdict}")}
        if verdict == "revise" and followup and state.get("route") in {"docs", "both"}:
            update["followup_query"] = str(followup)
        return update

    def finalize(state: AgentState) -> AgentState:
        return {"answer": state.get("draft", ""), "steps": step(state, "finalize")}

    # ---- edges -------------------------------------------------------------
    def after_router(state: AgentState) -> str:
        return {"docs": "retriever", "both": "retriever", "data": "sql_analyst"}.get(state["route"], "synthesizer")

    def after_retrieve(state: AgentState) -> str:
        if state["route"] == "both" and not state.get("sql_attempted"):
            return "sql_analyst"
        return "synthesizer"

    def after_critic(state: AgentState) -> str:
        critique = state.get("critique", {})
        if critique.get("verdict") != "revise":
            return "finalize"
        # The critic increments `revisions` on each rejection, so stop once
        # the revision budget is spent and ship the best draft we have.
        if state.get("revisions", 0) > settings.max_revisions:
            return "finalize"
        return "retriever" if state.get("followup_query") else "synthesizer"

    g = StateGraph(AgentState)
    g.add_node("router", router)
    g.add_node("retriever", retrieve)
    g.add_node("sql_analyst", sql_analyst)
    g.add_node("synthesizer", synthesize)
    g.add_node("critic", critic)
    g.add_node("finalize", finalize)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router, ["retriever", "sql_analyst", "synthesizer"])
    g.add_conditional_edges("retriever", after_retrieve, ["sql_analyst", "synthesizer"])
    g.add_edge("sql_analyst", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", after_critic, ["finalize", "retriever", "synthesizer"])
    g.add_edge("finalize", END)
    return g.compile()
