"""LLM and embedding clients.

`GeminiClient` talks to Gemini through the google-genai SDK, using either an
AI Studio API key or Vertex AI on Google Cloud. `MockLLM` is a deterministic
offline stand-in so the whole pipeline runs (and is testable) without
credentials. `TracedLLM` wraps any client and records latency, tokens and
cost on the active trace.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from .config import Settings
from .tracing import LLMCall, current_trace

EmbedKind = Literal["document", "query"]


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    model: str

    def generate(
        self, prompt: str, *, system: str | None = None, json_mode: bool = False, task: str = "generate"
    ) -> LLMResponse: ...

    def embed(self, texts: list[str], *, kind: EmbedKind = "document") -> list[list[float]]: ...


# --------------------------------------------------------------------------
# Gemini
# --------------------------------------------------------------------------
class GeminiClient:
    EMBED_BATCH = 100

    def __init__(self, settings: Settings):
        from google import genai
        from google.genai import types

        # Fail fast instead of hanging when a model is overloaded: each request
        # gets a timeout, and transient errors (429/5xx) are retried a few times.
        http = types.HttpOptions(
            timeout=settings.llm_timeout_s * 1000,
            retry_options=types.HttpRetryOptions(attempts=settings.llm_retries),
        )
        if settings.use_vertex_ai:
            if not settings.gcp_project:
                raise ValueError("GCP_PROJECT is required when USE_VERTEX_AI=true")
            self._client = genai.Client(vertexai=True, project=settings.gcp_project,
                                        location=settings.gcp_location, http_options=http)
        else:
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY is required when LLM_PROVIDER=gemini (or set USE_VERTEX_AI=true)")
            self._client = genai.Client(api_key=settings.google_api_key, http_options=http)
        self.model = settings.chat_model
        self.thinking_level = settings.thinking_level.strip().upper() or None
        self.embedding_model = settings.embedding_model
        self.embedding_dim = settings.embedding_dim

    def generate(self, prompt, *, system=None, json_mode=False, task="generate") -> LLMResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.0,
            response_mime_type="application/json" if json_mode else None,
            thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None,
        )
        resp = self._client.models.generate_content(model=self.model, contents=prompt, config=config)
        usage = resp.usage_metadata
        # Thinking models spend hidden "thought" tokens that are billed as
        # output, so include them or cost-per-request is under-reported.
        output = ((usage.candidates_token_count or 0) + (getattr(usage, "thoughts_token_count", None) or 0)
                  if usage else 0)
        return LLMResponse(
            text=resp.text or "",
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            output_tokens=output,
        )

    def embed(self, texts, *, kind="document") -> list[list[float]]:
        from google.genai import types

        task_type = "RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT"
        out: list[list[float]] = []
        for i in range(0, len(texts), self.EMBED_BATCH):
            batch = texts[i : i + self.EMBED_BATCH]
            resp = self._client.models.embed_content(
                model=self.embedding_model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.embedding_dim),
            )
            out.extend(_normalize(list(e.values)) for e in resp.embeddings)
        return out


# --------------------------------------------------------------------------
# Offline mock
# --------------------------------------------------------------------------
_STOP = frozenset(
    "a an the of to in on for and or is are was were be been with by at from as it this that what which who how "
    "do does did can could should would will i we you our your my me their they them its not no if than then so "
    "about into per much many any all".split()
)
# The mock routes to "data" only when the question names something that lives
# in the tables; aggregation words alone ("how many days of leave") are not enough.
_DATA_WORDS = {"revenue", "units sold", "sales", "ticket", "region"}
_DOC_WORDS = {"policy", "allowed", "can i", "process", "rule", "eligible", "leave", "expense", "hotel", "flight",
              "security", "password", "onboarding", "warranty", "premium support", "response time",
              "first-response", "reimburse", "reviewer", "battery"}


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in _STOP]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class MockLLM:
    """Deterministic, credential-free LLM used for local demos and tests.

    It is intentionally simple: keyword routing, a safe default SQL query,
    extractive answers built from the retrieved context, and a critic that
    always passes. Swap LLM_PROVIDER=gemini for real behaviour.
    """

    def __init__(self, embedding_dim: int = 256):
        self.model = "mock"
        self.embedding_dim = embedding_dim

    # ---- generation -------------------------------------------------------
    def generate(self, prompt, *, system=None, json_mode=False, task="generate") -> LLMResponse:
        handler = getattr(self, f"_task_{task}", None)
        text = handler(prompt) if handler else "OK"
        return LLMResponse(text=text, input_tokens=_estimate_tokens((system or "") + prompt),
                           output_tokens=_estimate_tokens(text))

    @staticmethod
    def _question(prompt: str) -> str:
        m = re.findall(r"^Question:\s*(.+)$", prompt, flags=re.MULTILINE)
        return m[-1].strip() if m else prompt

    def _task_router(self, prompt: str) -> str:
        q = self._question(prompt).lower()
        data = any(w in q for w in _DATA_WORDS)
        docs = any(w in q for w in _DOC_WORDS)
        route = "both" if data and docs else "data" if data else "docs"
        if len(tokenize(q)) <= 1 and q.strip() in {"hi", "hello", "hey", "thanks"}:
            route = "direct"
        return json.dumps({"route": route, "search_query": q, "reason": "keyword heuristic (mock)"})

    def _task_sql(self, prompt: str) -> str:
        tables = re.findall(r"^TABLE (\w+)", prompt, flags=re.MULTILINE)
        q = set(tokenize(self._question(prompt)))
        best = max(tables, key=lambda t: len(q & set(tokenize(t.replace("_", " ")))), default=None)
        sql = f"SELECT * FROM {best} LIMIT 10" if best else "SELECT 1"
        return json.dumps({"sql": sql, "explanation": "mock query"})

    def _task_answer(self, prompt: str) -> str:
        q = set(tokenize(self._question(prompt)))
        blocks = re.findall(r"^\[(\d+)\] \(([^)]*)\)\n(.*?)(?=^\[\d+\] \(|^SQL RESULT|^Question:|\Z)",
                            prompt, flags=re.MULTILINE | re.DOTALL)
        scored = []
        for idx, _src, body in blocks:
            for sent in re.split(r"(?<=[.!?])\s+", body.strip()):
                overlap = len(q & set(tokenize(sent)))
                if overlap:
                    scored.append((overlap, sent.strip(), idx))
        scored.sort(key=lambda x: -x[0])
        parts = [f"{s} [{i}]" for _, s, i in scored[:2]]
        sql = re.search(r"^SQL RESULT[^\n]*\n(.*?)(?=^Question:|\Z)", prompt, flags=re.MULTILINE | re.DOTALL)
        if sql and sql.group(1).strip():
            parts.append("From the data: " + sql.group(1).strip().splitlines()[0][:300] + " [SQL]")
        return " ".join(parts) or "I could not find this in the knowledge base."

    def _task_critic(self, prompt: str) -> str:
        return json.dumps({"verdict": "pass", "issues": [], "followup_query": None})

    def _task_judge(self, prompt: str) -> str:
        return json.dumps({"score": 1.0, "reason": "mock judge"})

    # ---- embeddings -------------------------------------------------------
    def embed(self, texts, *, kind="document") -> list[list[float]]:
        """Hashing-trick bag of words: crude, but semantically non-trivial."""
        out = []
        for text in texts:
            vec = [0.0] * self.embedding_dim
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                vec[h % self.embedding_dim] += 1.0 if (h >> 8) % 2 else -1.0
            out.append(_normalize(vec))
        return out


# --------------------------------------------------------------------------
# Tracing wrapper
# --------------------------------------------------------------------------
class TracedLLM:
    """Records latency, tokens and cost of every call on the active trace."""

    def __init__(self, inner: LLMClient, price_in_per_m: float, price_out_per_m: float):
        self.inner = inner
        self.model = inner.model
        self.price_in = price_in_per_m
        self.price_out = price_out_per_m

    def generate(self, prompt, *, system=None, json_mode=False, task="generate") -> LLMResponse:
        start = time.perf_counter()
        resp = self.inner.generate(prompt, system=system, json_mode=json_mode, task=task)
        latency = (time.perf_counter() - start) * 1000
        trace = current_trace()
        if trace is not None:
            cost = (resp.input_tokens * self.price_in + resp.output_tokens * self.price_out) / 1_000_000
            trace.record_llm(LLMCall(task, self.model, resp.input_tokens, resp.output_tokens,
                                     round(latency, 2), round(cost, 8)))
        return resp

    def generate_json(self, prompt, *, system=None, task="generate") -> dict[str, Any]:
        resp = self.generate(prompt, system=system, json_mode=True, task=task)
        return parse_json(resp.text)

    def embed(self, texts, *, kind="document"):
        return self.inner.embed(texts, kind=kind)


def parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences and prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        val = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Model did not return JSON: {text[:200]!r}") from None
        val = json.loads(m.group(0))
    if not isinstance(val, dict):
        raise ValueError(f"Expected a JSON object, got {type(val).__name__}")
    return val


def build_llm(settings: Settings) -> TracedLLM:
    inner: LLMClient
    if settings.llm_provider == "gemini":
        inner = GeminiClient(settings)
    else:
        inner = MockLLM(embedding_dim=settings.embedding_dim)
    return TracedLLM(inner, settings.price_input_per_m, settings.price_output_per_m)
