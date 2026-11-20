from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from app.config import Settings
from app.llm import LLMResponse, MockLLM, TracedLLM
from app.service import KnowledgeAgent

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data"


class ScriptedLLM:
    """Test double: returns queued responses per task, falling back to MockLLM.

    Responses may be strings, dicts (serialized to JSON) or callables that
    receive the prompt. Every call is recorded for assertions.
    """

    def __init__(self, script: dict[str, list] | None = None):
        self.model = "scripted"
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[tuple[str, str]] = []
        self._fallback = MockLLM(embedding_dim=64)

    def generate(self, prompt, *, system=None, json_mode=False, task="generate"):
        self.calls.append((task, prompt))
        queue = self.script.get(task)
        if queue:
            item = queue.pop(0)
            if isinstance(item, Callable):
                item = item(prompt)
            text = json.dumps(item) if isinstance(item, dict) else str(item)
            return LLMResponse(text=text, input_tokens=100, output_tokens=20)
        return self._fallback.generate(prompt, system=system, json_mode=json_mode, task=task)

    def embed(self, texts, *, kind="document"):
        return self._fallback.embed(texts, kind=kind)

    def tasks(self) -> list[str]:
        return [t for t, _ in self.calls]


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, llm_provider="mock", data_dir=tmp_path / "data", embedding_dim=64,
                    trace_dir=tmp_path / "traces", api_key=None, chunk_size=400, chunk_overlap=60)


@pytest.fixture
def make_agent(settings) -> Callable[..., KnowledgeAgent]:
    def _make(script: dict | None = None, ingest: bool = True) -> KnowledgeAgent:
        llm = TracedLLM(ScriptedLLM(script), 0.30, 2.50)
        agent = KnowledgeAgent(settings, llm=llm, persist_traces=False)
        if ingest:
            agent.ingest_directory(SAMPLE)
        return agent

    return _make
