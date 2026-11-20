"""Application settings, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM provider -------------------------------------------------------
    # "gemini" uses Google Gemini (AI Studio API key or Vertex AI on GCP).
    # "mock" is a deterministic offline provider used for demos and tests.
    llm_provider: Literal["gemini", "mock"] = "mock"
    google_api_key: str | None = None
    use_vertex_ai: bool = False
    gcp_project: str | None = None
    gcp_location: str = "us-central1"

    chat_model: str = "gemini-3.6-flash"
    # Gemini 3 "thinking" depth: minimal | low | medium | high, or "" for the
    # model default. Low keeps latency down; raise it if answer quality drops.
    thinking_level: str = "low"
    llm_timeout_s: int = 60
    llm_retries: int = 3
    embedding_model: str = "gemini-embedding-001"
    embedding_dim: int = 768

    # USD per 1M tokens, used for cost-per-request tracking. Check current
    # Gemini pricing and override these in .env if they have changed.
    price_input_per_m: float = 0.30
    price_output_per_m: float = 2.50

    # --- Storage ------------------------------------------------------------
    data_dir: Path = Path(".data")
    vector_backend: Literal["local", "chroma"] = "local"
    # If set and the index is empty at startup, ingest this folder (handy on
    # Cloud Run, whose filesystem is ephemeral).
    seed_dir: Path | None = None

    # --- Retrieval ----------------------------------------------------------
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # --- Agent loop ---------------------------------------------------------
    max_revisions: int = 2
    max_sql_attempts: int = 3
    sql_row_limit: int = 50

    # --- API ----------------------------------------------------------------
    api_key: str | None = None
    max_upload_mb: int = 10
    trace_dir: Path = Path(".data/traces")


@lru_cache
def get_settings() -> Settings:
    return Settings()
