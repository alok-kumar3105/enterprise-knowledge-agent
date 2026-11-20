"""FastAPI service exposing ingestion, question answering and metrics."""
from __future__ import annotations

import hmac
import logging
import re
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .ingest.loaders import DOC_EXTENSIONS, TABLE_EXTENSIONS
from .service import IngestReport, KnowledgeAgent

log = logging.getLogger("knowledge_agent")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(agent: KnowledgeAgent | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or (agent.settings if agent else get_settings())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is None:
            app.state.agent = KnowledgeAgent(settings)
        else:
            app.state.agent = agent
        a: KnowledgeAgent = app.state.agent
        if settings.seed_dir and a.store.count() == 0 and Path(settings.seed_dir).exists():
            report = a.ingest_directory(Path(settings.seed_dir))
            log.info("Seeded index from %s: %s", settings.seed_dir, report.to_dict())
        if not settings.api_key:
            log.warning("API_KEY is not set: the API is unauthenticated. Set API_KEY in production.")
        yield

    app = FastAPI(title="Enterprise Knowledge Agent", version=__version__, lifespan=lifespan)

    def get_agent(request: Request) -> KnowledgeAgent:
        return request.app.state.agent

    def require_key(key: str | None = Security(_api_key_header)) -> None:
        if settings.api_key and not (key and hmac.compare_digest(key, settings.api_key)):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(_STATIC / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/stats", dependencies=[Depends(require_key)])
    def stats(a: KnowledgeAgent = Depends(get_agent)):
        return a.stats()

    @app.get("/metrics", dependencies=[Depends(require_key)])
    def metrics(a: KnowledgeAgent = Depends(get_agent)):
        return a.metrics.snapshot()

    @app.post("/ask", dependencies=[Depends(require_key)])
    async def ask(body: AskRequest, a: KnowledgeAgent = Depends(get_agent)):
        try:
            return await run_in_threadpool(a.ask, body.question)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/ingest", dependencies=[Depends(require_key)])
    async def ingest(files: list[UploadFile] = File(...), a: KnowledgeAgent = Depends(get_agent)):
        allowed = DOC_EXTENSIONS | TABLE_EXTENSIONS
        limit = settings.max_upload_mb * 1024 * 1024
        report = IngestReport()
        with tempfile.TemporaryDirectory() as tmp:
            for upload in files:
                # Strip any path components and odd characters from the client filename.
                name = re.sub(r"[^A-Za-z0-9._-]", "_", Path(upload.filename or "upload").name)
                ext = Path(name).suffix.lower()
                if ext not in allowed:
                    report.skipped.append(f"{name}: unsupported type (allowed: {', '.join(sorted(allowed))})")
                    continue
                dest = Path(tmp) / name
                size = 0
                with dest.open("wb") as out:
                    while chunk := await upload.read(1024 * 1024):
                        size += len(chunk)
                        if size > limit:
                            break
                        out.write(chunk)
                if size > limit:
                    report.skipped.append(f"{name}: larger than {settings.max_upload_mb} MB")
                    continue
                await run_in_threadpool(a.ingest_file, dest, report)
        return report.to_dict()

    @app.delete("/index", dependencies=[Depends(require_key)])
    def reset(a: KnowledgeAgent = Depends(get_agent)):
        a.reset()
        return {"status": "cleared"}

    return app


app = create_app()
