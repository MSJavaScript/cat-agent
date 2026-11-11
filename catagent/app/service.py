"""FastAPI service: ``/chat`` streaming, job callbacks and plan inspection."""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain.messages import HumanMessage
from langgraph.types import Command

from .events import JOBMANAGER_SOURCE, DebugEventLog
from .models import ChatRequest
from .plan import get_plan
from .streaming import _stream_chat

_DEBUG_UI_HTML = (Path(__file__).with_name("debug_ui.html")).read_text(
    encoding="utf-8"
)


def _resume_from_request(
    req: ChatRequest,
    last_thread_id: str | None,
) -> tuple[str, Any]:
    """Build the thread id and graph input for a resume request."""
    # JobManager callback (JobResult).  The inner StepNode interrupt expects
    # the answer under the "payload" key.
    #
    # The debug CLI may have submitted a job
    # without registering a resume target.  The job manager then posts a
    # callback with empty thread_id/interrupt_id; ignore it instead of
    # resuming whichever request happened to be last.
    if req.interrupt_id:
        thread_id = req.thread_id or last_thread_id
        if not thread_id:
            raise HTTPException(
                status_code=400,
                detail="thread_id is required to resume an interrupt.",
            )
        if req.interrupt_resume is not None:
            response = req.interrupt_resume
        else:
            response = {
                "payload": {
                    "local_dir": req.local_dir or "",
                    "status": req.status or "error",
                    "error_msg": req.error_msg or "",
                }
            }
        return thread_id, Command(resume={req.interrupt_id: response})

    resume_request = req.interrupt_resume
    if resume_request is None:
        raise HTTPException(status_code=400, detail="interrupt_resume is required.")

    thread_id = req.thread_id
    interrupt_id = None
    response: Any = resume_request
    if isinstance(resume_request, dict):
        thread_id = thread_id or resume_request.get("thread_id")
        interrupt_id = resume_request.get("interrupt_id")
        for key in ("resume", "response", "value"):
            if key in resume_request:
                response = resume_request[key]
                break

    thread_id = thread_id or last_thread_id
    if not thread_id:
        raise HTTPException(
            status_code=400,
            detail="thread_id is required to resume an interrupt.",
        )

    if interrupt_id:
        return thread_id, Command(resume={interrupt_id: response})
    return thread_id, Command(resume=response)


def _normal_input(req: ChatRequest) -> tuple[str, Any]:
    """Build the thread id and initial state for a new user message."""
    if req.message is None or len(req.message) == 0:
        raise HTTPException(
            status_code=400,
            detail="Either message or interrupt_resume must be provided.",
        )
    thread_id = req.thread_id or str(uuid.uuid4())
    agent_input = {
        "messages": [HumanMessage(content=req.message)],
        "finalization": False,
        "orchestrator_is_working": False,
    }
    return thread_id, agent_input


def _sse(payload: dict[str, Any]) -> str:
    """Frame one payload as a Server-Sent Event."""
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def build_app(
    dotenv_path: str,
    db_path: str,
    job_manager_url: str,
    skill_index_json: str | None = None,
) -> FastAPI:
    """Create the cat-agent HTTP application and its runtime resources."""
    # Resolve through the package facade so ``catagent.app.build_runtime``
    # stays patchable, as it was when everything lived in one module.
    from . import build_runtime as _build_runtime

    runtime_kwargs = {
        "dotenv_path": dotenv_path,
        "db_path": db_path,
        "job_manager_url": job_manager_url,
    }
    if skill_index_json:
        runtime_kwargs["skill_index_json"] = skill_index_json
    catagent, httpx_client, db_conn = await _build_runtime(**runtime_kwargs)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await httpx_client.aclose()
        await db_conn.close()

    app = FastAPI(
        title="cat-agent",
        version="0.0.1",
        lifespan=lifespan,
    )
    app.state.catagent = catagent
    app.state.httpx_client = httpx_client
    # A stateless client may resume with only an interrupt value; remember the
    # most recent thread id as a convenience for that single-user debug case.
    app.state.last_thread_id = None
    # Every payload is also appended here so the debug UI can follow turns it
    # did not initiate itself (notably jobmanager callback resumes).
    app.state.event_log = DebugEventLog()

    @app.get("/", response_class=HTMLResponse)
    async def debug_ui():
        """Single-page debug console (the POST routes below stay unchanged)."""
        return HTMLResponse(_DEBUG_UI_HTML)

    @app.get("/events/{thread_id}")
    async def events_route(thread_id: str, after: int = 0):
        """Poll payloads for one thread after a known sequence number."""
        return {
            "thread_id": thread_id,
            "events": app.state.event_log.since(thread_id, after),
        }

    @app.post("/jobmanager/messages")
    async def jobmanager_messages(request: Request):
        """Receive progress messages posted to jobmanager's ``front_end_url``.

        ``run_job_agent`` posts ``{"role", "message", "tool_calls", ...}``.
        They have no cat-agent thread id, so they all go to the synthetic
        jobmanager bucket used by the side panel.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {"message": body}

        payload = {
            "type": "jobmanager",
            "role": body.get("role") or "message",
            "message": body.get("message"),
            "tool_calls": body.get("tool_calls") or "",
            "phase": body.get("phase"),
            "message_id": str(uuid.uuid4()),
            "source": JOBMANAGER_SOURCE,
        }
        event = app.state.event_log.append(JOBMANAGER_SOURCE, payload)
        return {"ok": True, "seq": event["seq"]}

    # jobmanager posts its JobResult directly to `callback_url`; accepting it
    # on "/" as well as "/chat" means no extra proxy is required.
    @app.post("/chat")
    @app.post("/")
    async def chat(req: ChatRequest):
        last_thread_id = app.state.last_thread_id

        if req.interrupt_id is not None or req.interrupt_resume is not None:
            thread_id, agent_input = _resume_from_request(req, last_thread_id)
        else:
            thread_id, agent_input = _normal_input(req)

        app.state.last_thread_id = thread_id

        async def event_stream():
            try:
                async for payload in _stream_chat(
                    catagent=catagent,
                    agent_input=agent_input,
                    thread_id=thread_id,
                    job_manager_url=job_manager_url,
                    http_client=httpx_client,
                    register_jobs=req.register_jobs,
                ):
                    app.state.event_log.append(thread_id, payload)
                    yield _sse(payload)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                payload = {
                    "type": "custom",
                    "message": f"{type(exc).__name__}: {exc}",
                    "message_id": str(uuid.uuid4()),
                    "thread_id": thread_id,
                }
                app.state.event_log.append(thread_id, payload)
                yield _sse(payload)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Disable Nginx buffering so streaming is not delayed.
                "X-Accel-Buffering": "no",
                "X-Thread-Id": thread_id,
            },
        )

    @app.get("/plan/{thread_id}")
    async def plan_route(thread_id: str):
        return await get_plan(catagent, thread_id)

    return app
