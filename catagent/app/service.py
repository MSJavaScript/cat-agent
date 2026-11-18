"""FastAPI service: ``/chat`` streaming, job callbacks and plan inspection."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from langchain.messages import HumanMessage
from langgraph.types import Command

from .events import JOBMANAGER_SOURCE, DebugEventLog
from .models import ChatRequest
from .plan import get_plan
from .streaming import _stream_chat

logger = logging.getLogger("catagent")

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


def _expected_resume_id(req: ChatRequest) -> str | None:
    """Return the outer interrupt id a resume request is answering.

    The HTTP UI sends it at the top level; other clients nest it inside
    ``interrupt_resume``.  Returning ``None`` means "cannot tell" and disables
    the staleness guard.
    """
    if req.interrupt_id:
        return req.interrupt_id
    resume_request = req.interrupt_resume
    if isinstance(resume_request, dict):
        interrupt_id = resume_request.get("interrupt_id")
        if isinstance(interrupt_id, str) and interrupt_id:
            return interrupt_id
    return None


def _resolve_workdir(workdir: str | None) -> str | None:
    """Normalize a requested project workdir, validating that it is a dir."""
    if workdir is None:
        return None
    workdir = workdir.strip()
    if not workdir:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(workdir))).resolve()
    if not path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"workdir does not exist or is not a directory: {path}",
        )
    return str(path)


def _normal_input(
    req: ChatRequest,
    workdir: str | None = None,
) -> tuple[str, Any]:
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
    # Do not overwrite the workdir persisted with an existing thread when a
    # follow-up message omits it.  A brand-new thread always gets one: the
    # value requested by the client, or the server process cwd.
    if workdir is not None or not req.thread_id:
        agent_input["project_workdir"] = workdir or os.path.abspath(os.getcwd())
    return thread_id, agent_input


def _sse(payload: dict[str, Any]) -> str:
    """Frame one payload as a Server-Sent Event."""
    return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"


async def build_app(
    dotenv_path: str,
    db_path: str,
    job_manager_url: str,
    skill_index_json: str | None = None,
    recursion_limit: int | None = None,
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
        # Detached job-result runs are not tied to an HTTP connection, so stop
        # them explicitly before closing the checkpoint DB / HTTP client.
        for task in list(app.state.run_tasks):
            task.cancel()
        for task in list(app.state.run_tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        await httpx_client.aclose()
        await db_conn.close()

    app = FastAPI(
        title="cat-agent",
        version="0.0.1",
        lifespan=lifespan,
    )
    app.state.catagent = catagent
    app.state.httpx_client = httpx_client
    # LangGraph superstep bound for every graph invocation (outer workflow,
    # orchestrator and each StepNode).  None means "use LangGraph's default".
    app.state.recursion_limit = recursion_limit
    # A stateless client may resume with only an interrupt value; remember the
    # most recent thread id as a convenience for that single-user debug case.
    app.state.last_thread_id = None
    # Every payload is also appended here so the debug UI can follow turns it
    # did not initiate itself (notably jobmanager callback resumes).
    app.state.event_log = DebugEventLog()
    # Job results are executed as detached tasks.  Keeping the task objects
    # alive prevents the event loop from garbage-collecting them mid-run and
    # lets shutdown cancel them cleanly.
    app.state.run_tasks: set[asyncio.Task] = set()

    def _spawn_run(coro) -> asyncio.Task:
        """Run one graph turn without tying its lifetime to an HTTP request."""
        task = asyncio.create_task(coro)
        app.state.run_tasks.add(task)

        def _done(finished: asyncio.Task) -> None:
            app.state.run_tasks.discard(finished)
            if finished.cancelled():
                return
            exc = finished.exception()
            if exc is not None:
                logger.error("Detached graph run failed: %s", exc, exc_info=exc)

        task.add_done_callback(_done)
        return task

    async def _run_detached(
        thread_id: str,
        agent_input: Any,
        workdir: str | None,
        register_jobs: bool = True,
        expected_interrupt_id: str | None = None,
    ) -> None:
        """Consume a graph run and publish its payloads to the event log.

        This is the important difference from the old callback path: the graph
        run is *not* owned by the jobmanager's HTTP request.  Closing that
        request (a read timeout, a retry, a restart) can no longer cancel the
        StepNode/orchestrator task and drop the interrupts it collected.
        """
        try:
            async for payload in _stream_chat(
                catagent=catagent,
                agent_input=agent_input,
                thread_id=thread_id,
                job_manager_url=job_manager_url,
                http_client=httpx_client,
                register_jobs=register_jobs,
                recursion_limit=app.state.recursion_limit,
                project_workdir=workdir,
                expected_interrupt_id=expected_interrupt_id,
            ):
                app.state.event_log.append(thread_id, payload)
        except asyncio.CancelledError:
            # Only server shutdown cancels this task now; the jobmanager has
            # already received its ACK and is not waiting for the body.
            logger.info("Detached graph run cancelled: thread=%s", thread_id)
            raise
        except Exception as exc:
            logger.exception("Detached graph run failed: thread=%s", thread_id)
            app.state.event_log.append(
                thread_id,
                {
                    "type": "custom",
                    "message": f"{type(exc).__name__}: {exc}",
                    "message_id": str(uuid.uuid4()),
                    "thread_id": thread_id,
                },
            )

    async def _accept_job_result(req: ChatRequest) -> JSONResponse:
        """Accept a JobResult and schedule the resume, then return immediately.

        Jobmanager's callback is a control-plane message, not a stream the
        caller needs to read.  Holding an SSE response open for a multi-minute
        StepNode turn is what caused the 30 s ``ReadTimeout`` churn and the
        overlapping retries.  A 202 ACK lets it mark the callback delivered
        while the graph continues in the background.
        """
        last_thread_id = app.state.last_thread_id
        workdir = _resolve_workdir(req.workdir)
        thread_id, agent_input = _resume_from_request(req, last_thread_id)
        app.state.last_thread_id = thread_id
        _spawn_run(
            _run_detached(
                thread_id=thread_id,
                agent_input=agent_input,
                workdir=workdir,
                register_jobs=req.register_jobs,
                expected_interrupt_id=req.interrupt_id,
            )
        )
        return JSONResponse(
            status_code=202,
            content={
                "ok": True,
                "accepted": True,
                "thread_id": thread_id,
                "interrupt_id": req.interrupt_id,
                "job_uuid": req.job_uuid,
            },
        )

    @app.get("/", response_class=HTMLResponse)
    async def debug_ui():
        """Single-page debug console (the POST routes below stay unchanged)."""
        return HTMLResponse(_DEBUG_UI_HTML)

    @app.get("/events/{thread_id}")
    async def events_route(thread_id: str, after: int = 0):
        """Poll payloads for one thread after a known sequence number.

        ``generation``/``latest_seq`` let the browser notice that its cursor
        belongs to a previous server process (or an evicted bucket) and replay
        from zero instead of silently polling a stream that will never advance.
        """
        return {"thread_id": thread_id, **app.state.event_log.read(thread_id, after)}

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
    # on "/" and "/chat" as well as "/job-result" means no extra proxy or
    # config change is required.
    #
    # A JobResult is detected by its `job_uuid`: an ordinary chat/resume never
    # sets that field.  Such a request is ACKed with 202 and the graph resume
    # runs as a detached task, so the jobmanager never holds an SSE connection
    # open for the whole StepNode turn.
    @app.post("/job-result")
    @app.post("/chat")
    @app.post("/")
    async def chat(req: ChatRequest):
        if req.job_uuid is not None:
            return await _accept_job_result(req)

        last_thread_id = app.state.last_thread_id
        workdir = _resolve_workdir(req.workdir)

        if req.interrupt_id is not None or req.interrupt_resume is not None:
            thread_id, agent_input = _resume_from_request(req, last_thread_id)
            # Apply the same stale-answer guard to *every* resume, not only to
            # jobmanager callbacks.  A double-clicked button, a browser retry,
            # or an old client resending an already-consumed interrupt id makes
            # LangGraph re-run the task that is currently at the head of the
            # checkpoint (it does not turn into a no-op).  That is exactly how
            # one persisted "Correction request from Router" was executed 11
            # times in logger.txt: the retries carried an interrupt id that had
            # already been consumed, so the pending orchestrator task ran again.
            expected_interrupt_id = _expected_resume_id(req)
        else:
            thread_id, agent_input = _normal_input(req, workdir)
            expected_interrupt_id = None

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
                    recursion_limit=app.state.recursion_limit,
                    project_workdir=workdir,
                    expected_interrupt_id=expected_interrupt_id,
                ):
                    app.state.event_log.append(thread_id, payload)
                    yield _sse(payload)
            except asyncio.CancelledError:
                # The ASGI server cancels this generator when the HTTP client
                # disconnects (the old jobmanager path hit this after its 30 s
                # read timeout).  Re-raise so Starlette closes the task; the
                # per-thread lock in `_stream_chat` is released by its
                # `async with`.  Job callbacks no longer go through this path.
                logger.info(
                    "SSE client disconnected; graph run cancelled: thread=%s",
                    thread_id,
                )
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
