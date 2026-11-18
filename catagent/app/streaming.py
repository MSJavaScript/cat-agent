"""Streaming helpers shared by the HTTP service and debug CLI."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

import httpx

from .models import StreamPayload
from .payloads import _jsonable, _payloads_from_chunk

logger = logging.getLogger("catagent")

# One run at a time per workflow thread.
#
# A thread is a single LangGraph checkpoint lineage: two concurrent ``astream``
# calls on the same thread both start from the same parent checkpoint and fork
# it.  That is what produced several StepNode runs (and several ``submit_job``
# calls) for one router dispatch.  The jobmanager used to create exactly this
# overlap: it held the callback's SSE connection open, timed out after 30 s,
# retried, and the retry started a second run while the first one was still
# executing.
#
# Keep the lock in this module (not in the Asyncio task that serves one HTTP
# request) so every entry point -- ``/chat`` SSE, the detached job-result
# runner, and the debug CLI -- shares it.
_THREAD_RUN_LOCKS: Dict[str, asyncio.Lock] = {}


def thread_run_lock(thread_id: str) -> asyncio.Lock:
    """Return the process-local run lock for one workflow thread.

    Everything runs on one event loop and there is no ``await`` between the
    lookup and the insert, so the registry itself is race-free.
    """
    lock = _THREAD_RUN_LOCKS.get(thread_id)
    if lock is None:
        lock = _THREAD_RUN_LOCKS[thread_id] = asyncio.Lock()
    return lock


async def _register_job_submission(
    payload: StreamPayload,
    thread_id: str,
    job_manager_url: str,
    http_client: httpx.AsyncClient,
) -> None:
    """Tell jobmanager which outer interrupt resumes a submitted job.

    This is the counterpart of ``JobManager._post_callback()``: the manager
    stores ``(thread_id, interrupt_id)`` under ``job_uuid`` and posts a
    ``JobResult`` back when the job reaches a terminal state.
    """
    value = payload.get("message")
    values = value if isinstance(value, (list, tuple)) else [value]
    for item in values:
        item = _jsonable(item)
        if not isinstance(item, dict) or item.get("kind") != "JobSubmission":
            continue
        # The interrupt value emitted by the outer interrupt_node is an
        # InterruptState: the job payload (including job_uuid) is nested under
        # "payload".  Accept a top-level job_uuid as well for direct callers.
        payload_value = item.get("payload")
        job_uuid = item.get("job_uuid")
        if not job_uuid and isinstance(payload_value, dict):
            job_uuid = payload_value.get("job_uuid")
        if not job_uuid:
            continue
        body = {
            "job_uuid": job_uuid,
            "thread_id": thread_id,
            "interrupt_id": payload.get("interrupt_id"),
        }
        try:
            response = await http_client.post(
                f"{job_manager_url.rstrip('/')}/set-thread-id", json=body
            )
            response.raise_for_status()
        except Exception as exc:
            # A failed bookkeeping call must not kill the agent stream.
            payload["registration_error"] = str(exc)


async def _pending_interrupt_ids(catagent, config: Dict[str, Any]) -> set[str] | None:
    """Return the interrupt ids currently pending on a thread.

    ``None`` means "could not determine" (a fresh thread, a test double without
    state access, ...).  Callers must treat that as "do not skip the resume":
    dropping a real job result is much worse than replaying one.
    """
    get_state = getattr(catagent, "aget_state", None)
    if get_state is None:
        return None
    try:
        snapshot = await get_state(config)
    except Exception:
        return None
    pending: set[str] = set()
    for task in (getattr(snapshot, "tasks", None) or ()):
        for interrupt_obj in (getattr(task, "interrupts", None) or ()):
            pending.add(str(interrupt_obj.id))
    # ``StateSnapshot.interrupts`` is also populated for some graph shapes.
    for interrupt_obj in (getattr(snapshot, "interrupts", None) or ()):
        pending.add(str(interrupt_obj.id))
    return pending


async def _stream_chat(
    catagent,
    agent_input: Any,
    thread_id: str,
    job_manager_url: str,
    http_client: httpx.AsyncClient,
    register_jobs: bool = True,
    recursion_limit: int | None = None,
    project_workdir: str | None = None,
    expected_interrupt_id: str | None = None,
):
    """Run the outer graph and yield UI/debug-friendly payloads.

    ``register_jobs`` controls whether JobSubmission interrupts are registered
    with the job manager, which would make it POST a resume back to the HTTP
    service.  The debug CLI polls the job manager itself, so it passes
    ``register_jobs=False`` to avoid a double resume.

    ``recursion_limit`` bounds LangGraph supersteps per graph invocation.  The
    value is carried in the config, and ``agent.py`` copies it to the inner
    orchestrator and StepNode graphs so one CLI/server option applies to all
    of them.  ``None`` keeps LangGraph's default.

    ``project_workdir`` is the root the StepNodes may access.  It normally
    comes from the request/agent input, but a resume that only carries an
    interrupt id has to read the value persisted with the thread.

    ``expected_interrupt_id`` is set for jobmanager callbacks.  Delivery is
    at-least-once, so a duplicate callback can arrive after the interrupt was
    already resumed; in that case the run is skipped instead of replaying the
    old resume.
    """
    lock = thread_run_lock(thread_id)
    # Serialize the whole graph run, including the inner orchestrator and every
    # StepNode it dispatches.  Without this, two overlapping requests on one
    # thread fork the same checkpoint and each replay can execute `submit_job`.
    async with lock:
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if recursion_limit is not None:
            config["recursion_limit"] = recursion_limit

        if expected_interrupt_id:
            pending = await _pending_interrupt_ids(catagent, config)
            # A job callback always carries an *outer* interrupt id that was
            # written to this thread's checkpoint before the registration was
            # sent.  If that id is no longer pending, the interrupt was already
            # resumed (at-least-once delivery sent it twice) and replaying the
            # old resume would only re-enter a pending node from the checkpoint.
            # That replay is what produced the repeated "Correction request"
            # orchestrator runs.  If the state cannot be inspected, do not skip.
            if pending is not None and expected_interrupt_id not in pending:
                logger.info(
                    "Skipping stale job-result resume for thread=%s "
                    "interrupt=%s: pending interrupts are %s.",
                    thread_id,
                    expected_interrupt_id,
                    sorted(pending),
                )
                return

        resource_dirs: dict[str, str] = {}
        if isinstance(agent_input, dict):
            project_workdir = project_workdir or agent_input.get("project_workdir")
            resource_dirs = agent_input.get("resource_dirs") or {}

        if not project_workdir:
            get_state = getattr(catagent, "aget_state", None)
            if get_state is not None:
                try:
                    snapshot = await get_state(config)
                    values = getattr(snapshot, "values", None) or {}
                    project_workdir = values.get("project_workdir")
                    resource_dirs = values.get("resource_dirs") or resource_dirs
                except Exception:
                    # A fresh thread has no checkpoint yet, and lightweight test
                    # doubles may not implement state access at all.
                    pass

        # A client that predates the workdir field may resume an old thread.
        # The server cwd is the same fallback used for a regular headless request.
        if not project_workdir:
            project_workdir = os.path.abspath(os.getcwd())

        context = {
            "project_workdir": project_workdir,
            "resource_dirs": resource_dirs,
        }

        async for chunk in catagent.astream(
            input=agent_input,
            config=config,
            context=context,
            stream_mode=["custom", "updates"],
            version="v2",
        ):
            for payload in _payloads_from_chunk(chunk):
                payload["thread_id"] = thread_id
                if register_jobs and payload.get("type") == "interrupt":
                    await _register_job_submission(
                        payload, thread_id, job_manager_url, http_client
                    )
                yield payload
