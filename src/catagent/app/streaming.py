"""Streaming helpers shared by the HTTP service and debug CLI."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict

import httpx

from ..workflow.liveness import PAUSED, clear_step_run, step_run_phase
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


class PendingInterruptError(RuntimeError):
    """A new user turn would discard interrupts that are still waiting.

    LangGraph treats a fresh ``input=`` on an interrupted thread as a new turn
    starting at the graph entry point.  The pending ``interrupt_node`` tasks
    (and therefore the only route back to the paused StepNode/orchestrator)
    are dropped.  That is how ``plan-0:step-4:sample_configs_iter1`` stayed
    ``running`` forever: a chat message sent while its job was in flight
    discarded step 4's interrupt, and the later job callback was rejected as
    stale.  Refusing the new turn is the only safe answer: the caller has to
    resume the pending interrupt (or wait for a JobSubmission callback) first.
    """

    def __init__(self, thread_id: str, pending: list[tuple[str, str | None]]):
        self.thread_id = thread_id
        self.pending = pending
        described = []
        for interrupt_id, source in pending:
            # ``interrupt_node`` annotates step-node interrupts with
            # ``_debug_source = "<thread_id>:<step_id>"`` so the UI and this
            # guard can name the paused step without a second state scan.
            prefix = f"{thread_id}:"
            if source and source.startswith(prefix):
                described.append(f"{interrupt_id} (step {source[len(prefix):]})")
            else:
                described.append(interrupt_id)
        super().__init__(
            f"Thread {thread_id} is paused on pending interrupt(s): "
            f"{', '.join(described)}. Answer or resume them first (a job "
            "submission resumes automatically when its result arrives). A new "
            "message would start a new graph turn and discard the paused work."
        )


async def _pending_interrupts(
    catagent,
    config: Dict[str, Any],
) -> list[tuple[str, str | None]] | None:
    """Return ``(interrupt_id, debug_source)`` for every pending interrupt.

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
    if snapshot is None:
        return None
    pending: dict[str, str | None] = {}

    def _add(interrupt_obj) -> None:
        value = getattr(interrupt_obj, "value", None)
        source = value.get("_debug_source") if isinstance(value, dict) else None
        pending.setdefault(str(interrupt_obj.id), source)

    for task in (getattr(snapshot, "tasks", None) or ()):
        for interrupt_obj in (getattr(task, "interrupts", None) or ()):
            _add(interrupt_obj)
    # ``StateSnapshot.interrupts`` is also populated for some graph shapes.
    for interrupt_obj in (getattr(snapshot, "interrupts", None) or ()):
        _add(interrupt_obj)
    return list(pending.items())


async def _pending_interrupt_ids(catagent, config: Dict[str, Any]) -> set[str] | None:
    """Return only the ids of the interrupts currently pending on a thread."""
    pending = await _pending_interrupts(catagent, config)
    if pending is None:
        return None
    return {interrupt_id for interrupt_id, _ in pending}


async def _reconcile_stale_job_result(
    catagent,
    config: Dict[str, Any],
    thread_id: str,
    expected_interrupt_id: str,
    job_local_dir: str | None,
) -> None:
    """Make a discarded job result visible instead of silently losing a step.

    A stale result cannot be applied: its outer interrupt is gone.  But the
    StepNode run it belonged to is usually still marked ``running`` while the
    in-process liveness record says ``paused`` (written by ``interrupt_node``
    before it suspended).  ``paused`` deliberately tells the request queue
    "not dead", so without this reconciliation nothing would ever report the
    step again.  Clear that mark so the next queue pass reports the dead run
    and the router can ask the orchestrator to correct the plan.
    """
    logger.info(
        "Skipping stale resume for thread=%s interrupt=%s: the interrupt is "
        "no longer pending (workdir=%s).",
        thread_id,
        expected_interrupt_id,
        job_local_dir or "<unknown>",
    )
    if not job_local_dir:
        return
    get_state = getattr(catagent, "aget_state", None)
    if get_state is None:
        return
    try:
        snapshot = await get_state(config)
        values = getattr(snapshot, "values", None) or {}
    except Exception:
        return

    # A job's ``local_dir`` may be the step workdir itself or a subdirectory
    # (``step4_md_iter1/smoke_test``).  Match the most specific workdir first.
    wanted = os.path.realpath(job_local_dir)
    matches = []
    for step_id, step in (values.get("steps") or {}).items():
        workdir = getattr(step, "workdir", "") or ""
        if not workdir:
            continue
        root = os.path.realpath(workdir)
        if wanted == root or wanted.startswith(root + os.sep):
            matches.append((len(root), step_id, step))
    matches.sort(key=lambda item: item[0], reverse=True)
    if not matches:
        return

    _, step_id, step = matches[0]
    # Only clear a *paused* record.  A live ``running``/``scheduled`` owner
    # must keep its liveness entry: the stale result concerns a different
    # interrupt, and marking a live step dead would make the queue dispatch it
    # twice.
    if getattr(step, "status", "") != "running":
        return
    if step_run_phase(thread_id, step_id) != PAUSED:
        return
    clear_step_run(thread_id, step_id)
    logger.warning(
        "Step %s lost the outer interrupt that would have resumed its job "
        "result; cleared its 'paused' liveness mark so the router reports it "
        "as dead instead of waiting forever.",
        step_id,
    )


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
    job_local_dir: str | None = None,
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
    old resume.  ``job_local_dir`` (the callback's job directory) is used only
    to name the step whose result had to be dropped, so it can be reported as
    dead instead of staying ``paused`` forever.
    """
    lock = thread_run_lock(thread_id)
    # Serialize the whole graph run, including the inner orchestrator and every
    # StepNode it dispatches.  Without this, two overlapping requests on one
    # thread fork the same checkpoint and each replay can execute `submit_job`.
    async with lock:
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        if recursion_limit is not None:
            config["recursion_limit"] = recursion_limit

        # A normal user turn must never start while an interrupt is pending.
        #
        # LangGraph starts a fresh `input=` turn at the graph entry point and
        # drops the pending `interrupt_node` tasks.  Those tasks hold the only
        # link back to the paused inner StepNode/orchestrator thread, so the
        # step would stay `running` with no owner and its later job result
        # would be rejected as stale.  This is exactly what killed
        # `plan-0:step-4:sample_configs_iter1` in the 2026-11-18 run.
        #
        # `agent_input` is a dict only for a new user message; resumes use a
        # `Command`, and job callbacks carry `expected_interrupt_id`.
        is_new_turn = isinstance(agent_input, dict) and bool(agent_input.get("messages"))
        if is_new_turn and expected_interrupt_id is None:
            pending = await _pending_interrupts(catagent, config)
            if pending:
                raise PendingInterruptError(thread_id, pending)

        if expected_interrupt_id:
            pending_ids = await _pending_interrupt_ids(catagent, config)
            # A job callback always carries an *outer* interrupt id that was
            # written to this thread's checkpoint before the registration was
            # sent.  If that id is no longer pending, the interrupt was already
            # resumed (at-least-once delivery sent it twice) and replaying the
            # old resume would only re-enter a pending node from the checkpoint.
            # That replay is what produced the repeated "Correction request"
            # orchestrator runs.  If the state cannot be inspected, do not skip.
            if pending_ids is not None and expected_interrupt_id not in pending_ids:
                await _reconcile_stale_job_result(
                    catagent,
                    config,
                    thread_id,
                    expected_interrupt_id,
                    job_local_dir,
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

        # Direct callers (the CLI and the e2e test) build the input dict
        # themselves and may omit `project_workdir`.  The runtime context below
        # is enough for the inner agents, but the outer `WorkflowState` also
        # persists the root and `router.write_readme()` reads it.  Seed it for a
        # new dict turn; a `Command(resume=...)` cannot carry state and the
        # value is already in the checkpoint for that path.
        if isinstance(agent_input, dict) and not agent_input.get("project_workdir"):
            agent_input = {**agent_input, "project_workdir": project_workdir}

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
