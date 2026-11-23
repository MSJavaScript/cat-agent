"""In-process liveness registry for StepNode runs.

The outer workflow keeps a *state* copy of every step, but ``Step.status`` is
written by the router and says only what the workflow *intends* the step to do.
It cannot tell whether the coroutine that was supposed to execute the step is
still alive: an uncaught exception aborts the superstep without rewriting the
status, and a process kill leaves the last checkpoint untouched.

This module is the out-of-band side channel the router and the request queue
use to answer that question.  It is deliberately process-local: the whole app
runs on one event loop in one uvicorn process, and the orchestrator lock/claim
registry in ``agent.py`` already relies on the same assumption.  See
``DEVELOPMENT.md`` for the multi-process alternative and for the cases this
registry does not cover.

Lifecycle of one step run::

    scheduled ──▶ running ──▶ finished
        │            │
        │            └──▶ (crashed/cancelled) ──▶ absent
        └──▶ paused ──▶ scheduled ──▶ ...

``scheduled``   the router has dispatched a ``Send("step_node", ...)`` but the
                node has not started yet.  Needed because the router sends the
                request queue and the step node in the same superstep, so the
                queue can run before ``mark_step_running``.
``running``     ``step_node`` is executing; the record keeps the actual
                ``asyncio.Task`` for an exact liveness check.
``finished``    ``step_node`` returned normally and its ``StepResult`` is on
                its way to the router.  The router clears the record when it
                consumes the result.
``paused``      the outer ``interrupt_node`` is waiting for the user.  Not
                dead: the graph will resume the same step when the answer
                arrives.
``absent``      nothing owns this step.  Combined with ``status == "running"``
                in workflow state this means the step died (crash, cancel,
                restart) and the queue must reconcile it instead of returning
                ``END`` forever.

All operations are synchronous and contain no ``await``, so on the single
event loop they are atomic with respect to graph nodes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, Iterable, Optional

logger = logging.getLogger("catagent")

SCHEDULED = "scheduled"
RUNNING = "running"
FINISHED = "finished"
PAUSED = "paused"
DEAD = "dead"

# (thread_id, step_id) -> {"phase": str, "task": asyncio.Task | None, "since": float}
_RUNS: Dict[tuple, Dict[str, object]] = {}


def _key(thread_id: str, step_id: str) -> tuple:
    return (thread_id, step_id)


def _store(thread_id: str, step_id: str, phase: str,
           task: Optional[asyncio.Task] = None) -> None:
    _RUNS[_key(thread_id, step_id)] = {
        "phase": phase,
        "task": task,
        "since": time.monotonic(),
    }


def mark_step_scheduled(thread_id: str, step_id: str) -> None:
    """Record that the router dispatched a step node for this step.

    Called by ``router`` (sync) *before* it returns the ``Send`` list, which is
    what closes the race with the request queue dispatched in the same
    superstep.
    """
    if not thread_id or not step_id:
        return
    _store(thread_id, step_id, SCHEDULED)


def mark_step_running(thread_id: str, step_id: str,
                      task: Optional[asyncio.Task]) -> None:
    """Record that ``step_node`` is executing, with its actual asyncio task."""
    if not thread_id or not step_id:
        return
    _store(thread_id, step_id, RUNNING, task)


def mark_step_finished(thread_id: str, step_id: str) -> None:
    """Record a normal return; the result still has to reach the router."""
    record = _RUNS.get(_key(thread_id, step_id))
    if record is None or record["phase"] != RUNNING:
        return
    record["phase"] = FINISHED
    record["task"] = None
    record["since"] = time.monotonic()


def mark_step_paused(thread_id: str, step_id: str) -> None:
    """Record that the outer ``interrupt_node`` is waiting for the user."""
    if not thread_id or not step_id:
        return
    _store(thread_id, step_id, PAUSED)


def clear_step_run(thread_id: str, step_id: str) -> None:
    """Forget a step.  Absence is the only "dead" signal."""
    _RUNS.pop(_key(thread_id, step_id), None)


def step_run_phase(thread_id: str, step_id: str) -> str:
    """Return the current phase, or ``DEAD`` when no run owns the step.

    ``task.done()`` is a safety net for a task that was cancelled without the
    node's cleanup running (for example an event-loop shutdown race).  The
    normal crash/cancel path clears the record in ``step_node`` itself.
    """
    record = _RUNS.get(_key(thread_id, step_id))
    if record is None:
        return DEAD
    if record["phase"] == RUNNING:
        task = record.get("task")
        if task is not None and task.done():
            return DEAD
    return str(record["phase"])


def prune_step_runs(thread_id: str, live_step_ids: Iterable[str]) -> None:
    """Drop records for steps no longer present in ``state["steps"]``.

    Steps can disappear when the orchestrator replaces a plan.  Without this
    sweep an abandoned record would live until process exit.
    """
    live = set(live_step_ids)
    for key in [key for key in _RUNS if key[0] == thread_id and key[1] not in live]:
        _RUNS.pop(key, None)


def reset_step_runs(thread_id: Optional[str] = None) -> None:
    """Clear the registry (tests, or a thread that is being discarded)."""
    if thread_id is None:
        _RUNS.clear()
        return
    for key in [key for key in _RUNS if key[0] == thread_id]:
        _RUNS.pop(key, None)
