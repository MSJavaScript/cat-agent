"""Deterministic tests for StepNode liveness detection and queue recovery.

Run from the repository root:

    python -m check.test_agent_liveness

No LLM and no graph execution are involved.  The tests pin the contract that
`status == "running"` alone is never used to decide whether a StepNode is
alive: the in-process registry in `catagent/workflow/liveness.py` is the
source of truth, and the request queue reports a dead step as an error result
instead of returning `END` forever.
"""
from __future__ import annotations

import asyncio

from langgraph.graph import END
from langgraph.types import Send

from catagent.workflow import liveness
from catagent.workflow.models import Step
from catagent.workflow.router import orchestrator_request_queue


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_step(step_id: str, status: str = "pending", **overrides) -> Step:
    values = dict(step_id=step_id, status=status,
                  preparation_done=True, workdir="/w")
    values.update(overrides)
    return Step(**values)


def config(thread_id: str = "t") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def test_phase_lifecycle() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"

    check(liveness.step_run_phase("t", step_id) == liveness.DEAD,
          "an unknown step must be dead")

    liveness.mark_step_scheduled("t", step_id)
    check(liveness.step_run_phase("t", step_id) == liveness.SCHEDULED,
          "the router placeholder must be visible before the node starts")

    task = asyncio.get_running_loop().create_future()
    liveness.mark_step_running("t", step_id, task)
    check(liveness.step_run_phase("t", step_id) == liveness.RUNNING,
          "an unfinished task must be running")

    liveness.mark_step_finished("t", step_id)
    check(liveness.step_run_phase("t", step_id) == liveness.FINISHED,
          "a normal return must stay visible until the router consumes it")

    liveness.clear_step_run("t", step_id)
    check(liveness.step_run_phase("t", step_id) == liveness.DEAD,
          "clearing the record must mean dead")


async def test_completed_task_is_dead() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"

    async def work() -> None:
        return None

    task = asyncio.ensure_future(work())
    liveness.mark_step_running("t", step_id, task)
    check(liveness.step_run_phase("t", step_id) == liveness.RUNNING,
          "pending task must be alive")
    await task
    check(liveness.step_run_phase("t", step_id) == liveness.DEAD,
          "done task must not count as alive")


def test_paused_is_not_dead() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    liveness.mark_step_paused("t", step_id)
    check(liveness.step_run_phase("t", step_id) == liveness.PAUSED,
          "a HITL pause must not be reported as dead")


async def test_running_step_with_owner_waits() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    liveness.mark_step_scheduled("t", step_id)
    state = {
        "finalization": False,
        "steps": {step_id: make_step(step_id, status="running")},
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state, config())
    check(command.goto == END,
          f"a scheduled step must make the queue wait, got {command.goto!r}")


async def test_dead_running_step_is_reported() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    state = {
        "finalization": False,
        "steps": {step_id: make_step(step_id, status="running")},
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state, config())
    check(command.goto == "router",
          f"a dead step must be routed to the router, got {command.goto!r}")
    result = (command.update or {}).get("step_results", {}).get(step_id)
    check(result is not None and result.status == "error",
          f"a dead step must produce an error StepResult, got {result!r}")


async def test_dead_running_step_is_not_masked_by_pending_sibling() -> None:
    """A pending sibling must not hide a running step that has no owner.

    Returning END on the first pending step was the old behaviour.  The graph
    then stopped with the dead step still ``running``: no interrupt, no result,
    no further router pass.
    """
    liveness.reset_step_runs()
    dead_id = "plan-0:step-0:a"
    pending_id = "plan-0:step-1:b"
    state = {
        "finalization": False,
        "steps": {
            dead_id: make_step(dead_id, status="running"),
            pending_id: make_step(pending_id, status="pending",
                                  preparation_done=False),
        },
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state, config())
    check(command.goto == "router",
          f"a dead step must not be masked, got {command.goto!r}")
    result = (command.update or {}).get("step_results", {}).get(dead_id)
    check(result is not None and result.status == "error",
          f"the dead step must produce an error StepResult, got {result!r}")
    check(pending_id not in (command.update or {}).get("step_results", {}),
          "a pending sibling must not be reported as crashed")


async def test_paused_running_step_waits() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    liveness.mark_step_paused("t", step_id)
    state = {
        "finalization": False,
        "steps": {step_id: make_step(step_id, status="running")},
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state, config())
    check(command.goto == END,
          f"a paused step must make the queue wait, got {command.goto!r}")


async def test_pending_step_waits() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    state = {
        "finalization": False,
        "steps": {step_id: make_step(step_id, status="pending")},
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state, config())
    check(command.goto == END,
          f"a pending step must make the queue wait, got {command.goto!r}")


async def test_finished_step_waits_for_router() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    liveness.mark_step_scheduled("t", step_id)
    liveness.mark_step_running("t", step_id, None)
    liveness.mark_step_finished("t", step_id)
    state = {
        "finalization": False,
        "steps": {step_id: make_step(step_id, status="running")},
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state, config())
    check(command.goto == END,
          f"a finishing step must make the queue wait, got {command.goto!r}")


async def test_running_step_without_thread_id_waits() -> None:
    """Without a thread id the registry cannot be consulted; never guess dead."""
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    state = {
        "finalization": False,
        "steps": {step_id: make_step(step_id, status="running")},
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state)
    check(command.goto == END,
          f"a missing thread id must fall back to waiting, got {command.goto!r}")


async def test_all_terminal_steps_still_replan() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    state = {
        "finalization": False,
        "steps": {step_id: make_step(step_id, status="done")},
        "orchestrator_requests": [],
    }
    command = await orchestrator_request_queue(state, config())
    send = command.goto
    check(isinstance(send, Send) and send.node == "orchestrator_node",
          f"unexpected replanning goto: {send!r}")
    check(send.arg.get("phase") == "replanning",
          f"unexpected replanning phase: {send.arg.get('phase')}")


# --------------------------------------------------------------------------
# step_node behaviour.  These build the real outer graph with a fake StepNode
# agent and call the registered `step_node` closure directly, so no LLM turn
# is needed.
# --------------------------------------------------------------------------
class _FakeState:
    def __init__(self, values: dict) -> None:
        self.values = values


class _FailingStepNode:
    async def astream(self, *args, **kwargs):
        raise RuntimeError("boom")
        yield  # pragma: no cover - makes this an async generator


class _CancellingStepNode:
    async def astream(self, *args, **kwargs):
        raise asyncio.CancelledError()
        yield  # pragma: no cover


class _SuccessfulStepNode:
    async def astream(self, *args, **kwargs):
        if False:  # pragma: no cover
            yield {}

    async def aget_state(self, config):
        return _FakeState({"step_result": {
            "status": "success",
            "summary": "done",
            "output_artifacts": {},
        }})


def _build_step_node(fake_stepnode):
    from langgraph.checkpoint.memory import InMemorySaver

    import catagent.agent as agent_module
    from check.scripted_model import ScriptedModel

    original_factory = agent_module.create_stepnode_agent
    try:
        agent_module.create_stepnode_agent = lambda **kwargs: fake_stepnode
        graph = agent_module.create_catagent(
            model=ScriptedModel(),
            job_manager_url="http://127.0.0.1:1",
            http_client=None,
            checkpointer=InMemorySaver(),
        )
    finally:
        agent_module.create_stepnode_agent = original_factory
    bound = graph.nodes["step_node"].bound
    return bound.afunc or bound.func


def _step_state(step_id: str) -> dict:
    return {
        "step_id": step_id,
        "workdir": "/tmp",
        "task_content": "do it",
        "loaded_skills": {},
        "interrupt_resume": {},
    }


async def test_step_node_crash_becomes_error_result() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    node = _build_step_node(_FailingStepNode())
    command = await node(_step_state(step_id), config("t"))

    from langgraph.types import Command

    check(isinstance(command, Command), f"expected a Command, got {command!r}")
    check(command.goto == "router",
          f"a crashed StepNode must route to the router, got {command.goto!r}")
    result = (command.update or {}).get("step_results", {}).get(step_id)
    check(result is not None and result.status == "error",
          f"a crashed StepNode must publish an error result, got {result!r}")
    check("RuntimeError: boom" in result.summary,
          f"the error summary must name the exception: {result.summary!r}")
    check(liveness.step_run_phase("t", step_id) == liveness.DEAD,
          "a crashed StepNode must not stay registered as alive")


async def test_step_node_cancel_stays_dead_and_propagates() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    node = _build_step_node(_CancellingStepNode())
    try:
        await node(_step_state(step_id), config("t"))
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("cancellation must not be converted into a result")
    check(liveness.step_run_phase("t", step_id) == liveness.DEAD,
          "a cancelled StepNode must not stay registered as alive")


async def test_step_node_success_marks_finished() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    node = _build_step_node(_SuccessfulStepNode())
    command = await node(_step_state(step_id), config("t"))
    check(command.goto == "router", f"unexpected goto: {command.goto!r}")
    check(liveness.step_run_phase("t", step_id) == liveness.FINISHED,
          "a normal return must stay finished until the router consumes it")


class _CapturingStepNode:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def astream(self, *args, **kwargs):
        self.calls.append(kwargs)
        if False:  # pragma: no cover
            yield {}

    async def aget_state(self, config):
        return _FakeState({"step_result": {
            "status": "success",
            "summary": "done",
            "output_artifacts": {},
        }})


async def test_step_node_forwards_context_on_resume() -> None:
    liveness.reset_step_runs()
    step_id = "plan-0:step-0:a"
    fake = _CapturingStepNode()
    node = _build_step_node(fake)
    state = {
        **_step_state(step_id),
        "workdir": "",
        "project_workdir": "/tmp/project",
        "interrupt_resume": {"inner-int": "yes"},
    }
    await node(state, config("t"))

    check(len(fake.calls) == 1, f"expected one inner run, got {fake.calls!r}")
    context = fake.calls[0].get("context") or {}
    check(context.get("project_workdir") == "/tmp/project",
          f"the resumed StepNode lost its project workdir: {context!r}")
    check(context.get("step_workdir") == "/tmp/project",
          f"a resumed step without its own workdir must fall back to the "
          f"project root: {context!r}")
    check(context.get("resource_dirs") == {},
          f"the StepNode context must always contain every schema key: {context!r}")


async def test_stream_chat_forwards_recursion_limit() -> None:
    from catagent.app.streaming import _stream_chat
    from catagent.agent import _child_config

    child = _child_config({"recursion_limit": 7,
                           "configurable": {"thread_id": "outer"}}, "inner")
    check(child == {"configurable": {"thread_id": "inner"},
                    "recursion_limit": 7},
          f"the recursion limit was not copied to the inner graph: {child!r}")

    captured: dict = {}
    captured_context: dict = {}

    class FakeAgent:
        async def astream(self, input, config, **kwargs):
            captured.update(config)
            captured_context.update(kwargs.get("context") or {})
            if False:  # pragma: no cover - async generator
                yield {}

    async for _ in _stream_chat(FakeAgent(), {}, "t", "", None,
                                register_jobs=False, recursion_limit=42):
        pass
    check(captured.get("recursion_limit") == 42,
          f"_stream_chat dropped the recursion limit: {captured!r}")
    check(captured_context.get("project_workdir"),
          f"_stream_chat dropped the project workdir context: {captured_context!r}")

    captured.clear()
    captured_context.clear()
    async for _ in _stream_chat(FakeAgent(), {}, "t", "", None,
                                register_jobs=False):
        pass
    check("recursion_limit" not in captured,
          f"an unset limit must keep LangGraph's default: {captured!r}")


async def main() -> None:
    tests = [
        ("phase lifecycle", test_phase_lifecycle),
        ("completed task is dead", test_completed_task_is_dead),
        ("paused is not dead", test_paused_is_not_dead),
        ("running step with owner waits", test_running_step_with_owner_waits),
        ("dead running step is reported", test_dead_running_step_is_reported),
        ("dead step is not masked by pending sibling",
         test_dead_running_step_is_not_masked_by_pending_sibling),
        ("paused running step waits", test_paused_running_step_waits),
        ("pending step waits", test_pending_step_waits),
        ("finished step waits for router", test_finished_step_waits_for_router),
        ("running step without thread id waits",
         test_running_step_without_thread_id_waits),
        ("all terminal steps still replan", test_all_terminal_steps_still_replan),
        ("step_node crash becomes error result",
         test_step_node_crash_becomes_error_result),
        ("step_node cancel stays dead and propagates",
         test_step_node_cancel_stays_dead_and_propagates),
        ("step_node success marks finished",
         test_step_node_success_marks_finished),
        ("step_node forwards context on resume",
         test_step_node_forwards_context_on_resume),
        ("stream_chat forwards recursion limit",
         test_stream_chat_forwards_recursion_limit),
    ]
    for name, function in tests:
        if asyncio.iscoroutinefunction(function):
            await function()
        else:
            function()
        print(f"  ok: {name}")
    print("\nAGENT LIVENESS TESTS: OK")


if __name__ == "__main__":
    import logging

    import catagent.agent as _agent_module

    # Keep the expected crash/warning logs out of the test output.
    _logger = logging.getLogger("catagent")
    _logger.addHandler(logging.NullHandler())
    _logger.propagate = False

    # `step_node` publishes progress through the graph stream writer.  The
    # direct-call tests below run outside a graph, so use a no-op writer for
    # the whole test run.
    _original_writer = _agent_module.get_stream_writer
    _agent_module.get_stream_writer = lambda: (lambda payload: None)
    try:
        asyncio.run(main())
    finally:
        _agent_module.get_stream_writer = _original_writer
