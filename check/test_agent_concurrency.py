"""Regression tests for duplicate runs, duplicate jobs and stale callbacks.

These tests do not call an LLM.  They cover the three mechanisms that made the
same StepNode submit the same job several times:

1. ``(?)/chat`` runs on one thread are serialized, so a retry cannot fork the
   same checkpoint.
2. ``submit_job`` derives a stable ``job_uuid`` from the replayed tool call, so
   even an accidental replay addresses the same jobmanager row.
3. A JobResult whose outer interrupt is no longer pending is ignored instead of
   replaying an old resume (which re-entered the orchestrator and printed the
   same "Correction request from Router" many times).

Run from the repository root:

    python -m check.test_agent_concurrency
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import tempfile
from types import SimpleNamespace
from typing import Any, List

import httpx

from catagent.app.streaming import PendingInterruptError, _stream_chat
from catagent.stepnode.init_tools import init_job_manager_tools, tool_call_wrapper


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok: {name}")


class FakeCatagent:
    """Minimal a stream-able graph double."""

    def __init__(self, snapshot: Any = None) -> None:
        self.snapshot = snapshot
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self.delay = 0.0

    async def aget_state(self, config):
        return self.snapshot

    async def astream(self, **kwargs):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            # An async generator that yields one harmless custom chunk.
            yield {"type": "custom", "ns": (), "data": {"hello": self.calls}}
        finally:
            self.active -= 1


class FakeInterrupt:
    def __init__(self, interrupt_id: str, value: Any = None) -> None:
        self.id = interrupt_id
        self.value = value if value is not None else {}


class FakeSnapshot:
    def __init__(
        self,
        interrupt_ids: List[str],
        values: Any = None,
        interrupt_values: Any = None,
    ) -> None:
        values = values or {}
        interrupt_values = interrupt_values or {}
        self.tasks = [
            SimpleNamespace(
                interrupts=[
                    FakeInterrupt(
                        interrupt_id,
                        interrupt_values.get(interrupt_id),
                    )
                ]
            )
            for interrupt_id in interrupt_ids
        ]
        self.interrupts = ()
        self.values = values
        self.next = ("interrupt_node",) if interrupt_ids else ()


async def test_serialized_runs() -> None:
    catagent = FakeCatagent()
    catagent.delay = 0.15

    async def consume() -> None:
        async for _ in _stream_chat(
            catagent=catagent,
            agent_input={},
            thread_id="concurrency-serialized",
            job_manager_url="http://unused",
            http_client=httpx.AsyncClient(),
            register_jobs=False,
            project_workdir="/tmp",
        ):
            pass

    await asyncio.gather(consume(), consume(), consume())
    check("three concurrent runs are serialized", catagent.max_active == 1,
          f"max_active={catagent.max_active}")
    check("all three runs still executed", catagent.calls == 3,
          f"calls={catagent.calls}")


async def test_stale_resume_is_skipped() -> None:
    stale = FakeCatagent(FakeSnapshot(["still-pending"]))
    async for _ in _stream_chat(
        catagent=stale,
        agent_input={},
        thread_id="concurrency-stale",
        job_manager_url="http://unused",
        http_client=httpx.AsyncClient(),
        register_jobs=False,
        project_workdir="/tmp",
        expected_interrupt_id="already-resumed",
    ):
        pass
    check("stale job callback does not start a run", stale.calls == 0,
          f"calls={stale.calls}")

    live = FakeCatagent(FakeSnapshot(["live-interrupt"]))
    async for _ in _stream_chat(
        catagent=live,
        agent_input={},
        thread_id="concurrency-live",
        job_manager_url="http://unused",
        http_client=httpx.AsyncClient(),
        register_jobs=False,
        project_workdir="/tmp",
        expected_interrupt_id="live-interrupt",
    ):
        pass
    check("pending job callback still resumes", live.calls == 1,
          f"calls={live.calls}")


async def test_new_message_is_rejected_while_paused() -> None:
    """A new turn must not silently discard a paused step's interrupt."""
    step_id = "plan-0:step-4:sample_configs_iter1"
    thread_id = "concurrency-paused-message"
    paused = FakeCatagent(
        FakeSnapshot(
            ["step4-interrupt"],
            values={
                "steps": {
                    step_id: SimpleNamespace(
                        workdir="/tmp/case/step4", status="running"
                    )
                }
            },
            interrupt_values={
                "step4-interrupt": {"_debug_source": f"{thread_id}:{step_id}"}
            },
        )
    )
    error = None
    try:
        async for _ in _stream_chat(
            catagent=paused,
            agent_input={"messages": ["skip step 4"]},
            thread_id=thread_id,
            job_manager_url="http://unused",
            http_client=httpx.AsyncClient(),
            register_jobs=False,
            project_workdir="/tmp",
        ):
            pass
    except PendingInterruptError as exc:
        error = exc

    check("new message while paused is rejected", error is not None,
          "the message was allowed to discard the pending interrupt")
    check("rejection names the paused step",
          error is not None and step_id in str(error), f"error={error}")
    check("rejection does not start a graph run", paused.calls == 0,
          f"calls={paused.calls}")

    fresh = FakeCatagent(FakeSnapshot([]))
    async for _ in _stream_chat(
        catagent=fresh,
        agent_input={"messages": ["start"]},
        thread_id="concurrency-idle-message",
        job_manager_url="http://unused",
        http_client=httpx.AsyncClient(),
        register_jobs=False,
        project_workdir="/tmp",
    ):
        pass
    check("ordinary message on an idle thread still runs", fresh.calls == 1,
          f"calls={fresh.calls}")


async def test_stale_job_result_marks_paused_step_dead() -> None:
    """A result that cannot be delivered must not leave the step 'paused'.

    The queue treats ``paused`` as alive, so an orphaned paused record hides a
    dead ``running`` step exactly as it hid step 4 in the 2026-11-18 run.
    """
    from catagent.workflow.liveness import (
        DEAD,
        clear_step_run,
        mark_step_paused,
        step_run_phase,
    )

    thread_id = "concurrency-stale-result"
    step_id = "plan-0:step-4:sample_configs_iter1"
    catagent = FakeCatagent(
        FakeSnapshot(
            ["some-other-interrupt"],
            values={
                "steps": {
                    step_id: SimpleNamespace(
                        workdir="/tmp/case/step4", status="running"
                    )
                }
            },
        )
    )
    clear_step_run(thread_id, step_id)
    mark_step_paused(thread_id, step_id)
    async for _ in _stream_chat(
        catagent=catagent,
        agent_input={},
        thread_id=thread_id,
        job_manager_url="http://unused",
        http_client=httpx.AsyncClient(),
        register_jobs=False,
        project_workdir="/tmp",
        expected_interrupt_id="stale-step4-interrupt",
        job_local_dir="/tmp/case/step4/smoke_test",
    ):
        pass
    check("stale job result does not start a run", catagent.calls == 0,
          f"calls={catagent.calls}")
    check("stale job result marks the paused step dead",
          step_run_phase(thread_id, step_id) == DEAD,
          f"phase={step_run_phase(thread_id, step_id)}")
    clear_step_run(thread_id, step_id)


async def test_deterministic_job_uuid() -> None:
    posted: List[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        posted.append(json.loads(request.content))
        return httpx.Response(200, json={"status": "success", "message": "ok"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tools = init_job_manager_tools("http://jobmanager", client)
        submit_tool = next(tool for tool in tools if tool.name == "submit_job")
        runtime = SimpleNamespace(
            config={"configurable": {"thread_id": "outer:step-1"}},
            tool_call_id="call-1",
        )

        # submit_job validates the step directory before it posts anything.
        # Use a throwaway directory with run.sh so the test does not depend on
        # /tmp/step1 being left behind by an earlier run.
        with tempfile.TemporaryDirectory() as local_dir:
            pathlib.Path(local_dir, "run.sh").write_text("#!/bin/sh\n")
            pathlib.Path(local_dir, "job_description.json").write_text("{}\n")

            def make_request(tool_call_id: str):
                return SimpleNamespace(
                    tool=submit_tool,
                    tool_call={
                        "name": "submit_job",
                        "id": tool_call_id,
                        "args": {
                            "job_description": "relax",
                            "local_dir": local_dir,
                            "provider_name": "local",
                            "job_env_name": "mock",
                        },
                    },
                    runtime=runtime,
                    state={},
                )

            async def handler_fn(req):
                raise AssertionError("submit_job must be handled by the wrapper")

            await tool_call_wrapper(make_request("call-1"), handler_fn)
            await tool_call_wrapper(make_request("call-1"), handler_fn)
            await tool_call_wrapper(make_request("call-2"), handler_fn)

    check("replayed tool call reuses job_uuid",
          posted[0]["job_uuid"] == posted[1]["job_uuid"],
          f"{posted[0]['job_uuid']} != {posted[1]['job_uuid']}")
    check("a new tool call gets a new job_uuid",
          posted[0]["job_uuid"] != posted[2]["job_uuid"],
          f"{posted[0]['job_uuid']} == {posted[2]['job_uuid']}")


async def test_job_result_ack() -> None:
    """`POST /` must ACK a JobResult immediately and run the resume detached."""
    import catagent.app as app_facade
    import catagent.app.service as service

    runtime_client = httpx.AsyncClient()

    class FakeDB:
        async def close(self) -> None:
            pass

    class FakeGraph:
        pass

    async def fake_build_runtime(**kwargs: Any):
        return FakeGraph(), runtime_client, FakeDB()

    seen: List[dict] = []

    async def fake_stream_chat(**kwargs: Any):
        seen.append(kwargs)
        yield {
            "type": "agent",
            "message": "resumed",
            "message_id": "m1",
            "thread_id": kwargs["thread_id"],
        }

    original_build = app_facade.build_runtime
    original_stream = service._stream_chat
    app_facade.build_runtime = fake_build_runtime
    service._stream_chat = fake_stream_chat
    try:
        app = await service.build_app(
            dotenv_path=".env",
            db_path=":memory:",
            job_manager_url="http://jobs",
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            response = await client.post(
                "/",
                json={
                    "job_uuid": "job-1",
                    "thread_id": "t-job",
                    "interrupt_id": "interrupt-1",
                    "local_dir": "/tmp/step1",
                    "status": "success",
                    "error_msg": "",
                },
            )
            check("job result is ACKed with 202", response.status_code == 202,
                  response.text)
            for _ in range(100):
                if not app.state.run_tasks:
                    break
                await asyncio.sleep(0.01)
            check("detached resume used the callback interrupt",
                  seen and seen[-1]["expected_interrupt_id"] == "interrupt-1",
                  f"seen={seen}")
            events = (
                await client.get("/events/t-job?after=0")
            ).json()["events"]
            check("detached resume published its payloads",
                  bool(events) and events[-1]["payload"]["message"] == "resumed",
                  f"events={events}")
    finally:
        app_facade.build_runtime = original_build
        service._stream_chat = original_stream
        await runtime_client.aclose()


async def test_stale_ui_resume_is_skipped() -> None:
    """A duplicated UI answer must not replay the pending task.

    LangGraph does not turn a resume for an already-consumed interrupt into a
    no-op: it re-runs whatever task is at the head of the checkpoint.  That is
    how one persisted correction ran 11 times in ``logger.txt``.
    """
    import catagent.app as app_facade
    import catagent.app.service as service

    runtime_client = httpx.AsyncClient()

    class FakeDB:
        async def close(self) -> None:
            pass

    stale = FakeCatagent(FakeSnapshot(["pending-id"]))

    async def fake_build_runtime(**kwargs: Any):
        return stale, runtime_client, FakeDB()

    original_build = app_facade.build_runtime
    app_facade.build_runtime = fake_build_runtime
    try:
        app = await service.build_app(
            dotenv_path=".env",
            db_path=":memory:",
            job_manager_url="http://jobs",
        )
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            async with client.stream(
                "POST",
                "/chat",
                json={
                    "thread_id": "t-ui-stale",
                    "interrupt_id": "already-consumed",
                    "interrupt_resume": {"payload": {}},
                },
            ) as response:
                check("stale UI resume is accepted", response.status_code == 200,
                      f"status={response.status_code}")
                await response.aread()
            check("stale UI resume does not run the graph", stale.calls == 0,
                  f"calls={stale.calls}")

            async with client.stream(
                "POST",
                "/chat",
                json={
                    "thread_id": "t-ui-live",
                    "interrupt_id": "pending-id",
                    "interrupt_resume": {"payload": {}},
                },
            ) as response:
                check("live UI resume is accepted", response.status_code == 200,
                      f"status={response.status_code}")
                await response.aread()
            check("live UI resume still runs the graph", stale.calls == 1,
                  f"calls={stale.calls}")

            # A normal message on a paused thread would start a new graph turn
            # at the entry point and drop the pending interrupts.  The service
            # must refuse it instead of silently abandoning the step.
            response = await client.post(
                "/chat",
                json={
                    "thread_id": "t-ui-paused",
                    "message": "skip the paused step",
                },
            )
            check("new message on a paused thread is rejected",
                  response.status_code == 409, response.text)
            check("rejected message did not run the graph", stale.calls == 1,
                  f"calls={stale.calls}")
    finally:
        app_facade.build_runtime = original_build
        await runtime_client.aclose()


async def main() -> None:
    await test_serialized_runs()
    await test_stale_resume_is_skipped()
    await test_new_message_is_rejected_while_paused()
    await test_stale_job_result_marks_paused_step_dead()
    await test_deterministic_job_uuid()
    await test_job_result_ack()
    await test_stale_ui_resume_is_skipped()
    print("\nCONCURRENCY / IDEMPOTENCE TESTS: OK")


if __name__ == "__main__":
    asyncio.run(main())
