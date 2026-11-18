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
from types import SimpleNamespace
from typing import Any, List

import httpx

from catagent.app.streaming import _stream_chat
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


class FakeSnapshot:
    def __init__(self, interrupt_ids: List[str]) -> None:
        self.tasks = [
            SimpleNamespace(interrupts=[SimpleNamespace(id=interrupt_id)])
            for interrupt_id in interrupt_ids
        ]
        self.interrupts = ()
        self.values = {}
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

        def make_request(tool_call_id: str):
            return SimpleNamespace(
                tool=submit_tool,
                tool_call={
                    "name": "submit_job",
                    "id": tool_call_id,
                    "args": {
                        "job_description": "relax",
                        "local_dir": "/tmp/step1",
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
    finally:
        app_facade.build_runtime = original_build
        await runtime_client.aclose()


async def main() -> None:
    await test_serialized_runs()
    await test_stale_resume_is_skipped()
    await test_deterministic_job_uuid()
    await test_job_result_ack()
    await test_stale_ui_resume_is_skipped()
    print("\nCONCURRENCY / IDEMPOTENCE TESTS: OK")


if __name__ == "__main__":
    asyncio.run(main())
