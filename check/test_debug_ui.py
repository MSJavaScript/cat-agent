"""Debug event log and single-page UI routes.

No model is called: ``build_runtime`` and ``_stream_chat`` are replaced with
small fakes.  The test checks the pieces the browser relies on: the bounded
per-thread event log, the jobmanager bucket, the HTML route, and the unchanged
``POST /`` callback route.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from langgraph.types import Interrupt

import catagent.app as app_facade
import catagent.app.service as service
from catagent.app.events import JOBMANAGER_SOURCE, DebugEventLog
from catagent.app.payloads import _interrupt_payloads


class FakeDB:
    async def close(self) -> None:
        pass


def test_event_log() -> None:
    log = DebugEventLog(max_events_per_thread=3)
    for index in range(5):
        log.append("t", {"n": index})

    events = log.since("t", 0)
    assert [event["seq"] for event in events] == [3, 4, 5], events
    assert [event["payload"]["n"] for event in events] == [2, 3, 4], events
    assert log.latest_seq("t") == 5
    assert [event["seq"] for event in log.since("t", 4)] == [5]
    assert log.since("other", 0) == []

    # Sequences keep increasing even after old payloads are trimmed.
    assert log.append("t", {"n": 99})["seq"] == 6


def test_interrupt_source() -> None:
    value = {
        "kind": "AskForArgs",
        "message": "need input",
        "payload": {},
        "_debug_source": "tid:plan-0:step-1",
    }
    payloads = _interrupt_payloads(Interrupt(value=value, id="i1"), None, ())
    assert payloads[0]["source"] == "tid:plan-0:step-1"
    assert "_debug_source" not in payloads[0]["message"]

    # Without the private key the caller-provided source is preserved.
    payloads = _interrupt_payloads(
        Interrupt(value={"kind": "AgentInterrupt"}, id="i2"), "orig", ()
    )
    assert payloads[0]["source"] == "orig"


async def _test_routes() -> None:
    runtime_client = httpx.AsyncClient()

    class FakeGraph:
        pass

    async def fake_build_runtime(**kwargs: Any):
        return FakeGraph(), runtime_client, FakeDB()

    async def fake_stream_chat(
        *,
        catagent: Any,
        agent_input: Any,
        thread_id: str,
        job_manager_url: str,
        http_client: httpx.AsyncClient,
        register_jobs: bool = True,
    ):
        yield {
            "type": "agent",
            "message": "orchestrator says hi",
            "message_id": "m1",
            "source": thread_id,
            "thread_id": thread_id,
        }
        yield {
            "type": "agent",
            "message": "step says hi",
            "message_id": "m2",
            "source": f"{thread_id}:plan-0:step-1",
            "thread_id": thread_id,
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
            page = await client.get("/")
            assert page.status_code == 200
            assert "Jobmanager agent" in page.text
            assert "/events/" in page.text
            assert 'id="jumpChips"' in page.text
            assert "updateJumpbar" in page.text

            response = await client.post(
                "/jobmanager/messages",
                json={
                    "role": "agent",
                    "message": "preparing inputs",
                    "phase": "preprocessing",
                },
            )
            assert response.status_code == 200, response.text
            jm_events = (
                await client.get(f"/events/{JOBMANAGER_SOURCE}?after=0")
            ).json()["events"]
            assert len(jm_events) == 1, jm_events
            payload = jm_events[0]["payload"]
            assert payload["type"] == "jobmanager"
            assert payload["role"] == "agent"
            assert payload["message"] == "preparing inputs"
            assert payload["phase"] == "preprocessing"

            async with client.stream(
                "POST",
                "/chat",
                json={
                    "message": "hello",
                    "thread_id": "t1",
                    "register_jobs": False,
                },
            ) as chat:
                assert chat.status_code == 200
                assert chat.headers["X-Thread-Id"] == "t1"
                body = await chat.aread()
            assert b"orchestrator says hi" in body

            events = (await client.get("/events/t1?after=0")).json()["events"]
            assert [event["payload"]["message"] for event in events] == [
                "orchestrator says hi",
                "step says hi",
            ]
            assert events[0]["payload"]["source"] == "t1"
            assert events[1]["payload"]["source"] == "t1:plan-0:step-1"

            after = (await client.get("/events/t1?after=1")).json()["events"]
            assert len(after) == 1 and after[0]["seq"] == 2, after

            # Adding GET / must not take over the jobmanager callback on POST /.
            async with client.stream(
                "POST",
                "/",
                json={
                    "message": "via root",
                    "thread_id": "t2",
                    "register_jobs": False,
                },
            ) as chat:
                assert chat.status_code == 200
                await chat.aread()
            t2_events = (await client.get("/events/t2?after=0")).json()["events"]
            assert t2_events, t2_events
            assert t2_events[0]["payload"]["source"] == "t2"
    finally:
        app_facade.build_runtime = original_build
        service._stream_chat = original_stream
        await runtime_client.aclose()


def main() -> None:
    test_event_log()
    test_interrupt_source()
    asyncio.run(_test_routes())
    print("DEBUG UI TESTS: OK")


if __name__ == "__main__":
    main()
