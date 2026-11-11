"""Smoke test for the event-log demo; no network or browser required.

    cd learn/event_log_demo
    python test_demo.py
"""

from __future__ import annotations

import asyncio

import httpx

from app import JOBMANAGER_SOURCE, app


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://demo") as client:
        response = await client.get("/")
        assert response.status_code == 200
        assert "serverResetRequired" in response.text

        # A normal turn writes three payloads and returns them over SSE.
        chat = await client.post("/chat", json={"message": "hello"})
        tid = chat.headers["X-Thread-Id"]
        assert "received: 'hello'" in chat.text

        read = (await client.get(f"/events/{tid}?after=0")).json()
        assert read["latest_seq"] == 3, read
        assert [event["seq"] for event in read["events"]] == [1, 2, 3]
        generation = read["generation"]

        # The callback is server-side only; the browser would not have an SSE
        # response to read, but the payloads still land in the log.
        await client.post(f"/callback/{tid}", json={})
        await asyncio.sleep(3.0)
        read = (await client.get(f"/events/{tid}?after=3")).json()
        assert [event["seq"] for event in read["events"]] == [4, 5], read
        assert "total energy" in read["events"][-1]["payload"]["message"]

        await client.post(
            "/jobmanager/messages",
            json={"role": "agent", "message": "preparing inputs", "phase": "prep"},
        )
        jm = (await client.get(f"/events/{JOBMANAGER_SOURCE}?after=0")).json()
        assert jm["latest_seq"] == 1
        assert jm["events"][0]["payload"]["message"] == "preparing inputs"

        # A restart resets sequence numbers and changes the generation.  The
        # old cursor (5) is now ahead of latest_seq (0), which is exactly what
        # the page checks before replaying from zero.
        restarted = (await client.post("/_debug/restart")).json()
        assert restarted["new_generation"] != generation
        stale = (await client.get(f"/events/{tid}?after=5")).json()
        assert stale["latest_seq"] == 0
        assert stale["events"] == []
        assert stale["generation"] == restarted["new_generation"]

        chat = await client.post("/chat", json={"message": "again", "thread_id": tid})
        assert chat.headers["X-Thread-Id"] == tid
        fresh = (await client.get(f"/events/{tid}?after=0")).json()
        assert [event["seq"] for event in fresh["events"]] == [1, 2, 3], fresh

    print("EVENT LOG DEMO: OK")


if __name__ == "__main__":
    asyncio.run(main())
