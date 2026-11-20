"""Headless checks for the job-resume demo; no network required.

    cd learn/job_resume_demo
    python test_demo.py
"""

from __future__ import annotations

import asyncio

from demo import (
    CallbackChannel,
    EventBus,
    JobManager,
    register_resume,
)


async def test_callback_resumes() -> None:
    channel = CallbackChannel()
    manager = JobManager(channel.deliver)

    future = register_resume(channel, "job-ok")
    await manager.submit("job-ok", delay=0.01)
    result = await asyncio.wait_for(future, timeout=1)
    await manager.wait_idle()

    assert result.job_uuid == "job-ok"
    assert result.status == "success"
    assert result.local_dir == "/tmp/job-ok"


async def test_callback_lost_without_registration() -> None:
    channel = CallbackChannel()
    manager = JobManager(channel.deliver)

    await manager.submit("job-late", delay=0.01)
    try:
        await manager.wait_idle()
    except LookupError as exc:
        assert "job-late" in str(exc)
    else:
        raise AssertionError("a callback without registration should not succeed")


async def test_event_bus_replays_for_late_subscriber() -> None:
    bus = EventBus()
    manager = JobManager(bus.publish)

    await manager.submit("job-event", delay=0.01)
    await manager.wait_idle()  # the event is published before anyone waits

    result = await asyncio.wait_for(bus.wait_for("job-event"), timeout=1)
    assert result.job_uuid == "job-event"
    assert result.status == "success"


async def test_event_bus_wakes_waiter() -> None:
    bus = EventBus()
    manager = JobManager(bus.publish)

    waiter = asyncio.create_task(bus.wait_for("job-wait"))
    await asyncio.sleep(0)  # let the waiter subscribe first
    await manager.submit("job-wait", delay=0.01)

    result = await asyncio.wait_for(waiter, timeout=1)
    await manager.wait_idle()
    assert result.status == "success"


async def main() -> None:
    for test in (
        test_callback_resumes,
        test_callback_lost_without_registration,
        test_event_bus_replays_for_late_subscriber,
        test_event_bus_wakes_waiter,
    ):
        await test()
        print(f"ok: {test.__name__}")
    print("JOB RESUME DEMO: OK")


if __name__ == "__main__":
    asyncio.run(main())
