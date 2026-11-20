"""How a finished job wakes the agent, without the HTTP callback hop.

Current catagent flow (two processes):

    StepNode --POST /submit---------------------> JobManager
    StepNode --POST /set-thread-id--------------> JobManager  (resume target)
    JobManager --POST JobResult (callback_url)--> cat-agent   (Command(resume=...))

The last arrow is the one this demo isolates.  A job result is just an event;
the only question is *how it crosses the producer/consumer boundary*:

    callback  : a Python function registered per job_uuid.  Same process only.
    event bus : publish/wait_for by job_uuid.  In production this is a broker
                (Redis Streams, NATS, Kafka) or a durable DB outbox.

Run:

    cd learn/job_resume_demo
    python demo.py

The demo deliberately keeps everything in one asyncio loop and uses no
external packages.  That is exactly the assumption an in-process callback
makes; see README.md for what changes when the two services stay separate.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List


@dataclass(frozen=True)
class JobResult:
    """The payload the agent needs in order to resume the graph."""

    job_uuid: str
    status: str
    local_dir: str = ""
    error_msg: str = ""


# --------------------------------------------------------------------- manager


class JobManager:
    """Stand-in for ``catagent/jobmanager/manager.py``.

    Only ``deliver`` changes between the demos.  In the real code it is an
    HTTP POST to ``callback_url``; here it is a function call or a bus publish.
    """

    def __init__(self, deliver: Callable[[JobResult], Awaitable[None]]) -> None:
        self._deliver = deliver
        self._tasks: set[asyncio.Task] = set()

    async def submit(
        self,
        job_uuid: str,
        *,
        delay: float = 0.05,
        status: str = "success",
    ) -> None:
        """Accept a job and return immediately, like the real ``/submit``."""
        print(f"jobmanager: accepted {job_uuid}")
        task = asyncio.create_task(self._run(job_uuid, delay, status))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run(self, job_uuid: str, delay: float, status: str) -> None:
        # The real manager polls the provider, downloads, validates and runs
        # the checking agent here.  All of that only matters as "not yet".
        await asyncio.sleep(delay)
        print(f"jobmanager: downloaded {job_uuid}; delivering result")
        await self._deliver(
            JobResult(
                job_uuid=job_uuid,
                status=status,
                local_dir=f"/tmp/{job_uuid}",
            )
        )

    async def wait_idle(self) -> None:
        """Wait for every submitted job to finish (demo/test convenience)."""
        if self._tasks:
            await asyncio.gather(*self._tasks)


# ---------------------------------------------------------------- pattern 1: fn


class CallbackChannel:
    """A per-job Python callback.  Works only inside one process.

    Mapping to the real code:

    * ``register(job_uuid, callback)`` is the agent's ``/set-thread-id`` call:
      it tells the manager how to reach the paused graph turn.
    * ``deliver(result)`` is ``JobManager._deliver_callback()``.  In the repo
      it builds a ``JobResult`` request and POSTs it; here it calls the
      callback directly.
    * The callback is the agent's ``_accept_job_result()``: ACK the result and
      schedule ``Command(resume={interrupt_id: ...})`` as a background task.

    There is no network, no serialization and no second server.  The price is
    that producer and consumer must share an address space and an event loop.
    """

    def __init__(self) -> None:
        self._callbacks: Dict[str, Callable[[JobResult], Awaitable[None]]] = {}

    def register(
        self,
        job_uuid: str,
        callback: Callable[[JobResult], Awaitable[None]],
    ) -> None:
        if job_uuid in self._callbacks:
            raise RuntimeError(f"callback already registered for {job_uuid}")
        self._callbacks[job_uuid] = callback

    async def deliver(self, result: JobResult) -> None:
        callback = self._callbacks.pop(result.job_uuid, None)
        if callback is None:
            # The real manager must NOT drop the result here.  It keeps the
            # row in its DB outbox and retries, because the agent may register
            # its resume target after the job already finished.  The demo
            # raises so the race is visible.
            raise LookupError(
                f"job {result.job_uuid} finished before a callback was registered"
            )
        await callback(result)


# ------------------------------------------------------------- pattern 2: event


class EventBus:
    """A tiny topic bus with replay for late subscribers.

    ``publish`` stores the result before waking waiters, so a consumer that
    subscribes after completion still gets it.  That "history" is the part
    people forget when replacing callbacks with a broadcast event: a plain
    ``asyncio.Event`` has no memory, and a result published one millisecond
    before the agent subscribes is simply lost.

    A production bus (Redis Streams/NATS/Kafka) adds:

    * cross-process transport,
    * durable retention (so a consumer restart does not lose the event),
    * consumer groups / acknowledgement (at-least-once delivery).

    The real jobmanager already has the durable half in its SQLite outbox
    (``callback_pending`` + ``result_status``); a bus would replace only the
    transport half.
    """

    def __init__(self) -> None:
        self._history: Dict[str, JobResult] = {}
        self._waiters: Dict[str, List[asyncio.Future]] = {}

    async def publish(self, result: JobResult) -> None:
        # No await between these statements: on one event loop this is atomic
        # with respect to ``wait_for`` below.
        self._history[result.job_uuid] = result
        for waiter in self._waiters.pop(result.job_uuid, []):
            if not waiter.done():
                waiter.set_result(result)

    async def wait_for(self, job_uuid: str) -> JobResult:
        result = self._history.pop(job_uuid, None)
        if result is not None:
            print(f"agent: {job_uuid} had already finished; replayed from history")
            return result

        waiter: asyncio.Future = asyncio.get_running_loop().create_future()
        self._waiters.setdefault(job_uuid, []).append(waiter)
        return await waiter


# ------------------------------------------------------------------------- agent


def register_resume(
    channel: CallbackChannel,
    job_uuid: str,
) -> "asyncio.Future[JobResult]":
    """Register the agent's resume callback and return the awaited result.

    This is the in-process equivalent of emitting a LangGraph interrupt and
    later resuming it.  The real graph does not block on a Future; it persists
    an interrupt and a later ``Command(resume=...)`` re-enters the graph.  The
    trigger for that command is the callback.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future = loop.create_future()

    async def resume(result: JobResult) -> None:
        print(
            f"agent: resumed by callback for {result.job_uuid} "
            f"(status={result.status}, dir={result.local_dir})"
        )
        if not future.done():
            future.set_result(result)

    channel.register(job_uuid, resume)
    return future


# ------------------------------------------------------------------------ demos


async def demo_callback() -> None:
    print("\n--- 1. in-process callback: register, then submit ---")
    channel = CallbackChannel()
    manager = JobManager(channel.deliver)

    future = register_resume(channel, "job-callback")
    await manager.submit("job-callback")
    result = await asyncio.wait_for(future, timeout=1)
    await manager.wait_idle()
    assert result.status == "success"


async def demo_late_callback() -> None:
    print("\n--- 2. in-process callback: result arrives too early ---")
    channel = CallbackChannel()
    manager = JobManager(channel.deliver)

    await manager.submit("job-too-early")
    try:
        await manager.wait_idle()
    except LookupError as exc:
        print(f"jobmanager: {exc}")
        print("agent: never resumed  <-- a durable outbox is what prevents this")


async def demo_event_bus() -> None:
    print("\n--- 3. event bus: result arrives before the subscriber ---")
    bus = EventBus()
    manager = JobManager(bus.publish)

    await manager.submit("job-event")
    await manager.wait_idle()  # deliberately finish before the agent waits

    print("agent: starting to wait after the job is already done")
    result = await asyncio.wait_for(bus.wait_for("job-event"), timeout=1)
    print(
        f"agent: resumed from event for {result.job_uuid} "
        f"(status={result.status}, dir={result.local_dir})"
    )


async def main() -> None:
    await demo_callback()
    await demo_late_callback()
    await demo_event_bus()
    print(
        "\nTakeaway: the callback/event replaces only the resume trigger.\n"
        "Durability (result persisted before delivery, at-least-once, no lost\n"
        "wakeup) is a separate property that both patterns still need."
    )


if __name__ == "__main__":
    asyncio.run(main())
