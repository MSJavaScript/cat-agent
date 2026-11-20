# Job-resume demo: callback vs. event bus

A small, dependency-free answer to one question:

> When jobmanager has downloaded a job, can it resume the agent with an
> in-process event/callback instead of POSTing a `JobResult` back to
> `catagent/app/service.py`?

**Short answer:** yes, if jobmanager and agent live in the same process. If
they stay two processes, a "callback function" cannot cross the boundary:
an event still has to travel over *some* transport, and HTTP is already the
simplest one. The current callback is not redundant indirection; it is the
resume trigger plus a durable outbox for a distributed system.

Run the demo and its checks:

```bash
cd learn/job_resume_demo
python demo.py
python test_demo.py
```

## What the current code actually does

```text
StepNode                     JobManager                       cat-agent
   |                              |                                |
   |-- POST /submit ------------->|  persist job, return accepted  |
   |                              |                                |
   |  LangGraph interrupt         |                                |
   |-- POST /set-thread-id ------>|  store (thread_id, interrupt_id)|
   |                              |                                |
   |                              |  poll/download/check           |
   |                              |-- POST JobResult ------------->| 202 ACK
   |                              |                                | resume graph
   |                              |<-------- outbox marked sent ---|
```

Four separate jobs are being done by those two POSTs:

1. **Trigger** — something must wake a paused LangGraph interrupt. The
   interrupt itself cannot fire; it is passive state in a checkpoint.
2. **Routing** — a result belongs to one `job_uuid`, which belongs to one
   paused graph turn (`thread_id`, `interrupt_id`).
3. **Delivery guarantee** — jobmanager may finish before the agent has
   registered its interrupt, or the agent may be down. The result is persisted
   (`result_status`) and kept in an outbox (`callback_pending`) until a 2xx.
4. **Failure isolation** — the jobmanager must not be able to kill the
   agent's graph run, and a slow agent turn must not block job polling. That is
   why `_accept_job_result()` returns `202` immediately and runs the resume as
   a detached task.

The two HTTP calls are the distributed version of (1)+(2). The DB columns and
the ACK/retry loop are (3)+(4). Replacing HTTP with an event does not remove
(3)+(4); it only changes the wire.

## Options

| option | works across processes? | extra infrastructure | lose-a-result risk | comment |
| --- | --- | --- | --- | --- |
| Current HTTP callback | yes | none (both sides already are HTTP servers) | low: durable outbox + at-least-once | smallest thing that satisfies all four properties |
| In-process Python callback / `Future` | no | none | high unless results are persisted | best when jobmanager is embedded in the agent process |
| Event bus (Redis Streams, NATS, Kafka) | yes | broker + client library | depends on retention/ack; can be low | useful for fan-out, multiple consumers, replay, multiple agent replicas |
| Poll `GET /status/{job_uuid}` | yes | none | low | no inbound endpoint on the agent; adds latency and poll load. The debug CLI already uses this (`catagent/app/jobs.py`) |
| In-process DB outbox + dispatcher | no (same process) | none | low if the DB is durable | callback + persistence, still single-process |

### 1. In-process callback (this is what `demo.py` shows first)

```python
channel = CallbackChannel()
manager = JobManager(channel.deliver)   # instead of callback_url + httpx

future = register_resume(channel, job_uuid)  # replaces POST /set-thread-id
await manager.submit(job_uuid)               # still a normal submit
result = await future                        # replaces the callback POST
```

The jobmanager calls the function registered for `job_uuid` directly. There is
no serialization, no `/job-result` endpoint and no second HTTP client. This
requires:

- one OS process and one asyncio event loop;
- a shared reference to the `JobManager`;
- the agent's resume code callable as a coroutine;
- the same durability story if the shared process can restart (the DB outbox
  should still be written *before* invoking the callback).

### 2. Event bus (what "event system" usually means)

```python
bus = EventBus()                 # Redis Streams / NATS in production
manager = JobManager(bus.publish)
await manager.submit(job_uuid)
result = await bus.wait_for(job_uuid)
```

`publish(job_uuid, result)` and `wait_for(job_uuid)` replace the two POSTs.
The important detail is **history**: `EventBus` stores a result before waking
waiters, so an agent that subscribes *after* the download still receives it.
A bare `asyncio.Event` or a broadcast pub/sub topic without retention does not.
`demo.py` scenario 3 demonstrates exactly this late-subscriber case.

A real broker adds cross-process transport and retention, but also a new
service to run, credentials, serialization, reconnection and monitoring. If
the only requirement is "wake this one agent when this one job finishes", HTTP
already provides that with fewer moving parts.

## What the demo shows

| scenario | result |
| --- | --- |
| `demo_callback()` | callback registered first; the jobmanager invokes it and the agent resumes |
| `demo_late_callback()` | job finishes first; the callback is lost because there is no outbox |
| `demo_event_bus()` | job finishes first; the bus replays the stored result to the late subscriber |

`test_demo.py` checks the same three behaviours plus the normal wake-up path.
It needs no model, no jobmanager server and no network.

## Mapping back to the real code

| demo | real code |
| --- | --- |
| `JobManager.submit()` | `catagent/jobmanager/manager.py` → `submit_job()` |
| `CallbackChannel.register()` | `catagent/app/streaming.py` → `_register_job_submission()` → `POST /set-thread-id` |
| `CallbackChannel.deliver()` | `JobManager._deliver_callback()` → POST `callback_url` |
| `register_resume()` callback | `catagent/app/service.py` → `_accept_job_result()` → detached `_stream_chat(Command(resume=...))` |
| `EventBus.wait_for()` | no direct equivalent; closest is the CLI's `poll_job_results()` in `catagent/app/jobs.py` |
| `EventBus` history | `JobRecord.result_status` + `callback_pending` (the durable outbox) |

## Invariants to preserve if this is implemented for real

These are the reasons the current HTTP callback is more than a redundant POST.
Any replacement must keep them.

1. **Correlate, do not broadcast blindly.** Every result carries `job_uuid`.
   A listener must only resume the graph turn whose interrupt value contains
   that `job_uuid` / `interrupt_id`. Resuming the "last thread" (or all
   listeners) is the bug `job_uuid` exists to prevent.
2. **Persist before delivering.** Write `result_status`/`result_error` first,
   then enqueue delivery. A crash between download and callback must not lose
   or re-decide the result. The current `FinalizeState` state machine exists
   for this.
3. **At-least-once, then deduplicate on the agent.** A retry, a reconnect or a
   broker redelivery will call the resume path twice. `_stream_chat()` already
   checks `expected_interrupt_id` against the pending interrupts and skips a
   stale resume; keep that guard.
4. **Handle result-before-subscriber.** The agent may register its interrupt
   after the job has completed (the outer interrupt is emitted after the
   StepNode tool call returns; `submit_job` itself may be slow). A callback
   implementation must not drop the result in that window; the demo shows the
   failure, and the DB outbox is the fix.
5. **Do not let delivery own the job.** Whether the callback is a function or
   an HTTP request, run the graph turn as a separate task with its own
   failure handling. The jobmanager's polling loop must continue even if the
   agent is slow or down.
6. **Survive an independent restart.** With HTTP, either side can restart and
   recover from its own database. An in-process callback makes them fail and
   restart together; a broker reintroduces the broker as a dependency. Choose
   deliberately which failure domain you want.
7. **Bound it.** Callbacks can hang. Use timeouts, a bounded queue, and a
   dead-letter/outbox path; "fire and forget" is not enough for a durable
   result.

## If you want to change the real code later

The smallest refactor that keeps both deployments working is to introduce a
delivery seam instead of deleting HTTP:

```python
# jobmanager side (sketch)
class ResultDelivery(Protocol):
    async def register(self, job_uuid: str, target: ResumeTarget) -> None: ...
    async def deliver(self, result: JobResult) -> bool: ...

class HttpCallbackDelivery:   # current POST + 2xx ACK
class InProcessDelivery:      # calls a registered coroutine
class BrokerDelivery:         # publishes to a durable stream
```

`JobManager` would keep its existing DB outbox and call `deliver()` from
`_deliver_callback()`; only the transport under that method changes. The agent
would keep `_accept_job_result()`; only how the result arrives changes. That
way the durable half and the transport half stay separable.

**Do not** replace the callback with a plain `asyncio.Event` while the two
services remain separate processes: the event object lives in one process and
does not exist in the other. And do not replace it with an in-memory pub/sub
without replay unless you are certain the subscriber always registers before
the job can finish — the current registration race (result ready before
`/set-thread-id`) proves that assumption is false.
