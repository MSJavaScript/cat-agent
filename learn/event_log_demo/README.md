# Event-log demo

A ~250-line FastAPI app that reproduces the mechanism behind
`catagent/app/events.py` and the polling half of `catagent/app/debug_ui.html`.
It exists so the data flow can be watched directly, without a model, a job
manager, or a real cluster job.

## Run

```bash
cd learn/event_log_demo
uvicorn app:app --reload --port 8010
# open http://localhost:8010
```

No network or credentials. `python test_demo.py` runs a headless smoke test.

## Files

| file | what it is |
| --- | --- |
| `app.py` | the HTTP service: `POST /chat` (SSE + log), `POST /callback/{thread}` (server-side resume), `POST /jobmanager/messages` (synthetic bucket), `GET /events/{thread}?after=N`, `POST /_debug/restart` |
| `index.html` | the page: polls `/events`, renders both buckets, detects a replaced log |
| `test_demo.py` | headless check of the same flow |

The `DebugEventLog` in `app.py` is a condensed copy of the real one so the demo
stands alone. The real class is `catagent/app/events.py`; it additionally
sanitises payloads through JSON and uses `threading.Lock` instead of
`asyncio.Lock`.

## The mechanism in four points

1. **Every payload goes through one place.** In catagent, `_stream_chat()`
   yields a payload, `service.py` appends it to the log, then yields it as SSE.
   Same bytes, two delivery paths.
2. **Sequence numbers are per bucket.** A bucket is a cat-agent thread id or
   the synthetic `__jobmanager__` one. `after=N` means "I have already rendered
   through N"; the server returns only `seq > N`.
3. **Some turns are not started by the browser.** When jobmanager finishes a
   job it posts the result to `/chat` server-side. The browser has no SSE
   stream for that request, so polling is the only way it learns what happened.
4. **The log is volatile and bounded.** Restarting the process loses everything
   and restarts sequence numbers at 1. To keep that from freezing a page whose
   cursor is already at 7, `read()` returns:
   - `generation`: identity of this log instance;
   - `latest_seq`: highest sequence ever assigned to the bucket.

   A client must replay from `after=0` when the generation changed **or** when
   `after > latest_seq` (also true for an evicted bucket).

## What to try

| action | what it demonstrates |
| --- | --- |
| type a message, **send** | the SSE body and the poll view carry identical payloads; the page ignores the former (`await response.text()` with no parsing) |
| in devtools, look at `POST /chat` | its response stream ends, yet the page keeps updating — rendering is 100% `/events` polling |
| **server-side callback** | two new lines appear although the browser never made that request; this is the jobmanager resume path |
| **jobmanager message** | progress lands in a separate bucket, because it has no thread id |
| **restart server log**, then send again | the page clears, the status shows a new generation, and sequence numbers start at 1 again |
| **new thread**, then go back | a fresh cursor replays whatever the server still retains for that thread |
| create 17+ threads | the least recently used bucket is evicted; revisiting it hits the `after > latest_seq` rule |

The status line shows exactly what the page knows:
`generation <id> · thread cursor <rendered>/<latest> · jm cursor <n>`.

## Mapping back to catagent

| demo | real |
| --- | --- |
| `DebugEventLog` | `catagent/app/events.py` |
| `POST /chat` | `catagent/app/service.py` → `chat()` + `_stream_chat()` |
| `GET /events/{thread_id}` | `service.py` → `events_route()` |
| `POST /jobmanager/messages` | `service.py` → `jobmanager_messages()` |
| `POST /callback/{thread_id}` | `JobManager._deliver_callback()` in `jobmanager/manager.py` |
| polling + reset detection | `pollThread()` / `pollJobmanager()` / `serverResetRequired()` in `catagent/app/debug_ui.html` |

`/events` is a debug convenience only. Real API consumers read the SSE stream
and never touch this log.
