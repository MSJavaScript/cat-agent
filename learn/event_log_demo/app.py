"""A tiny FastAPI model of catagent's debug event log.

Why this exists
---------------
``catagent/app/events.py`` exists because not every turn starts in the browser.
A jobmanager callback resumes the graph through a server-side ``POST /chat``;
the browser never sees that request, so it has no SSE response to read.  The
server therefore records every payload in a small in-memory log, and the page
*polls* one endpoint for both kinds of traffic.

Run it:
    cd learn/event_log_demo
    uvicorn app:app --reload --port 8010

Then open http://localhost:8010 and follow ``README.md``.

The ``DebugEventLog`` below is a condensed copy of the real one so this demo
has no catagent dependencies.  The real class also JSON-sanitises payloads and
keeps its own lock; the important parts reproduced here are:

* one sequence counter per bucket (thread id, or the synthetic jobmanager id),
* a bounded deque per bucket and a bounded number of buckets (LRU eviction),
* a ``generation`` token identifying this log instance.

The generation token is what makes a browser notice a server restart: sequence
numbers start again at 1, and a cursor left over from the previous process
would otherwise wait forever for ``seq > cursor``.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections import OrderedDict, deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

JOBMANAGER_SOURCE = "__jobmanager__"
HTML_PATH = Path(__file__).with_name("index.html")


# --------------------------------------------------------------------------- log


class DebugEventLog:
    """Bounded, thread-safe, monotonically sequenced event buckets."""

    def __init__(
        self,
        max_events_per_thread: int = 500,
        max_threads: int = 16,
    ) -> None:
        self._max_events = max_events_per_thread
        self._max_threads = max(1, max_threads)
        self._events: Dict[str, Deque[Dict[str, Any]]] = OrderedDict()
        self._seq: Dict[str, int] = {}
        self._generation = uuid.uuid4().hex
        self._lock = asyncio.Lock()

    @property
    def generation(self) -> str:
        return self._generation

    async def append(self, thread_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        async with self._lock:
            seq = self._seq.get(thread_id, 0) + 1
            self._seq[thread_id] = seq
            event = {"seq": seq, "thread_id": thread_id, "payload": payload}
            bucket = self._events.get(thread_id)
            if bucket is None:
                bucket = deque(maxlen=self._max_events)
                self._events[thread_id] = bucket
            else:
                self._events.move_to_end(thread_id)
            bucket.append(event)
            while len(self._events) > self._max_threads:
                old_thread, _ = self._events.popitem(last=False)
                self._seq.pop(old_thread, None)
            return event

    async def read(self, thread_id: str, after: int = 0) -> Dict[str, Any]:
        async with self._lock:
            events = [
                event
                for event in self._events.get(thread_id, ())
                if event["seq"] > after
            ]
            return {
                "events": events,
                "latest_seq": self._seq.get(thread_id, 0),
                "generation": self._generation,
            }


# ------------------------------------------------------------------------- app


app = FastAPI(title="event log demo")
app.state.log = DebugEventLog()


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class CallbackRequest(BaseModel):
    message: Optional[str] = None


class JobmanagerMessage(BaseModel):
    role: str = "agent"
    message: str
    phase: Optional[str] = None


def _payload(thread_id: str, message: str, kind: str = "agent") -> Dict[str, Any]:
    return {
        "type": kind,
        "message": message,
        "message_id": str(uuid.uuid4()),
        "source": thread_id,
        "thread_id": thread_id,
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HTML_PATH)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Normal turn: the payloads go to the log *and* down the SSE response.

    The demo page deliberately ignores the response body (it only drains it),
    exactly like ``debug_ui.html`` does.  Everything it shows comes from
    polling ``/events``.
    """
    thread_id = req.thread_id or "demo-" + uuid.uuid4().hex[:6]

    async def stream():
        messages = [
            f"received: {req.message!r}",
            "thinking (simulated)…",
            "turn done. Now click “server-side callback”.",
        ]
        for message in messages:
            payload = _payload(thread_id, message)
            await app.state.log.append(thread_id, payload)
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Thread-Id": thread_id,
        },
    )


@app.post("/callback/{thread_id}")
async def callback(thread_id: str, req: CallbackRequest) -> Dict[str, Any]:
    """Out-of-band turn: no browser request is involved.

    This stands in for ``JobManager._deliver_callback()`` posting a JobResult
    to the cat-agent service.  The HTTP response returns immediately; the work
    happens in a background task and only becomes visible through the log.
    """
    message = req.message or "job finished: total energy = -43.357165 eV"
    asyncio.create_task(_server_resume(thread_id, message))
    return {
        "accepted": True,
        "thread_id": thread_id,
        "note": "the browser never sees this response; watch the poll view",
    }


async def _server_resume(thread_id: str, message: str) -> None:
    await asyncio.sleep(1.5)
    for text in ("resuming graph after jobmanager callback…", message):
        await app.state.log.append(thread_id, _payload(thread_id, text))
        await asyncio.sleep(0.6)


@app.post("/jobmanager/messages")
async def jobmanager_message(req: JobmanagerMessage) -> Dict[str, Any]:
    """Jobmanager progress has no cat-agent thread id -> synthetic bucket."""
    payload = {
        "type": "jobmanager",
        "role": req.role,
        "message": req.message,
        "phase": req.phase,
        "message_id": str(uuid.uuid4()),
        "source": JOBMANAGER_SOURCE,
    }
    event = await app.state.log.append(JOBMANAGER_SOURCE, payload)
    return {"ok": True, "seq": event["seq"]}


@app.get("/events/{thread_id}")
async def events(thread_id: str, after: int = 0) -> Dict[str, Any]:
    """The one endpoint the page polls, for both kinds of traffic."""
    return {"thread_id": thread_id, **await app.state.log.read(thread_id, after)}


@app.post("/_debug/restart")
async def restart_log() -> Dict[str, Any]:
    """Replace the log, simulating a process restart.

    New generation, sequence numbers back to 1.  Watch the page: it clears and
    replays from zero instead of freezing on its old cursor.
    """
    old_generation = app.state.log.generation
    app.state.log = DebugEventLog()
    return {
        "old_generation": old_generation,
        "new_generation": app.state.log.generation,
    }
