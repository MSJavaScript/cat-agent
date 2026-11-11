"""In-memory event log backing the single-page debug UI.

The HTTP service is the only place where every payload is produced: normal
turns stream through ``POST /chat``, and a resumed job also enters through
``POST /chat`` when jobmanager posts its callback.  Keeping a small per-thread
log here lets a browser poll *one* endpoint and see both kinds of traffic,
instead of trying to consume the SSE response of a request it did not make.

The log is intentionally volatile and bounded: it exists only for debugging
and is rebuilt when the process restarts.

Because a restart resets every sequence number, each ``read()`` also returns a
``generation`` token for the log instance.  A polling client compares it with
the token it saw before: a change means the server was restarted, its cursor
now points past the end of the stream, and it must replay from zero instead of
waiting for a sequence number that will never be reached.  The same rule
applies when a whole thread bucket was evicted to stay under ``max_threads``.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List

__all__ = ["JOBMANAGER_SOURCE", "DebugEventLog"]

# The debug UI shows jobmanager agent progress in its own panel.  Jobmanager
# posts those messages to ``front_end_url`` without a cat-agent thread id, so
# they all share this synthetic source.
JOBMANAGER_SOURCE = "__jobmanager__"


class DebugEventLog:
    """Bounded, thread-safe, monotonically sequenced event buckets.

    Two bounds keep a long debugging session from growing without limit:
    ``max_events_per_thread`` trims the oldest payloads of one bucket, and
    ``max_threads`` evicts the least recently written buckets entirely.
    """

    def __init__(
        self,
        max_events_per_thread: int = 2000,
        max_threads: int = 32,
    ) -> None:
        self._max_events = max_events_per_thread
        self._max_threads = max(1, max_threads)
        self._events: Dict[str, Deque[Dict[str, Any]]] = OrderedDict()
        self._seq: Dict[str, int] = {}
        self._generation = uuid.uuid4().hex
        self._lock = threading.Lock()

    @staticmethod
    def _plain(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Store a detached JSON-safe copy.

        Custom graph values are arbitrary Python objects.  The SSE route uses
        ``default=str``; doing the same round-trip here keeps ``/events``
        serialisable even when a node writes something exotic.
        """
        try:
            return json.loads(json.dumps(payload, ensure_ascii=False, default=str))
        except Exception:
            # Never break the live stream because a debug payload is circular
            # or otherwise unserialisable.
            if not isinstance(payload, dict):
                return {"type": "custom", "message": repr(payload)}
            return {
                "type": str(payload.get("type", "custom")),
                "message": repr(payload.get("message")),
                "message_id": str(payload.get("message_id", "")),
                "source": str(payload.get("source")) if payload.get("source") else None,
                "thread_id": str(payload.get("thread_id")) if payload.get("thread_id") else None,
            }

    def append(self, thread_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Append one payload and return the stored event (with its sequence)."""
        payload = self._plain(payload)
        with self._lock:
            seq = self._seq.get(thread_id, 0) + 1
            self._seq[thread_id] = seq
            event = {"seq": seq, "thread_id": thread_id, "payload": payload}
            bucket = self._events.get(thread_id)
            if bucket is None:
                bucket = deque(maxlen=self._max_events)
                self._events[thread_id] = bucket
            else:
                # Recency decides which bucket is evicted next.
                self._events.move_to_end(thread_id)
            bucket.append(event)
            self._evict_threads()
            return event

    def _evict_threads(self) -> None:
        """Drop least recently used buckets.  Caller holds ``self._lock``."""
        while len(self._events) > self._max_threads:
            old_thread, _ = self._events.popitem(last=False)
            # Forget its sequence too: a thread that comes back starts at 1,
            # which the client detects as "cursor ahead of the stream".
            self._seq.pop(old_thread, None)

    def since(self, thread_id: str, after: int = 0) -> List[Dict[str, Any]]:
        """Return events with ``seq > after``, oldest first.

        A caller that remembers the last sequence it rendered gets only new
        events; ``after=0`` replays the retained history for a thread.
        """
        return self.read(thread_id, after)["events"]

    def latest_seq(self, thread_id: str) -> int:
        return self.read(thread_id, 0)["latest_seq"]

    def read(self, thread_id: str, after: int = 0) -> Dict[str, Any]:
        """Return one consistent poll result.

        ``generation`` identifies this log instance.  ``latest_seq`` is the
        highest sequence ever assigned to the bucket, even if the payloads
        themselves were trimmed.  A client whose ``after`` is greater than
        ``latest_seq`` has a cursor from a previous server process (or from an
        evicted bucket) and must reset it to zero.
        """
        with self._lock:
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
