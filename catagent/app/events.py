"""In-memory event log backing the single-page debug UI.

The HTTP service is the only place where every payload is produced: normal
turns stream through ``POST /chat``, and a resumed job also enters through
``POST /chat`` when jobmanager posts its callback.  Keeping a small per-thread
log here lets a browser poll *one* endpoint and see both kinds of traffic,
instead of trying to consume the SSE response of a request it did not make.

The log is intentionally volatile and bounded: it exists only for debugging
and is rebuilt when the process restarts.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from typing import Any, Deque, Dict, List

__all__ = ["JOBMANAGER_SOURCE", "DebugEventLog"]

# The debug UI shows jobmanager agent progress in its own panel.  Jobmanager
# posts those messages to ``front_end_url`` without a cat-agent thread id, so
# they all share this synthetic source.
JOBMANAGER_SOURCE = "__jobmanager__"


class DebugEventLog:
    """Bounded, thread-safe, monotonically sequenced event buckets."""

    def __init__(self, max_events_per_thread: int = 2000) -> None:
        self._max_events = max_events_per_thread
        self._events: Dict[str, Deque[Dict[str, Any]]] = {}
        self._seq: Dict[str, int] = {}
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
            bucket = self._events.setdefault(
                thread_id, deque(maxlen=self._max_events)
            )
            bucket.append(event)
            return event

    def since(self, thread_id: str, after: int = 0) -> List[Dict[str, Any]]:
        """Return events with ``seq > after``, oldest first.

        A caller that remembers the last sequence it rendered gets only new
        events; ``after=0`` replays the retained history for a thread.
        """
        with self._lock:
            return [
                event
                for event in self._events.get(thread_id, ())
                if event["seq"] > after
            ]

    def latest_seq(self, thread_id: str) -> int:
        with self._lock:
            return self._seq.get(thread_id, 0)
