"""Shared data models for the cat-agent application layer.

The HTTP service and the debug CLI exchange the same plain dictionary shape.
Keeping the shape in one place makes the boundary explicit and prevents the
transports from drifting apart.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel


class StreamPayload(TypedDict, total=False):
    """One UI-friendly event produced by the agent workflow."""

    type: Literal["user", "agent", "tool", "interrupt", "custom"]

    # ToolMessage content, an Interrupt value, arbitrary custom data, or a
    # plain string for ordinary messages.
    message: Any

    # LangChain messages always have an id; custom stream values are given one
    # by the payload converter.
    message_id: str

    # AI messages may contain tool calls.
    tool_calls: List[Dict[str, Any]]

    tool_call_id: str
    tool_call_name: str | None

    interrupt_id: str
    interrupt_response_schema: Any

    node: str | None
    namespace: tuple[str, ...]

    # Which workflow/step produced the payload.  For the orchestrator this is
    # just the thread id; for a step node it is "<thread_id>:<step_id>".
    source: str | None

    # The outer graph thread id, useful for resuming from a stateless client.
    thread_id: str | None


class ChatRequest(BaseModel):
    """Request body for ``POST /chat`` (also used for jobmanager callbacks)."""

    # When present, the request resumes a pending interrupt.
    interrupt_resume: Optional[Dict[str, Any]] = None

    # An ordinary user message.
    message: Optional[str] = None

    # Optional session id.  A normal chat without it starts a new thread.
    thread_id: Optional[str] = None

    # Fields posted by jobmanager.manager._post_callback (JobResult).
    job_uuid: Optional[str] = None
    interrupt_id: Optional[str] = None
    local_dir: Optional[str] = None
    status: Optional[Literal["success", "error"]] = None
    error_msg: Optional[str] = None

    # A client that polls jobmanager itself (for example the debug CLI) sets
    # this to False so jobmanager does not also POST a resume callback.
    register_jobs: bool = True
