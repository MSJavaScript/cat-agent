"""Convert LangGraph stream chunks into :class:`StreamPayload` dictionaries.

The workflow is built from two nested compiled graphs.  Their stream chunks are
forwarded through ``get_stream_writer()`` with a small envelope, so this module
normalises both the outer graph's chunks and the forwarded inner chunks.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, List, Tuple

from langchain.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langgraph.types import Interrupt
from pydantic import BaseModel

from .models import StreamPayload

# These tools return skill documents, file lists and summaries that are useful
# to the model but unreadable in a chat transcript.  Only the streamed payload
# is shortened; the ToolMessage in graph state stays intact.
_SKILL_TOOL_NAMES = frozenset(
    {
        "get_skill_summaries",
        "get_skill_content",
        "load_skill_from_filesystem",
    }
)
_SKILL_TOOL_PAYLOAD_LIMIT = 100


def _payload_message_content(message: ToolMessage) -> Any:
    """Shorten noisy skill-tool output for display only."""
    if message.name not in _SKILL_TOOL_NAMES:
        return message.content
    if isinstance(message.content, str):
        return message.content[:_SKILL_TOOL_PAYLOAD_LIMIT]
    return json.dumps(
        message.content, ensure_ascii=False, default=str
    )[:_SKILL_TOOL_PAYLOAD_LIMIT]


def _jsonable(value: Any) -> Any:
    """Convert pydantic / LangChain objects into plain Python data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")
    return value


def _message_id(message: BaseMessage) -> str:
    return str(message.id) if message.id is not None else str(uuid.uuid4())


def _message_payload(
    message: BaseMessage,
    source: str | None = None,
    namespace: Tuple[str, ...] = (),
    node: str | None = None,
) -> StreamPayload:
    """Map one LangChain message onto the public StreamPayload shape."""
    if isinstance(message, AIMessage):
        return {
            "type": "agent",
            "message": message.content,
            "message_id": _message_id(message),
            "tool_calls": list(message.tool_calls or []),
            "source": source,
            "node": node,
            "namespace": namespace,
        }
    if isinstance(message, ToolMessage):
        return {
            "type": "tool",
            "message": _payload_message_content(message),
            "message_id": _message_id(message),
            "tool_call_id": message.tool_call_id,
            "tool_call_name": message.name,
            "source": source,
            "node": node,
            "namespace": namespace,
        }
    if isinstance(message, HumanMessage):
        message_type = "user"
    else:
        # SystemMessage and any future message type are shown as custom data.
        message_type = "custom"
    return {
        "type": message_type,
        "message": message.content,
        "message_id": _message_id(message),
        "source": source,
        "node": node,
        "namespace": namespace,
    }


def _interrupt_payloads(
    interrupts: Any,
    source: str | None,
    namespace: Tuple[str, ...],
) -> List[StreamPayload]:
    if isinstance(interrupts, Interrupt):
        interrupts = (interrupts,)

    payloads: List[StreamPayload] = []
    for interrupt_obj in interrupts or ():
        value = _jsonable(interrupt_obj.value)
        # ``agent.interrupt_node`` adds the producing StepNode/orchestrator
        # source to the outer interrupt value.  Use it for grouping and hide
        # the private key from the displayed payload.
        if isinstance(value, dict) and "_debug_source" in value:
            value = dict(value)
            debug_source = value.pop("_debug_source", None)
            if debug_source:
                source = str(debug_source)
        payloads.append(
            {
                "type": "interrupt",
                # JobSubmission / AskForArgs / AgentInterrupt are pydantic
                # models while they travel through the inner agents; make them
                # printable.
                "message": value,
                "message_id": str(interrupt_obj.id),
                "interrupt_id": str(interrupt_obj.id),
                "interrupt_response_schema": _jsonable(
                    getattr(interrupt_obj, "response_schema", None)
                ),
                "source": source,
                "node": None,
                "namespace": namespace,
            }
        )
    return payloads


def _payloads_from_updates(
    data: Any,
    source: str | None,
    namespace: Tuple[str, ...],
) -> List[StreamPayload]:
    """Convert one ``updates`` payload (possibly nested) into payloads."""
    if not isinstance(data, dict):
        return []

    payloads: List[StreamPayload] = []
    for node, update in data.items():
        if node == "__interrupt__":
            payloads.extend(_interrupt_payloads(update, source, namespace))
            continue
        if not isinstance(update, dict):
            continue

        if "__interrupt__" in update:
            payloads.extend(
                _interrupt_payloads(update["__interrupt__"], source, namespace)
            )

        messages = update.get("messages") or []
        if isinstance(messages, BaseMessage):
            messages = [messages]
        for message in messages:
            if isinstance(message, BaseMessage):
                payloads.append(
                    _message_payload(message, source, namespace, node)
                )
    return payloads


def _payloads_from_custom_value(
    data: Any,
    source: str | None,
    namespace: Tuple[str, ...],
) -> List[StreamPayload]:
    """Convert a value written with ``get_stream_writer()`` into payloads."""
    if isinstance(data, BaseMessage):
        return [_message_payload(data, source, namespace)]
    if isinstance(data, (list, tuple)) and any(
        isinstance(item, BaseMessage) for item in data
    ):
        return [
            _message_payload(item, source, namespace)
            for item in data
            if isinstance(item, BaseMessage)
        ]
    return [
        {
            "type": "custom",
            "message": data,
            "message_id": str(uuid.uuid4()),
            "source": source,
            "node": None,
            "namespace": namespace,
        }
    ]


def _payloads_from_custom(
    data: Any,
    namespace: Tuple[str, ...],
) -> List[StreamPayload]:
    """Parse the custom chunks written by the nodes in ``agent.py``.

    Those nodes forward inner graphs as
    ``{"source": ..., "type": "updates"|"custom", "ns": ..., "data": ...}``.
    Anything else is ordinary custom data.
    """
    if (
        isinstance(data, dict)
        and "source" in data
        and data.get("type") in {"updates", "values", "custom"}
    ):
        source = data.get("source")
        inner_type = data.get("type")
        inner_data = data.get("data")
        inner_ns = tuple(data.get("ns") or namespace)

        if inner_type == "updates":
            return _payloads_from_updates(inner_data, source, inner_ns)
        if inner_type == "custom":
            return _payloads_from_custom_value(inner_data, source, inner_ns)
        return _payloads_from_custom_value(data, source, namespace)

    return _payloads_from_custom_value(data, None, namespace)


def _payloads_from_chunk(chunk: Any) -> List[StreamPayload]:
    """Dispatch one ``stream_mode=["custom", "updates"]`` chunk."""
    if not isinstance(chunk, dict):
        return []
    chunk_type = chunk.get("type")
    namespace = tuple(chunk.get("ns") or ())

    if chunk_type == "custom":
        return _payloads_from_custom(chunk.get("data"), namespace)
    if chunk_type == "updates":
        return _payloads_from_updates(chunk.get("data"), None, namespace)
    return []
