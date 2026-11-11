"""Stream a LangGraph agent's output as UI friendly payloads.

The original helper only understood *message* updates. Anything else that a
graph can legitimately emit would raise or be silently dropped:

* ``interrupt()`` -- surfaced by LangGraph as ``{"__interrupt__": (Interrupt(...),)}``
  which is not a ``dict`` and therefore blew up on ``update.get(...)``;
* ``None`` updates -- middleware nodes (and nodes that return nothing) show up
  as ``{"some_node": None}``;
* nodes returning plain state (no ``messages`` key) -- previously ignored;
* several messages in a single update -- only the last one was streamed, which
  hid the ``AIMessage`` of a structured-output turn.

This module keeps the original payload shape (``role`` / ``message`` /
``tool_calls`` / ``structured_response``) so existing consumers keep working,
and extends it so a caller can:

* ask a human whenever an interrupt shows up and then resume the graph;
* stream subgraphs (``subgraphs=True``) without seeing anything twice:
  LangGraph re-emits every subgraph interrupt at the parent namespace level
  (same ``Interrupt.id``) and repeats subgraph messages in the parent node's
  aggregate update (same message objects) -- :func:`run_agent` collapses both
  to the first, deepest occurrence;
* inspect which node / subgraph produced a chunk;
* never crash on a chunk it does not understand.

Consumers normally only need :func:`run_agent`; :func:`payloads_from_chunk` is
exposed so the classification logic can be unit tested in isolation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, TypedDict

from langchain.messages import AIMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt


class StreamPayload(TypedDict, total=False):
    """One streamed item.

    ``role`` is the discriminant and is one of:

    ``"agent"``     an ``AIMessage`` (may carry ``tool_calls``);
    ``"tool"``      a ``ToolMessage`` (carries ``tool_call_id`` / ``name``);
    ``"user"`` / ``"system"``  a human / system message produced by a node;
    ``"interrupt"`` the graph is paused and needs a human decision
                    (``message`` holds ``Interrupt.value``);
    ``"custom"``    arbitrary data written with ``get_stream_writer()``;
    ``None``        anything else -- inspect ``raw``.
    """

    role: str | None
    message: Any
    tool_calls: list[dict[str, Any]]
    tool_call_id: str | None
    name: str | None
    structured_response: Any
    interrupt_id: str
    interrupt: Interrupt
    response_schema: Any
    node: str | None
    namespace: tuple[str, ...]
    raw: Any


_INTERRUPT_KEY = "__interrupt__"

# LangChain message ``type`` values that are not AI/Tool but still useful to
# surface with a friendlier role. Everything else keeps ``role=None``.
_ROLE_BY_MESSAGE_TYPE: dict[str, str] = {
    "human": "user",
    "system": "system",
}

# Payloads with these roles are candidates for de-duplication (see
# ``_dedup_key``). Unknown message types (e.g. ``RemoveMessage``) are not:
# a removal is not a duplicate of the message it removes.
_DEDUP_ROLES = frozenset({"agent", "tool", "user", "system"})


def _annotate(payload: StreamPayload, node: str | None, namespace: tuple[str, ...]) -> StreamPayload:
    """Stamp the producing node and subgraph namespace onto a payload."""
    payload["node"] = node
    payload["namespace"] = namespace
    return payload


def _message_payload(message: BaseMessage) -> StreamPayload:
    """Map a single message to its payload (before node/namespace are added)."""
    if isinstance(message, AIMessage):
        return {
            "role": "agent",
            "message": message.content,
            "tool_calls": list(message.tool_calls or []),
            "raw": message,
        }
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "message": message.content,
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "raw": message,
        }
    # Unknown / custom message: keep the payload schema stable and expose the
    # original object so callers can still render it.
    return {
        "role": _ROLE_BY_MESSAGE_TYPE.get(message.type),
        "message": getattr(message, "content", message),
        "raw": message,
    }


def _interrupt_payloads(
    interrupts: Any, namespace: tuple[str, ...]
) -> list[StreamPayload]:
    """One payload per ``Interrupt`` -- value, id and the object itself."""
    if isinstance(interrupts, Interrupt):
        interrupts = (interrupts,)
    return [
        _annotate(
            {
                "role": "interrupt",
                "message": interrupt.value,
                "interrupt_id": interrupt.id,
                "interrupt": interrupt,
                "response_schema": interrupt.response_schema,
                "raw": interrupt,
            },
            None,
            namespace,
        )
        for interrupt in interrupts or ()
    ]


def _update_payloads(
    node: str, update: dict[str, Any], namespace: tuple[str, ...]
) -> list[StreamPayload]:
    """Turn a single ``{node: update}`` entry into payloads."""
    messages = update.get("messages") or []
    if isinstance(messages, BaseMessage):
        messages = [messages]

    payloads = [
        _annotate(_message_payload(message), node, namespace)
        for message in messages
    ]

    structured_response = update.get("structured_response")
    if structured_response is not None:
        # A structured-output turn carries the response next to its messages.
        # Attach it to the last message so the caller sees both together, and
        # fall back to a standalone payload if the node returned no messages.
        if payloads:
            payloads[-1]["structured_response"] = structured_response
        else:
            payloads.append(
                _annotate(
                    {
                        "role": "agent",
                        "message": None,
                        "structured_response": structured_response,
                        "raw": structured_response,
                    },
                    node,
                    namespace,
                )
            )

    return payloads


def _custom_payloads(data: Any, namespace: tuple[str, ...]) -> list[StreamPayload]:
    """Turn data written with ``get_stream_writer()`` into payloads.

    Custom data is arbitrary. A message, or a list containing messages, is
    mapped exactly like a normal message update; anything else (a string, a
    dict, a list of scalars, ...) is surfaced with ``role="custom"``.
    """
    if isinstance(data, (list, tuple)) and any(
        isinstance(item, BaseMessage) for item in data
    ):
        items = data
    else:
        items = [data]

    return [
        _annotate(
            _message_payload(item) if isinstance(item, BaseMessage) else
            {"role": "custom", "message": item, "raw": item},
            # Custom chunks do not carry the producing node, only the namespace.
            None,
            namespace,
        )
        for item in items
    ]


def _normalize_chunk(chunk: Any) -> dict[str, Any] | None:
    """Return a chunk in the ``{"type", "ns", "data"}`` shape, or ``None``.

    ``astream(version="v2")`` -- what :func:`run_agent` uses by default --
    always yields such dicts. Version ``"v1"`` yields raw tuples instead; the
    two unambiguous layouts, ``(namespace, mode, data)`` (``subgraphs=True``
    with a ``stream_mode`` list) and ``(mode, data)`` (list without subgraphs),
    are converted so the classifier keeps working. The third v1 layout,
    ``(namespace, data)`` from a single stream mode, cannot be told apart from
    custom data and stays unsupported.
    """
    if isinstance(chunk, dict):
        return chunk
    if isinstance(chunk, tuple):
        if len(chunk) == 3:
            namespace, mode, data = chunk
        elif len(chunk) == 2 and isinstance(chunk[0], str):
            namespace, (mode, data) = (), chunk
        else:
            return None
        return {"type": mode, "ns": namespace, "data": data}
    return None


def payloads_from_chunk(chunk: Any) -> list[StreamPayload]:
    """Convert one ``astream(...)`` chunk into payloads.

    Understands ``updates`` (messages, interrupts) and ``custom`` (data written
    with ``get_stream_writer()``), as v2 dicts or the unambiguous v1 tuples
    (see :func:`_normalize_chunk`). Returns an empty list for chunks that carry
    nothing to stream, which makes the function safe to call on every chunk.
    """
    chunk = _normalize_chunk(chunk)
    if chunk is None:
        return []

    namespace = tuple(chunk.get("ns") or ())
    chunk_type = chunk.get("type")

    if chunk_type == "custom":
        return _custom_payloads(chunk.get("data"), namespace)
    if chunk_type != "updates":
        return []

    payloads: list[StreamPayload] = []

    for node, update in (chunk.get("data") or {}).items():
        if node == _INTERRUPT_KEY:
            payloads.extend(_interrupt_payloads(update, namespace))
            continue
        # Nodes may return ``None`` (middleware, empty nodes); they are not
        # message updates so there is nothing to stream.
        if not isinstance(update, dict):
            continue
        payloads.extend(_update_payloads(node, update, namespace))

    return payloads


def _dedup_key(payload: StreamPayload, pinned: list[BaseMessage]) -> Any | None:
    """Stable identity used to drop re-emitted payloads, or ``None`` when the
    payload cannot repeat.

    With ``subgraphs=True`` LangGraph emits each subgraph interrupt once per
    namespace level -- always with the same ``Interrupt.id`` -- and repeats the
    subgraph's messages in the parent node's aggregate update, as the *same*
    message objects. Both are collapsed to the first (deepest) occurrence.
    Messages that carry no ``id`` are keyed by object identity; ``pinned``
    keeps them referenced so their ``id()`` cannot be recycled mid-stream.
    """
    if payload.get("role") == "interrupt":
        return ("interrupt", payload["interrupt_id"])
    if payload.get("role") in _DEDUP_ROLES:
        message = payload["raw"]
        if message.id is None:
            pinned.append(message)
            return ("message", message.type, id(message))
        return ("message", message.type, message.id)
    return None


async def run_agent(
    agent: CompiledStateGraph,
    input: Any,
    context: Any,
    config: RunnableConfig,
    **astream_kwargs: Any,
) -> AsyncIterator[StreamPayload]:
    """Stream ``agent`` and yield :class:`StreamPayload` items.

    Args:
        agent: A compiled LangGraph graph (``create_agent`` output works too).
        input: Initial input, or a ``Command`` to resume a paused graph
            (``Command(resume=...)``) -- see :func:`resume_agent`.
        context: Runtime context forwarded to the graph.
        config: ``RunnableConfig``; must contain a ``configurable.thread_id``
            when the graph has a checkpointer.
        **astream_kwargs: Forwarded to ``agent.astream``. ``stream_mode``
            defaults to ``["updates", "custom"]`` so both state updates and
            ``get_stream_writer()`` output are parsed; pass a mode to narrow it.
            ``version`` defaults to ``"v2"``. ``subgraphs=True`` streams the
            innards of subgraphs (namespaced payloads) and is de-duplicated:
            each interrupt / message is yielded exactly once, at its deepest
            namespace.

    Yields:
        One payload per message / custom item / interrupt. An ``interrupt``
        payload means the graph is paused: collect ``payload["interrupt_id"]``
        and ``payload["message"]``, ask the human, then resume. When several
        interrupts are pending, resume them together with
        :func:`resume_interrupts`. De-duplication is per call: a resumed run
        may legitimately re-show messages written before the pause (a node
        that invokes a subgraph returns the subgraph's full output).
    """
    stream_mode = astream_kwargs.pop("stream_mode", ["updates", "custom"])
    version = astream_kwargs.pop("version", "v2")

    seen: set[Any] = set()
    pinned: list[BaseMessage] = []

    async for chunk in agent.astream(
        input=input,
        config=config,
        context=context,
        stream_mode=stream_mode,
        version=version,
        **astream_kwargs,
    ):
        for payload in payloads_from_chunk(chunk):
            key = _dedup_key(payload, pinned)
            if key is not None:
                if key in seen:
                    continue
                seen.add(key)
            yield payload


def resume_agent(
    agent: CompiledStateGraph,
    resume: Any,
    context: Any,
    config: RunnableConfig,
    **astream_kwargs: Any,
) -> AsyncIterator[StreamPayload]:
    """Resume a graph paused on an interrupt and keep streaming.

    Equivalent to ``run_agent(agent, Command(resume=resume), ...)``. Works for
    interrupts raised inside subgraphs too: LangGraph routes the resume value
    into the paused sub-agent, whether it was added as a node or is being
    invoked inside one.
    """
    return run_agent(
        agent, Command(resume=resume), context, config, **astream_kwargs
    )


def is_interrupt(payload: StreamPayload) -> bool:
    """True when ``payload`` is an interrupt that needs a human decision."""
    return payload.get("role") == "interrupt"


def resume_interrupts(
    agent: CompiledStateGraph,
    responses: Mapping[str, Any],
    context: Any,
    config: RunnableConfig,
    **astream_kwargs: Any,
) -> AsyncIterator[StreamPayload]:
    """Resume several pending interrupts at once.

    Parallel branches can interrupt independently; LangGraph then has more than
    one pending ``Interrupt``. Resume them in one call by mapping each
    ``Interrupt.id`` (``payload["interrupt_id"]``) to the value that interrupt
    expects::

        pending = {p["interrupt_id"]: answer(p) for p in payloads if is_interrupt(p)}
        async for payload in resume_interrupts(agent, pending, None, config):
            ...

    This is ``resume_agent(agent, dict(responses), ...)``; it exists to make
    the id-keyed form explicit. The value per id is whatever that node passed
    to ``interrupt()`` expects (for ``HumanInTheLoopMiddleware`` that is
    ``{"decisions": [...]}``). A subgraph interrupt paused at several
    namespace levels shares one id, so the id-keyed form stays correct there.
    """
    return resume_agent(agent, dict(responses), context, config, **astream_kwargs)
