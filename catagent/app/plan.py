"""Read-only workflow plan projection used by ``GET /plan``."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _step_sort_key(step_id: str) -> Tuple[Any, ...]:
    """Natural ordering for ids such as ``plan-2:step-10:geometry``."""
    parts: List[Any] = []
    for chunk in str(step_id).split(":"):
        pieces: List[Any] = []
        for token in chunk.replace("-", " ").split():
            pieces.append(int(token) if token.isdigit() else token)
        parts.append(tuple(pieces))
    return tuple(parts)


async def get_plan(catagent, thread_id: str) -> Dict[str, Any]:
    """Return the workflow plan for ``thread_id`` as plain JSON data.

    This powers the HTTP ``/plan/{thread_id}`` route.
    """
    config = {"configurable": {"thread_id": thread_id}}
    agent_state = await catagent.aget_state(config)
    values = getattr(agent_state, "values", None) or {}

    raw_steps = values.get("steps") or {}
    steps: List[Dict[str, Any]] = []
    for step_id, step in raw_steps.items():
        if hasattr(step, "model_dump"):
            data = step.model_dump(mode="json")
        elif isinstance(step, dict):
            data = dict(step)
        else:
            data = {"content": str(step)}
        data.setdefault("step_id", str(step_id))
        steps.append(data)

    steps.sort(key=lambda item: _step_sort_key(item.get("step_id", "")))

    return {
        "thread_id": thread_id,
        "steps": steps,
        "finalization": bool(values.get("finalization", False)),
    }
