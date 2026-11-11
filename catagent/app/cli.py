"""Interactive debug chat loop (in-process graph)."""

from __future__ import annotations

import json
import pprint
import uuid
from typing import Any, Dict

from langchain.messages import HumanMessage
from langgraph.types import Command

from .jobs import job_uuid_from_interrupt, poll_job_results
from .models import StreamPayload
from .runtime import build_runtime
from .streaming import _stream_chat

_COLORS = {
    "user": "\033[93m",       # yellow
    "agent": "\033[96m",      # cyan
    "tool": "\033[92m",       # green
    "interrupt": "\033[95m",  # magenta
    "custom": "\033[90m",     # grey
    "error": "\033[91m",      # red
}
_RESET = "\033[0m"


def _print_payload(payload: StreamPayload) -> None:
    """Print one payload in a readable, colored form (debug only)."""
    kind = payload.get("type", "custom")
    color = _COLORS.get(kind, _COLORS["custom"])
    source = payload.get("source")
    source_text = f" source={source}" if source else ""
    header = f"[{kind}{source_text}]"

    if kind == "agent":
        text = payload.get("message")
        if not isinstance(text, str):
            text = pprint.pformat(text)
        print(f"{color}{header} {text}{_RESET}")
        tool_calls = payload.get("tool_calls") or []
        if tool_calls:
            print(f"{color}  tool_calls: {pprint.pformat(tool_calls)}{_RESET}")
    elif kind == "tool":
        text = payload.get("message")
        if not isinstance(text, str):
            text = pprint.pformat(text)
        print(
            f"{color}{header} {payload.get('tool_call_name')} "
            f"(id={payload.get('tool_call_id')}): {text}{_RESET}"
        )
    elif kind == "interrupt":
        print(f"{color}{header} id={payload.get('interrupt_id')}{_RESET}")
        print(f"{color}{pprint.pformat(payload.get('message'))}{_RESET}")
    else:
        text = payload.get("message")
        if not isinstance(text, str):
            text = pprint.pformat(text)
        print(f"{color}{header} {text}{_RESET}")


def _parse_cli_response(raw: str, interrupt_value: Any) -> Any:
    """Parse a debug user's answer, with friendly handling of free text."""
    raw = raw.strip()
    try:
        parsed: Any = json.loads(raw)
    except Exception:
        parsed = raw

    if not isinstance(parsed, str) or not isinstance(interrupt_value, dict):
        return parsed

    kind = interrupt_value.get("kind")
    if kind == "AgentInterrupt":
        return {"payload": {"result": parsed}}
    if kind == "AskForArgs":
        return {"payload": {"command": "edit", "message": parsed}}
    return parsed


async def run_cli(args) -> None:
    """Run the interactive debug chat loop against an in-process graph."""
    runtime_kwargs = {
        "dotenv_path": args.dotenv,
        "db_path": args.db,
        "job_manager_url": args.job_manager_url,
    }
    skill_index_json = getattr(args, "skill_index_json", None)
    if skill_index_json:
        runtime_kwargs["skill_index_json"] = skill_index_json
    catagent, httpx_client, db_conn = await build_runtime(**runtime_kwargs)

    thread_id = args.thread_id or f"cli-{uuid.uuid4()}"
    pending: Dict[str, Any] = {}

    print(f"cat-agent debug CLI, thread_id={thread_id}")
    print("Type 'exit' or 'quit' to leave. Interrupt answers may be JSON.")

    try:
        while True:
            if pending:
                # JobSubmission is automatic: poll the job manager and resume.
                # Only the remaining interrupts are answered by the user on
                # stdin.
                jobs = {
                    interrupt_id: value
                    for interrupt_id, value in pending.items()
                    if isinstance(value, dict)
                    and value.get("kind") == "JobSubmission"
                }
                answers: Dict[str, Any] = {}
                for interrupt_id, value in pending.items():
                    if interrupt_id in jobs:
                        continue
                    print(
                        f"\n{_COLORS['interrupt']}Interrupt {interrupt_id}{_RESET}"
                    )
                    print(pprint.pformat(value))
                    raw = input("answer (JSON or free text) > ")
                    if raw.strip().lower() in {"exit", "quit"}:
                        return
                    answers[interrupt_id] = _parse_cli_response(raw, value)

                if jobs:
                    labels = ", ".join(
                        job_uuid_from_interrupt(value) or interrupt_id
                        for interrupt_id, value in jobs.items()
                    )
                    print(
                        f"\n{_COLORS['interrupt']}Waiting for submitted "
                        f"job(s): {labels}{_RESET}"
                    )
                    try:
                        job_answers = await poll_job_results(
                            httpx_client,
                            args.job_manager_url,
                            jobs,
                        )
                    except KeyboardInterrupt:
                        return
                    except Exception as exc:
                        print(
                            f"{_COLORS['error']}ERROR while polling jobs: "
                            f"{type(exc).__name__}: {exc}{_RESET}"
                        )
                        # Keep the job interrupts; retry on the next loop.
                        pending = jobs
                        continue

                    answers.update(job_answers)
                    for interrupt_id, response in job_answers.items():
                        value = jobs.get(interrupt_id) or {}
                        job_uuid = job_uuid_from_interrupt(value) or interrupt_id
                        payload = (
                            response.get("payload")
                            if isinstance(response, dict)
                            else None
                        ) or {}
                        print(
                            f"{_COLORS['tool']}Job {job_uuid} returned: "
                            f"{payload.get('status', 'error')}{_RESET}"
                        )

                pending = {}
                agent_input: Any = Command(resume=answers)
            else:
                raw = input(f"\n{_COLORS['user']}user > {_RESET}")
                if raw.strip().lower() in {"exit", "quit"}:
                    break
                if not raw.strip():
                    continue
                agent_input = {
                    "messages": [HumanMessage(content=raw)],
                    "finalization": False,
                    "orchestrator_is_working": False,
                }

            try:
                async for payload in _stream_chat(
                    catagent=catagent,
                    agent_input=agent_input,
                    thread_id=thread_id,
                    job_manager_url=args.job_manager_url,
                    http_client=httpx_client,
                    # The CLI polls jobmanager itself, so the manager must not
                    # also POST a callback to a nonexistent HTTP endpoint for
                    # this process.
                    register_jobs=False,
                ):
                    _print_payload(payload)
                    if payload.get("type") == "interrupt":
                        pending[payload["interrupt_id"]] = payload.get("message")
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(
                    f"{_COLORS['error']}ERROR: {type(exc).__name__}: "
                    f"{exc}{_RESET}"
                )
    finally:
        await httpx_client.aclose()
        await db_conn.close()
