"""Simulated front end: a tiny HTTP server that only prints to the terminal.

It receives two kinds of POSTs from the job manager:

- JobResult : sent to `callback_url` when a job reaches its final state.
- messages  : agent progress sent to `front_end_url`.

The demo config uses one port with `/callback` and `/stream` paths. The real
`jobmanager_config.json` uses two ports with bare `/` paths, so this server can
bind both:

    python -m catagent.jobmanager.demo.frontend --callback-port 5600 --stream-port 5601

or a single port for both:

    python -m catagent.jobmanager.demo.frontend --callback-port 9001
"""

import argparse
import ast
import datetime
import json
import logging
import os
import shutil
import sys
import textwrap
import threading

# Make `jobmanager` importable when this file is run as a plain script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_PARENT not in sys.path:
    sys.path.insert(0, _REPO_PARENT)

import uvicorn
from fastapi import FastAPI, Request
from catagent.jobmanager.models import JobResult

logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

_ROLE_COLOR = {"agent": "36", "tool": "32", "human": "34", "message": "37"}


def _width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 20)).columns, 120))


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if sys.stdout.isatty() else text


def _indent_wrap(text, indent: str = "    ", subsequent: str = None) -> str:
    """Wrap long lines to the terminal width and indent every line.

    `subsequent` is used for continuation lines (hanging indent).
    """
    width = _width()
    sub = subsequent if subsequent is not None else indent
    out = []
    for raw in str(text).split("\n"):
        if not raw.strip():
            out.append("")
            continue
        pieces = textwrap.wrap(raw, width=max(20, width - len(indent)),
                               break_long_words=False, break_on_hyphens=False) or [""]
        out.append(indent + pieces[0])
        out.extend(sub + piece for piece in pieces[1:])
    return "\n".join(out)


def _stringify_message(message) -> str:
    """AIMessage.content may be a string or a list of content blocks."""
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for block in message:
            if isinstance(block, dict):
                parts.append(str(block.get("text", block.get("content", block))))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(message)


def _stringify_tool_calls(raw) -> str:
    """Render the tool_calls repr as one bullet per call."""
    if isinstance(raw, str):
        try:
            calls = ast.literal_eval(raw)
        except Exception:
            return raw
    else:
        calls = raw
    if not isinstance(calls, (list, tuple)):
        return str(calls)
    lines = []
    for call in calls:
        if isinstance(call, dict):
            name = call.get("name")
            if name is None and isinstance(call.get("function"), dict):
                name = call["function"].get("name")
            args = call.get("args", call.get("arguments", ""))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            args_str = (json.dumps(args, ensure_ascii=False)
                        if isinstance(args, (dict, list)) else str(args))
            lines.append(f"\u2022 {name}({args_str})")
        else:
            lines.append(f"\u2022 {call}")
    return "\n".join(lines)


def _print_result(result: JobResult) -> None:
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    ok = result.status == "success"
    color = "32" if ok else "31"
    bar = "\u2550" * _width()
    print(f"\n{_c(color, bar)}", flush=True)
    print(_c(color, f" JOB RESULT  {result.status.upper()}   ({ts})"), flush=True)
    print(f"   job_uuid  : {result.job_uuid}", flush=True)
    print(f"   local_dir : {result.local_dir}", flush=True)
    if result.error_msg:
        print("   error_msg :", flush=True)
        print(_indent_wrap(result.error_msg, "     "), flush=True)
    print(f"{_c(color, bar)}\n", flush=True)


def _print_stream(payload: dict) -> None:
    role = payload.get("role") or "message"
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    color = _ROLE_COLOR.get(role, "37")
    message = _stringify_message(payload.get("message"))
    tool_calls = payload.get("tool_calls") or ""

    print(flush=True)
    print(_c(color, f"[{role} {ts}]"), flush=True)
    if message.strip():
        print(_indent_wrap(message), flush=True)
    if tool_calls:
        print(_indent_wrap("tool calls:", "    "), flush=True)
        print(_indent_wrap(_stringify_tool_calls(tool_calls), "      ", "        "), flush=True)


async def _do_callback(body: dict) -> dict:
    _print_result(JobResult(**body))
    return {"ok": True}


async def _do_stream(body: dict) -> dict:
    _print_stream(body)
    return {"ok": True}


def build_callback_app() -> FastAPI:
    app = FastAPI(title="Job Manager Demo Frontend (callback)")

    @app.post("/callback")
    async def callback(result: JobResult) -> dict:
        _print_result(result)
        return {"ok": True}

    @app.post("/")
    async def root(result: JobResult) -> dict:
        _print_result(result)
        return {"ok": True}

    return app


def build_stream_app() -> FastAPI:
    app = FastAPI(title="Job Manager Demo Frontend (stream)")

    @app.post("/stream")
    async def stream(payload: dict) -> dict:
        _print_stream(payload)
        return {"ok": True}

    @app.post("/")
    async def root(payload: dict) -> dict:
        _print_stream(payload)
        return {"ok": True}

    return app


def build_combined_app() -> FastAPI:
    """Single-port mode: disambiguate by body content on `/`."""
    app = FastAPI(title="Job Manager Demo Frontend")

    @app.post("/callback")
    async def callback(result: JobResult) -> dict:
        _print_result(result)
        return {"ok": True}

    @app.post("/stream")
    async def stream(payload: dict) -> dict:
        _print_stream(payload)
        return {"ok": True}

    @app.post("/")
    async def root(request: Request) -> dict:
        body = await request.json()
        if isinstance(body, dict) and "job_uuid" in body:
            return await _do_callback(body)
        return await _do_stream(body)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo front end (JobResult/message printer).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--callback-port", type=int, default=9001)
    parser.add_argument("--stream-port", type=int, default=None,
                        help="Port for agent messages. Defaults to --callback-port.")
    args = parser.parse_args()

    stream_port = args.stream_port or args.callback_port
    if stream_port == args.callback_port:
        print(f"[FRONTEND] listening on http://{args.host}:{args.callback_port} "
              f"(paths: /callback, /stream)", flush=True)
        uvicorn.run(build_combined_app(), host=args.host, port=args.callback_port,
                    log_level="warning")
        return

    # Two ports: the stream server runs in a daemon thread, callbacks in the main thread.
    stream_thread = threading.Thread(
        target=uvicorn.run,
        kwargs=dict(app=build_stream_app(), host=args.host, port=stream_port,
                    log_level="warning"),
        daemon=True)
    stream_thread.start()
    print(f"[FRONTEND] callback on http://{args.host}:{args.callback_port} (/), "
          f"stream on http://{args.host}:{stream_port} (/)", flush=True)
    uvicorn.run(build_callback_app(), host=args.host, port=args.callback_port,
                log_level="warning")


if __name__ == "__main__":
    main()
