"""cat-agent application layer.

The former single ``catagent/app.py`` module is split by responsibility:

``models.py``
    Shared request and stream-payload types.
``payloads.py``
    Conversion from LangGraph stream chunks to UI/debug payloads.
``runtime.py``
    Model selection, checkpoint database and compiled graph construction.
``skills.py``
    Loading of ``skill_index_json`` into initial ``loaded_skills``.
``jobs.py``
    Polling of durable job results for the debug CLI.
``streaming.py``
    Graph streaming plus jobmanager resume-target registration.
``plan.py``
    Read-only plan projection used by HTTP.
``events.py``
    Bounded in-memory payload log for the single-page debug UI.
``service.py``
    FastAPI application and HTTP routes.
``cli.py``
    Interactive debug chat loop.
``server.py``
    HTTP server launcher.
``log.py``
    Application logging setup shared by the CLI and the server.

This module is also the common entry point: ``_build_parser``/``main`` are
used by ``python -m catagent.app`` and can still be imported as
``catagent.app.main``.  The imports below preserve the historical
``catagent.app.<name>`` surface so existing callers such as ``check/`` and
external tools keep working.
"""

from __future__ import annotations

import argparse
import asyncio

from .._args import _build_parser, resolve_common_args
from .cli import (
    _COLORS,
    _RESET,
    _parse_cli_response,
    _print_payload,
    run_cli,
)
from .jobs import job_uuid_from_interrupt, poll_job_results, wait_for_job
from .log import setup_logging, shutdown_logging
from .models import ChatRequest, StreamPayload
from .payloads import (
    _interrupt_payloads,
    _jsonable,
    _message_id,
    _message_payload,
    _payloads_from_chunk,
    _payloads_from_custom,
    _payloads_from_custom_value,
    _payloads_from_updates,
)
from .plan import _step_sort_key, get_plan
from .runtime import build_runtime, create_llm_model
from .server import _parse_bind_url, run_server
from .service import _resume_from_request, build_app
from .skills import load_skill_index
from .streaming import _register_job_submission, _stream_chat


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser | None = None) -> None:
    """Run the mode selected by ``args`` (shared by both entry points)."""
    serve = getattr(args, "serve", False) or bool(getattr(args, "url", None))
    if getattr(args, "url", None):
        try:
            args.host, args.port = _parse_bind_url(args.url)
        except ValueError as exc:
            if parser is not None:
                parser.error(str(exc))
            raise

    # The agent logger is configured for both runtime modes, so the log.info()
    # calls in the graph are visible without relying on uvicorn's log setup.
    listener = setup_logging(args.log, args.agent_log_level)
    try:
        if serve:
            asyncio.run(run_server(args))
        else:
            asyncio.run(run_cli(args))
    finally:
        shutdown_logging(listener)


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        resolve_common_args(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    _run(args, parser)


__all__ = [
    "ChatRequest",
    "StreamPayload",
    "_build_parser",
    "build_app",
    "build_runtime",
    "create_llm_model",
    "get_plan",
    "job_uuid_from_interrupt",
    "load_skill_index",
    "main",
    "poll_job_results",
    "run_cli",
    "run_server",
    "setup_logging",
    "shutdown_logging",
    "wait_for_job",
]
