"""Entry point for the installed ``catagent`` command.

This module only imports :mod:`catagent._args`, so ``catagent --help`` and
``--version`` work without loading LangGraph or the model stack.  The App layer
is imported lazily inside :func:`main` once a real run starts.
"""

from __future__ import annotations

from typing import Sequence

from ._args import _build_command_parser, resolve_common_args


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_command_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "jobmanager":
        from .jobmanager.server import run as run_jobmanager

        run_jobmanager(
            setting=args.setting,
            host=args.host,
            port=args.port,
            log=args.log,
            level=args.level,
        )
        return

    # ``server`` is the only mode with a listening socket; ``cli`` is the
    # default in-process mode.  Keep ``--serve`` semantics for run_server().
    args.serve = args.command == "server"

    try:
        resolve_common_args(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    from .app import _run

    _run(args, parser)


if __name__ == "__main__":
    main()
