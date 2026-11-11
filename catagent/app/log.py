"""Application logging setup shared by the debug CLI and the HTTP server."""

from __future__ import annotations

import logging
import logging.handlers
import queue


def setup_logging(log_path: str | None = None,
                  level: str = "info") -> logging.handlers.QueueListener:
    """Configure the ``catagent`` logger without blocking the event loop.

    Records are handed to a :class:`logging.handlers.QueueHandler` and a
    background thread performs the actual I/O.  When ``log_path`` is omitted
    the records go to stderr, which is convenient for the debug CLI and for
    ``uvicorn`` deployments that already collect stderr.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log = logging.getLogger("catagent")
    log.setLevel(numeric_level)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    output: logging.Handler
    if log_path:
        output = logging.FileHandler(log_path)
    else:
        output = logging.StreamHandler()
    output.setFormatter(formatter)
    output.setLevel(numeric_level)

    log_queue: queue.Queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(numeric_level)
    log.addHandler(queue_handler)

    listener = logging.handlers.QueueListener(log_queue, output,
                                              respect_handler_level=True)
    listener.start()
    return listener


def shutdown_logging(listener: logging.handlers.QueueListener) -> None:
    """Stop a listener created by :func:`setup_logging` and detach handlers."""
    listener.stop()
    for handler in getattr(listener, "handlers", ()):
        handler.close()

    log = logging.getLogger("catagent")
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
