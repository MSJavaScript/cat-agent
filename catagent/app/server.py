"""HTTP server launcher for the cat-agent FastAPI application."""

from __future__ import annotations

from typing import Tuple


def _parse_bind_url(url: str) -> Tuple[str, int]:
    """Split ``http://host:port`` into a uvicorn host/port pair."""
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"http://{url}")
    if not parsed.hostname:
        raise ValueError(f"Invalid bind URL: {url!r}")
    return parsed.hostname, parsed.port or 5600


async def run_server(args) -> None:
    """Run the HTTP API in this process (``--serve``).

    The app and its aiosqlite/httpx clients are created in the same event loop
    that uvicorn later uses, so nothing is bound to a foreign loop.
    """
    # Look ``build_app`` up on the package facade rather than the module
    # globals.  This keeps ``catagent.app.build_app`` patchable, matching the
    # behaviour of the former single-module ``app.py``.
    from . import build_app as _build_app

    app_kwargs = {
        "dotenv_path": args.dotenv,
        "db_path": args.db,
        "job_manager_url": args.job_manager_url,
    }
    skill_index_json = getattr(args, "skill_index_json", None)
    if skill_index_json:
        app_kwargs["skill_index_json"] = skill_index_json
    app = await _build_app(**app_kwargs)
    import uvicorn

    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    await uvicorn.Server(config).serve()
