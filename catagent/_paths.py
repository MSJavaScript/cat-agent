"""Locate data files that ship with cat-agent.

This module and :mod:`catagent._args` are deliberately dependency-free so the
installed ``catagent --help`` does not need to import the LangGraph stack.

The skills directory is part of the source tree (``<repo>/skills``) and is
installed as the ``skills`` package, so the same lookup works for a source
checkout and for a wheel installed into site-packages.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

__all__ = ["default_skill_index"]

SKILLS_ENV = "CATAGENT_SKILL_INDEX"


def _installed_skill_index() -> Path | None:
    try:
        spec = importlib.util.find_spec("skills")
    except (ImportError, ValueError):
        return None
    for location in spec.submodule_search_locations or ():
        candidate = Path(location) / "skill_index.json"
        if candidate.is_file():
            return candidate
    return None


def default_skill_index() -> Path:
    """Return the skill index to use when ``--skill-index-json`` is omitted.

    Resolution order:

    1. ``CATAGENT_SKILL_INDEX`` (an explicit user override),
    2. the installed/source ``skills`` package,
    3. ``<repo>/skills/skill_index.json`` for a checkout that is only on
       ``PYTHONPATH``.
    """
    override = os.environ.get(SKILLS_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{SKILLS_ENV} points at {str(path)!r}, which does not exist."
            )
        return path.resolve()

    installed = _installed_skill_index()
    if installed is not None:
        return installed.resolve()

    # catagent/_paths.py -> repository root
    checkout = Path(__file__).resolve().parents[1] / "skills" / "skill_index.json"
    if checkout.is_file():
        return checkout

    raise FileNotFoundError(
        "Could not find skills/skill_index.json. Pass --skill-index-json, set "
        f"{SKILLS_ENV}, or reinstall cat-agent with its skills package."
    )
