"""Locate data files that ship with cat-agent.

This module and :mod:`catagent._args` are deliberately dependency-free so the
installed ``catagent --help`` does not need to import the LangGraph stack.

Everything ships inside the ``catagent`` package, so the skills directory is
``<catagent>/skills`` both in a source checkout and in an installed wheel.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["default_skill_index"]

SKILLS_ENV = "CATAGENT_SKILL_INDEX"

# catagent/_paths.py -> catagent/
_PACKAGE_DIR = Path(__file__).resolve().parent


def default_skill_index() -> Path:
    """Return the skill index to use when ``--skill-index-json`` is omitted.

    Resolution order:

    1. ``CATAGENT_SKILL_INDEX`` (an explicit user override),
    2. ``<catagent>/skills/skill_index.json`` (source checkout or wheel).
    """
    override = os.environ.get(SKILLS_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{SKILLS_ENV} points at {str(path)!r}, which does not exist."
            )
        return path.resolve()

    packaged = _PACKAGE_DIR / "skills" / "skill_index.json"
    if packaged.is_file():
        return packaged.resolve()

    raise FileNotFoundError(
        "Could not find catagent/skills/skill_index.json. Pass "
        f"--skill-index-json, set {SKILLS_ENV}, or reinstall cat-agent with "
        "its bundled skills."
    )
