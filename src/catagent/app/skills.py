"""Load the skill index used to seed every new workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..workflow.models import SkillArtifact, SkillMetadata

__all__ = ["load_skill_index"]


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def _resolve_skill_path(
    declared: str,
    *,
    index_dir: Path,
    name: str,
) -> Path:
    """Resolve a skill document, tolerating a moved/renamed skill checkout.

    Index files are often generated on one machine and reused on another.  If
    the recorded absolute path does not exist, fall back to
    ``<index_dir>/<name>/skill.md``; that is the layout used by this repository
    (``catagent/skills/<name>/skill.md``).
    """
    path = _expand(declared)
    if not path.is_absolute():
        path = index_dir / path
    if path.is_file():
        return path.resolve()

    fallback = index_dir / name / "skill.md"
    if fallback.is_file():
        return fallback.resolve()

    fallback = index_dir / name / _expand(declared).name
    if fallback.is_file():
        return fallback.resolve()

    raise FileNotFoundError(
        f"Skill {name!r} points at {declared!r}, which does not exist and "
        f"could not be found under {str(index_dir / name)!r}."
    )


def _resolve_artifact(
    artifact: Any,
    *,
    index_dir: Path,
    skill_path: Path,
    old_skill_dir: Path | None,
) -> SkillArtifact:
    """Keep artifact metadata, remapping its path when the checkout moved."""
    if not isinstance(artifact, dict):
        raise ValueError(f"Skill artifact must be an object, got {type(artifact)!r}.")

    declared = str(artifact.get("path", ""))
    purpose = str(artifact.get("purpose", ""))
    raw = _expand(declared)
    skill_dir = skill_path.parent

    if raw.is_absolute():
        path = raw
    else:
        # Relative paths in the index are relative to the index file itself,
        # matching the ``path`` field (``<skill>/skill.md``).  As a convenience,
        # also accept paths relative to the individual skill directory.
        path = index_dir / raw
        if not path.is_file():
            path = skill_dir / raw

    if path.is_file():
        resolved = path.resolve()
    else:
        # Preserve the subdirectory (references/ or scripts/) but replace the
        # stale checkout prefix with the actual skill directory.
        candidates = []
        if old_skill_dir is not None:
            try:
                candidates.append(skill_dir / raw.relative_to(old_skill_dir))
            except ValueError:
                pass
        candidates.extend(
            [
                skill_dir / raw.name,
                skill_dir / "references" / raw.name,
                skill_dir / "scripts" / raw.name,
            ]
        )
        resolved = next(
            (candidate.resolve() for candidate in candidates if candidate.is_file()),
            path,
        )

    return SkillArtifact(path=str(resolved), purpose=purpose)


def _artifacts(
    values: Any,
    *,
    index_dir: Path,
    skill_path: Path,
    old_skill_dir: Path | None,
) -> list[SkillArtifact]:
    if not values:
        return []
    if not isinstance(values, (list, tuple)):
        raise ValueError("Skill references/scripts must be a list.")
    return [
        _resolve_artifact(
            item,
            index_dir=index_dir,
            skill_path=skill_path,
            old_skill_dir=old_skill_dir,
        )
        for item in values
    ]


def load_skill_index(path: str) -> Dict[str, SkillMetadata]:
    """Read ``skill_index_json`` and return ``{name: SkillMetadata}``.

    The file must contain a JSON array.  Each entry has the same shape as
    ``catagent/skills/skill_index.json``: ``name``, ``path``, ``summary``,
    ``references`` and ``scripts``.  Paths are resolved against the index
    location when the recorded checkout no longer exists, so an index from
    another machine still works as long as the skills sit next to it.
    """
    index_path = _expand(path)
    if not index_path.is_file():
        raise FileNotFoundError(f"Skill index {path!r} does not exist.")

    with index_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Skill index must be a JSON array of skill objects.")

    index_dir = index_path.resolve().parent
    skills: Dict[str, SkillMetadata] = {}
    for entry in data:
        if not isinstance(entry, dict):
            raise ValueError("Each skill index entry must be a JSON object.")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("Each skill index entry needs a non-empty 'name'.")
        declared_path = str(entry.get("path") or "")
        if not declared_path:
            raise ValueError(f"Skill {name!r} is missing its 'path'.")

        skill_path = _resolve_skill_path(
            declared_path,
            index_dir=index_dir,
            name=name,
        )
        declared = _expand(declared_path)
        old_skill_dir = declared.parent if declared.is_absolute() else None
        skills[name] = SkillMetadata(
            name=name,
            path=str(skill_path),
            summary=str(entry.get("summary") or ""),
            references=_artifacts(
                entry.get("references"),
                index_dir=index_dir,
                skill_path=skill_path,
                old_skill_dir=old_skill_dir,
            ),
            scripts=_artifacts(
                entry.get("scripts"),
                index_dir=index_dir,
                skill_path=skill_path,
                old_skill_dir=old_skill_dir,
            ),
        )

    return skills
