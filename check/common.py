"""Small helpers shared by the check/ test scripts."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import dotenv
from langchain.chat_models import BaseChatModel

from catagent.app import create_llm_model
from catagent.workflow.models import SkillArtifact, SkillMetadata

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = Path(__file__).resolve().parent / "mock_skills" / "mock_compute"
MOCK_BIN = REPO_ROOT / "check" / "mock_software" / "bin"


def load_env() -> None:
    """Load the model credentials used by the rest of the repo."""
    candidates = [
        Path("/home/luandong/bin/.env"),
        REPO_ROOT / ".env",
        REPO_ROOT / "catagent" / "jobmanager" / ".env",
    ]
    for path in candidates:
        if path.is_file():
            dotenv.load_dotenv(path, override=False)
            return
    raise FileNotFoundError("No .env file found; set MODEL_NAME/BASE_URL/API_KEY.")


def make_model() -> BaseChatModel:
    load_env()
    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise RuntimeError("MODEL_NAME is not set.")
    lower = model_name.lower()
    if "qwen" in lower:
        kind = "qwen"
    elif "glm" in lower:
        kind = "glm"
    elif "deepseek" in lower:
        kind = "deepseek"
    else:
        raise ValueError(f"Cannot infer model type from MODEL_NAME={model_name!r}.")
    return create_llm_model(
        model_type=kind,
        model_name=model_name,
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
    )


def _frontmatter(text: str) -> Dict[str, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, flags=re.S)
    values: Dict[str, str] = {}
    if not match:
        return values
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip().strip('"\'')
    return values


def skill_metadata(skill_dir: Path) -> SkillMetadata:
    skill_dir = skill_dir.resolve()
    skill_path = skill_dir / "skill.md"
    if not skill_path.is_file():
        raise FileNotFoundError(skill_path)
    text = skill_path.read_text(encoding="utf-8")
    front = _frontmatter(text)
    name = front.get("name") or skill_dir.name
    summary = front.get("description") or ""

    refs_dir = skill_dir / "references"
    references = []
    if refs_dir.is_dir():
        for path in sorted(refs_dir.iterdir()):
            if path.is_file():
                references.append(SkillArtifact(path=str(path),
                                                purpose=f"Reference file {path.name}"))

    scripts_dir = skill_dir / "scripts"
    scripts = []
    if scripts_dir.is_dir():
        for path in sorted(scripts_dir.iterdir()):
            if path.is_file():
                scripts.append(SkillArtifact(path=str(path),
                                             purpose=f"Mock software script {path.name}"))

    return SkillMetadata(name=name,
                         path=str(skill_path),
                         summary=summary,
                         references=references,
                         scripts=scripts)


def load_mock_skills() -> Dict[str, SkillMetadata]:
    return {m["name"]: m for m in (skill_metadata(SKILL_ROOT),)}


def add_mock_path_to_env() -> None:
    """Make mock-* commands available to execute_shell in this process."""
    current = os.environ.get("PATH", "")
    if str(MOCK_BIN) not in current.split(os.pathsep):
        os.environ["PATH"] = f"{MOCK_BIN}{os.pathsep}{current}"


def iter_v2_updates(stream) -> Iterable[Any]:
    """Yield only 'updates' chunks from a v2 astream."""
    async def _inner():
        async for chunk in stream:
            if isinstance(chunk, dict) and chunk.get("type") == "updates":
                yield chunk
    return _inner()
