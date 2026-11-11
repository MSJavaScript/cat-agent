"""Skill-index loading, graph seeding and skill-payload truncation.

No LLM call is made: the graph is only constructed, then its ``init_skills``
node is called directly.  The loader and payload tests need no network either.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import httpx
from langchain.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from catagent.agent import create_catagent
from catagent.app import _build_parser
from catagent._args import _build_command_parser, resolve_common_args
from catagent._paths import SKILLS_ENV, default_skill_index
from catagent.app.payloads import _message_payload
from catagent.app.skills import load_skill_index
from check.scripted_model import ScriptedModel

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_INDEX = REPO_ROOT / "skills" / "skill_index.json"


def _skill(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "path": f"/old/checkout/skills/{name}/skill.md",
        "summary": f"Summary of {name}.",
        "references": [
            {
                "path": f"/old/checkout/skills/{name}/references/ref.md",
                "purpose": "Reference.",
            }
        ],
        "scripts": [
            {
                "path": f"/old/checkout/skills/{name}/scripts/run.py",
                "purpose": "Script.",
            }
        ],
    }


def test_repository_index() -> None:
    skills = load_skill_index(str(SKILL_INDEX))
    assert skills, "skills/skill_index.json must not be empty"
    for metadata in skills.values():
        assert Path(metadata["path"]).is_file(), metadata["path"]
        artifacts = (metadata["references"] or []) + (metadata["scripts"] or [])
        for artifact in artifacts:
            assert Path(artifact["path"]).is_file(), artifact["path"]


def test_moved_checkout() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        skill_dir = root / "skills" / "demo"
        (skill_dir / "references").mkdir(parents=True)
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "skill.md").write_text("# demo\n", encoding="utf-8")
        (skill_dir / "references" / "ref.md").write_text("ref\n", encoding="utf-8")
        (skill_dir / "scripts" / "run.py").write_text("print('ok')\n", encoding="utf-8")
        index = root / "skills" / "skill_index.json"
        index.write_text(json.dumps([_skill("demo")]), encoding="utf-8")

        skills = load_skill_index(str(index))

    metadata = skills["demo"]
    assert metadata["path"] == str(skill_dir / "skill.md")
    assert metadata["references"][0]["path"] == str(skill_dir / "references" / "ref.md")
    assert metadata["scripts"][0]["path"] == str(skill_dir / "scripts" / "run.py")


def test_payload_limit() -> None:
    long_text = "x" * 500
    for name in (
        "get_skill_summaries",
        "get_skill_content",
        "load_skill_from_filesystem",
    ):
        message = ToolMessage(content=long_text, tool_call_id="1", name=name)
        payload = _message_payload(message)
        assert payload["message"] == long_text[:100]
        assert message.content == long_text, "the ToolMessage must stay intact"

    message = ToolMessage(content=long_text, tool_call_id="2", name="other_tool")
    assert _message_payload(message)["message"] == long_text


async def _test_graph_seeding() -> None:
    initial = {
        "demo": {
            "name": "demo",
            "path": "/tmp/demo/skill.md",
            "summary": "Demo.",
            "references": [],
            "scripts": [],
        }
    }
    async with httpx.AsyncClient() as client:
        catagent = create_catagent(
            model=ScriptedModel(),
            job_manager_url="http://jobs",
            http_client=client,
            checkpointer=InMemorySaver(),
            loaded_skills=initial,
        )

    init_node = catagent.nodes["init_skills"].bound.func
    command = init_node({"messages": []})
    assert command.goto == "orchestrator_node"
    assert command.update == {"loaded_skills": initial}

    command = init_node({"messages": [], "loaded_skills": {"existing": {}}})
    assert command.goto == "orchestrator_node"
    assert not command.update, "an existing thread must keep its loaded skills"


def test_cli_option_default() -> None:
    """The skill index is bundled, so the flag is optional."""
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.skill_index_json is None
    resolve_common_args(args)
    assert Path(args.skill_index_json) == SKILL_INDEX.resolve()

    # An explicit value is preserved for load_skill_index() to expand.
    args = parser.parse_args(["--skill-index-json", "custom.json"])
    resolve_common_args(args)
    assert args.skill_index_json == "custom.json"

    command = _build_command_parser().parse_args(["cli"])
    resolve_common_args(command)
    assert command.command == "cli"
    assert Path(command.skill_index_json) == SKILL_INDEX.resolve()

    server = _build_command_parser().parse_args(["server", "--port", "1234"])
    assert server.command == "server"
    assert server.port == 1234

    with mock.patch.dict(os.environ, {SKILLS_ENV: str(SKILL_INDEX)}):
        assert default_skill_index() == SKILL_INDEX.resolve()


def test_agent_logging() -> None:
    from catagent.app import setup_logging, shutdown_logging

    log = logging.getLogger("catagent")
    old_handlers = list(log.handlers)
    old_level = log.level
    old_propagate = log.propagate
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "catagent.log"
            listener = setup_logging(str(path), "info")
            try:
                log.info("logging smoke test")
            finally:
                shutdown_logging(listener)
            text = path.read_text()
        assert "INFO catagent logging smoke test" in text, text
    finally:
        for handler in list(log.handlers):
            log.removeHandler(handler)
        for handler in old_handlers:
            log.addHandler(handler)
        log.setLevel(old_level)
        log.propagate = old_propagate


def main() -> None:
    test_repository_index()
    test_moved_checkout()
    test_payload_limit()
    asyncio.run(_test_graph_seeding())
    test_cli_option_default()
    test_agent_logging()
    print("APP SKILL/PAYLOAD TESTS: OK")


if __name__ == "__main__":
    main()
