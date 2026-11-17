"""Deterministic tests for the StepNode path policy in ``util_tools``.

The policy is inlined in ``execute_shell`` (there is no standalone
``_path_policy_error`` helper anymore).  These tests exercise the real tool
functions:

* for a ``StepNode``, ``execute_shell`` may only touch existing paths under
  the project workdir or under a directory containing a skill artifact,
* safe device paths such as ``/dev/null`` are always allowed,
* an ``Orchestrator`` is not bound by the project workdir,
* ``read_file`` always stays inside the project workdir,
* a command whose ``cwd`` is outside the project is rejected.

Known limitation, asserted explicitly: the inline policy only checks paths that
already exist (``is_file() or is_dir()``), so a non-existent outside path is not
denied.  Missing ``project_workdir`` context raises ``KeyError`` instead of
returning a friendly message.

Run from the repository root:

    python -m check.test_path_guard
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from catagent.stepnode.stepnode import create_stepnode_agent
from catagent.tools.other.util_tools import execute_shell, read_file
from check.scripted_model import ScriptedModel

_SYSTEM_SKILLS = Path(__file__).resolve().parent.parent / "catagent" / "system_skills"
_STEPNODE_SKILLS = [
    str(_SYSTEM_SKILLS / name)
    for name in (
        "stepnode_prompt.md",
        "file_and_state_management.md",
        "job_submission.md",
        "file_processing_for_step_node.md",
    )
]


def _runtime(project: str, artifacts=(), role: str = "StepNode"):
    return SimpleNamespace(
        state={"role": role, "skill_artifacts_pathes": set(artifacts)},
        context={"project_workdir": project},
        tool_call_id="call-1",
    )


async def _run_shell(runtime, command: str, cwd: str):
    return await execute_shell.coroutine(runtime, command, cwd, 10)


def _message(command) -> ToolMessage:
    messages = command.update["messages"]
    assert len(messages) == 1, messages
    return messages[0]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def test_project_boundary() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        other = os.path.join(root, "other")
        os.makedirs(project)
        os.makedirs(other)
        inside = os.path.join(project, "input.txt")
        outside = os.path.join(other, "secret.txt")
        Path(inside).write_text("inside", encoding="utf-8")
        Path(outside).write_text("outside", encoding="utf-8")
        runtime = _runtime(project)

        # A file inside the project is allowed.
        message = _message(await _run_shell(runtime, "ls input.txt", project))
        check("Access denied" not in message.content, message.content)

        # An existing file outside the project is denied and named.
        message = _message(await _run_shell(runtime, f"cat {outside}", project))
        check(message.status == "error", f"expected an error, got {message}")
        check("Access denied" in message.content, message.content)
        check(outside in message.content, message.content)

        # A relative escape is resolved against `cwd` and denied.
        message = _message(
            await _run_shell(runtime, "cat ../other/secret.txt", project)
        )
        check("Access denied" in message.content, message.content)
        check(outside in message.content, message.content)

        # Current behaviour: non-existent paths are not checked, so a path
        # outside the project that does not exist yet is not denied.
        missing = os.path.join(other, "not-created.txt")
        message = _message(await _run_shell(runtime, f"cat {missing}", project))
        check("Access denied" not in message.content, message.content)


async def test_skill_artifacts() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        skill_root = os.path.join(root, "skill")
        scripts = os.path.join(skill_root, "scripts")
        os.makedirs(project)
        os.makedirs(scripts)
        script = os.path.join(scripts, "run.py")
        sibling = os.path.join(skill_root, "other.py")
        Path(script).write_text("print('run')\n", encoding="utf-8")
        Path(sibling).write_text("print('other')\n", encoding="utf-8")
        runtime = _runtime(project, artifacts=[script])

        # The directory of a declared skill artifact is allowed.
        message = _message(await _run_shell(runtime, f"ls {script}", project))
        check("Access denied" not in message.content, message.content)

        # A sibling outside the artifact's directory is not.
        message = _message(await _run_shell(runtime, f"ls {sibling}", project))
        check("Access denied" in message.content, message.content)
        check(sibling in message.content, message.content)


async def test_device_paths() -> None:
    with tempfile.TemporaryDirectory() as project:
        runtime = _runtime(project)
        for device in ("/dev/null", "/dev/stdout"):
            message = _message(
                await _run_shell(runtime, f"ls {device}", project)
            )
            check("Access denied" not in message.content, message.content)


async def test_forbidden_network_command() -> None:
    with tempfile.TemporaryDirectory() as project:
        runtime = _runtime(project)
        message = _message(
            await _run_shell(runtime, "curl http://example.com", project)
        )
        check(message.status == "error", f"expected an error, got {message}")
        check(
            "not allowed to send requests" in message.content,
            message.content,
        )


async def test_cwd_boundary_and_missing_directory() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        other = os.path.join(root, "other")
        os.makedirs(project)
        os.makedirs(other)
        runtime = _runtime(project)

        # `ls` names no path, so only the cwd reveals that the command would
        # run outside the project.
        message = _message(await _run_shell(runtime, "ls", other))
        check(message.status == "error", f"expected an error, got {message}")
        check(other in message.content, message.content)

        missing = os.path.join(project, "does-not-exist")
        message = _message(await _run_shell(runtime, "ls", missing))
        check(message.status == "error", f"expected an error, got {message}")
        check("is not a directory" in message.content, message.content)


async def test_orchestrator_is_not_bound() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        other = os.path.join(root, "other")
        os.makedirs(project)
        os.makedirs(other)
        runtime = _runtime(project, role="Orchestrator")

        message = _message(await _run_shell(runtime, "ls", other))
        check("Access denied" not in message.content, message.content)
        check("is not a directory" not in message.content, message.content)


async def test_missing_context_fails_loudly() -> None:
    with tempfile.TemporaryDirectory() as project:
        runtime = SimpleNamespace(
            state={"role": "StepNode", "skill_artifacts_pathes": set()},
            context={},
            tool_call_id="call-1",
        )
        try:
            await _run_shell(runtime, "ls", project)
        except KeyError:
            pass
        else:
            raise AssertionError(
                "missing `project_workdir` should raise KeyError, not silently "
                "skip the path policy"
            )


async def test_read_file_boundary() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        other = os.path.join(root, "other")
        os.makedirs(project)
        os.makedirs(other)
        inside = os.path.join(project, "input.txt")
        outside = os.path.join(other, "secret.txt")
        Path(inside).write_text("line 1\nline 2\n", encoding="utf-8")
        Path(outside).write_text("secret\n", encoding="utf-8")
        runtime = _runtime(project)

        result = await read_file.coroutine(runtime, inside)
        check(isinstance(result, dict), f"expected a dict, got {result!r}")
        check(result["total_lines"] == 2, result)
        check(result["lines"][0][1] == "line 1", result)

        result = await read_file.coroutine(runtime, outside)
        check(isinstance(result, str), f"expected a denial string, got {result!r}")
        check(result.startswith("Access denied"), result)


async def test_stepnode_graph_receives_context() -> None:
    """The real StepNode graph must pass its runtime context to tools."""
    with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as other:
        model = ScriptedModel().call_tool(
            "execute_shell", {"command": "ls", "cwd": other}
        ).say("done")
        agent = create_stepnode_agent(
            model=model,
            checkpointer=InMemorySaver(),
            job_manager_url="http://127.0.0.1:1",
            http_client=None,
            system_skill_pathes=_STEPNODE_SKILLS,
        )
        state = await agent.ainvoke(
            {
                "messages": [HumanMessage(content="do it")],
                "role": "StepNode",
                "workdir": project,
                "loaded_skills": {},
                "skill_artifacts_pathes": set(),
                "stepnode_finalization": False,
            },
            config={"configurable": {"thread_id": "path-guard-e2e"}},
            context={
                "project_workdir": project,
                "step_workdir": project,
                "resource_dirs": {},
            },
        )
        tool_messages = [
            message for message in state["messages"] if isinstance(message, ToolMessage)
        ]
        assert tool_messages, state["messages"]
        assert tool_messages[0].status == "error", tool_messages[0]
        assert other in tool_messages[0].content, tool_messages[0].content


def main() -> None:
    tests = [
        test_project_boundary,
        test_skill_artifacts,
        test_device_paths,
        test_forbidden_network_command,
        test_cwd_boundary_and_missing_directory,
        test_orchestrator_is_not_bound,
        test_missing_context_fails_loudly,
        test_read_file_boundary,
        test_stepnode_graph_receives_context,
    ]
    for test in tests:
        if asyncio.iscoroutinefunction(test):
            asyncio.run(test())
        else:
            test()
        print(f"{test.__name__}: OK")
    print("path guard tests: OK")


if __name__ == "__main__":
    main()
