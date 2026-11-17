"""Deterministic tests for the StepNode path policy in ``util_tools``.

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
from langgraph.types import Command

from catagent.stepnode.stepnode import create_stepnode_agent
from catagent.tools.other.util_tools import _path_policy_error, execute_shell
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


def test_project_boundary() -> None:
    with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as other:
        inside = os.path.join(project, "input.txt")
        outside = os.path.join(other, "secret.txt")
        runtime = _runtime(project)

        assert _path_policy_error(
            runtime, [inside], tool_name="execute_shell"
        ) is None
        # Non-existent outside paths are denied too: a command could create
        # them, and the boundary must not depend on the current filesystem.
        error = _path_policy_error(
            runtime,
            [inside, outside, os.path.join(other, "new.txt")],
            tool_name="execute_shell",
        )
        assert error is not None
        assert outside in error
        assert os.path.join(other, "new.txt") in error

        # Relative paths are interpreted relative to the project workdir.
        assert _path_policy_error(
            runtime, ["sub/out.txt"], tool_name="execute_shell"
        ) is None
        assert _path_policy_error(
            runtime, ["../escape.txt"], tool_name="execute_shell"
        ) is not None


def test_skill_artifacts() -> None:
    with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as skill_root:
        script = os.path.join(skill_root, "scripts", "run.py")
        sibling = os.path.join(skill_root, "scripts", "other.py")
        os.makedirs(os.path.dirname(script))
        runtime = _runtime(project, artifacts=[script])

        # The bug in the original loop: `continue` only left the inner loop,
        # so an allowed skill script was denied immediately afterwards.
        assert _path_policy_error(
            runtime,
            [script],
            tool_name="execute_shell",
            allow_skill_artifacts=True,
        ) is None
        assert _path_policy_error(
            runtime,
            [sibling],
            tool_name="execute_shell",
            allow_skill_artifacts=True,
        ) is not None
        # edit_file/read_file do not opt into skill artifacts.
        assert _path_policy_error(
            runtime,
            [script],
            tool_name="read_file",
        ) is not None


def test_device_paths_and_missing_context() -> None:
    with tempfile.TemporaryDirectory() as project:
        runtime = _runtime(project)
        assert _path_policy_error(
            runtime, ["/dev/null"], tool_name="execute_shell"
        ) is None
        # /dev/stdout is a symlink into /proc; it must still be recognised
        # before its resolved form is checked against the project root.
        assert _path_policy_error(
            runtime, ["/dev/stdout"], tool_name="execute_shell"
        ) is None

        runtime.context = {}
        error = _path_policy_error(
            runtime, [os.path.join(project, "a.txt")], tool_name="execute_shell"
        )
        assert error is not None and "not configured" in error


def test_execute_shell_checks_cwd() -> None:
    with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as other:
        runtime = _runtime(project)
        # `ls` names no path, so only the cwd reveals that the command would
        # run outside the project.
        command = asyncio.run(
            execute_shell.coroutine(runtime, "ls", other, 5)
        )
        assert isinstance(command, Command), command
        message = command.update["messages"][0]
        assert message.status == "error"
        assert other in message.content


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
                "workdir": project,
                "loaded_skills": {},
                "skill_artifacts_pathes": set(),
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
        test_device_paths_and_missing_context,
        test_execute_shell_checks_cwd,
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
