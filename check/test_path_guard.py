"""Deterministic tests for the workspace path policy in ``util_tools``.

The policy is shared by the orchestrator and StepNodes:

* ``execute_shell`` may only touch paths under the project workdir, the files
  of loaded skills, and executables that PATH resolves to,
* safe device paths such as ``/dev/null`` are always allowed,
* ``read_file`` may read project files and skill references, ``edit_file`` may
  only write inside the project,
* ``cwd`` defaults to the step workdir and is rejected when outside the project,
* ``set_step_workdir`` cannot move a step outside the project.

The parser is regex based, so the guard stops accidental wandering rather than
an adversarial command (see DEVELOPMENT.md).  Missing ``project_workdir``
context raises ``KeyError`` instead of returning a friendly message.

Run from the repository root:

    python -m check.test_path_guard
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

from langchain.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from catagent.stepnode.stepnode import create_stepnode_agent
from catagent.tools.other.util_tools import edit_file, execute_shell, read_file
from catagent.workflow.step_edit_tool import set_step_workdir
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

        # Current behaviour: a path outside the project is denied whether or
        # not it exists, so a write cannot create files elsewhere either.
        missing = os.path.join(other, "not-created.txt")
        message = _message(await _run_shell(runtime, f"cat {missing}", project))
        check("Access denied" in message.content, message.content)
        check(missing in message.content, message.content)

        # A missing path *inside* the project is not a policy violation.
        missing_inside = os.path.join(project, "not-created.txt")
        message = _message(
            await _run_shell(runtime, f"cat {missing_inside}", project)
        )
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

        # The orchestrator keeps the paths inside `loaded_skills` rather than
        # in the flattened field; the same allow-list must be derived from it.
        orchestrator = SimpleNamespace(
            state={
                "role": "Orchestrator",
                "loaded_skills": {
                    "demo": {"references": [], "scripts": [{"path": script}]},
                },
            },
            context={"project_workdir": project},
            tool_call_id="call-2",
        )
        message = _message(await _run_shell(orchestrator, f"ls {script}", project))
        check("Access denied" not in message.content, message.content)


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


async def test_orchestrator_is_bound() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        other = os.path.join(root, "other")
        os.makedirs(project)
        os.makedirs(other)
        runtime = _runtime(project, role="Orchestrator")

        message = _message(await _run_shell(runtime, "ls", other))
        check(message.status == "error", f"expected an error, got {message}")
        check(other in message.content, message.content)

        # The orchestrator can still work inside the project.
        message = _message(await _run_shell(runtime, "ls", project))
        check("Access denied" not in message.content, message.content)


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


async def test_skill_reference_read_but_not_edit() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        skill_root = os.path.join(root, "skill")
        os.makedirs(project)
        os.makedirs(skill_root)
        reference = os.path.join(skill_root, "reference.md")
        Path(reference).write_text("docs\n", encoding="utf-8")
        runtime = _runtime(project, artifacts=[reference])

        result = await read_file.coroutine(runtime, reference)
        check(isinstance(result, dict), f"expected a dict, got {result!r}")
        check(result["lines"][0][1] == "docs", result)

        result = await edit_file.coroutine(runtime, reference, ["changed"], "overwrite")
        check(isinstance(result, str), f"expected a denial string, got {result!r}")
        check(result.startswith("Access denied"), result)
        check(Path(reference).read_text(encoding="utf-8") == "docs\n", "skill file was modified")


async def test_symlink_escape() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        other = os.path.join(root, "other")
        os.makedirs(project)
        os.makedirs(other)
        outside = os.path.join(other, "secret.txt")
        Path(outside).write_text("secret\n", encoding="utf-8")
        link = os.path.join(project, "link")
        os.symlink(other, link)
        runtime = _runtime(project)

        result = await read_file.coroutine(runtime, os.path.join(link, "secret.txt"))
        check(isinstance(result, str), f"expected a denial string, got {result!r}")
        check(result.startswith("Access denied"), result)

        message = _message(
            await _run_shell(runtime, f"cat {os.path.join(link, 'secret.txt')}", project)
        )
        check(message.status == "error", f"expected an error, got {message}")
        check("Access denied" in message.content, message.content)


async def test_default_cwd_uses_step_workdir() -> None:
    with tempfile.TemporaryDirectory() as project, tempfile.TemporaryDirectory() as other:
        runtime = SimpleNamespace(
            state={"role": "StepNode", "skill_artifacts_pathes": set()},
            context={"project_workdir": project, "step_workdir": project},
            tool_call_id="call-1",
        )
        message = _message(
            await execute_shell.coroutine(runtime, "echo ok > probe.txt", None, 10)
        )
        check(message.status != "error", message.content)
        check(Path(project, "probe.txt").is_file(), message.content)
        check(not Path(other, "probe.txt").exists(), "command ran in the wrong cwd")

        # A relative `cwd` is resolved against the step workdir, not the server
        # process cwd (which may be the repository checkout).
        subdir = Path(project, "sub")
        subdir.mkdir()
        message = _message(
            await execute_shell.coroutine(runtime, "echo rel > rel.txt", "sub", 10)
        )
        check(message.status != "error", message.content)
        check((subdir / "rel.txt").is_file(), message.content)


async def test_executable_on_path_is_allowed() -> None:
    with tempfile.TemporaryDirectory() as project:
        executable = shutil.which("sh") or shutil.which("ls")
        check(executable is not None, "no shell executable found")
        runtime = _runtime(project)
        message = _message(
            await _run_shell(runtime, f"{executable} --version", project)
        )
        check("Access denied" not in message.content, message.content)


def test_shell_cwd_is_optional() -> None:
    """The model may omit `cwd`; the tool then uses the step workdir."""
    schema = execute_shell.tool_call_schema.model_json_schema()
    check("cwd" not in schema.get("required", []), schema)
    check("command" in schema.get("required", []), schema)


def test_set_step_workdir_boundary() -> None:
    with tempfile.TemporaryDirectory() as root:
        project = os.path.join(root, "project")
        other = os.path.join(root, "other")
        subdir = os.path.join(project, "jobs")
        os.makedirs(subdir)
        os.makedirs(other)
        os.symlink(other, os.path.join(project, "link"))
        runtime = SimpleNamespace(
            state={"steps": {"s1": object()}},
            context={"project_workdir": project},
            tool_call_id="call-1",
        )

        for bad in (other, os.path.join(project, "link")):
            command = set_step_workdir.func(runtime, "s1", bad)
            message = _message(command)
            check(message.status == "error", f"expected an error, got {message}")
            check("Access denied" in message.content, message.content)

        command = set_step_workdir.func(runtime, "s1", subdir)
        check(
            command.update["steps"]["s1"]["workdir"] == str(Path(subdir).resolve()),
            command,
        )


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
        test_orchestrator_is_bound,
        test_missing_context_fails_loudly,
        test_read_file_boundary,
        test_skill_reference_read_but_not_edit,
        test_symlink_escape,
        test_default_cwd_uses_step_workdir,
        test_executable_on_path_is_allowed,
        test_shell_cwd_is_optional,
        test_set_step_workdir_boundary,
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
