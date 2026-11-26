from langchain.messages import ToolMessage
from langgraph.types import Command
from langchain.tools import  ToolRuntime, tool

import asyncio, aiofiles, os, shlex 
from typing import List, Optional, Literal, Any
from datetime import datetime
from .parse_path import parse_path
from catagent.workflow.models import skill_artifacts_from_skills
from pathlib import Path

# 这里的路径检测先保持现状。只有当文件或目录存在才拦截，否则会拦截掉一些正常的命令

_SAFE_DEVICE_PATHS = {
    "/dev/null",
    "/dev/stdin",
    "/dev/stdout",
    "/dev/stderr",
}


def _context_value(runtime: Any, key: str) -> Any:
    """Read a key from a dict-like or model-like runtime context."""
    context = getattr(runtime, "context", None)
    if context is None:
        return None
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _resolved(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path))).resolve()


def _allowed_dirs(runtime: Any) -> list[Path]:
    """Directories an agent tool may access for this run.

    ``project_workdir`` is always allowed.  Skill artifacts live outside the
    project, so their directories are added for the run; both the orchestrator
    and StepNode use the same allow-list (skill references are needed while
    preparing a step, and scripts while running it).
    """
    project = _context_value(runtime, "project_workdir")
    if not project:
        # Fail closed instead of silently disabling the policy.
        raise KeyError("project_workdir")
    allowed = [_resolved(project)]

    resource_dirs = _context_value(runtime, "resource_dirs") or {}
    for directory in resource_dirs:
        if directory:
            allowed.append(_resolved(directory))

    state = getattr(runtime, "state", None) or {}
    artifacts = set(state.get("skill_artifacts_pathes") or set())
    # The orchestrator state has `loaded_skills` but not the flattened
    # `skill_artifacts_pathes` field, which is derived for StepNodes.  Derive it
    # here as well so both agents see the same skill files.
    artifacts.update(skill_artifacts_from_skills(state.get("loaded_skills") or {}))
    for artifact in artifacts:
        if artifact:
            # Artifacts are files; allow their directory so helper modules and
            # sibling scripts can be imported/executed.
            directory = _resolved(os.path.dirname(artifact))
            if directory not in allowed:
                allowed.append(directory)
    return allowed


def _is_executable_on_path(path: Path) -> bool:
    """True when ``path`` is an executable that PATH resolves to.

    Shell commands routinely call ``/usr/bin/python3`` or a conda interpreter
    by absolute path.  Those are legitimate even though they are outside the
    project; allow them without opening up the directory they live in.
    """
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return False
    for entry in (os.environ.get("PATH") or "").split(os.pathsep):
        if not entry:
            continue
        try:
            if (Path(entry).resolve() / resolved.name) == resolved:
                return True
        except OSError:
            continue
    return False


def _path_policy_error(runtime: Any, cwd: str, command: str) -> Optional[str]:
    """Return an error message if ``command`` touches a path outside the roots.

    The parser is regex based (``parse_path``), so this is a guard against
    accidental wandering, not a sandbox: an obfuscated command can still read
    outside paths.  See DEVELOPMENT.md for the remaining limitations.
    """
    allowed = _allowed_dirs(runtime)
    project = allowed[0]

    checked = [cwd, *parse_path(command, cwd)]
    for raw in checked:
        if not raw or raw in _SAFE_DEVICE_PATHS:
            continue
        # ``parse_path`` sees the ``//host`` in a URL as an absolute path.
        # Network access is rejected separately by ``execute_shell``, so do not
        # turn a harmless local ``--api-url http://...`` option into a denial.
        if "://" in raw:
            continue
        expanded = os.path.expandvars(os.path.expanduser(raw))
        if not os.path.isabs(expanded):
            expanded = os.path.join(cwd, expanded)
        path = Path(expanded)
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()

        if (not path.is_dir()) and (not path.is_file()):
            continue 

        if any(resolved == root or resolved.is_relative_to(root) for root in allowed):
            continue
        if _is_executable_on_path(path):
            continue

        return f"Access denied. Cannot access {raw}. `execute_shell` is only allowed to access the content under project workdir {project} and the files of loaded skills.\nYou get this error because `execute_shell` detected that you are accessing outside of the workdir. Please focus on the computation task. If this is a misjudgment, rewrite the command with a different format or write them into a script file then execute."
    
    return None


def _file_policy_error(
    runtime: Any,
    file_path: str,
    action: str,
    allow_skill_dirs: bool = False,
) -> Optional[str]:
    """Return an error message when ``file_path`` is outside the allowed roots.

    ``project_workdir`` is the boundary for file tools.  Reading skill files is
    allowed because the orchestrator and StepNode are given those references;
    writing them is not, because installed skills are shared by every thread.
    """
    allowed = _allowed_dirs(runtime)
    project = allowed[0]
    if not allow_skill_dirs:
        allowed = [project]

    resolved = _resolved(file_path)
    if any(resolved == root or resolved.is_relative_to(root) for root in allowed):
        return None
    return f"Access denied. You are only allowed to {action} files under project dir {project}."


@tool
async def execute_shell(runtime:ToolRuntime[Any], 
                        command: str,  
                        cwd, 
                        timeout: int = 60) -> dict:
    """
    Execute a shell command. Tips: Use `cat`, `head`, `grep`, `sed`, `awk` or `echo` etc. to quickly read or edit files. 
    Args:
        command: The command you want to execute.
        cwd: Current working directory. Defaults to the step workdir (the project workdir for the orchestrator).
        timeout: Time limit for command execution (in seconds).
    """
    if not cwd or not os.path.isdir(cwd):
        return Command(update={
            "messages": [ToolMessage(
                            status="error",
                            content=f"Error: {cwd} is not a valid workdir.",
                            tool_call_id=runtime.tool_call_id,
                            name="execute_shell"
                        )]
        })
    # Resolve a relative `cwd` against the step workdir.  Resolving it against
    # the server cwd would silently run in the repository checkout.
    cwd = os.path.expandvars(os.path.expanduser(cwd))
    if not os.path.isabs(cwd):
        base = (_context_value(runtime, "step_workdir")
                or _context_value(runtime, "project_workdir"))
        cwd = os.path.join(base, cwd)
    cwd = os.path.abspath(cwd)
    forbidden = {'curl', 'wget', 'requests', 'httpx', 'aiohttp', 'urllib3'}
    for item in shlex.split(command):
        for forbid_item in forbidden:
            if forbid_item in item:
                return Command(update={
                    "messages": [ToolMessage(
                                    status="error",
                                    content=f"Error: Find {forbid_item} in command, you are not allowed to send requests by execute_shell. Please focus on the computation task. ",
                                    tool_call_id=runtime.tool_call_id,
                                    name="execute_shell"
                                )]
                })

    if not os.path.isdir(cwd):
        return Command(update={
            "messages": [ToolMessage(
                            status="error",
                            content=f"Error: {cwd} is not a directory.",
                            tool_call_id=runtime.tool_call_id,
                            name="execute_shell"
                        )]
        })

    # Both agents are bound by the project workdir.  The orchestrator needs to
    # inspect and prepare files, but it can do that inside the project; the
    # only outside files it may touch are the loaded skill artifacts.
    policy_error = _path_policy_error(runtime, cwd, command)
    if policy_error:
        return Command(update={
                        "messages": [ToolMessage(
                            status="error",
                            content=policy_error,
                            tool_call_id=runtime.tool_call_id,
                            name="execute_shell"
                        )]
            })

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as e:
            proc.kill()
            await proc.wait()
            return Command(update={
                "messages": [ToolMessage(
                    status="error",
                    content=f"Error, timeout: {e}",
                    tool_call_id=runtime.tool_call_id,
                    name="execute_shell"
                )]
            })

        content = {
            "return_code": proc.returncode,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
        }

        # 对于包引用错误，提示向用户寻求帮助，否则agent会像coding agent一样开始debug
        inject_message = ""
        if "ImportError" in content["stderr"] or "ModuleNotFoundError" in content["stderr"]:
            inject_message = "You may call `pause_to_ask` to ask user for help." 

        return Command(update={
            "messages":[
                ToolMessage(
                    content=f"{content}\n{inject_message}",
                    tool_call_id=runtime.tool_call_id,
                    name="execute_shell"
                )
            ]
        })
        
    except Exception as e:
        return Command(update={
            "messages": [ToolMessage(
                status="error",
                content=f"Error: {e}",
                tool_call_id=runtime.tool_call_id,
                name="execute_shell"
            )]
        })


def _with_timestamp(lines: List[str], add_timestamp: bool) -> List[str]:
    """Return a copy of `lines` with a timestamp prefix on the first line."""
    if not add_timestamp:
        return lines
    if not lines:
        return lines  # nothing to prefix
    ts = datetime.now().strftime("[%Y-%m-%d %H:%M:%S] ")
    new_lines = list(lines)
    new_lines[0] = ts + new_lines[0]
    return new_lines

@tool
async def edit_file(
    runtime:ToolRuntime[Any],
    file_path: str,
    lines: List[str],
    mode: Literal['append', 'overwrite', 'replace', 'insert'] = 'append',
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    add_timestamp:bool = False ):
    '''
    Perform complex editing on a text file within a specified range of line numbers.
    Args:
        file_path: Absolute path to the file.
        lines: The content need to be written.
        mode: Edit mode.
            - 'append': Append `lines` to the file.
            - 'overwrite': Overwrite the file by content in `lines`.
            - 'replace': Replace old content in the file from `start_line` to `end_line` by `lines`.
            - 'insert': Insert `lines` before `start_line` in the file.
        start_line: Start line number, begin from 1. `None` means the start of the line.
        end_line: End line number. `None` means the end of the line.
        add_timestamp: Add timestamp at the start of first line.
    '''
    policy_error = _file_policy_error(runtime, file_path, "edit")
    if policy_error:
        return policy_error

    if not os.path.isfile(file_path):
        dirname = os.path.dirname(file_path)
        if not os.path.isdir(dirname):
            raise NotADirectoryError(f"{dirname} is not a directory.")
        async with aiofiles.open(file_path, mode="w") as f:
            pass

    lines = _with_timestamp(lines, add_timestamp)

    if mode == 'append':
        async with aiofiles.open(file_path, mode="a") as f:
            await f.write('\n')
            for line in lines:
                await f.write(line+'\n')
        return f"Appended {len(lines)} lines to {file_path}"
    if mode == 'overwrite':
        async with aiofiles.open(file_path, mode="w") as f:
            for line in lines:
                await f.write(line+'\n')
        return f"Overwrote {file_path} with {len(lines)} lines"

    content2 = [item+'\n' for item in lines]

    async with aiofiles.open(file_path, 'r') as f:
        lines = await f.readlines()
    nline = len(lines)
    start_idx = start_line - 1 if start_line else 0
    start_idx = max(0, start_idx)
    end_idx = end_line if end_line else nline
    end_idx = min(end_idx, nline)

    if mode == 'insert':
        if start_idx > nline - 1:
            new_lines = lines + content2
        else:
            new_lines = lines[0:start_idx] + content2 + lines[start_idx:]
        result = f"Inserted {len(content2)} lines before line {start_idx + 1} of {file_path}"
    else:
        if start_idx >= end_idx:
            raise ValueError(f"`start_line` should be less or equal to `end_line`.")
        new_lines = lines[0:start_idx] + content2 + lines[end_idx:]
        result = (
            f"Replaced lines {start_idx + 1}-{end_idx} of {file_path} "
            f"with {len(content2)} lines"
        )

    async with aiofiles.open(file_path, 'w') as f:
        for line in new_lines:
            await f.write(line)
    return result


@tool
async def read_file(
    runtime:ToolRuntime[Any],
    file_path: str,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> dict:
    '''
    Read a text file by line number. This is the counterpart of `edit_file`.
    Args:
        file_path: Absolute path to the file.
        start_line: First line to read, begin from 1. `None` means the start of the file.
        end_line: Last line to read (inclusive). `None` means the end of the file.
    Returns:
        A dict containing the file path, the total number of lines, the actual
        line range that was read, and `lines`, a list of `[line_number, line_content]`
        pairs so that every line number maps exactly to its content.
    '''
    policy_error = _file_policy_error(runtime, file_path, "read", allow_skill_dirs=True)
    if policy_error:
        return policy_error

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    async with aiofiles.open(file_path, mode="r") as f:
        file_lines = await f.readlines()

    total_lines = len(file_lines)
    first = 1 if start_line is None else start_line
    last = total_lines if end_line is None else end_line

    if first < 1:
        raise ValueError(f"`start_line` should be greater than or equal to 1.")
    if end_line is not None and last < first:
        raise ValueError(f"`end_line` should be greater than or equal to `start_line`.")

    # Clamp the requested range to the file boundaries.
    first = min(first, total_lines + 1)
    last = min(max(last, first - 1), total_lines)

    content = [
        [line_num, file_lines[line_num - 1].rstrip('\r\n')]
        for line_num in range(first, last + 1)
    ]

    return {
        "file_path": os.path.abspath(file_path),
        "total_lines": total_lines,
        "start_line": first if content else None,
        "end_line": last if content else None,
        "returned_lines": len(content),
        "lines": content,
    }
