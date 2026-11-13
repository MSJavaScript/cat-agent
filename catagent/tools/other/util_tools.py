from langchain.tools import tool
import asyncio, aiofiles, os
from typing import List, Optional, Literal
from datetime import datetime
from langchain.messages import ToolMessage
from langgraph.types import Command
from langchain.tools import  ToolRuntime

@tool
async def execute_shell(runtime:ToolRuntime, command: str, cwd: str = None, timeout: int = 60) -> dict:
    """
    Execute a shell command. Tips: Use `cat`, `head`, `grep`, `sed`, `awk` or `echo` etc. to quickly read or edit files. 
    Args:
        command: The command you want to execute.
        cwd: Current working directory.
        timeout: Time limit for command execution (in seconds).
    """

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
