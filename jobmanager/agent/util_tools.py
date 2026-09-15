from langchain.tools import tool
import asyncio, aiofiles, os
from typing import List, Optional, Literal

@tool
async def execute_shell(command: str, cwd: str = None, timeout: int = 60) -> dict:
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
            raise e
        content = {
            "return_code": proc.returncode,
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else "",
        }
        return content
    except Exception as e:
        raise e


@tool
async def edit_file(
    file_path: str,
    lines: List[str],
    mode: Literal['append', 'overwrite', 'replace', 'insert'] = 'append',
    start_line: Optional[int] = None,
    end_line: Optional[int] = None ):
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
    '''
    if not os.path.isfile(file_path):
        await execute_shell.ainvoke(dict(command=f"echo > {file_path}"))
        
    if mode == 'append':
        async with aiofiles.open(file_path, mode="a") as f:
            await f.write('\n')
            for line in lines:
                await f.write(line+'\n')
        return 
    if mode == 'overwrite':
        async with aiofiles.open(file_path, mode="w") as f:
            for line in lines:
                await f.write(line+'\n')
        return 

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
    else:
        if start_idx >= end_idx:
            raise ValueError(f"`start_line` shoule be less or equal to `end_line`.")
        new_lines = lines[0:start_idx] + content2 + lines[end_idx:]

    async with aiofiles.open(file_path, 'w') as f:
        for line in new_lines:
            await f.write(line)


