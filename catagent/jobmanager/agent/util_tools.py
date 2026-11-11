from langchain.tools import tool
import asyncio, aiofiles, os, shlex, glob
from typing import List, Optional, Literal
from pathlib import Path

_MARKDOWN_EXTENSIONS = (".md", ".markdown", ".mdx")
_SHELL_OPERATORS = {"&&", "||", ";", "|", "&", "(", ")", "{", "}", ";;", "|&"}

def _runs_ls_or_find(command: str) -> bool:
    """Return True if the command invokes `ls` or `find`."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars="();<>|&;")
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False

    expect_command = True
    for token in tokens:
        if token in _SHELL_OPERATORS:
            expect_command = True
            continue
        if not expect_command:
            continue
        # Skip environment variable assignments and common command wrappers.
        if "=" in token and token.split("=", 1)[0].isidentifier():
            continue
        if token in ("command", "env", "exec", "nohup", "sudo", "time"):
            continue
        if os.path.basename(token) in ("ls", "find"):
            return True
        expect_command = False
    return False

def _is_markdown_line(line: str) -> bool:
    """Check whether an `ls`/`find` output line refers to a markdown file."""
    stripped = line.strip()
    if not stripped:
        return False

    # `ls -l` prints symlinks as `name -> target`; hide if either is markdown.
    parts = stripped.split(" -> ", 1)
    names = [parts[0].rsplit(maxsplit=1)[-1]]
    if len(parts) == 2:
        names.append(parts[1].rsplit(maxsplit=1)[-1])

    for name in names:
        name = name.rstrip("/@*=|").lower()
        if name.endswith(_MARKDOWN_EXTENSIONS):
            return True
    return False

def _filter_markdown(stdout: str) -> str:
    return "".join(
        line for line in stdout.splitlines(keepends=True)
        if not _is_markdown_line(line)
    )

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
        stdout_text = stdout.decode() if stdout else ""
        if _runs_ls_or_find(command):
            stdout_text = _filter_markdown(stdout_text)
        content = {
            "return_code": proc.returncode,
            "stdout": stdout_text,
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
            raise ValueError(f"`start_line` shoule be less or equal to `end_line`.")
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
def preprocessing_validation(
    local_dir:str, 
    n_group:int = 0, 
    group_dirs:List[str] = None):
    '''
    Validate the preprocessing. Check if there is one and only one `job.json` and `run.sh` in job dir (local_dir) or group dir.
    Args:
        local_dir: Absolute path to job local dir.
        n_group: The number of job groups. For a single job, this is 0.
        group_dirs: The dir names or dir wildcards for each job group, relative to local_dir.
    '''
    if not os.path.isdir(local_dir):
        raise NotADirectoryError(f"{local_dir} is not a directory.")

    if not group_dirs:
        base = Path(local_dir)
        for a in ["run.sh", "job.json"]:
            files = [str(fpath) for fpath in base.rglob(a) if fpath.is_file()]
            nfiles = len(files)
            if nfiles != 1:
                if nfiles == 0:
                    return f"Validation failed. There is no {a} in {local_dir} or its subdirs."
                else:
                    return f"Validation failed. There are {nfiles} {a} in {local_dir} or its subdirs:\n{files}."
                
        return f"Validation pass. Only one `run.sh` and `job.json` exist in {local_dir}."


    dir_names = []
    for item in group_dirs:
        if '*' in item:
            names = [name for name in glob.glob(os.path.join(local_dir, item)) if os.path.isdir(name)]
            if len(names) == 0:
                raise NotADirectoryError(f"Wildcard {item} does not match any dir under {local_dir}.")
            dir_names.extend(names) 
        else:
            p = os.path.join(local_dir, item)
            if not os.path.isdir(p):
                raise NotADirectoryError(f"{p} is not a directory.")
            dir_names.append(p)

    if n_group != len(dir_names):
        raise RuntimeError(f"The number of groups {n_group} and the number of found group dirs are not match. Group dirs found:\n {dir_names}")

    for name in dir_names:
        base = Path(name)
        for a in ["run.sh", "job.json"]:
            files = [str(fpath) for fpath in base.rglob(a) if fpath.is_file()]
            nfiles = len(files)
            if nfiles != 1:
                if nfiles == 0:
                    return f"Validation failed. There is no {a} in group dir {name} or its subdirs."
                else:
                    return f"Validation failed. There are {nfiles} {a} in group dir {name} or its subdirs:\n{files}."
                
    return f"Validation pass. Only one `run.sh` and `job.json` exist in each group dir:\n{dir_names}"


    

    