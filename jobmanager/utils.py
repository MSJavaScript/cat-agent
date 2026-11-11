import asyncio, hashlib

async def run_shell_command(command: str, cwd: str = None, timeout: int = 60)->dict:
    '''
    Run a bash command.
    '''
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


def to_md5_hex(input:str):
    for char in input:
        if ord(char) > 127:
            raise ValueError(f"Non ascii character is not allowed. {char}")
    md5_hash = hashlib.md5(input.encode('ascii'))
    return md5_hash.hexdigest()





