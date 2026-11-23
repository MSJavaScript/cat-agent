from catagent.jobmanager.base import BaseProvider
from catagent.jobmanager.utils import run_shell_command
from catagent.jobmanager.models import JobStatus, JobSetting
from pydantic import BaseModel
from typing import List, Dict, Literal, Optional, Set
import os, json, logging, shlex, signal, subprocess, tempfile


logger = logging.getLogger("jobmanager")

# Where the pid -> job directory registry lives.  It must survive a restart of
# the jobmanager process, otherwise a finished job cannot be distinguished from
# a lost one.  It does not need to survive a machine reboot: in that case the
# job state is genuinely UNKNOWN, which is safer than assuming success.
STATE_DIR_ENV = "CATAGENT_LOCAL_STATE_DIR"

class LocalJobEnv(BaseModel):
    type:Literal["local"] = "local"
    name: str
    softwares: List[str]
    initialize_command: Optional[List[str]]


def _state_root() -> str:
    '''
    返回临时文件的存放目录
    '''
    root = os.environ.get(STATE_DIR_ENV) or os.path.join(
        tempfile.gettempdir(), "catagent-local-jobs")
    os.makedirs(root, exist_ok=True)
    return root


def _read_json(path: str) -> dict:
    try:
        with open(path, mode="r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _atomic_write_json(path: str, payload: dict) -> None:
    '''
    原子化写入json
    '''
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    # 创建一个唯一的临时文件，fd 是文件描述符，tmp_path是文件路径
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-")
    try:
        # 文件描述符用os.fdopen打开，不能用open
        with os.fdopen(fd, mode="w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except ValueError:
        return False


class LocalProvider(BaseProvider):
    type: str = "local"
    def __init__(self, name, processing_units:List[str], job_envs_setting: dict):
        super().__init__(name, processing_units)
        self.job_envs:Dict[str, LocalJobEnv] = dict()
        for env_key in job_envs_setting.keys():
            self.job_envs[env_key] = LocalJobEnv(name=env_key,
                                                softwares = job_envs_setting[env_key]["softwares"],
                                                initialize_command = job_envs_setting[env_key].get("initialize_command"))
        self.resources = []
        # 保存Popen句柄，便于用 poll() 回收子进程。
        # 否则退出的子进程会变成僵尸，os.kill(pid, 0) 仍然成功，状态永远停在RUNNING。
        self._procs: Dict[int, subprocess.Popen] = {}

    def get_info(self)->dict:
        return {
            "name": self.name, 
            "processing_units": self.processing_units,
            "job_envs": {key:self.job_envs[key].softwares for key in self.job_envs.keys()},
            "remote_files": []
        }

    def get_remote_files_list(self) -> List[dict]:
        return []
    
    def get_job_env(self, job_env_name:str)->dict:
        return self.job_envs[job_env_name].model_dump(mode="python")

    def get_resources(self) -> List[dict]:
        return []

    def _registry_path(self, pid: int) -> str:
        return os.path.join(_state_root(), f"{pid}.json")

    def _register(self, pid: int, local_dir: str, token: str) -> None:
        _atomic_write_json(self._registry_path(pid), {
            "pid": pid,
            "local_dir": os.path.abspath(local_dir),
            "token": token,
        })

    def _local_dir_for_pid(self, pid: int) -> Optional[str]:
        entry = _read_json(self._registry_path(pid))
        local_dir = entry.get("local_dir")
        if isinstance(local_dir, str) and os.path.isdir(local_dir):
            return local_dir
        return None

    @staticmethod
    def _read_exit_code(local_dir: str) -> Optional[int]:
        status = _read_json(os.path.join(local_dir, ".catagent", "status.json"))
        code = status.get("exit_code")
        return code if isinstance(code, int) else None

    @staticmethod
    def _read_pid_file(run_dir: str) -> Optional[int]:
        try:
            with open(os.path.join(run_dir, "pid"), mode="r", encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return None

    async def get_job_states(self, local_pids:Set[str]) -> Dict[str, JobStatus]:
        result = dict()
        for pid in local_pids:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                result[pid] = JobStatus.UNKNOWN
                continue

            local_dir = self._local_dir_for_pid(pid_int)
            proc = self._procs.get(pid_int)

            if proc is not None:
                # poll() 会回收已退出的子进程并返回退出码。
                rc = proc.poll()
                if rc is None:
                    result[pid] = JobStatus.RUNNING
                    continue
                self._procs.pop(pid_int, None)
                exit_code = self._read_exit_code(local_dir) if local_dir else None
                if exit_code is None:
                    exit_code = rc
                result[pid] = (JobStatus.COMPLETED if exit_code == 0
                               else JobStatus.FAILED)
                continue

            # No live Popen handle (e.g. the service restarted).  The wrapper
            # writes an atomic status file on exit; it is the only reliable
            # source of the exit code.  A dead pid without that file is
            # UNKNOWN, never COMPLETED.
            exit_code = self._read_exit_code(local_dir) if local_dir else None
            if exit_code is not None:
                result[pid] = (JobStatus.COMPLETED if exit_code == 0
                               else JobStatus.FAILED)
            elif _pid_alive(pid_int):
                result[pid] = JobStatus.RUNNING
            else:
                result[pid] = JobStatus.UNKNOWN
        return result

    async def cancel(self, local_pid: str):
        try:
            pid = int(local_pid)
            # The job runs in its own session; signal the whole group so child
            # processes started by run.sh are stopped too.
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except ProcessLookupError:
                os.kill(pid, signal.SIGTERM)
            logger.info(f"Local job {local_pid} is killed.")
        except (ProcessLookupError, ValueError):
            logger.error(f"Local job {local_pid} does not exist.")
        except PermissionError:
            logger.error(f"Cannot kill {local_pid}, permission error.")
 
    
    async def download(self, local_pid:str, reomte_dir:str, local_dir:str)->bool:
        # 本地作业的结果本来就在 local_dir 中，remote_dir 通常为空。
        if reomte_dir:
            await run_shell_command(f"mv {reomte_dir} {local_dir}")
        return True

    def _write_runner(self, local_dir: str, job_env_name: str) -> str:
        """Write a wrapper that records the real exit code atomically.

        The jobmanager process is not the parent of `run.sh` across a restart,
        so `waitpid` is unavailable later.  The wrapper persists the exit code
        before exiting, which makes the status recoverable and prevents a dead
        failed process from being reported as completed.
        """
        run_dir = os.path.join(local_dir, ".catagent")
        os.makedirs(run_dir, exist_ok=True)
        wrapper = os.path.join(run_dir, "run.sh")
        env = self.job_envs[job_env_name]
        init_lines = "\n".join(env.initialize_command or [])
        script = f"""#!/usr/bin/env bash
set +e
cd {shlex.quote(os.path.abspath(local_dir))} || exit 127
mkdir -p .catagent
printf '%s\\n' "$$" > .catagent/pid
{init_lines}
bash run.sh > .catagent/stdout.log 2> .catagent/stderr.log
rc=$?
tmp="$(mktemp .catagent/.status.XXXXXX)"
printf '{{"exit_code": %d, "finished_at": "%s"}}\\n' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$tmp"
mv "$tmp" .catagent/status.json
exit "$rc"
"""
        with open(wrapper, mode="w", encoding="utf-8") as f:
            f.write(script)
        return wrapper

    async def submit(self, setting:JobSetting)->str:
        if not os.path.isdir(setting.local_dir):
            raise NotADirectoryError(f"{setting.local_dir} is not a directory.")

        local_run = os.path.join(setting.local_dir, "run.sh")
        if not os.path.isfile(local_run):
            raise FileNotFoundError(f"There should be a run.sh file in directory {setting.local_dir}. ")

        token = setting.idempotency_key or setting.job_name or setting.local_dir
        run_dir = os.path.join(setting.local_dir, ".catagent")
        submit_file = os.path.join(run_dir, "submit.json")

        # Idempotent submit: if this exact group was already started, return
        # its pid instead of launching a second copy after a crash.
        previous = _read_json(submit_file)
        if previous.get("token") == token:
            pid = previous.get("pid")
            if pid:
                pid = int(pid)
                logger.info(f"Local job for {token} is already started with pid {pid}.")
                self._register(pid, setting.local_dir, token)
                return str(pid)
            # Crash window: the wrapper was spawned but the manager died before
            # writing the pid.  The wrapper publishes its own pid, so recover it
            # instead of starting a second process.
            recovered = self._read_pid_file(run_dir)
            if recovered is not None and _pid_alive(recovered):
                logger.info(f"Recovered already-launched local job for {token} "
                            f"with pid {recovered}.")
                _atomic_write_json(submit_file, {
                    "token": token,
                    "local_dir": os.path.abspath(setting.local_dir),
                    "status": "running",
                    "pid": recovered,
                })
                self._register(recovered, setting.local_dir, token)
                return str(recovered)

        os.makedirs(run_dir, exist_ok=True)
        for path in (submit_file, os.path.join(run_dir, "status.json"),
                     os.path.join(run_dir, "pid")):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

        wrapper = self._write_runner(setting.local_dir, setting.job_env_name)

        # Mark the intent before spawning so a retry has a chance to detect it.
        _atomic_write_json(submit_file, {
            "token": token,
            "local_dir": os.path.abspath(setting.local_dir),
            "status": "launching",
        })

        proc = subprocess.Popen(['bash', wrapper],
            cwd=setting.local_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        _atomic_write_json(submit_file, {
            "token": token,
            "local_dir": os.path.abspath(setting.local_dir),
            "status": "running",
            "pid": proc.pid,
        })
        self._register(proc.pid, setting.local_dir, token)
        self._procs[proc.pid] = proc
        return str(proc.pid)
        

    async def cleanup(self, remote_dir: str):
        pass 

    def disconnect(self):
        pass
