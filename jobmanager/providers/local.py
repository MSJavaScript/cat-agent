from jobmanager.base import BaseProvider
from jobmanager.utils import run_shell_command
from jobmanager.models import JobStatus, JobSetting
from pydantic import BaseModel
from typing import List, Dict, Literal, Optional, Set
import os, logging, subprocess, signal


logger = logging.getLogger("jobmanager")

class LocalJobEnv(BaseModel):
    type:Literal["local"] = "local"
    name: str
    softwares: List[str]
    initialize_command: Optional[List[str]]

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

    async def get_job_states(self, local_pids:Set[str]) -> Dict[str, JobStatus]:
        result = dict()
        for pid in local_pids:
            proc = self._procs.get(int(pid)) if str(pid).isdigit() else None
            if proc is not None:
                # poll() 会回收已退出的子进程并返回退出码。
                rc = proc.poll()
                if rc is None:
                    result[pid] = JobStatus.RUNNING
                else:
                    self._procs.pop(int(pid), None)
                    result[pid] = JobStatus.COMPLETED if rc == 0 else JobStatus.FAILED
                continue
            # 进程句柄不在内存中（例如服务重启过），退回按pid探测。
            try:
                os.kill(int(pid), 0)
                result[pid] = JobStatus.RUNNING
            except (ProcessLookupError, ValueError):
                result[pid] = JobStatus.COMPLETED
            except PermissionError:
                result[pid] = JobStatus.RUNNING
        return result

    async def cancel(self, local_pid: str):
        try:
            os.kill(int(local_pid), signal.SIGTERM)
            logger.info(f"Local job {local_pid}  is killed.")
        except (ProcessLookupError, ValueError):
            logger.error(f"Local job {local_pid} does not exist.")
        except PermissionError:
            logger.error(f"Cannot kill {local_pid}, permission error.")
 
    
    async def download(self, local_pid:str, reomte_dir:str, local_dir:str)->bool:
        # 本地作业的结果本来就在 local_dir 中，remote_dir 通常为空。
        if reomte_dir:
            await run_shell_command(f"mv {reomte_dir} {local_dir}")
        return True

    async def submit(self, setting:JobSetting)->str:
        if not os.path.isdir(setting.local_dir):
            raise NotADirectoryError(f"{setting.local_dir} is not a directory.")

        local_run = os.path.join(setting.local_dir, "run.sh")
        if not os.path.isfile(local_run):
            raise FileNotFoundError(f"There should be a run.sh file in directory {setting.local_dir}. ")
        proc = subprocess.Popen(['bash', local_run],
            cwd=setting.local_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._procs[proc.pid] = proc
        return str(proc.pid)
        

    async def cleanup(self, remote_dir: str):
        pass 

    def disconnect(self):
        pass 