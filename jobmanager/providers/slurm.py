from jobmanager.base import BaseProvider
from jobmanager.utils import run_shell_command
from jobmanager.models import JobStatus, JobSetting

from pydantic import BaseModel
from typing import List, Dict, Literal, Optional, Set
import os, logging
import asyncio,paramiko, atexit, scp 

logger = logging.getLogger("jobmanager")

class SlurmJobEnv(BaseModel):
    type:Literal["bohr"] = "slurm"
    name: str 
    softwares: List[str]
    initialize_command: List[str]
    # dataset_path: Optional[str]
    # datasets:Optional[List[str]]

class SlurmResource(BaseModel):
    type:Literal["bohr"] = "slurm"
    partition_name: str
    partition_type: Literal["cpu", "gpu", "cpu,gpu"]
    n_nodes:int
    n_cpu_core:Optional[int]
    n_gpu_core:Optional[int]
    memeory_size: Optional[float]
    video_memeory_size: Optional[float]
    price:Optional[float]


def run_ssh_remote_command(client:paramiko.SSHClient, command:str) -> List[List[str]]|None:
    try:
        channelFiles = client.exec_command(command)
        out = []
        for cf in channelFiles:
            try:
                lines = cf.readlines()
            except IOError as e:
                lines = []
            cf.close()
            out.append(lines)
        return out
    except paramiko.SSHException as e:
        logger.error(f"Command {command} failed, {e}.")
        return None  

class SlurmProvider(BaseProvider):
    type: str = "slurm"
    def __init__(self, 
                 name,
                 processing_units: List[str],
                 job_envs_setting: dict,
                 remote_files_setting:List[dict],
                 host: str,
                 user: str,
                 port: int,
                 password: str,
                 remote_root: str = "~/jobs",
                 keyfile: str = "",
                 resources_setting: list = None):
        
        super().__init__(name, processing_units)
        self.job_envs:Dict[str, SlurmJobEnv] = dict()
        for env_key in job_envs_setting.keys():
            self.job_envs[env_key] = SlurmJobEnv(name = env_key,
                                                 softwares = job_envs_setting[env_key]["softwares"],
                                                 initialize_command = job_envs_setting[env_key]["initialize_command"],
                                                 dataset_path=job_envs_setting[env_key].get("dataset_path"),
                                                 datasets=job_envs_setting[env_key].get("datasets"))

        self.remote_files = remote_files_setting
        self.remote_root = remote_root
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            if len(keyfile) > 0:
                self.ssh_client.connect(
                    hostname=host,
                    username=user,
                    key_filename=keyfile,
                    port=port
                )
            else:
                self.ssh_client.connect(
                    hostname=host,
                    username=user,
                    password=password,
                    port=port
                )
            logger.info(f"Connect {user}@{host} successfully.")
        except Exception as e:
            raise RuntimeError(f"Cannot connect to {user}@{host}. Error: {e}")

        self.scp = scp.SCPClient(self.ssh_client.get_transport())
        atexit.register(self.disconnect)

        self.resources:List[SlurmResource] = []
        if resources_setting is None:
            command="sinfo -O 'partition,cpus,nodes,gres' --noheader | sed 's/\\*//g'"
            result = run_ssh_remote_command(self.ssh_client, command) 
            if result:
                for line in result[1]:
                    a,b,c,d = line.strip().split()
                    partition_type = "cpu,gpu" if "gpu" in d else "cpu"
                    self.resources.append(SlurmResource(partition_name=a,
                                                        partition_type=partition_type,
                                                        n_cpu_core=int(b),
                                                        n_nodes=int(c)))

        else:
            for item in resources_setting:
                self.resources.append(SlurmResource(partition_name=item['partition_name'],
                                                    partition_type=item['partition_type'],
                                                    n_cpu_core=item['n_cpu_core'],
                                                    n_nodes=item['n_nodes'],
                                                    n_gpu_core=item.get('n_gpu_core'),
                                                    memeory_size=item.get('memeory_size'),
                                                    video_memeory_size=item.get('video_memeory_size'),
                                                    price=item.get('price')))

    def disconnect(self):
        self.ssh_client.close()
        self.scp.close()
        print("======== SSH disconnect ======== ")
        logger.info(f"SSH connection closed.")
    
    def get_info(self) -> dict:
        return {
            "name": self.name, 
            "processing_units": self.processing_units,
            "job_envs": {key:self.job_envs[key].softwares for key in self.job_envs.keys()},
            "remote_files": [ item["description"] for item in self.remote_files ]
        }
    
    def get_remote_files_list(self) -> List[dict]:
            return self.remote_files
    
    def get_job_env(self, job_env_name:str)->dict:
        return self.job_envs[job_env_name].model_dump(mode="python")
    
    def get_resources(self) -> List[dict]:
        return [item.model_dump(mode="python") for item in self.resources]

    async def get_job_states(self, slurm_job_ids:Set[str]) -> Dict[str, JobStatus]:
        # slurm 无法区分正常完成和报错退出的任务，它们都是complete
        # 而且有的平台上不能用 sacct 命令，scontrol 命令只能查询当前排队或运行中的任务
        result = dict()
        for slurm_job_id in slurm_job_ids:
            command = f"scontrol show jobid {slurm_job_id} | grep JobState | awk '{{print $1}}' | awk -F= '{{print $2}}'"
            output = await asyncio.to_thread(run_ssh_remote_command, client=self.ssh_client, command=command) 
            if output is None:
                logger.error(f"Fail to get the job state of job {slurm_job_id} on {self.name}, SSH error.")
                result[slurm_job_id] = JobStatus.NONE
            elif len(output[1]) > 0:
                state = output[1][0].strip()
                if state == "RUNNING":
                    result[slurm_job_id] = JobStatus.RUNNING
                elif state == "PENDING" or state == "SUSPENDED":
                    result[slurm_job_id] = JobStatus.SUBMITTED
                elif state == "CANCELLED": #一个任务scancel足够长时间后slurm无法得到它的scancel状态
                    result[slurm_job_id] = JobStatus.CANCELLED
                else:
                    result[slurm_job_id] = JobStatus.UNKNOWN
            elif len(output[2]) > 0 and "Invalid job id" in output[2][0]:
                logger.error(f"Fail to get the job state of job {slurm_job_id} on {self.name}. Job may not exist.")
                result[slurm_job_id] = JobStatus.UNKNOWN
            else: 
                logger.error(f"Fail to get the job state of job {slurm_job_id} on {self.name}. Job may not exist.")
                result[slurm_job_id] = JobStatus.NONE
        return result

    async def cancel(self, slurm_job_id: str):
        result = await asyncio.to_thread(run_ssh_remote_command, 
                                         client=self.ssh_client, 
                                         command=f"scancel -v {slurm_job_id}")
        if result is None:
            logger.error(f"Fail to cancel job {slurm_job_id} on {self.name}, SSH error.")
            return
        # scancel 的输出无论成功与否都在标准错误result[2]中
        if len(result[2]) == 2 and "Invalid job id" in result[2][1]:
            logger.info(f"Job {slurm_job_id} does not exist on {self.name}.")
        else:
            logger.info(f"Job {slurm_job_id} on slurm {self.name} is cancelled.")

    async def download(self, slurm_job_id:str, reomte_dir:str, local_dir:str)->bool:
        command = f"cd {reomte_dir} && tar -czf out.tar.gz --exclude='out.tar.gz' * "
        result = await asyncio.to_thread(run_ssh_remote_command,
                                client=self.ssh_client,
                                command=command)
        if result is None:
            logger.error(f"Cannot compress files in {reomte_dir}, SSH error.")
            return False
        if len(result[2]) != 0:
            logger.error(f"Cannot compress files in {reomte_dir}. Error: {''.join(result[2])}.")
            return False 
        # 如果文件不存在会抛出异常 scp.SCPException
        await asyncio.to_thread(self.scp.get, 
                                remote_path=os.path.join(reomte_dir, 'out.tar.gz'), 
                                local_path=local_dir)
        await run_shell_command(f"tar -xf out.tar.gz && rm -f out.tar.gz", cwd=local_dir)
        logger.info(f"The result files of {slurm_job_id} is downloaded in to {local_dir}. ")
        # 清理远程的压缩包文件
        command = f"cd {reomte_dir} && rm -f out.tar.gz"
        await asyncio.to_thread(run_ssh_remote_command, client=self.ssh_client, command=command)
        return True 

    async def submit(self, setting:JobSetting) -> str:
        # 运行这个函数之前假设agent已经把所有的额外内容在run.sh 中修改好
        # 例如要激活哪些环境，export什么变量，NEB计算时的核数。多个任务并行时核数的分配等。
        if not os.path.isdir(setting.local_dir):
            raise NotADirectoryError(f"{setting.local_dir} is not a directory.")
        local_run_sh = os.path.join(setting.local_dir, "run.sh")
        if not os.path.isfile(local_run_sh):
            raise FileNotFoundError(f"There should be a run.sh file in directory {setting.local_dir}. ")

        # 这里要做的只是把目录上传到远端然后提交。
        command = f"if [ -d {setting.remote_dir} ]; then rm -f -r {setting.remote_dir}; fi && mkdir -p {setting.remote_dir}"
        await asyncio.to_thread(run_ssh_remote_command,
                                client=self.ssh_client,
                                command=command)
        # 把当前目录下的内容打包
        await run_shell_command(command="tar -czf input.tar.gz --exclude='input.tar.gz' * ",
                                cwd=setting.local_dir,
                                timeout=120)
        # 上传到远端
        await asyncio.to_thread(self.scp.put,
                                files=os.path.join(setting.local_dir, 'input.tar.gz'),
                                remote_path=setting.remote_dir)
        # 在远端解压然后提交任务
        command = f"cd {setting.remote_dir} && tar -xf input.tar.gz && rm -f input.tar.gz "
        await asyncio.to_thread(run_ssh_remote_command,
                                         client=self.ssh_client,
                                         command=command)
        # 删除本地的Input.tar.gz 
        await run_shell_command(command="rm -f input.tar.gz",
                                cwd=setting.local_dir,
                                timeout=120)

        command = f"bash -l -c 'cd {setting.remote_dir} && sbatch run.sh'"
        result = await asyncio.to_thread(run_ssh_remote_command,
                                                 client=self.ssh_client,
                                                 command=command)
        if result is None:
            raise RuntimeError(f"Fail to submit job {setting.job_name} to {self.name}. SSH error.")
        if len(result[1]) > 0:  
            return result[1][-1].strip().split()[-1]
        else:
            raise RuntimeError(f"Fail to submit job {setting.job_name} to {self.name}. Error: {''.join(result[2])}")
        
    async def cleanup(self, remote_dir:str):
        # 删除远端的目录
        await asyncio.to_thread(run_ssh_remote_command,
                                client=self.ssh_client,
                                command=f"rm -f -r {remote_dir}. ")


    
    




