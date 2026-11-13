from catagent.jobmanager.base import BaseProvider
from catagent.jobmanager.utils import run_shell_command
from catagent.jobmanager.models import JobStatus, JobSetting

from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional, Set, Tuple
import os, logging, json, asyncio, shlex, shutil
from subprocess import check_output

logger = logging.getLogger("jobmanager")

# 同时查询状态的最大并发数，避免一次性拉起过多CLI进程。
_STATUS_QUERY_CONCURRENCY = 16
class BohrJobEnv(BaseModel):
    type:Literal["bohr"] = "bohr"
    name: str
    image_path: str
    softwares: List[str]
    initialize_command: Optional[List[str]]
    # dataset_path: Optional[str]
    # datasets:Optional[List[str]]

class BohrResource(BaseModel):
    '''
    Information of a machine on Bohr platform.
    '''
    type:Literal["bohr"] = "bohr"
    machine_name: str = Field(description="Identifier of the machine.")
    machine_type: Literal["cpu", "gpu"]
    n_cpu_core: Optional[int]
    n_gpu_core: Optional[int]
    memeory_size: Optional[float]
    video_memeory_size: Optional[float]
    price: Optional[float]

class BohrProvider(BaseProvider):
    type: str = "bohr"
    def __init__(self, 
                 name, 
                 processing_units:List[str],
                 job_envs_setting: dict,
                 remote_files_setting:List[dict],
                 project_id:str,
                 platform:str = "ali",
                 job_type:str = "container",
                 resources_setting: list = None):
        
        super().__init__(name, processing_units)

        self.job_envs:Dict[str, BohrJobEnv] = dict()
        for env_key in job_envs_setting.keys():
            self.job_envs[env_key] = BohrJobEnv(name=env_key,
                                                image_path=job_envs_setting[env_key]["image_path"],
                                                softwares = job_envs_setting[env_key]["softwares"],
                                                initialize_command = job_envs_setting[env_key].get("initialize_command"))
        self.remote_files = remote_files_setting
        self.project_id = project_id
        self.platform = platform
        self.job_type = job_type
        # 获取bohr平台的资源信息，这里可以过滤一下机型
        self.resources:List[BohrResource] = []

        if resources_setting is None:
            output = check_output("bohr machine list -c 'gpu' --json", shell=True)
            result:list = json.loads(output)
            for item in result:
                self.resources.append(BohrResource(machine_name=item['skuName'],
                                                machine_type='gpu',
                                                n_cpu_core=item['cpuCoreNum'],
                                                n_gpu_core=item['gpuCoreNum'],
                                                memeory_size=item['memory'],
                                                video_memeory_size=item['vmemory'],
                                                price=item['discountPrice']))
            output = check_output("bohr machine list -c 'cpu' --json", shell=True)
            result:list = json.loads(output)
            for item in result:
                self.resources.append(BohrResource(machine_name=item['skuName'],
                                                machine_type='cpu',
                                                n_cpu_core=item['cpuCoreNum'],
                                                n_gpu_core=0,
                                                memeory_size=item['memory'],
                                                video_memeory_size=0,
                                                price=item['discountPrice']))
        else:
            for item in resources_setting:
                self.resources.append(BohrResource(machine_name=item['machine_name'],
                                                machine_type=item['machine_type'],
                                                n_cpu_core=item.get('n_cpu_core'),
                                                n_gpu_core=item.get('n_gpu_core'),
                                                memeory_size=item.get('memeory_size'),
                                                video_memeory_size=item.get('video_memeory_size'),
                                                price=item.get('price')))

    # 该函数只用于显示简要的作业环境和机器信息，是否具有CPU和GPU
    def get_info(self)->dict:
        return {
            "name": self.name, 
            "processing_units": self.processing_units,
            "job_envs": {key:self.job_envs[key].softwares for key in self.job_envs.keys()},
            "remote_files": [ {"description":item["description"], 
                               "files":item["files"]} for item in self.remote_files ]
        }

    def get_remote_files_list(self) -> List[dict]:
        return self.remote_files

    def get_job_env(self, job_env_name:str)->dict:
        return self.job_envs[job_env_name].model_dump(mode="python")

    def get_resources(self) -> List[dict]:
        return [item.model_dump(mode="python") for item in self.resources]

    async def _get_one_job_state(self, jid: str, semaphore: asyncio.Semaphore) -> Tuple[str, JobStatus]:
        command = f"bohr job describe -j {jid} | awk -F, '{{print $12}} ' | tail -1"
        async with semaphore:
            try:
                output = await run_shell_command(command)
            except Exception as e:
                logger.error(f"Cannot get the state of job with id {jid}. Error:{e}")
                return jid, JobStatus.NONE
        if output["return_code"] != 0:
            logger.error(f"Cannot get the state of job with id {jid}: {output['stderr'].strip()}")
            return jid, JobStatus.NONE
        statusStr = output["stdout"].strip()
        if statusStr:
            # `awk ... | tail -1` also returns the header cell while the job
            # is still being created, and the CLI may emit other transient
            # words before the job is running.  Treat anything we cannot map
            # as "state not known yet" (NONE) instead of UNKNOWN: UNKNOWN is a
            # terminal state for the manager and would make it download the
            # results of a job that has not produced any output yet.
            statusStr = statusStr.splitlines()[-1].strip()
        if statusStr in ("Finished", "Completed", "Succeeded", "Success"):
            return jid, JobStatus.COMPLETED
        if statusStr in ("Failed", "Error"):
            return jid, JobStatus.FAILED
        if statusStr == "Running":
            return jid, JobStatus.RUNNING
        if statusStr in ("Scheduling", "Pending", "Queued", "Queueing",
                         "Waiting", "Creating", "Submitted"):
            return jid, JobStatus.SUBMITTED
        if statusStr in ("Cancelled", "Canceled", "Terminated"):
            return jid, JobStatus.CANCELLED
        if statusStr in ("", "statusStr"):
            logger.debug(f"Job {jid} has no status yet (describe returned {statusStr!r}).")
        else:
            logger.warning(f"Job {jid} has an unrecognized status {statusStr!r}; "
                           f"will query it again.")
        return jid, JobStatus.NONE

    async def get_job_states(self, bohr_job_ids:Set[str]) -> Dict[str, JobStatus]:
        # 并发查询，避免顺序执行几百个CLI调用拖长整个轮询周期。
        semaphore = asyncio.Semaphore(_STATUS_QUERY_CONCURRENCY)
        results = await asyncio.gather(
            *(self._get_one_job_state(jid, semaphore) for jid in bohr_job_ids))
        return dict(results)

    async def cancel(self, bohr_job_id: str):
        command = f"bohr job terminate {bohr_job_id}" 
        output = await run_shell_command(command, None, 120)
        if output["return_code"] != 0:
            logger.error(f"Failed to cancel the job {bohr_job_id} on bohr, error: {output["stderr"]}")
            #raise RuntimeError(f"Failed to cancel the job {bohr_job_id} on bohr, error: {output["stderr"]}")
        std_out_err = output["stdout"] + output["stderr"]
        if "Finished" in std_out_err or "Failed" in std_out_err:
            logger.warning(f"Job {bohr_job_id} on bohr is Finished or Failed, do not need to cancel.")
        elif "Successfully" in std_out_err:
            logger.info(f"Job {bohr_job_id}  on bohr is cancelled.")


    async def download(self, bohr_job_id:str, reomte_dir:str, local_dir:str)->bool:
        if not os.path.isdir(local_dir):
            raise NotADirectoryError(f"{local_dir} is not a directory.")

        zipdir = os.path.join(local_dir, bohr_job_id)
        # `bohr job download` writes <jobid>/out.zip below local_dir.  Remove
        # anything left by an earlier failed attempt so a stale archive is
        # never mistaken for the new download.
        await asyncio.to_thread(shutil.rmtree, zipdir, ignore_errors=True)

        command = f"bohr job download -j {bohr_job_id}"
        try:
            output = await run_shell_command(command, local_dir, 300)
            if output["return_code"] != 0:
                logger.error(f"Fail to download the job {bohr_job_id} result on bohr. {output['stderr'].strip()}")
                return False 
            if "Downloading outfile to" in output["stdout"]:
                logger.info(f"Successfully download the job {bohr_job_id} from bohr.")
        except Exception as e:
            logger.exception(f"Fail to download the job {bohr_job_id} result on bohr. {e}")
            return False 

        out_zip = os.path.join(zipdir, "out.zip")
        if not os.path.isfile(out_zip) or os.path.getsize(out_zip) == 0:
            logger.error(f"Download of job {bohr_job_id} produced no out.zip in {zipdir}.")
            return False

        try:
            # Reject a truncated/incomplete zip before extracting anything.
            output = await run_shell_command("unzip -t out.zip", zipdir, 120)
            if output["return_code"] != 0:
                logger.error(f"Downloaded out.zip for job {bohr_job_id} is corrupt: "
                             f"{output['stderr'].strip()}")
                return False

            output = await run_shell_command(
                "unzip -o out.zip -x \"*.md\" \"*.json\" \"*.sh\" \"*.py\"",
                zipdir, 120)
            if output["return_code"] != 0:
                logger.error(f"Fail to unzip out.zip in {zipdir}. {output['stderr'].strip()}")
                return False 

            await run_shell_command("rm -f out.zip", zipdir, 120)
            # Merge the downloaded files into local_dir.  `mv <dir>/* .` nests a
            # directory when local_dir already contains a non-empty directory
            # with the same name; rsync merges recursively instead.  Result
            # files overwrite older ones so a retry can replace data from a
            # failed transfer, while the user's inputs are left untouched.
            output = await run_shell_command(
                f"rsync -a --exclude=INCAR --exclude=POSCAR --exclude='POTCAR*' "
                f"--exclude=run.sh --exclude=job.json --exclude=job_description.json "
                f"--exclude='*.md' --exclude='*.sh' --exclude='*.py' "
                f"{shlex.quote(bohr_job_id)}/ . "
                f"&& rm -rf {shlex.quote(bohr_job_id)}",
                local_dir, 120)
            if output["return_code"] != 0:
                logger.error(f"Fail to merge the downloaded results into {local_dir}. "
                             f"{output['stderr'].strip()}")
                return False
            logger.info(f"Result files are unzip to {local_dir}")
        except Exception as e:
            logger.exception(f"Fail to unzip and clean the downloaded results in {zipdir}")
            return False

        return True 

    async def submit(self, setting:JobSetting)->str:
        if not os.path.isdir(setting.local_dir):
            raise NotADirectoryError(f"{setting.local_dir} is not a directory.")
        if not os.path.isfile(os.path.join(setting.local_dir, "run.sh")):
            raise FileNotFoundError(f"There should be a run.sh file in directory {setting.local_dir}. ")

        command = f'''bohr job submit \\
--job_name "{setting.job_name}" \\
--project_id "{self.project_id}" \\
--machine_type "{setting.resource_name}" \\
--image_address "{self.job_envs[setting.job_env_name].image_path}" \\
--dataset_path "{','.join(setting.access_remote_dirs) if setting.access_remote_dirs else ''}" \\
--command "bash run.sh " \\
--log_file "tmp.log" \\
--backward_files "*" \\
--input_directory "./"
'''
        try:
            output = await run_shell_command(command, setting.local_dir, 300)
            if output["return_code"] != 0 or (not 'succeed' in output["stdout"]):
                logger.warning(f"Job submission {setting.job_name} to bohr failed.")
                raise RuntimeError(f"Failed to submit job {setting.job_name} to bohr. \
                                   Return code: {output["return_code"]}. Stdout: {output["stdout"]}")
            jobid = output["stdout"].split('\n')[1]
            jobid = jobid.split()[1]
            logger.info(f"Job is submitted to bohr, with jobid {jobid}.")
            return jobid
        except Exception as e:
            logger.warning(f"Job submission {setting.job_name} to bohr failed.")
            raise RuntimeError(f"Failed to submit job {setting.job_name} to bohr. Error: {e}." )
        

    async def cleanup(self, remote_dir: str):
        pass 

    def disconnect(self):
        pass 