from abc import ABC, abstractmethod
from catagent.jobmanager.models import JobSetting, JobStatus
from typing import Dict, List, Set

class BaseProvider(ABC):
    type: str = "base"
    def __init__(self, name:str, processing_units:List[str]):
        self.processing_units = processing_units
        self.name = name

    @abstractmethod
    def get_info(self) -> dict:
        '''
        Return the basic informations of the provider.
        Returns: 
            A dict with 3 keys:
            - "name": Name of provider.
            - "processing_units": Available processing units of provider, can be cpu or gpu.
            - "job_envs": Available job environments and softwares.
            - "remote_files": Available remote files on the provider.
        '''
        ...


    @abstractmethod
    def get_remote_files_list(self) -> List[dict]:
        ...
    
    @abstractmethod
    def get_resources(self) -> list:
        '''
        Informations of computational resources of provider.
        '''
        ...

    @abstractmethod
    def get_job_env(self, job_env_name:str)->dict:
        ...


    @abstractmethod
    async def get_job_states(self, remote_ids:Set[str]) -> Dict[str, JobStatus]:
        '''
        Return the status of jobs on remote. 
        '''
        ...

    @abstractmethod
    async def cancel(self, remote_id: str) -> None:
        ...

    @abstractmethod
    async def download(self, remote_id: str, reomte_dir:str, local_dir:str) -> bool:
        ...
    
    @abstractmethod
    async def submit(self, setting:JobSetting) -> str:
        '''
        Submit a job to remote, using a specific job environment. 
        A job id assigned by bohr or slurm system is returned.
        '''
        ...

    @abstractmethod
    async def cleanup(self, remote_dir: str) -> None:
        return None

    @abstractmethod
    def disconnect(self):
        return None 

  