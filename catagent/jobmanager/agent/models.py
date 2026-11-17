from dataclasses import dataclass
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


@dataclass
class JobManagerAgentContext:
    provider_id: str = None 
    job_env_name: str = None 
    phase: str = None 
    local_dir: str = ""


class PreprocessingOutput(BaseModel):
    '''
    Structured output for preprocessing phase.
    '''
    batch_job: bool    = Field(description="If this is a batch job.", default=False)
    n_jobs: int        = Field(description="The total number of jobs.", default=1)
    n_groups: Optional[int] = Field(description="The number of groups, only valid for batch job.", default=None)
    group_dirs: Optional[List[str]] = Field(description="The dir names or dir wildcards for each job group. A group has ONLY one `run.sh` and `job_description.json` file in the group dir.", default=None)

class CheckingOutput(BaseModel):
    '''
    Structured output for checking phase.
    '''
    job_result_status: Literal["Success", "InputError", "EnvironmentError"] = Field(description="The status of the downloaded job results.", default=None)
    message:str = Field(description="The brief summary or error messages of job results, within 3 to 5 sentences.", default=None)

class JobManagerAgentOutput(BaseModel):
    '''
    Final structured output.
    '''
    phase: Literal["preprocessing", "checking"] = Field(description="Phase of the current task. Preprocessing of remote job submissions or checking the results of downloaded jobs.")
    preprocessing_output: Optional[PreprocessingOutput] = Field(description="Output for preprocessing phase.", default=None)
    checking_output: Optional[CheckingOutput] = Field(description="Output for checking phase.", default=None)