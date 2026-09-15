
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Literal
from pydantic import BaseModel

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

class JobSetting(BaseModel):
    job_name: str
    local_dir: str

    # 只在slurm平台上需要
    remote_dir: Optional[str] = None
    # 返回文件的形式
    result_patterns: Optional[List[str]] = None
    # 可访问的远程路径，在波尔平台上称为“数据集路径”
    access_remote_dirs: Optional[List[str]] = None
    # 下面两个选项要由科学计算agent指定。目的是为了保证计算环境的一致性。
    provider_name: str #提供商的名称。在配置json文件中指定
    job_env_name:  str
    # 机器类型或分区类型由任务agent指定。科学计算agent不知道这些信息。
    resource_name: Optional[str] = None


class JobType(str, Enum):
    VASP = "vasp"
    ABACUS = "abacus"
    FENICS = "fenics"
    LAMMPS = "lammps"

class JobStatus(str, Enum):
    SUBMITTED = "submitted"   
    RUNNING   = "running"  
    COMPLETED = "completed"    
    FAILED    = "failed"
    CANCELLED = "cancelled"
    UNKNOWN   = "unknown"  # 查询状态的函数成功，但没有返回这个作业的信息
    NONE      = "none"     # 调用查询状态的函数失败返回None

class SubJobRecord(BaseModel):

    job_uuid:    str # 所属作业的uuid
    group_index: int # 组的序号，从0开始。单个作业只有group_index=0一个子作业
    local_dir:   str # 该组的本地路径。单个作业时与作业的local_dir相同
    remote_id:   Optional[str] = None # 该组提交到远端时被赋予的id
    remote_dir:  Optional[str] = None # 该组在远端的路径。对于波尔平台为空
    status:      JobStatus
    created_at: str = ""
    updated_at: str = ""
    error: Optional[str] = None

class JobRecord(BaseModel):
    job_uuid:    str # Agent 分配的uuid
    description: str # 作业的描述
    provider_id: str 
    job_env_name:str # 作业运行的环境名称
    local_dir:   str # 本地路径。该路径是提交任务时文件的输入目录，同时也是下载目录

    result_patterns: List[str] = []
    status:      JobStatus
    created_at: str = ""
    updated_at: str = ""
    callback_pending: bool
    attempt: int = 0 #尝试次数
    # 如果任务出错记录错误原因，由一个agent提取。目前没有agent参与设为空字符串。
    error: Optional[str] = None 
    sub_jobs:List[SubJobRecord] = []
    
class JobRequest(BaseModel):
    job_uuid: str     # Agent 分配的uuid
    description: str  # 工作的描述，上层agent需要有一个提示词模板
    local_dir: str    # 本地路径。该路径是提交任务时文件的输入目录，同时也是下载目录
    provider_id: str  # 提供商的id
    job_env_name: str # 提供商的环境名称
    result_patterns: List[str] #回收文件的模式

class JobResult(BaseModel):
    job_uuid: str 
    local_dir: str
    status: Literal["error", "success"]
    error_msg: Optional[str]
    