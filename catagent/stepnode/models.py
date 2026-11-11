from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

from catagent.workflow.models import SkillMetadata
from pydantic import BaseModel, Field
from typing import List, Dict, Annotated, Literal, Any, Set, TypedDict
import operator

# 工具的命名空间，只是为了编程上方便。
class ToolNamespace(BaseModel):
    description:str 
    tool_names: List[str]

class RawStepResult(BaseModel):
    '''
    Metadata of step result.
    '''
    status: Literal["success", "error"] = Field(description="If this step success.", default="success")
    summary: str  = Field(description="Summary and error messages (if any) of this step.", default="")
    output_artifacts: Dict[str, str] = Field(description="Important output files of this step, key is filename or wildcard, value is the purpose of this file.", default_factory=dict)
    
def dict_reducer(current:dict, update:dict) -> dict:
    current = current or dict()
    if update is None:
        return current
    result = current.copy()
    for k, v in update.items():
        if v is None:
            result.pop(k, None)
        else:
            result[k] = v 
    return result


class StepNodeState(TypedDict):
    # loaded_skills 专供主agent 使用，键是skill名称，值是skill的元数据
    loaded_skills: Annotated[Dict[str, SkillMetadata], dict_reducer]

    messages: Annotated[List[AnyMessage], add_messages]

    accessible_tool_names: Annotated[Set[str], operator.or_]

    # 用于处理中断。key 是tool_call_id
    interrupts: Annotated[Dict[str, dict], dict_reducer]

    #可以跳过的函数，不需要询问
    skip_review_tool_names: Annotated[Set[str], operator.or_]

    step_result: RawStepResult

    workdir: str

# Interrupt Node 的私有state
class InterruptState(TypedDict):
    kind: str
    message: str 
    tool_call_id: str 
    tool_name: str
    payload: Dict[str, Any] = dict()


# 以下是关于中断的数据模型。这些数据模型只负责在组装payload 时有固定的格式
# 实际在agent 状态里存放的只是一个dict 
class Interruption(BaseModel):
    pending: bool = True 
    kind: str
    message: str 
    tool_call_id: str 
    tool_name: str

# 对于提交作业的中断，这里特意添加了job_uuid字段
# 目的是在外侧调用时如果发现是一个JobSubmission 中断，则立即调用jobmanager的
# api 根据job_uuid 去在db中填写对应的 interrupu_id和thread_id，以便于resume agent
class JobSubmission(Interruption):
    kind: Literal["JobSubmission"] = "JobSubmission"
    job_uuid: str 

class AskForArgs(Interruption):
    """One pending ask, surfaced through interrupt() when args need user input."""
    kind: Literal["AskForArgs"] = "AskForArgs"
    current_args: Dict[str, Any] = dict()
    args_schema: Dict[str, Any] = dict()
    missing: List[str] = list()

class AgentInterrupt(Interruption):
    kind: Literal["AgentInterrupt"] = "AgentInterrupt"
    type: Literal["any", "confirmation", "choice-single", "choice-multiple"] = "any"
    meta_data: dict = dict()