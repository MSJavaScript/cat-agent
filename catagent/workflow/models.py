from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from typing import TypedDict, Annotated, List, Dict, Literal, Optional, Set, Tuple, Any 
from pydantic import BaseModel, Field
from copy import deepcopy
from typing import NamedTuple

class RawStep(BaseModel):
    '''
    Metadata of a step. 
    '''
    need_further_planning: bool = Field(description="Whether the current step need further planning. If the granularity of the current step is sufficiently fine and the solution is clear this should be False. ", default=False)

    name:str = Field(description="A descriptive name of the step, only letters, numbers, hyphens, and underscores are allowed.")
    
    content: str = Field(description="Content of the current step.")

    related_skills: Optional[Set[str]] = Field(description="Names of skills related to this step.", default=None)

    solution: Optional[str] = Field(description="Solution for the current step. This is usually included in the skill docs or user's prompt, such as the usage of tool functions, scripts and softwares, example code snippets and shell commands. For a step need further planning, this field can be omitted.", default=None)

    required_artifacts: Optional[Dict[str, str]] = Field(description="Artifacts that are required for this step, usually some input files or the results from previous steps. Key is filename or wildcard, value is the purpose of the file. ", default=None)
    
    output_artifacts: Optional[Dict[str, str]] = Field(description="Artifacts that will be generated in this step. Key is filename or wildcard, value is the purpose of the file.", default=None)

class RawPlan(BaseModel):
    '''
    Plan for a task.
    '''
    steps: List[RawStep] = Field(description="List of plan steps.", default_factory=list)

    topology: Optional[List[Tuple[int, int]]] = Field(description="Workflow execution topology, a directed acyclic graph. Step is represented by the index, each tuple represents an edge in the graph. For example, `[(0,1),(1,2)]` represents the sequential execution of '0→1→2'. `[(0,1),(0,2)]` represents parallel branches '0→1' and '0→2'. If the plan has only one step, this should be None.", default=None)

class Step(BaseModel):
    step_id: str # 格式 plan-数字:step-数字:名字
    status: Literal["pending", "done", "failed", "running", "cancelled"] = "pending"
    preparation_stage: Literal["done", "ongoing", "notyet"] = "notyet"
    workdir: str = ""
    
 
    need_further_planning: bool = False 
    content: str = ""
    related_skills: Set[str] = set() 
    solution: str = "" 
    required_artifacts: Dict[str, str] = dict() 
    output_artifacts: Dict[str, str] = dict() 

    depend_on:   Set[str] = set()
    required_by: Set[str] = set()

class StepResult(BaseModel):
    status: Literal["success", "error"] = "success"
    processed: bool = False # 是否已经处理了该结果
    step_id: str  # 格式 plan-数字:step-数字:名字
    summary: str  # 该步骤的总结。输出了什么，如果有错误是什么错误
    artifacts: Dict[str, str] # 输出的关键文件
    
# Skill相关的数据结构
class SkillArtifact(TypedDict):
    path:str
    purpose:str

class SkillMetadata(TypedDict):
    name: str #名称
    path: str #绝对路径
    summary: str #总结
    references: Optional[List[SkillArtifact]]
    scripts: Optional[List[SkillArtifact]]

# 用于Agent的结构化输出，需要字段描述
class SkillSummaryOutput(BaseModel):
    name: str = Field(description="A descriptive name of skill, only letters, numbers, hyphens, and underscores are allowed. Directly use the name in metadata if exists." )
    summary: str = Field(description="Content summary of the skill, no more than 5 sentences.", default="")
    reference_purposes: Optional[Dict[str, str]] = Field(description="The purposes of reference files of the skill. Key is the file basename, value is the purpose of this file.", default=None)
    script_purposes: Optional[Dict[str, str]] = Field(description="The purposes of scripts of the skill. Key is the file basename, value is the purpose of this script.", default=None)


class Delete(NamedTuple):
    indices: List[int]

def list_reducer(current:list, update) -> list:
    if current is None:
        current = []
    if update is None:
        return current
    if isinstance(update, Delete):
        n = len(current)
        return [current[i] for i in range(n) if not i in update.indices]
    if isinstance(update, list):  
        return current + update
    return current + [update] 

def dict_reducer(current:dict, update:dict) -> dict:
    current = current or dict()
    if update is None:
        return current
    result = current.copy()
    for k, v in update.items():
        if v is None:
            result.pop(k, None)
        elif isinstance(v, dict) and hasattr(result.get(k), "model_copy"):
            # A plain dict is a partial patch of a pydantic state object
            # (currently Step).  This keeps concurrent edits of different
            # fields from clobbering each other: several tools may run in
            # one ToolNode batch, each seeing the same pre-batch state.
            result[k] = result[k].model_copy(update=v)
        else:
            result[k] = v 
    return result


class WorkflowState(TypedDict, total=False):
    # Initial user input arrives here; later nodes only read it when they need
    # to start a fresh orchestrator turn.
    messages: Annotated[List[AnyMessage], add_messages]

    orchestrator_is_working: bool
    finalization: bool
    orchestrator_requests: Annotated[List[Dict[str, Any]], list_reducer]

    steps: Annotated[Dict[str, Step], dict_reducer] # 唯一与orchestrator共享的字段
    step_results: Annotated[Dict[str, StepResult], dict_reducer]
    # Skills loaded by the orchestrator. StepNode receives a copy so it can
    # call get_skill_content() without loading the skill again.
    loaded_skills: Annotated[Dict[str, SkillMetadata], dict_reducer]



 