from langchain.chat_models import BaseChatModel
from langchain.agents import create_agent, AgentState 
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, ModelRequest, ModelResponse
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command, Checkpointer, interrupt
from langchain.messages import ToolMessage
from langchain.tools import tool, ToolRuntime

from .models import dict_reducer, Step, SkillMetadata, RawPlan
from catagent.stepnode.models import InterruptState

import httpx
from typing import List, Annotated, Dict, Literal, Callable, Any 
from tools.other.util_tools import execute_shell, edit_file, read_file
from .skill_tool import init_skill_tools
from .planner_tool import init_planner_tool
from .job_tool import init_job_manager_tools
from .step_edit_tool import (get_steps_layout,
                            get_step_content,
                            set_step_status,
                            set_step_workdir,
                            set_step_preparation_stage,
                            set_step_dependence,
                            set_step_content,
                            create_step)


@tool 
def pause_to_ask(runtime:ToolRuntime, 
                 message:str, 
                 type: Literal["any", "confirmation", "choice-single", "choice-multiple"] = "any",
                 meta_data:Dict[str, Any] = None ):
    '''
    Emit an interrupt to pause the agent for obtaining authorization, confirmation, or supplementary information from the user. For example, pause to ask user for a python environment path or let them manually install a package, ask the user to set key calculation parameters, such as DFT functional, MD temperature, linear solver type, etc., if they didn't provide.
    Args:
        message: The message sent to user.
        type: Type of interrupt, can be
            - "confirmation": Ask the user to confirm something.
            - "choice-single": Ask the user to choice a single value from the given options. Options should be explained in `meta_data`, keys are the options, values are the meaning.
            - "choice-multiple": Ask the user to choice multiple values from the given options.
            - "any": For complex data, use "any". For example, ask the user to set a range or a list.
        meta_data: Extra metadata. Can be the choices requiring the user to select or data fields requiring the user to fill. Key is the choice or field name, value is the meaning.
    '''
 
    payload = {"type": type}
    if meta_data:
        payload.update(meta_data)
    interrupt_payload = InterruptState(kind="AgentInterrupt",
                                       message=message,
                                       tool_call_id=runtime.tool_call_id,
                                       tool_name="pause_to_ask",
                                       payload=payload)
    response = interrupt(interrupt_payload)
    return response 


@tool 
def set_finalization_flag(runtime:ToolRuntime, finalization:bool):
    '''
    Set `finalization` flag of the agent. ONLY for replanning phase, before call this tool, please call `get_current_phase` to confirm the agent working phase. 
    Args:
        finalization: set `True` to finalize this run.
    '''
    return Command(update={
        "finalization": finalization,
        "messages": [ToolMessage(content=f"Set `finalization = {finalization}`.",
                                 tool_call_id = runtime.tool_call_id,
                                 name="set_finalization_flag")]
        }) 

@tool 
def get_current_phase(runtime:ToolRuntime):
    '''
    Get the current working phase.
    '''
    phase = runtime.state.get("phase", "user")
    if phase == "user":
        return "User request (fresh planning) phase."
    elif phase == "replanning":
        return "Replanning phase. Further decompose a step into sub-steps, or finalize the whole run and make a new plan."
    elif phase == "preparation":
        return "Step preparation phase. Make workdir for this step (if not exist) and copy required files to the workdir."
    elif phase == "correction":
        return "Correction phase. Fix the problem in the failed step."
    elif phase == "logging":
        return "We are not in a specific phase. Just waiting for user's request or logging the working state if `state.md` is not updated."
    else:
        raise ValueError(f"Unknow phase {phase}.")

@wrap_model_call 
async def filter_tools(
    request: ModelRequest, 
    handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
    request_type = request.state.get("phase", "user")
    common_tools = {"execute_shell", "edit_file", "pause_to_ask", "read_file", 
                    "get_steps_layout", "get_step_content", "get_current_phase"}
    allowed_tools = dict()
    if request_type == "preparation":
        allowed_tools = common_tools | {
            "set_step_workdir", "set_step_preparation_stage",
            "get_skill_summaries", "get_skill_content",
            "load_skill_from_filesystem", "get_providers_info",
        }
    elif request_type == "logging":
        allowed_tools = common_tools
    if allowed_tools:
        tools = [t for t in request.tools if t.name in allowed_tools]
        request = request.override(tools=tools)
    return await handler(request)

 
@wrap_tool_call
async def handle_tool_error(request, handler):
    try:
        return await handler(request)
    except GraphBubbleUp:
        raise
    except Exception as e:
        return Command(update={
            "messages": [
                ToolMessage(status="error",
                            content=f"Tool error: {e}",
                            tool_call_id=request.tool_call["id"],
                            name=request.tool_call["name"])
            ]
        })


class OrchestratorAgentState(AgentState):
    finalization: bool = False 
    loaded_skills: Annotated[Dict[str, SkillMetadata], dict_reducer]
    steps: Annotated[Dict[str, Step], dict_reducer]
    raw_plan_dict: Annotated[Dict[str, RawPlan], dict_reducer] 
    phase: Literal["user", "preparation", "correction", "replanning", "logging"] = "user"
    interrupt_resume: Dict[str, Any]

def create_orchestrator_agent(model:BaseChatModel, 
                              system_skill_path:List[str], 
                              job_manager_url:str,
                              http_client:httpx.AsyncClient,
                              checkpointer:Checkpointer = None):

    system_prompt = ""
    for path in system_skill_path:
        with open(path, mode="r") as f:
            system_prompt += ''.join(f.readlines())

    tools = [execute_shell, edit_file, read_file, pause_to_ask, get_current_phase,
             set_finalization_flag,
             get_steps_layout, set_step_status, set_step_workdir,
             set_step_preparation_stage, set_step_dependence,
             set_step_content, get_step_content, create_step] 

    tools.append(init_job_manager_tools(job_manager_url, http_client))
    tools.extend(init_skill_tools(model=model))
    tools.extend(init_planner_tool(model=model))

    orchestrator_agent = create_agent(model=model,
                                      tools = tools,
                                      system_prompt=system_prompt,
                                      state_schema=OrchestratorAgentState,
                                      checkpointer=checkpointer,
                                      middleware=[handle_tool_error]) 
    # debug 阶段先不考虑任何中间件
    return orchestrator_agent
