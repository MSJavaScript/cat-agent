from langchain.tools import tool, ToolRuntime
from langgraph.types import Command
from langchain.messages import ToolMessage
from .models import Step
from typing import List, Dict, Set, Literal
import os 


@tool 
def get_steps_layout(runtime:ToolRuntime):
    '''
    Get the layout of all the steps in the state, useful for getting a glance at the progress of the plans and the dependence of plan steps.
    Returns:
        A dict containing the status and dependence of every step.
    ''' 
    plans_dict = dict()
    for key in runtime.state["steps"].keys():
        plan_id, step_id, name = key.split(':')
        info = {"step_id": key, "name": name, 
                "status":runtime.state["steps"][key].status,
                "depend_on":runtime.state["steps"][key].depend_on  }
        if plan_id in plans_dict:
            plans_dict[plan_id][step_id] = info 
        else:
            plans_dict[plan_id] = dict()
            plans_dict[plan_id][step_id] = info
    return plans_dict


@tool
def set_step_status(runtime:ToolRuntime, 
                    step_id:str,
                    status:Literal["pending", "done", "failed", "cancelled", "running"]):
    '''
    Set the status of a step.
    Args:
        step_id: Unique id of the step.
        status: Step status. Set a step to "cancelled" to delete a step.
    '''
    steps = runtime.state["steps"]
    if not step_id in steps:
        raise KeyError(f"{step_id} does not exist.")

    update = {step_id: {"status": status}}
    return Command(update={
        "steps": update,
        "messages": [ToolMessage(content=f"Status of {step_id} is set to {status}.", 
                                 tool_call_id=runtime.tool_call_id,
                                 name="set_step_status")]
    })

@tool 
def set_step_workdir(runtime:ToolRuntime, step_id:str, workdir:str):
    '''
    Set the workdir of a step.
    Args:
        step_id: Unique id of the step.
        workdir: Absolute path to workdir (should exist).
    ''' 
    if not os.path.isdir(workdir):
        raise NotADirectoryError(f"{workdir} is not a directory.")
    steps = runtime.state["steps"]
    if not step_id in steps:
        raise KeyError(f"{step_id} does not exist.")

    update = {step_id: {"workdir": workdir}}
    return Command(update={
        "steps": update,
        "messages": [ToolMessage(content=f"Workdir of {step_id} is set to {workdir}.", 
                                 tool_call_id=runtime.tool_call_id,
                                 name="set_step_workdir")]
    })


@tool 
def set_step_preparation_stage(runtime:ToolRuntime, 
                               step_id:str, 
                               preparation_stage:Literal["done", "ongoing", "notyet"] = "notyet"): 
    '''
    Set the `preparation_stage` of a step. ONLY for preparation phase, before call this tool, please call `get_current_phase` to confirm the agent working phase. For `preparation_stage = 'notyet'` or `'ongoing'`, `status` must be `'pending'`. 
    Args:
        step_id: Unique id of the step.
        preparation_stage: Can only be "done", "ongoing" or "notyet".
    '''
    steps = runtime.state["steps"]
    if not step_id in steps:
        raise KeyError(f"{step_id} does not exist.")

    update = {step_id: {"preparation_stage": preparation_stage}}
    return Command(update={
        "steps": update,
        "messages": [ToolMessage(content=f"`preparation_stage` of {step_id} is set to {preparation_stage}.", 
                                 tool_call_id=runtime.tool_call_id,
                                 name="set_step_preparation_stage")]
    })

@tool 
def set_step_dependence(runtime:ToolRuntime, 
                        step_id:str, 
                        depend_on:Set[str]=None, 
                        required_by:Set[str]=None):
    '''
    Set the workflow dependence of a step.
    Args:
        step_id: Unique id of the step.
        depend_on: The ids of steps which the current step depend on. Can be a standard step_id or "plan-{id}" which means the step depend on the whole plan "plan-{id}". Use `None` to keep the old values.
        required_by: The ids of steps which require the current step. Can be a standard step_id or "plan-{id}". Use `None` to keep the old values.
    ''' 
    steps = runtime.state["steps"]
    if not step_id in steps:
        raise KeyError(f"{step_id} does not exist.")

    patch = dict()
    message = ""
    if not depend_on is None:
        patch["depend_on"] = depend_on
        message += f"`depend_on` of {step_id} is set to {depend_on}.\n"
    else:
        message += f"`depend_on` of {step_id} is unchanged.\n"
    if not required_by is None:
        patch["required_by"] = required_by
        message += f"`required_by` of {step_id} is set to {required_by}.\n"
    else:
        message += f"`required_by` of {step_id} is unchanged.\n"
        
    return Command(update={
        "steps": {step_id: patch},
        "messages": [ToolMessage(content=message,  
                                 tool_call_id=runtime.tool_call_id,
                                 name="set_step_dependence")]
    })


@tool 
def set_step_content(runtime:ToolRuntime, 
                    step_id:str,
                    need_further_planning:bool = None,
                    content:str = None,
                    related_skills:Set[str] = None,
                    solution:str = None,
                    required_artifacts:Dict[str, str] = None,
                    output_artifacts:Dict[str, str] = None):
    '''
    Set the content of a step. Use `None` for an argument to keep the old value.
    Args:
        step_id: Unique id of the step.
        need_further_planning: If the step need further planning.
        content: Content of step.
        related_skills: Related skill doc names.
        solution: Solution of the step.
        required_artifacts: Required files of the current step, key is file name or wildcard, value is the purpose of the file.
        output_artifacts: Files that will be generated in current step.
    '''
    steps = runtime.state["steps"]
    if not step_id in steps:
        raise KeyError(f"{step_id} does not exist.")

    keys_and_vals = {
        "need_further_planning": need_further_planning,
        "content": content,
        "related_skills": related_skills,
        "solution": solution,
        "required_artifacts": required_artifacts,
        "output_artifacts": output_artifacts
    }

    patch = dict()
    message = ""
    for key in keys_and_vals.keys():
        if not keys_and_vals[key] is None:
            patch[key] = keys_and_vals[key]
            message += f"`{key}` of {step_id} is set to {keys_and_vals[key]}.\n"
        else:
            message += f"`{key}` of {step_id} is unchanged.\n"
    
    return Command(update={
        "steps": {step_id: patch},
        "messages": [ToolMessage(content=message,  
                                 tool_call_id=runtime.tool_call_id,
                                 name="set_step_content")]
    })



@tool 
def create_step(runtime:ToolRuntime, 
                step_name:str, 
                content: str,
                plan_index:int, 
                solution: str = "",
                step_index:int = 0):
    '''
    Directly add a registered step in agent state with step id "plan-{plan_index}:step-{step_index}:{step_name}".
    Args:
        step_name: Name of the step. Only letters, numbers, hyphens, and underscores are allowed.
        content: Job content of the step.
        plan_index: Index of the plan.
        solution: Solution of this step.
        step_index: Index of the step in the plan.
    ''' 
    step_id = f"plan-{plan_index}:step-{step_index}:{step_name}" 
    steps = runtime.state["steps"]
    if step_id in steps:
        raise ValueError(f"Step with id {step_id} already exist.")

    update_step = {
        step_id: Step(step_id=step_id,
                      status='pending',
                      preparation_stage='notyet',
                      workdir='',
                      need_further_planning=False,
                      content=content,
                      solution=solution)
    }
    return Command(
        update={
            "steps":update_step,
            "messages": [ToolMessage(content=f"Step {step_id} is added to agent state.", 
                                     tool_call_id=runtime.tool_call_id,
                                     name="create_step" )]
        }
    )


@tool 
def get_step_content(runtime:ToolRuntime, 
                     step_id_list:List[str], 
                     return_content: bool = False,
                     return_solution: bool = False,
                     return_artifacts: bool = False,
                     return_dependence:bool = False):
    '''
    Get the :class:`Step` objects in agent state. Use the `return_*` flags to control which fields will be returned. 
    Args:
        step_id_list: The list of step_ids.
        return_content: If `content` field is returned.
        return_solution: If `solution` field is returned.
        return_artifacts: If `required_artifacts` and `output_artifacts` fields are returned.
        return_dependence: If `depend_on` and `required_by` fields are returned.
    Returns:
        A list of Step object. Each object has the following fields.
        - step_id(str): Unique id of the step.
        - status(str): Step status.
        - workdir(str): Workdir of the step.
        - preparation_stage(bool): Step preparation stage.
        - need_further_planning(bool): If the step need further planning.
        - content(str): Step content.
        - related_skills(set): Names of skills related to this step.
        - solution(str): The solution of the step.
        - required_artifacts(dict): Files required for this step. 
        - output_artifacts(dict): Files expected to be generated in this step. 
        - depend_on(set): The ids of steps which the current step depend on.
        - required_by(set): The ids of steps which require the current step.
    '''
    result = dict()
    steps = runtime.state["steps"]
    for step_id in step_id_list:
        if not step_id in steps:
            raise KeyError(f"{step_id} does not exist.")
        result[step_id] = steps[step_id].model_dump(mode="python")

        if not return_content:
            result[step_id].pop("content", None)
        if not return_solution:
            result[step_id].pop("solution", None)
        if not return_artifacts:
            result[step_id].pop("required_artifacts", None)
            result[step_id].pop("output_artifacts", None)
        if not return_dependence:
            result[step_id].pop("depend_on", None)
            result[step_id].pop("required_by", None)

    return result