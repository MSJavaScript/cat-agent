from langchain.chat_models import BaseChatModel
from langchain.tools import tool, ToolRuntime
from langchain.messages import ToolMessage, HumanMessage, SystemMessage
from langgraph.types import Command 
from .models import SkillMetadata, Step, RawPlan, RawStep, Delete
from typing import List, Dict, Literal, Set

def init_planner_tool(model:BaseChatModel):
    @tool 
    def planner(runtime:ToolRuntime,
                related_skill_names:Set[str], 
                task_content:str,
                plan_steps_guidance: List[str],
                task_context:str = ""):
        '''
        Make a raw plan for solving a question/job. A guidance is required to guide the planner. For simple tasks, divide them into 1 to 5 steps, while complex tasks are divided into 5-8 steps, no more than 10 steps. If a certain step is too complex, or multi steps should be merged to limit the total number of steps, add "the step need further planning" in the corresponding entry of `plan_steps_guidance` to guide the planner set this step to be `need_further_planning = True`.
        Args:
            related_skill_names: Names of skills related to this question.
            task_content: Question/job content.
            plan_steps_guidance: Quidances for the plan, planner will generate the same number of steps. Add a prompt "the step need further planning" to set a RawStep `need_further_planning = True`. 
            task_context: The context of the question. Such as the file path and calculation setting specified by the user, the error message and possible correction of a step.
        '''

        system_prompt = "You are an agent responsible for planning the execution flow of scientific computing works. Please follow the guidance to make the plan. At last, make sure to generate a structued output `RawPlan` object, instead of plain text."

        loaded_skills:Dict[str, SkillMetadata] = runtime.state["loaded_skills"]
        skill_doc = ""
        for skill_name in related_skill_names:
            skill_doc += f"Skill: {skill_name}\n"
            file_path = loaded_skills[skill_name]["path"]
            with open(file_path, 'r') as f:
                lines = f.readlines() 
                skill_doc += ''.join(lines)

            script_content = ""
            for script in loaded_skills[skill_name]["scripts"]:
                script_content += f"|{script['path']}|{script['purpose']}|\n"
            if len(script_content) > 0:
                script_content = '\nAvailable scripts:\n\n|Name|Purpose|\n|---|---|\n'+script_content

            reference_content = ""
            for reference in loaded_skills[skill_name]["references"]:
                #不要把提交任务相关的内容交给planner，它不需要知道
                if "job_submission" in reference['path']:
                    continue
                reference_content += f"|{reference['path']}|{reference['purpose']}|\n"
            if len(reference_content) > 0:
                reference_content = '\nAvailable reference files:\n\n|Name|Purpose|\n|---|---|\n'+reference_content

            skill_doc += script_content + reference_content


        guidance = "**The plan step guidances you should follow**:\n"
        for i, item in enumerate(plan_steps_guidance):
            guidance += f"{i}. {item}\n"


        prompt = f"Please make plan for the question/job below:\n {task_content}"
        if len(task_content) > 0:
            prompt += f"Additional informations:\n {task_context}"

        prompt += f"{skill_doc}\n\n{guidance}"


        try:
            response = model.with_structured_output(RawPlan).invoke([
                SystemMessage(content=system_prompt), 
                HumanMessage(content=prompt)])
            response = RawPlan.model_validate(response)
        except Exception as e:
            return Command(update={
                "messages":[ ToolMessage(status="error",
                                         content=f"Error:{e}", 
                                         tool_call_id=runtime.tool_call_id,
                                         name="planner")]
            })
    
        # Key 的格式为rawplan-id
        raw_plan_id_list = []
        for key in runtime.state["raw_plan_dict"].keys():
            _, b = key.rsplit('-', 1)
            raw_plan_id_list.append(int(b))
        rawplan_id = f"rawplan-{max(raw_plan_id_list)+1 if raw_plan_id_list else 0}"

        returned_message = f"Raw plan abstract:\nExecution flow: {response.topology}\nSteps:\n"
        for i, raw_step in enumerate(response.steps):
            returned_message += f"Step {i}({raw_step.name}): {raw_step.content}\n"
        returned_message += (f"The raw plan has been written to the agent state under the id "
                             f"`{rawplan_id}`. Call `register_raw_plan` with "
                             f"`raw_plan_id=\"{rawplan_id}\"` to register it.")

        return Command(update={
            "raw_plan_dict": {rawplan_id: response },
            "messages":[ ToolMessage(content=returned_message, 
                                     tool_call_id=runtime.tool_call_id,
                                     name="planner")]
        })


    @tool 
    def get_raw_plan_content(runtime:ToolRuntime, detailed:bool = False):
        '''
        Return all the unregistered `RawPlan` objects.
        Args:
            detailed: If the detailed content is returned. If `False`, fields `"related_skills", "solution",  "required_artifacts", "output_artifacts"` in `RawStep` will be omitted.
        Return the content of unregistered raw plan.
        Returns:
            A dict of `RawPlan` objects. Key is the id of `RawPlan`, value is the `RawPlan` object, each has the following keys:
            - steps(list): A list of `RawStep` object. Each one has the following fields:
                - need_further_planning(bool): If the step need further planning.
                - name(str): Step name.
                - content(str): Step content.
                - related_skills(set): Names of skills related to this step.
                - solution(str): The solution of the step.
                - required_artifacts(dict): Files required for this step.
                - output_artifacts(dict): Files expected to be generated in this step.
            - topology(list): A list of int tuples representing the execution topology of the workflow. Each int is the index of the step. For example, `[(0,1),(1,2)]` represents the sequential execution of '0→1→2'. `[(0,1),(0,2)]` represents parallel branches '0→1' and '0→2'. Optional.
        '''
        omit_keys = ["related_skills", "solution",  "required_artifacts", "output_artifacts"]
        raw_plan_dict:Dict[str, RawPlan] = runtime.state.get("raw_plan_dict") or dict()
        result = dict()
        for key in raw_plan_dict.keys():
            raw_plan = raw_plan_dict[key].model_dump(mode='python')

            info = {key:raw_plan[key] for key in raw_plan.keys() if key != "steps"}
            if detailed:
                info["steps"] = raw_plan["steps"]
            else:
                simple_steps = []
                for step in raw_plan["steps"]:
                    simple_steps.append({key:step[key] for key in step.keys() if not key in omit_keys})
                info["steps"] = simple_steps
            
            result[key] = info 
        return result 


    @tool 
    def register_raw_plan(runtime:ToolRuntime,
                          raw_plan_id: str,
                          action:Literal["append", "replace"] = "append",
                          depend_on_external:List[List[str]] = None,
                          raw_step_indices_group:List[List[int]|int] = None,
                          step_ids_group:List[List[str]|str] = None): 
        '''
        Register a raw plan. 
        
        For `"append"` mode, use `depend_on_external` to set the external dependence of each raw step. For example, if raw step 0 depends `"plan-1:step-1:xxx", "plan-2:step-2:yyy"`, then the first list of `depend_on_external` should be `["plan-1:step-1:xxx", "plan-2:step-2:yyy"]`. If a step depend the whole plan-x, just set `["plan-x"]`
         
        For `"replace"`, use `raw_step_indices_group` and `step_ids_group` to set the correspondence between raw steps and the steps to be replaced. The correspondence can ONLY be one-to-one (most recommended), many-to-one or one-to-many. Groups must be disjointed. For example, `raw_step_indices_group=[0,1]; step_ids_group=['a', 'b']` gives 0→'a', 1→'b'. `raw_step_indices_group=[[0,1],2]; step_ids_group=['a', ['b', 'c']]` gives [0,1]→'a', 2→['b', 'c']. 
        Args:
            raw_plan_id: Id of raw plan, the format is "rawplan-{id}", "{id}" is a int.
            action: registration action.
                - "append": Append raw steps to the agent state. 
                - "replace": Replace the steps in the agent state.
            depend_on_external: Valid in `"append"` mode. Set the external dependence of each raw step. Each list corresponds to a raw step, each str is the `step_id` of a external step which is not contained in the current raw plan.
            raw_step_indices_group: Valid in `"replace"` mode. The group of raw step indices.
            step_ids_group: Valid in `"replace"` mode. The group of step ids. 
        '''

        def _helper(step_id:str, 
                    raw_step:RawStep, 
                    depend_on:Set[str], 
                    required_by:Set[str]):
            
            step = Step(step_id=step_id,
                        status='pending',
                        workdir='',

                        need_further_planning=raw_step.need_further_planning,
                        content=raw_step.content or "",
                        # RawStep uses None for optional fields, while Step
                        # expects concrete containers/strings.
                        related_skills=raw_step.related_skills or set(),
                        solution=raw_step.solution or "",
                        required_artifacts=raw_step.required_artifacts or {},
                        output_artifacts=raw_step.output_artifacts or {},
  
                        depend_on=depend_on,
                        required_by=required_by)
            return step 
         
        raw_plan_dict:Dict[str, RawPlan] = runtime.state["raw_plan_dict"]
        raw_plan_keys = list(raw_plan_dict.keys())

        if not raw_plan_id in raw_plan_dict:
            # Return a ToolMessage instead of raising: the orchestrator can then
            # recover by retrying with the correct id.  Raising here aborts the
            # whole graph because tool errors are not handled by default.
            available = ", ".join(sorted(raw_plan_dict.keys())) or "(none)"
            return Command(update={
                "messages": [ToolMessage(
                    status="error",
                    content=(f"Error: raw plan `{raw_plan_id}` does not exist. "
                             f"Available raw plan ids: {available}. "
                             f"Call `get_raw_plan_content` to inspect them."),
                    tool_call_id=runtime.tool_call_id,
                    name="register_raw_plan")]
            })

        raw_plan = raw_plan_dict[raw_plan_id]
        n_steps = len(raw_plan.steps)

        steps_keys = list((runtime.state.get("steps") or {}).keys())
        plan_id_set = set()
        for key in steps_keys:
            a, _, _ = key.split(':') # plan-数字:step-数字:名字
            _, num = a.split('-') # plan-数字
            plan_id_set.add(int(num))
        plan_id = max(plan_id_set) + 1 if len(plan_id_set) > 0 else 0

        depend_on = {i:[] for i in range(n_steps)}
        required_by = {i:[] for i in range(n_steps)}
        topology = raw_plan.topology
        if topology:
            for i,j in topology: # i → j
                required_by[i].append(j)
                depend_on[j].append(i)

        if action == "append":
            update = dict()

            raw_plan_keys.remove(raw_plan_id)

            message = f"Raw plan {raw_plan_id} is registered and removed. {len(raw_plan_keys)} raw plan(s) lefted in agent state: {raw_plan_keys}.\n"

            message += "Workflow topology of added steps:\n\n|step_id|name|depend_on|required_by|\n|---|---|---|---|\n"
            for i in range(n_steps):
                step_id = f"plan-{plan_id}:step-{i}:{raw_plan.steps[i].name}"

                _depend_on = [f"plan-{plan_id}:step-{k}:{raw_plan.steps[k].name}" for k in depend_on[i]] 

                if depend_on_external and depend_on_external[i]:
                    _depend_on = _depend_on + depend_on_external[i]

                _required_by = [f"plan-{plan_id}:step-{k}:{raw_plan.steps[k].name}" for k in required_by[i]]

                message += f"|{step_id}|{raw_plan.steps[i].name}|{',\n'.join(_depend_on)}|{',\n'.join(_required_by)}|\n"
                update[step_id] = _helper(step_id, 
                                          raw_plan.steps[i], 
                                          set(_depend_on), 
                                          set(_required_by))
                
            return Command(update={
                "messages": [ToolMessage(content=message, 
                                         tool_call_id=runtime.tool_call_id,
                                         name="register_raw_plan")],
                "steps": update,
                "raw_plan_dict": { raw_plan_id: None }
            })

        else:
            # 替换模式下的更新策略。
            # 把所有的新增的Step加入到更新，更改依赖的也加入更新。
            # 最后去把它们依赖中已经删掉的Step的id去掉
            
            if (not raw_step_indices_group) or (not step_ids_group):
                raise ValueError("You should set `raw_step_indices_group` and `step_ids_group` in \"replace\" mode. ")
            if len(raw_step_indices_group) != len(step_ids_group):
                raise ValueError("`raw_step_indices_group` and `step_ids_group` must have the same length.")

            raw_plan_keys.remove(raw_plan_id)

            message = f"Raw plan {raw_plan_id} is registered and removed. {len(raw_plan_keys)} raw plan(s) lefted in agent state:{raw_plan_keys}\nStep object replacement ( old → new ):\n"

            step_dict:Dict[str, Step] = runtime.state.get("steps") or dict()
            update = dict()
            for rg, sg in zip(raw_step_indices_group, step_ids_group):
                rg = [rg] if isinstance(rg, int) else rg
                sg = [sg] if isinstance(sg, str) else sg 

                n_rg = len(rg)
                n_sg = len(sg)

                if n_rg > 1 and n_sg > 1:
                    raise RuntimeError(f"The correspondence between raw steps indices and step ids must be one-to-one, one-to-many or many-to-one, got {rg}→{sg}.")
                
                if n_rg == 1: # raw step只有一个，对应要替换的Step可能有多个
                    
                    # 收集所有对应的step对象中的依赖关系，它的依赖关系要全部赋予给这个raw step
                    _depend_on_set = set()
                    _required_by_set = set()
                    for step_id in sg:
                        _depend_on_set.update(step_dict[step_id].depend_on)
                        _required_by_set.update(step_dict[step_id].required_by)
                    # 后面这些依赖需要修改，去掉所有被删除的step
                    _depend_on   = _depend_on_set.union(
                        [f"plan-{plan_id}:step-{k}:{raw_plan.steps[k].name}" for k in depend_on[rg[0]]]
                    )
                    _required_by = _required_by_set.union(
                        [f"plan-{plan_id}:step-{k}:{raw_plan.steps[k].name}" for k in required_by[rg[0]]]
                    )

                    step_id = f"plan-{plan_id}:step-{rg[0]}:{raw_plan.steps[rg[0]].name}"
                    update[step_id] = _helper(step_id, raw_plan.steps[rg[0]],  _depend_on, _required_by)

                    if n_sg == 1:
                        message += f"{sg[0]} → {step_id}\n"
                    else:
                        message += f"{sg} → {step_id}\n"

                    # _required_by_set 中保存的是那些依赖于被删除节点的节点。它们需要更新depend_on
                    # 所以我们也要把它加入更新，后面更新它的depend_on
                    for item_id in _required_by_set:
                        update[item_id] = step_dict[item_id].model_copy()
                        update[item_id].depend_on.add(step_id)
                elif n_sg == 1:  # 多个raw_step对应一个step
                    new_id_list = []
                    for raw_step_id in rg:
                        step_id = f"plan-{plan_id}:step-{raw_step_id}:{raw_plan.steps[raw_step_id].name}"
                        new_id_list.append(step_id)

                        _depend_on = {f"plan-{plan_id}:step-{k}:{raw_plan.steps[k].name}" 
                                      for k in depend_on[raw_step_id]}.union(step_dict[sg[0]].depend_on)
                        
                        _required_by = {f"plan-{plan_id}:step-{k}:{raw_plan.steps[k].name}" 
                                        for k in required_by[raw_step_id]}.union(step_dict[sg[0]].required_by)

                        update[step_id] = _helper(step_id, raw_plan.steps[raw_step_id],  _depend_on, _required_by)

                    message += f"{sg[0]} → {new_id_list}\n"

                    for item_id in step_dict[sg[0]].required_by:
                        update[item_id] = step_dict[item_id].model_copy()
                        update[item_id].depend_on.update([ 
                            f"plan-{plan_id}:step-{raw_step_id}:{raw_plan.steps[raw_step_id].name}" for raw_step_id in rg
                            ])

            # 现在更新所有update的depend on，去掉那些将要删除的节点
            
            step_ids_will_be_deleted = set()
            for item in step_ids_group:
                if isinstance(item, list):
                    step_ids_will_be_deleted.update(item)
                else:
                    step_ids_will_be_deleted.add(item)
            
            new_add_message = "Below are the newly added steps:\n\n|step_id|name|depend_on|required_by|\n|---|---|---|---|\n"
            modified_message = "Below are the steps whose dependence changed:\n\n|step_id|name|depend_on|required_by|\n|---|---|---|---|\n"
            plan_key = f"plan-{plan_id}"
            for key in update.keys():
                update[key].depend_on.difference_update(step_ids_will_be_deleted)
                row_msg = f"|{key}|{update[key].step_id}|{', '.join(update[key].depend_on)}|{', '.join(update[key].required_by)}|\n"
                if plan_key in key:
                    new_add_message += row_msg
                else:
                    modified_message += row_msg
            message += f"\n {new_add_message}\n{modified_message}\n"
            message += f"Below are steps that have been deleted:\n {step_ids_will_be_deleted}"

            for key in step_ids_will_be_deleted:
                update[key] = None 

            return Command(
                update={
                    "messages": [ToolMessage(content=message, 
                                             tool_call_id=runtime.tool_call_id,
                                             name="register_raw_plan")],
                    "steps": update,
                    "raw_plan_dict": { raw_plan_id:None }
                }
            )

    return [planner, get_raw_plan_content, register_raw_plan]