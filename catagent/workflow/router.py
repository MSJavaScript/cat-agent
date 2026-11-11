
from .models import WorkflowState, Delete, Step
from langchain.messages import HumanMessage
from langgraph.types import Command, Send
from langgraph.graph import END 
from typing import Dict
import asyncio, os 

# 这里变成串行请求是因为任务步骤之间都是相互关联的。Orchestrator需要知道全局的状态
# NOTE: this node is wrapped by `orchestrator_request_queue_node` in agent.py,
# which guarantees that no request is dispatched while another orchestrator
# call is running.  Never busy-wait on a graph-state flag here: the state seen
# by a running node is an immutable snapshot and would never change.
async def orchestrator_request_queue(state:WorkflowState):
    if state.get("finalization", False):
        return Command(goto=END)

    requests = state.get("orchestrator_requests") or []
    steps = state.get("steps") or {}

    if len(requests) == 0:
        # 没有任何请求
        for key in steps.keys(): 
            # 有正在跑或还没跑的步骤。这里goto=END不意味着整个运行的结束
            # 因为在router那里使用Send()并行地分发给stepnode任务，所以此处也是
            # 处于并行执行状态。即使这里Goto=END，还有没结束的执行流程
            if not steps[key].status in ['done', 'cancelled', 'failed']:
                return Command(goto=END)
 
        replanning_prompt = "Replanning requests from Router:\nAll the steps finished (done, cancelled or failed). Determine if further actions are need. If not, summary this run and set `finalization = True`. "
        return Command(update={ "orchestrator_is_working": True },
                       goto = Send("orchestrator_node", arg={
                           "messages": [HumanMessage(content=replanning_prompt)],
                           "phase": "replanning",
                           # Keep the same full-step baseline as normal
                           # dispatches so the orchestrator only returns a
                           # delta and cannot overwrite concurrent changes.
                           "steps": {key: step.model_copy(deep=True)
                                     for key, step in steps.items()},
                       }))

    request = dict(requests[0])
    # Always hand the orchestrator the current outer step snapshot, not the
    # snapshot captured when the request was queued.  A stale preparation
    # request could otherwise reset `workdir`/`preparation_stage` that a newer
    # request had already completed, making the router re-issue the same work.
    request["steps"] = {key: step.model_copy(deep=True)
                        for key, step in steps.items()}
    return Command(
        update= {
            "orchestrator_requests": Delete(indices=[0]),
            "orchestrator_is_working": True
        },
        goto=Send("orchestrator_node", arg=request)
    )


def _dependency_done(depend_on_id:str, steps:Dict[str, Step]) -> bool:
    """Whether a dependency is resolved.

    A dependency can be a concrete step id ("plan-1:step-2:name") or a whole
    plan ("plan-1"), as documented by `set_step_dependence`.
    """
    if depend_on_id in steps:
        return steps[depend_on_id].status == "done"

    if depend_on_id.startswith("plan-") and ":" not in depend_on_id:
        members = [key for key in steps if key.split(":", 1)[0] == depend_on_id]
        return len(members) > 0 and all(steps[key].status == "done" for key in members)

    return False


def router(state:WorkflowState):
    req_list = []
    stepnode_task_list = []
    step_running_update = dict()

    step_results = state.get("step_results") or {}
    steps = state.get("steps") or {} 
    # A node's `update` is not visible to the rest of this function, so keep a
    # local view of the step states after applying results/preparation changes.
    # 这里的意思是可能在同一次调用 router 过程中，步骤i完成了，而i+1恰好依赖于i，那么
    # 这里直接可以进行依赖的相关处理。而如果只是读state["steps"] 获取的永远都是旧值
    steps_view = {key: step.model_copy(deep=True) for key, step in steps.items()}

    update_step_results = dict()
    #1. 先处理 StepNode 的返回结果
    # 1.1 出错的步骤需要更正

    update_failed_steps = dict()
    message_for_failed_steps = "Correction request from Router:\nThe following steps failed. Please analyze the problem and fix it, and update `state.md`. If necessary, make a new plan, or modify the step and set it to 'pending'(to let the StepNode try again). \n\n"

    # 1.2 如果步骤正常完成，则仅仅记录状态
    update_success_steps = dict()
    message_for_success_steps = "Status logging request from Router:\n The following steps complete successfully. Please just log them in `state.md`.\n\n"
    
    for key in step_results.keys():
        result = step_results[key]
        if result.processed:
            continue
        if key not in steps_view:
            # A result for a step that no longer exists (e.g. replaced by a
            # new plan). Mark it processed so it does not loop forever.
            update_step_results[key] = result.model_copy()
            update_step_results[key].processed = True
            continue

        if result.status == "error":
            update_step_results[key] = result.model_copy()
            update_step_results[key].processed = True 

            steps_view[key].status = "failed"
            update_failed_steps[key] = steps_view[key].model_copy()

            message_for_failed_steps += f"step_id: {key}\nworkdir:{steps_view[key].workdir}\nsummary and error messages:{result.summary}\n\n"

        elif result.status == "success":
            update_step_results[key] = result.model_copy()
            update_step_results[key].processed = True 

            steps_view[key].status = "done"
            update_success_steps[key] = steps_view[key].model_copy()

            message_for_success_steps += f"step_id:{key}\nworkdir:{steps_view[key].workdir}\nsummary:{result.summary}\n\n"
        else: #这里实际上永远不会被访问
            raise RuntimeError(f"Unknown status {result.status} for step {key}.")

    if update_failed_steps:
        req_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_failed_steps)],
            "steps": update_failed_steps
        })

    if update_success_steps:
        req_list.append({
            "phase": "logging",
            "messages": [HumanMessage(content=message_for_success_steps)],
            "steps": update_success_steps
        })

    # 2. 检查步骤的依赖
    # 2.1 如果是需要replanning，且依赖已经完成，则进行replanning
    # 2.2 如果是一个pending步骤：
    #   2.2.1 如果依赖已经完成，则进行步骤准备，例如建立工作目录、把前面步骤的文件复制到当前目录
    #   2.2.2 如果依赖已经完成，步骤准备也完成，则可以分发任务到stepnode

    update_replanning_steps = dict()
    message_for_replanning = "Replanning requests from Router\n: Dependences resolved, the following steps now can be replanned (further decompose into sub-steps). NOTE: set `preparation_stage = 'done' when you finish replanning. `\n\n"


    update_pending_steps_preparation = dict()
    message_for_preparation = "Preparation requests from Router\n: Dependences resolved. Set and create workdirs for the following steps, and copy required files (if any) to their workdir, e.g., results files from previous steps. NOTE: set `preparation_stage = 'done' when you finish preparation. \n\n"

    message_for_workdir_correction = ""
    message_for_status_correction = ""
    message_for_dependence_correction = ""

    for key in steps_view.keys():
        step = steps_view[key]
        go_to_replanning = False 
        go_to_prepare = False 
        go_to_stepnode = False
        error_workdir_not_set = False  # 错误：已经准备好，但是工作目录未设置
        error_workdir_not_exist = False # 错误：工作目录不存在
        error_notyet_status_not_pending = False # 错误：尚未准备，但是status不是pending
        error_ongoing_status_not_pending = False # 错误：正在准备，但是status不是pending
        error_done_dependence_broken = False     # 错误：准备已经完成，但是它的依赖不全都是done，依赖损坏

        dependencies_done = all(_dependency_done(d_id, steps_view) for d_id in step.depend_on)

        if step.preparation_stage == "notyet":
            # 还没有准备的步骤 
            if step.status != "pending":
                error_notyet_status_not_pending = True 
            elif dependencies_done:
                # 检查它的依赖是否已经全部满足？如果依赖满足，则可以进行步骤准备
                if step.need_further_planning:
                    go_to_replanning = True  # 如果是需要进一步规划的步骤，且依赖完成，则去规划
                else:
                    go_to_prepare = True   # 不需要进一步规划的步骤，且依赖完成，则去准备该步骤。
        elif step.preparation_stage == "ongoing":
            if step.status != "pending":
                error_ongoing_status_not_pending = True 
        else:
            if step.status == "pending":
                # 已经准备好，并且状态是pending，那么这个步骤可以提交给stepnode计算了
                # 但是还要进行下列检查
                # 工作目录是否已经设置？如果有工作目录，它是否存在？
                if len(step.workdir) == 0:
                    error_workdir_not_set = True 
                elif not os.path.isdir(step.workdir):
                    error_workdir_not_exist = True 
                if not dependencies_done:
                    error_done_dependence_broken = True 
                elif not error_workdir_not_set and not error_workdir_not_exist:
                    go_to_stepnode = True 

        # 依赖的步骤全都等于done 
        step_description = f"step_id: {key}\ncontent: {step.content}\n"

        step_require_artifacts = ""
        if step.required_artifacts:
            step_require_artifacts = f"required_artifacts:\n|filename or wildcard|purpose|\n|---|---|\n"
            step_require_artifacts += "".join([f"|{fname}|{step.required_artifacts[fname]}|\n" for fname in step.required_artifacts.keys()])

        step_depend_on = ""
        if step.depend_on:
            step_depend_on = f"dependence:\n|step_id|workdir|status|\n|---|---|---|\n"
            for d_id in step.depend_on:
                if d_id in steps_view:
                    dep = steps_view[d_id]
                    step_depend_on += f"|{d_id}|{dep.workdir}|{dep.status}|\n"
                else:
                    dep_status = "done" if _dependency_done(d_id, steps_view) else "pending"
                    step_depend_on += f"|{d_id}|-|{dep_status}|\n"

        if error_workdir_not_set:
            message_for_workdir_correction += f"Workdir of step {key} is not set.\n" 
        if error_workdir_not_exist:
            message_for_workdir_correction += f"Workdir {step.workdir} for step {key} does not exist.\n"

        if error_notyet_status_not_pending:
            message_for_status_correction += f"The preparation stage of {key} is 'notyet', but its status is {step.status}.\n"
        if error_ongoing_status_not_pending:
            message_for_status_correction += f"The preparation stage of {key} is 'ongoing', but its status is {step.status}.\n"
        
        if error_done_dependence_broken:
            message_for_dependence_correction += f"{step_description}{step_require_artifacts}{step_depend_on}\n"

        if go_to_prepare:
            step.preparation_stage = "ongoing"
            update_pending_steps_preparation[key] = step.model_copy()
            message_for_preparation += f"{step_description}{step_require_artifacts}{step_depend_on}\n" 

        if go_to_replanning:
            step.need_further_planning = False 
            step.preparation_stage = "ongoing"
            update_replanning_steps[key] = step.model_copy()
            message_for_replanning += f"{step_description}{step_require_artifacts}{step_depend_on}\n"

        if go_to_stepnode:
            step.status = "running"
            step_running_update[key] = step.model_copy()

            # 写入readme.md 
            write_readme(step.workdir, step)
            message_for_stepnode = f"Complete this step. Read `readme.md` in workdir to get the task content.\nWorkdir: {step.workdir}\nWhen task completes, write a summary section in `report.md` to summarize this run and error messages (if any)."
            stepnode_task_list.append(Send(node="step_node", 
                                    arg= {
                                        "workdir": step.workdir,
                                        "step_id": key,
                                        "task_content": message_for_stepnode,
                                        "loaded_skills": state.get("loaded_skills") or {},
                                        "interrupt_resume": {}
                                    }))


    if len(message_for_workdir_correction) > 0:
        header = "Correction requests from Router:\nThe workdirs of following steps are not set or does not exist\n"
        message_for_workdir_correction = header + message_for_workdir_correction
        req_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_workdir_correction)],
        })

    if len(message_for_status_correction) > 0:
        header = "Correction requests from Router:\nIf the preparation stage is 'notyet' or 'ongoing', the status of a step must be 'pending'. The following steps violate this rule.\n"
        message_for_status_correction = header + message_for_status_correction
        req_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_status_correction)],
        })

    if len(message_for_dependence_correction) > 0:
        header = "Correction request from Router:\nError: Dependence broken. The preparation stages of following steps are 'done', but the status of their dependent steps are not 'done'.\n"
        message_for_dependence_correction = header + message_for_dependence_correction
        req_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_dependence_correction)],
        })

    if update_pending_steps_preparation:
        req_list.append({
            "phase": "preparation",
            "steps": update_pending_steps_preparation,
            "messages": [HumanMessage(content=message_for_preparation)]
        })

    if update_replanning_steps:
        req_list.append({
            "phase": "replanning", 
            "steps": update_replanning_steps,
            "messages": [HumanMessage(content=message_for_replanning)]
        })


    goto_list = stepnode_task_list
    goto_list.append("orchestrator_request_queue")
    if step_running_update:
        req_list = [ { "steps": step_running_update }, *req_list]

    # Keep the outer workflow state consistent with the local view immediately.
    # The same changes are also forwarded to the orchestrator through the
    # request queue so its private state stays in sync.
    step_state_updates = dict()
    for changes in (update_failed_steps, update_success_steps,
                    update_pending_steps_preparation, update_replanning_steps,
                    step_running_update):
        step_state_updates.update(changes)

    return Command(goto=goto_list, update={
        "step_results": update_step_results,
        "steps": step_state_updates,
        "orchestrator_requests": req_list
    })



def write_readme(workdir_path:str, step:Step):
    readme_path = os.path.join(workdir_path, "readme.md")
    with open(readme_path, mode="w") as f:
        f.write(f"# readme: {step.step_id}\n\n")

        f.write(f"## Content\n")
        f.write(f"{step.content}\n\n")

        f.write(f"## Solution\n")
        f.write(f"{step.solution}\n\n")

        f.write(f"## Related skills\n")
        f.write(f"{step.related_skills}\n\n")

        f.write(f"## Required Artifacts\n")
        f.write(f"|file name or wildcard|purpose|\n|---|---|\n")
        for req_art in step.required_artifacts.keys():
            f.write(f"|{req_art}|{step.required_artifacts[req_art]}|\n")

        f.write(f"## Output Artifacts\n")
        f.write(f"|file name or wildcard|purpose|\n|---|---|\n")
        for req_art in step.output_artifacts.keys():
            f.write(f"|{req_art}|{step.output_artifacts[req_art]}|\n")

        f.write("\n") 