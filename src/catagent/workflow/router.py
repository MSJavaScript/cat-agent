
from .models import WorkflowState, Step, StepResult, skill_artifacts_from_skills, OrchestratorRequest
from .liveness import (
    DEAD,
    clear_step_run,
    mark_step_scheduled,
    prune_step_runs,
    step_run_phase,
)
from langchain.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from langgraph.graph import END 
from typing import Any, Dict, Optional
import asyncio, os, hashlib, datetime, json
import logging
logger = logging.getLogger("catagent")


def _json_safe(value: Any) -> Any:
    """Return a deterministic JSON-serialisable view of a request payload.

    Requests contain LangChain messages.  ``json.dumps`` cannot serialise
    them, and their ``id``/metadata are volatile, so only the message type and
    content are kept.  Dicts, lists, sets and pydantic models are handled
    recursively; everything else falls back to ``str``.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_json_safe(item) for item in value), key=repr)
    if hasattr(value, "content") and hasattr(value, "type"):
        return {"type": value.type, "content": _json_safe(value.content)}
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    return str(value)


def request_fingerprint(payload: Dict[str, Any]) -> str:
    """Return a stable key for a request payload.

    The router uses this as the queue key.  Two identical requests generated
    on different router passes therefore have the same key and cannot both be
    dispatched.
    """
    data = json.dumps(
        _json_safe(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

# 这里变成串行请求是因为任务步骤之间都是相互关联的。Orchestrator需要知道全局的状态
# NOTE: this node is wrapped by `orchestrator_request_queue_node` in agent.py,
# which guarantees that no request is dispatched while another orchestrator
# call is running.  Never busy-wait on a graph-state flag here: the state seen
# by a running node is an immutable snapshot and would never change.
async def orchestrator_request_queue(state:WorkflowState,
                                     config:Optional[RunnableConfig] = None):
    if state.get("finalization", False):
        return Command(goto=END)

    requests = state.get("orchestrator_requests") or {}
    if not isinstance(requests, dict):
        # Old list-based queue format.  The router regenerates any request
        # that is still needed, so treat it as empty instead of crashing.
        logger.warning("Ignoring legacy list-shaped orchestrator_requests.")
        requests = {}
    steps = state.get("steps", {})
    thread_id = ((config or {}).get("configurable", {})).get("thread_id", "")

    # Dispatch in creation order.  The key is the payload fingerprint, so an
    # identical request regenerated on a later pass keeps its original key and
    # cannot be queued twice.
    req_list = [
        (req.get("creation_timestamp", 0.0), bool(req.get("sent", False)), key)
        for key, req in requests.items()
    ]
    req_list.sort(key=lambda item: (item[0], item[2]))

    send_key = None 
    for _, s, k in req_list:
        if not s:
            send_key = k
            break 

    if send_key is None:
        dead_steps = []
        # True when at least one non-terminal step is legitimately waiting:
        # pending/dependency-blocked, or running with a live owner.
        #
        # Do not return early on the first waiting step: a dead `running` step
        # must still be reported even when a sibling is pending.  The old loop
        # returned END as soon as it saw the pending sibling, hiding the dead
        # step forever (the graph produced no interrupt and no result, so the
        # request queue never moved again).
        waiting = False
        for key in steps.keys(): 
            step = steps[key]
            if step.status in ['done', 'deleted']: # pending, running, done, deleted
                continue
            if step.status != "running":
                # pending / dependency-blocked: the router still owns it.
                waiting = True
                continue
            if not thread_id:
                # No thread id means the registry cannot be consulted (direct
                # callers, tests).  Preserve the old "wait" behaviour instead
                # of declaring every running step dead.
                waiting = True
                continue
            # `status == "running"` is only an intent written by the router; it
            # survives a crash, a cancellation and a process restart.  Consult
            # the in-process run registry (see workflow/liveness.py) to decide
            # whether anything is still going to produce a StepResult.
            # `scheduled`, `running`, `finished` and `paused` are all "wait";
            # only `dead` means the queue would wait forever.
            if step_run_phase(thread_id, key) != DEAD:
                waiting = True
                continue
            dead_steps.append(key)

        if dead_steps:
            logger.warning(
                "orchestrator_request_queue: no owner for running step(s) %s; "
                "reporting them as crashed so the router can correct the plan.",
                dead_steps,
            )
            return Command(goto="router", update={
                "step_results": {
                    key: StepResult(
                        status="error",
                        processed=False,
                        step_id=key,
                        summary=(
                            "StepNode is no longer running but the step is still "
                            "marked 'running' (crashed, cancelled, or the server "
                            "was restarted). No worker will produce a result."
                        ),
                        artifacts={},
                    ) for key in dead_steps
                }
            })

        if waiting:
            return Command(goto=END)

        replanning_prompt = "Replanning requests from Router:\nAll the steps finished (done or deleted). Determine if further actions are need. If not, summary this run and set `finalization = True`. "

        return Command(update={ "orchestrator_is_working": True },
                       goto = Send("orchestrator_node", arg={
                           "messages": [HumanMessage(content=replanning_prompt)],
                           "phase": "replanning",
                           "project_workdir": state.get("project_workdir", ""),
                           # Keep the same full-step baseline as normal
                           # dispatches so the orchestrator only returns a
                           # delta and cannot overwrite concurrent changes.
                           "steps": {key: step.model_copy(deep=True) for key, step in steps.items()},
                       }))

    payload = requests[send_key]["payload"]
    request = payload.copy()
    request["steps"] = {key: step.model_copy(deep=True) for key, step in steps.items()}

    request.setdefault("project_workdir", state.get("project_workdir", ""))
    return Command(
        update= {
            "orchestrator_requests": {send_key: {"sent": True}},
            "orchestrator_is_working": True
        },
        goto=Send("orchestrator_node", arg=request)
    )



#########################################
#  整个工作流的路由，遍历每个作业结果，生成作业处理的请求。
#  遍历每个步骤，生成步骤准备或状态纠正的请求
#
#  1. 处理Step结果。每个作业有个processed的标志，如果处理过了这个Step结果就跳过
#  1.1 如果是失败的StepResult，生成一个correction请求。
#  1.2 如果是成功的StepResult，生成一个状态记录请求
#  2. 处理步骤间的依赖，如果步骤依赖满足，则发送准备请求。如果准备完成则派发给StepNode 进行计算
#########################################

async def router(state:WorkflowState, config:RunnableConfig):
    # 辅助函数，一个步骤的依赖是否是done 
    def _dependency_done(depend_on_id:str, steps_view:dict) -> bool:
        if depend_on_id in steps_view:
            return steps_view[depend_on_id]["status"] == "done"
        if depend_on_id.startswith("plan-") and ":" not in depend_on_id:
            members = [key for key in steps_view if key.split(":", 1)[0] == depend_on_id]
            return len(members) > 0 and all(steps_view[key]["status"] == "done" for key in members)
        return False

    logger.info("Enter router node.")

    thread_id = config["configurable"]["thread_id"]
    req_payload_list = []
    stepnode_task_list = []
    step_running_update = dict()

    step_results = state.get("step_results") or {}
    steps = state.get("steps") or {} 

    # 只有以下三个字段会被更新，所以step_view只包含它们
    # 其中 status 会根据作业结果即时更新，在下面的for循环中要看到最新的status
    steps_view = dict()
    for key in steps.keys():
        steps_view[key] = {
            "status": steps[key].status,
            "need_further_planning": steps[key].need_further_planning,
            "n_times": steps[key].n_times
        }

    # skill_artifacts_pathes 设置了StepNode可访问的路径。在Orchestrator中可能加载了新的skill
    # 所以这里更新一下
    skill_artifacts_pathes = set(state.get("skill_artifacts_pathes") or ())
    skill_artifacts_pathes.update(
        skill_artifacts_from_skills(state.get("loaded_skills") or {})
    )

    update_step_results = dict()

    update_failed_steps = dict()
    message_for_failed_steps = "Correction request from Router:\nThe following steps may failed. If it really failed, please analyze the problem and fix it, and update `state.md`. To retry, you have two choices: (1) Revise the step solution and set status = 'pending', this can reuse chat history of StepNode but likely pollute the context and make StepNode fail again. (2) Set the old step status = 'deleted' to delete it, and create a new 'pending' step (call `create_step`) with different `step_id`. In this case, you must call `set_step_dependence` to set the workflow dependence. \n\n"

    update_success_steps = dict()
    message_for_success_steps = "Status logging request from Router:\n The following steps complete successfully. Please call `set_step_status` to set their status = 'done' so that subsequent steps can be prepared, and log the status in `state.md`.\n\n"

    logger.info("Router node, processing the results of steps.")
    for key in step_results.keys():
        result = step_results[key]
        if result.processed:
            continue
        # The step node returned normally (or the queue reported a dead run);
        # its liveness record has served its purpose.
        clear_step_run(thread_id, key)

        if key not in steps_view:
            # `register_raw_plan(replace)` really removes Step keys from the
            # shared state, and it may do so while this StepNode is running or
            # while its result is still queued.  The step no longer exists, so
            # there is nothing to correct; drop the result instead of
            # resurrecting a deleted step.
            logger.warning(
                "Router: dropping result for step %s because the step was "
                "deleted (register_raw_plan replace).", key
            )
            update_step_results[key] = {"processed": True}
            continue

        assert key in steps
        if result.status == "error":
            update_step_results[key] = {"processed": True}
            steps_view[key]["status"] = "deleted"
            update_failed_steps[key] = steps_view[key]
            message_for_failed_steps += f"step_id: {key}\nworkdir:{steps[key].workdir}\nsummary and error messages:{result.summary}\n\n"
        elif result.status == "success":
            update_step_results[key] = {"processed": True}
            steps_view[key]["status"] = "done"
            update_success_steps[key] = steps_view[key]
            message_for_success_steps += f"step_id:{key}\nworkdir:{steps[key].workdir}\nsummary:{result.summary}\n\n"
        else: #这里实际上永远不会被访问
            raise RuntimeError(f"Unknown status {result.status} for step {key}.")

    if update_failed_steps:
        req_payload_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_failed_steps)],
            "steps": update_failed_steps
        })

    if update_success_steps:
        req_payload_list.append({
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

    message_for_replanning = ""
    message_for_preparation = ""
    message_for_workdir_correction = ""
    message_for_status_correction = ""
    message_for_dependence_correction = ""

    logger.info("Router node, checking the dependence of steps.")
    for key in steps_view.keys():

        go_to_replanning = False 
        go_to_prepare = False 
        go_to_stepnode = False
        error_workdir_not_set = False        # 错误：已经准备好，但是工作目录未设置
        error_workdir_not_exist = False      # 错误：工作目录不存在
        error_preparation_status = False     # 错误：没有准备，但是status不是pending，或准备了，但是status仍然是pending
        error_done_dependence_broken = False # 错误：准备已经完成，但是它的依赖不全都是done，依赖损坏

        dependencies_done = all(_dependency_done(d_id, steps_view) for d_id in steps[key].depend_on)

        # 提问：steps_view中的status会根据前面的StepResult修改
        # 那么这里结合 steps 中的 preparation_done 来判定各种状态是否会造成冲突？
        # 回答：不会，因为未准备的一定不会运行，所以不存在它们的结果。
        # 如果未准备，且status是非pending 会让Orchestrator修正。
        # 可能有点问题的是 if error_done_dependence_broken 这个分支
        # 我们先假定步骤结果的Correction或logging Orchestrator 会正确执行标志位的设置
        # 即使没有，if error_done_dependence_broken 会让它去修正
        if not steps[key].preparation_done: #未准备
            if steps_view[key]["status"] != "pending":
                error_preparation_status = True 
            elif dependencies_done:
                if steps_view[key]["need_further_planning"]:
                    go_to_replanning = True  # 如果是需要进一步规划的步骤，且依赖完成，则去规划
                else:
                    go_to_prepare = True   # 不需要进一步规划的步骤，且依赖完成，则去准备该步骤。
        else:
            if steps_view[key]["status"] == "pending":
                if len(steps[key].workdir) == 0:
                    error_workdir_not_set = True 
                elif not os.path.isdir(steps[key].workdir):
                    error_workdir_not_exist = True 
                if not dependencies_done:
                    error_done_dependence_broken = True 
                elif not error_workdir_not_set and not error_workdir_not_exist:
                    go_to_stepnode = True 

        # 依赖的步骤全都等于done 
        step_description = f"step_id: {key}\ncontent: {steps[key].content}\n"

        step_require_artifacts = ""
        if steps[key].required_artifacts:
            step_require_artifacts = f"required_artifacts:\n|filename or wildcard|purpose|\n|---|---|\n"
            step_require_artifacts += "".join([f"|{fname}|{steps[key].required_artifacts[fname]}|\n" for fname in steps[key].required_artifacts.keys()])

        step_depend_on = ""
        if steps[key].depend_on:
            step_depend_on = f"dependence:\n|step_id|workdir|status|\n|---|---|---|\n"
            for d_id in steps[key].depend_on:
                # A dependence can be a concrete step id or a whole plan
                # ("plan-{id}").  The latter has no entry in `steps_view`.
                if d_id in steps_view:
                    depend_workdir = steps[d_id].workdir
                    depend_status = steps_view[d_id]["status"]
                elif d_id.startswith("plan-") and ":" not in d_id:
                    members = [
                        member for member in steps_view
                        if member.split(":", 1)[0] == d_id
                    ]
                    depend_workdir = "-"
                    depend_status = (
                        "done" if members and all(
                            steps_view[member]["status"] == "done"
                            for member in members
                        ) else "pending"
                    )
                else:
                    depend_workdir = "-"
                    depend_status = "pending"
                step_depend_on += f"|{d_id}|{depend_workdir}|{depend_status}|\n"
 

        if error_workdir_not_set:
            message_for_workdir_correction += f"Workdir of step {key} is not set.\n" 
        if error_workdir_not_exist:
            message_for_workdir_correction += f"Workdir {steps[key].workdir} for step {key} does not exist.\n"
        if error_preparation_status:
            message_for_status_correction += f"{key}: preparation_done = {steps[key].preparation_done}, status = {steps_view[key]["status"]}\n"
        if error_done_dependence_broken:
            message_for_dependence_correction += f"{step_description}{step_require_artifacts}{step_depend_on}\n"

        if go_to_prepare:
            message_for_preparation += f"{step_description}{step_require_artifacts}{step_depend_on}\n" 

        if go_to_replanning:
            steps_view[key]["need_further_planning"] = False
            update_replanning_steps[key] = steps_view[key]
            message_for_replanning += f"{step_description}{step_require_artifacts}{step_depend_on}\n"

        if go_to_stepnode:
            steps_view[key]["n_times"] += 1
            steps_view[key]["status"] = "running"

            step_running_update[key] = steps_view[key]
            # Register the intent *before* returning the Send list.  The queue
            # is dispatched in the same superstep and may run before the step
            # node gets a chance to mark itself running.
            mark_step_scheduled(thread_id, key)

            # 写入readme.md 
            write_readme(steps[key].workdir, steps[key], state["project_workdir"])
            message_for_stepnode = f"Complete this step. Read `readme.md` in workdir to get the task content.\nWorkdir: {steps[key].workdir}\nCall `get_tool_collection_infos` to get the descriptions of all the tool collections. Call `load_tool_collection` to load the tools.\nPrefer to use the tools in collections and scripts provided by skill (firstly use -h option to learn the usage, do not read them).\nWhen task completes, write a summary section in `report.md` to summarize this run and error messages (if any)."
            if steps_view[key]["n_times"] > 1:
                message_for_stepnode = f"Retry {steps_view[key]["n_times"] - 1}." + message_for_stepnode

            stepnode_task_list.append(Send(node="step_node", 
                                    arg= {
                                        "workdir": steps[key].workdir,
                                        "step_id": key,
                                        "task_content": message_for_stepnode,
                                        "project_workdir": state.get("project_workdir", ""),
                                        "loaded_skills": state.get("loaded_skills") or {},
                                        "skill_artifacts_pathes": skill_artifacts_pathes,
                                        "interrupt_resume": {}
                                    }))


    logger.info("Router node, all the task done.")

    if len(message_for_workdir_correction) > 0:
        header = "Correction requests from Router:\nThe workdirs of following steps are not set or does not exist\n"
        message_for_workdir_correction = header + message_for_workdir_correction
        req_payload_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_workdir_correction)],
        })

    if len(message_for_status_correction) > 0:
        header = "Correction requests from Router:\n`preparation_done` and `status` of a step should satisfy: `(not preparation_done) and status == 'pending'` or `preparation_done and status != 'pending'`. The following steps violate this rule.\n"
        message_for_status_correction = header + message_for_status_correction
        req_payload_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_status_correction)],
        })

    if len(message_for_dependence_correction) > 0:
        header = "Correction request from Router:\nError: Dependence broken. The `preparation_done` flags of following steps are `True`, but the status of their dependent steps are not 'done'.\n"
        message_for_dependence_correction = header + message_for_dependence_correction
        req_payload_list.append({
            "phase": "correction",
            "messages": [HumanMessage(content=message_for_dependence_correction)],
        })

    if len(message_for_preparation) > 0:
        header = "Preparation requests from Router\n: Dependences resolved. Set and create workdirs for the following steps, and copy required files (if any) to their workdir, e.g., results files from previous steps. When you finish preparation, call `set_step_preparation_done` to set `preparation_done = True`. \n\n"
        message_for_preparation = header + message_for_preparation
        req_payload_list.append({
            "phase": "preparation",
            "messages": [HumanMessage(content=message_for_preparation)]
        })

    if len(message_for_replanning) > 0:
        header =  "Replanning requests from Router\n: Dependences resolved, the following steps now can be replanned (further decompose into sub-steps). When you finish replanning, call `set_step_preparation_done` to set `preparation_done = True`.\n\n"
        message_for_replanning = header + message_for_replanning
        req_payload_list.append({
            "phase": "replanning", 
            "steps": update_replanning_steps,
            "messages": [HumanMessage(content=message_for_replanning)]
        })


    goto_list = stepnode_task_list
    goto_list.append("orchestrator_request_queue")

    if step_running_update:
       req_payload_list = [ { "steps": step_running_update }, *req_payload_list]

    # Queue requests under a stable fingerprint.  The reducer keeps an entry
    # that is already queued (and, in particular, preserves `sent=True`), so
    # regenerating the same request on a later router pass cannot dispatch it
    # a second time.
    req_dict = dict()
    now = datetime.datetime.now().timestamp()
    for index, req in enumerate(req_payload_list):
        fingerprint = request_fingerprint(req)
        if fingerprint in req_dict:
            continue
        req_dict[fingerprint] = OrchestratorRequest(
            sent=False,
            # Microsecond offsets keep the queue in generation order even
            # when several requests are created in the same clock tick.
            creation_timestamp=now + index * 1e-6,
            payload=req,
        )

    step_state_updates = dict()
    for changes in (update_failed_steps, update_success_steps, step_running_update, update_replanning_steps):
        step_state_updates.update(changes)

    prune_step_runs(thread_id, steps_view.keys())
    logger.info("Router node, exit.")
    return Command(goto=goto_list, update={
        "step_results": update_step_results,
        "steps": step_state_updates,
        "orchestrator_requests": req_dict
    })



def write_readme(workdir_path:str, 
                 step:Step,
                 project_workdir:str):
    readme_path = os.path.join(workdir_path, "readme.md")
    with open(readme_path, mode="w") as f:
        f.write(f"# readme: {step.step_id}\n\n")

        msg = "Maintain a report.md during execution. When the run completes, add a Summary section in report.md."
        f.write("## Content\n")
        f.write(f"{step.content}\n{msg}\n\n")

        f.write("## Solution\n")
        f.write(f"{step.solution}\n\n")

        f.write("## Related skills\n")
        f.write(f"{step.related_skills}\n\n")

        f.write("## Required Artifacts\n")
        f.write("|file name or wildcard|purpose|\n|---|---|\n")
        for req_art in step.required_artifacts.keys():
            f.write(f"|{req_art}|{step.required_artifacts[req_art]}|\n")

        f.write("## Output Artifacts\n")
        f.write("|file name or wildcard|purpose|\n|---|---|\n")
        for req_art in step.output_artifacts.keys():
            f.write(f"|{req_art}|{step.output_artifacts[req_art]}|\n")

        f.write('\n')
        rules = ["You should only focus on computation task, please following the following rules:",
                 f"- Please only work in workdir {step.workdir}. You can only access files in project workdir {project_workdir} and files of provided skills. Do not access other dirs or files.",
                 f"- If you need to submit a job, firstly write a run.sh and job_description.json. Move all the files related to job to a separate dir, don't let them pollute you workdir (especially when you submit a batch job).",
                 f"- Leave the job submission and downloading to Jobmanager, you should not care about them.",
                 f"- Don't try to debug any repo. Focus the scientific problem and computation task.",
                 f"- Don't use curl, wget or git etc. Don't install anything by yourself. If you really need, call `pause_to_ask` to ask user for help.",
                 f"- Be concise and fast, don't do the analysis that is not asked by the user."]
        f.write("## Rules\n")
        for line in rules:
            f.write(f"{line}\n")
        f.write("\n") 