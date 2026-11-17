from langchain.chat_models import BaseChatModel
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.config import get_stream_writer
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send, interrupt
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import HumanMessage
from datetime import datetime

from .workflow.models import (
    WorkflowState,
    StepResult,
    Step,
    SkillMetadata,
    WorkflowContext,
    skill_artifacts_from_skills,
)
from .workflow.orchestrator import create_orchestrator_agent, OrchestratorAgentState, OrchestratorContext
from .workflow.router import orchestrator_request_queue, router
from .workflow.liveness import (
    clear_step_run,
    mark_step_finished,
    mark_step_paused,
    mark_step_running,
    mark_step_scheduled,
)
from .stepnode.stepnode import create_stepnode_agent, InterruptState
from .stepnode.models import StepNodeContext

import logging
from typing import Literal, TypedDict, Dict, Any, Optional
import os, httpx, asyncio, traceback
from langgraph.errors import GraphBubbleUp
from langgraph.runtime import Runtime

logger = logging.getLogger("catagent")

class StepNodeInput(TypedDict, total=False):
    workdir: str       # 一个步骤首次运行时需要添加workdir
    step_id: str       # plan-id:step-id:name
    task_content: str 
    # Skills already loaded by the orchestrator, forwarded to the inner agent.
    loaded_skills: Dict[str, Any]
    # Skill files/directories the StepNode may access outside the project.
    skill_artifacts_pathes: set[str]
    # Copied from the outer workflow so direct node calls (tests) still work.
    project_workdir: str
    # 键是interrupt_id，值是针对interrupt的response
    interrupt_resume: Dict[str, Any]

class InterruptNodeInput(TypedDict, total=False):
    source:Literal["stepnode", "orchestrator"]
    step_id: str
    workdir: str
    project_workdir: str
    interrupt_ori_id: str 
    interrupt_payload: InterruptState
    # Outer step snapshot at dispatch time, carried through the pause so the
    # resumed orchestrator can return a field-level delta (including deleted
    # steps) instead of overwriting the outer state wholesale.
    steps: Dict[str, Step]

# The orchestrator is a singleton *per user thread*, not per process.  A global
# lock serialized every session in the process, which made one user's long
# orchestrator call block everyone else.  Locks and claims are keyed by
# thread_id so different users run concurrently while a single thread still
# remains serialized.
_ORCHESTRATOR_LOCKS: Dict[str, asyncio.Lock] = {}
_ORCHESTRATOR_CLAIMS: Dict[str, asyncio.Event] = {}


def get_orchestrator_sync(thread_id: str) -> tuple[asyncio.Lock, asyncio.Event]:
    """Return the lock/claim pair for one workflow thread.

    Everything runs on one event loop and there is no await between lookup and
    creation, so the dict operations are atomic with respect to other nodes.
    """
    lock = _ORCHESTRATOR_LOCKS.get(thread_id)
    if lock is None:
        lock = _ORCHESTRATOR_LOCKS[thread_id] = asyncio.Lock()
    claim = _ORCHESTRATOR_CLAIMS.get(thread_id)
    if claim is None:
        claim = _ORCHESTRATOR_CLAIMS[thread_id] = asyncio.Event()
    return lock, claim


def release_orchestrator_sync(thread_id: str) -> None:
    """Drop the registry entries once a workflow is finalized and idle."""
    lock = _ORCHESTRATOR_LOCKS.get(thread_id)
    claim = _ORCHESTRATOR_CLAIMS.get(thread_id)
    if lock is not None and (lock.locked() or (claim is not None and claim.is_set())):
        return
    _ORCHESTRATOR_LOCKS.pop(thread_id, None)
    _ORCHESTRATOR_CLAIMS.pop(thread_id, None)


def _child_config(config: RunnableConfig, thread_id: str) -> Dict[str, Any]:
    """Build the config for an inner graph on its own checkpoint thread.

    The inner orchestrator and every StepNode are separate graph invocations,
    so LangGraph applies the recursion limit per invocation.  Copy the
    app-level limit to them as well; otherwise ``--recursion-limit`` would only
    bound the outer workflow, which is not where the tool loops run.
    """
    child: Dict[str, Any] = {"configurable": {"thread_id": thread_id}}
    recursion_limit = (config or {}).get("recursion_limit")
    if recursion_limit is not None:
        child["recursion_limit"] = recursion_limit
    return child


def _outer_workdirs(
    runtime: Optional[Runtime],
    state: Dict[str, Any],
) -> tuple[str, Dict[str, str]]:
    """Return (project_workdir, resource_dirs) for an inner-agent context.

    The normal path is the outer graph runtime context supplied by
    ``_stream_chat``.  ``state`` is a fallback for direct node calls in tests
    and for older checkpoints that persisted ``project_workdir`` in the graph
    state.
    """
    context = getattr(runtime, "context", None) if runtime is not None else None
    project_workdir = _context_value(context, "project_workdir")
    resource_dirs = _context_value(context, "resource_dirs")
    if not project_workdir:
        project_workdir = state.get("project_workdir") or ""
    if resource_dirs is None:
        resource_dirs = state.get("resource_dirs") or {}
    return project_workdir, resource_dirs


def _context_value(context: Any, key: str) -> Any:
    """Read a key from a dict-like or model-like runtime context."""
    if context is None:
        return None
    if isinstance(context, dict):
        return context.get(key)
    return getattr(context, key, None)


def _steps_delta(baseline: Dict[str, Step],
                 current: Dict[str, Step]) -> Dict[str, Any]:
    """Return only the step fields the orchestrator actually changed.

    The outer workflow and the inner orchestrator each keep a copy of
    `steps`.  Returning the whole inner copy would overwrite router-owned
    transitions (e.g. preparation done, status running) that happened while
    the orchestrator was busy, which made the router re-issue the same
    preparation request.  A field-level delta avoids that lost update.
    """
    delta: Dict[str, Any] = dict()
    # `register_raw_plan(replace)` really deletes Step keys inside the inner
    # graph.  Those deletions must be propagated too, otherwise the outer
    # workflow keeps a step that no longer exists and may dispatch it again.
    for step_id in baseline:
        if step_id not in current:
            delta[step_id] = None
    for step_id, step in current.items():
        old = baseline.get(step_id)
        if old is None:
            # A newly created step must stay a Step instance; a plain dict
            # would be stored verbatim by dict_reducer because there is no
            # existing Step to patch.
            delta[step_id] = step.model_copy(deep=True)
            continue
        patch = dict()
        for field in Step.model_fields:
            if getattr(step, field) != getattr(old, field):
                patch[field] = getattr(step, field)
        if patch:
            delta[step_id] = patch
    return delta


def node_set_flag(state:WorkflowState, config:RunnableConfig):
    if state.get("finalization", False):
        release_orchestrator_sync(config["configurable"]["thread_id"])
        return Command(goto=END)

    MAXTIME = 300
    now = datetime.now().timestamp()
    # Drop requests that were dispatched but not acknowledged for MAXTIME.
    # Until then the fingerprint stays in the queue, which is what prevents an
    # identical request from being dispatched again.
    requests = state.get("orchestrator_requests") or {}
    if not isinstance(requests, dict):
        requests = {}
    outdated_requests = dict()
    for key, request in requests.items():
        if request.get("sent") and \
            now - request.get("creation_timestamp", now) > MAXTIME:
            outdated_requests[key] = None 

    # The orchestrator just changed the plan/step state, so let the router
    # re-check dependences and dispatch whatever is now ready.
    return Command(goto="router", update={ "orchestrator_is_working": False,
                                          "orchestrator_requests":outdated_requests })

def create_catagent(model:BaseChatModel, 
                    job_manager_url:str,
                    http_client:httpx.AsyncClient,
                    checkpointer:Checkpointer,
                    loaded_skills:Dict[str, SkillMetadata] | None = None):

    loaded_skills = loaded_skills or {}
    # skill 文件所在的路径由 init_skills_node 写入 WorkflowState，StepNode 至多能访问它们。

    dirname = os.path.join(os.path.dirname(__file__), "system_skills")
    skill_path = []
    for f in ["orchestrator_prompt.md", "bash_tips.md", "file_and_state_management.md"]:
        p = os.path.join(dirname, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} does not exist.")
        else:
            skill_path.append(p)
    orchestrator_agent = create_orchestrator_agent(model=model,
                                                   system_skill_path=skill_path,
                                                   job_manager_url=job_manager_url,
                                                   http_client=http_client,
                                                   checkpointer=checkpointer) 

    skill_path = []
    for f in [ "stepnode_prompt.md", "file_and_state_management.md", 
              "job_submission.md", "file_processing_for_step_node.md"]:
        p = os.path.join(dirname, f)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} does not exist.")
        else:
            skill_path.append(p)
    
    stepnode_agent:CompiledStateGraph = create_stepnode_agent(model=model,
                                           system_skill_pathes=skill_path,
                                           job_manager_url=job_manager_url,
                                           http_client=http_client,
                                           checkpointer=checkpointer) # 这里需要一个checkpointer 

    # langgraph 中节点跳转有两种方式，一种是在条件边中返回节点名称，或用Send api
    # 条件边只通过 add_conditional_edges()添加
    # 另一种方式是在一个节点中使用Command(goto=..., update=...) 命令
    # goto= 接受的参数可以是一个节点名称，一个Send()或一个列表。当传入节点名称时，update 中的输入
    # 会被目标节点使用，并且当你在列表里使用同样的节点名称时那个节点仅会被运行一次。但是Send()
    # 指定的节点是可以多次运行的，所以可以用来执行并行任务。
    # 当使用Command()之后原来已有的边不受影响。但为了防止混乱，最好不要用add_edge 在用了Command的节点
    # 和下游节点之间再建立边。

    # The orchestrator is a singleton per workflow: it shares the same state and
    # must not run twice concurrently.  The graph state flag alone is only a
    # snapshot inside a running node, so real coordination needs an asyncio
    # primitive.  The claim prevents two queue nodes from dispatching the same
    # request at the same time.  Both are keyed by thread_id (see
    # get_orchestrator_sync) so separate users do not block each other.

    def _interrupt_values(payload:dict) -> list:
        """Return the Interrupt objects carried by one stream chunk."""
        interrupts = payload["data"].get("__interrupt__")
        if interrupts is None:
            return []
        if not isinstance(interrupts, (list, tuple)):
            interrupts = (interrupts,)
        return list(interrupts)

    async def step_node(state:StepNodeInput,
                        config:RunnableConfig,
                        runtime: Optional[Runtime[WorkflowContext]] = None):
        logger.info(f"Enter step_node: step_id = {state.get('step_id')}, workdir = {state.get('workdir')}\ninterrupt_resume = {state.get('interrupt_resume')}.")

        writer = get_stream_writer()
        thread_id = config["configurable"]["thread_id"]
        step_id = state.get("step_id", "")
        chat_step_node_id = f"{thread_id}:{step_id}"
        step_node_config = _child_config(config, chat_step_node_id)
        project_workdir, resource_dirs = _outer_workdirs(runtime, state)
        # A resumed call may not carry the step workdir in its Send arg.  The
        # inner graph still needs it for get_workdir() and the path guard, so
        # fall back to the project root rather than passing an empty string.
        step_workdir = state.get("workdir") or project_workdir
        step_context: StepNodeContext = {
            "project_workdir": project_workdir,
            "step_workdir": step_workdir,
            "resource_dirs": resource_dirs,
        }

        # Publish liveness *before* the first await so the request queue, which
        # the router dispatches in the same superstep, can see that this step
        # has an owner.  `mark_step_scheduled` in the router covers the narrow
        # window before this node starts.
        mark_step_running(thread_id, step_id, asyncio.current_task())

        try:
            interrupt_resume = state.get("interrupt_resume") or {}
            if interrupt_resume: # 中断后接着运行
                payloads = stepnode_agent.astream(input = Command(resume=interrupt_resume),
                                                  config = step_node_config,
                                                  context = step_context,
                                                  stream_mode = ["custom", "updates"],
                                                  version = "v2")
            else: # 首次运行
                payloads = stepnode_agent.astream(
                    input={ # 强制要求 task_content 和workdir必须存在，否则报错
                        "role": "StepNode",
                        "stepnode_finalization": False,
                        "messages": [HumanMessage(content=state["task_content"])],
                        "workdir": state["workdir"],
                        "loaded_skills": state.get("loaded_skills") or {},
                        "skill_artifacts_pathes": set(state.get("skill_artifacts_pathes") or set())},
                    config = step_node_config,
                    context = step_context,
                    stream_mode = ["custom", "updates"],
                    version = "v2")

            collected_interrupts = dict()
            async for payload in payloads:
                # payload 是一个字典
                # 有三个键 type, ns 和 data。如果是updates类型，data只有一个键是节点名称
                # {'type': 'custom', 'ns': (), 'data': {'messages': 'Hello'}}
                if payload.get("type") == "updates" and "__interrupt__" in (payload.get("data") or {}):
                    for interrupt_obj in _interrupt_values(payload):
                        collected_interrupts[interrupt_obj.id] = interrupt_obj.value
                else:
                    writer({"source":chat_step_node_id, **payload })

            # 这里收集的中断可能是很多个提交作业的中断
            if collected_interrupts:
                send_list = [Send("interrupt_node", arg={
                    "source": "stepnode",
                    "step_id": step_id,
                    "workdir": state.get("workdir", ""),
                    "project_workdir": project_workdir,
                    "interrupt_ori_id": key,
                    "interrupt_payload": collected_interrupts[key]
                }) for key in collected_interrupts.keys()]
                command = Command(goto=send_list)
            else:
                stepnode_state = await stepnode_agent.aget_state(config=step_node_config)
                raw_step_result = stepnode_state.values.get("step_result") or {}
                if hasattr(raw_step_result, "model_dump"):
                    raw_step_result = raw_step_result.model_dump(mode="python")

                step_result = StepResult(status=raw_step_result.get("status", "error"),
                                         processed=False,
                                         summary=raw_step_result.get("summary", ""))

                # 转到router，更新steps_result
                command = Command(goto="router",
                                  update={ "step_results": {step_id: step_result} })
        except asyncio.CancelledError:
            # A cancelled run (shutdown, client disconnect) must not look like
            # a live StepNode after the event loop comes back.
            clear_step_run(thread_id, step_id)
            raise
        except GraphBubbleUp:
            # Control-flow exceptions (interrupts) must reach LangGraph.
            clear_step_run(thread_id, step_id)
            raise
        except Exception as exc:
            # An uncaught exception used to abort the whole superstep: the
            # queue's writes were discarded, `Step.status` stayed "running" and
            # nothing ever moved the workflow again.  Convert it into the same
            # StepResult the router already knows how to correct.
            clear_step_run(thread_id, step_id)
            logger.exception(f"StepNode {step_id} crashed.")
            summary = traceback.format_exc()
            if len(summary) > 4000:
                summary = "..." + summary[-4000:]
            if not step_id:
                # Nothing to attribute the failure to; keep the old behaviour.
                raise
            return Command(goto="router", update={
                "step_results": {step_id: StepResult(
                    status="error",
                    processed=False,
                    summary=f"StepNode crashed: {type(exc).__name__}: {exc}\n\n{summary}")}
            })

        mark_step_finished(thread_id, step_id)
        return command

    

    async def orchestrator_node(state:OrchestratorAgentState,
                                config:RunnableConfig,
                                runtime: Optional[Runtime[WorkflowContext]] = None):
        # A resume enters this node through Send(..., {"interrupt_resume": ...}),
        # so the outer state (messages/steps) is not part of ``state`` here.
        last_message = state["messages"][-1].content if state.get("messages") else "<none>"

        length = len(last_message)
        logger.info(f"Enter orchestrator_node. last messages = {last_message[:250] if length > 250 else last_message}\ninterrupt_resume = {state.get('interrupt_resume')}")

        writer = get_stream_writer()
        # 这里假设它必须提供thread_id
        thread_id = config["configurable"]["thread_id"]
        # The inner orchestrator is a standalone compiled graph, not a real
        # subgraph.  It must use a different checkpoint thread from the outer
        # workflow, otherwise both graphs overwrite each other's state when
        # they share the same checkpointer.
        orchestrator_config = _child_config(config, f"{thread_id}:orchestrator")
        project_workdir, resource_dirs = _outer_workdirs(runtime, state)
        orchestrator_context: OrchestratorContext = {
            "project_workdir": project_workdir,
            "resource_dirs": resource_dirs,
        }
        orchestrator_lock, orchestrator_claimed = get_orchestrator_sync(thread_id)
        orchestrator_claimed.clear()

        async with orchestrator_lock:
            interrupt_resume = state.get("interrupt_resume") or {}
            baseline_steps = state.get("steps") or {}
            if interrupt_resume:
                payloads = orchestrator_agent.astream(input=Command(resume=interrupt_resume),
                                                      config = orchestrator_config,
                                                      context = orchestrator_context,
                                                      stream_mode=["custom", "updates"],
                                                      version="v2")  
            else:
                messages = state.get("messages", [])
                if not messages: 
                    # 没有消息，只更新状态。router里有个请求是只把steps的status设置成running
                    # 这个不需要运行agent 
                    await orchestrator_agent.aupdate_state(config=orchestrator_config, values=state)
                    # 回头更新WorkflowState的steps字段
                    return Command(goto="node_set_flag",
                                   update={"steps": state.get("steps", {})})

                # `steps` is the full outer snapshot taken when this request
                # was dispatched.  Applying it to the inner graph makes the
                # orchestrator work on the same status/workdir/
                # need_further_planning values the router sees; the delta on
                # the way back still protects concurrent outer updates.
                agent_input = {key:state[key] for key in state.keys() if key != "messages"}
                agent_input["messages"] = [ messages[-1] ]
                agent_input["role"] = "Orchestrator"
                payloads = orchestrator_agent.astream(input = agent_input,
                                                      config = orchestrator_config,
                                                      context = orchestrator_context,
                                                      stream_mode = ["custom", "updates"],
                                                      version = "v2")

            collected_interrupts = dict()
            async for payload in payloads:
                if payload.get("type") == "updates" and "__interrupt__" in (payload.get("data") or {}):
                    for interrupt_obj in _interrupt_values(payload):
                        collected_interrupts[interrupt_obj.id] = interrupt_obj.value
                else:
                    writer({"source":thread_id, **payload})

            if collected_interrupts:
                send_list = [Send("interrupt_node", arg={
                    "source": "orchestrator",
                    "step_id": "",
                    "project_workdir": project_workdir,
                    "interrupt_ori_id": key,
                    "interrupt_payload": collected_interrupts[key],
                    # Carry the baseline across the pause.  Without it the
                    # resumed turn cannot tell which steps the orchestrator
                    # deleted, so a replacement would leave a phantom step in
                    # the outer workflow.
                    "steps": baseline_steps,
                }) for key in collected_interrupts.keys()]
                return Command(goto=send_list)

            lastest_state = await orchestrator_agent.aget_state(config=orchestrator_config)
            post_steps = lastest_state.values.get("steps", {}) or {}
            post_skills = lastest_state.values.get("loaded_skills", {}) or {}
            # Only propagate what the orchestrator actually changed; the rest of
            # the outer `steps` may have advanced concurrently (preparation
            # done, a stepnode result, ...).  `_steps_delta` also emits `None`
            # for steps the orchestrator deleted.  A resume without a carried
            # baseline falls back to treating every inner step as new, which is
            # the old whole-state behaviour.
            steps_update = _steps_delta(baseline_steps, post_steps)

            return Command(goto="node_set_flag",
                           update={
                               "steps": steps_update,
                               "loaded_skills": post_skills,
                               # Skills loaded during this turn must be usable by
                               # the StepNodes dispatched in the next superstep.
                               "skill_artifacts_pathes":
                                   skill_artifacts_from_skills(post_skills),
                               "finalization": lastest_state.values.get("finalization", False)
                           })

    def interrupt_node(state:InterruptNodeInput, config:RunnableConfig):
        logger.info(f"Enter interrupt_node. source = {state['source']}, step_id = {state['step_id']}.")
        thread_id = config["configurable"]["thread_id"]
        if state['interrupt_payload']["kind"] == "JobSubmission":
            logger.info(f"JobSubmission interrupt. step_id = {state['step_id']}.")

        # A paused step is *not* dead.  Mark it before suspending so the request
        # queue does not mistake `status == "running"` with no StepNode task for
        # a crash (this matters when a sibling step finishes while this one is
        # waiting for the user).  The mark survives the GraphInterrupt; it is
        # replaced by `scheduled` when the node runs again after the resume.
        if state["source"] == "stepnode":
            mark_step_paused(thread_id, state.get("step_id", ""))

        # 该节点把内层agent发送的interrupt二次发送到外部
        # Attach the producing source to the value for the debug UI.  The
        # payload converter removes this private key before display, and the
        # resume response is unaffected because it never reads the value back.
        value = state["interrupt_payload"]
        if isinstance(value, dict):
            if state.get("source") == "stepnode":
                debug_source = f"{thread_id}:{state.get('step_id', '')}"
            elif state.get("source") == "orchestrator":
                debug_source = thread_id
            else:
                debug_source = None
            value = {**value, "_debug_source": debug_source}
        try:
            response = interrupt(value=value)
        except GraphBubbleUp:
            # Still suspended: keep the `paused` mark for the next superstep.
            raise
        # The outer interrupt id is not the id used by the inner graph.  Keep
        # the original id so the answer can be routed back to the exact inner
        # interrupt when several interrupts are pending at the same time.
        resume_map = {state["interrupt_ori_id"]: response}
        if state["source"] == "orchestrator":
            return Command(goto=Send("orchestrator_node",
                                     arg={"interrupt_resume": resume_map,
                                          "project_workdir": state.get("project_workdir", ""),
                                          # Keep the dispatch-time baseline so
                                          # the delta can propagate deletions.
                                          "steps": state.get("steps") or {}}))
        else:
            # 这里不需要设置workdir，直接resume，原来的workdir不会被覆盖
            mark_step_scheduled(thread_id, state.get("step_id", ""))
            return Command(goto=Send("step_node",
                                     arg={"step_id": state["step_id"],
                                          "workdir": state.get("workdir", ""),
                                          "project_workdir": state.get("project_workdir", ""),
                                          "interrupt_resume": resume_map}))

    async def orchestrator_request_queue_node(state:WorkflowState,
                                               config:RunnableConfig):
        logger.info("orchestrator_request_queue_node")
        thread_id = config["configurable"]["thread_id"]
        orchestrator_lock, orchestrator_claimed = get_orchestrator_sync(thread_id)
        # A request may only be dispatched when the orchestrator of *this*
        # thread is idle.  Otherwise leave it in `orchestrator_requests`; the
        # orchestrator will return through node_set_flag -> router when it
        # finishes, and the queue will be retried then.  The state flag also
        # covers the case where the orchestrator is paused on an interrupt
        # (the lock is not held while a graph is waiting for a callback).
        if (state.get("orchestrator_is_working", False)
                or orchestrator_lock.locked()
                or orchestrator_claimed.is_set()):
            return Command(goto=END)

        orchestrator_claimed.set()
        try:
            command = await orchestrator_request_queue(state, config)
        except BaseException:
            orchestrator_claimed.clear()
            raise

        goto = getattr(command, "goto", None)
        sends = goto if isinstance(goto, (list, tuple)) else [goto]
        if not any(isinstance(item, Send) and item.node == "orchestrator_node"
                   for item in sends):
            orchestrator_claimed.clear()
        return command

    
    def init_skills_node(state:WorkflowState):
        skills = state.get("loaded_skills") or loaded_skills
        updates: Dict[str, Any] = {
            "skill_artifacts_pathes": skill_artifacts_from_skills(skills)
        }
        if not state.get("loaded_skills"):
            updates["loaded_skills"] = loaded_skills
        return Command(goto="orchestrator_node", update=updates)

    graph = StateGraph(WorkflowState, context_schema=WorkflowContext)
    graph.add_node("init_skills", init_skills_node)
    graph.add_node("orchestrator_node", orchestrator_node)
    graph.add_node("node_set_flag", node_set_flag)
    graph.add_node("step_node", step_node)
    graph.add_node("orchestrator_request_queue", orchestrator_request_queue_node)
    graph.add_node("router", router)
    graph.add_node("interrupt_node", interrupt_node)
    # 每个节点里都有Command(goto=...)所以只需要设置入口点，不需要添加边
    graph.add_edge(START, "init_skills")
    graph = graph.compile(checkpointer=checkpointer)
    return graph 

 