from langchain.chat_models import BaseChatModel
from langgraph.types import Checkpointer
from langgraph.graph import START, END, StateGraph
from langgraph.config import get_stream_writer
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send, interrupt
from langgraph.graph.state import CompiledStateGraph
from langchain.messages import HumanMessage

from .workflow.models import WorkflowState, StepResult, Step, SkillMetadata
from .workflow.orchestrator import create_orchestrator_agent, OrchestratorAgentState
from .workflow.router import orchestrator_request_queue, router
from .stepnode.stepnode import create_stepnode_agent, InterruptState

import logging
from typing import Literal, TypedDict, Dict, Any
import os, httpx, asyncio

logger = logging.getLogger("catagent")

class StepNodeInput(TypedDict, total=False):
    workdir: str       # 一个步骤首次运行时需要添加workdir
    step_id: str       # plan-id:step-id:name
    task_content: str 
    # Skills already loaded by the orchestrator, forwarded to the inner agent.
    loaded_skills: Dict[str, Any]
    # 键是interrupt_id，值是针对interrupt的response
    interrupt_resume: Dict[str, Any]

class InterruptNodeInput(TypedDict):
    source:Literal["stepnode", "orchestrator"]
    step_id: str
    interrupt_ori_id: str 
    interrupt_payload: InterruptState

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
    # The orchestrator just changed the plan/step state, so let the router
    # re-check dependences and dispatch whatever is now ready.
    return Command(goto="router", update={ "orchestrator_is_working": False })

def create_catagent(model:BaseChatModel, 
                    job_manager_url:str,
                    http_client:httpx.AsyncClient,
                    checkpointer:Checkpointer,
                    loaded_skills:Dict[str, SkillMetadata] | None = None):

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

    async def step_node(state:StepNodeInput, config:RunnableConfig):
        logger.info(f"Enter step_node: step_id = {state.get('step_id')}, workdir = {state.get('workdir')}\ninterrupt_resume = {state.get('interrupt_resume')}.")

        writer = get_stream_writer()
        thread_id = config["configurable"]["thread_id"]
        step_id = state.get("step_id", "")
        chat_step_node_id = f"{thread_id}:{step_id}"
        step_node_config = {"configurable": {"thread_id": chat_step_node_id}}

        interrupt_resume = state.get("interrupt_resume") or {}
        if interrupt_resume: # 中断后接着运行
            payloads = stepnode_agent.astream(input = Command(resume=interrupt_resume),
                                              config = step_node_config,
                                              stream_mode = ["custom", "updates"],
                                              version = "v2")
        else: # 首次运行
            payloads = stepnode_agent.astream(
                input={ # 强制要求 task_content 和workdir必须存在，否则报错
                    "messages": [HumanMessage(content=state["task_content"])],
                    "workdir": state["workdir"],
                    "loaded_skills": state.get("loaded_skills") or {}},
                config = step_node_config,
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
                "interrupt_ori_id": key,
                "interrupt_payload": collected_interrupts[key]
            }) for key in collected_interrupts.keys()]
            return Command(goto=send_list)

        stepnode_state = await stepnode_agent.aget_state(config=step_node_config)
        raw_step_result = stepnode_state.values.get("step_result") or {}
        if hasattr(raw_step_result, "model_dump"):
            raw_step_result = raw_step_result.model_dump(mode="python")

        step_result = StepResult(status=raw_step_result.get("status", "error"),
                                 processed=False,
                                 step_id=step_id,
                                 summary=raw_step_result.get("summary", ""),
                                 artifacts=raw_step_result.get("output_artifacts", {}))

        # 转到router，更新steps_result
        return Command(goto="router",
                       update={ "step_results": {step_id: step_result} })

    

    async def orchestrator_node(state:OrchestratorAgentState, config:RunnableConfig):
        # A resume enters this node through Send(..., {"interrupt_resume": ...}),
        # so the outer state (messages/steps) is not part of ``state`` here.
        last_message = state["messages"][-1].content if state.get("messages") else "<none>"
        logger.info(f"Enter orchestrator_node. last messages = {last_message}\ninterrupt_resume = {state.get('interrupt_resume')}")

        writer = get_stream_writer()
        # 这里假设它必须提供thread_id
        thread_id = config["configurable"]["thread_id"]
        # The inner orchestrator is a standalone compiled graph, not a real
        # subgraph.  It must use a different checkpoint thread from the outer
        # workflow, otherwise both graphs overwrite each other's state when
        # they share the same checkpointer.
        orchestrator_config = {
            "configurable": {"thread_id": f"{thread_id}:orchestrator"}
        }
        orchestrator_lock, orchestrator_claimed = get_orchestrator_sync(thread_id)
        orchestrator_claimed.clear()

        async with orchestrator_lock:
            interrupt_resume = state.get("interrupt_resume") or {}
            baseline_steps = state.get("steps") or {}

            if interrupt_resume:
                payloads = orchestrator_agent.astream(input=Command(resume=interrupt_resume),
                                                      config = orchestrator_config,
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
                agent_input = {key:state[key] for key in state.keys() if key != "messages"}
                agent_input["messages"] = [ messages[-1] ]
                payloads = orchestrator_agent.astream(input = agent_input,
                                                      config = orchestrator_config,
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
                    "interrupt_ori_id": key,
                    "interrupt_payload": collected_interrupts[key]
                }) for key in collected_interrupts.keys()]
                return Command(goto=send_list)

            lastest_state = await orchestrator_agent.aget_state(config=orchestrator_config)
            post_steps = lastest_state.values.get("steps", {}) or {}
            if interrupt_resume:
                # A resumed turn must return the whole inner state: changes made
                # before the interrupt were never synced to the outer workflow.
                steps_update = post_steps
            else:
                # Only propagate what the orchestrator actually changed; the
                # rest of the outer `steps` may have advanced concurrently
                # (preparation done, a stepnode result, ...).
                steps_update = _steps_delta(baseline_steps, post_steps)

            return Command(goto="node_set_flag",
                           update={
                               "steps": steps_update,
                               "loaded_skills": lastest_state.values.get("loaded_skills", {}),
                               "finalization": lastest_state.values.get("finalization", False)
                           })

    def interrupt_node(state:InterruptNodeInput, config:RunnableConfig):
        logger.info(f"Enter interrupt_node. source = {state['source']}, step_id = {state['step_id']}.")
        if state['interrupt_payload']["kind"] == "JobSubmission":
            logger.info(f"JobSubmission interrupt. step_id = {state['step_id']}.")

        # 该节点把内层agent发送的interrupt二次发送到外部
        # Attach the producing source to the value for the debug UI.  The
        # payload converter removes this private key before display, and the
        # resume response is unaffected because it never reads the value back.
        value = state["interrupt_payload"]
        if isinstance(value, dict):
            thread_id = config["configurable"]["thread_id"]
            if state.get("source") == "stepnode":
                debug_source = f"{thread_id}:{state.get('step_id', '')}"
            elif state.get("source") == "orchestrator":
                debug_source = thread_id
            else:
                debug_source = None
            value = {**value, "_debug_source": debug_source}
        response = interrupt(value=value)
        # The outer interrupt id is not the id used by the inner graph.  Keep
        # the original id so the answer can be routed back to the exact inner
        # interrupt when several interrupts are pending at the same time.
        resume_map = {state["interrupt_ori_id"]: response}
        if state["source"] == "orchestrator":
            return Command(goto=Send("orchestrator_node",
                                     arg={"interrupt_resume": resume_map}))
        else:
            # 这里不需要设置workdir，直接resume，原来的workdir不会被覆盖
            return Command(goto=Send("step_node",
                                     arg={"step_id": state["step_id"],
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
            command = await orchestrator_request_queue(state)
        except BaseException:
            orchestrator_claimed.clear()
            raise

        goto = getattr(command, "goto", None)
        sends = goto if isinstance(goto, (list, tuple)) else [goto]
        if not any(isinstance(item, Send) and item.node == "orchestrator_node"
                   for item in sends):
            orchestrator_claimed.clear()
        return command

    initial_skills = loaded_skills or {}

    def init_skills_node(state:WorkflowState):
        # Seed a fresh thread with the configured skill index.  A thread that
        # already has skills (for example after loading one from the file
        # system) keeps them across user turns.
        if state.get("loaded_skills") or not initial_skills:
            return Command(goto="orchestrator_node")
        return Command(goto="orchestrator_node",
                       update={"loaded_skills": dict(initial_skills)})

    graph = StateGraph(WorkflowState)
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

 