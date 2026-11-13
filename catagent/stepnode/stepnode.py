from langchain.chat_models import BaseChatModel
from langgraph.types import Checkpointer
from langchain_core.tools.base import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from langchain.tools import ToolRuntime
from langgraph.prebuilt import ToolNode

from .models import StepNodeState, InterruptState, RawStepResult
from .init_tools import get_tool_node, TOOLS_BY_NAME
from typing import List, Dict, Any
import copy, httpx, json, logging, os

logger = logging.getLogger(__name__)


def _drop_defaults(node):
    """Recursively remove JSON-schema `default` keys (in place)."""
    if isinstance(node, dict):
        node.pop("default", None)
        for value in node.values():
            _drop_defaults(value)
    elif isinstance(node, list):
        for value in node:
            _drop_defaults(value)


def _optional_args_schema(tool:BaseTool) -> Dict[str, Any]:
    """Build an extraction schema in which every tool argument is optional.

    The tool's own `tool_call_schema` can declare non-nullable fields with
    defaults (e.g. `ntimes: int = -1`). The extracting model legitimately
    returns `null` for arguments the user did not mention, which would fail
    Pydantic validation of the tool schema. Wrapping every field in
    `anyOf: [<type>, null]` and requiring nothing lets the parser report only
    the values the user actually provided.
    """
    schema = tool.tool_call_schema.model_json_schema() if tool.tool_call_schema else {}
    properties = {}
    for name, prop in (schema.get("properties") or {}).items():
        prop = copy.deepcopy(prop)
        _drop_defaults(prop)
        properties[name] = {
            "anyOf": [prop, {"type": "null"}],
            "description": prop.get("description")
                           or f"Argument `{name}`. Use null if the user did not specify it.",
        }
    return {
        "title": f"{tool.name}_optional_args",
        "description": "Only include arguments the user explicitly specified; use null for everything else.",
        "type": "object",
        "properties": properties,
        "required": [],
    }


def create_parse_user_reply_util(model:BaseChatModel):
    def parse_user_reply(tool:BaseTool,
                        current_args: Dict[str, Any], 
                        user_text: str) -> Dict[str, Any]:
        """Turn a free-form user reply into concrete tool args with an LLM.

        The user may answer a whole sentence ("a is 1.5 and b is 2.5"), so we ask
        the parser model to fill the tool's own args schema. Values the user did
        not mention come back as None and keep the model's current guess.
        """
        # Use the model-facing schema: `args_schema` contains injected args
        # such as `runtime: ToolRuntime`, which has no JSON schema. Every field
        # is nullable so the model can omit arguments the user did not mention.
        schema = _optional_args_schema(tool)
        structured = model.with_structured_output(schema)
        parsed = structured.invoke(
            [
                SystemMessage(
                    content=(
                        "You convert a user's free-form reply into arguments for a "
                        "tool call. Extract only the values the user explicitly "
                        "provided; use null for anything the user did not specify."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Tool: {tool.name} — {tool.description}\n"
                        f"Schema: {json.dumps(schema)}\n"
                        f"Current arguments: {json.dumps(current_args)}\n"
                        f"User reply: {user_text}"
                    )
                ),
            ]
        )
        # `with_structured_output` returns a dict for a JSON-schema input, but
        # support Pydantic output too in case the schema handling changes.
        parsed_args = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed or {})
        # Drop `None` values from the current args: they mean "not specified",
        # so the tool must see its own defaults instead of an explicit None.
        merged = {key: value for key, value in current_args.items() if value is not None}
        for key, value in parsed_args.items():
            if value is not None:
                merged[key] = value
        return merged

    return parse_user_reply


def create_model_and_tool_node(model:BaseChatModel, 
                      system_skill_pathes:List[str], 
                      job_manager_url:str,
                      http_client:httpx.AsyncClient):
    
    static_tools, tool_node = get_tool_node(model, job_manager_url, http_client)
    system_prompt = ""
    for skill_path in system_skill_pathes:
        with open(skill_path, mode="r") as f:
            content = ''.join(f.readlines())
            system_prompt += content+'\n'
    
    def model_node(state:StepNodeState):
        """Agent node — what middleware's wrap_model_call did, spelled out."""
        accessible = state.get("accessible_tool_names", {})
        existing_names = {t.name for t in static_tools}
        extra = [TOOLS_BY_NAME[name] for name in accessible if name not in existing_names]
        tools = [*static_tools, *extra]
        response = model.bind_tools(tools).invoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    return model_node, tool_node


# 条件分支，是否走interrupt_node
def has_interrupts(state:StepNodeState):
    interrupt_list = []
    state_keys = ["kind", "message", "tool_call_id", "tool_name"]
    for key in (state.get("interrupts") or {}).keys():
        obj = state["interrupts"][key]

        payload = dict()
        for obj_key in obj.keys():
            if not obj_key in state_keys:
                payload[obj_key] = obj[obj_key]

        args = dict()
        for state_key in state_keys:
            args[state_key] = obj[state_key]
        args["payload"] = payload

        interrupt_list.append(Send(node="interrupt_node", arg=args))

    return interrupt_list if len(interrupt_list) > 0 else "model_node"


def create_summary_node(model:BaseChatModel):
    def summary_node(state:StepNodeState):
        report_md = os.path.join(state.get("workdir", ""), "report.md")
        if not os.path.isfile(report_md):
            return {"step_result": RawStepResult(
                status="error",
                summary=f"{report_md} was not created; the step result is unknown.")}

        with open(report_md, mode="r") as f:
            content = ''.join(f.readlines())

        content = content.strip()
        if len(content) == 0:
            return {"step_result": RawStepResult(
                status="error",
                summary=f"{report_md} is empty; the step result is unknown.")}
        else:
            system_prompt = "Given a report on the execution results, you extract the `status` (success or error), summary and error messages, and description of important output artifacts from the results."
            result = model.with_structured_output(RawStepResult).invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=content)])
            return {"step_result": result}

    return summary_node

def create_interrupt_node(model:BaseChatModel):
    parse_user_reply = create_parse_user_reply_util(model)
    # 这个节点接在tool_node之后
    async def interrupt_node(state:InterruptState,  config:RunnableConfig, runtime:Runtime):
        response = interrupt(state)
        # Allow both {"payload": {...}} (the documented shape) and a bare
        # payload dict, so users and the job manager can resume directly.
        response_payload = response.get("payload", response) if isinstance(response, dict) else {}
        if state["kind"] == "AskForArgs":
            # 对于"AskForArgs" 需要payload里有一个command字段
            command = response_payload.get("command")
            message = response_payload.get("message")
            tool_called = TOOLS_BY_NAME[state["tool_name"]]

            # 保留旧的 ToolRuntime 构造方式作为参考。
            # 现在改用 execute_tool 里的 ToolNode 执行：把原始 ToolCall 传给
            # ToolNode 会让它从 graph channels 里恢复 runtime.state，因此工具
            # 能看到真实的 graph state，不需要手动构造 ToolRuntime。
            # tool_runtime = ToolRuntime(state = dict(),
            #                             context = runtime.context,
            #                             config = config,
            #                             stream_writer=runtime.stream_writer,
            #                             tool_call_id = state["tool_call_id"],
            #                             store = runtime.store,
            #                             tools=[tool_called],
            #                             execution_info=runtime.execution_info,
            #                             server_info=runtime.server_info)

            async def execute_tool(tool:BaseTool, args:Dict[str, Any]) -> List[ToolMessage]:
                """Execute a confirmed tool through a clean ToolNode.

                Passing the raw ToolCall (instead of a synthetic state dict) lets
                ToolNode hydrate `runtime.state` from the graph channels, so the
                tool sees the real graph state.
                """
                # `None` means "not specified". Dropping those keys lets the
                # tool's own defaults apply (e.g. ntimes=-1) instead of passing
                # an invalid None for a non-nullable parameter.
                args = {key: value for key, value in args.items() if value is not None}
                tool_call = {
                    "name": tool.name,
                    "args": args,
                    "id": state["tool_call_id"],
                    "type": "tool_call",
                }
                try:
                    output = await ToolNode([tool]).ainvoke([tool_call], config)
                except Exception:
                    logger.exception("Failed to execute tool %s (tool_call_id=%s)",
                                     tool.name, state["tool_call_id"])
                    raise
                if isinstance(output, dict):
                    return list(output.get("messages", []))
                # A tool that returns Command(s) needs its update merged into the
                # graph, which this resume node cannot do. Fail loudly instead
                # of silently dropping the update.
                raise RuntimeError("The tool returned a Command, which is not supported in interrupt_node.")

            def prefix_tool_messages(messages:List[ToolMessage], prefix:str) -> List[ToolMessage]:
                if not messages:
                    return [ToolMessage(status="error", content=prefix,
                                        tool_call_id=state["tool_call_id"], name=state["tool_name"])]
                last = messages[-1]
                messages[-1] = last.model_copy(update={"content": f"{prefix}{last.content}"})
                return messages

            msg = f"Ask the user to review and edit args.\n"

            if command == "use-default": # 采用现有的默认值
                try:
                    msg += f"User chose to use default args, run the tool by default args.\n"

                    returned_messages = prefix_tool_messages(
                        await execute_tool(tool_called, state["payload"]["current_args"]), msg)
                except Exception as e:
                    returned_messages = [
                        ToolMessage(status="error",
                                    content=f"{msg}, Error ({type(e).__name__}): {e}",
                                    tool_call_id=state["tool_call_id"], name=state["tool_name"])
                    ]
            elif command == "edit":
                if not message:
                    returned_messages = [
                        ToolMessage(status="error", 
                                    content="The user chose to edit the args, but input nothing.",
                                    tool_call_id=state["tool_call_id"], name=state["tool_name"])
                    ]
                else:
                    try:
                        msg += f"Try to parse the args from user's response: {message} \n"
                        merged_args = parse_user_reply(
                            tool = tool_called,
                            current_args = state["payload"]["current_args"],
                            user_text = message)

                        try:
                            returned_messages = prefix_tool_messages(
                                await execute_tool(tool_called, merged_args),
                                f"{msg}parsed args: {merged_args}.\nCall the tool by parsed args, result: ")
                        except Exception as e:
                            returned_messages = [
                                ToolMessage(status="error",
                                            content=f"{msg}parsed args: {merged_args}.\nCall the tool by parsed args, error ({type(e).__name__}): {e}",
                                            tool_call_id = state["tool_call_id"],
                                            name = state["tool_name"])
                            ]

                    except Exception as e:
                        returned_messages = [
                            ToolMessage(status="error",
                                        content=f"{msg}Failed to parse args from user's response. Error ({type(e).__name__}): {e}\n Toolcall is cancelled.",
                                        tool_call_id = state["tool_call_id"],
                                        name = state["tool_name"])
                        ]
                    
        elif state["kind"] == "JobSubmission":
            local_dir = response_payload.get("local_dir", state.get("local_dir", ""))
            returned_messages = [
                ToolMessage(
                    status = response_payload.get("status", "error"),
                    content = f"The job in {local_dir} returned from the job manager.\n{response_payload.get('error_msg', '')}",
                    tool_call_id = state["tool_call_id"],
                    name = state["tool_name"]
                )]

        elif state["kind"] == "AgentInterrupt":
            result = response_payload.get("result", response_payload)
            returned_messages = [
                ToolMessage(content=f"Paused with message: \"{state['message']}\"\nUser's response: {result}.",
                            tool_call_id = state["tool_call_id"],
                            name = state["tool_name"])
                ]

        return Command(update={ 
            "interrupts":{state["tool_call_id"]: None },
            "messages": returned_messages
            })

    return interrupt_node

def route_after_model(state: StepNodeState) :
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tool_node"
    return "summary_node"

    
def create_stepnode_agent(model:BaseChatModel,
                          checkpointer:Checkpointer,
                          job_manager_url:str,
                          http_client:httpx.AsyncClient,
                          system_skill_pathes:List[str]):

    model_node, tool_node = create_model_and_tool_node(model, 
                                                       system_skill_pathes, 
                                                       job_manager_url, 
                                                       http_client)
    
    interrupt_node = create_interrupt_node(model)
    summary_node = create_summary_node(model)

    graph = StateGraph(StepNodeState)
    graph.add_node("model_node", model_node)
    graph.add_node("tool_node", tool_node)
    graph.add_node("interrupt_node", interrupt_node)
    graph.add_node("summary_node", summary_node)

    graph.add_edge(START, "model_node")
    graph.add_conditional_edges("model_node", route_after_model)
    graph.add_conditional_edges("tool_node", has_interrupts)
    graph.add_edge("interrupt_node", "model_node")
    graph.add_edge("summary_node", END)

    agent = graph.compile(checkpointer=checkpointer)
    return agent 

