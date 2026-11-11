from langchain.chat_models import BaseChatModel
from langgraph.types import Checkpointer
from langchain_core.tools.base import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt


from .models import StepNodeState, InterruptState, RawStepResult
from .init_tools import get_tool_node, TOOLS_BY_NAME
from typing import List, Dict, Any
import httpx, json, os

def create_parse_user_reply_util(model:BaseChatModel):
    def parse_user_reply(tool:BaseTool,
                        current_args: Dict[str, Any], 
                        user_text: str) -> Dict[str, Any]:
        """Turn a free-form user reply into concrete tool args with an LLM.

        The user may answer a whole sentence ("a is 1.5 and b is 2.5"), so we ask
        the parser model to fill the tool's own args schema. Values the user did
        not mention come back as None and keep the model's current guess.
        """
        schema = tool.args_schema.model_json_schema() if tool.args_schema else {}
        structured = model.with_structured_output(tool.args_schema)
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
        merged = dict(current_args)
        for key, value in parsed.model_dump().items():
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
    def interrupt_node(state:InterruptState):
        response = interrupt(state)
        # Allow both {"payload": {...}} (the documented shape) and a bare
        # payload dict, so users and the job manager can resume directly.
        response_payload = response.get("payload", response) if isinstance(response, dict) else {}
        if state["kind"] == "AskForArgs":
            # 对于"AskForArgs" 需要payload里有一个command字段
            command = response_payload.get("command")
            message = response_payload.get("message")
            tool_called = TOOLS_BY_NAME[state["tool_name"]]

            msg = f"Ask the user to review and edit args.\n"

            if command == "use-default": # 采用现有的默认值
                try:
                    msg += f"User chose to use default args, run the tool by default args.\n"
                    tool_result = tool_called.invoke(input = state["payload"]["current_args"])
                    returned_messages = [
                        ToolMessage(content=str(tool_result), tool_call_id=state["tool_call_id"], name=state["tool_name"])
                    ]
                except Exception as e:
                    returned_messages = [
                        ToolMessage(status="error", content=f"{msg}, Error: {e}",
                                    tool_call_id=state["tool_call_id"], name=state["tool_name"])
                    ]
            elif command == "edit":
                if not message:
                    raise RuntimeError("AskForArgs interrupt, user did not provide anything to edit the parameters.")

                try:
                    msg += f"Try to parse the args from user's response: {message} \n"
                    merged_args = parse_user_reply(
                        tool = tool_called,
                        current_args = state["payload"]["current_args"],
                        user_text = message)
                    
                    try:
                        result = tool_called.invoke(merged_args)
                        returned_messages = [
                            ToolMessage(
                                content=f"{msg}parsed args: {merged_args}.\nCall the tool by parsed args, result: {result}",
                                tool_call_id = state["tool_call_id"],
                                name = state["tool_name"]
                            )
                        ]
                    except Exception as e:
                        returned_messages = [
                            ToolMessage(status="error",
                                        content=f"{msg}parsed args: {merged_args}.\nCall the tool by parsed args, error: {e}",
                                        tool_call_id = state["tool_call_id"],
                                        name = state["tool_name"])
                        ] 

                except Exception as e:
                    returned_messages = [
                        ToolMessage(status="error", 
                                    content=f"{msg}Failed to parse args from user's response. Error: {e}\n Toolcall is cancelled.",
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

