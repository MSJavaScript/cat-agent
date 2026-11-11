from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing import List
from pydantic import Field

def _tool_names(tools) -> List[str]:
    return [t["function"]["name"] if isinstance(t, dict) else t.name for t in tools]

class ScriptedModel(BaseChatModel):
    """Chat model that replays a scripted list of responses.

    Builder helpers append to the script; the agent pops one response per
    model turn. Logs record what the agent actually did with the model.
    """
    responses: List[AIMessage] = Field(default_factory=list)
    bind_log: List[List[str]] = Field(default_factory=list)    # every bind_tools() call
    invoke_log: List[dict] = Field(default_factory=list)       # every _generate() call

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def say(self, content: str) -> "ScriptedModel":
        """Script a plain (final) answer."""
        self.responses.append(AIMessage(content=content))
        return self

    def call_tool(self, name: str, args: dict, id: str | None = None) -> "ScriptedModel":
        """Script a tool call — the agent will execute the REAL tool."""
        self.responses.append(AIMessage(content="", tool_calls=[{
            "name": name, "args": args,
            "id": id or f"call_{len(self.responses) + 1}", "type": "tool_call",
        }]))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.invoke_log.append({
            "n_messages": len(messages),
            "kwargs_tools": _tool_names(kwargs.get("tools", [])),
        })
        if not self.responses:
            raise RuntimeError(
                f"ScriptedModel ran out of scripted responses after "
                f"{len(self.invoke_log)} model calls — add more .say()/.call_tool()."
            )
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # direct async path, so ainvoke() doesn't hop to a thread executor
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        # record what the agent binds (and WHEN — once or per turn), then
        # return self, i.e. the scripted model ignores tool schemas entirely.
        self.bind_log.append(_tool_names(tools))
        return self
