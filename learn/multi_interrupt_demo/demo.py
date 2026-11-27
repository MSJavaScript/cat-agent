"""Several interrupts in one superstep, and why they must be resumed together.

Run from this directory:

    python demo.py

Four experiments, all dependency-free (no API key, no network):

1. Two parallel nodes call ``interrupt()``; B interrupts 5 s after A.
   Does one ``ainvoke()`` surface both, or only the first one?

2. Resume only one of the two.  Does LangGraph start a second parallel run,
   or does it keep the sibling pending?

3. The shape used by cat-agent: an outer graph forwards an inner graph's
   interrupts to the user/jobmanager, then resumes the inner graph when the
   answers arrive.  When the answers arrive separately, the outer graph
   advances only once, and the naive "one resume per answer" dispatches two
   concurrent runs on one inner checkpoint.  Compare with the collector used
   by ``resume_collector_node`` in ``src/catagent/agent.py``.

4. ``langchain.agents.create_agent`` with a model that emits three
   ``pause_to_ask`` calls in one turn.  This is the orchestrator case: one
   model turn can park three questions at the same time.

The takeaway is in the README; the short version is:

* one ``ainvoke`` returns *all* interrupts of the superstep;
* a partial resume leaves the sibling pending (no second run starts by
  itself), but the outer graph defers the resume until the whole superstep is
  answered, so separate answers become concurrent resumes;
* accumulate every answer of the same inner graph and resume it once with
  ``Command(resume={id1: v1, id2: v2, ...})``.
"""

from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain.agents import create_agent
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt


# ---------------------------------------------------------------------------
# 1 + 2: one graph, two parallel interrupt nodes
# ---------------------------------------------------------------------------


class ParallelState(TypedDict):
    seen: Annotated[list, operator.add]


def build_parallel_graph(delay_b: float = 0.0):
    """Two independent nodes, both call ``interrupt()`` in the same superstep."""

    async def ask_a(state):
        answer = interrupt({"question": "A"})
        return {"seen": [f"A={answer}"]}

    async def ask_b(state):
        if delay_b:
            # The superstep does not finish until this node finishes too.
            await asyncio.sleep(delay_b)
        answer = interrupt({"question": "B"})
        return {"seen": [f"B={answer}"]}

    graph = StateGraph(ParallelState)
    graph.add_node("ask_a", ask_a)
    graph.add_node("ask_b", ask_b)
    graph.add_edge(START, "ask_a")
    graph.add_edge(START, "ask_b")
    graph.add_edge("ask_a", END)
    graph.add_edge("ask_b", END)
    return graph.compile(checkpointer=InMemorySaver())


def pending_interrupts(graph, config) -> List[tuple[str, Any]]:
    return [
        (str(item.id), item.value)
        for task in (graph.get_state(config).tasks or ())
        for item in (task.interrupts or ())
    ]


async def experiment_1_one_invoke(delay_b: float) -> Dict[str, Any]:
    """Question 1: what does a single ainvoke() return?"""
    graph = build_parallel_graph(delay_b)
    config = {"configurable": {"thread_id": f"one-invoke-{delay_b}"}}
    result = await graph.ainvoke({"seen": []}, config)
    pending = pending_interrupts(graph, config)
    return {
        "delay_b": delay_b,
        "pending": pending,
        "seen": result.get("seen", []),
        "next": graph.get_state(config).next,
    }


async def experiment_2_partial_resume() -> Dict[str, Any]:
    """Question 2: resume one interrupt, then the other."""
    graph = build_parallel_graph(0.0)
    config = {"configurable": {"thread_id": "partial-resume"}}
    await graph.ainvoke({"seen": []}, config)

    pending = pending_interrupts(graph, config)
    first_id, second_id = pending[0][0], pending[1][0]

    after_first = await graph.ainvoke(
        Command(resume={first_id: "answer-A"}), config
    )
    first_state = graph.get_state(config)
    first_pending = pending_interrupts(graph, config)

    after_second = await graph.ainvoke(
        Command(resume={second_id: "answer-B"}), config
    )
    final_state = graph.get_state(config)

    return {
        "pending_before": pending,
        "after_first_seen": after_first.get("seen", []),
        "after_first_pending": first_pending,
        "after_first_next": first_state.next,
        "after_second_seen": after_second.get("seen", []),
        "final_pending": pending_interrupts(graph, config),
        "final_next": final_state.next,
    }


# ---------------------------------------------------------------------------
# 3: the cat-agent shape -- outer graph forwards inner interrupts
# ---------------------------------------------------------------------------


class FrozenReadSaver(InMemorySaver):
    """``InMemorySaver`` whose reads can be pinned to one snapshot.

    The production checkpointer is SQLite.  Two concurrent resumes both read
    the checkpoint before either writes, so both start from the same inner
    superstep.  ``InMemorySaver`` is fast enough that the second resume often
    reads the checkpoint the first one just wrote, which hides the race.  The
    freeze makes the load-before-write window deterministic so the failure is
    reproducible in-process.
    """

    def __init__(self) -> None:
        super().__init__()
        self.enabled = False
        self._frozen: Dict[tuple, Any] = {}

    def freeze(self) -> None:
        self.enabled = True
        self._frozen = {}

    def unfreeze(self) -> None:
        self.enabled = False
        self._frozen = {}

    async def aget_tuple(self, config):  # type: ignore[override]
        if not self.enabled:
            return await super().aget_tuple(config)
        key = (
            config["configurable"].get("thread_id"),
            config["configurable"].get("checkpoint_ns", ""),
        )
        if key not in self._frozen:
            self._frozen[key] = await super().aget_tuple(config)
        return self._frozen[key]


class InnerState(TypedDict):
    # The two jobs/questions of one StepNode turn.
    seen: Annotated[list, operator.add]


def build_inner_graph():
    async def ask_a(state):
        answer = interrupt({"question": "A"})
        return {"seen": [f"A={answer}"]}

    async def ask_b(state):
        answer = interrupt({"question": "B"})
        return {"seen": [f"B={answer}"]}

    graph = StateGraph(InnerState)
    graph.add_node("ask_a", ask_a)
    graph.add_node("ask_b", ask_b)
    graph.add_edge(START, "ask_a")
    graph.add_edge(START, "ask_b")
    graph.add_edge("ask_a", END)
    graph.add_edge("ask_b", END)
    return graph.compile(checkpointer=FrozenReadSaver())


def merge_pending(current, update):
    """Same idea as ``pending_resume_reducer`` in workflow/models.py."""
    result = {key: {"answers": dict(value["answers"])} for key, value in (current or {}).items()}
    for key, value in (update or {}).items():
        if value is None:
            result.pop(key, None)
            continue
        answers = dict(result.get(key, {}).get("answers") or {})
        answers.update(value.get("answers") or {})
        result[key] = {"answers": answers}
    return result


class OuterState(TypedDict, total=False):
    pending: Annotated[dict, merge_pending]
    log: Annotated[list, operator.add]


@dataclass
class OuterRun:
    mode: str
    outer_before: List[tuple[str, Any]] = field(default_factory=list)
    outer_after_a: List[tuple[str, Any]] = field(default_factory=list)
    outer_after_b: List[tuple[str, Any]] = field(default_factory=list)
    resume_calls: List[Dict[str, str]] = field(default_factory=list)
    inner_seen: List[str] = field(default_factory=list)
    inner_pending: List[str] = field(default_factory=list)
    outer_next: tuple = ()
    inner_resumes: int = 0


def build_outer_graph(mode: str):
    """``mode='naive'`` resumes once per answer; ``mode='collector'`` merges."""
    inner = build_inner_graph()
    inner_config = {"configurable": {"thread_id": "inner"}}
    resume_calls: List[Dict[str, str]] = []

    async def dispatch(state):
        # In cat-agent this is ``step_node`` starting the StepNode for the
        # first time; the inner graph stops on both job submissions at once.
        snapshot = await inner.aget_state(inner_config)
        pending = {
            str(item.id): item.value
            for task in (snapshot.tasks or ())
            for item in (task.interrupts or ())
        }
        if not pending and not (snapshot.values or {}).get("seen"):
            await inner.ainvoke({"seen": []}, inner_config)
            snapshot = await inner.aget_state(inner_config)
            pending = {
                str(item.id): item.value
                for task in (snapshot.tasks or ())
                for item in (task.interrupts or ())
            }
        if not pending:
            return Command(goto=END)
        return Command(
            goto=[
                Send("ask_outer", {"inner_id": iid, "payload": value})
                for iid, value in pending.items()
            ]
        )

    def ask_outer(state):
        # This is the outer ``interrupt_node``: it re-emits one inner question
        # and later receives the user's / jobmanager's answer.
        response = interrupt(state["payload"])
        if mode == "naive":
            return Command(
                goto=Send(
                    "resume_inner",
                    {"answers": {state["inner_id"]: response}},
                )
            )
        # Record the answer; the collector merges all siblings of this group.
        return Command(
            goto="collector",
            update={"pending": {"step": {"answers": {state["inner_id"]: response}}}},
        )

    def collector(state):
        groups = state.get("pending") or {}
        sends = []
        cleared = {}
        for key, group in groups.items():
            sends.append(Send("resume_inner", {"answers": dict(group["answers"])}))
            cleared[key] = None
        return Command(goto=sends, update={"pending": cleared})

    async def resume_inner(state):
        answers = dict(state["answers"])
        resume_calls.append(answers)
        still_pending: Dict[str, Any] = {}
        async for chunk in inner.astream(
            Command(resume=answers), inner_config, stream_mode="updates"
        ):
            # ``step_node`` forwards whatever interrupt the inner run ends on.
            if "__interrupt__" in (chunk or {}):
                for item in chunk["__interrupt__"]:
                    still_pending[str(item.id)] = item.value
        if still_pending:
            return Command(
                goto=[
                    Send("ask_outer", {"inner_id": iid, "payload": value})
                    for iid, value in still_pending.items()
                ]
            )
        return Command(goto=END, update={"log": ["step done"]})

    graph = StateGraph(OuterState)
    graph.add_node("dispatch", dispatch)
    graph.add_node("ask_outer", ask_outer)
    graph.add_node("resume_inner", resume_inner)
    if mode == "collector":
        graph.add_node("collector", collector)
    graph.add_edge(START, "dispatch")
    return graph.compile(checkpointer=InMemorySaver()), inner, inner_config, resume_calls


async def experiment_3_outer_inner(mode: str) -> OuterRun:
    """Two callbacks arrive at different times, like a real jobmanager.

    Both callbacks answer the *original* outer interrupt ids.  With the
    collector, the outer graph resumes the inner graph once with both answers.
    Without it, each answer dispatches its own resume; both run concurrently,
    each re-interrupts the sibling and the outer graph parks on *new*
    interrupt ids that were never registered with the jobmanager.
    """
    graph, inner, inner_config, resume_calls = build_outer_graph(mode)
    config = {"configurable": {"thread_id": f"outer-{mode}"}}

    await graph.ainvoke({}, config)
    run = OuterRun(mode=mode)
    run.outer_before = pending_interrupts(graph, config)
    id_a, id_b = run.outer_before[0][0], run.outer_before[1][0]

    # SQLite reads and writes are separate I/O operations: allow the two
    # resumes that the outer graph dispatches together to load the same
    # inner checkpoint.
    inner.checkpointer.freeze()

    # Callback for the first job arrives now.
    await graph.ainvoke(Command(resume={id_a: "result-A"}), config)
    run.outer_after_a = pending_interrupts(graph, config)

    # Callback for the second job arrives minutes later.
    await graph.ainvoke(Command(resume={id_b: "result-B"}), config)
    run.outer_after_b = pending_interrupts(graph, config)

    inner.checkpointer.unfreeze()

    snapshot = await inner.aget_state(inner_config)
    run.inner_seen = list(snapshot.values.get("seen") or [])
    run.inner_pending = [
        str(item.value.get("question"))
        for task in (snapshot.tasks or ())
        for item in (task.interrupts or ())
    ]
    run.resume_calls = resume_calls
    run.inner_resumes = len(resume_calls)
    run.outer_next = graph.get_state(config).next
    return run


# ---------------------------------------------------------------------------
# 4: the orchestrator case -- several pause_to_ask calls in one model turn
# ---------------------------------------------------------------------------


class FakeToolModel(BaseChatModel):
    """First turn: emit N ``pause_to_ask`` tool calls.  Later: finish."""

    n_calls: int = 2

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(
        self,
        messages: Sequence[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        if messages and isinstance(messages[-1], ToolMessage):
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="done"))]
            )
        calls = [
            {
                "name": "pause_to_ask",
                "args": {"message": f"question {i}"},
                "id": f"call_{i}",
                "type": "tool_call",
            }
            for i in range(self.n_calls)
        ]
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=calls))]
        )


@tool
def pause_to_ask(runtime: ToolRuntime[Any], message: str):
    """Ask the user a question and wait for the answer."""
    answer = interrupt({"message": message, "tool_call_id": runtime.tool_call_id})
    return f"answer={answer}"


async def experiment_4_pause_to_ask(n_calls: int = 3) -> Dict[str, Any]:
    graph = create_agent(
        model=FakeToolModel(n_calls=n_calls),
        tools=[pause_to_ask],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": f"orchestrator-{n_calls}"}}
    await graph.ainvoke({"messages": [("user", "go")]}, config)

    pending = pending_interrupts(graph, config)
    await graph.ainvoke(
        Command(resume={iid: f"answer-{i}" for i, (iid, _) in enumerate(pending)}),
        config,
    )
    final = graph.get_state(config)
    tool_messages = [
        message.content
        for message in final.values.get("messages", [])
        if isinstance(message, ToolMessage)
    ]
    return {
        "pending_count": len(pending),
        "pending": pending,
        "tool_messages": tool_messages,
        "final_next": final.next,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def show_run(run: OuterRun) -> None:
    print(f"  outer paused before callbacks : {[(i, v['question']) for i, v in run.outer_before]}")
    print(f"  after callback A             : {[(i, v['question']) for i, v in run.outer_after_a]}")
    print(f"  after callback B             : {[(i, v['question']) for i, v in run.outer_after_b]}")
    print(f"  inner resume calls           : {run.inner_resumes}")
    for call in run.resume_calls:
        print(f"      answers={call}")
    print(f"  inner seen                   : {run.inner_seen}")
    print(f"  inner still pending          : {run.inner_pending}")
    print(f"  outer next                   : {run.outer_next}")


async def main() -> None:
    print("=" * 78)
    print("1. Two parallel interrupt nodes in one ainvoke()")
    for delay in (0.0, 5.0):
        result = await experiment_1_one_invoke(delay)
        label = "simultaneous" if not delay else f"B interrupts {delay:g}s after A"
        print(f"  [{label}]")
        print(f"    ainvoke returned after both nodes finished; seen={result['seen']}")
        print(f"    interrupts returned in one result: {len(result['pending'])}")
        for iid, value in result["pending"]:
            print(f"      {value['question']}: {iid}")
        assert len(result["pending"]) == 2

    print("=" * 78)
    print("2. Resume only one of the two answered interrupts")
    partial = await experiment_2_partial_resume()
    print(f"  before resume        : {[(i, v['question']) for i, v in partial['pending_before']]}")
    print(f"  after resuming A     : seen={partial['after_first_seen']}")
    print(f"                         still pending={[(i, v['question']) for i, v in partial['after_first_pending']]}")
    print(f"                         next={partial['after_first_next']}")
    print(f"  after resuming B too : seen={partial['after_second_seen']} next={partial['final_next']}")
    print("  -> a partial resume does not start a second run; the first answer is kept")
    print("     and the graph stays paused on the sibling until it is answered too.")

    print("=" * 78)
    print("3. cat-agent shape: outer graph forwards an inner graph's interrupts")
    print("   The two answers arrive at different times (callbacks are separate).")
    for mode in ("naive", "collector"):
        run = await experiment_3_outer_inner(mode)
        print(f"\n  mode={mode}")
        show_run(run)
        if mode == "naive":
            print("  note: this in-memory run happened to merge both inner writes, yet the")
            print("        outer graph still parked on the re-interrupts.  The production")
            print("        dump has the same outer shape: inner step 149 held both")
            print("        __resume__ writes and outer step 13 was paused on the two")
            print("        already-delivered interrupt ids.")

    print()
    print("  naive     : after both original callbacks the outer graph is paused on")
    print("              NEW interrupt ids (the sibling re-interrupt raised by each")
    print("              concurrent inner run).  The jobmanager marked the original")
    print("              results DELIVERED, so no callback will answer the new ids.")
    print("  collector : one inner resume with {A: result-A, B: result-B}; the step")
    print("              finishes and the outer graph reaches END.")

    print("=" * 78)
    print("4. Orchestrator: several pause_to_ask calls in one model turn")
    pause = await experiment_4_pause_to_ask(3)
    print(f"  one ainvoke surfaced {pause['pending_count']} pause_to_ask interrupts")
    print(f"  final next={pause['final_next']} tool replies={pause['tool_messages']}")
    print("  -> the orchestrator has exactly the same shape as the StepNode case.")

    print()
    print("SUMMARY")
    print("  * One ainvoke collects every interrupt of the superstep (even a 5s gap).")
    print("  * Resuming one interrupt alone keeps the sibling pending; no parallel run")
    print("    is created by that call.")
    print("  * But an outer graph defers all sibling resumes to one superstep, so")
    print("    'one resume per answer' becomes two concurrent runs on one checkpoint.")
    print("  * Merge the answers per inner graph and resume it once.")


if __name__ == "__main__":
    asyncio.run(main())
