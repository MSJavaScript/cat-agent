"""Regression test for sibling-interrupt collection in the outer workflow.

Two job results can answer two interrupts of the *same* StepNode (or of the
orchestrator) in one outer superstep.  Resuming the inner graph once per answer
started two concurrent runs on one inner checkpoint; each run re-raised the
other answer's interrupt, the outer graph parked on those new ids, and the
already-delivered job results could never resume them again.

The fix is split in two places:

* ``interrupt_node`` records its answer in ``pending_resumes`` and routes to
  ``resume_collector`` instead of dispatching its own resume;
* ``resume_collector_node`` unions the answers of one inner graph and emits a
  single ``Send`` with the complete ``{inner_interrupt_id: answer}`` map.

This test wires the real nodes into a small outer graph.  It checks that the
answers may arrive in separate invocations and are still merged before the
single resume.  It also checks that two *different* steps are not merged.

Run from the repository root:

    python -m check.test_agent_multi_interrupt
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send

from catagent.agent import interrupt_node, resume_collector_node
from catagent.workflow.liveness import clear_step_run
from catagent.workflow.models import pending_resume_reducer


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name}: {detail}")
    print(f"  ok: {name}")


class CollectorState(TypedDict, total=False):
    pending_resumes: Annotated[
        Dict[str, Dict[str, Any]], pending_resume_reducer
    ]
    resumes: Annotated[List[Dict[str, str]], operator.add]


def build_graph(
    source: str,
    steps: List[str],
    interrupts_per_step: int = 2,
):
    """Build an outer graph with the real ``interrupt_node``/collector nodes.

    ``steps`` is the list of StepNode ids.  For the orchestrator there is one
    group only.  ``resumes`` records exactly what each resumed inner graph
    received, so the test can count dispatches.
    """
    target = "step_node" if source == "stepnode" else "orchestrator_node"

    def dispatch(state):
        sends = []
        for step_id in steps:
            for index in range(interrupts_per_step):
                sends.append(
                    Send(
                        "interrupt_node",
                        {
                            "source": source,
                            "step_id": step_id,
                            "workdir": f"/tmp/{step_id or 'orchestrator'}",
                            "project_workdir": "/tmp",
                            "interrupt_ori_id": f"inner-{step_id or 'orchestrator'}-{index}",
                            "interrupt_payload": {
                                "kind": "Test",
                                "question": f"{step_id or 'orchestrator'}-{index}",
                            },
                        },
                    )
                )
        return Command(goto=sends)

    def resume_target(state):
        return Command(
            goto=END,
            update={"resumes": [dict(state.get("interrupt_resume") or {})]},
        )

    graph = StateGraph(CollectorState)
    graph.add_node("dispatch", dispatch)
    # The real nodes are annotated with their production TypedDicts, which
    # redeclare some channels with different reducers.  Pin them to this small
    # test schema; at runtime a node receives the plain state dict.
    graph.add_node("interrupt_node", interrupt_node, input_schema=CollectorState)
    graph.add_node("resume_collector", resume_collector_node, input_schema=CollectorState)
    graph.add_node("step_node", resume_target)
    graph.add_node("orchestrator_node", resume_target)
    graph.add_edge(START, "dispatch")
    return graph.compile(checkpointer=InMemorySaver())


def pending_ids(graph, config) -> List[str]:
    snapshot = graph.get_state(config)
    ids = [
        str(item.id)
        for task in (snapshot.tasks or ())
        for item in (task.interrupts or ())
    ]
    if not ids:
        ids = [str(item.id) for item in (snapshot.interrupts or ())]
    return ids


async def test_separate_answers_are_merged() -> None:
    """g1 and g2 arrive in two invocations; the step is resumed once."""
    step_id = "plan-0:step-1:two-jobs"
    graph = build_graph("stepnode", [step_id], interrupts_per_step=2)
    thread_id = "multi-interrupt-separate"
    config = {"configurable": {"thread_id": thread_id}}

    await graph.ainvoke({}, config)
    pending = pending_ids(graph, config)
    check("both interrupts are pending", len(pending) == 2, f"{pending}")

    # First job result: only one outer interrupt is answered.
    await graph.ainvoke(
        Command(resume={pending[0]: "result-0"}), config
    )
    snapshot = graph.get_state(config)
    check(
        "first callback does not resume the inner graph yet",
        not snapshot.values.get("resumes"),
        f"resumes={snapshot.values.get('resumes')}",
    )
    remaining = pending_ids(graph, config)
    check("sibling interrupt is still pending", remaining == [pending[1]], f"{remaining}")

    # Second job result: the collector now merges both answers.
    await graph.ainvoke(
        Command(resume={pending[1]: "result-1"}), config
    )
    snapshot = graph.get_state(config)
    resumes = snapshot.values.get("resumes") or []
    check("inner graph resumed exactly once", len(resumes) == 1, f"{resumes}")
    answers = resumes[0] if resumes else {}
    check(
        "the single resume carries both answers",
        len(answers) == 2 and set(answers.values()) == {"result-0", "result-1"},
        f"{answers}",
    )
    check("outer graph is no longer paused", not pending_ids(graph, config))
    check("outer graph finished", snapshot.next == (), f"next={snapshot.next}")

    clear_step_run(thread_id, step_id)


async def test_different_steps_are_not_merged() -> None:
    """Two steps answered together keep one resume each."""
    steps = ["plan-0:step-1:alpha", "plan-0:step-1:beta"]
    graph = build_graph("stepnode", steps, interrupts_per_step=2)
    thread_id = "multi-interrupt-two-steps"
    config = {"configurable": {"thread_id": thread_id}}

    await graph.ainvoke({}, config)
    pending = pending_ids(graph, config)
    check("four interrupts are pending", len(pending) == 4, f"{pending}")

    await graph.ainvoke(
        Command(
            resume={interrupt_id: f"answer-{index}" for index, interrupt_id in enumerate(pending)}
        ),
        config,
    )
    snapshot = graph.get_state(config)
    resumes = snapshot.values.get("resumes") or []
    check("each step is resumed once", len(resumes) == 2, f"{resumes}")
    check(
        "each step receives only its own two answers",
        all(len(answers) == 2 for answers in resumes),
        f"{resumes}",
    )
    check("outer graph finished", snapshot.next == (), f"next={snapshot.next}")

    for step_id in steps:
        clear_step_run(thread_id, step_id)


async def test_orchestrator_pause_to_ask_is_merged() -> None:
    """Several pause_to_ask answers resume the orchestrator only once."""
    graph = build_graph("orchestrator", [""], interrupts_per_step=3)
    thread_id = "multi-interrupt-orchestrator"
    config = {"configurable": {"thread_id": thread_id}}

    await graph.ainvoke({}, config)
    pending = pending_ids(graph, config)
    check("three orchestrator interrupts are pending", len(pending) == 3, f"{pending}")

    # Answer the first one alone: nothing runs while siblings are pending.
    await graph.ainvoke(Command(resume={pending[0]: "answer-0"}), config)
    check(
        "partial orchestrator answer waits",
        not (graph.get_state(config).values or {}).get("resumes"),
        "the orchestrator ran before all pause_to_ask answers arrived",
    )

    await graph.ainvoke(
        Command(resume={interrupt_id: f"answer-{index}" for index, interrupt_id in enumerate(pending[1:], start=1)}),
        config,
    )
    snapshot = graph.get_state(config)
    resumes = snapshot.values.get("resumes") or []
    check("orchestrator resumed exactly once", len(resumes) == 1, f"{resumes}")
    check(
        "the resume carries all three answers",
        len(resumes[0]) == 3 if resumes else False,
        f"{resumes}",
    )
    check("outer graph finished", snapshot.next == (), f"next={snapshot.next}")


async def main() -> None:
    await test_separate_answers_are_merged()
    await test_different_steps_are_not_merged()
    await test_orchestrator_pause_to_ask_is_merged()
    print("\nMULTI-INTERRUPT RESUME TEST: OK")


if __name__ == "__main__":
    asyncio.run(main())
