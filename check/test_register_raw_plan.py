"""Deterministic test for register_raw_plan(append/replace).

Run from the repository root:

    python -m check.test_register_raw_plan

No LLM is called: a ScriptedModel drives a real LangGraph agent whose tools
are the real planner tools.  The test checks the actual Step objects stored in
the agent state.
"""

import asyncio
from typing import Annotated, Dict

from langchain.agents import AgentState, create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage

from check.scripted_model import ScriptedModel
from catagent.workflow.models import (
    RawPlan, RawStep, Step, SkillMetadata, dict_reducer,
)
from catagent.workflow.planner_tool import init_planner_tool


class PlannerTestState(AgentState):
    loaded_skills: Annotated[Dict[str, SkillMetadata], dict_reducer]
    steps: Annotated[Dict[str, Step], dict_reducer]
    raw_plan_dict: Annotated[Dict[str, RawPlan], dict_reducer]


def raw_step(name: str, content: str = "", **kw) -> RawStep:
    return RawStep(name=name, content=content or f"mock step {name}", **kw)


# rawplan-0: prepare -> (relax_a, relax_b) -> analyze
PLAN_0 = RawPlan(
    steps=[
        raw_step("prepare", "Create inputs for samples A and B."),
        raw_step("relax_a", "Relax sample A."),
        raw_step("relax_b", "Relax sample B."),
        raw_step("analyze", "Compare A and B."),
    ],
    topology=[(0, 1), (0, 2), (1, 3), (2, 3)],
)

# rawplan-1: replace relax_a + relax_b with revised relax steps.
PLAN_1 = RawPlan(
    steps=[
        raw_step("relax_a_v2", "Relax sample A with the revised settings."),
        raw_step("relax_b_v2", "Relax sample B with the revised settings."),
    ],
    topology=None,
)

# rawplan-2: replace both revised relax steps by one combined step (many-to-one).
PLAN_2 = RawPlan(
    steps=[
        raw_step("relax_both", "Relax samples A and B together."),
    ],
    topology=None,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


async def main() -> None:
    model = ScriptedModel()
    # First agent turn: append rawplan-0.
    model.call_tool(
        "register_raw_plan",
        {
            "raw_plan_id": "rawplan-0",
            "action": "append",
            "depend_on_external": [[], [], [], ["plan-9:step-0:external"]],
        },
        id="append-1",
    )
    model.say("append finished")

    # Second agent turn: replace the two relax steps with plan-1.
    model.call_tool(
        "register_raw_plan",
        {
            "raw_plan_id": "rawplan-1",
            "action": "replace",
            "raw_step_indices_group": [0, 1],
            "step_ids_group": [
                "plan-0:step-1:relax_a",
                "plan-0:step-2:relax_b",
            ],
        },
        id="replace-1",
    )
    model.say("replace finished")

    # Third agent turn: many-to-one replacement of the two revised steps.
    model.call_tool(
        "register_raw_plan",
        {
            "raw_plan_id": "rawplan-2",
            "action": "replace",
            "raw_step_indices_group": [0],
            "step_ids_group": [[
                "plan-1:step-0:relax_a_v2",
                "plan-1:step-1:relax_b_v2",
            ]],
        },
        id="replace-2",
    )
    model.say("many-to-one finished")

    agent = create_agent(
        model=model,
        state_schema=PlannerTestState,
        tools=init_planner_tool(model=model),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "register-plan-test"}}
    initial = {
        "messages": [],
        "loaded_skills": {},
        "steps": {},
        "raw_plan_dict": {
            "rawplan-0": PLAN_0,
            "rawplan-1": PLAN_1,
            "rawplan-2": PLAN_2,
        },
    }

    await agent.ainvoke(
        {**initial, "messages": [HumanMessage(content="register the first plan")]},
        config=config,
    )
    state = await agent.aget_state(config)
    steps = state.values["steps"]
    raw_plans = state.values["raw_plan_dict"]

    check("rawplan-0" not in raw_plans, "append must consume rawplan-0")
    check(set(steps) == {
        "plan-0:step-0:prepare",
        "plan-0:step-1:relax_a",
        "plan-0:step-2:relax_b",
        "plan-0:step-3:analyze",
    }, f"unexpected append steps: {sorted(steps)}")
    check(steps["plan-0:step-0:prepare"].depend_on == set(),
          "prepare should have no dependence")
    check(steps["plan-0:step-1:relax_a"].depend_on == {"plan-0:step-0:prepare"},
          "relax_a should depend on prepare")
    check(steps["plan-0:step-3:analyze"].depend_on == {
        "plan-0:step-1:relax_a", "plan-0:step-2:relax_b",
        "plan-9:step-0:external",
    }, "analyze should depend on both relax steps and the external step")

    await agent.ainvoke(
        {"messages": [HumanMessage(content="replace the relax steps")]},
        config=config,
    )
    state = await agent.aget_state(config)
    steps = state.values["steps"]
    check("rawplan-1" not in state.values["raw_plan_dict"],
          "replace must consume rawplan-1")
    check("plan-0:step-1:relax_a" not in steps, "old relax_a must be deleted")
    check("plan-0:step-2:relax_b" not in steps, "old relax_b must be deleted")
    check("plan-1:step-0:relax_a_v2" in steps, "new relax_a must exist")
    check("plan-1:step-1:relax_b_v2" in steps, "new relax_b must exist")
    check(steps["plan-1:step-0:relax_a_v2"].depend_on == {"plan-0:step-0:prepare"},
          "new relax_a should inherit prepare dependence")
    check(steps["plan-1:step-1:relax_b_v2"].depend_on == {"plan-0:step-0:prepare"},
          "new relax_b should inherit prepare dependence")
    check(steps["plan-0:step-3:analyze"].depend_on == {
        "plan-1:step-0:relax_a_v2",
        "plan-1:step-1:relax_b_v2",
        "plan-9:step-0:external",
    }, f"analyze dependence was not rewired: {steps['plan-0:step-3:analyze'].depend_on}")

    await agent.ainvoke(
        {"messages": [HumanMessage(content="merge the two relax steps")]},
        config=config,
    )
    state = await agent.aget_state(config)
    steps = state.values["steps"]
    check("plan-1:step-0:relax_a_v2" not in steps, "revised relax_a must be deleted")
    check("plan-1:step-1:relax_b_v2" not in steps, "revised relax_b must be deleted")
    new_id = "plan-2:step-0:relax_both"
    check(new_id in steps, "combined relax step must exist")
    check(steps[new_id].depend_on == {"plan-0:step-0:prepare"},
          f"combined step should depend on prepare, got {steps[new_id].depend_on}")
    check(steps["plan-0:step-3:analyze"].depend_on == {
        new_id, "plan-9:step-0:external",
    }, f"analyze should now depend on the combined step and its external "
       f"dependence, got {steps['plan-0:step-3:analyze'].depend_on}")

    print("register_raw_plan append/replace tests: OK")
    print("raw_plan_dict left:", list(state.values["raw_plan_dict"]))
    print("steps left:", sorted(steps))


if __name__ == "__main__":
    asyncio.run(main())
