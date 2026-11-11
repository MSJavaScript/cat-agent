"""Deterministic test for the planner tool.

The planner calls the LLM through ``with_structured_output(RawPlan)``.  This
test replaces that one model call with a fixed RawPlan, so no network access is
needed.  It checks that:

  * the tool stores the plan in ``raw_plan_dict``,
  * it tells the agent which ``rawplan-N`` id to pass to
    ``register_raw_plan`` (an earlier version omitted the id and the model
    guessed a wrong one),
  * ``get_raw_plan_content`` can read the stored plan.

Run from the repository root:

    python -m check.test_planner_tool
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated, Dict

from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from catagent.workflow.models import (
    RawPlan, RawStep, SkillMetadata, Step, dict_reducer,
)
from catagent.workflow.planner_tool import init_planner_tool
from check.scripted_model import ScriptedModel


PLAN = RawPlan(
    steps=[
        RawStep(name="prepare", content="Create the inputs."),
        RawStep(name="relax", content="Run the fake relaxation."),
        RawStep(name="analyze", content="Compare the results."),
    ],
    topology=[(0, 1), (1, 2)],
)


class FixedPlannerModel:
    """Stand-in for the LLM used only by the planner tool."""

    def __init__(self, plan: RawPlan) -> None:
        self.plan = plan
        self.calls = []

    def with_structured_output(self, schema):
        if schema is not RawPlan:
            raise AssertionError(f"unexpected structured output schema: {schema}")
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        return self.plan


class PlannerTestState(AgentState):
    loaded_skills: Annotated[Dict[str, SkillMetadata], dict_reducer]
    steps: Annotated[Dict[str, Step], dict_reducer]
    raw_plan_dict: Annotated[Dict[str, RawPlan], dict_reducer]


def make_skill(tmpdir: Path) -> SkillMetadata:
    skill_path = tmpdir / "skill.md"
    skill_path.write_text(
        "---\nname: mock-skill\ndescription: planner test skill\n---\n\n"
        "# Mock skill\n\nThis file only exists for the planner test.\n",
        encoding="utf-8",
    )
    return SkillMetadata(name="mock-skill",
                         path=str(skill_path),
                         summary="planner test skill",
                         references=[],
                         scripts=[])


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="planner_test_") as tmp:
        skill = make_skill(Path(tmp))
        planner_model = FixedPlannerModel(PLAN)
        agent_model = ScriptedModel()
        agent_model.call_tool(
            "planner",
            {
                "related_skill_names": ["mock-skill"],
                "task_content": "Plan a small fake computation.",
                "plan_steps_guidance": [
                    "Step 0: prepare the inputs.",
                    "Step 1: run the relaxation.",
                    "Step 2: analyze the result.",
                ],
                "task_context": "This is a deterministic test.",
            },
            id="planner-1",
        )
        agent_model.call_tool("get_raw_plan_content", {}, id="content-1")
        agent_model.say("planner test finished")

        agent = create_agent(
            model=agent_model,
            state_schema=PlannerTestState,
            tools=init_planner_tool(model=planner_model),
            checkpointer=InMemorySaver(),
        )
        config = {"configurable": {"thread_id": "planner-tool-test"}}
        await agent.ainvoke(
            {
                "messages": [HumanMessage(content="make a plan")],
                "loaded_skills": {skill["name"]: skill},
                "steps": {},
                "raw_plan_dict": {},
            },
            config=config,
        )

        state = await agent.aget_state(config)
        raw_plans = state.values.get("raw_plan_dict") or {}
        if list(raw_plans) != ["rawplan-0"]:
            raise AssertionError(f"unexpected raw plans: {list(raw_plans)}")

        stored = raw_plans["rawplan-0"]
        if [step.name for step in stored.steps] != ["prepare", "relax", "analyze"]:
            raise AssertionError(f"unexpected stored plan: {stored}")

        if state.values.get("steps"):
            raise AssertionError("the planner must not register steps by itself")

        if len(planner_model.calls) != 1:
            raise AssertionError(
                f"planner model should be called once, got {len(planner_model.calls)}"
            )

        tool_messages = [
            message for message in state.values.get("messages", [])
            if getattr(message, "name", None) == "planner"
        ]
        if not tool_messages:
            raise AssertionError("planner returned no ToolMessage")
        if "rawplan-0" not in str(tool_messages[-1].content):
            raise AssertionError(
                "the planner reply must tell the agent the raw plan id: "
                f"{tool_messages[-1].content!r}"
            )

        print("planner tool test: OK")
        print("stored raw plan id: rawplan-0")
        print("steps generated by plan:", [step.name for step in stored.steps])
        print("planner model calls:", len(planner_model.calls))


if __name__ == "__main__":
    asyncio.run(main())
