"""Real-LLM smoke test for the orchestrator.

The deterministic append/replace details live in
check/test_register_raw_plan.py.  This script checks that the orchestrator can
actually read the fake skill, call the planner, and register an append plan
through the real tool path.

Run from the repository root:

    python -m check.test_agent_orchestrator
"""
import asyncio

import httpx
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from catagent.workflow.orchestrator import create_orchestrator_agent
from check.common import load_mock_skills, make_model, REPO_ROOT


SYSTEM_SKILLS = REPO_ROOT / "catagent" / "system_skills"


def system_skill_paths():
    return [str(SYSTEM_SKILLS / name) for name in (
        "orchestrator_prompt.md",
        "bash_tips.md",
        "file_and_state_management.md",
    )]


async def run_orchestrator(agent, input_data, config):
    """Run until the agent stops producing interrupts, auto-answering them."""
    current = input_data
    while True:
        pending = {}
        async for chunk in agent.astream(
                input=current,
                config=config,
                stream_mode=["custom", "updates"],
                version="v2"):
            if not isinstance(chunk, dict) or chunk.get("type") != "updates":
                continue
            data = chunk.get("data") or {}
            interrupts = data.get("__interrupt__")
            if not interrupts:
                continue
            if not isinstance(interrupts, (list, tuple)):
                interrupts = (interrupts,)
            for item in interrupts:
                value = getattr(item, "value", item)
                pending[getattr(item, "id", "interrupt")] = value
                print(f"[interrupt] {value}")

        if not pending:
            break

        # The fake task does not need any user-specific parameters.
        answers = {}
        for key, value in pending.items():
            if isinstance(value, dict) and value.get("kind") == "AgentInterrupt":
                answers[key] = {"payload": {"result": "Use the defaults from the skill and continue."}}
            elif isinstance(value, dict) and value.get("kind") == "AskForArgs":
                answers[key] = {"payload": {"command": "use-default"}}
            else:
                answers[key] = "Proceed with the plan using the skill defaults."
        from langgraph.types import Command
        current = Command(resume=answers)


def mock_transport() -> httpx.MockTransport:
    """Answer the /providers call without starting the real jobmanager."""
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/providers":
            return httpx.Response(200, json=[{
                "name": "local-mock",
                "processing_units": ["cpu"],
                "job_envs": {
                    "mock-compute": [
                        "mock-prepare", "mock-relax", "mock-analyze", "mock-report",
                    ],
                },
                "remote_files": [],
            }])
        return httpx.Response(404, json={"detail": "not found"})
    return httpx.MockTransport(handler)


async def main() -> None:
    model = make_model()
    async with httpx.AsyncClient(transport=mock_transport()) as client:
        agent = create_orchestrator_agent(
            model=model,
            system_skill_path=system_skill_paths(),
            job_manager_url="http://mock-jobmanager",
            http_client=client,
            checkpointer=InMemorySaver(),
        )

        await _run_plan_test(agent)


async def _run_plan_test(agent) -> None:
    config = {"configurable": {"thread_id": "orchestrator-smoke"}}
    request = (
        "Use the `mock-compute` skill to plan a fake computation. "
        "Make a 4-step plan: prepare the inputs, relax sample A, relax sample B "
        "in parallel with sample A, then analyze both results. "
        "Use the planner tool and register the plan with register_raw_plan in "
        "append mode. Do not execute the steps; only produce and register the plan. "
        "Do not ask the user questions; use the default settings from the skill."
    )
    await run_orchestrator(
        agent,
        {
            "messages": [HumanMessage(content=request)],
            "loaded_skills": load_mock_skills(),
            "steps": {},
            "raw_plan_dict": {},
            "phase": "user",
        },
        config,
    )

    state = await agent.aget_state(config)
    steps = state.values.get("steps") or {}
    if len(steps) < 4:
        raise AssertionError(
            f"orchestrator registered only {len(steps)} steps: {sorted(steps)}\n"
            f"messages:\n" + "\n".join(
                f"{type(m).__name__}: {getattr(m, 'content', '')!r}"
                for m in state.values.get("messages", [])
            )
        )

    print("\nRegistered steps:")
    for step_id in sorted(steps):
        step = steps[step_id]
        print(f"  {step_id} status={step.status} "
              f"depend_on={sorted(step.depend_on)} "
              f"required_by={sorted(step.required_by)}")

    print("\norchestrator append plan test: OK")


if __name__ == "__main__":
    asyncio.run(main())
