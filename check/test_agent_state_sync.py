"""Deterministic tests for outer-workflow <-> orchestrator state sync.

Run from the repository root:

    python -m check.test_agent_state_sync

No LLM is involved.  These tests cover the lost-update bug that made the
router re-issue the same preparation request: the inner orchestrator used to
return its whole `steps` copy, overwriting router-owned transitions, and a
queued request used to carry a stale step snapshot.  Both are checked here.
"""
from __future__ import annotations

import asyncio

from langgraph.types import Send

from catagent.agent import _steps_delta
from catagent.workflow.models import Step, dict_reducer
from catagent.workflow.router import orchestrator_request_queue


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def make_step(step_id: str, **overrides) -> Step:
    values = dict(step_id=step_id, status="pending",
                  preparation_stage="notyet", workdir="")
    values.update(overrides)
    return Step(**values)


def test_delta_is_empty_when_nothing_changed() -> None:
    a = make_step("plan-0:step-0:a", status="running", workdir="/w")
    baseline = {a.step_id: a.model_copy(deep=True)}
    current = {a.step_id: a.model_copy(deep=True)}
    check(_steps_delta(baseline, current) == {},
          f"unchanged steps produced a delta: {_steps_delta(baseline, current)}")


def test_delta_contains_only_changed_fields() -> None:
    old = make_step("plan-0:step-0:a", workdir="/w")
    new = old.model_copy(deep=True)
    new.preparation_stage = "done"
    delta = _steps_delta({old.step_id: old}, {new.step_id: new})
    check(delta == {old.step_id: {"preparation_stage": "done"}},
          f"unexpected delta: {delta}")


def test_delta_does_not_clobber_concurrent_outer_changes() -> None:
    # The orchestrator still saw preparation_stage=ongoing; meanwhile the
    # router has already advanced the outer step to done.  The delta must not
    # contain the stale field, so applying it leaves the outer change intact.
    baseline = make_step("plan-0:step-0:a", preparation_stage="ongoing",
                         workdir="/w")
    current = baseline.model_copy(deep=True)
    current.content = "prepared instructions"
    delta = _steps_delta({baseline.step_id: baseline},
                         {current.step_id: current})
    check("preparation_stage" not in delta[baseline.step_id],
          f"stale preparation stage was returned: {delta}")

    outer = baseline.model_copy(deep=True)
    outer.preparation_stage = "done"  # router advanced while agent ran
    merged = dict_reducer({outer.step_id: outer}, delta)
    check(merged[outer.step_id].preparation_stage == "done",
          "delta reverted a concurrent preparation transition")
    check(merged[outer.step_id].content == "prepared instructions",
          "delta lost the orchestrator change")


def test_delta_keeps_new_steps_as_objects() -> None:
    new = make_step("plan-0:step-9:new", content="do it")
    delta = _steps_delta({}, {new.step_id: new})
    check(isinstance(delta[new.step_id], Step),
          f"new step became a {type(delta[new.step_id])}")
    merged = dict_reducer({}, delta)
    check(isinstance(merged[new.step_id], Step),
          "new step was stored as a plain dict")


async def test_dispatch_rehydrates_stale_request_steps() -> None:
    """A queued preparation request must not be dispatched with stale fields."""
    current = make_step("plan-0:step-0:a", preparation_stage="done",
                        workdir="/prepared")
    stale = make_step("plan-0:step-0:a", preparation_stage="ongoing",
                      workdir="")
    state = {
        "finalization": False,
        "steps": {current.step_id: current},
        "orchestrator_requests": [{
            "phase": "preparation",
            "steps": {stale.step_id: stale},
            "messages": [],
        }],
    }
    command = await orchestrator_request_queue(state)
    send = command.goto
    check(isinstance(send, Send) and send.node == "orchestrator_node",
          f"unexpected goto: {send}")
    sent = send.arg["steps"][current.step_id]
    check(sent.preparation_stage == "done",
          f"stale preparation stage was dispatched: {sent.preparation_stage}")
    check(sent.workdir == "/prepared",
          f"stale workdir was dispatched: {sent.workdir!r}")


async def test_dispatch_does_not_mutate_queued_request() -> None:
    current = make_step("plan-0:step-0:a", preparation_stage="done")
    queued = {"phase": "preparation",
              "steps": {current.step_id: make_step(current.step_id)},
              "messages": []}
    state = {"finalization": False,
             "steps": {current.step_id: current},
             "orchestrator_requests": [queued]}
    await orchestrator_request_queue(state)
    check(state["orchestrator_requests"][0]["steps"][current.step_id]
          .preparation_stage == "notyet",
          "the queued request was mutated in place")


async def test_replanning_dispatch_has_full_baseline() -> None:
    """The replanning branch must also send the current steps as baseline."""
    done = make_step("plan-0:step-0:a", status="done",
                     preparation_stage="done", workdir="/w")
    state = {"finalization": False,
             "steps": {done.step_id: done},
             "orchestrator_requests": []}
    command = await orchestrator_request_queue(state)
    send = command.goto
    check(isinstance(send, Send) and send.node == "orchestrator_node",
          f"unexpected replanning goto: {send}")
    check(send.arg.get("phase") == "replanning",
          f"unexpected phase: {send.arg.get('phase')}")
    check(send.arg["steps"][done.step_id].status == "done",
          "replanning dispatch omitted the current step baseline")


async def main() -> None:
    tests = [
        ("delta is empty when unchanged", test_delta_is_empty_when_nothing_changed),
        ("delta contains only changed fields", test_delta_contains_only_changed_fields),
        ("delta preserves concurrent outer changes", test_delta_does_not_clobber_concurrent_outer_changes),
        ("delta keeps new steps as objects", test_delta_keeps_new_steps_as_objects),
        ("dispatch rehydrates stale steps", test_dispatch_rehydrates_stale_request_steps),
        ("dispatch does not mutate the queue", test_dispatch_does_not_mutate_queued_request),
        ("replanning dispatch has full baseline", test_replanning_dispatch_has_full_baseline),
    ]
    for name, function in tests:
        if asyncio.iscoroutinefunction(function):
            await function()
        else:
            function()
        print(f"  ok: {name}")
    print("\nAGENT STATE SYNC TESTS: OK")


if __name__ == "__main__":
    asyncio.run(main())
