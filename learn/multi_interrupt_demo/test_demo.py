"""Headless checks for the multi-interrupt demo; no network or API key.

    cd learn/multi_interrupt_demo
    python test_demo.py
"""

from __future__ import annotations

import asyncio

from demo import (
    experiment_1_one_invoke,
    experiment_2_partial_resume,
    experiment_3_outer_inner,
    experiment_4_pause_to_ask,
)


async def test_one_invoke_collects_all_interrupts() -> None:
    """Both parallel interrupts are surfaced by a single ainvoke()."""
    for delay in (0.0, 0.05):
        result = await experiment_1_one_invoke(delay)
        assert len(result["pending"]) == 2, result
        assert result["seen"] == [], result["seen"]
        questions = sorted(value["question"] for _, value in result["pending"])
        assert questions == ["A", "B"], questions


async def test_partial_resume_keeps_sibling_pending() -> None:
    """Resuming A does not start a second run; B stays pending, A is kept."""
    result = await experiment_2_partial_resume()

    assert result["after_first_seen"] == ["A=answer-A"], result
    assert len(result["after_first_pending"]) == 1, result
    assert result["after_first_pending"][0][1]["question"] == "B", result
    assert result["after_first_next"] == ("ask_b",), result

    assert sorted(result["after_second_seen"]) == ["A=answer-A", "B=answer-B"], result
    assert result["final_pending"] == [], result
    assert result["final_next"] == (), result


async def test_naive_outer_reinterrupts_after_two_callbacks() -> None:
    """One resume per answer races two runs and parks the outer graph again."""
    run = await experiment_3_outer_inner("naive")

    original_ids = {iid for iid, _ in run.outer_before}
    assert len(original_ids) == 2, run.outer_before
    assert len(run.outer_after_a) == 1, run.outer_after_a

    # Both original callbacks were accepted, yet the outer graph is paused on
    # two NEW interrupts.  Those are the sibling interrupts each concurrent
    # inner run re-raised; no jobmanager callback is registered for them.
    new_ids = {iid for iid, _ in run.outer_after_b}
    assert len(new_ids) == 2, run.outer_after_b
    assert new_ids.isdisjoint(original_ids), (original_ids, new_ids)
    assert run.outer_next == ("ask_outer", "ask_outer"), run.outer_next

    # Two separate inner resumes, each carrying only one answer.
    assert run.inner_resumes == 2, run.resume_calls
    assert all(len(call) == 1 for call in run.resume_calls), run.resume_calls


async def test_collector_resumes_inner_graph_once() -> None:
    """The collector merges both answers into one resume and the step ends."""
    run = await experiment_3_outer_inner("collector")

    assert len(run.outer_before) == 2, run.outer_before
    assert len(run.outer_after_a) == 1, run.outer_after_a
    assert run.outer_after_b == [], run.outer_after_b

    assert run.inner_resumes == 1, run.resume_calls
    assert len(run.resume_calls[0]) == 2, run.resume_calls
    assert sorted(run.inner_seen) == ["A=result-A", "B=result-B"], run.inner_seen
    assert run.inner_pending == [], run.inner_pending
    assert run.outer_next == (), run.outer_next


async def test_orchestrator_pause_to_ask_batch() -> None:
    """Several pause_to_ask calls in one turn are all surfaced together."""
    result = await experiment_4_pause_to_ask(3)
    assert result["pending_count"] == 3, result
    assert result["final_next"] == (), result
    assert result["tool_messages"] == [
        "answer=answer-0",
        "answer=answer-1",
        "answer=answer-2",
    ], result["tool_messages"]


async def main() -> None:
    for test in (
        test_one_invoke_collects_all_interrupts,
        test_partial_resume_keeps_sibling_pending,
        test_naive_outer_reinterrupts_after_two_callbacks,
        test_collector_resumes_inner_graph_once,
        test_orchestrator_pause_to_ask_batch,
    ):
        await test()
        print(f"ok: {test.__name__}")
    print("MULTI INTERRUPT DEMO: OK")


if __name__ == "__main__":
    asyncio.run(main())
