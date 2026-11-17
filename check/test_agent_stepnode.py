"""Deterministic test for StepNode's finalization extraction.

Finalization is inlined in ``create_model_and_tool_node(...)[0]``
(``model_node``): when ``stepnode_finalization`` is true it asks the model for a
structured ``RawStepResult``, retries three times, and then falls back to an
error result.  This test replaces the LLM with a fixed stand-in and checks:

* a successful structured result is forwarded unchanged,
* three failed attempts fall back to ``status="error"``,
* the system prompt is passed as a ``SystemMessage`` whose ``content`` is a
  string (guarding the old bug where a ``SystemMessage`` was nested as the
  content of another message).

Run from the repository root:

    python -m check.test_agent_stepnode
"""

from __future__ import annotations

from langchain.messages import HumanMessage, SystemMessage

from catagent.stepnode.models import RawStepResult
from catagent.stepnode.stepnode import create_model_and_tool_node


class FixedSummaryModel:
    """Stand-in for the LLM used by the finalization branch."""

    def __init__(self, result: RawStepResult) -> None:
        self.result = result
        self.fail = False
        self.calls: list[list] = []

    def with_structured_output(self, schema):
        if schema is not RawStepResult:
            raise AssertionError(f"unexpected structured output schema: {schema}")
        return self

    def invoke(self, messages):
        if not isinstance(messages[0], SystemMessage):
            raise AssertionError("the first message must be a SystemMessage")
        if not isinstance(messages[0].content, str):
            raise AssertionError("SystemMessage.content must be a string")
        if not isinstance(messages[-1], HumanMessage):
            raise AssertionError("the last message must be a HumanMessage")
        if "finalization" not in messages[-1].content.lower():
            raise AssertionError(
                f"unexpected finalization prompt: {messages[-1].content!r}"
            )
        self.calls.append(messages)
        if self.fail:
            raise ValueError("simulated structured-output failure")
        return self.result


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    expected = RawStepResult(
        status="success",
        summary="Created input_A.json and input_B.json.",
    )
    model = FixedSummaryModel(expected)
    # Building the node also initialises the shared tool registry, so build it
    # once and reuse it for both scenarios.
    model_node, _ = create_model_and_tool_node(
        model, [], "http://127.0.0.1:1", None
    )

    history = [
        HumanMessage(content="do the work"),
        HumanMessage(content="the work is finished"),
    ]
    state = {"stepnode_finalization": True, "messages": history}

    # 1. A successful structured result is forwarded unchanged.
    result = model_node(state)["step_result"]
    check(result == expected, f"unexpected finalization result: {result}")
    check(len(model.calls) == 1, f"model should be called once, got {len(model.calls)}")
    check(
        model.calls[0][1:-1] == history,
        "the step history must be passed between the system prompt and the "
        "finalization prompt",
    )

    # 2. Three failed attempts fall back to an error result.
    model.fail = True
    model.calls.clear()
    result = model_node(state)["step_result"]
    check(result.status == "error", f"fallback should be an error, got {result}")
    check(
        "Cannot extract structured output" in result.summary,
        f"unexpected fallback summary: {result.summary!r}",
    )
    check(len(model.calls) == 3, f"expected three attempts, got {len(model.calls)}")

    print("StepNode finalization test: OK")
    print("structured result: forwarded")
    print("three failures: error fallback")


if __name__ == "__main__":
    main()
