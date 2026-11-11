"""Deterministic test for StepNode's summary extraction.

``summary_node`` reads ``report.md`` from the step workdir and asks the model
to extract a structured ``RawStepResult``.  This test replaces the LLM call
with a fixed response and verifies both the success path and the two error
paths (missing / empty report).  It also guards against the regression where
a ``SystemMessage`` was passed as the *content* of another ``SystemMessage``.

Run from the repository root:

    python -m check.test_agent_stepnode
"""

import tempfile
from pathlib import Path

from langchain.messages import HumanMessage, SystemMessage

from catagent.stepnode.models import RawStepResult
from catagent.stepnode.stepnode import create_summary_node


class FixedSummaryModel:
    """Stand-in for the LLM used by summary_node."""

    def __init__(self, result: RawStepResult) -> None:
        self.result = result
        self.calls = []

    def with_structured_output(self, schema):
        if schema is not RawStepResult:
            raise AssertionError(f"unexpected structured output schema: {schema}")
        return self

    def invoke(self, messages):
        if len(messages) != 2:
            raise AssertionError(f"expected 2 messages, got {len(messages)}")
        if not isinstance(messages[0], SystemMessage):
            raise AssertionError("the first message must be a SystemMessage")
        if not isinstance(messages[0].content, str):
            raise AssertionError("SystemMessage.content must be a string")
        if not isinstance(messages[1], HumanMessage):
            raise AssertionError("the second message must be a HumanMessage")
        self.calls.append(messages)
        return self.result


def write_report(workdir: Path, text: str) -> None:
    (workdir / "report.md").write_text(text, encoding="utf-8")


def main() -> None:
    expected = RawStepResult(
        status="success",
        step_id="plan-0:step-0:prepare",
        summary="Created input_A.json and input_B.json.",
        artifacts={"input_A.json": "sample A input",
                   "input_B.json": "sample B input"},
    )
    model = FixedSummaryModel(expected)
    summary_node = create_summary_node(model)

    with tempfile.TemporaryDirectory(prefix="stepnode_summary_") as tmp:
        workdir = Path(tmp)

        # 1. Missing report.md -> error result, no model call.
        result = summary_node({"workdir": tmp})["step_result"]
        if result.status != "error":
            raise AssertionError(f"missing report should be an error, got {result}")
        if model.calls:
            raise AssertionError("the model must not be called for a missing report")

        # 2. Empty report.md -> error result, no model call.
        write_report(workdir, "   \n\n")
        result = summary_node({"workdir": tmp})["step_result"]
        if result.status != "error":
            raise AssertionError(f"empty report should be an error, got {result}")
        if model.calls:
            raise AssertionError("the model must not be called for an empty report")

        # 3. Valid report.md -> model result is forwarded.
        write_report(workdir, "# Report\n\nEverything succeeded.\n")
        result = summary_node({"workdir": tmp})["step_result"]
        if result != expected:
            raise AssertionError(f"unexpected summary result: {result}")
        if len(model.calls) != 1:
            raise AssertionError(
                f"the model should be called once, got {len(model.calls)}"
            )

    print("StepNode summary_node test: OK")
    print("missing report: error")
    print("empty report: error")
    print("valid report: success")


if __name__ == "__main__":
    main()
