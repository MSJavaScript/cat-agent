"""The debug CLI polls and resumes submitted jobs itself.

No LLM, no server: a fake graph stream emits a JobSubmission interrupt and a
mock HTTP transport serves the durable job result.  The CLI must opt out of
jobmanager callbacks, poll ``/status/{job_uuid}``, and resume with the result
instead of asking the user about the job.
"""

from __future__ import annotations

import asyncio
import builtins
from types import SimpleNamespace
from typing import Any

import httpx
from langgraph.types import Command

import catagent.app.cli as cli


class FakeDB:
    async def close(self) -> None:
        pass


async def main() -> None:
    requests: list[tuple[str, str]] = []
    stream_calls: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.method, request.url.path))
        assert request.url.path == "/status/job-1", request.url.path
        return httpx.Response(
            200,
            json={
                "job_uuid": "job-1",
                "local_dir": "/tmp/job-1",
                "result_status": "success",
                "result_error": "",
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def fake_build_runtime(*, dotenv_path: str, db_path: str,
                                 job_manager_url: str, skill_index_json: str):
        return object(), http_client, FakeDB()

    async def fake_stream_chat(*, catagent, agent_input, thread_id,
                               job_manager_url, http_client, register_jobs=True):
        stream_calls.append(
            {"input": agent_input, "register_jobs": register_jobs}
        )
        if len(stream_calls) == 1:
            yield {
                "type": "interrupt",
                "message": {
                    "kind": "JobSubmission",
                    "message": "submitted",
                    "payload": {"job_uuid": "job-1"},
                },
                "message_id": "int-1",
                "interrupt_id": "int-1",
                "thread_id": thread_id,
            }
        else:
            yield {
                "type": "custom",
                "message": "resumed-ok",
                "message_id": "m-1",
                "thread_id": thread_id,
            }

    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return "go" if len(prompts) == 1 else "exit"

    original_build = cli.build_runtime
    original_stream = cli._stream_chat
    original_input = builtins.input
    cli.build_runtime = fake_build_runtime
    cli._stream_chat = fake_stream_chat
    builtins.input = fake_input
    try:
        args = SimpleNamespace(
            dotenv=".env",
            db=":memory:",
            job_manager_url="http://jobs",
            skill_index_json="skills/skill_index.json",
            thread_id="t-cli",
        )
        await cli.run_cli(args)
    finally:
        cli.build_runtime = original_build
        cli._stream_chat = original_stream
        builtins.input = original_input
        await http_client.aclose()

    assert len(stream_calls) == 2, stream_calls
    assert all(call["register_jobs"] is False for call in stream_calls), (
        "the CLI must not register a jobmanager callback"
    )
    assert isinstance(stream_calls[1]["input"], Command)
    assert stream_calls[1]["input"].resume == {
        "int-1": {
            "payload": {
                "local_dir": "/tmp/job-1",
                "status": "success",
                "error_msg": "",
            }
        }
    }
    assert requests == [("GET", "/status/job-1")], requests
    # Only user prompts; JobSubmission must never be asked on stdin.
    assert len(prompts) == 2, prompts
    assert all("answer" not in prompt for prompt in prompts), prompts
    print("CLI JOB POLLING TEST: OK")


if __name__ == "__main__":
    asyncio.run(main())
