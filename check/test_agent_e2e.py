"""End-to-end test: orchestrator -> router -> StepNode -> jobmanager callback.

This uses the real LLM and the real local jobmanager, but every "calculation"
is the fake sleep-based software in check/mock_software.

What it exercises:
  * the orchestrator reads the mock-compute skill and makes a plan,
  * the router prepares workdirs and dispatches the two relax steps,
  * StepNode submits real local jobs through jobmanager,
  * jobmanager posts JobResults back to the callback endpoint,
  * every job interrupt is resumed through catagent.app._stream_chat,
  * the final report files are produced.

Run from the repository root:

    python -m check.test_agent_e2e

Optional:
    python -m check.test_agent_e2e --workdir /tmp/my_e2e --timeout 300
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import httpx
import uvicorn
from fastapi import FastAPI
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from catagent.agent import create_catagent
from catagent.app import _stream_chat
from check.common import (
    MOCK_BIN, REPO_ROOT, add_mock_path_to_env, load_mock_skills, make_model,
)

JOB_MANAGER_PORT = 18000
CALLBACK_PORT = 5600
JOB_MANAGER_URL = f"http://127.0.0.1:{JOB_MANAGER_PORT}"
TEST_CONFIG = REPO_ROOT / "check" / "jobmanager_test_config.json"


class CallbackServer:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.server: uvicorn.Server | None = None
        self.task: asyncio.Task | None = None

    def build(self) -> FastAPI:
        app = FastAPI()

        @app.post("/")
        async def receive_callback(result: dict) -> dict:
            await self.queue.put(result)
            return {"status": "success"}

        return app

    async def start(self) -> None:
        config = uvicorn.Config(self.build(), host="127.0.0.1",
                                port=CALLBACK_PORT, log_level="warning")
        self.server = uvicorn.Server(config)
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(200):
            if self.server.started:
                return
            await asyncio.sleep(0.05)
        raise RuntimeError("callback server did not start")

    async def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.task is not None:
            await self.task


async def wait_http(url: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=2) as client:
        while time.monotonic() < deadline:
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.25)
    raise TimeoutError(f"service at {url} did not become ready")


def print_payload(payload: dict[str, Any]) -> None:
    kind = payload.get("type", "custom")
    source = payload.get("source")
    source_text = f" [{source}]" if source else ""
    message = payload.get("message")
    if kind == "interrupt":
        print(f"[{kind}{source_text}] id={payload.get('interrupt_id')} {message!r}")
    elif kind == "agent":
        calls = payload.get("tool_calls") or []
        print(f"[{kind}{source_text}] {message!r}")
        for call in calls:
            print(f"    -> {call.get('name')}({call.get('args')})")
    elif kind == "tool":
        text = str(message)
        if len(text) > 300:
            text = text[:300] + "..."
        print(f"[{kind}{source_text}] {payload.get('tool_call_name')}: {text}")
    elif kind == "custom":
        text = str(message)
        if len(text) > 200:
            text = text[:200] + "..."
        print(f"[{kind}{source_text}] {text}")


def interrupt_kind(value: Any) -> str | None:
    if isinstance(value, dict):
        return value.get("kind")
    return None


def job_uuid_from_interrupt(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    payload = value.get("payload") or {}
    if isinstance(payload, dict) and payload.get("job_uuid"):
        return payload["job_uuid"]
    return value.get("job_uuid")


def auto_answer(value: Any) -> Any:
    kind = interrupt_kind(value)
    if kind == "AgentInterrupt":
        return {"payload": {"result":
                "Use the defaults from the mock-compute skill and continue."}}
    if kind == "AskForArgs":
        return {"payload": {"command": "use-default"}}
    return "Proceed using the mock-compute skill defaults."


async def run_workflow(thread_id: str, workdir: Path,
                       timeout: float) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    callback = CallbackServer()
    await callback.start()

    pending_callbacks: Dict[str, dict] = {}

    async def wait_callback(interrupt_id: str) -> dict:
        while interrupt_id not in pending_callbacks:
            item = await asyncio.wait_for(callback.queue.get(), timeout=timeout)
            print(f"[jobmanager callback] {item}")
            pending_callbacks[item["interrupt_id"]] = item
        return pending_callbacks.pop(interrupt_id)

    try:
        model = make_model()
        async with httpx.AsyncClient(timeout=60) as http_client:
            catagent = create_catagent(
                model=model,
                job_manager_url=JOB_MANAGER_URL,
                http_client=http_client,
                checkpointer=InMemorySaver(),
            )

            request = (
                "Use the already-loaded `mock-compute` skill to run the fake "
                "computation workflow for samples A and B. Work inside "
                f"`{workdir}`. Create step workdirs under it. "
                "Use the planner and register_raw_plan in append mode. "
                "The plan must prepare inputs, run relax_A and relax_B in "
                "parallel, then analyze both results and write final_report.md. "
                "Use provider `local-mock` and job environment `mock-compute` "
                "for the relax steps; use the mock-* commands from the skill. "
                "Do not ask questions: use the defaults in the skill. "
                "Every step must write report.md before it finishes. "
                "When all steps are done, log state.md and call "
                "set_finalization_flag with True."
            )
            current: Any = {
                "messages": [HumanMessage(content=request)],
                "loaded_skills": load_mock_skills(),
                "finalization": False,
                "orchestrator_is_working": False,
            }

            while True:
                pending: dict[str, Any] = {}
                async for payload in _stream_chat(
                        catagent=catagent,
                        agent_input=current,
                        thread_id=thread_id,
                        job_manager_url=JOB_MANAGER_URL,
                        http_client=http_client):
                    print_payload(payload)
                    if payload.get("type") == "interrupt":
                        pending[payload["interrupt_id"]] = payload.get("message")

                if not pending:
                    break

                answers: dict[str, Any] = {}
                for interrupt_id, value in pending.items():
                    kind = interrupt_kind(value)
                    if kind == "JobSubmission":
                        job_uuid = job_uuid_from_interrupt(value)
                        print(f"[job] waiting for {job_uuid} (interrupt {interrupt_id})")
                        result = await wait_callback(interrupt_id)
                        answers[interrupt_id] = {
                            "payload": {
                                "local_dir": result.get("local_dir", ""),
                                "status": result.get("status", "error"),
                                "error_msg": result.get("error_msg", ""),
                            }
                        }
                    else:
                        answers[interrupt_id] = auto_answer(value)
                current = Command(resume=answers)

            state = await catagent.aget_state(config)
            return dict(state.values)
    finally:
        await callback.stop()


def start_jobmanager(log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["PATH"] = f"{MOCK_BIN}{os.pathsep}{env.get('PATH', '')}"
    TEST_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    (TEST_CONFIG.parent / ".tmp").mkdir(parents=True, exist_ok=True)
    db_path = Path("/mnt/t/programming/myagents/cat-agent-dev/check/.tmp/jobmanager_test.db")
    for suffix in ("", "-shm", "-wal"):
        try:
            Path(str(db_path) + suffix).unlink()
        except FileNotFoundError:
            pass
    log_file = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        [sys.executable, "-m", "jobmanager.server",
         "--setting", str(TEST_CONFIG),
         "--host", "127.0.0.1",
         "--port", str(JOB_MANAGER_PORT),
         "--log", str(log_path),
         "--level", "info"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="cat-agent mock end-to-end test.")
    parser.add_argument("--workdir", default=None,
                        help="Work directory for the fake calculation.")
    parser.add_argument("--timeout", type=float, default=180.0,
                        help="Seconds to wait for one job callback.")
    args = parser.parse_args()

    add_mock_path_to_env()
    workdir = Path(args.workdir) if args.workdir else (
        REPO_ROOT / "check" / ".tmp" / f"e2e_{uuid.uuid4().hex[:8]}")
    workdir.mkdir(parents=True, exist_ok=True)
    log_path = workdir / "jobmanager.log"

    jobmanager = start_jobmanager(log_path)
    try:
        asyncio.run(wait_http(f"{JOB_MANAGER_URL}/providers"))
        state = asyncio.run(run_workflow(
            thread_id=f"e2e-{uuid.uuid4().hex[:8]}",
            workdir=workdir,
            timeout=args.timeout,
        ))
    finally:
        jobmanager.terminate()
        try:
            jobmanager.wait(timeout=10)
        except subprocess.TimeoutExpired:
            jobmanager.kill()
            jobmanager.wait(timeout=10)

    steps = state.get("steps") or {}
    print("\nFinal steps:")
    for step_id in sorted(steps):
        step = steps[step_id]
        print(f"  {step_id}: status={step.status} "
              f"preparation={step.preparation_stage} workdir={step.workdir}")

    found = sorted(str(p.relative_to(workdir)) for p in workdir.rglob("*")
                   if p.is_file())
    print("\nFiles produced:")
    for path in found:
        print(f"  {path}")

    if not state.get("finalization", False):
        raise AssertionError("workflow did not set finalization=True")
    if not steps:
        raise AssertionError("workflow produced no steps")
    bad = [step_id for step_id, step in steps.items() if step.status != "done"]
    if bad:
        raise AssertionError(f"steps not done: {bad}")
    if not any(p.endswith("analysis.json") for p in found):
        raise AssertionError("analysis.json was not produced")
    if not any(p.endswith("final_report.md") for p in found):
        raise AssertionError("final_report.md was not produced")

    print("\nEND-TO-END TEST: OK")
    print(f"workdir = {workdir}")
    print(f"jobmanager log = {log_path}")


if __name__ == "__main__":
    main()
