"""Real-provider integration test: StepNode -> jobmanager -> bohr/slurm-1.

This test uses the real LLM, the real jobmanager and the real remote providers
``bohr`` and ``slurm-1``.  It copies the four VASP single-point input sets from
``/mnt/t/workdir/agent_test/case0/test_cases`` into a temporary workdir and
runs one StepNode conversation per case.

Why StepNode level instead of the full orchestrator?  The failure this test
guards against was in the StepNode <-> jobmanager path: a job downloaded the
wrong workdir's archive (shared ``scp.SCPClient``), or a transient transfer
error was reported as "results lost", which made the LLM submit a duplicate
job.  Running the four StepNodes concurrently reproduces exactly that
contention while keeping the workflow deterministic (no orchestrator planning).

What it checks:
  * each StepNode submits exactly one job for its workdir;
  * all four jobs complete on the real providers and the callback resumes the
    StepNode;
  * OUTCAR/CONTCAR contain the requested element and atom count (the old bug
    put Cu results into the Si workdir);
  * the jobmanager log contains none of the old misleading download errors.

Run from the repository root with the project venv:

    /mnt/t/softwares_install/lang-graph/bin/python3 \
        -m check.test_agent_real_jobmanager

Useful options:

    --workdir /tmp/opencode/real_jm
    --cases bohr_Si,slurm1_Cu
    --timeout 1800          # seconds to wait for one job callback
    --poll-interval 15      # jobmanager poll interval
    --agent-prepares-files  # let the StepNode create run.sh/job_description.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx
import uvicorn
from ase.io import read as ase_read
from fastapi import FastAPI
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from catagent.app import _stream_chat, load_skill_index
from catagent.stepnode.stepnode import create_stepnode_agent
from check.common import REPO_ROOT, load_env, make_model

TEST_CASES = Path("/mnt/t/workdir/agent_test/case0/test_cases")
BASE_CONFIG = REPO_ROOT / "jobmanager" / "jobmanager_config.json"
SKILL_INDEX = REPO_ROOT / "skills" / "skill_index.json"
PYTHON = "/mnt/t/softwares_install/lang-graph/bin/python3"
EXPECTED_ARTIFACTS = ("OUTCAR", "CONTCAR", "vasp.stdout")


@dataclass(frozen=True)
class Case:
    name: str
    source: Path
    provider: str
    job_env: str
    element: str
    atoms: int
    description: str


CASES: Dict[str, Case] = {
    "bohr_Si": Case(
        name="bohr_Si",
        source=TEST_CASES / "bohr" / "Si",
        provider="bohr",
        job_env="vasp-6.3.2",
        element="Si",
        atoms=8,
        description=("VASP single-point calculation for the 8-atom conventional "
                     "Si diamond bulk cell (PBE, ENCUT=400, KSPACING=0.3, "
                     "ISPIN=1, IBRION=-1, NSW=0)."),
    ),
    "bohr_Cu": Case(
        name="bohr_Cu",
        source=TEST_CASES / "bohr" / "Cu",
        provider="bohr",
        job_env="vasp-6.3.2",
        element="Cu",
        atoms=4,
        description=("VASP single-point calculation for the 4-atom conventional "
                     "fcc Cu bulk cell (PBE, ENCUT=400, KSPACING=0.3, "
                     "ISPIN=1, IBRION=-1, NSW=0)."),
    ),
    "slurm1_Si": Case(
        name="slurm1_Si",
        source=TEST_CASES / "slurm" / "Si",
        provider="slurm-1",
        job_env="vasp-6.4.0",
        element="Si",
        atoms=8,
        description=("VASP single-point calculation for the 8-atom conventional "
                     "Si diamond bulk cell (PBE, ENCUT=400, KSPACING=0.3, "
                     "ISPIN=1, IBRION=-1, NSW=0)."),
    ),
    "slurm1_Cu": Case(
        name="slurm1_Cu",
        source=TEST_CASES / "slurm" / "Cu",
        provider="slurm-1",
        job_env="vasp-6.4.0",
        element="Cu",
        atoms=4,
        description=("VASP single-point calculation for the 4-atom conventional "
                     "fcc Cu bulk cell (PBE, ENCUT=400, KSPACING=0.3, "
                     "ISPIN=1, IBRION=-1, NSW=0)."),
    ),
}


# ---------------------------------------------------------------------------
# callback server / broker
# ---------------------------------------------------------------------------
class CallbackBroker:
    """Collect jobmanager callbacks and let each StepNode wait for its own."""

    def __init__(self) -> None:
        self.results: Dict[str, dict] = {}
        self.events: Dict[str, asyncio.Event] = {}
        self.lock = asyncio.Lock()
        self.received = 0
        self.ignored: List[dict] = []

    async def put(self, result: Any) -> None:
        if not isinstance(result, dict):
            self.ignored.append({"raw": repr(result)[:500]})
            return
        interrupt_id = str(result.get("interrupt_id") or "")
        job_uuid = str(result.get("job_uuid") or "")
        if not interrupt_id or not job_uuid:
            # The jobmanager should never send a callback without a resume
            # target; record it so a malformed callback is visible in the
            # summary instead of being silently dropped.
            self.ignored.append(result)
            return
        async with self.lock:
            self.received += 1
            self.results[interrupt_id] = result
            event = self.events.setdefault(interrupt_id, asyncio.Event())
            event.set()

    async def wait(self, interrupt_id: str, timeout: float) -> dict:
        async with self.lock:
            if interrupt_id in self.results:
                return self.results.pop(interrupt_id)
            event = self.events.setdefault(interrupt_id, asyncio.Event())
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"no jobmanager callback for interrupt {interrupt_id} "
                f"within {timeout:.0f}s") from exc
        async with self.lock:
            event = self.events.pop(interrupt_id, None)
            result = self.results.pop(interrupt_id, None)
        if result is None:
            raise RuntimeError(
                f"callback for interrupt {interrupt_id} was consumed twice")
        return result


class CallbackServer:
    def __init__(self, broker: CallbackBroker) -> None:
        self.broker = broker
        self.server: Optional[uvicorn.Server] = None
        self.task: Optional[asyncio.Task] = None
        self.port: Optional[int] = None

    def build(self) -> FastAPI:
        app = FastAPI()

        @app.post("/")
        async def receive_callback(result: dict) -> dict:
            await self.broker.put(result)
            return {"status": "success"}

        @app.get("/health")
        async def health() -> dict:
            return {"status": "ok"}

        return app

    async def start(self) -> None:
        # Port 0 lets the OS choose; read it back after startup so the
        # jobmanager config can be written without a port race.
        config = uvicorn.Config(self.build(), host="127.0.0.1", port=0,
                                log_level="warning")
        self.server = uvicorn.Server(config)
        self.task = asyncio.create_task(self.server.serve())
        for _ in range(400):
            if self.server.started and self.server.servers:
                self.port = self.server.servers[0].sockets[0].getsockname()[1]
                return
            if self.task.done():
                self.task.result()
            await asyncio.sleep(0.05)
        raise RuntimeError("callback server did not start")

    async def stop(self) -> None:
        if self.server is not None:
            self.server.should_exit = True
        if self.task is not None:
            await self.task


# ---------------------------------------------------------------------------
# jobmanager process / config
# ---------------------------------------------------------------------------
def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def write_jobmanager_config(workdir: Path, callback_url: str,
                            poll_interval: float) -> Path:
    with BASE_CONFIG.open(encoding="utf-8") as handle:
        base = json.load(handle)

    providers = {
        name: base["providers"][name]
        for name in ("bohr", "slurm-1")
    }
    config = {
        "callback_url": callback_url,
        "db_path": str(workdir / "jobmanager.db"),
        "front_end_url": None,
        # Keep the real provider settings, but poll faster than the 60 s
        # production default so the test finishes in minutes, not hours.
        "poll_interval": poll_interval,
        "retry_job_download": [5, 10, 20],
        "retry_callback": [2, 5, 10],
        "n_retry_get_status": 3,
        "n_retry_job_submit": 3,
        "max_finalize_retries": 5,
        "providers": providers,
    }
    path = workdir / "jobmanager_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return path


def start_jobmanager(config_path: Path, port: int,
                     log_path: Path) -> tuple[subprocess.Popen, Any, Path]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = log_path.with_name("jobmanager_stdout.log")
    stdout_file = stdout_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    # Make `python3` inside jobmanager's execute_shell resolve to the venv
    # that has ase, so its checking agent can inspect VASP outputs.
    venv_bin = str(Path(sys.executable).parent)
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "jobmanager.server",
         "--setting", str(config_path),
         "--host", "127.0.0.1",
         "--port", str(port),
         "--log", str(log_path),
         "--level", "info"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=stdout_file,
        stderr=subprocess.STDOUT,
    )
    return proc, stdout_file, stdout_path


async def wait_http(url: str, proc: subprocess.Popen, log_path: Path,
                    timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=3) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"jobmanager exited with code {proc.returncode}.\n"
                    f"{tail_text(log_path)}")
            try:
                response = await client.get(url)
                if response.status_code < 500:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"service at {url} did not become ready.\n"
                       f"{tail_text(log_path)}")


def tail_text(path: Path, lines: int = 80) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return f"({path} does not exist)"
    return "\n".join(content.splitlines()[-lines:])


def stop_jobmanager(proc: subprocess.Popen, stdout_file: Any) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    stdout_file.close()


# ---------------------------------------------------------------------------
# case preparation
# ---------------------------------------------------------------------------
def case_readme(case: Case, case_dir: Path) -> str:
    return f"""# readme: {case.name}

## Content
Run one VASP single-point calculation through the JobManager and report the result.

- System: {case.element} bulk, {case.atoms} atoms.
- Provider: `{case.provider}` (CPU).
- Job environment: `{case.job_env}`.
- Workdir: `{case_dir}`.

The input files are already in the workdir:
- `INCAR`, `POSCAR`, `POTCAR`
- `run.sh` (submission template, uses the `<np>` placeholder)
- `job_description.json` (job metadata)

## What to do
1. Verify that `INCAR`, `POSCAR`, `POTCAR`, `run.sh` and
   `job_description.json` exist in the workdir.
2. Call `submit_job` **exactly once** with:
   - `job_description`: "{case.description}"
   - `local_dir`: "{case_dir}"
   - `provider_name`: "{case.provider}"
   - `job_env_name`: "{case.job_env}"
   - `result_patterns`: ["OUTCAR", "CONTCAR", "vasp.stdout"]
3. The call pauses until the job manager posts the result back. When it
   returns, inspect `OUTCAR` and `CONTCAR` with ASE
   (`{PYTHON}`) and confirm:
   - the structure contains only {case.element},
   - it has {case.atoms} atoms,
   - `vasp.stdout` contains a final `F=` / `E0=` line.
4. Write `report.md` in the workdir with a short summary (status, atom count,
   final energy).

## Rules
- Do not modify `INCAR`, `POSCAR` or `POTCAR`.
- Do not run VASP locally; the JobManager runs it remotely.
- Do not submit a second job, even if the result mentions a download problem.
  A transfer error is not evidence that the calculation failed.
"""


def prepare_case(case: Case, workdir: Path,
                 agent_prepares_files: bool) -> Path:
    if not case.source.is_dir():
        raise FileNotFoundError(f"test case {case.source} does not exist")
    case_dir = workdir / "cases" / case.name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    shutil.copytree(case.source, case_dir)

    (case_dir / "readme.md").write_text(case_readme(case, case_dir),
                                        encoding="utf-8")
    if not agent_prepares_files:
        (case_dir / "run.sh").write_text(
            "#!/bin/bash\n"
            "mpirun -n <np> vasp_std > vasp.stdout\n",
            encoding="utf-8")
        (case_dir / "run.sh").chmod(0o755)
        job_description = {
            "description": case.description,
            "job_name": f"{case.name}_sp",
            "use_gpu": False,
            "batch_job": False,
            "n_jobs": 1,
            "workload": "mini",
            "placeholders": {
                "np": "The number of MPI processes",
            },
        }
        (case_dir / "job_description.json").write_text(
            json.dumps(job_description, indent=2), encoding="utf-8")
    return case_dir


# ---------------------------------------------------------------------------
# StepNode execution
# ---------------------------------------------------------------------------
def auto_answer(value: Any) -> Any:
    if isinstance(value, dict):
        kind = value.get("kind")
        if kind == "AskForArgs":
            return {"payload": {"command": "use-default"}}
        if kind == "AgentInterrupt":
            return {"payload": {
                "result": ("The inputs and settings are correct as described "
                           "in readme.md. Continue without asking again.")}}
    return {"payload": {"result": "Proceed using the instructions in readme.md."}}


def print_payload(case_name: str, payload: dict[str, Any]) -> None:
    kind = payload.get("type", "custom")
    if kind == "interrupt":
        value = payload.get("message")
        inner = value.get("kind") if isinstance(value, dict) else None
        print(f"[{case_name}] interrupt {inner} id={payload.get('interrupt_id')}")
        return
    if kind == "agent":
        calls = payload.get("tool_calls") or []
        text = str(payload.get("message") or "")
        print(f"[{case_name}] agent {text[:200]!r}")
        for call in calls:
            print(f"[{case_name}]   -> {call.get('name')}({call.get('args')})")
        return
    if kind == "tool":
        text = str(payload.get("message") or "")
        print(f"[{case_name}] tool {payload.get('tool_call_name')}: {text[:240]!r}")
        return
    text = str(payload.get("message") or "")
    print(f"[{case_name}] {kind}: {text[:200]!r}")


async def run_case(case: Case, case_dir: Path, stepnode: Any,
                   broker: CallbackBroker, http_client: httpx.AsyncClient,
                   skills: Dict[str, Any], run_id: str, timeout: float) -> dict:
    thread_id = f"{run_id}:{case.name}"
    config = {"configurable": {"thread_id": thread_id}}
    current: Any = {
        "messages": [HumanMessage(content=(case_dir / "readme.md").read_text(
            encoding="utf-8"))],
        "workdir": str(case_dir),
        "loaded_skills": skills,
        "accessible_tool_names": set(),
        "skip_review_tool_names": set(),
        "interrupts": {},
    }

    for _turn in range(60):
        pending: Dict[str, Any] = {}
        async for payload in _stream_chat(
                catagent=stepnode,
                agent_input=current,
                thread_id=thread_id,
                job_manager_url=JOB_MANAGER_URL,
                http_client=http_client):
            print_payload(case.name, payload)
            if payload.get("type") == "interrupt":
                pending[payload["interrupt_id"]] = payload.get("message")

        if not pending:
            break

        answers: Dict[str, Any] = {}
        for interrupt_id, value in pending.items():
            kind = value.get("kind") if isinstance(value, dict) else None
            if kind == "JobSubmission":
                job_uuid = (value.get("payload") or {}).get("job_uuid") \
                    if isinstance(value, dict) else None
                print(f"[{case.name}] waiting for job {job_uuid} "
                      f"(interrupt {interrupt_id})")
                result = await broker.wait(interrupt_id, timeout)
                print(f"[{case.name}] callback {result}")
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
    else:
        raise RuntimeError(f"{case.name}: StepNode did not finish after 60 turns")

    state = await stepnode.aget_state(config)
    return dict(state.values)


# ---------------------------------------------------------------------------
# result checks
# ---------------------------------------------------------------------------
def normalise_step_result(value: Any) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if isinstance(value, dict):
        return value
    return {}


def check_case(case: Case, case_dir: Path, state: dict,
               rows: List[dict]) -> List[str]:
    errors: List[str] = []
    step_result = normalise_step_result(state.get("step_result"))
    status = step_result.get("status")
    if status != "success":
        errors.append(f"{case.name}: StepNode result status is {status!r}: "
                      f"{step_result.get('summary')!r}")

    report = case_dir / "report.md"
    if not report.is_file() or report.stat().st_size == 0:
        errors.append(f"{case.name}: report.md was not written")

    matching = [row for row in rows
                if os.path.realpath(row.get("local_dir", "")) ==
                os.path.realpath(str(case_dir))]
    if len(matching) != 1:
        errors.append(f"{case.name}: expected exactly one jobmanager row for "
                      f"{case_dir}, found {len(matching)}: "
                      f"{[row.get('job_uuid') for row in matching]}")
        return errors

    row = matching[0]
    expected = {
        "result_status": "success",
        "finalize_state": "delivered",
        "submission_state": "submitted",
    }
    for key, want in expected.items():
        got = row.get(key)
        if got != want:
            errors.append(f"{case.name}: jobmanager {key}={got!r}, expected {want!r} "
                          f"(result_error={row.get('result_error')!r})")
    # slurm purges a job from `scontrol show jobid` shortly after it finishes,
    # so its aggregated status can legitimately be "unknown" even though the
    # results downloaded and the checking agent confirmed success.
    if row.get("status") not in ("completed", "unknown"):
        errors.append(f"{case.name}: jobmanager status={row.get('status')!r}, "
                      f"expected 'completed' (or 'unknown' for a purged slurm job)")
    if not row.get("thread_id") or not row.get("interrupt_id"):
        errors.append(f"{case.name}: callback target was never registered "
                      f"(thread_id={row.get('thread_id')!r}, "
                      f"interrupt_id={row.get('interrupt_id')!r})")
    if row.get("result_error"):
        errors.append(f"{case.name}: result_error is not empty: "
                      f"{row.get('result_error')!r}")
    if len(row.get("sub_jobs") or []) != 1:
        errors.append(f"{case.name}: expected one sub job, got "
                      f"{len(row.get('sub_jobs') or [])}")

    for name in EXPECTED_ARTIFACTS:
        path = case_dir / name
        if not path.is_file():
            errors.append(f"{case.name}: missing downloaded artifact {name}")
        elif path.stat().st_size == 0:
            errors.append(f"{case.name}: downloaded artifact {name} is empty")

    contcar = case_dir / "CONTCAR"
    if contcar.is_file() and contcar.stat().st_size:
        try:
            atoms = ase_read(contcar, format="vasp")
            symbols = atoms.get_chemical_symbols()
            if set(symbols) != {case.element}:
                errors.append(
                    f"{case.name}: CONTCAR contains {sorted(set(symbols))}, "
                    f"expected only {case.element}")
            if len(atoms) != case.atoms:
                errors.append(f"{case.name}: CONTCAR has {len(atoms)} atoms, "
                              f"expected {case.atoms}")
        except Exception as exc:
            errors.append(f"{case.name}: cannot parse CONTCAR: {exc}")

    outcar = case_dir / "OUTCAR"
    if outcar.is_file() and outcar.stat().st_size:
        try:
            atoms = ase_read(outcar, format="vasp-out", index=-1)
            symbols = atoms.get_chemical_symbols()
            if set(symbols) != {case.element}:
                errors.append(
                    f"{case.name}: OUTCAR contains {sorted(set(symbols))}, "
                    f"expected only {case.element}")
            if len(atoms) != case.atoms:
                errors.append(f"{case.name}: OUTCAR has {len(atoms)} atoms, "
                              f"expected {case.atoms}")
        except Exception as exc:
            errors.append(f"{case.name}: cannot parse OUTCAR: {exc}")

    stdout = case_dir / "vasp.stdout"
    if stdout.is_file() and stdout.stat().st_size:
        text = stdout.read_text(encoding="utf-8", errors="replace")
        if "E0=" not in text:
            errors.append(f"{case.name}: vasp.stdout has no final E0= line")

    patterns = row.get("result_patterns") or []
    if not patterns:
        print(f"NOTE {case.name}: agent did not declare result_patterns; "
              f"the download was still verified from the files on disk.")
    return errors


async def fetch_jobs(client: httpx.AsyncClient, base_url: str) -> List[dict]:
    response = await client.get(f"{base_url}/jobs", params={"limit": 500})
    response.raise_for_status()
    return response.json()


async def cancel_active_jobs(client: httpx.AsyncClient, base_url: str,
                             rows: Iterable[dict]) -> None:
    for row in rows:
        if row.get("status") in ("submitted", "running"):
            try:
                await client.post(f"{base_url}/cancel/{row['job_uuid']}")
            except Exception as exc:
                print(f"WARNING: could not cancel {row.get('job_uuid')}: {exc}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
JOB_MANAGER_URL = ""


async def run_all(args: argparse.Namespace) -> None:
    global JOB_MANAGER_URL
    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    run_id = f"realjm-{uuid.uuid4().hex[:8]}"
    log_path = workdir / "jobmanager.log"

    selected = [name.strip() for name in args.cases.split(",") if name.strip()]
    unknown = [name for name in selected if name not in CASES]
    if unknown:
        raise SystemExit(f"unknown case(s): {unknown}; choose from {list(CASES)}")

    prepared: List[tuple[Case, Path]] = []
    for name in selected:
        case = CASES[name]
        prepared.append((case, prepare_case(case, workdir,
                                            args.agent_prepares_files)))
    print(f"workdir = {workdir}")
    print(f"cases   = {[case.name for case, _ in prepared]}")

    load_env()
    if not os.getenv("MODEL_NAME"):
        raise RuntimeError("MODEL_NAME is not set; check jobmanager/.env")

    broker = CallbackBroker()
    callback = CallbackServer(broker)
    await callback.start()
    print(f"callback server on 127.0.0.1:{callback.port}")

    jm_port = _free_port()
    JOB_MANAGER_URL = f"http://127.0.0.1:{jm_port}"
    config_path = write_jobmanager_config(
        workdir, f"http://127.0.0.1:{callback.port}", args.poll_interval)
    proc, stdout_file, stdout_path = start_jobmanager(
        config_path, jm_port, log_path)

    success = False
    try:
        await wait_http(f"{JOB_MANAGER_URL}/providers", proc, stdout_path)
        print("jobmanager is ready")

        model = make_model()
        skills = load_skill_index(str(SKILL_INDEX))
        system_skills = [
            str(REPO_ROOT / "catagent" / "system_skills" / name)
            for name in ("stepnode_prompt.md", "file_and_state_management.md",
                         "job_submission.md", "file_processing_for_step_node.md")
        ]
        checkpointer = InMemorySaver()
        async with httpx.AsyncClient(timeout=120) as http_client:
            stepnode = create_stepnode_agent(
                model=model,
                checkpointer=checkpointer,
                job_manager_url=JOB_MANAGER_URL,
                http_client=http_client,
                system_skill_pathes=system_skills,
            )
            case_timeout = args.case_timeout or (args.timeout + 1200)
            tasks = [
                asyncio.create_task(
                    asyncio.wait_for(
                        run_case(case, case_dir, stepnode, broker, http_client,
                                 skills, run_id, args.timeout),
                        timeout=case_timeout))
                for case, case_dir in prepared
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            rows = await fetch_jobs(http_client, JOB_MANAGER_URL)
            errors: List[str] = []
            states: Dict[str, dict] = {}
            for (case, case_dir), result in zip(prepared, results):
                if isinstance(result, BaseException):
                    errors.append(f"{case.name}: StepNode failed: {result!r}")
                    states[case.name] = {}
                else:
                    states[case.name] = result
                    errors.extend(check_case(case, case_dir, result, rows))

            log_text = log_path.read_text(encoding="utf-8", errors="replace") \
                if log_path.is_file() else ""
            # A transient "download failed, retrying" line is acceptable: the
            # manager is supposed to retry it.  These markers are terminal
            # failures (or the old shared-SCP symptom) and must not appear.
            fatal_log_markers = [
                "Only part of job results",
                "Underlying channel was closed",
                "Declared result files are missing",
                "Failed to obtain the job results",
                "exhausted retries",
                "Failed to submit job",
            ]
            for marker in fatal_log_markers:
                if marker in log_text:
                    errors.append(f"jobmanager log contains {marker!r}")

            if broker.received < len(prepared):
                errors.append(f"only {broker.received} callback(s) received for "
                              f"{len(prepared)} job(s)")
            if broker.ignored:
                errors.append(f"{len(broker.ignored)} malformed callback(s): "
                              f"{broker.ignored[:3]}")
            if "callback ok" not in log_text and broker.received >= len(prepared):
                print("NOTE: 'callback ok' was not found in the log "
                      "(it may not have been flushed yet).")

            print("\nJobmanager rows:")
            for row in sorted(rows, key=lambda item: item.get("created_at", "")):
                print(f"  {row.get('job_uuid')}  {row.get('local_dir')}  "
                      f"status={row.get('status')} "
                      f"result={row.get('result_status')} "
                      f"finalize={row.get('finalize_state')} "
                      f"attempts={row.get('attempt')} "
                      f"finalize_attempts={row.get('finalize_attempt')}")
                for sub in row.get("sub_jobs") or []:
                    print(f"      group {sub.get('group_index')}: "
                          f"remote_id={sub.get('remote_id')} "
                          f"remote_dir={sub.get('remote_dir')}")

            print("\nResults:")
            for case, case_dir in prepared:
                step_result = normalise_step_result(
                    states.get(case.name, {}).get("step_result"))
                files = [name for name in EXPECTED_ARTIFACTS
                         if (case_dir / name).is_file()]
                print(f"  {case.name}: {step_result.get('status')!r} "
                      f"files={files} dir={case_dir}")

            if errors:
                print("\nFAILURES:")
                for error in errors:
                    print(f"  - {error}")
                raise AssertionError(f"{len(errors)} check(s) failed")

            success = True
            print("\nREAL-PROVIDER AGENT/JOBMANAGER TEST: OK")
            print(f"jobmanager log = {log_path}")
            print(f"jobmanager db  = {workdir / 'jobmanager.db'}")
    finally:
        if not success:
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    rows = await fetch_jobs(client, JOB_MANAGER_URL)
                    await cancel_active_jobs(client, JOB_MANAGER_URL, rows)
            except Exception:
                pass
        stop_jobmanager(proc, stdout_file)
        await callback.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the real bohr/slurm-1 agent <-> jobmanager test.")
    parser.add_argument(
        "--workdir", default=str(Path("/tmp/opencode") /
                                 f"real_jm_{uuid.uuid4().hex[:8]}"),
        help="Directory for the test workdir and jobmanager DB/log.")
    parser.add_argument(
        "--cases", default=",".join(CASES),
        help="Comma-separated subset of cases to run.")
    parser.add_argument(
        "--timeout", type=float, default=1800.0,
        help="Seconds to wait for one job callback.")
    parser.add_argument(
        "--case-timeout", type=float, default=0.0,
        help="Overall seconds per case (default: --timeout + 1200).")
    parser.add_argument(
        "--poll-interval", type=float, default=15.0,
        help="Jobmanager poll interval in seconds.")
    parser.add_argument(
        "--agent-prepares-files", action="store_true",
        help="Let the StepNode create run.sh and job_description.json "
             "instead of pre-creating them.")
    args = parser.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
