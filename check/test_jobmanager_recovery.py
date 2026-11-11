"""Deterministic tests for jobmanager crash recovery and multi-user progress.

Run from the repository root:

    python -m check.test_jobmanager_recovery

The tests use a fake provider and a fake jobmanager agent, so no LLM, network
or real computation is involved.  They cover the failure paths that the
end-to-end test cannot exercise:

  * submission resumes from a persisted plan instead of duplicating jobs,
  * a crash during download is re-finalized, never swept as success,
  * a completed job with missing result files is reported as an error,
  * a callback 404 is retried instead of being treated as delivered,
  * the local provider persists real exit codes and survives a restart,
  * orchestrator locks are per user thread, not per process.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import httpx

from jobmanager.base import BaseProvider
from jobmanager.db import JobDB
from jobmanager.manager import JobManager
from jobmanager.models import (
    FinalizeState, JobRecord, JobRequest, JobSetting, JobStatus,
    SubmissionState, SubJobRecord, _now,
)
from jobmanager.providers.local import LocalProvider


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeProvider(BaseProvider):
    type = "fake"

    def __init__(self) -> None:
        super().__init__("fake", ["cpu"])
        self.submit_calls: list[JobSetting] = []
        self.download_fails = False

    def get_info(self) -> dict:
        return {"name": self.name, "processing_units": self.processing_units,
                "job_envs": {}, "remote_files": []}

    def get_remote_files_list(self) -> list[dict]:
        return []

    def get_resources(self) -> list:
        return []

    def get_job_env(self, job_env_name: str) -> dict:
        return {}

    async def get_job_states(self, remote_ids: set[str]) -> dict[str, JobStatus]:
        return {remote_id: JobStatus.RUNNING for remote_id in remote_ids}

    async def cancel(self, remote_id: str) -> None:
        return None

    async def download(self, remote_id: str, reomte_dir: str,
                       local_dir: str) -> bool:
        return not self.download_fails

    async def submit(self, setting: JobSetting) -> str:
        self.submit_calls.append(setting)
        return f"remote-{len(self.submit_calls)}"

    async def cleanup(self, remote_dir: str) -> None:
        return None

    def disconnect(self) -> None:
        return None


class CallbackRecorder:
    def __init__(self, status_code: int = 200) -> None:
        self.status_code = status_code
        self.payloads: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.payloads.append(json.loads(request.content))
            return httpx.Response(self.status_code, json={"status": "success"})
        return httpx.MockTransport(handler)


async def make_manager(tmp: Path, provider: FakeProvider,
                       callback: CallbackRecorder) -> JobManager:
    db = JobDB(str(tmp / "jobs.db"))
    await db.connect()
    manager = JobManager(agent=None, db=db, providers={"fake": provider},
                         callback_url="http://callback.invalid/",
                         poll_interval=1, retry_job_download=[0],
                         retry_callback=[0], n_retry_get_status=1,
                         n_retry_job_submit=1)
    manager.client = httpx.AsyncClient(transport=callback.transport())
    return manager


async def close_manager(manager: JobManager) -> None:
    await manager.stop()


async def drain(manager: JobManager) -> None:
    """Wait for all background tasks, including tasks spawned by tasks."""
    for _ in range(20):
        await asyncio.sleep(0)
        tasks = list(manager._tasks)
        if not tasks:
            return
        await asyncio.gather(*tasks, return_exceptions=True)


async def save_record(manager: JobManager, record: JobRecord) -> None:
    """Persist a test record and its resume registration.

    ``save()`` intentionally does not write thread_id/interrupt_id (a
    placeholder save must never erase a registration made by /set-thread-id),
    so tests that expect callback delivery register them explicitly.
    """
    await manager.db.save(record)
    if record.thread_id or record.interrupt_id:
        await manager.db.set_thread_id(
            record.job_uuid, record.thread_id, record.interrupt_id)


def base_record(job_uuid: str, **overrides) -> JobRecord:
    values = dict(
        job_uuid=job_uuid,
        # A callback is only sent once the agent has registered its resume
        # target; tests that expect a delivery need a valid registration.
        thread_id="test-thread",
        interrupt_id="test-interrupt",
        description="test job",
        provider_id="fake",
        job_env_name="fake-env",
        local_dir=tempfile.mkdtemp(prefix="catagent-job-"),
        result_patterns=[],
        status=JobStatus.COMPLETED,
        created_at=_now(),
        updated_at=_now(),
        callback_pending=False,
        attempt=0,
        error="",
        sub_jobs=[],
        submission_state=SubmissionState.SUBMITTED,
        finalize_state=FinalizeState.PENDING,
    )
    values.update(overrides)
    return JobRecord(**values)


def sub_job(job_uuid: str, local_dir: str, remote_id: str | None = "r1",
            group_index: int = 0) -> SubJobRecord:
    return SubJobRecord(job_uuid=job_uuid, group_index=group_index,
                        local_dir=local_dir, remote_id=remote_id,
                        remote_dir=None, status=JobStatus.COMPLETED,
                        created_at=_now(), updated_at=_now())


async def test_old_database_is_migrated(tmp: Path) -> None:
    """An existing v1 database gains the new columns and enters finalize."""
    import sqlite3

    path = tmp / "old.db"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE jobs (
            job_uuid         TEXT PRIMARY KEY,
            thread_id        TEXT,
            interrupt_id     TEXT,
            description      TEXT,
            provider_id      TEXT NOT NULL,
            job_env_name     TEXT NOT NULL,
            local_dir        TEXT NOT NULL,
            result_patterns  TEXT NOT NULL DEFAULT '[]',
            status           TEXT NOT NULL DEFAULT 'pending',
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            callback_pending INTEGER NOT NULL DEFAULT 0,
            attempt          INTEGER NOT NULL DEFAULT 0,
            error            TEXT DEFAULT ''
        );
        CREATE TABLE sub_jobs (
            sub_job_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            job_uuid    TEXT NOT NULL,
            group_index INTEGER NOT NULL,
            local_dir   TEXT NOT NULL,
            remote_id   TEXT,
            remote_dir  TEXT,
            status      TEXT NOT NULL DEFAULT 'pending',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            error       TEXT NOT NULL DEFAULT '',
            UNIQUE(job_uuid, group_index),
            FOREIGN KEY(job_uuid) REFERENCES jobs(job_uuid) ON DELETE CASCADE
        );
    """)
    con.execute(
        "INSERT INTO jobs (job_uuid, thread_id, interrupt_id, description,"
        " provider_id, job_env_name, local_dir, result_patterns, status,"
        " created_at, updated_at, callback_pending, attempt, error)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("old-job", "t", "i", "d", "fake", "env", "/tmp", "[]",
         "completed", _now(), _now(), 0, 0, ""))
    con.commit()
    con.close()

    db = JobDB(str(path))
    await db.connect()
    try:
        record = await db.get("old-job")
        check(record is not None, "old row disappeared during migration")
        check(record.submission_state == SubmissionState.SUBMITTED,
              f"unexpected submission_state: {record.submission_state}")
        check(record.finalize_state == FinalizeState.PENDING,
              f"terminal row did not enter finalize: {record.finalize_state}")
        check(record.result_status is None, "old row invented a result")
        columns = await db._job_columns()
        for column in ("submission_state", "finalize_state",
                       "result_status", "result_error"):
            check(column in columns, f"missing migrated column {column}")
    finally:
        await db.close()


async def test_submission_resumes_persisted_plan(tmp: Path) -> None:
    provider = FakeProvider()
    manager = await make_manager(tmp, provider, CallbackRecorder())
    try:
        job_uuid = "resume-plan"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        record = base_record(
            job_uuid, local_dir=local_dir, status=JobStatus.SUBMITTED,
            finalize_state=FinalizeState.NONE,
            submission_state=SubmissionState.PLANNED,
            sub_jobs=[sub_job(job_uuid, local_dir, remote_id=None)])
        await save_record(manager, record)

        await manager._boot_recover()
        await drain(manager)

        fresh = await manager.db.get(job_uuid)
        check(fresh is not None, "job disappeared")
        check(fresh.submission_state == SubmissionState.SUBMITTED,
              f"submission not completed: {fresh.submission_state}")
        check(fresh.sub_jobs[0].remote_id is not None,
              "remote id was not persisted")
        check(len(provider.submit_calls) == 1,
              f"expected exactly one submit, got {len(provider.submit_calls)}")
    finally:
        await close_manager(manager)


async def test_submission_skips_already_started_groups(tmp: Path) -> None:
    provider = FakeProvider()
    manager = await make_manager(tmp, provider, CallbackRecorder())
    try:
        job_uuid = "partial-plan"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        record = base_record(
            job_uuid, local_dir=local_dir, status=JobStatus.SUBMITTED,
            finalize_state=FinalizeState.NONE,
            submission_state=SubmissionState.SUBMITTING,
            sub_jobs=[
                sub_job(job_uuid, local_dir, remote_id="already-there", group_index=0),
                sub_job(job_uuid, local_dir, remote_id=None, group_index=1),
            ])
        await save_record(manager, record)

        await manager._run_submission(manager._job_request(record))

        fresh = await manager.db.get(job_uuid)
        check(fresh.sub_jobs[0].remote_id == "already-there",
              "an already submitted group was resubmitted")
        check(fresh.sub_jobs[1].remote_id is not None,
              "the missing group was not submitted")
        check(len(provider.submit_calls) == 1,
              f"expected one new submit, got {len(provider.submit_calls)}")
        check(len({s.idempotency_key for s in provider.submit_calls}) == 1,
              "idempotency key missing")
    finally:
        await close_manager(manager)


async def test_explicit_retry_bumps_attempt(tmp: Path) -> None:
    """An explicit /submit after failure must use a fresh idempotency key.

    Crash retries reuse `job_uuid:gN:aM`; an explicit retry increments M so a
    provider cannot return the previous failed process.
    """
    provider = FakeProvider()
    manager = await make_manager(tmp, provider, CallbackRecorder())

    async def fake_agent(**kwargs):
        return {"preprocessing_output": {"batch_job": False}}

    import jobmanager.manager as manager_module
    original = manager_module.run_job_agent
    manager_module.run_job_agent = fake_agent
    try:
        job_uuid = "explicit-retry"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        record = base_record(
            job_uuid, local_dir=local_dir, status=JobStatus.FAILED,
            submission_state=SubmissionState.FAILED,
            finalize_state=FinalizeState.PENDING,
            attempt=0,
            sub_jobs=[sub_job(job_uuid, local_dir, remote_id="old")])
        await save_record(manager, record)

        request = manager._job_request(record)
        response = await manager.submit_job(request)
        check(response["status"] == "success",
              f"retry was rejected: {response}")
        await drain(manager)

        fresh = await manager.db.get(job_uuid)
        check(fresh.attempt == 1,
              f"explicit retry did not bump the attempt: {fresh.attempt}")
        check(len(provider.submit_calls) == 1,
              f"expected one new submit, got {len(provider.submit_calls)}")
        key = provider.submit_calls[0].idempotency_key
        check(key is not None and key.endswith(":a1"),
              f"retry reused the old idempotency key: {key}")
    finally:
        manager_module.run_job_agent = original
        await close_manager(manager)


async def test_missing_result_files_are_an_error(tmp: Path) -> None:
    callback = CallbackRecorder()
    provider = FakeProvider()
    manager = await make_manager(tmp, provider, callback)

    async def fake_agent(**kwargs):
        return {"checking_output": {"job_result_status": "Success"}}

    import jobmanager.manager as manager_module
    original = manager_module.run_job_agent
    manager_module.run_job_agent = fake_agent
    try:
        job_uuid = "missing-files"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        # One declared output exists, the other does not.  A partial result set
        # must still fail: `any()` semantics would report a false success.
        with open(os.path.join(local_dir, "relaxed.json"), "w",
                  encoding="utf-8") as f:
            f.write("{}")
        record = base_record(
            job_uuid, local_dir=local_dir,
            result_patterns=["relaxed.json", "relax.log"],
            status=JobStatus.COMPLETED, finalize_state=FinalizeState.PENDING,
            sub_jobs=[sub_job(job_uuid, local_dir)])
        await save_record(manager, record)

        await manager._finalize_job(job_uuid)

        check(len(callback.payloads) == 1,
              f"expected one callback, got {len(callback.payloads)}")
        check(callback.payloads[0]["status"] == "error",
              f"missing artifacts were reported as {callback.payloads[0]['status']}")
        check("missing" in callback.payloads[0]["error_msg"].lower(),
              f"unexpected error message: {callback.payloads[0]['error_msg']}")
        check("relax.log" in callback.payloads[0]["error_msg"],
              f"missing pattern not named: {callback.payloads[0]['error_msg']}")
        fresh = await manager.db.get(job_uuid)
        check(fresh.finalize_state == FinalizeState.DELIVERED,
              "callback was not marked delivered")
    finally:
        manager_module.run_job_agent = original
        await close_manager(manager)


async def test_crash_during_download_is_not_a_success(tmp: Path) -> None:
    callback = CallbackRecorder()
    provider = FakeProvider()
    provider.download_fails = True
    manager = await make_manager(tmp, provider, callback)
    try:
        job_uuid = "crash-download"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        record = base_record(
            job_uuid, local_dir=local_dir,
            status=JobStatus.COMPLETED,
            finalize_state=FinalizeState.FINALIZING,
            sub_jobs=[sub_job(job_uuid, local_dir)])
        await save_record(manager, record)

        # Simulates the boot sweep after a crash in the middle of finalize.
        await manager._boot_recover()
        await drain(manager)

        # A failed transfer is retryable: one attempt must not report a result
        # and must leave the job finalizing for the next polling cycle.
        check(callback.payloads == [],
              f"reported a result after one failed download: {callback.payloads}")
        fresh = await manager.db.get(job_uuid)
        check(fresh.finalize_state == FinalizeState.FINALIZING,
              f"failed download left finalize_state={fresh.finalize_state}")
        check(fresh.finalize_attempt == 1,
              f"unexpected finalize_attempt: {fresh.finalize_attempt}")

        # Only after the retry budget is exhausted may it become an error.
        for _ in range(manager.max_finalize_retries):
            await manager._finalize_job(job_uuid)
            await drain(manager)

        check(len(callback.payloads) == 1,
              f"expected one terminal callback, got {len(callback.payloads)}")
        check(callback.payloads[0]["status"] == "error",
              "a failed download was reported as success")
    finally:
        await close_manager(manager)


async def test_polling_sweep_retries_finalizing(tmp: Path) -> None:
    callback = CallbackRecorder()
    provider = FakeProvider()
    provider.download_fails = True
    manager = await make_manager(tmp, provider, callback)
    try:
        job_uuid = "sweep-finalize"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        record = base_record(
            job_uuid, local_dir=local_dir, status=JobStatus.COMPLETED,
            finalize_state=FinalizeState.FINALIZING,
            sub_jobs=[sub_job(job_uuid, local_dir)])
        await save_record(manager, record)

        await manager._retry_pending_finalizations()
        await drain(manager)

        # One failed download keeps the row finalizing instead of signalling
        # "results lost"; the sweep tries again on the next cycle.
        check(callback.payloads == [],
              f"reported a result after one failed download: {callback.payloads}")
        fresh = await manager.db.get(job_uuid)
        check(fresh.finalize_state == FinalizeState.FINALIZING,
              f"failed download left finalize_state={fresh.finalize_state}")

        for _ in range(manager.max_finalize_retries):
            await manager._finalize_job(job_uuid)
            await drain(manager)

        check(len(callback.payloads) == 1,
              f"expected one terminal callback, got {len(callback.payloads)}")
        check(callback.payloads[0]["status"] == "error",
              "polling sweep reported success for a failed download")
    finally:
        await close_manager(manager)


async def test_callback_404_is_not_delivered(tmp: Path) -> None:
    callback = CallbackRecorder(status_code=404)
    provider = FakeProvider()
    manager = await make_manager(tmp, provider, callback)
    try:
        job_uuid = "callback-404"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        result = Path(local_dir) / "result.json"
        result.write_text("{}", encoding="utf-8")
        record = base_record(
            job_uuid, local_dir=local_dir, result_patterns=["result.json"],
            status=JobStatus.COMPLETED, finalize_state=FinalizeState.PENDING,
            sub_jobs=[sub_job(job_uuid, local_dir)])
        await save_record(manager, record)

        async def fake_agent(**kwargs):
            return {"checking_output": {"job_result_status": "Success"}}
        import jobmanager.manager as manager_module
        original = manager_module.run_job_agent
        manager_module.run_job_agent = fake_agent
        try:
            await manager._finalize_job(job_uuid)
        finally:
            manager_module.run_job_agent = original

        fresh = await manager.db.get(job_uuid)
        check(fresh.finalize_state == FinalizeState.READY,
              f"404 changed finalize state to {fresh.finalize_state}")
        check(fresh.callback_pending, "404 cleared callback_pending")

        # The outbox must still be deliverable once the endpoint is back.
        callback.status_code = 200
        await manager._deliver_callback(job_uuid)
        fresh = await manager.db.get(job_uuid)
        check(fresh.finalize_state == FinalizeState.DELIVERED,
              "callback was not delivered after the endpoint recovered")
    finally:
        await close_manager(manager)


async def test_missing_agent_output_fails_closed(tmp: Path) -> None:
    callback = CallbackRecorder()
    provider = FakeProvider()
    manager = await make_manager(tmp, provider, callback)

    async def fake_agent(**kwargs):
        return {}

    import jobmanager.manager as manager_module
    original = manager_module.run_job_agent
    manager_module.run_job_agent = fake_agent
    try:
        job_uuid = "no-agent-output"
        local_dir = tempfile.mkdtemp(prefix="catagent-job-")
        record = base_record(
            job_uuid, local_dir=local_dir, status=JobStatus.COMPLETED,
            finalize_state=FinalizeState.PENDING,
            sub_jobs=[sub_job(job_uuid, local_dir)])
        await save_record(manager, record)

        await manager._finalize_job(job_uuid)

        check(len(callback.payloads) == 1, "expected one callback")
        check(callback.payloads[0]["status"] == "error",
              "missing agent output defaulted to success")
    finally:
        manager_module.run_job_agent = original
        await close_manager(manager)


async def wait_terminal(provider: LocalProvider, pid: str,
                        timeout: float = 10.0) -> JobStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        states = await provider.get_job_states({pid})
        state = states.get(pid)
        if state not in (JobStatus.RUNNING, JobStatus.SUBMITTED, None):
            return state
        await asyncio.sleep(0.05)
    raise TimeoutError(f"local job {pid} did not finish")


def make_local_provider() -> LocalProvider:
    return LocalProvider("local-test", ["cpu"], {
        "test-env": {"softwares": [], "initialize_command": []},
    })


async def test_local_provider_exit_codes_and_restart(tmp: Path) -> None:
    os.environ["CATAGENT_LOCAL_STATE_DIR"] = str(tmp / "local-state")

    success_dir = tmp / "success"
    success_dir.mkdir()
    (success_dir / "run.sh").write_text("echo ok > result.txt\n",
                                        encoding="utf-8")
    provider = make_local_provider()
    pid = await provider.submit(JobSetting(
        job_name="ok", local_dir=str(success_dir), provider_name="local-test",
        job_env_name="test-env", idempotency_key="ok:a0"))
    check(await wait_terminal(provider, pid) == JobStatus.COMPLETED,
          "successful local job was not completed")

    # A fresh provider has no Popen handle; it must still read the persisted
    # exit code instead of assuming success.
    restarted = make_local_provider()
    check(await restarted.get_job_states({pid}) == {pid: JobStatus.COMPLETED},
          "restarted provider lost the successful exit code")

    # Re-submitting the same attempt is idempotent.
    same_pid = await restarted.submit(JobSetting(
        job_name="ok", local_dir=str(success_dir), provider_name="local-test",
        job_env_name="test-env", idempotency_key="ok:a0"))
    check(same_pid == pid, "idempotent submit started a second process")

    failure_dir = tmp / "failure"
    failure_dir.mkdir()
    (failure_dir / "run.sh").write_text("exit 3\n", encoding="utf-8")
    failing_pid = await provider.submit(JobSetting(
        job_name="bad", local_dir=str(failure_dir), provider_name="local-test",
        job_env_name="test-env", idempotency_key="bad:a0"))
    check(await wait_terminal(provider, failing_pid) == JobStatus.FAILED,
          "failing local job was not reported as failed")

    restarted = make_local_provider()
    check(await restarted.get_job_states({failing_pid})
          == {failing_pid: JobStatus.FAILED},
          "restart turned a failed job into success")

    # A dead pid with no status file is UNKNOWN, not COMPLETED.
    dead = subprocess.Popen(["true"])
    dead.wait()
    unknown_dir = tmp / "unknown"
    unknown_dir.mkdir()
    restarted._register(dead.pid, str(unknown_dir), "unknown:a0")
    check(await restarted.get_job_states({str(dead.pid)})
          == {str(dead.pid): JobStatus.UNKNOWN},
          "dead pid without a status file was assumed completed")


async def test_local_submit_recovers_published_pid(tmp: Path) -> None:
    """A crash between spawn and pid write must not start a second process."""
    os.environ["CATAGENT_LOCAL_STATE_DIR"] = str(tmp / "local-state-recover")
    job_dir = tmp / "recover"
    job_dir.mkdir()
    (job_dir / "run.sh").write_text("echo done\n", encoding="utf-8")
    run_dir = job_dir / ".catagent"
    run_dir.mkdir()
    (run_dir / "submit.json").write_text(
        json.dumps({"token": "recover:a0", "status": "launching",
                    "local_dir": str(job_dir)}), encoding="utf-8")
    live = subprocess.Popen(["sleep", "30"])
    (run_dir / "pid").write_text(str(live.pid), encoding="utf-8")
    try:
        provider = make_local_provider()
        pid = await provider.submit(JobSetting(
            job_name="recover", local_dir=str(job_dir),
            provider_name="local-test", job_env_name="test-env",
            idempotency_key="recover:a0"))
        check(pid == str(live.pid),
              f"recovery launched a second process: {pid} != {live.pid}")
        saved = json.loads((run_dir / "submit.json").read_text(encoding="utf-8"))
        check(saved.get("pid") == live.pid,
              f"recovered pid was not persisted: {saved}")
    finally:
        live.terminate()
        live.wait()


async def test_local_provider_runs_initialize_command(tmp: Path) -> None:
    os.environ["CATAGENT_LOCAL_STATE_DIR"] = str(tmp / "local-state-init")
    job_dir = tmp / "init"
    job_dir.mkdir()
    (job_dir / "run.sh").write_text("echo \"$CATAGENT_TEST_VAR\" > out.txt\n",
                                    encoding="utf-8")
    provider = LocalProvider("local-test", ["cpu"], {
        "test-env": {"softwares": [],
                     "initialize_command": ["export CATAGENT_TEST_VAR=hello"]},
    })
    pid = await provider.submit(JobSetting(
        job_name="init", local_dir=str(job_dir), provider_name="local-test",
        job_env_name="test-env", idempotency_key="init:a0"))
    check(await wait_terminal(provider, pid) == JobStatus.COMPLETED,
          "initialize_command job failed")
    check((job_dir / "out.txt").read_text(encoding="utf-8").strip() == "hello",
          "initialize_command was not applied before run.sh")


async def test_orchestrator_locks_are_per_thread() -> None:
    from catagent.agent import (
        get_orchestrator_sync, release_orchestrator_sync,
    )

    lock_a, claim_a = get_orchestrator_sync("user-a")
    lock_a_again, claim_a_again = get_orchestrator_sync("user-a")
    lock_b, claim_b = get_orchestrator_sync("user-b")
    check(lock_a is lock_a_again and claim_a is claim_a_again,
          "the same thread did not reuse its lock")
    check(lock_a is not lock_b, "different threads share one lock")

    order: list[tuple[str, str]] = []

    async def worker(tag: str, lock: asyncio.Lock) -> None:
        async with lock:
            order.append((tag, "start"))
            await asyncio.sleep(0.05)
            order.append((tag, "end"))

    await asyncio.gather(worker("a", lock_a), worker("b", lock_b))
    check([event for _, event in order[:2]] == ["start", "start"],
          f"different threads were serialized: {order}")

    # The same thread must still be serialized.
    order.clear()
    await asyncio.gather(worker("a1", lock_a), worker("a2", lock_a))
    check(order[0] == ("a1", "start") and order[1] == ("a1", "end"),
          f"same-thread work overlapped: {order}")

    claim_a.set()
    release_orchestrator_sync("user-a")
    check(get_orchestrator_sync("user-a")[0] is lock_a,
          "a claimed thread was released while busy")
    claim_a.clear()
    release_orchestrator_sync("user-a")
    lock_new, claim_new = get_orchestrator_sync("user-a")
    check(lock_new is not lock_a and claim_new is not claim_a,
          "idle thread entries were not released")
    release_orchestrator_sync("user-a")
    release_orchestrator_sync("user-b")


async def test_slurm_remote_path_quoting() -> None:
    """A leading ``~`` in slurm's remote_root must stay shell-expandable."""
    from jobmanager.providers.slurm import remote_shell_path

    quoted = remote_shell_path("~/jobs/abc")
    check(quoted == '"$HOME"/jobs/abc',
          f"tilde path was quoted into a literal: {quoted!r}")
    check("~" not in quoted, f"tilde survived shell quoting: {quoted!r}")
    check(remote_shell_path("~") == '"$HOME"',
          "bare tilde was not expanded safely")
    check(remote_shell_path("/tmp/a b/x") == "'/tmp/a b/x'",
          "a path with spaces was not quoted")


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="catagent-recovery-"))
    tests = [
        ("old database is migrated", test_old_database_is_migrated),
        ("submission resumes persisted plan", test_submission_resumes_persisted_plan),
        ("submission skips started groups", test_submission_skips_already_started_groups),
        ("explicit retry bumps attempt", test_explicit_retry_bumps_attempt),
        ("missing result files are an error", test_missing_result_files_are_an_error),
        ("crash during download is not success", test_crash_during_download_is_not_a_success),
        ("polling sweep retries finalizing", test_polling_sweep_retries_finalizing),
        ("callback 404 is not delivered", test_callback_404_is_not_delivered),
        ("missing agent output fails closed", test_missing_agent_output_fails_closed),
        ("local exit codes survive restart", test_local_provider_exit_codes_and_restart),
        ("local submit recovers published pid", test_local_submit_recovers_published_pid),
        ("initialize_command is applied", test_local_provider_runs_initialize_command),
        ("orchestrator locks are per thread", test_orchestrator_locks_are_per_thread),
        ("slurm remote path quoting", test_slurm_remote_path_quoting),
    ]
    for name, function in tests:
        case_dir = root / name.replace(" ", "_")
        case_dir.mkdir(parents=True, exist_ok=True)
        if function in (test_orchestrator_locks_are_per_thread,
                        test_slurm_remote_path_quoting):
            await function()
        else:
            await function(case_dir)
        print(f"  ok: {name}")
    print("\nJOBMANAGER RECOVERY TESTS: OK")
    print(f"workdir = {root}")


if __name__ == "__main__":
    asyncio.run(main())
