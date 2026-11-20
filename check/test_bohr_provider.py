"""Deterministic tests for Bohr output parsing.

Run from the repository root:

    python -m check.test_bohr_provider

No ``bohr`` command is executed: the subprocess helper is replaced by a fake
that returns captured output.  The tests cover the regression where the job id
was read from a fixed stdout line, which stored ``0`` after a submission that
Bohrium rejected internally.
"""

from __future__ import annotations

import asyncio
import json

import catagent.jobmanager.providers.bohr as bohr_module
from catagent.jobmanager.models import JobStatus
from catagent.jobmanager.providers.bohr import BohrProvider, _parse_submit_job_id


def test_parse_standard_output() -> None:
    stdout = (
        "Submit job succeed.\n"
        "JobId:  23777027\n"
        "JobGroupId:  16739648\n"
    )
    assert _parse_submit_job_id(stdout) == "23777027"


def test_parse_ignores_leading_noise() -> None:
    # The old implementation used ``splitlines()[1]`` and read this as "job".
    stdout = (
        "Uploading input files... done\n"
        "Some warning from the CLI\n"
        "Submit job succeed.\n"
        "JobId: 23777030\n"
        "JobGroupId: 16739649\n"
    )
    assert _parse_submit_job_id(stdout) == "23777030"


def test_parse_single_line_variant() -> None:
    stdout = "Submit job succeed. JobId: 23777033 JobGroupId: 16739652\n"
    assert _parse_submit_job_id(stdout) == "23777033"


def test_parse_crlf_and_spacing() -> None:
    stdout = "Submit job succeed.\r\nJobId:\t23777032\r\nJobGroupId: 16739651\r\n"
    assert _parse_submit_job_id(stdout) == "23777032"


def test_parse_does_not_use_job_group_id() -> None:
    stdout = "Submit job succeed.\nJobGroupId: 16739648\n"
    assert _parse_submit_job_id(stdout) is None


def test_parse_rejects_zero() -> None:
    stdout = (
        "Submit job succeed.\n"
        "JobId: 0\n"
        "JobGroupId: 0\n"
    )
    assert _parse_submit_job_id(stdout) is None


def test_parse_rejects_missing_id() -> None:
    assert _parse_submit_job_id("Submit job succeed.\n") is None
    assert _parse_submit_job_id("") is None


def test_parse_rejects_ambiguous_ids() -> None:
    stdout = (
        "Submit job succeed.\n"
        "JobId: 23777027\n"
        "JobId: 23777030\n"
    )
    try:
        _parse_submit_job_id(stdout)
    except RuntimeError as exc:
        assert "multiple JobIds" in str(exc)
    else:
        raise AssertionError("multiple distinct JobIds must be rejected")


async def _state_with_output(
        jid: str, output: dict) -> tuple[tuple[str, JobStatus], list[str]]:
    calls: list[str] = []

    async def fake_run_shell_command(command: str, *args, **kwargs) -> dict:
        calls.append(command)
        return output

    original = bohr_module.run_shell_command
    bohr_module.run_shell_command = fake_run_shell_command
    try:
        provider = object.__new__(BohrProvider)
        result = await provider._get_one_job_state(jid, asyncio.Semaphore(1))
    finally:
        bohr_module.run_shell_command = original
    return result, calls


async def test_status_uses_json() -> None:
    payload = [{
        "id": 23777027,
        "statusStr": "Running",
        "status": 1,
    }]
    (jid, status), calls = await _state_with_output(
        "23777027", {"return_code": 0, "stdout": json.dumps(payload), "stderr": ""})
    assert (jid, status) == ("23777027", JobStatus.RUNNING)
    assert calls == ["bohr job describe -j 23777027 --json"]


async def test_status_terminal_values() -> None:
    for word, expected in [
        ("Success", JobStatus.COMPLETED),
        ("Finished", JobStatus.COMPLETED),
        ("Failed", JobStatus.FAILED),
        ("Stopped", JobStatus.CANCELLED),
        ("Pending", JobStatus.SUBMITTED),
    ]:
        payload = [{"id": 23777027, "statusStr": word}]
        (_, status), _ = await _state_with_output(
            "23777027",
            {"return_code": 0, "stdout": json.dumps(payload), "stderr": ""})
        assert status == expected, f"{word}: {status}"


async def test_status_invalid_id_is_failed_without_query() -> None:
    (jid, status), calls = await _state_with_output(
        "0", {"return_code": 0, "stdout": "", "stderr": ""})
    assert (jid, status) == ("0", JobStatus.FAILED)
    assert calls == []


async def test_status_api_error_is_none() -> None:
    (jid, status), _ = await _state_with_output(
        "999999999",
        {"return_code": 0,
         "stdout": "Error: title: 请求错误, message: record not found\n",
         "stderr": ""})
    assert (jid, status) == ("999999999", JobStatus.NONE)


async def test_status_bad_json_is_none() -> None:
    (jid, status), _ = await _state_with_output(
        "23777027", {"return_code": 0, "stdout": "not json", "stderr": ""})
    assert (jid, status) == ("23777027", JobStatus.NONE)


async def main() -> None:
    test_parse_standard_output()
    test_parse_ignores_leading_noise()
    test_parse_single_line_variant()
    test_parse_crlf_and_spacing()
    test_parse_does_not_use_job_group_id()
    test_parse_rejects_zero()
    test_parse_rejects_missing_id()
    test_parse_rejects_ambiguous_ids()
    await test_status_uses_json()
    await test_status_terminal_values()
    await test_status_invalid_id_is_failed_without_query()
    await test_status_api_error_is_none()
    await test_status_bad_json_is_none()
    print("BOHR PROVIDER TEST: OK")


if __name__ == "__main__":
    asyncio.run(main())
