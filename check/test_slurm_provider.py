"""Deterministic tests for the Slurm provider.

Run from the repository root:

    python -m check.test_slurm_provider

No SSH connection or cluster is used.  The tests cover:

* ``sbatch`` output parsing (the remote id used to be taken from the last
  stdout line's last token, so an asynchronous login-shell banner could be
  persisted instead of the real id);
* mapping of ``scontrol``/``sacct`` state strings to ``JobStatus``;
* falling back to ``sacct`` when Slurm has purged a completed job from
  ``scontrol`` (``MinJobAge`` defaults to 5 minutes);
* reconnecting a dead paramiko session before a status query.
"""

from __future__ import annotations

import asyncio
import threading

from catagent.jobmanager.models import JobStatus
from catagent.jobmanager.providers.slurm import (
    SlurmProvider,
    _job_status_from_slurm_state,
    _parse_sacct_state,
    _parse_scontrol_state,
    _parse_submit_job_id,
)

BANNER = (
    "\n"
    "-------------   TH Storage Info   ----------------\n"
    "\n"
    "HomePath:    /HOME/user   HomeUsed: 4.84G   HomeQuota: 100G\n"
    "\n"
    "--------------------------------------------------\n"
    "\n"
    "UserStorage: HDD_POOL UserStorageUsed: .18M UserStorageQuota: 0M\n"
    "\n"
    "--------------------------------------------------\n"
)


def test_parse_standard_output() -> None:
    assert _parse_submit_job_id("Submitted batch job 9330190\n") == "9330190"


def test_parse_banner_after_submit_message() -> None:
    # The old implementation returned 50 hyphens for exactly this output.
    stdout = "Submitted batch job 9330190\n" + BANNER
    assert _parse_submit_job_id(stdout) == "9330190"


def test_parse_banner_before_submit_message() -> None:
    stdout = BANNER + "Submitted batch job 9330190\n"
    assert _parse_submit_job_id(stdout) == "9330190"


def test_parse_ignores_leading_warnings() -> None:
    stdout = (
        "sbatch: warning: something harmless\n"
        "Submitted batch job 9330190\n"
        "sbatch: another trailing warning\n"
    )
    assert _parse_submit_job_id(stdout) == "9330190"


def test_parse_cluster_suffix() -> None:
    # Newer Slurm versions may append "on cluster <name>".
    stdout = "Submitted batch job 9330190 on cluster deimos\n"
    assert _parse_submit_job_id(stdout) == "9330190"


def test_parse_crlf_and_tabs() -> None:
    stdout = "Submitted batch job\t9330190\r\n"
    assert _parse_submit_job_id(stdout) == "9330190"


def test_parse_rejects_banner_only() -> None:
    assert _parse_submit_job_id(BANNER) is None


def test_parse_rejects_missing_message() -> None:
    assert _parse_submit_job_id("") is None
    assert _parse_submit_job_id("sbatch: error: invalid partition\n") is None


def test_parse_rejects_zero() -> None:
    assert _parse_submit_job_id("Submitted batch job 0\n") is None


def test_parse_same_id_twice() -> None:
    stdout = "Submitted batch job 9330190\nSubmitted batch job 9330190\n"
    assert _parse_submit_job_id(stdout) == "9330190"


def test_parse_rejects_ambiguous_ids() -> None:
    stdout = "Submitted batch job 9330190\nSubmitted batch job 9330191\n"
    try:
        _parse_submit_job_id(stdout)
    except RuntimeError as exc:
        assert "multiple job ids" in str(exc)
    else:
        raise AssertionError("multiple distinct job ids must be rejected")


# ---------------------------------------------------------------------------
# State mapping / parsing
# ---------------------------------------------------------------------------

def test_scontrol_state_is_parsed_from_oneliner() -> None:
    stdout = ("JobId=9330352 JobName=Rh-sp-dataset1-g3 UserId=pxyl281(23632) "
              "JobState=RUNNING Reason=None Dependency=(null)\n")
    assert _parse_scontrol_state(stdout) == "RUNNING"
    # A banner before the real line must not confuse the parser.
    assert _parse_scontrol_state(BANNER + stdout) == "RUNNING"
    assert _parse_scontrol_state(
        "slurm_load_jobs error: Invalid job id specified\n") is None


def test_maps_running_states() -> None:
    for raw in ("RUNNING", "COMPLETING", "CONFIGURING", "RESUMING",
                "running"):
        assert _job_status_from_slurm_state(raw) is JobStatus.RUNNING, raw


def test_maps_pending_states() -> None:
    for raw in ("PENDING", "SUSPENDED"):
        assert _job_status_from_slurm_state(raw) is JobStatus.SUBMITTED, raw


def test_maps_terminal_states() -> None:
    assert _job_status_from_slurm_state("COMPLETED") is JobStatus.COMPLETED
    assert _job_status_from_slurm_state("CANCELLED") is JobStatus.CANCELLED
    # sacct writes cancellation as "CANCELLED by <uid>", and appends "+" to
    # some states when the job was killed by a signal.
    assert _job_status_from_slurm_state(
        "CANCELLED by 1234") is JobStatus.CANCELLED
    assert _job_status_from_slurm_state("CANCELLED+") is JobStatus.CANCELLED
    for raw in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED",
                "SPECIAL_EXIT"):
        assert _job_status_from_slurm_state(raw) is JobStatus.FAILED, raw


def test_unknown_states_are_not_terminal() -> None:
    for raw in ("", "UNKNOWN", "NONE", "SOMETHING_NEW"):
        assert _job_status_from_slurm_state(raw) is None, raw


def test_sacct_state_must_match_the_exact_job_id() -> None:
    stdout = ("9330314|COMPLETED\n"
              "9330314.batch|COMPLETED\n"
              "9330314.extern|COMPLETED\n")
    assert _parse_sacct_state(stdout, "9330314") == "COMPLETED"


def test_sacct_state_ignores_job_steps() -> None:
    stdout = "9330314.batch|COMPLETED\n9330314.extern|COMPLETED\n"
    assert _parse_sacct_state(stdout, "9330314") is None


def test_sacct_parses_spaces_and_empty_output() -> None:
    assert _parse_sacct_state(" 9330314 | COMPLETED \n", "9330314") == \
        "COMPLETED"
    assert _parse_sacct_state("", "9330314") is None
    assert _parse_sacct_state("slurm_load_jobs error\n", "9330314") is None


# ---------------------------------------------------------------------------
# Purged-job fallback and reconnection
# ---------------------------------------------------------------------------

def _bare_provider() -> SlurmProvider:
    """Build a provider without connecting to a cluster."""
    provider = SlurmProvider.__new__(SlurmProvider)
    provider.name = "slurm-test"
    return provider


def _fake_output(stdout: str = "", stderr: str = "",
                 status: int = 0):
    return [[], stdout.splitlines(keepends=True),
            stderr.splitlines(keepends=True), [str(status)]]


def test_running_job_is_read_from_scontrol() -> None:
    provider = _bare_provider()
    calls: list[str] = []

    def fake_run(command, retry_on_disconnect=False):
        calls.append(command)
        return _fake_output(
            "JobId=9330352 JobState=RUNNING Reason=None\n")

    provider._run_ssh = fake_run
    assert asyncio.run(provider._get_one_job_state("9330352")) is \
        JobStatus.RUNNING
    assert len(calls) == 1 and calls[0].startswith("scontrol")
    assert "sacct" not in calls[0]


def test_completed_job_purged_from_scontrol_uses_sacct() -> None:
    provider = _bare_provider()
    calls: list[str] = []

    def fake_run(command, retry_on_disconnect=False):
        calls.append(command)
        if command.startswith("scontrol"):
            return _fake_output(
                stderr="slurm_load_jobs error: Invalid job id specified\n",
                status=1)
        return _fake_output("9330314|COMPLETED\n")

    provider._run_ssh = fake_run
    assert asyncio.run(provider._get_one_job_state("9330314")) is \
        JobStatus.COMPLETED
    assert calls[0].startswith("scontrol")
    assert calls[1].startswith("sacct")


def test_completed_job_uses_plain_sacct_state() -> None:
    provider = _bare_provider()

    def fake_run(command, retry_on_disconnect=False):
        if command.startswith("scontrol"):
            return _fake_output(
                stderr="slurm_load_jobs error: Invalid job id specified\n",
                status=1)
        return _fake_output("9330314|CANCELLED by 1234\n")

    provider._run_ssh = fake_run
    assert asyncio.run(provider._get_one_job_state("9330314")) is \
        JobStatus.CANCELLED


def test_job_absent_from_both_sources_is_unknown() -> None:
    provider = _bare_provider()

    def fake_run(command, retry_on_disconnect=False):
        return _fake_output(
            stderr="slurm_load_jobs error: Invalid job id specified\n",
            status=1)

    provider._run_ssh = fake_run
    assert asyncio.run(provider._get_one_job_state("9330314")) is \
        JobStatus.UNKNOWN


def test_state_query_retries_after_connection_dies() -> None:
    provider = _bare_provider()
    provider._ssh_lock = threading.RLock()
    provider._user = "user"
    provider._host = "host"
    provider.ssh_client = _FakeClient(active=True, die_on_exec=True)
    reconnects: list[int] = []

    def fake_connect():
        reconnects.append(1)
        provider.ssh_client = _FakeClient(
            active=True,
            stdout="JobId=9330352 JobState=RUNNING Reason=None\n")

    provider._connect_locked = fake_connect

    result = provider._run_ssh("scontrol show jobid -o 9330352",
                               retry_on_disconnect=True)
    assert result is not None
    assert result[1] == ["JobId=9330352 JobState=RUNNING Reason=None\n"]
    assert len(reconnects) == 1


def test_run_ssh_reconnects_dead_transport_before_exec() -> None:
    provider = _bare_provider()
    provider._ssh_lock = threading.RLock()
    provider._user = "user"
    provider._host = "host"
    provider.ssh_client = _FakeClient(active=False)
    reconnects: list[int] = []

    def fake_connect():
        reconnects.append(1)
        provider.ssh_client = _FakeClient(active=True, stdout="ok\n")

    provider._connect_locked = fake_connect

    result = provider._run_ssh("echo ok")
    assert result is not None and result[1] == ["ok\n"]
    assert len(reconnects) == 1


class _FakeChannel:
    def __init__(self, lines=()):
        self._lines = list(lines)
        self.channel = self

    def readlines(self):
        return list(self._lines)

    def close(self):
        pass

    def recv_exit_status(self):
        return 0


class _FakeTransport:
    def __init__(self, active: bool):
        self._active = active

    def is_active(self) -> bool:
        return self._active


class _FakeClient:
    """Minimal paramiko.SSHClient stand-in for reconnection tests."""

    def __init__(self, active: bool, stdout: str = "",
                 die_on_exec: bool = False):
        self._transport = _FakeTransport(active)
        self._stdout = stdout
        self._die_on_exec = die_on_exec

    def get_transport(self):
        return self._transport

    def exec_command(self, command):
        if self._die_on_exec:
            # Emulate a transport that dies between the liveness check and
            # exec_command: the session is unusable and the transport reports
            # inactive on the retry path.
            self._transport._active = False
            import paramiko
            raise paramiko.SSHException("SSH session not active")
        return (_FakeChannel(), _FakeChannel([self._stdout]),
                _FakeChannel())

    def close(self):
        pass


def main() -> None:
    test_parse_standard_output()
    test_parse_banner_after_submit_message()
    test_parse_banner_before_submit_message()
    test_parse_ignores_leading_warnings()
    test_parse_cluster_suffix()
    test_parse_crlf_and_tabs()
    test_parse_rejects_banner_only()
    test_parse_rejects_missing_message()
    test_parse_rejects_zero()
    test_parse_same_id_twice()
    test_parse_rejects_ambiguous_ids()
    test_scontrol_state_is_parsed_from_oneliner()
    test_maps_running_states()
    test_maps_pending_states()
    test_maps_terminal_states()
    test_unknown_states_are_not_terminal()
    test_sacct_state_must_match_the_exact_job_id()
    test_sacct_state_ignores_job_steps()
    test_sacct_parses_spaces_and_empty_output()
    test_running_job_is_read_from_scontrol()
    test_completed_job_purged_from_scontrol_uses_sacct()
    test_completed_job_uses_plain_sacct_state()
    test_job_absent_from_both_sources_is_unknown()
    test_state_query_retries_after_connection_dies()
    test_run_ssh_reconnects_dead_transport_before_exec()
    print("SLURM PROVIDER TEST: OK")


if __name__ == "__main__":
    main()
