"""Deterministic tests for Slurm ``sbatch`` output parsing.

Run from the repository root:

    python -m check.test_slurm_provider

No SSH connection or cluster is used: the tests call the pure parser with
captured stdout.  They cover the regression where the remote id was taken from
the last stdout line's last token.  On this cluster a login-shell banner
(``TH Storage Info``) is printed asynchronously, so its decorative
``--------------------------------------------------`` line can arrive *after*
``Submitted batch job <id>`` and was persisted as the remote id.
"""

from __future__ import annotations

from catagent.jobmanager.providers.slurm import _parse_submit_job_id

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
    print("SLURM PROVIDER TEST: OK")


if __name__ == "__main__":
    main()
