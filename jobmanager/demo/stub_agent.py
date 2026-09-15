"""A deterministic stand-in for the LLM job agent.

There is no calculation agent yet, so the demo must exercise the whole
jobmanager pipeline (submit -> poll -> download -> callback) without an LLM.

`stub_run_job_agent` mimics the signature/return value of
`jobmanager.agent.agent.run_job_agent`:

- preprocessing phase: reads `job_description.json` next to the inputs and
  reports the grouping (batch vs single job).
- checking phase: inspects the downloaded results and returns a
  job_result_status. Two markers let you force a failure:

    DEMO_INPUT_FAIL  -> "InputError"      (no retry, error reported)
    DEMO_ENV_FAIL    -> "EnvironmentError" (job is resubmitted up to the limit)

It also streams a few messages to `front_end_url`, so the demo frontend can
show the agent progress, exactly like the real agent does.
"""

import glob
import json
import logging
import os

logger = logging.getLogger("jobmanager")


def _read_json(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


async def _stream(http_client, url, payload: dict) -> None:
    if not (http_client and url):
        return
    try:
        await http_client.post(url=url, json=payload)
    except Exception as e:  # streaming is best-effort
        logger.warning(f"[stub-agent] stream post failed: {e}")


async def stub_run_job_agent(agent,
                             phase: str,
                             thread_id: str,
                             local_dir: str,
                             log_msg: bool = False,
                             provider_id: str = None,
                             job_env_name: str = None,
                             http_client=None,
                             url: str = None) -> dict:
    await _stream(http_client, url, {
        "role": "agent",
        "message": f"[stub-agent:{phase}] working on {local_dir}",
        "tool_calls": "",
    })

    if phase == "preprocessing":
        desc = _read_json(os.path.join(local_dir, "job_description.json"))
        batch_job = bool(desc.get("batch_job", False))
        output = {"batch_job": batch_job, "n_jobs": int(desc.get("n_jobs", 1) or 1)}
        if batch_job:
            groups = sorted(d for d in glob.glob(os.path.join(local_dir, "group_*"))
                            if os.path.isdir(d))
            output["n_groups"] = len(groups)
            output["group_dirs_wildcard"] = "group_*"
        else:
            output["n_groups"] = None
            output["group_dirs_wildcard"] = None
        logger.info(f"[stub-agent] preprocessing output: {output}")
        await _stream(http_client, url, {"role": "tool", "message": f"[stub-agent] {output}"})
        return {"phase": "preprocessing", "preprocessing_output": output, "checking_output": None}

    # checking phase
    if os.path.exists(os.path.join(local_dir, "DEMO_INPUT_FAIL")):
        result = {"job_result_status": "InputError",
                  "error_message": "DEMO_INPUT_FAIL marker found: simulated bad input."}
    elif os.path.exists(os.path.join(local_dir, "DEMO_ENV_FAIL")):
        result = {"job_result_status": "EnvironmentError",
                  "error_message": "DEMO_ENV_FAIL marker found: simulated environment failure."}
    else:
        result = {"job_result_status": "Success", "error_message": ""}
    logger.info(f"[stub-agent] checking output: {result}")
    await _stream(http_client, url, {"role": "tool", "message": f"[stub-agent] {result}"})
    return {"phase": "checking", "preprocessing_output": None, "checking_output": result}
