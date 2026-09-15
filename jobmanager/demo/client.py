"""Interactive stand-in for the calculation agent.

There is no calculation agent yet, so this CLI lets you type the fields of a
JobRequest by hand. It also plays the role of the parent agent that prepares
the inputs: it creates the job directory, `run.sh` and `job_description.json`,
then POSTs the JobRequest to the job manager and polls `/status` until the job
is finished.

Run it in its own terminal while `frontend` and `server` are running:

    python -m jobmanager.demo.client --url http://127.0.0.1:8000
"""

import argparse
import json
import os
import sys
import time
import uuid as uuidlib

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))

TERMINAL_STATUSES = {"completed", "failed", "cancelled", "unknown"}

RUN_SH = """#!/bin/bash
# Demo workload: write progress into run.stdout, then exit.
cd "$(dirname "$0")"
{
  echo "job start $(date)"
  for i in 1 2 3; do
    echo "step ${i}"
    sleep 1
  done
  echo "job done $(date)"
} > run.stdout
"""


def prompt(text: str, default=None, required: bool = False):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        value = input(f"{text}{suffix}: ").strip()
        if value:
            return value
        if default is not None:
            return default
        if not required:
            return ""
        print("  (this field is required)")


def prompt_choice(text: str, choices: list, default=None):
    print(f"{text} options: {', '.join(choices)}")
    while True:
        value = prompt(text, default=default)
        if value in choices:
            return value
        print(f"  (choose one of: {', '.join(choices)})")


def prompt_bool(text: str, default: bool = False) -> bool:
    default_str = "y" if default else "n"
    value = prompt(f"{text} (y/n)", default=default_str).lower()
    return value in ("y", "yes", "true", "1")


def build_job_dir(base_dir: str, job_uuid: str, batch: bool, n_groups: int,
                  input_fail: bool, env_fail: bool) -> str:
    job_dir = os.path.join(base_dir, job_uuid)
    os.makedirs(job_dir, exist_ok=True)

    if batch:
        for i in range(n_groups):
            group_dir = os.path.join(job_dir, f"group_{i}")
            os.makedirs(group_dir, exist_ok=True)
            with open(os.path.join(group_dir, "run.sh"), "w") as f:
                f.write(RUN_SH)
    else:
        with open(os.path.join(job_dir, "run.sh"), "w") as f:
            f.write(RUN_SH)
        os.chmod(os.path.join(job_dir, "run.sh"), 0o755)

    description = {
        "description": "Demo job typed from the command line.",
        "job_name": job_uuid,
        "use_gpu": False,
        "batch_job": batch,
        "n_jobs": n_groups if batch else 1,
        "workload": "mini",
        "placeholders": {},
    }
    with open(os.path.join(job_dir, "job_description.json"), "w") as f:
        json.dump(description, f, indent=2)

    if input_fail:
        open(os.path.join(job_dir, "DEMO_INPUT_FAIL"), "w").close()
    if env_fail:
        open(os.path.join(job_dir, "DEMO_ENV_FAIL"), "w").close()
    return job_dir


def wait_for_terminal(http: httpx.Client, job_uuid: str) -> dict:
    last = None
    while True:
        resp = http.get(f"/status/{job_uuid}")
        if resp.status_code == 404:
            print("  (job not visible yet ...)")
            time.sleep(1)
            continue
        recorder = resp.json()
        status = recorder["status"]
        if status != last:
            print(f"  status: {status}")
            last = status
        if status in TERMINAL_STATUSES:
            return recorder
        time.sleep(1)


def submit_one(http: httpx.Client, providers: dict, base_dir: str) -> None:
    provider_id = prompt_choice("provider_id", list(providers.keys()), default=next(iter(providers)))
    envs = list(providers[provider_id].get("job_envs", {}).keys())
    if envs:
        job_env_name = prompt_choice("job_env_name", envs, default=envs[0])
    else:
        job_env_name = prompt("job_env_name", required=True)

    default_uuid = uuidlib.uuid4().hex[:12]
    job_uuid = prompt("job_uuid", default=default_uuid)
    description = prompt("description", default="demo job")
    wildcard = prompt("result_patterns (comma separated)", default="*")
    result_patterns = [p.strip() for p in wildcard.split(",") if p.strip()]

    # Empty -> generate a self-contained demo job; otherwise use an existing dir
    # (with your own run.sh / job_description.json) as the parent agent would.
    local_dir_input = prompt("local_dir (empty = auto-create a demo job)", default="")

    batch = False
    n_groups = 1
    input_fail = env_fail = False
    if local_dir_input:
        job_dir = os.path.abspath(os.path.expanduser(local_dir_input))
        if not os.path.isdir(job_dir):
            print(f"  {job_dir} is not a directory, aborting.")
            return
        if not os.path.isfile(os.path.join(job_dir, "run.sh")):
            print("  warning: no run.sh in that directory; the provider will reject the submission.")
    else:
        batch = prompt_bool("batch_job", default=False)
        if batch:
            n_groups = int(prompt("number of groups", default="3"))
        print("Optional failure injection (leave both 'n' for a successful run):")
        input_fail = prompt_bool("simulate InputError (DEMO_INPUT_FAIL)", default=False)
        env_fail = prompt_bool("simulate EnvironmentError (DEMO_ENV_FAIL)", default=False)
        job_dir = build_job_dir(base_dir, job_uuid, batch, n_groups, input_fail, env_fail)
    request = {
        "job_uuid": job_uuid,
        "description": description,
        "local_dir": job_dir,
        "provider_id": provider_id,
        "job_env_name": job_env_name,
        "result_patterns": result_patterns,
    }
    print(f"\n--> POST /submit\n{json.dumps(request, indent=2)}")
    resp = http.post("/submit", json=request)
    print(f"<-- {resp.status_code} {resp.json()}\n")
    if resp.status_code != 200:
        return

    print("Waiting for the job to finish (watch the frontend terminal for the JobResult)...")
    record = wait_for_terminal(http, job_uuid)
    print(f"final status: {record['status']}   groups: {len(record.get('sub_jobs', []))}")
    if record.get("error"):
        print(f"error: {record['error']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive JobRequest client for the demo.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Job manager base URL.")
    parser.add_argument("--workdir", default=os.path.join(_HERE, "runs"),
                        help="Directory where demo job dirs are created.")
    args = parser.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    with httpx.Client(base_url=args.url, timeout=30) as http:
        providers = None
        last_error = None
        for _ in range(20):  # the server may still be importing dependencies
            try:
                providers = http.get("/providers").json()
                break
            except Exception as e:
                last_error = e
                time.sleep(0.5)
        if providers is None:
            print(f"Cannot reach the job manager at {args.url}: {last_error}")
            print("Start it first:  python -m jobmanager.demo.server")
            return 1

        if not providers:
            print("No providers configured.")
            return 1

        print(f"Connected to {args.url}. providers: {list(providers.keys())}")
        print(f"Demo job dirs are created under: {args.workdir}\n")
        while True:
            try:
                submit_one(http, providers, args.workdir)
            except (KeyboardInterrupt, EOFError):
                print("\nAborted.")
                break
            if not prompt_bool("\nSubmit another job?", default=False):
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())