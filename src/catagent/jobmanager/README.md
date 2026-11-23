# Job Manager

This is a job manager which can be used as a harness for material computation agent. It hides all the details of job submission and downloading from the main agent, let it focus on the calculation.

The main agent should have the ability to actively interrupt, for example, the `interrupt` api of `langgraph`. The workflow is:

- Main agent prepares the input files for job, but it doesn't care about the details resources allocation, machine or partition selection. 
- Main agent get the informations of providers from job manager. It will return minimal informations of softwares, and if the provider have CPU or GPU. Main agent select a specific provider and job environment to ensure the consistency of calculation.
- Main agent post job request to the manager, and interrupt to wait for the results.
- When the job is done, the job manager post back the results and resume the main agent.

Main agent should provide some informations of the job: run.sh and job_description.json

## Job inputs

The main agent owns the scientific content of a job, the job manager owns everything
about *where* and *how* it runs. Concretely the main agent must guarantee that every
input directory is complete and self-contained, and must provide two extra files:

### `run.sh`

The submission script. It must not encode any resource allocation (no `#SBATCH`,
no `mpirun -n <number>`, no machine specific paths). Use placeholders instead:

```bash
for d in sp_*
do
  mpirun -n <np> vasp_std > vasp.stdout
done
```

The placeholders are declared and explained in `job_description.json`. The job
manager agent replaces them during preprocessing, adds the environment
initialization commands and, on Slurm, the `#SBATCH` directives.

The main agent also provides the raw inputs (`INCAR`, `POSCAR`, `KPOINTS`,
`POTCAR`, ...). For NEB the directory must contain the images:

```bash
00 01 02 03 04 INCAR KPOINTS POTCAR
```

### `job_description.json`

```json
{
  "description": "A batch of vasp jobs to do structure relaxation.",
  "job_name": "NH3-Fe-relax",
  "use_gpu": false,
  "batch_job": true,
  "n_jobs": 500,
  "workload": "medium",
  "placeholders": {
    "np": "The number of mpi processes",
    "model_name": "Name of pre-trained model.",
    "model_path": "Path to the model."
  }
}
```

| Field | Meaning |
| --- | --- |
| `description` | Short description of the job. |
| `job_name` | Job name. For a batch job it is used as a prefix. |
| `use_gpu` | Whether the job needs a GPU. |
| `batch_job` | Whether this is a batch job. |
| `n_jobs` | Total number of jobs (`1` when `batch_job` is `false`). |
| `workload` | Average workload per job: `mini`, `low`, `medium`, `high`, `ultra high`. |
| `placeholders` | Every placeholder in `run.sh`, mapped to its meaning. |

For a batch job the job manager agent groups the sub-directories and reports the
number of groups plus a wildcard (for example `group_*`); the manager then submits
each group as one remote job. See `agent/skill.md` for the full preprocessing rules
(machine/partition selection, `NCORE`, NEB `np / n_images`, parallel jobs on
exclusive Slurm nodes, ...).

## HTTP API

The manager is a FastAPI app. All paths are relative to the server root.

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | – | `{"status": "ok"}` |
| `POST` | `/submit` | `JobRequest` | `{"status": "success"\|"error", "message": "..."}` |
| `POST` | `/cancel/{job_uuid}` | – | `{"status": "success"\|"error", "message": "..."}` |
| `GET` | `/status/{job_uuid}` | – | `JobRecord` (404 if unknown) |
| `GET` | `/jobs?limit=200` | – | list of `JobRecord` |
| `GET` | `/providers` | – | provider id → softwares / CPU-GPU info |

`JobRequest`:

```json
{
  "job_uuid": "0f0f...",
  "description": "A batch of vasp jobs to do structure relaxation.",
  "local_dir": "/abs/path/to/inputs",
  "provider_id": "slurm-1",
  "job_env_name": "vasp-6.4.0",
  "result_patterns": ["OUTCAR", "CONTCAR", "vasp.stdout", "slurm*.out"]
}
```

- `job_uuid` is chosen by the main agent and must be unique.
- `local_dir` is both the submission directory and the download directory; results
  are written back into it.
- `result_patterns` selects the files to bring back. On Bohr it is passed as
  `--backward_files`, on Slurm the whole remote directory is archived, and on the
  local provider it is not a transfer filter but is still used to verify that the
  declared result files exist before a `completed` job is reported as successful.

`/submit` returns as soon as the request is accepted; preprocessing and submission
run in a background task. With a batch job, one directory per group is created on the
remote and tracked as a *sub job*.

### Callback

When a job reaches a terminal state the manager POSTs a `JobResult` to
`callback_url`:

```json
{
  "job_uuid": "0f0f...",
  "local_dir": "/abs/path/to/inputs",
  "status": "success",
  "error_msg": ""
}
```

`status` is `success` or `error`. The result is decided once, persisted
(`result_status`/`result_error`), and then delivered from that persisted payload.
Delivery is retried `retry_callback` times; only a 2xx response marks the job as
delivered. A `404` is **not** treated as success: if the main agent is gone, the
callback stays pending and is retried (and re-posted at the next startup), so a
result is never silently lost. Delivery is therefore at-least-once.

A `callback_pending` row is an outbox entry: it is always paired with a persisted
`result_status`, so a crash cannot turn a failed download into a success callback.

A failed transfer is not a result. If the archive cannot be fetched or fails its
integrity check, the row stays in `FINALIZING` and the idempotent download is
retried on later polls, up to `max_finalize_retries`. Only a verified archive
whose declared `result_patterns` are still missing is a terminal `error`. This
distinction matters because reporting "results lost" for a transient SCP/tar
failure makes the StepNode submit a duplicate job.

Result classification (produced by the job manager agent when the results are
downloaded):

| Status | Meaning |
| --- | --- |
| `Success` | No error. |
| `InputError` | Caused by the inputs (bad script, wrong `INCAR`/`json` parameter, ...). Reported to the main agent, **not** retried. |
| `EnvironmentError` | Caused by the network, the resources or the software environment (file/command missing, CUDA mismatch, ...). The job is resubmitted, up to `n_retry_job_submit` times. |

### Progress streaming

While the agent works, `run_job_agent` POSTs its messages to `front_end_url`, so a
front end can show live progress:

```json
{"role": "agent", "message": "...", "tool_calls": "...", "phase": "preprocessing"}
{"role": "tool",  "message": "...", "phase": "preprocessing"}
```

This is best-effort: a failure to stream never affects the job.  The cat-agent
debug console exposes `POST /jobmanager/messages` for these payloads; set
`front_end_url` to `http://127.0.0.1:5600/jobmanager/messages` to see them in
the console's right-hand panel.

### Job statuses

`JobRecord.status` aggregates the statuses of its sub jobs:

`submitted → running → completed`, or a terminal `failed` / `cancelled` / `unknown`.
`none` means the status query itself failed. Once **all** sub jobs are terminal the
manager downloads the results, runs the checking agent and posts the callback.

### Crash recovery

Submission and finalization are durable state machines stored on the `jobs` row:

| Column | Values |
| --- | --- |
| `submission_state` | `planning → planned → submitting → submitted` (or `failed`) |
| `finalize_state` | `none → pending → finalizing → ready → delivered` |
| `result_status` | `success` / `error`, written once |
| `result_error` | error string carried by the callback |

- The group plan is written **before** the first provider call, and each remote id
  is written immediately after its own submission. A restart resumes only the
  groups that have no remote id yet, so nothing is submitted twice.
- Each group has an idempotency key `job_uuid:gN:aM` (attempt `M`). A crash retry
  reuses the same key; an explicit retry increments the attempt and starts fresh.
- `finalize_state` is claimed atomically. A crash mid-download leaves the row in
  `finalizing`; the polling loop retries it (and startup retries it after a
  restart) instead of guessing a result.
- A `completed` job is reported as an error if **any** declared
  `result_patterns` file is missing in any group (producing only some of the
  declared outputs is not a success). If the checking agent produces no
  `job_result_status`, the result fails closed rather than defaulting to success.
- Startup recovery order is: incomplete submissions → pending finalizations →
  pending callbacks.

The local provider additionally stores an atomic exit-code file at
`<local_dir>/.catagent/status.json`, an idempotent submission marker at
`<local_dir>/.catagent/submit.json`, and a pid → job-directory registry under
`$CATAGENT_LOCAL_STATE_DIR` (default `$TMPDIR/catagent-local-jobs`). After a
restart, a dead pid **without** a status file is reported as `unknown`, never
`completed`. Jobs run in their own session so `cancel` can signal the whole
process group.

## Failure modes and invariants (for maintainers)

A four-case run against the real `bohr` and `slurm-1` providers
(`check/test_agent_real_jobmanager.py`) exposed a chain of failures. They are
fixed, but the same mistakes are easy to reintroduce when adding a provider or
moving the state machine to another database. This section records the incident
and the invariants the current code relies on.

### What went wrong

1. **A non-terminal provider state was treated as terminal.** Bohr returned an
   empty/header-only `describe` output and Slurm returned `COMPLETING`; both were
   mapped to `UNKNOWN`. The manager considered the job finished a few seconds
   after submission, and `bohr job download` then failed with
   `index out of range [0] with length 0`.
2. **One `scp.SCPClient` was shared by concurrent downloads.** The client keeps
   the channel and transfer state on the instance, so two simultaneous Slurm
   downloads produced `Underlying channel was closed` and a truncated archive.
   The truncated (or other job's) archive was extracted into the local workdir,
   which is how Cu results appeared in the `slurm1_Si` directory.
3. **A transfer failure was reported as a result.** After the download retry
   loop the old code emitted `Only part of job results are downloaded, the
   remaining part are lost.` That is a statement about the calculation, but the
   only evidence was a failed SCP/tar operation. The StepNodes reacted by
   submitting the same job again (observed 2 submissions for `bohr_Si`,
   `slurm1_Si`, `slurm1_Cu`; 1 for `bohr_Cu`). Those were LLM-driven recovery
   calls, not a code retry loop. A duplicate job can run on the cluster and
   return a second result.
4. **The callback raced resume-target registration.** The result could be ready
   before `/set-thread-id` stored `(thread_id, interrupt_id)`. The callback had an
   empty `interrupt_id`, was rejected with HTTP 400, exhausted `retry_callback`,
   and the row stayed at `finalize_state=ready`.
5. **A leading `~` was quoted into a literal path.** Slurm's `remote_root` is
   `~/jobs`. `shlex.quote(remote_dir)` produced `'~/jobs/<uuid>'`, so the remote
   shell did not expand the tilde and the download failed with
   `cd: ~/jobs/<uuid>: No such file or directory`. This regression was caught by
   the real-provider test, not by the offline tests.

### Invariants

**Status and finalization**

- `JobStatus.NONE` means *not known yet*. It must never be written as a
  sub-job/job terminal state and must not trigger a download or callback.
- `JobStatus.UNKNOWN` is terminal only when the provider gave a definitive
  answer (for example Slurm `Invalid job id`). Unrecognized or empty output maps
  to `NONE`.
- A transfer error is not a result. It returns `download_retry`, increments
  `finalize_attempt`, and leaves `finalize_state=finalizing` for a later poll.
  Only a *verified* archive with a declared `result_patterns` file still missing
  becomes a terminal `error`.
- A missing checking-agent output fails closed (`error`); it never defaults to
  success.
- Results are written once (`result_status`/`result_error`), then delivered from
  the persisted payload. Do not recompute a result for a delivered row.
- Compare group/local directories with `os.path.realpath`; a textual `==`
  breaks on symlinked or relative workdirs.

**Files and transfers**

- One fresh `scp.SCPClient` per transfer, protected by a lock. Never put a
  client on a provider if two jobs can download or upload concurrently.
- Download into a temporary directory, validate the whole archive
  (`tar -tzf`, `unzip -t`), and only then write into `local_dir`. A truncated
  transfer must not overwrite the previous good result.
- Exclude local inputs (`INCAR`, `POSCAR`, `POTCAR*`, `*.sh`, `*.json`, ...) when
  extracting, so the manager never mutates the scientific inputs.
- Remote paths may contain `~`. Quote them with `remote_shell_path()` (which
  emits `"$HOME"/...`) or resolve them to absolute paths at submission. Passing
  `$HOME` to `SCPClient.get` is also unsafe: its sanitizer quotes `$`, while a
  bare `~` is left expandable.

**Callback outbox**

- The callback is at-least-once. Only a 2xx response marks it delivered.
- A result may be ready before the agent registers its resume target. Delivery
  must wait for a non-empty `thread_id`/`interrupt_id`, and an empty value must
  never overwrite an existing registration.
- `list_callback_pending` must include both never-delivered and retry-pending
  rows so the polling loop and startup recovery can re-deliver them.

### Adding a provider

`BaseProvider` is the seam; the manager owns all state transitions. A new
provider should satisfy this contract:

1. **Discovery** — expose `processing_units`, `job_envs`, and optional remote
   files. If `resources` is not configured, discover it at startup without
   preventing submissions.
2. **Status** — `get_job_states(remote_ids)` must return one entry per requested
   id. Map only well-defined terminal states to `COMPLETED`/`FAILED`/
   `CANCELLED`; map queued/running/transitional states to `SUBMITTED`/`RUNNING`;
   map blank, parse failures, and unrecognized states to `NONE`. Return
   `UNKNOWN` only for a definitive “no such job”. Batch the query where the CLI
   allows it, but tolerate per-id failures.
3. **Submit** — honor `JobSetting.idempotency_key`: retries with the same key
   must not start a second remote job. Persist the remote id as soon as it is
   known; the manager records it immediately after `submit` returns.
4. **Download** — signature `download(remote_id, remote_dir, local_dir) -> bool`
   (returning `False` or raising means “try again”, not “calculation failed”).
   Use a fresh transfer client, a temp dir, archive validation, and input
   exclusions. Never call `set_result_ready` or post callbacks from a provider.
5. **Cancel/cleanup** — `cancel` must be idempotent and tolerate a job that has
   already disappeared. Do not delete remote data before the result is persisted
   and delivered (a later retry may still need it).
6. **Logging** — log the remote command and stderr tail for diagnosis, but never
   credentials. Log retries at `WARNING` and terminal failures at `ERROR`.

After implementing a provider, extend `write_jobmanager_config()` in
`check/test_agent_real_jobmanager.py` (or add a provider-specific test) and run
one small real job through the StepNode, including a non-default `remote_root`
and two concurrent jobs.

### Replacing the database backend

`JobDB` currently uses SQLite/aiosqlite. Any replacement must preserve these
semantics:

1. **Atomic claim** — `claim_finalize(job_uuid)` moves a row into `finalizing`
   exactly once (compare-and-set). Two pollers or a restart must not finalize
   the same row concurrently.
2. **Atomic counter** — `increment_finalize_attempt` must return the new value
   under the same transaction/lock. The manager uses the returned count to
   decide when to stop retrying.
3. **Write-once result** — once `result_status`/`result_error` is set, later
   finalize passes must not overwrite it. Preserve `finalize_attempt` on
   ordinary saves; reset it only in `reset_finalize` (a new submission attempt).
4. **Outbox queries** — `list_finalize_pending` must return non-terminal rows
   left in `finalizing`, and `list_callback_pending` must return every
   `callback_pending` row. Startup recovery and the polling loop depend on both.
5. **Registration safety** — `set_thread_id` must ignore empty `thread_id` or
   `interrupt_id` values. The agent may register after a result is ready.
6. **Job + sub-job upsert** — writing a job must replace/merge its `sub_jobs`
   atomically, with deletion cascading from `jobs`. Preserve the
   `(job_uuid, group_index)` uniqueness used by grouping.
7. **Migrations** — run idempotent migrations before creating indexes, and give
   legacy rows safe defaults: `submission_state=submitted`,
   `finalize_state=none`, `finalize_attempt=0`, empty callback target. Rows from
   the old schema must remain readable without rewriting the whole DB.
8. **Concurrency** — use a process-wide write lock (current design) or a
   backend-level transaction that provides the same guarantees. If several
   manager processes can share one database, every read-modify-write above must
   be a conditional update, not “read, decide, write”.
9. **JSON columns** — `result_patterns` and `sub_jobs` are serialized/expanded by
   the persistence layer; keep that boundary in one module so the manager never
   touches backend-specific rows.

### When changing status or finalize logic

- Keep `NONE` non-terminal in the provider mapping, the polling loop, and the
  aggregation step.
- Keep the three outcome classes distinct: `success`, terminal `error`, and
  retryable transfer failure (`download_retry`). Do not collapse them into a
  boolean.
- Keep the download idempotent: retries may run after a partial write, so clean
  stale archives and overwrite result files, but never inputs.
- If you add a new failure message, make clear whether it describes the
  calculation or the transfer. The StepNode reads this text and decides whether
  to resubmit; “results lost” must only be used when that is actually known.

### Regression tests

Before merging changes in this area, run:

```bash
python -m check.test_jobmanager_recovery          # state machine, outbox, ~ quoting
python -m check.test_agent_real_jobmanager --timeout 1800
```

The first test is deterministic and covers the claim/retry/outbox invariants and
`remote_shell_path`. The second submits four real VASP single points, two per
provider, concurrently. It asserts one job row per workdir, success/delivered
state, registered callback targets, the expected element and atom count in
`OUTCAR`/`CONTCAR` (Si 8, Cu 4), a final `E0=` line, and the absence of the old
fatal log messages. It is the only test that exercises two concurrent Slurm
downloads and a `~/jobs` remote root.

## Configuration

Two files are involved:

- `.env` — the LLM used by the job manager agent.
- `jobmanager_config.json` — everything else; passed with `--setting`.

### `.env`

```dotenv
MODEL_NAME="GLM-5.3-Flash"
BASE_URL="https://llmapi.paratera.com"
API_KEY="..."
```

`MODEL_NAME` also selects the client class: names containing `qwen`, `glm` or
`deepseek` map to `langchain-qwq`, `langchain-openai` and `langchain-deepseek`
respectively. The file is loaded automatically from the repository, no environment
variable needs to be exported.

### `jobmanager_config.json`

```json
{
  "db_path": "jobmanager.db",
  "callback_url": "http://localhost:5600",
  "front_end_url": "http://localhost:5600/jobmanager/messages",
  "poll_interval": 60,
  "retry_job_download": [5, 10, 20],
  "retry_callback": [5, 10, 20],
  "n_retry_get_status": 3,
  "n_retry_job_submit": 3,
  "max_finalize_retries": 5,
  "providers": { }
}
```

| Key | Meaning |
| --- | --- |
| `db_path` | SQLite database. Optional, defaults to `jobmanager.db` next to the config file. |
| `callback_url` | Where `JobResult` is POSTed. |
| `front_end_url` | Where agent progress messages are POSTed. |
| `poll_interval` | Seconds between two status polls per provider. |
| `retry_job_download` | Delays between download attempts. |
| `retry_callback` | Delays between callback attempts. |
| `n_retry_get_status` | Retries of a failing remote status query. |
| `n_retry_job_submit` | Resubmissions allowed for an `EnvironmentError`. |
| `max_finalize_retries` | Polls spent retrying a failed result transfer before the job is reported as an error (default 5). |

`providers` maps a provider id (what the main agent passes as `provider_id`) to a
provider definition. Each provider declares `processing_units` (`cpu` and/or `gpu`)
and `job_envs`, which is what `/providers` exposes so the main agent can pick a
consistent environment.

`local` — jobs run as local `bash run.sh` processes:

```json
"local": {
  "type": "local",
  "processing_units": ["cpu", "gpu"],
  "job_envs": {
    "deepmd-kit": {
      "softwares": ["deepmd-kit with DPA3", "lammps", "ase", "scipy"],
      "initialize_command": ["conda activate /path/to/deepmd-kit"]
    }
  }
}
```

`bohr` — cloud platform; the agent picks a machine from `resources`:

```json
"bohr": {
  "type": "bohr",
  "processing_units": ["cpu", "gpu"],
  "project_id": 29496,
  "platform": "ali",
  "job_type": "container",
  "resources": [
    {"machine_name": "c32_m64_cpu", "machine_type": "cpu", "n_cpu_core": 32, "memeory_size": 64, "price": 2.56},
    {"machine_name": "1 * NVIDIA 4090_24g", "machine_type": "gpu", "video_memeory_size": 24, "price": 5.5}
  ],
  "remote_files": [
    {"description": "Pre-trained MLFF models.", "path": "/bohr/.../v7", "files": ["DPA4-Neo.pt"]}
  ],
  "job_envs": {
    "vasp-6.3.2": {
      "image_path": "registry.dp.tech/.../vasp-ase:6.3.2-vtst",
      "softwares": ["vasp-6.3.2", "ase"],
      "initialize_command": ["source /opt/intel/oneapi/setvars.sh", "export FI_PROVIDER=tcp"]
    }
  }
}
```

`slurm` — the agent picks a queue partition; `run.sh` must be a valid Slurm script:

```json
"slurm-1": {
  "type": "slurm",
  "processing_units": ["cpu"],
  "host": "ssh.example.com",
  "user": "user@ACCOUNT",
  "port": 22,
  "password": "",
  "keyfile": "/home/user/.ssh/id_ed25519",
  "remote_root": "~/jobs",
  "resources": [
    {"partition_name": "mars_l", "partition_type": "cpu", "n_cpu_core": 64, "n_nodes": 254}
  ],
  "remote_files": [],
  "job_envs": {
    "vasp-6.4.0": {
      "softwares": ["vasp-6.4.0 with vtst"],
      "initialize_command": ["module load intel/oneapi2023.2_impi", "export PATH=...:$PATH"]
    }
  }
}
```

Notes:

- Slurm providers open the SSH connection when the provider is constructed, i.e. at
  server startup. Every configured Slurm host must be reachable or the server will
  not start.
- `password` is used only when `keyfile` is empty.
- `remote_root` is the remote directory created per job; `local_dir` is uploaded
  there, and the result archive is downloaded back into `local_dir`.
- On Slurm, the full remote job directory is downloaded, so `result_patterns` is not
  needed there.
- Values the main agent is not supposed to know (machine SKU, partition, resource
  names) are chosen by the job manager agent from `resources`.

## Running the server

The job manager is part of the `catagent` package. Start it from anywhere; pass
an explicit setting file:

```bash
catagent jobmanager \
    --setting catagent/jobmanager/jobmanager_config.json \
    --host 127.0.0.1 --port 8000 \
    --log jobmanager.log --level info
```

`--level` is `info` or `debug`. Logs go to `--log` from a background thread, so file
logging never blocks the event loop.

## Demo

`demo/` is a self-contained playground: it replaces the calculation agent with an
interactive command line client and the main agent front end with a server that
prints everything it receives. Jobs are executed with the `local` provider.

The demo needs no LLM and no remote platform by default, because the job manager
agent is replaced by a deterministic stub (`demo/stub_agent.py`).

### Offline demo

```bash
cd /path/to/parent-of-jobmanager
python -m catagent.jobmanager.demo.run_demo
```

This starts the front end and the job manager as child processes, then hands the
terminal over to the interactive client. Answer the prompts (Enter accepts the
default) and watch the front end print the streamed agent messages and the final
`JobResult`:

```
[agent 15:36:13]
    [stub-agent:preprocessing] working on /.../runs/f7738ba603ba

════════════════════════════════════════════════════════════════════════════════
 JOB RESULT  SUCCESS   (15:36:17)
   job_uuid  : f7738ba603ba
   local_dir : /.../runs/f7738ba603ba
════════════════════════════════════════════════════════════════════════════════
```

The client, which plays the role of the main agent, asks for:

- `provider_id` and `job_env_name`, chosen from `GET /providers`.
- `job_uuid`, `description`, `result_patterns`.
- `local_dir`: leave it empty to auto-create a demo job (a small `run.sh` plus a
  `job_description.json`), or type the path of an existing job directory to submit
  real inputs.
- For auto-created jobs: `batch_job`, the number of groups, and two optional failure
  injections — `DEMO_INPUT_FAIL` (reported as `InputError`, no retry) and
  `DEMO_ENV_FAIL` (reported as `EnvironmentError`, resubmitted up to
  `n_retry_job_submit`).

### Demo with the real settings

To use the real LLM agent (from `.env`) and the real providers, add
`--real-agent`:

```bash
python -m catagent.jobmanager.demo.run_demo \
    --config catagent/jobmanager/jobmanager_config.json --real-agent
```

The launcher reads `callback_url` and `front_end_url` from the config (in
`jobmanager_config.json` these are `:5600` and `:5601`) and binds the demo front end
to those ports, so it receives exactly the POSTs the manager sends. In this mode a
real calculation can be submitted to `bohr`/`slurm` by pointing `local_dir` at a
prepared job directory.

Because provider construction opens SSH connections, all Slurm hosts in the config
must be reachable; trim the config to the single platform you want to test otherwise.

### Separate terminals

Equivalent to the launcher, useful when you want to see each stream on its own:

```bash
# main agent front end (printer). One port when the config uses /callback and
# /stream paths (demo/config.json -> :9001), two ports when it uses bare URLs on
# separate ports (jobmanager_config.json -> :5600 and :5601).
python -m catagent.jobmanager.demo.frontend --callback-port 9001
# python -m catagent.jobmanager.demo.frontend --callback-port 5600 --stream-port 5601

# job manager (stub agent and local provider; add --real-agent for the real agent)
python -m catagent.jobmanager.demo.server --config catagent/jobmanager/demo/config.json --port 8000 --fresh

# interactive main agent
python -m catagent.jobmanager.demo.client --url http://127.0.0.1:8000
```

### Demo files

| File | Role |
| --- | --- |
| `demo/run_demo.py` | Launcher: starts the front end and the job manager, then the client. |
| `demo/frontend.py` | Main agent front end stand-in: prints progress messages and `JobResult`s. Binds one or two ports and bare `/` paths, matching both demo and real configs. |
| `demo/server.py` | Job manager with the stub agent (default) or the real agent (`--real-agent`). |
| `demo/client.py` | Interactive `JobRequest` client; also creates demo job directories. |
| `demo/stub_agent.py` | Deterministic agent: reads `job_description.json`, and reports `Success` / `InputError` / `EnvironmentError` from the marker files. |
| `demo/config.json` | Offline demo settings: single-port front end and the `local` provider. |
| `demo/runs/`, `demo/demo.db`, `demo/demo-server.log` | Runtime artifacts, safe to delete. |

## Layout

| Path | Role |
| --- | --- |
| `server.py` | FastAPI app and CLI entry point. |
| `manager.py` | `JobManager`: submission, grouping, polling, download, callback, retries. |
| `db.py` | SQLite (aiosqlite) persistence for jobs and sub jobs. |
| `models.py` | `JobRequest`, `JobResult`, `JobRecord`, `SubJobRecord`, `JobSetting`, `JobStatus`. |
| `base.py` | `BaseProvider` interface. |
| `providers/` | `local.py`, `bohr.py`, `slurm.py` implementations. |
| `agent/` | Job manager agent (LLM), its tools and `skill.md`. |
| `utils.py` | Async shell helper. |
| `demo/` | Offline demo and interactive client. |
