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
  `--backward_files`, on Slurm the whole remote directory is archived, and it is
  unused for local jobs.

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

`status` is `success` or `error`. Delivery is retried `retry_callback` times; on
success (or a `404`, meaning the main agent already moved on) the job is marked as
delivered. Undelivered callbacks are left pending and re-posted at the next startup,
so delivery is at-least-once.

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
{"role": "agent", "message": "...", "tool_calls": "..."}
{"role": "tool",  "message": "..."}
```

This is best-effort: a failure to stream never affects the job.

### Job statuses

`JobRecord.status` aggregates the statuses of its sub jobs:

`submitted → running → completed`, or a terminal `failed` / `cancelled` / `unknown`.
`none` means the status query itself failed. Once **all** sub jobs are terminal the
manager downloads the results, runs the checking agent and posts the callback.

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
  "front_end_url": "http://localhost:5601",
  "poll_interval": 60,
  "retry_job_download": [5, 10, 20],
  "retry_callback": [5, 10, 20],
  "n_retry_get_status": 3,
  "n_retry_job_submit": 3,
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

Run from the directory that contains the `jobmanager` package, so imports and the
config path resolve:

```bash
python -m jobmanager.server \
    --setting jobmanager/jobmanager_config.json \
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
python -m jobmanager.demo.run_demo
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
python -m jobmanager.demo.run_demo \
    --config jobmanager/jobmanager_config.json --real-agent
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
python -m jobmanager.demo.frontend --callback-port 9001
# python -m jobmanager.demo.frontend --callback-port 5600 --stream-port 5601

# job manager (stub agent and local provider; add --real-agent for the real agent)
python -m jobmanager.demo.server --config jobmanager/demo/config.json --port 8000 --fresh

# interactive main agent
python -m jobmanager.demo.client --url http://127.0.0.1:8000
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
