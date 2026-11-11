# Agent test suite (`check/`)

This directory contains tests for the cat-agent workflow.  Most of them use a
**fake computation pipeline** (`mock-compute`) whose "software" is a set of
short, sleep-based Python scripts.  No real scientific software is executed and
no GPU is needed.  The exception is `test_agent_real_jobmanager.py`, which runs
four small VASP single-point jobs on the real `bohr` and `slurm-1` providers.

The tests are grouped by what they exercise:

| File | Real LLM? | Real jobmanager? | Purpose |
|---|---|---|---|
| `test_planner_tool.py` | no | no | Planner tool stores a `RawPlan` and reports its id. |
| `test_register_raw_plan.py` | no | no | `register_raw_plan` in `append` and `replace` mode. |
| `test_agent_stepnode.py` | no | no | `summary_node` reads `report.md` and extracts a `RawStepResult`. |
| `test_agent_state_sync.py` | no | no | Outer↔orchestrator step sync: no lost updates, no stale preparation requests. |
| `test_jobmanager_recovery.py` | no | embedded | Crash recovery: resumable submission/finalization, callback outbox, local exit codes, per-thread locks. |
| `test_cli_jobs.py` | no | mocked HTTP | The debug CLI polls `/status/{job_uuid}` and resumes a JobSubmission itself. |
| `test_app_skill_init.py` | no | no | Skill-index loading/remapping, graph `loaded_skills` seeding, 100-character skill-tool payloads, CLI parser options, and agent logging. |
| `test_debug_ui.py` | no | no | Bounded event log, `GET /` HTML route, `/events/{thread_id}`, jobmanager messages, and the unchanged `POST /` callback. |
| `test_agent_orchestrator.py` | yes | mocked HTTP | The orchestrator reads a skill, plans, and registers an append plan. |
| `test_agent_e2e.py` | yes | yes | Full workflow: orchestrator → router → StepNode → jobmanager callback → final report. |
| `test_agent_real_jobmanager.py` | yes | yes (real `bohr` + `slurm-1`) | Four concurrent StepNodes submit real VASP single-point jobs; checks downloads, callbacks, and element/atom counts. |

Run everything from the repository root.  The deterministic tests are quick
(seconds); the two real-LLM tests make many model calls and typically take
5–20 minutes depending on provider latency.  `--timeout` in the e2e test is
the maximum wait for one job callback, not for the whole run.

---

## 1. Prerequisites

### Python environment

Use the same environment as the rest of the project (the tests import
`catagent`, `jobmanager`, `langgraph`, `langchain`, `httpx`, `fastapi` and
`uvicorn`).

### Model credentials

The real-LLM tests call `check.common.make_model()`, which loads the first
existing file among:

1. `/home/luandong/bin/.env`
2. `<repo>/.env`
3. `<repo>/catagent/jobmanager/.env`

It expects:

```dotenv
MODEL_NAME=DeepSeek-V4.1-Flash
BASE_URL=...
API_KEY=...
```

`MODEL_NAME` is used to choose the chat-model wrapper (`qwen`, `glm`, or
`deepseek`).  The deterministic tests do not need credentials.

### Free ports

`test_agent_e2e.py` starts a local jobmanager on **18000** and a callback
server on **5600**.  Check that they are free:

```bash
ss -ltnp | grep -E ':(5600|18000)\b'
```

If something is using them, stop it or edit the constants
`JOB_MANAGER_PORT` / `CALLBACK_PORT` in `check/test_agent_e2e.py`.

---

## 2. The fake computation

### `mock_skills/mock_compute/`

Layout is the same as a real skill (compare `catagent/skills/dpa4/`):

```text
check/mock_skills/mock_compute/
├── skill.md
├── references/
│   ├── job_submission.md
│   └── workflow.md
└── scripts/
    ├── mock_analyze.py -> ../../../mock_software/mock_analyze.py
    ├── mock_prepare.py -> ../../../mock_software/mock_prepare.py
    ├── mock_relax.py   -> ../../../mock_software/mock_relax.py
    └── mock_report.py  -> ../../../mock_software/mock_report.py
```

The skill describes the intended plan:

```text
prepare ──┬──> relax_A ──┐
          └──> relax_B ──┴──> analyze ──> report
```

`relax_A` and `relax_B` are independent and are executed in parallel.  The
skill also documents `run.sh` / `job_description.json` for the two job steps.

### `mock_software/`

| Command | What it does |
|---|---|
| `mock-prepare --outdir DIR` | Writes `input_A.json` and `input_B.json`. |
| `mock-relax --input FILE --output FILE` | Sleeps, writes `relaxed_<sample>.json` and a `.log`. |
| `mock-analyze --input-dir DIR --output FILE` | Compares `relaxed_*.json`, writes `analysis.json` and `analysis.md`. |
| `mock-report --analysis FILE --output FILE` | Writes `final_report.md`. |

The Python scripts live in `check/mock_software/`; the executables on `PATH`
are the wrappers in `check/mock_software/bin/`.  The e2e harness prepends
`check/mock_software/bin` to `PATH` for local (non-job) steps.

The fake energies are deterministic: sample A ends lower than sample B
(11.7 vs 13.05 in the current implementation).

### Job environment

`catagent/jobmanager/jobmanager_config.json` contains a provider named `local-mock` with
one job environment, `mock-compute`.  Its `initialize_command` puts
`check/mock_software/bin` on `PATH`; `check/test_agent_e2e.py` also starts the
jobmanager with that directory on `PATH`.  The local provider applies each
`initialize_command` line inside its run wrapper before `run.sh`, so jobs
submitted from a bare jobmanager process see the same tools.

`check/jobmanager_test_config.json` is the fast config actually used by the
e2e test:

- database: `check/.tmp/jobmanager_test.db`
- callback: `http://127.0.0.1:5600`
- `poll_interval: 1` second
- only the `local-mock` provider

---

## 3. Running the tests

### Deterministic tests (no network, no jobmanager)

```bash
python -m check.test_planner_tool
python -m check.test_register_raw_plan
python -m check.test_agent_stepnode
python -m check.test_agent_state_sync
python -m check.test_jobmanager_recovery
python -m check.test_cli_jobs
python -m check.test_app_skill_init
python -m check.test_debug_ui
```

Expected output:

```text
planner tool test: OK
...
register_raw_plan append/replace tests: OK
...
StepNode summary_node test: OK
...
AGENT STATE SYNC TESTS: OK
...
JOBMANAGER RECOVERY TESTS: OK
```

### State sync between the workflow and the orchestrator

```bash
python -m check.test_agent_state_sync
```

The outer workflow and the inner orchestrator each keep a copy of `steps`, so a
naive full-state copy-back can lose updates.  This test covers the two rules
that prevent the router from re-issuing the same preparation request:

- `orchestrator_node` returns only the step **fields the orchestrator changed**
  (`_steps_delta`), so a concurrent router transition such as
  `preparation_stage = done` is never reverted.
- `orchestrator_request_queue` **rehydrates** a queued request's `steps` from
  the current outer state, so an old request cannot dispatch stale
  `workdir`/`preparation_stage` values.

### Debug CLI job polling

```bash
python -m check.test_cli_jobs
```

The debug CLI has no HTTP endpoint for jobmanager callbacks, so it must poll
jobs itself: `_stream_chat` is called with `register_jobs=False`, the CLI polls
`GET /status/{job_uuid}` until `result_status` is final, and then resumes the
graph with `{interrupt_id: {"payload": {...}}}`. The test uses a fake graph
stream and an `httpx.MockTransport`; no jobmanager process is started.

### Skill index and payload filtering

```bash
python -m check.test_app_skill_init
```

This checks that `catagent/skills/skill_index.json` loads, that stale absolute skill
paths are remapped to the local `catagent/skills/<name>/` tree, that the compiled graph's
`init_skills` node seeds `loaded_skills`, and that skill-tool `ToolMessage`s are
truncated to 100 characters in streamed payloads without changing the original
message.

### Debug UI event log and routes

```bash
python -m check.test_debug_ui
```

No model or browser is needed.  The test builds the FastAPI app with a fake
runtime and checks the bounded per-thread event log, `GET /` (the HTML console),
`GET /events/{thread_id}`, `POST /jobmanager/messages`, and that the jobmanager
callback on `POST /` still works.  It also checks that a StepNode interrupt
carries its `<thread_id>:<step_id>` source into the payload.

### Jobmanager crash-recovery tests

```bash
python -m check.test_jobmanager_recovery
```

This test does **not** call an LLM, start a server, or run real software.  A
fake provider and a fake jobmanager agent exercise the failure paths that the
e2e test cannot trigger deterministically.  It covers:

- **Resumable submission** — a persisted group plan is resumed at boot; groups
  that already have a remote id are skipped, so a restart never starts a second
  remote job.  The idempotency key includes the attempt
  (`job_uuid:gN:aM`), so crash retries are idempotent while an explicit retry
  increments the attempt and starts a fresh run.
- **Crash during download** — a row left in `finalizing` is re-finalized by the
  polling loop (and again at startup) and never swept as a success.
- **Missing artifacts** — a `completed` job is an error if any declared
  `result_patterns` file is absent; having only some of the declared outputs is
  not a success.
- **Callback outbox** — a 404 response or a crash before delivery keeps the
  row `ready`/pending; it is retried and marked `delivered` only on 2xx.
- **Missing analysis output** — if the checking agent returns no
  `job_result_status`, the result fails closed instead of defaulting to
  success.
- **Local exit codes** — the wrapper writes `.catagent/status.json`
  atomically; a restarted provider reads it, and a dead pid without a status
  file is `unknown`, never `completed`.  The wrapper also publishes its pid, so
  a crash between spawn and pid persistence does not launch a second process.
- **`initialize_command`** — the local run wrapper applies it before `run.sh`.
- **Per-thread orchestrator locks** — different `thread_id`s run concurrently
  while one thread stays serialized.

The durable state is stored in the `jobs` table as four columns:

| Column | Meaning |
|---|---|
| `submission_state` | `planning` → `planned` → `submitting` → `submitted` (or `failed`). |
| `finalize_state` | `none` → `pending` → `finalizing` → `ready` → `delivered`. |
| `result_status` | `success` / `error`, written once when the result is decided. |
| `result_error` | Human-readable error carried by the callback. |

`callback_pending` is the outbox flag: it is set together with
`result_status`, so a boot sweep re-posts the exact persisted result instead of
inferring one from the current job status.  Recovery order at startup is
submissions → finalizations → callbacks.

The local provider keeps a pid → job-directory registry under
`$CATAGENT_LOCAL_STATE_DIR` (default `$TMPDIR/catagent-local-jobs`) and an
idempotent `.catagent/submit.json` marker per job, which is what makes status
recoverable across a jobmanager restart.

### Orchestrator with the real LLM

```bash
python -m check.test_agent_orchestrator
```

This creates a real orchestrator graph with `InMemorySaver`.  The jobmanager
HTTP call is replaced by an `httpx.MockTransport` that only answers
`GET /providers`, so no jobmanager process is needed.  The agent must read the
mock skill, call the planner, and register at least four steps with the
expected dependencies.  Expected output:

```text
Registered steps:
  plan-0:step-0:...
  ...
orchestrator append plan test: OK
```

### Full end-to-end test with jobmanager

```bash
python -m check.test_agent_e2e --timeout 180
```

Optional arguments:

```bash
python -m check.test_agent_e2e --workdir /tmp/my_e2e --timeout 300
```

What happens:

1. A jobmanager subprocess is started on `127.0.0.1:18000` with
   `check/jobmanager_test_config.json`.
2. A FastAPI callback server listens on `127.0.0.1:5600`.
3. The test builds the real `create_catagent()` graph with the real LLM.
4. It drives the graph through `catagent.app._stream_chat` and answers every
   interrupt:
   - `JobSubmission` interrupts are matched with the jobmanager callback by
     `interrupt_id`;
   - all other interrupts use the mock-skill defaults.
5. The workflow must finish with `finalization=True` and all steps `done`.
6. The test asserts that `analysis.json` and `final_report.md` exist.

Expected tail:

```text
Final steps:
  plan-0:step-0:...: status=done preparation=done ...
  ...
END-TO-END TEST: OK
workdir = .../check/.tmp/e2e_<id>
jobmanager log = .../check/.tmp/e2e_<id>/jobmanager.log
```

By default each run creates a new directory under `check/.tmp/`.  Keep it for
inspection or delete `check/.tmp/` to start clean.

### Real remote-provider test (`bohr` + `slurm-1`)

```bash
python -m check.test_agent_real_jobmanager
```

This is the regression test for the real-provider download/callback failure.
It copies the four input sets from
`/mnt/t/workdir/agent_test/case0/test_cases` (Si and Cu, for `bohr` and
`slurm-1`) into a temporary workdir and runs **four StepNode conversations
concurrently**, one per case.  Running them together is deliberate: the old
bug shared one `scp.SCPClient` between two concurrent slurm downloads, which
truncated one archive and put Cu results into the Si workdir.

The test writes a temporary jobmanager config derived from
`catagent/jobmanager/jobmanager_config.json`: only the `bohr` and `slurm-1` providers,
a temporary DB, the test callback URL, and a shorter `poll_interval`.  The
repository config is never modified.  `run.sh` and `job_description.json` are
pre-created so the test is deterministic; pass `--agent-prepares-files` to let
the StepNode author them instead.

It checks that:

- each StepNode calls `submit_job` exactly once for its workdir;
- all four rows reach `result_status=success`, `finalize_state=delivered`;
- the callback carries a registered `thread_id`/`interrupt_id` and resumes the
  StepNode;
- `OUTCAR`/`CONTCAR` contain only the requested element and atom count
  (Si = 8, Cu = 4) — this catches the cross-workdir data corruption;
- `vasp.stdout` contains a final `E0=` line;
- the jobmanager log has no `Only part of job results`, `Underlying channel was
  closed`, `Declared result files are missing`, or exhausted-retry messages.

Useful options:

```bash
python -m check.test_agent_real_jobmanager \
    --workdir /tmp/opencode/real_jm \
    --cases bohr_Cu,slurm1_Si \
    --timeout 1800 \
    --case-timeout 3000 \
    --poll-interval 15
```

`--case-timeout` caps the whole per-case StepNode conversation (default
`--timeout + 1200`).  `--agent-prepares-files` makes the StepNode author
`run.sh`/`job_description.json` instead of using the deterministic versions the
test pre-creates.

The test uses the real `bohr` CLI and the configured slurm SSH key, so it
needs network access and valid provider credentials.  It prints the job UUIDs,
remote ids, result files and the jobmanager log/DB paths for inspection.

---

## 4. What `register_raw_plan` should do

`check/test_register_raw_plan.py` drives a real LangGraph agent with the real
planner tools but a `ScriptedModel` that replays tool calls.  It covers:

### Append

```text
rawplan-0: prepare -> (relax_a, relax_b) -> analyze
```

- All four steps are added under `plan-0:step-N:<name>`.
- `topology` becomes `depend_on` / `required_by` sets.
- `depend_on_external` can add dependencies on steps outside the plan.
- The registered raw plan is removed from `raw_plan_dict`.

### Replace

Two replacement shapes are tested:

1. One-to-one: `rawplan-1` replaces `relax_a` and `relax_b` with
   `relax_a_v2` and `relax_b_v2`.  The new steps inherit the replaced steps'
   dependencies, and steps that depended on the old ids are rewired.
2. Many-to-one: `rawplan-2` replaces both revised relax steps with one
   `relax_both` step.  `analyze` must ultimately depend only on the new step
   plus its original external dependency.

The test also checks that `raw_plan_dict` becomes empty after registration.

---

## 5. Troubleshooting

- **`Address already in use`** — a previous jobmanager or callback server is
  still running.  Check ports 5600/18000 and stop the process.
- **No model credentials** — create `/home/luandong/bin/.env` or a repo
  `.env` with `MODEL_NAME`, `BASE_URL`, `API_KEY`.
- **E2E fails with `workflow did not set finalization=True`** — inspect
  `check/.tmp/e2e_<id>/state.md` and the final step table printed by the test.
  `jobmanager.log` in the same directory contains the jobmanager side.
- **A job stays `submitted`** — check `jobmanager.log` for the `run.sh`
  invocation and whether `check/mock_software/bin` is on `PATH`.
- **Stale state from an old run** — every e2e run uses a random thread id and
  a fresh `InMemorySaver`, so old runs should not leak.  The jobmanager test
  database is deleted at startup.

All tests except `test_agent_real_jobmanager.py` stay on `local-mock` or a
mocked HTTP transport; only that one submits to the real `bohr` and `slurm-1`
providers.
