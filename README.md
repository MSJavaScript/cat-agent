# cat-agent

An agent for computational catalysis.

`cat-agent` turns a scientific-computation request into a plan, prepares a
working directory for every step, runs each step with an LLM agent, submits heavy
calculations to a separate [jobmanager](#jobmanager), and loops until the user's
task is finished. There is a CLI and a single-page web console for debug.

This README is the onboarding document for the project. It explains what the
pieces are, how a request flows through them, how to run every mode, and which
invariants you must preserve when changing the code.

---

## Installation

```bash
./install.sh                  # current Python environment
./install.sh --user           # ~/.local, no root
./install.sh --venv .venv     # create a virtualenv and install there
./install.sh --editable       # editable install for development
./install.sh --no-deps        # dependencies are already installed
```

The script installs the dependencies declared in `pyproject.toml` and a
`catagent` console script. Only runtime files are installed: the development
directories (`ai-changes/`, `check/`, `learn/`, `recycle/`), `jobmanager/demo/`,
all `.env` files and `jobmanager/jobmanager_config.json` are excluded.
Jobmanager configuration and model credentials are therefore always supplied by
the user.

The skills directory is part of the package, so `--skill-index-json` is
optional. Override it with the flag or with `CATAGENT_SKILL_INDEX` when using a
different skill set.

---

## Quick start

```bash
# 1. Install cat-agent and the `catagent` command (see Installation)
./install.sh

# 2. Model credentials (create .env; see Configuration)
# MODEL_NAME=...
# BASE_URL=...
# API_KEY=...

# 3. Optional: start the job manager for real job submission
python -m jobmanager.server \
    --setting /path/to/jobmanager_config.json \
    --host 127.0.0.1 --port 8000 --log jobmanager.log

# 4. Run the agent with the debug CLI (in-process graph)
catagent cli --dotenv .env --db catagent_checkpoints.db
```

The checkpoint database is created automatically. `catagent --help`,
`catagent cli --help` and `catagent server --help` list all options. The legacy
`python -m catagent.app` entry point continues to work with the same flags.

---

## Repository layout

```text
catagent/
├── agent.py                  Outer workflow: builds and wires the main graph.
├── run_agent.py              Standalone generic LangGraph streamer helper.
├── _args.py                  Argument parsers (dependency-light, shared).
├── _paths.py                 Locates the bundled skills index.
├── _entrypoint.py            Entry point for the installed `catagent` command.
├── app/                      Application layer (HTTP service, CLI, runtime).
│   ├── __init__.py           Package facade and common CLI/server entry point.
│   ├── __main__.py           Entry point for `python -m catagent.app`.
│   ├── models.py             ChatRequest / StreamPayload types.
│   ├── payloads.py           LangGraph chunk -> UI payload conversion.
│   ├── runtime.py            Model + checkpoint DB + compiled graph factory.
│   ├── skills.py             `skill_index_json` -> initial `loaded_skills`.
│   ├── jobs.py               Polling of durable job results for the debug CLI.
│   ├── streaming.py          Graph streaming and job callback registration.
│   ├── plan.py               Read-only plan projection (`/plan`).
│   ├── events.py             Bounded in-memory payload log for the debug UI.
│   ├── debug_ui.html         Single-page debug console (served at `GET /`).
│   ├── service.py            FastAPI app and HTTP/SSE routes.
│   ├── server.py             uvicorn launcher (`--serve`).
│   ├── cli.py                Interactive debug chat loop.
│   └── log.py                `catagent` logger setup (file or stderr).
├── workflow/                 Planning and orchestration (outer graph pieces).
│   ├── models.py             RawPlan/RawStep/Step/WorkflowState + reducers.
│   ├── orchestrator.py       The phase-aware orchestrator agent and its tools.
│   ├── router.py             Dependency dispatch, step preparation, requests.
│   ├── planner_tool.py       planner / get_raw_plan_content / register_raw_plan.
│   ├── step_edit_tool.py     Read/edit `Step` objects in orchestrator state.
│   ├── skill_tool.py         Load skills and summarize them into state.
│   └── job_tool.py           `get_providers_info` for the orchestrator.
├── stepnode/                 Per-step execution agent (inner graph).
│   ├── stepnode.py           model/tool/interrupt/summary nodes.
│   ├── init_tools.py         Static tool registry + job/interrupt tools.
│   ├── models.py             StepNodeState, StepResult and interrupt models.
│   └── skill_tool.py         Skill lookup for a running step.
└── system_skills/            System prompts injected into the agents.

jobmanager/                   Independent HTTP job submission service.
tools/                        Computation/shell tools used by agents.
skills/                       Scientific skills shipped with the package.
check/                        Fast deterministic tests + full e2e test harness.
learn/                        Self-contained runnable demos of internal mechanisms.
install.sh                    Installs dependencies and the `catagent` command.
pyproject.toml, MANIFEST.in   Packaging: runtime files only, dev dirs excluded.
```

`catagent/app.py` used to be one large module. It is now the `catagent/app`
package; `catagent/app/__init__.py` re-exports the historical names so existing
imports keep working (`from catagent.app import _stream_chat, build_runtime,
create_llm_model, get_plan, ...`).

---

## Architecture

There are two nested LangGraph graphs and one independent HTTP service:

```text
                   ┌──────────────────────── outer workflow (catagent/agent.py) ───────────────────────┐
 user message ────▶│ START → init_skills → orchestrator_node → node_set_flag → router                   │
 (HTTP /           │              ▲                    │                                            │
  debug CLI)       │              │                    ├──(Send)──▶ step_node  (one per ready step)   │
                   │              │                    │                    │                       │
                   │        interrupt_node ◀───────────┴── orchestrator_request_queue ◀──────────────┤
                   │              │                                                              │    │
                   │              └────────────── resume ────────────────────────────────────────┘    │
                   └───────────────────────────────────────────────────────────────────────────────────┘
                              │ inner calls                         │ inner calls
                              ▼                                     ▼
                   ┌─────────────────────┐              ┌─────────────────────────┐
                   │ Orchestrator agent  │              │ StepNode agent          │
                   │ plan / prepare /    │              │ generate inputs,        │
                   │ correct / finalize  │              │ submit, analyze, report │
                   └─────────────────────┘              └─────────────────────────┘
                              │                                     │
                              └──────────────┬──────────────────────┘
                                             ▼
                                  ┌─────────────────────┐
                                  │ jobmanager (HTTP)   │──▶ HPC / local providers
                                  └─────────────────────┘
```

### The four components

| Component | Responsibility |
| --- | --- |
| **Orchestrator** | A phase-aware agent that talks to the user, makes plans, prepares step directories, handles failures, logs `state.md`, and decides when the run is finalized. |
| **Router** | Pure workflow logic. Consumes `Step` objects, checks dependencies, marks steps running, writes each step's `readme.md`, dispatches `step_node` tasks in parallel, and queues work for the orchestrator. |
| **StepNode** | A per-step agent. It reads `readme.md`, generates inputs, submits jobs through jobmanager, analyzes results, and writes `report.md`. |
| **JobManager** | An independent FastAPI service that owns providers, resource selection, submission, polling, result download, and the callback that resumes the graph. It does not share the graph's process or database. |

The orchestrator only plans and prepares; it never generates calculation inputs
or submits jobs. The StepNode owns execution. Keeping this boundary is what lets
the workflow run many steps in parallel while keeping the global state coherent.

### Request lifecycle

1. A user message reaches the outer graph. On a new thread, `init_skills`
   fills `loaded_skills` from the configured skill index. `orchestrator_node`
   then runs the orchestrator in the `user` phase. It gathers context, reads
   skills, calls the `planner` tool, and registers a `RawPlan`
   (`register_raw_plan`) as `Step` objects.
2. `node_set_flag` sends control to `router`.
3. `router` checks each step:
   - dependencies resolved + `need_further_planning` → queue a replanning
     request;
   - dependencies resolved + prepared → queue a preparation request;
   - `preparation_stage == "done"`, `status == "pending"`, workdir exists →
     mark `running`, write `readme.md`, and `Send` a `step_node` task.
4. Ready steps may run in parallel (one `Send` per step). Each StepNode returns
   a `StepResult` to `router`.
5. `router` records success/failure, asks the orchestrator to log or correct it,
   and queues any newly-ready work. The loop repeats.
6. When all steps are terminal, `router` sends a replanning request. The
   orchestrator either plans more work or calls `set_finalization_flag(True)`.
   `node_set_flag` then routes to `END`.

### State model

The outer graph state is `WorkflowState` in `catagent/workflow/models.py`:

| Field | Meaning |
| --- | --- |
| `messages` | Conversation history; only the latest message is forwarded on a fresh orchestrator turn. |
| `steps` | `Dict[step_id, Step]`; the only field shared with the orchestrator. Uses a patch-aware reducer. |
| `step_results` | Latest results returned by StepNodes. |
| `orchestrator_requests` | Queue of requests for the orchestrator (preparation, correction, logging, replanning). |
| `loaded_skills` | Skill metadata seeded from `--skill-index-json`; the orchestrator can add more and StepNodes receive a copy. |
| `orchestrator_is_working` | Guard flag for the single in-flight orchestrator call. |
| `finalization` | Set by the orchestrator when the run is complete. |

A `Step` has an id of the form `plan-{n}:step-{m}:{name}` and a small lifecycle
(`status` × `preparation_stage`). The important transitions are:

```text
pending + notyet  ──deps ok──▶ pending + ongoing ──prepared──▶ pending + done
                                                                    │
                                                                    ▼
                                                            running
                                                          /          \
                                                     done            failed
```

The orchestrator keeps its own copy of `steps` in `OrchestratorAgentState`.
Because the outer graph and the orchestrator can both mutate step state, the
sync rules are deliberate:

- `orchestrator_node` returns only the **field-level delta** the orchestrator
  changed (`_steps_delta`), never the whole inner copy.
- `orchestrator_request_queue` **rehydrates** each queued request's `steps`
  from the current outer snapshot, so a stale request cannot reset a workdir or
  preparation stage that was completed in the meantime.

Both rules exist to avoid lost updates; see `check/test_agent_state_sync.py`.

### Checkpointing and thread ids

Everything is checkpointed in one `AsyncSqliteSaver`, but the nested graphs use
different thread ids:

- outer workflow: `<thread_id>`
- orchestrator: `<thread_id>:orchestrator`
- each step node: `<thread_id>:<step_id>`

Never reuse the outer id for an inner graph: both graphs would overwrite each
other's checkpoints. The `thread_id` is also the unit of concurrency — the
orchestrator lock/claim registry in `agent.py` is keyed by it, so different users
run concurrently while one user stays serialized.

### Interrupts and human-in-the-loop

LangGraph interrupts are used for three things:

| Kind | Raised by | User-facing? | Resume shape |
| --- | --- | --- | --- |
| `JobSubmission` | `submit_job` → StepNode inner interrupt | No; handled automatically | job result payload |
| `AskForArgs` | tools that need user-edited arguments | Yes | `{"payload": {"command": "...", "message": "..."}}` |
| `AgentInterrupt` | `pause_to_ask` | Yes | `{"payload": {"result": ...}}` |

The inner StepNode interrupts once; `step_node` catches the interrupt and re-emits
it through the outer `interrupt_node`, preserving the original interrupt id. On
resume, `interrupt_node` routes the answer back to the exact inner graph
(`Command(resume={original_id: response})`). The same pattern is used for the
orchestrator. When several interrupts are pending, a resume map keyed by
interrupt id resumes them together.

`catagent/app/streaming.py::_stream_chat` turns stream chunks into payloads and,
for `JobSubmission`, optionally posts `{job_uuid, thread_id, interrupt_id}` to
jobmanager's `/set-thread-id`. That registration is what lets jobmanager POST a
`JobResult` back and resume the graph.

### Job submission flow

```text
StepNode ──submit_job──▶ jobmanager POST /submit
StepNode ──interrupt()──▶ pauses; outer graph surfaces JobSubmission
            │
            ▼
   register_jobs=True  ──▶ jobmanager stores (thread_id, interrupt_id)
   register_jobs=False ──▶ debug CLI polls /status/{job_uuid} itself
            │
            ▼
 jobmanager finishes ──▶ POST JobResult ──▶ catagent /chat (callback)
            │
            ▼
 outer interrupt_node resumes the exact StepNode interrupt
```

There must be exactly one owner of a pending job resume:

- HTTP server mode: `register_jobs=True`; jobmanager posts the callback.
- Debug CLI: `register_jobs=False`; the client polls `/status/{job_uuid}` and
  resumes the graph itself. The polling code lives in `catagent/app/jobs.py`.

Registering both would resume a job twice.

### Skills and tools

- The application requires `--skill-index-json` (for example
  `skills/skill_index.json`). `catagent/app/skills.py` loads it and seeds every
  new thread's `loaded_skills`; the graph initializer only fills the field when
  a thread does not already have skills, so a later
  `load_skill_from_filesystem` call is never overwritten.
- `skills/skill_index.json` lists the available skills: a JSON array of
  `{name, path, summary, references, scripts}` objects. If the recorded
  checkout no longer exists, paths are remapped next to the index file
  (`skills/<name>/skill.md` and its `references/` / `scripts/` files).
- The orchestrator can also load a skill at runtime with
  `load_skill_from_filesystem`, summarize it, and store `SkillMetadata` in
  `loaded_skills`.
- A StepNode receives that metadata and can call `get_skill_content` without
  re-reading the skill from disk.
- Tool messages from `get_skill_summaries`, `get_skill_content` and
  `load_skill_from_filesystem` are shortened to their first 100 characters in
  the **streamed payload** only. The full `ToolMessage` still reaches the model
  and stays in graph state; this keeps chat transcripts readable.
- `tools/` contains shell, file-editing, structure, NEB, and INCAR helpers.
  `catagent/stepnode/init_tools.py` builds the static tool registry and the
  job-related tools. Tool collections can be enabled at runtime by the StepNode.

---

## Runtime modes

### 1. Debug CLI (in-process)

```bash
catagent cli \
    --dotenv .env \
    --db catagent_checkpoints.db \
    --job-manager-url http://127.0.0.1:8000 \
    --thread-id my-thread
```

This runs the compiled graph in the current process, prints colored payloads,
and asks for interrupt answers on stdin (JSON or free text). Submitted jobs are
handled automatically: the CLI polls jobmanager's `/status/{job_uuid}` and
resumes the graph when the durable result is ready. The equivalent legacy
invocation is `python -m catagent.app` with the same options.

### 2. HTTP server

```bash
catagent server \
    --dotenv .env \
    --db catagent_checkpoints.db \
    --host 127.0.0.1 --port 5600 \
    --job-manager-url http://127.0.0.1:8000

# --url overrides --host/--port
catagent server --url http://127.0.0.1:5600
```

Routes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Single-page debug console. |
| `POST` | `/chat` and `/` | Start a turn or resume an interrupt; streams SSE payloads. |
| `GET` | `/plan/{thread_id}` | Current plan projection (steps + `finalization`). |
| `GET` | `/events/{thread_id}?after=N` | Payloads retained for a thread, for the debug console. |
| `POST` | `/jobmanager/messages` | Progress messages posted to jobmanager's `front_end_url`. |

In server mode set jobmanager's `callback_url` to the cat-agent address (for
example `http://127.0.0.1:5600`); `JobResult` callbacks are accepted on both `/`
and `/chat`.

Both runtime modes configure the `catagent` logger before starting. Records go
to stderr by default, or to a file with `--log`; `--agent-log-level` controls
that logger independently of uvicorn's `--log-level`:

```bash
catagent server \
    --log catagent.log --agent-log-level info \
    --log-level warning
```

### 3. Debug web UI

Open `http://127.0.0.1:5600/` after starting the server. The page posts to
`/chat`, polls `/events/{thread_id}`, and renders one card per
`StreamPayload.source`:

- the orchestrator (source is the thread id),
- each StepNode (source is `<thread_id>:<step_id>`),
- any other source reported by the graph.

A jump bar under the header lists one chip per card (orchestrator, each
StepNode, any other source) plus the jobmanager panel. Clicking a chip scrolls
that container into view and briefly highlights it.

The right-hand panel shows the jobmanager agent's progress. Point jobmanager's
`front_end_url` at the cat-agent service:

```json
"front_end_url": "http://127.0.0.1:5600/jobmanager/messages"
```

The panel is best-effort, exactly like jobmanager's progress stream. Job
submissions resume automatically through the normal `callback_url`; other
interrupts have an answer box in their own card. The event log is in memory and
bounded, so it is intended for debugging only. Each `/events` response carries
the log's `generation` and per-thread `latest_seq`; the page uses them to notice
a server restart (or an evicted bucket) and replay from zero instead of waiting
on a cursor that can never advance. `learn/event_log_demo/` demonstrates the
whole mechanism with a standalone FastAPI app.

---

## Configuration

### Model credentials (`.env`)

```dotenv
MODEL_NAME=DeepSeek-V4.1-Flash
BASE_URL=https://your-endpoint
API_KEY=...
```

`MODEL_NAME` also selects the client class: names containing `qwen`, `glm`, or
`deepseek` map to `langchain-qwq`, `langchain-openai`, and `langchain-deepseek`
respectively. `catagent/app/runtime.py` builds the model, the SQLite
checkpointer, and the compiled graph from these values.

The test helper in `check/common.py` additionally looks for
`/home/luandong/bin/.env` and `jobmanager/.env`.

### Skill index

`--skill-index-json` is required by `catagent.app`. Point it at a JSON array
such as `skills/skill_index.json`:

```json
[
  {
    "name": "dpa4",
    "path": "dpa4/skill.md",
    "summary": "DPA4 force-field finetuning and deployment.",
    "references": [{"path": "references/job_submission.md", "purpose": "Job template."}],
    "scripts": [{"path": "scripts/get_input_json.py", "purpose": "Build input.json."}]
  }
]
```

Paths may be absolute, relative to the index file, or use the standard
`skills/<name>/skill.md` layout next to the index. The loader verifies each
`skill.md` exists and remaps stale absolute paths to `skills/<name>/` next to
the index. The resulting metadata is available to
`get_skill_summaries` / `get_skill_content` in the first orchestrator turn.

### Jobmanager

The jobmanager has its own `.env` and `jobmanager_config.json`. See
[jobmanager/README.md](jobmanager/README.md) for providers, callbacks, retries,
crash recovery, and the demo. Start it with:

```bash
python -m jobmanager.server \
    --setting jobmanager/jobmanager_config.json \
    --host 127.0.0.1 --port 8000 \
    --log jobmanager.log --level info
```

---

## Tests

The fast tests use a fake computation pipeline and need no network, GPU, or LLM.
Run them from the repository root:

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

Real-LLM tests (require credentials; can take 5–20 minutes):

```bash
python -m check.test_agent_orchestrator
python -m check.test_agent_e2e --timeout 180
python -m check.test_agent_real_jobmanager --timeout 1800
```

The e2e test starts a local jobmanager on port `18000` and a callback server on
`5600`, drives the real graph, and requires `finalization=True` with all steps
`done`.  `test_agent_real_jobmanager` is the real-provider regression test: it
runs four concurrent StepNodes against the real `bohr` and `slurm-1` providers
using the VASP inputs under
`/mnt/t/workdir/agent_test/case0/test_cases`, and verifies the downloaded
`OUTCAR`/`CONTCAR` contain the requested element and atom count.
See [check/README.md](check/README.md) for the fake pipeline, ports,
environment variables, and troubleshooting.

A quick smoke test of the application package and debug CLI:

```bash
python -c "import catagent.app as a; print(a._stream_chat, a.build_runtime)"
python -m catagent.app --help
```

The development environment also has transport-level checks for the HTTP
backend, server routes, real uvicorn startup, and live remote mode.

---

## Key points for a new developer

### Architecture invariants

1. **The package facade is public API.** `catagent/app/__init__.py` re-exports
   `_stream_chat`, `build_runtime`, `create_llm_model`, `get_plan`,
   `build_app`, `_parse_bind_url`, and the CLI functions. `check/` and external
   tools import them from `catagent.app`. If you rename a module, update the
   facade.
2. **Entry-point collaborators stay patchable.** `cli.main` resolves
   `run_server`/`run_cli`, `run_server` resolves `build_app`, and `build_app`
   resolves `build_runtime` through the package facade at call time. This
   preserves the old single-module monkeypatch behaviour used by the tests.
3. **Outer and inner graphs need distinct thread ids.** See
   `catagent/app/runtime.py` and `catagent/agent.py`. Reusing one id corrupts
   checkpoints.
4. **Never copy the orchestrator's whole `steps` state back.** Return a delta
   (`_steps_delta`), and rehydrate queued requests from the live outer state.
5. **Keep the orchestrator serialized per thread.** `_ORCHESTRATOR_LOCKS` /
   `_ORCHESTRATOR_CLAIMS` in `agent.py` are keyed by `thread_id`; do not replace
   them with a process-global lock.
6. **Exactly one resume owner per job.** `register_jobs` selects between the
   jobmanager callback and client polling. Do not enable both.
7. **`router` uses a local `steps_view`.** A node's `update` is not visible to
   the rest of the same function, so dependency checks must operate on the local
   copy after applying results.
8. **Never busy-wait on graph state.** A running node sees an immutable snapshot;
   use the per-thread `asyncio` primitives instead.
9. **Tools report recoverable errors as `ToolMessage(status="error")` or
   `Command`, not by raising.** An uncaught exception aborts the graph.
10. **The step contract is files.** The orchestrator prepares `workdir/` and the
    router writes `readme.md`; the StepNode writes `report.md`. `summary_node`
    reads `report.md` and fails the step if it is missing or empty.
11. **Skill payload filtering is display-only.** `payloads.py` truncates
    `get_skill_summaries`, `get_skill_content` and
    `load_skill_from_filesystem` messages to 100 characters. Never truncate the
    `ToolMessage` itself; the model needs the full skill text.

### The `app` package

| Module | Responsibility |
| --- | --- |
| `models.py` | `ChatRequest`, `StreamPayload`. Update here when the wire shape changes. |
| `payloads.py` | LangGraph v2 chunks → payloads. Keep it transport-agnostic. |
| `skills.py` | Load `skill_index_json`; seed and remap initial `loaded_skills`. |
| `runtime.py` | Model inference, SQLite checkpointer, graph construction. |
| `jobs.py` | Poll `/status/{job_uuid}` and build the resume map for the debug CLI. |
| `streaming.py` | `_stream_chat`, payload annotation, job registration. |
| `plan.py` | `get_plan`; the read-only plan projection used by HTTP. |
| `service.py` | FastAPI routes, SSE framing, resume parsing, lifespan cleanup. |
| `cli.py` | Argument parsing, debug CLI, bind URL parsing, uvicorn launch. |

When adding a feature:

- Put pure data shapes in `models.py`.
- Put chunk/session translation in `payloads.py` or `streaming.py`.
- Put HTTP concerns in `service.py`, never in the graph.
- Test the behavior, not the module layout: callers should keep using
  `catagent.app.<name>`.

### Two payload/stream helpers

Do not confuse them:

- `catagent/app/payloads.py` (discriminant `type`: `user`/`agent`/`tool`/
  `interrupt`/`custom`) is what the HTTP service and tests consume.
- `catagent/run_agent.py` (discriminant `role`, with de-duplication for
  subgraphs) is a standalone generic helper. It is useful as a reference but is
  not the service path.

### Extending the workflow

- **New step tool**: add it to `catagent/stepnode/init_tools.py` and (if it
  should be gated) to a tool collection.
- **New orchestrator request phase**: add it to `router.py`, handle it in
  `orchestrator.py` (`get_current_phase` and the system prompt), and make sure
  `node_set_flag` still returns to `router`.
- **New interrupt kind**: add the model in `catagent/stepnode/models.py`, handle
  it in `create_interrupt_node`, and document its resume shape in this README.
- **New runtime mode**: build on `build_runtime`/`build_app` rather than
  duplicating model and checkpointer setup.
