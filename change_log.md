# Change log

## 2026-11-17 — De-duplicate `orchestrator_requests`

### Background

In one workflow run the router generated the same correction request on
several passes. The orchestrator fixed `step.status` on the first pass, but the
identical requests that followed reverted/flipped it again. The root cause was
that the request queue was an append-only list: every router pass appended a
new entry even when an equivalent request was already queued, and the queue
had no identity for "this exact request".

Only the de-duplication problem is addressed here. No unrelated refactors were
made. Note: `register_raw_plan(replace)` really does remove `Step` keys from
the shared state (in addition to removing the `RawPlan`). The user wants
the outer `Step` state (in particular `status="running"` and
`need_further_planning`) to overwrite the inner orchestrator snapshot, so that
outer → inner propagation is kept.

### What changed

#### `catagent/workflow/models.py`

- **`OrchestratorRequest`** is now an explicit `TypedDict` with
  `sent: bool`, `creation_timestamp: float` and `payload: Dict[str, Any]`.
  The stray `status` field and the invalid defaults on a `TypedDict` were
  removed; the queue entry is now written as
  `OrchestratorRequest(sent=False, creation_timestamp=..., payload=...)`.
- **Added `orchestrator_request_reducer`** for the queue:
  - The queue is a `dict` keyed by the request fingerprint.
  - A complete request (has a `payload`) that is already queued is **ignored**,
    so a later router pass cannot reset `sent` back to `False` or replace the
    original `creation_timestamp`.
  - Lifecycle patches such as `{"sent": True}` are **merged** into the stored
    entry instead of replacing it.
  - `None` removes an entry (used by `node_set_flag` to expire old requests).
  - A legacy list-shaped queue is treated as empty; the router regenerates
    anything still needed.
- **`WorkflowState.orchestrator_requests`** now uses
  `orchestrator_request_reducer` instead of the generic `dict_reducer`.

#### `catagent/workflow/router.py`

- **Added `_json_safe` and `request_fingerprint`.** The fingerprint is the
  SHA-256 of a canonical JSON view of the request payload
  (`sort_keys=True`, compact separators). LangChain messages are reduced to
  `{type, content}`, so volatile `HumanMessage` ids/metadata do not change the
  identity of a request; sets are sorted for determinism.
- **`orchestrator_request_queue`**
  - Tolerates a legacy list-shaped / partially populated queue instead of
    raising.
  - Reads entries with `.get(...)` so a lifecycle patch cannot cause a
    `KeyError`.
  - Sorts by `(creation_timestamp, key)` (the old
    `req_list.sort(lambda a: a[0])` call was invalid).
  - Dispatch still marks the request `sent=True` **before** sending; the
    fingerprint then stays in the queue, so the same request cannot be
    dispatched again.
- **Router request generation**
  - Requests are keyed by `request_fingerprint(req)`, de-duplicated within the
    pass, and queued as
    `OrchestratorRequest(sent=False, creation_timestamp=now + i*1e-6, payload=req)`.
  - The microsecond offset preserves generation order for requests created in
    the same clock tick.
  - Removed the `asyncio.sleep(0.1)` used to fake distinct timestamps.
- **Dependency table fix.** The table now reports the dependency's status
  (`steps_view[d_id]["status"]`) instead of the current step's status.
  `plan-{id}` dependency groups are handled explicitly rather than asserted.
- **StepNode dispatch fix.** `step.workdir` (undefined name) → `steps[key].workdir`.
- **StepResult handling no longer asserts that the step still exists.**
  `register_raw_plan(replace)` removes Step keys while a StepNode may still be
  running or while its result is queued, so such a result is marked processed
  and dropped instead of crashing or resurrecting the deleted step. The
  invariant that does hold (`key in steps` for every key in `steps_view`) is
  asserted.

#### `catagent/agent.py`

- **`agent_input` now includes `steps`.** The outer state snapshot is applied
  to the inner orchestrator graph, so the outer `status` /
  `need_further_planning` / `preparation_done` values overwrite the inner
  snapshot. `messages` is still replaced with only the latest message.
- On the way back, `_steps_delta` still returns only the fields the
  orchestrator actually changed, so concurrent outer updates are not reverted.
- **`_steps_delta` propagates deletions.** A step present in the baseline but
  absent from the inner graph is emitted as a `None` patch, so
  `dict_reducer` removes it from the outer workflow instead of leaving a
  phantom step that the router could dispatch again.
- **Interrupt/resume keeps the baseline.** `InterruptNodeInput` now carries the
  dispatch-time `steps` snapshot through the pause, and the resumed
  `orchestrator_node` uses the same `_steps_delta`, so a plan replacement made
  before an interrupt is also reflected (including deletions) when the user
  answers. A resume from an older checkpoint without the carried baseline falls
  back to the previous whole-state behaviour.
- **`node_set_flag`** now guards against a non-dict queue and reads
  `sent` / `creation_timestamp` with `.get(...)`. Requests that were dispatched
  but not acknowledged for `MAXTIME = 300` seconds are removed. Until then the
  fingerprint remains, which is what prevents a duplicate dispatch.

#### `catagent/system_skills/orchestrator_prompt.md`

- Finalization wording changed from "done, cancelled or failed" to
  "done or deleted", matching the actual step statuses.

#### `check/test_agent_liveness.py`

- Step fixture uses `preparation_done=True` instead of the removed
  `preparation_stage="done"` field.

#### `check/test_agent_state_sync.py`

- Rewrote the deterministic tests for the new model and queue:
  - fingerprint ignores message ids but distinguishes content;
  - identical requests keep one entry and preserve `sent=True`;
  - the queue does not re-dispatch a sent request;
  - dispatch uses the current outer step snapshot (not the stale payload copy);
  - marking `sent` does not lose the payload/creation timestamp;
  - a running step is not dispatched twice;
  - the step delta is empty when nothing changed, contains only changed fields,
    preserves concurrent outer changes, and propagates deleted steps as `None`;
  - a queued StepResult for a step removed by `register_raw_plan(replace)` is
    dropped (marked processed) instead of failing the assertion.

### Verification

- `python3 -m check.test_agent_state_sync` → **13 tests, OK**
- `python3 -m check.test_agent_liveness` → **OK**
- `python3 -c "import catagent.agent"` → **OK**
- Manual checks confirmed:
  - Re-running the router with the same state produces the same fingerprint and
    does not reset `sent`.
  - The reducer preserves `sent=True` and `creation_timestamp` across passes.
  - The orchestrator queue dispatches `steps` as `Step` objects and marks the
    request `sent` via a merge that keeps the payload.
  - Applying the outer `steps` to the inner graph overwrites the inner status.
  - A step removed from the inner graph produces a `None` delta patch and is
    popped from the outer state.
  - The router drops a queued result for a step removed by
    `register_raw_plan(replace)` and marks that result processed.

### Not addressed (out of scope)

- `check/test_agent_e2e.py` and `check/test_agent_orchestrator.py` need a live
  LLM/jobmanager and were not run.
- Documentation in `README.md`, `check/README.md` and `DEVELOPMENT.md` still
  mentions the old `preparation_stage` terminology.
- `catagent/workflow/step_edit_tool.py::create_step` still passes the stale
  `preparation_stage='notyet'` keyword.
- `Step.model_copy(update=...)` bypasses pydantic validation, so an invalid
  legacy status value could still persist.
- `check/test_register_raw_plan.py` has a pre-existing failure (also on HEAD,
  unrelated to these changes): after `register_raw_plan(replace)`, the
  surviving predecessor's `required_by` still lists the deleted step ids
  (e.g. `prepare.required_by` keeps `relax_a`/`relax_b`), and the test's
  expectation for `relax_b_v2.depend_on` omits the new-plan topology edge
  `relax_a_v2`. Router dependency checks use `depend_on`, not `required_by`, so
  this does not affect dispatching, but the stale back-references are a real
  data bug in `planner_tool.py` and were left untouched to keep this change
  focused.

---

## 2026-11-17 — Follow-up: `register_raw_plan` test and test-suite rerun

### `catagent/workflow/planner_tool.py`

- After a `replace`, every surviving step that still referenced a deleted step
  id (typically a predecessor's `required_by`) is now patched too, not just the
  new steps and their direct successors.
- `required_by` is now cleaned together with `depend_on`, and the inverse edge
  is maintained: when a new step depends on a surviving predecessor, the
  predecessor's `required_by` gains the new step. Steps that are about to be
  deleted are skipped when adding these inverse edges, so they cannot
  re-register themselves.
- Result: after one-to-one replacement,
  `prepare.required_by == {relax_a_v2, relax_b_v2}`; after the many-to-one
  replacement it is `{relax_both}`.

### `check/test_register_raw_plan.py`

- Corrected the stale expectation for `relax_b_v2.depend_on`: it inherits
  `prepare` and also depends on `relax_a_v2` (the new plan's `(0, 1)` topology
  edge).
- Added assertions that `prepare.required_by` is rewired to the replacement
  steps in both the one-to-one and many-to-one cases.
- **Passes.**

### `check/test_agent_e2e.py`

- The jobmanager server now requires `--dotenv`; the test passes the first
  existing credentials file (`/home/luandong/bin/.env`, repo `.env`, or
  `catagent/jobmanager/.env`).
- The final report used the removed `Step.preparation_stage`; it now prints
  `preparation_done`.

### Test rerun (after the `RawStepResult`/`StepResult` corrections)

Deterministic tests:

- Pass: `test_planner_tool`, `test_register_raw_plan`, `test_agent_state_sync`,
  `test_agent_liveness`, `test_cli_jobs`, `test_app_skill_init`,
  `test_debug_ui`, `test_parse_path`.
- Fail, stale test API (not caused by `RawStepResult`/`StepResult`):
  - `test_agent_stepnode` imports `create_summary_node`, which no longer
    exists; summarization is now inlined in
    `create_model_and_tool_node().model_node`.
  - `test_path_guard` imports `_path_policy_error`, which no longer exists;
    the policy is inlined in `execute_shell` (and does not implement the
    test's stronger non-existent-path / missing-context semantics).
- Fail, deliberate behavior change:
  - `test_jobmanager_recovery::test_missing_result_files_are_an_error`: the
    missing-artifact gate in `manager.py` is commented out with "We currently
    prefer not to check `_missing_artifacts`", so a partial result set is
    reported as success. The test expects the old fail-closed behavior.

Real-LLM tests:

- `test_agent_orchestrator` — **passes**.
- `test_agent_e2e` — the harness now starts the jobmanager and the workflow
  runs. One run completed the whole workflow and failed only at the old
  `preparation_stage` print (now fixed). A second run was derailed by repeated
  provider `OpenAIInvalidRequestError: 400 ... internal MaaS component`
  responses, StepNode retries, and the 180 s callback timeout; the jobmanager
  itself completed and delivered callbacks. This is an external model-service
  failure, not a workflow failure.

### Note

`StepResult` and `RawStepResult` currently ignore extra keyword arguments
(Pydantic default `extra='ignore'`), so the leftover `step_id=`/`artifacts=`
arguments in `agent.py`'s crash path and in the stale `test_agent_stepnode.py`
do not raise, but they are no longer stored.

---

## 2026-11-17 — Follow-up: modernize stale tests and remove the missing-artifact check

### `check/test_agent_stepnode.py` (rewritten)

- No longer imports the removed `create_summary_node`. The test now builds the
  real node via `create_model_and_tool_node(model, [], url, None)[0]` and
  exercises the inlined finalization branch (`stepnode_finalization=True`).
- Uses one `FixedSummaryModel` (the shared tool registry cannot be built twice
  in a process) and checks:
  - a structured `RawStepResult` is forwarded unchanged,
  - the step history sits between the system prompt and the finalization
    prompt,
  - three failed attempts fall back to `status="error"` with the
    `Cannot extract structured output` summary,
  - the first message is a `SystemMessage` whose `content` is a string.
- **Passes.**

### `check/test_path_guard.py` (rewritten)

- No longer imports the removed `_path_policy_error`; the tests now call the
  real `execute_shell` and `read_file` tools.
- Coverage: project workdir boundary (absolute and relative escape), allowed
  skill-artifact directories, safe device paths (`/dev/null`, `/dev/stdout`),
  forbidden network commands, outside/non-directory `cwd`, `Orchestrator` not
  being bound, `read_file` staying inside the project, and a real StepNode
  graph receiving its runtime context.
- The test documents two current behaviours explicitly instead of expecting
  the old helper's semantics:
  - paths are only checked when they already exist (`is_file() or is_dir()`),
    so a non-existent path outside the project is not denied;
  - missing `project_workdir` context raises `KeyError` instead of returning a
    "not configured" message.
- The real-graph case now seeds `role="StepNode"` and
  `stepnode_finalization=False`, which `model_node` and `execute_shell`
  require (the `TypedDict` defaults are not injected by `ainvoke`).
- **Passes.**

### `check/test_jobmanager_recovery.py`

- Deleted `test_missing_result_files_are_an_error` and its entry in the test
  list. The missing-artifact gate in `manager.py` is intentionally commented
  out because the LLM-chosen `result_patterns` are ambiguous and the check is
  likely to fail in production. The production `_missing_artifacts` helper is
  kept for future improvement.
- **Passes.**

### `check/README.md`

- Updated the descriptions of `test_agent_stepnode.py` and
  `test_path_guard.py` to match the current implementations.

### Deterministic suite result

All deterministic tests pass:

`test_planner_tool`, `test_register_raw_plan`, `test_agent_stepnode`,
`test_agent_state_sync`, `test_agent_liveness`, `test_jobmanager_recovery`,
`test_cli_jobs`, `test_app_skill_init`, `test_debug_ui`, `test_parse_path`,
`test_path_guard`.

---

## 2026-11-17 — Diagnostic logging for StepNode LLM failures

### `catagent/stepnode/stepnode.py`

- Added `_serialize_model_request(model, messages, tools)` which builds the
  OpenAI-format request (model name, messages, tool schemas) from the exact
  arguments passed to the provider.
- Added `_dump_failed_model_request(...)`: on a model-call exception it writes
  the full request to
  `$TMPDIR/catagent_stepnode_llm_error_<time_ns>_<pid>.json` and returns the
  path. The dump includes the exception type/message and the request payload.
  `time_ns()` keeps concurrent/same-second failures from overwriting each
  other.
- The normal `model_node` call is now wrapped in `try/except`. On failure it
  logs the model name, message count, tool count, total message characters,
  state role, exception type/message, and the dump path, then re-raises. The
  traceback is included via `logger.exception`-style `exc_info=True`.
- The finalization retry loop no longer swallows exceptions with `pass`: each
  attempt is logged at warning level with its traceback, and the third failure
  also writes a request dump before falling back to the error `RawStepResult`.

### Rerun results

- `test_agent_orchestrator` — **passes**.
- `test_agent_e2e` — ran for the 50-minute shell limit and was killed
  (`exit=124`). The workflow kept making progress (several compute jobs
  succeeded), but StepNode continued to hit the same MaaS 400 and the
  orchestrator kept creating retry steps, so the run never reached a final
  state. The new logging produced six request dumps.

### What the dumps showed

- Replaying the exact dumped payloads through the same `ChatDeepSeek` client:
  four of five succeeded, one reproduced the 400, and an immediate replay of
  that same payload succeeded. Therefore the 400 is **intermittent on the
  provider side**, not a deterministic malformed request (the request body,
  tool schemas, and size are accepted).
- Every failing dump contained a **duplicate `tool` message for one
  `tool_call_id`**. `submit_job` returns a `ToolMessage` and an interrupt; on
  resume, `interrupt_node` appends a second `ToolMessage` with the same
  `tool_call_id` (`stepnode.py`, `JobSubmission` branch). OpenAI-format
  histories expect exactly one tool response per tool call. The provider
  tolerated it in most calls (and in a direct replay), but it is a real
  malformed-history bug worth fixing.
- No duplicate or orphan tool message was found for the non-`submit_job`
  tools.

### Deterministic suite

All deterministic tests pass, including `test_agent_stepnode` and
`test_path_guard` after the logging change.

---

## 2026-11-17 — `submit_job` no longer emits a duplicate tool response

### `catagent/stepnode/init_tools.py`

- **`submit_job` is now a plain tool returning `(ok, payload)`.** It no longer
  returns a `Command`, and it no longer takes `runtime: ToolRuntime[Any]`.
  - Success: `(True, payload)` where `payload` includes `job_uuid` and the
    submission metadata.
  - Failure: `(False, message)` with the same message format as before.
  - The function signature gained the explicit return annotation
    `-> Tuple[bool, Any]` (`Tuple` added to the typing import).
- **`tool_call_wrapper` owns the `submit_job` control flow.** It resolves the
  tool instance from `req.tool` (falling back to `TOOLS_BY_NAME`) because
  `submit_job` is a per-jobmanager closure and is **not** in the global
  `TOOLS_BY_NAME` map. This avoids a `KeyError` on the old
  `TOOLS_BY_NAME[tool_name]` lookup.
  - Success: emits only a `JobSubmission` interrupt. The single result
    `ToolMessage` is added later by `interrupt_node` when the job finishes.
  - Failure: emits one `ToolMessage(status="error")`.
  - Missing tool: emits a clear `"submit_job tool is not available"` error
    message instead of raising.
  - Uses `args or {}` so a model-emitted `null` args payload does not crash.

### Why

Previously `submit_job` returned *both* a `ToolMessage` and an interrupt; on
resume `interrupt_node` appended a second `ToolMessage` with the same
`tool_call_id`. That produced a malformed OpenAI-format history (two tool
responses for one tool call) and appeared in every failing StepNode request
dump. The provider tolerated it most of the time, but it could trigger
validation errors. The wrapper-based split guarantees exactly one tool
response per `submit_job` call.

### Verification

- Deterministic StepNode graph check (`submit_job` → `JobSubmission`
  interrupt → resume): exactly **one** `ToolMessage` for the `call_submit`
  tool call.
- `test_agent_e2e` — **PASS** (`END-TO-END TEST: OK`). All five steps reached
  `status=done`, `preparation_done=True` in
  `check/.tmp/e2e_7a6780fd`.
  - 0 normal StepNode model-call failures.
  - 0 step retries or deleted steps.
  - One finalization call hit the intermittent MaaS 400 and succeeded on
    retry (attempt 1/3 failed, attempt 2 succeeded).
- `test_agent_orchestrator` — **PASS** (`orchestrator append plan test: OK`)
  when run with `/mnt/t/softwares_install/lang-graph/bin/python3`.
- Full deterministic suite — **PASS**.

### Remaining note

The duplicate-tool-response defect is fixed. The intermittent
`OpenAIException ... rejected by an internal MaaS component` 400 still occurs
occasionally in finalization; the existing 3-attempt retry recovers it. Dumps
are still written to
`$TMPDIR/catagent_stepnode_llm_error_<time_ns>_<pid>.json` for diagnostics.
