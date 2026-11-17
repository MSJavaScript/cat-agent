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
made, and `steps` are still never removed (only `RawPlan`s are). The user wants
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
- Kept the user's `assert key in steps_view` in the StepResult handling; steps
  are never deleted, so this invariant holds.

#### `catagent/agent.py`

- **`agent_input` now includes `steps`.** The outer state snapshot is applied
  to the inner orchestrator graph, so the outer `status` /
  `need_further_planning` / `preparation_done` values overwrite the inner
  snapshot. `messages` is still replaced with only the latest message.
- On the way back, `_steps_delta` still returns only the fields the
  orchestrator actually changed, so concurrent outer updates are not reverted.
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
    and preserves concurrent outer changes.

### Verification

- `python3 -m check.test_agent_state_sync` → **11 tests, OK**
- `python3 -m check.test_agent_liveness` → **OK**
- `python3 -c "import catagent.agent"` → **OK**
- Manual checks confirmed:
  - Re-running the router with the same state produces the same fingerprint and
    does not reset `sent`.
  - The reducer preserves `sent=True` and `creation_timestamp` across passes.
  - The orchestrator queue dispatches `steps` as `Step` objects and marks the
    request `sent` via a merge that keeps the payload.
  - Applying the outer `steps` to the inner graph overwrites the inner status.

### Not addressed (out of scope)

- `check/test_agent_e2e.py` and `check/test_agent_orchestrator.py` need a live
  LLM/jobmanager and were not run.
- Documentation in `README.md`, `check/README.md` and `DEVELOPMENT.md` still
  mentions the old `preparation_stage` terminology.
- `catagent/workflow/step_edit_tool.py::create_step` still passes the stale
  `preparation_stage='notyet'` keyword.
- `Step.model_copy(update=...)` bypasses pydantic validation, so an invalid
  legacy status value could still persist.
