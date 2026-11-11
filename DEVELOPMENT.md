# Development notes

## Replacing the embedded debug UI with an independent Next.js / React app

The current debug console is a single HTML file served by FastAPI at `GET /`.
Replacing it with a separate frontend does **not** require changes to the graph,
the tools, or jobmanager. The service is already API-first; the embedded page
is only one consumer of that API.

This document records the contract, the backend changes that are actually
required, the frontend structure, deployment caveats and a migration checklist.

---

## 1. Current API contract

The debug page in `catagent/app/debug_ui.html` uses only these routes:

| Frontend need | Endpoint | Notes |
| --- | --- | --- |
| Render a thread | `GET /events/{thread_id}?after=N` | Primary rendering path. Returns `events`, `latest_seq`, `generation`. |
| Render jobmanager progress | `GET /events/__jobmanager__?after=N` | Same route, synthetic bucket (`JOBMANAGER_SOURCE`). |
| Send a message / resume | `POST /chat` | Returns SSE; thread id is returned in the `X-Thread-Id` response header. |
| Jobmanager progress input | `POST /jobmanager/messages` | Called by jobmanager, not by the browser. |

`GET /plan/{thread_id}` exists and returns the plan projection, but the current
page does not use it. `POST /` is an alias of `POST /chat` used by the
jobmanager callback and must be preserved.

Relevant files:

- `catagent/app/service.py` — `build_app`, `events_route`, `jobmanager_messages`, `chat`
- `catagent/app/events.py` — `DebugEventLog`, `JOBMANAGER_SOURCE`
- `catagent/app/models.py` — `ChatRequest`, `StreamPayload`
- `catagent/app/streaming.py` — `_stream_chat`
- `catagent/app/debug_ui.html` — the reference client implementation

### Event payload shape

`StreamPayload` in `catagent/app/models.py` is the source of truth. Important
fields:

- `type`: `user`, `agent`, `tool`, `interrupt` or `custom`
- `message`: string or arbitrary object
- `message_id`: stable message identity
- `source`: thread id for the orchestrator, `<thread_id>:<step_id>` for a
  StepNode, other values for custom sources
- `thread_id`: outer graph thread id
- `tool_calls`, `tool_call_id`, `tool_call_name`
- `interrupt_id`, `interrupt_response_schema`

The server does not currently emit a `user` event; the debug page echoes user
text locally. A real app needs to decide whether to keep doing that or to have
the server record it (see §2.6).

---

## 2. Backend changes

### 2.1 CORS — required for a separate origin

If the frontend runs at `http://localhost:3000` and cat-agent at `:5600`, add
`CORSMiddleware` in `build_app()`:

```python
import os
from fastapi.middleware.cors import CORSMiddleware

origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "CATAGENT_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["X-Thread-Id"],
    max_age=600,
)
```

Notes:

- `POST /chat` sends JSON, so the browser performs a preflight; middleware
  handles it.
- `expose_headers` is mandatory, otherwise JavaScript cannot read
  `X-Thread-Id` from a cross-origin response.
- Do not use `allow_origins=["*"]` together with credentials. The service does
  not use cookies today, so no credentials are needed.
- CORS is not authentication. If the API is exposed beyond localhost, add a
  token check and allow the `Authorization` header.

Alternative to CORS: put both applications behind one reverse proxy:

```
/          -> Next.js :3000
/api/*     -> cat-agent :5600
```

This avoids CORS entirely, but proxying an SSE stream needs care (§4).

### 2.2 What happens to `GET /`

Keep `POST /`: jobmanager posts its `JobResult` there.

The embedded page can be retired by:

- removing `_DEBUG_UI_HTML` and the `GET /` route from `service.py`,
- deleting `catagent/app/debug_ui.html`,
- removing the `app/debug_ui.html` entry from `[tool.setuptools.package-data]`.

Alternatively keep the route as a redirect to the new UI URL, or return a small
JSON service description. During migration it is safer to keep the old page
working.

### 2.3 Make thread ids explicit

`service.py` keeps `app.state.last_thread_id` for stateless resume. That is a
single-user convenience and must not be used by a multi-user frontend.

Rules for the new client:

- generate a thread id with `crypto.randomUUID()` when starting a new chat;
- send it in the first `/chat` request and on every resume;
- keep it in the URL (`/chat/<thread_id>`) and optionally in `localStorage`;
- never omit `thread_id` for resume requests.

No backend change is strictly required for this, but the global
`last_thread_id` fallback should eventually be removed or made opt-in.

### 2.4 Add history endpoints

The in-memory event log is bounded and volatile. Conversation state survives in
the LangGraph checkpoint database, but there is no route to read it. A real UI
needs at least:

```python
GET /threads/{thread_id}/messages   # messages from catagent.aget_state()
GET /threads                        # thread ids from the checkpoint DB
```

The plan route already exists:

```python
GET /plan/{thread_id}               # steps + finalization flag
```

Without a messages endpoint, reloading the page after a server restart shows an
empty conversation even though the graph state exists.

### 2.5 SSE heartbeat

`event_stream()` can be silent for minutes during a long model or tool call.
Intermediate proxies often close an idle response. Emit a comment periodically:

```python
yield ": ping\n\n"
```

For example on a 15-second timeout while awaiting the next graph payload.
The service already sets `X-Accel-Buffering: no`; the proxy also needs
`proxy_buffering off` and a generous `proxy_read_timeout`.

### 2.6 Optional: record user messages

If replay should include the user's own text, append a user payload to the log
before starting the stream in `chat()`:

```python
app.state.event_log.append(thread_id, {
    "type": "user",
    "message": req.message,
    "message_id": str(uuid.uuid4()),
    "source": thread_id,
    "thread_id": thread_id,
})
```

Only do this for new messages (`_normal_input`), not for interrupt resumes. If
the embedded page is still in use, it must stop rendering its local echo for
this thread, otherwise the message appears twice.

### 2.7 Optional: push instead of polling

Polling is sufficient for a single-user console. For lower latency or many
clients, add one of:

- `GET /events/{thread_id}/stream` — SSE that tails the event log. This needs a
  per-bucket notification primitive (asyncio condition/event) because
  `DebugEventLog.append()` is synchronous.
- `GET /events/{thread_id}?after=N&wait=30` — long polling.

Keep the existing JSON route stable; the frontend can switch delivery later.

### 2.8 Optional: API prefix

If the frontend and backend share a domain behind a reverse proxy, mount the
API under `/api` (`APIRouter(prefix="/api")`) and keep the old paths as aliases
during migration. This avoids collisions with Next.js routes.

---

## 3. Frontend structure

Suggested layout if the app lives in this repository:

```text
web/                          # excluded from the Python wheel
├── app/
│   ├── layout.tsx
│   ├── page.tsx              # new chat
│   └── chat/[threadId]/page.tsx
├── lib/
│   ├── catagent.ts           # API client, SSE reader, polling
│   └── types.ts              # mirror of StreamPayload / event response
└── .env.local                # NEXT_PUBLIC_CATAGENT_API_URL=http://127.0.0.1:5600
```

Add `web` to the packaging exclusions, exactly like `check/` and `learn/`:

- `[tool.setuptools.packages.find] exclude` in `pyproject.toml`
- `prune web` in `MANIFEST.in`

If the frontend is a separate repository, none of this applies.

### 3.1 Sending and resuming

`EventSource` cannot POST, so use `fetch`:

```ts
const res = await fetch(`${API}/chat`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    message,
    thread_id: threadId,
    register_jobs: true, // let jobmanager resume the graph when the job finishes
  }),
});
```

The response is SSE. Two valid strategies:

1. **Drain and poll** — `await res.text()` and render from `/events`. This is
   what the current page does; polling remains the single rendering path.
2. **Parse the stream** — render immediately and still reconcile with polling:

```ts
const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
let buffer = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += value;
  const frames = buffer.split("\n\n");
  buffer = frames.pop()!;
  for (const frame of frames) {
    const data = frame.split("\n").find((line) => line.startsWith("data: "));
    if (data) onPayload(JSON.parse(data.slice(6)));
  }
}
```

If both paths render, deduplicate by `message_id` (or by event `seq`).

### 3.2 Polling and restart detection

Port `serverResetRequired` / `resetStreams` from `debug_ui.html`:

```ts
const data = await (await fetch(
  `${API}/events/${encodeURIComponent(threadId)}?after=${cursor}`,
  { cache: "no-store" },
)).json();

if (data.generation !== generation || cursor > data.latest_seq) {
  // Server restarted, or this bucket was evicted: clear and replay from zero.
  clearThread();
  cursor = 0;
  generation = data.generation;
  return poll(); // fetch once more from after=0
}
```

Ratchet `cursor` to `event.seq` and ignore lower sequence numbers. Poll the
jobmanager bucket separately from `__jobmanager__`.

### 3.3 Interrupt resume shapes

From the current page (`parseResume`):

- `AgentInterrupt` → `{ payload: { result: text } }`
- `AskForArgs` → `{ payload: { command: "edit", message: text } }`
- `JobSubmission` → no UI action; jobmanager resumes automatically.

Send:

```json
{
  "thread_id": "…",
  "interrupt_id": "…",
  "interrupt_resume": { "payload": { "result": "…" } }
}
```

### 3.4 Plan panel

Poll `GET /plan/{thread_id}` after each event batch or on an interval. It
returns:

```json
{ "thread_id": "…", "steps": [ ... ], "finalization": false }
```

Steps are sorted by `step_id` with a natural ordering.

### 3.5 Types

FastAPI exposes `/openapi.json` for the request/response schemas, but SSE
payloads are produced as plain dicts typed by `StreamPayload` (a `TypedDict`),
so they do not appear in the schema. Keep a hand-written TypeScript mirror of
`StreamPayload` and of the `/events` response, and update both when
`payloads.py` changes.

---

## 4. Deployment caveats

1. **Run a single uvicorn worker.** `DebugEventLog` is process-local. With
   `--workers N`, polling can reach a worker that has no history. Options:
   one worker, sticky routing, or replace the log with a shared store
   (Redis/DB). The SQLite checkpointer has its own single-process assumptions.
2. **SSE and proxies.** Disable buffering and raise read timeouts for `/chat`
   and `/events`. Verify that any Next.js rewrite or nginx layer does not buffer
   or time out the stream.
3. **Jobmanager progress is global.** All clients read the same
   `__jobmanager__` bucket. Acceptable for one user; for multi-user, make
   jobmanager include a `thread_id` with its progress posts and route those
   payloads into per-thread buckets in `service.py`.
4. **Jobmanager URLs do not change.** `front_end_url` and `callback_url` point
   at cat-agent, not at the Next app. The frontend only reads the bucket.
5. **Do not call jobmanager directly from the browser** unless CORS is added
   there too. Route provider/job queries through cat-agent if needed.
6. **Authentication.** If the API is not localhost-only, add an API token
   check. Prefer keeping the token in a Next route handler and proxying
   server-side rather than exposing it through `NEXT_PUBLIC_*`.

---

## 5. Suggested migration order

1. Add CORS, `expose_headers` and the SSE heartbeat. Keep the embedded page.
   Verify a minimal Next/React page can poll `/events` and post `/chat`.
2. Implement feature parity in the new app: cards grouped by `source`, jump bar,
   interrupt forms, jobmanager panel, generation-reset handling.
3. Add `GET /threads/{id}/messages` and optional `GET /threads` so reloads and
   restarts have real history.
4. Optionally record `user` events server-side and stop the local echo.
5. Retire `GET /` and `debug_ui.html`; update `pyproject.toml` and
   `MANIFEST.in`.
6. Only if needed: SSE push endpoint, API prefix, auth, per-thread jobmanager
   buckets, thread management routes.

---

## 6. Testing checklist

- Preflight: `OPTIONS /chat` from the UI origin returns the expected
  `Access-Control-Allow-*` headers.
- `X-Thread-Id` is readable by the browser.
- A message sent without `thread_id` gets one; a message sent with one reuses
  it.
- Reloading the page replays retained events; restarting the service clears the
  view and replays from zero instead of freezing.
- An interrupt can be resumed from the new UI.
- A real job submitted through jobmanager resumes automatically through
  `POST /` and appears in the new UI.
- Long streams survive the reverse proxy (heartbeat, no buffering).
- The existing deterministic checks still pass:
  `check/test_debug_ui.py`, `check/test_app_skill_init.py`,
  `check/test_cli_jobs.py`, `check/test_agent_state_sync.py`.
