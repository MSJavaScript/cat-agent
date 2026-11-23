"""Poll jobmanager for durable job results.

The debug CLI has no HTTP endpoint for the job manager to call back.  It
therefore passes ``register_jobs=False`` to
:func:`catagent.app._stream_chat` and polls ``GET /status/{job_uuid}`` until the
durable ``result_status`` is final, then resumes its local graph.

Keeping this logic in one module guarantees a consistent payload shape and
retry behaviour.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

import httpx


def job_uuid_from_interrupt(value: Any) -> str:
    """Extract ``job_uuid`` from a ``JobSubmission`` interrupt value.

    The normal shape is ``{"kind": "JobSubmission", "payload": {"job_uuid": ...}}``;
    a top-level ``job_uuid`` is accepted for direct callers.
    """
    if not isinstance(value, dict):
        return ""
    if value.get("job_uuid"):
        return str(value["job_uuid"])
    payload = value.get("payload")
    if isinstance(payload, dict) and payload.get("job_uuid"):
        return str(payload["job_uuid"])
    return ""


async def wait_for_job(
    http_client: httpx.AsyncClient,
    job_manager_url: str,
    job_uuid: str,
    *,
    poll_interval: float = 2.0,
    poll_interval_max: float = 10.0,
) -> Dict[str, Any]:
    """Poll jobmanager until ``job_uuid`` has a final result.

    ``JobRecord.status`` can stay non-terminal for a long time, so the durable
    ``result_status`` (``success``/``error``) is the only completion signal.
    Transient HTTP errors and a 404 while the row is being written are retried.
    """
    url = f"{job_manager_url.rstrip('/')}/status/{job_uuid}"
    delay = max(0.2, float(poll_interval))
    max_delay = max(delay, float(poll_interval_max))

    while True:
        data: Dict[str, Any] | None = None
        try:
            response = await http_client.get(url, timeout=30)
            if response.status_code != 404:
                response.raise_for_status()
                data = response.json()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Transient errors are retried.  The caller can cancel.
            data = None

        if isinstance(data, dict):
            result_status = data.get("result_status")
            if result_status in ("success", "error"):
                return {
                    "local_dir": str(data.get("local_dir") or ""),
                    "status": result_status,
                    "error_msg": str(
                        data.get("result_error") or data.get("error") or ""
                    ),
                }

        await asyncio.sleep(delay)
        delay = min(delay * 1.5, max_delay)


async def poll_job_results(
    http_client: httpx.AsyncClient,
    job_manager_url: str,
    jobs: Dict[str, Any],
    *,
    poll_interval: float = 2.0,
    poll_interval_max: float = 10.0,
) -> Dict[str, Any]:
    """Build a resume map for ``{interrupt_id: JobSubmission value}``.

    Each value is wrapped in the ``{"payload": ...}`` shape that the StepNode
    interrupt handler expects.  A submission without a ``job_uuid`` is answered
    with an error instead of hanging forever.
    """
    resume_map: Dict[str, Any] = {}
    for interrupt_id, value in jobs.items():
        job_uuid = job_uuid_from_interrupt(value)
        if not job_uuid:
            resume_map[interrupt_id] = {
                "payload": {
                    "local_dir": "",
                    "status": "error",
                    "error_msg": (
                        "JobSubmission interrupt did not include a job_uuid."
                    ),
                }
            }
            continue
        result = await wait_for_job(
            http_client,
            job_manager_url,
            job_uuid,
            poll_interval=poll_interval,
            poll_interval_max=poll_interval_max,
        )
        resume_map[interrupt_id] = {"payload": result}
    return resume_map
