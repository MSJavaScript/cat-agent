# learn

Small, self-contained runnable explanations of mechanisms used in this repo.
Nothing here is imported by the product code; the demos are meant to be read
and executed.

| demo | topic |
| --- | --- |
| [`event_log_demo/`](event_log_demo/) | why `catagent/app/events.py` exists: server-side event recording, polling with sequence cursors, out-of-band jobmanager callbacks, restart detection |

```bash
cd learn/event_log_demo
uvicorn app:app --reload --port 8010
```
