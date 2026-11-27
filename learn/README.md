# learn

Small, self-contained runnable explanations of mechanisms used in this repo.
Nothing here is imported by the product code; the demos are meant to be read
and executed.

| demo | topic |
| --- | --- |
| [`event_log_demo/`](event_log_demo/) | why `catagent/app/events.py` exists: server-side event recording, polling with sequence cursors, out-of-band jobmanager callbacks, restart detection |
| [`job_resume_demo/`](job_resume_demo/) | replacing the jobmanager → agent HTTP callback with an in-process callback or event bus: what gets simpler, what must stay durable |
| [`multi_interrupt_demo/`](multi_interrupt_demo/) | several pending interrupts on one graph: one `ainvoke()` returns all of them, a partial resume keeps the sibling pending, and why the outer graph must merge all answers before resuming the inner StepNode/orchestrator once |

```bash
cd learn/event_log_demo
uvicorn app:app --reload --port 8010

cd learn/job_resume_demo
python demo.py
python test_demo.py
```
