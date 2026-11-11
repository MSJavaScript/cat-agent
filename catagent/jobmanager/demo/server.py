"""Demo job manager server.

Same HTTP surface as `catagent.jobmanager.server`, but the LLM agent is replaced by the
deterministic `stub_run_job_agent`, so the demo runs offline. Uses the `local`
provider, i.e. jobs are executed as local `bash run.sh` processes.

Run it in its own terminal:

    python -m catagent.jobmanager.demo.server --config catagent/jobmanager/demo/config.json --port 8000
"""

import argparse
import json
import logging
import os
import sys
from contextlib import asynccontextmanager

# Make `jobmanager` and this demo package importable when run as a plain script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_HERE, _REPO_PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from catagent.jobmanager.db import JobDB
from catagent.jobmanager.manager import JobManager
from catagent.jobmanager.models import JobRequest
from catagent.jobmanager.server import build_app, create_providers, setup_logging
from catagent.jobmanager.demo.stub_agent import stub_run_job_agent


def build_demo_app(config_path: str, fresh: bool = False) -> FastAPI:
    with open(config_path) as f:
        setting = json.load(f)

    cfg_dir = os.path.dirname(os.path.abspath(config_path))
    db_path = setting.get("db_path") or "jobmanager.db"
    if not os.path.isabs(db_path):
        db_path = os.path.join(cfg_dir, db_path)
    if fresh and os.path.exists(db_path):
        os.remove(db_path)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)

    providers = create_providers(setting["providers"])
    db = JobDB(db_path)

    # Replace the LLM agent with the deterministic stub for the demo.
    import catagent.jobmanager.manager as manager_module
    manager_module.run_job_agent = stub_run_job_agent

    manager = JobManager(agent=None,
                         db=db,
                         providers=providers,
                         callback_url=setting["callback_url"],
                         poll_interval=setting.get("poll_interval", 3),
                         retry_job_download=setting.get("retry_job_download", [2, 4]),
                         retry_callback=setting.get("retry_callback", [2, 4]),
                         n_retry_get_status=setting.get("n_retry_get_status", 3),
                         n_retry_job_submit=setting.get("n_retry_job_submit", 1),
                         front_end_url=setting.get("front_end_url"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await db.connect()
        await manager.start()
        try:
            yield
        finally:
            await manager.stop()

    app = FastAPI(title="Job Manager DEMO", version="0.0.1", lifespan=lifespan)

    @app.post("/submit")
    async def submit(req: JobRequest) -> JSONResponse:
        return JSONResponse(await manager.submit_job(req))

    @app.post("/cancel/{job_uuid}")
    async def cancel(job_uuid: str) -> JSONResponse:
        return JSONResponse(await manager.cancel_job(job_uuid))

    @app.get("/status/{job_uuid}")
    async def status(job_uuid: str) -> JSONResponse:
        record = await manager.get_status(job_uuid)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        return JSONResponse(record.model_dump(mode="json"))

    @app.get("/jobs")
    async def list_jobs(limit: int = 200) -> JSONResponse:
        records = await manager.list_jobs(limit)
        return JSONResponse([r.model_dump(mode="json") for r in records])

    @app.get("/providers")
    async def get_providers_info() -> JSONResponse:
        info = {key: providers[key].get_info() for key in providers}
        return JSONResponse(info)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo job manager server (stub agent + local provider).")
    parser.add_argument("--config", default=os.path.join(_HERE, "config.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log", default=os.path.join(_HERE, "demo-server.log"))
    parser.add_argument("--fresh", action="store_true", help="Delete the demo database before starting.")
    parser.add_argument("--real-agent", action="store_true",
                        help="Use the production build_app: real LLM agent from .env and the "
                             "real providers from the config. Without this flag the deterministic "
                             "stub agent is used.")
    args = parser.parse_args()

    # Non-blocking file logging plus a console copy so you can watch the manager work.
    listener = setup_logging(args.log, "info")
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s [jobmanager] %(levelname)s %(message)s",
                                           "%H:%M:%S"))
    logging.getLogger("jobmanager").addHandler(console)
    try:
        if args.real_agent:
            if args.fresh:
                print("[JOBMANAGER] --fresh is ignored with --real-agent.", flush=True)
            app = build_app(args.config)
            print(f"[JOBMANAGER] real-agent mode: LLM from .env, providers from {args.config}", flush=True)
        else:
            app = build_demo_app(args.config, fresh=args.fresh)
        print(f"[JOBMANAGER] listening on http://{args.host}:{args.port}  (log: {args.log})", flush=True)
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    finally:
        listener.stop()


if __name__ == "__main__":
    main()