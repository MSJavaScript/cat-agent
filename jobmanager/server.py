import logging, logging.handlers, json, os, queue, dotenv, argparse, uvicorn
from jobmanager.models import JobRequest, AgentResumeInfo
from jobmanager.db import JobDB
from jobmanager.base import BaseProvider
from jobmanager.providers.bohr import BohrProvider
from jobmanager.providers.slurm import SlurmProvider
from jobmanager.providers.local import LocalProvider
 
from jobmanager.manager import JobManager
from jobmanager.agent.agent import JobManagerAgent

from typing import Dict, Any, Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger("jobmanager")


def setup_logging(log_path: str, level: str = "info") -> logging.handlers.QueueListener:
    """Configure the "jobmanager" logger to write to `log_path` without
    blocking the event loop. Records are handed to a QueueHandler and a
    background thread performs the actual file I/O."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    log = logging.getLogger("jobmanager")
    log.setLevel(numeric_level)
    log.propagate = False
    for handler in list(log.handlers):
        log.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    log_queue: queue.Queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(numeric_level)
    log.addHandler(queue_handler)

    listener = logging.handlers.QueueListener(log_queue, file_handler,
                                              respect_handler_level=True)
    listener.start()
    return listener


def create_llm_model(
        model_type:Literal["glm", "deepseek", "qwen"],
        model_name:str,
        base_url:str,
        api_key:str,
        **kwargs):
    
    if model_type == "glm":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, 
                 api_key=api_key, 
                 base_url=base_url,
                 **kwargs)
    elif model_type == "deepseek":
        from langchain_deepseek import ChatDeepSeek
        return ChatDeepSeek(model = model_name,
                            api_key = api_key,
                            base_url = base_url,
                            **kwargs) 
    else:
        from langchain_qwq import ChatQwen
        return ChatQwen(model=model_name,
                        base_url=base_url,
                        api_key=api_key, **kwargs)

 
# 把providers字段传递给这个函数。
def create_providers(setting:Dict[str, Any]) -> Dict[str, BaseProvider]:
    providers = dict()
    for key in setting.keys():
        obj = setting[key]
        if obj["type"] == "bohr":
            providers[key] = BohrProvider(name=key, 
                                          processing_units=obj["processing_units"],
                                          job_envs_setting=obj["job_envs"], 
                                          remote_files_setting=obj.get("remote_files", []),
                                          project_id=obj["project_id"],
                                          platform=obj["platform"], 
                                          job_type=obj["job_type"],
                                          resources_setting=obj["resources"])
        elif obj["type"] == "slurm":
            providers[key] = SlurmProvider(name=key, 
                                           processing_units=obj["processing_units"],
                                           job_envs_setting=obj["job_envs"], 
                                           remote_files_setting=obj.get("remote_files", []),
                                           host=obj["host"],
                                           user=obj["user"], 
                                           port=obj["port"], 
                                           password=obj["password"],
                                           remote_root=obj["remote_root"], 
                                           keyfile=obj["keyfile"], 
                                           resources_setting=obj["resources"] )
        elif obj["type"] == "local":
            providers[key] = LocalProvider(name=key, processing_units=obj["processing_units"],
                                           job_envs_setting=obj["job_envs"]) 
        else:
            raise ValueError(f"Unsupported provider type {obj["type"]}.")
    return providers


def build_app(setting_json_path:str) -> FastAPI:
    if not os.path.isfile(setting_json_path):
        raise FileNotFoundError(f"{setting_json_path} is not a file.")
    with open(setting_json_path, mode="r") as f:
        setting:dict = json.load(f)


    # db_path 可选：缺省时放在配置文件同目录下。
    db_path = setting.get("db_path") or os.path.join(
        os.path.dirname(os.path.abspath(setting_json_path)), "jobmanager.db")
    db = JobDB(db_path)
    providers = create_providers(setting["providers"])

    dotenv.load_dotenv()

    model_name = os.getenv("MODEL_NAME").lower()
    if "qwen" in model_name:
        model_type = "qwen"
    elif "glm" in model_name:
        model_type = "glm"
    elif "deepseek" in model_name:
        model_type = "deepseek"
    model = create_llm_model(model_type=model_type,
                             model_name=os.getenv("MODEL_NAME"),
                             base_url=os.getenv("BASE_URL"),
                             api_key=os.getenv("API_KEY"))
    obj = JobManagerAgent(providers)
    agent = obj.create(model=model, checkpointer=InMemorySaver()) 
    manager = JobManager(agent=agent,
                         db = db,
                         providers=providers,
                         callback_url=setting["callback_url"],
                         poll_interval=setting.get("poll_interval", 60),
                         retry_job_download=setting.get("retry_job_download", [5, 10, 20]),
                         retry_callback=setting.get("retry_callback", [5, 10, 20]),
                         n_retry_get_status=setting.get("n_retry_get_status", 3),
                         n_retry_job_submit=setting.get("n_retry_job_submit", 3),
                         max_finalize_retries=setting.get("max_finalize_retries", 5),
                         front_end_url=setting.get("front_end_url", None))
    
    @asynccontextmanager
    async def lifespan(app:FastAPI):
        await db.connect()
        await manager.start()
        yield 
        await manager.stop()

    app = FastAPI(title="Computation Job Manager", version="0.0.1", lifespan=lifespan)

    @app.get("/health")
    async def health() -> JSONResponse:
        # Some process/environment health checkers probe /health even when the
        # service does not need it.  Answering avoids noisy 404 lines in the
        # uvicorn log; it only reports that the HTTP process is alive.
        return JSONResponse({"status": "ok"})

    @app.post("/submit")
    async def submit(req:JobRequest) -> JSONResponse:
        response = await manager.submit_job(req)
        return JSONResponse(response)

    @app.post("/set-thread-id")
    async def set_thread_id(req:AgentResumeInfo) -> JSONResponse:
        await manager.set_thread_id(req.job_uuid, req.thread_id, req.interrupt_id)
        return JSONResponse({"status":"success", "message": "thread_id and interrupt_id is added into database." })
    
    @app.post("/cancel/{job_uuid}")
    async def cancel(job_uuid: str) -> JSONResponse:
        response = await manager.cancel_job(job_uuid)
        return JSONResponse(response)

    @app.get("/status/{job_uuid}")
    async def status(job_uuid:str) -> JSONResponse:
        record = await manager.get_status(job_uuid)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        return JSONResponse(record.model_dump(mode="json")) 

    @app.get("/jobs")
    async def list_jobs(limit:int = 200) -> JSONResponse:
        records = await manager.list_jobs(limit)
        return JSONResponse([r.model_dump(mode="json") for r in records])

    @app.get("/providers")
    async def get_providers_info() -> JSONResponse:
        info_list = []
        for key in providers.keys():
            info_list.append( providers[key].get_info())
        return JSONResponse(info_list)

    return app

def main() -> None:
    parser = argparse.ArgumentParser(description="Job manager fastapi backend.")
    parser.add_argument("--setting", type=str, required=True, help="Path to the setting json file.")
    parser.add_argument("--host", type=str, required=True, help="Host of app.")
    parser.add_argument("--port", type=int, required=True, help="Port of app.")
    parser.add_argument("--log", type=str, required=True, help="Path to log file.")
    parser.add_argument("--level", type=str, default="info", choices=["info", "debug"], help="Log level.")
    args = parser.parse_args()

    listener = setup_logging(args.log, args.level)
    try:
        app = build_app(args.setting)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.level)
    finally:
        listener.stop()

if __name__ == "__main__":
    main()





    
    
