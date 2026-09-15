import logging, json, httpx, asyncio, os, glob, contextlib
from jobmanager.models import JobRequest, JobStatus, JobResult, JobSetting, SubJobRecord, _now
from jobmanager.db import JobDB, JobRecord
from jobmanager.base import BaseProvider
from jobmanager.agent.agent import run_job_agent

from typing import Dict, Literal, Optional, List, Awaitable, Any 
from pydantic import BaseModel

logger = logging.getLogger("jobmanager")


class SimpleResponse(BaseModel):
    status: Literal["error", "success"] = "success"
    message: str = ""

def read_json(path:str) -> dict:
    try:
        with open(path, mode="r") as f:
            obj = json.load(f)
            return obj 
    except Exception as e:
        return dict()

class JobManager:
    def __init__(self,
                 agent,
                 db:JobDB,
                 providers:Dict[str, BaseProvider],
                 callback_url:str,

                 poll_interval:int = 60,
                 retry_job_download: List[int] = [5, 10, 20],
                 retry_callback:List[int] = [5, 10, 20],
                 n_retry_get_status:int = 3,
                 n_retry_job_submit:int = 3,
                 front_end_url:str = None):

        self.agent = agent
        self.db = db 
        self.callback_url = callback_url
        self.poll_interval = poll_interval
        self.retry_job_download = retry_job_download + [0]
        self.retry_callback = retry_callback + [0]
        self.n_retry_get_status = n_retry_get_status
        self.n_retry_job_submit = n_retry_job_submit
        self.polling_task = None 
        self.providers = providers
        self.front_end_url = front_end_url
        self.client = httpx.AsyncClient(timeout=30)
        self.stopping = False 
        # 后台任务（finalize、预处理的提交）统一登记，便于在关闭时回收并记录异常。
        self._tasks: set[asyncio.Task] = set()
        # 正在预处理/提交中的job_uuid，防止并发的重复提交请求。
        self._inflight: set[str] = set()

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        """把协程注册为后台任务，异常不会被静默丢弃。"""
        task = asyncio.create_task(coro)
        self._tasks.add(task)

        def _on_done(t: asyncio.Task):
            self._tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(f"Background task failed: {exc}", exc_info=exc)

        task.add_done_callback(_on_done)
        return task

    async def submit_job(self, req:JobRequest):
        """接收提交请求。预处理的耗时操作放到后台任务，接口立即返回。"""
        if req.provider_id not in self.providers:
            msg = f"Unknown provider_id {req.provider_id}."
            logger.warning(msg)
            return SimpleResponse(status="error", message=msg).model_dump(mode="python")

        if req.job_uuid in self._inflight:
            msg = f"Duplicate job. job={req.job_uuid} is being submitted."
            logger.info(msg)
            return SimpleResponse(status="error", message=msg).model_dump(mode="python")
        self._inflight.add(req.job_uuid)

        existing = await self.db.get(req.job_uuid)
        if existing:
            if existing.status != JobStatus.FAILED:
                self._inflight.discard(req.job_uuid)
                msg = f"Duplicate job. job={req.job_uuid} status={existing.status}."
                logger.info(msg)
                return SimpleResponse(status="error", message=msg).model_dump(mode="python")
            # 失败重试：清除旧的子作业记录后重新提交
            await self.db.delete_sub_jobs(req.job_uuid)

        # 先写入占位记录，保证 /status 可见，并防止并发的重复提交。
        placeholder = JobRecord(job_uuid = req.job_uuid,
                                description = req.description,
                                provider_id = req.provider_id,
                                job_env_name = req.job_env_name,
                                local_dir = req.local_dir,
                                result_patterns = req.result_patterns,
                                status = JobStatus.SUBMITTED,
                                created_at = (existing.created_at if existing else _now()),
                                updated_at = _now(),
                                callback_pending = False,
                                attempt = (existing.attempt if existing else 0),
                                error = "",
                                sub_jobs = [])
        await self.db.save(placeholder)

        self._spawn(self._submit_job_impl(req))
        msg = f"Accepted job. job_uuid = {req.job_uuid}, local_dir = {req.local_dir}. "
        logger.info(msg)
        return SimpleResponse(status="success", message=msg).model_dump(mode="python")

    async def _submit_job_impl(self, req:JobRequest):
        """执行预处理并把各分组提交到远端。运行在后台任务中。"""
        attempt = 0
        created_at = _now()
        try:
            existing = await self.db.get(req.job_uuid)
            attempt = existing.attempt if existing else 0
            created_at = existing.created_at if existing else _now()
            response = await run_job_agent(agent=self.agent,
                                           phase="preprocessing",
                                           thread_id="temp-123",
                                           local_dir=req.local_dir,
                                           log_msg=True,
                                           provider_id=req.provider_id,
                                           job_env_name=req.job_env_name,
                                           http_client=self.client,
                                           url=self.front_end_url)

            preprocessing_output = response.get("preprocessing_output", None)
            if not preprocessing_output:
                raise RuntimeError(f"Agent can not preprocess the job {req}.")

            provider = self.providers[req.provider_id]
            remote_root = getattr(provider, "remote_root", None)

            # 确定作业的分组。批量作业会被拆分成若干组，每组单独提交到远端，
            # 对应一个remote_id和remote_dir。单个作业视为只有一个组的特例。
            if preprocessing_output["batch_job"]:
                wildcard = preprocessing_output.get("group_dirs_wildcard") or "*"
                group_local_dirs = sorted(
                    d for d in glob.glob(os.path.join(req.local_dir, wildcard))
                    if os.path.isdir(d))
                if not group_local_dirs:
                    raise RuntimeError(f"no group dirs match '{wildcard}' in {req.local_dir}.")
                n_groups = preprocessing_output.get("n_groups")
                if n_groups and n_groups != len(group_local_dirs):
                    logger.warning(f"Job {req.job_uuid}: agent reported n_groups={n_groups}, "
                                   f"but found {len(group_local_dirs)} group dirs.")
            else:
                group_local_dirs = [req.local_dir]

            sub_jobs = []
            for group_index, group_local_dir in enumerate(group_local_dirs):
                # 单个作业时remote_dir为remote_root/job_uuid，
                # 批量作业时每组为remote_root/job_uuid/组目录名
                if remote_root:
                    if group_local_dir == req.local_dir:
                        remote_dir = os.path.join(remote_root, req.job_uuid)
                    else:
                        remote_dir = os.path.join(remote_root, req.job_uuid, os.path.basename(group_local_dir))
                else:
                    remote_dir = None

                remote_id = await self._submit_group(req, provider, group_local_dir, remote_dir)

                sub_jobs.append(SubJobRecord(job_uuid=req.job_uuid,
                                             group_index=group_index,
                                             local_dir=group_local_dir,
                                             remote_id=remote_id,
                                             remote_dir=remote_dir,
                                             status=JobStatus.SUBMITTED,
                                             created_at=_now(),
                                             updated_at=_now()))

            rec = JobRecord(job_uuid = req.job_uuid,
                            description = req.description,
                            provider_id = req.provider_id,
                            job_env_name = req.job_env_name,
                            local_dir = req.local_dir,
                            result_patterns = req.result_patterns,
                            status = JobStatus.SUBMITTED,
                            created_at = created_at,
                            updated_at = _now(),
                            callback_pending = False,
                            attempt = attempt,
                            error = "",
                            sub_jobs = sub_jobs)

            await self.db.save(rec)
            logger.info(f"Submitted job. job_uuid = {req.job_uuid}, local_dir = {req.local_dir}, "
                        f"n_groups = {len(sub_jobs)}. ")
        except Exception as e:
            logger.error(f"Failed to submit job {req.job_uuid}: {e}", exc_info=True)
            failed = JobRecord(job_uuid = req.job_uuid,
                               description = req.description,
                               provider_id = req.provider_id,
                               job_env_name = req.job_env_name,
                               local_dir = req.local_dir,
                               result_patterns = req.result_patterns,
                               status = JobStatus.FAILED,
                               created_at = created_at,
                               updated_at = _now(),
                               callback_pending = False,
                               attempt = attempt,
                               error = str(e),
                               sub_jobs = [])
            await self.db.save(failed)
            await self._post_callback(job_uuid=req.job_uuid,
                                      local_dir=req.local_dir,
                                      status="error",
                                      error_msg=f"Failed to submit job: {e}")
        finally:
            self._inflight.discard(req.job_uuid)


    async def _submit_group(self, req:JobRequest, provider:BaseProvider,
                            group_local_dir:str, remote_dir:Optional[str]) -> str:
        """提交一个组。组目录下可以有自己的job.json，否则使用作业根目录的job.json。"""
        meta_data = await asyncio.to_thread(read_json, os.path.join(group_local_dir, "job.json"))
        if not meta_data:
            meta_data = await asyncio.to_thread(read_json, os.path.join(req.local_dir, "job.json"))
        return await provider.submit(JobSetting(job_name=meta_data.get("job_name", req.job_uuid),
                                        local_dir=group_local_dir,
                                        remote_dir=remote_dir,
                                        result_patterns=req.result_patterns,
                                        provider_name=req.provider_id,
                                        job_env_name=req.job_env_name,
                                        access_remote_dirs=meta_data.get("access_remote_dirs"),
                                        resource_name=meta_data.get("resource_name")))


    async def cancel_job(self, job_uuid:str):
        # 这里还要考虑作业不存在，即已经被取消或已完成的情况
        rec = await self.db.get(job_uuid)
        if rec is None:
            msg = f"{job_uuid} does not exist."
            logger.warning(msg)
            return SimpleResponse(status="error", message=msg).model_dump(mode="python") 

        # 批量作业的每一个子作业（组）都要取消
        for sub in rec.sub_jobs:
            if sub.remote_id is None:
                continue
            try:
                await self.providers[rec.provider_id].cancel(sub.remote_id)
            except Exception as e:
                msg = f"Cannot cancel sub job. job_uuid={job_uuid}, group_index={sub.group_index}, {e}. "
                logger.warning(msg)
                return SimpleResponse(status="error", message=msg).model_dump(mode="python") 

        await self.db.set_error(job_uuid, JobStatus.CANCELLED, "cancelled by request", rec.attempt)
        msg = f"{job_uuid} is cancelled by request."
        logger.info(msg)
        return SimpleResponse(status="success", message=msg).model_dump(mode="python") 
        
    async def get_status(self, job_uuid: str):
        return await self.db.get(job_uuid)

    async def list_jobs(self, limit: int = 200) -> list[JobRecord]:
        return await self.db.list_all(limit)


    async def start(self):
        self.polling_task = asyncio.create_task(self._polling_loop())
        # 启动时补发因崩溃/重启而未送达的callback。
        self._spawn(self._boot_sweep_callbacks())

    async def stop(self):
        self.stopping = True
        if self.polling_task is not None:
            self.polling_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.polling_task
            self.polling_task = None
        # 等待仍在执行的后台任务结束，避免它们继续使用即将关闭的DB和HTTP客户端。
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self.client.aclose()
        await self.db.close()

    @staticmethod
    def _terminal_result(rec: JobRecord) -> tuple[Literal["error", "success"], str]:
        """根据作业记录的最终状态，推断补发callback时应携带的结果。"""
        if rec.status == JobStatus.COMPLETED:
            return "success", ""
        if rec.status == JobStatus.CANCELLED:
            return "error", "Job is cancelled."
        if rec.status == JobStatus.NONE:
            return "error", "Failed to get the status of job"
        return "error", rec.error or f"Job ended with status {rec.status.value}."

    async def _boot_sweep_callbacks(self):
        """把重启前未成功送达的callback重新POST一次，实现at-least-once投递。"""
        try:
            records = await self.db.list_callback_pending()
        except Exception as e:
            logger.error(f"Failed to list pending callbacks: {e}", exc_info=True)
            return
        if not records:
            return
        logger.info(f"Re-posting {len(records)} pending callback(s) from previous run.")
        for rec in records:
            status, error_msg = self._terminal_result(rec)
            await self._post_callback(job_uuid=rec.job_uuid,
                                      local_dir=rec.local_dir,
                                      status=status,
                                      error_msg=error_msg)


    async def _post_callback(self, job_uuid:str, local_dir:str, status:Literal["error", "success"], error_msg:str):
        job_result = JobResult(job_uuid=job_uuid,
                                local_dir=local_dir,
                                status=status,
                                error_msg=error_msg) 
        body = job_result.model_dump_json()
        for delay in self.retry_callback:
            try:
                response = await self.client.post(self.callback_url, content=body, headers={"Content-Type": "application/json"})
                if 200 <= response.status_code < 300 or response.status_code == 404:
                    await self.db.set_callback_pending(job_uuid, False)
                    logger.info(f"callback ok job_uuid = {job_uuid}.")
                    return
                logger.warning(f"callback job_uuid={job_uuid} http={response.status_code}, retrying.")
            except Exception as e:
                logger.warning(f"callback job_uuid={job_uuid} post failed:{e}.")
            await asyncio.sleep(delay)
        logger.error(f"callback job_uuid={job_uuid} exhausted retries (left pending)")


    async def _finalize_job(self, job_record:JobRecord):
        job_status = job_record.status

        # 进入终结流程后先标记callback待发，这样即使下载或检查过程中崩溃，
        # 重启时的boot sweep也能补发callback。
        await self.db.set_callback_pending(job_record.job_uuid, True)

        if job_status == JobStatus.CANCELLED:
            # 被cancel的任务直接返回。
            await self._post_callback(job_uuid=job_record.job_uuid,
                                      local_dir=job_record.local_dir,
                                      status="error",
                                      error_msg=f"Job is cancelled.")
            return 
        if job_status == JobStatus.NONE:
            # 持续多次查询任务状态函数都返回错误，这种任务直接报错返回。
            await self._post_callback(job_uuid=job_record.job_uuid,
                                      local_dir=job_record.local_dir,
                                      status="error",
                                      error_msg=f"Failed to get the status of job")
            return 
        
        # 剩下的状态是COMPLETED, FAILED, UNKNOWN
        # 现在尝试下载任务。批量作业需要下载每一个子作业（组）的结果。
        provider_id = job_record.provider_id
        provider = self.providers[provider_id]
        failed_groups = []
        for sub in job_record.sub_jobs:
            for delay in self.retry_job_download:
                try:
                    flag = await provider.download(sub.remote_id, sub.remote_dir, sub.local_dir)
                except Exception as e:
                    logger.warning(f"Failed to download sub job. "
                                   f"job_uuid={job_record.job_uuid}, "
                                   f"group_index={sub.group_index}, {e}.")
                    flag = False
                if flag:
                    break 
                await asyncio.sleep(delay)
            else:
                failed_groups.append(sub.group_index)

        if failed_groups:
            # 任务结果下载失败
            await self._post_callback(job_uuid=job_record.job_uuid,
                                      local_dir=job_record.local_dir,
                                      status="error",
                                      error_msg="Only part of job results are downloaded, the remaining part are lost.")
            return 

        # 这里使用agent 检查一下是否有任务提交上的错误。
        response = await run_job_agent(agent=self.agent,
                                       phase="checking",
                                       thread_id="temp-123",
                                       local_dir=job_record.local_dir,
                                       log_msg=True,
                                       provider_id=job_record.provider_id,
                                       job_env_name=job_record.job_env_name,
                                       http_client=self.client,
                                       url=self.front_end_url)
        
        checking_output = response.get("checking_output", None)
        if (checking_output is None) or (checking_output.get("job_result_status") is None):
            await self._post_callback(job_uuid=job_record.job_uuid,
                                    local_dir=job_record.local_dir,
                                    status="success",
                                    error_msg="Failed to analyze the status of job results, take it to be successful by default.")
            return 


        # 成功或输入文件错误则直接返回
        if checking_output.get("job_result_status") == "Success":
            await self._post_callback(job_uuid=job_record.job_uuid,
                                    local_dir=job_record.local_dir,
                                    status="success",
                                    error_msg="")
            return 
        elif checking_output.get("job_result_status") == "InputError":
            await self._post_callback(job_uuid=job_record.job_uuid,
                                    local_dir=job_record.local_dir,
                                    status="error",
                                    error_msg=checking_output.get("error_message"))
            return 


        error_msg = (job_record.error or "") + f'\n Retry {job_record.attempt}:' + checking_output.get("error_message")
        await self.db.set_error(job_uuid=job_record.job_uuid,
                                status=JobStatus.FAILED,
                                error=error_msg,
                                attempt=job_record.attempt+1)
        
        if job_record.attempt >= self.n_retry_job_submit:
            await self._post_callback(job_uuid=job_record.job_uuid,
                                    local_dir=job_record.local_dir,
                                    status="error",
                                    error_msg=f"Job failed {job_record.attempt} times. Stop retrying. Below is the error messages\n\n {error_msg}")
            await self.db.set_callback_pending(job_record.job_uuid, False)
            return 

        # 没有超过最大次数则重试：删除旧的子作业记录后重新提交
        await self.db.delete_sub_jobs(job_record.job_uuid)
        request = JobRequest(job_uuid=job_record.job_uuid,
                            description=job_record.description,
                            local_dir=job_record.local_dir,
                            provider_id=job_record.provider_id,
                            job_env_name=job_record.job_env_name,
                            result_patterns=job_record.result_patterns)
        await self._submit_job_impl(request)


    async def _polling_loop(self):
        while not self.stopping:
            for provider_id in list(self.providers.keys()):
                # 单个provider的失败不能终止整个轮询循环，否则所有作业都不再被查询。
                try:
                    await self._poll_provider(provider_id)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Polling provider {provider_id} failed: {e}", exc_info=True)
            await asyncio.sleep(self.poll_interval)

    async def _poll_provider(self, provider_id: str):
        # 列出数据库里状态是running或submitted的子作业（组）
        pairs = await self.db.list_active_sub_jobs(provider_id)

        # remote_id -> (作业, 子作业)
        pairs_dict = { sub.remote_id: (job, sub) for job, sub in pairs
                       if sub.remote_id is not None }
        remote_ids = set(pairs_dict.keys())
        if not remote_ids:
            return

        # 这里的states也是以remote_id为键的。
        remote_job_states = dict()
        for _ in range(self.n_retry_get_status):
            states = await self.providers[provider_id].get_job_states(remote_ids)
            remote_job_states.update(states)
            for key in states.keys():
                if states[key] != JobStatus.NONE:
                    remote_ids.discard(key)
            if len(remote_ids) == 0:
                break 
        else:
            logger.error(f"Failed to get the status of job {remote_ids} for {self.n_retry_get_status} times.")

        # 在这里更新数据库中子作业的状态。
        # update_sub_job_status内部会重新聚合作业（jobs表）的总状态。
        for remote_id, state in remote_job_states.items():
            await self.db.update_sub_job_status(remote_id, state)

        # 在这里分发任务：只有当一个作业的所有子作业都到达终止状态时
        # 才finalize，保证下载和callback对整个作业只做一次。
        finalized_jobs = set()
        for remote_id, state in remote_job_states.items():
            if state == JobStatus.SUBMITTED or state == JobStatus.RUNNING:
                continue
            job, sub = pairs_dict[remote_id]
            if job.job_uuid in finalized_jobs:
                continue
            # 重新读取作业，拿到聚合后的最新状态和全部子作业
            fresh = await self.db.get(job.job_uuid)
            if fresh is None:
                continue
            if fresh.status == JobStatus.SUBMITTED or fresh.status == JobStatus.RUNNING:
                continue
            finalized_jobs.add(job.job_uuid)
            self._spawn(self._finalize_job(fresh))
