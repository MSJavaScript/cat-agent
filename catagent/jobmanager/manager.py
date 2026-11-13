import logging, json, httpx, asyncio, os, glob, contextlib
from catagent.jobmanager.models import (
    JobRequest, JobStatus, JobResult, JobSetting, SubJobRecord, _now,
    SubmissionState, FinalizeState,
)
from catagent.jobmanager.db import JobDB, JobRecord
from catagent.jobmanager.base import BaseProvider
from catagent.jobmanager.agent.agent import run_job_agent

from typing import Dict, Literal, Optional, List, Awaitable, Any 
from pydantic import BaseModel
import uuid 

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
                 max_finalize_retries:int = 5,
                 front_end_url:str = None):

        self.agent = agent 
        self.db = db 
        self.callback_url = callback_url
        self.poll_interval = poll_interval
        self.retry_job_download = retry_job_download + [0]
        self.retry_callback = retry_callback + [0]
        self.n_retry_get_status = n_retry_get_status
        self.n_retry_job_submit = n_retry_job_submit
        # A download can fail for reasons unrelated to the calculation (SSH,
        # SCP, a partially written archive).  Such a failure must not end the
        # job: retry it on later polling cycles, up to this many times.
        self.max_finalize_retries = max_finalize_retries
        self.polling_task = None 
        self.providers = providers
        self.front_end_url = front_end_url
        self.client = httpx.AsyncClient(timeout=30)
        self.stopping = False 
        # 后台任务（finalize、预处理的提交）统一登记，便于在关闭时回收并记录异常。
        # 一个对象可以放在set中必须是可哈希的，需要重载__hash__和__eq__
        self._tasks: set[asyncio.Task] = set()
        # 正在预处理/提交中的job_uuid，防止并发的重复提交请求。
        self._inflight: set[str] = set()
        # 正在终结中的job_uuid，防止同一个作业被并发地下载/检查。
        self._finalizing: set[str] = set()
        # 正在投递callback的job_uuid，防止周期性补投与首次投递并发。
        self._delivering: set[str] = set()

    def _spawn(self, coro: Awaitable[Any]) -> asyncio.Task:
        """把协程注册为后台任务，异常不会被静默丢弃。"""
        # 当使用create_task 之后，coro （一个async函数的调用）就会开始执行
        task = asyncio.create_task(coro)
        self._tasks.add(task)

        def _on_done(t: asyncio.Task):
            self._tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(f"Background task failed: {exc}", exc_info=exc)
        
        # 执行结束后调用回调函数
        task.add_done_callback(_on_done)
        return task

    async def submit_job(self, req:JobRequest):
        """接收提交请求。预处理的耗时操作放到后台任务，接口立即返回。

        The call is idempotent on `job_uuid`: a retry of an in-flight or
        half-submitted job resumes it instead of creating a second remote job.
        """
        if req.provider_id not in self.providers:
            msg = f"Unknown provider_id {req.provider_id}."
            logger.warning(msg)
            return SimpleResponse(status="error", message=msg).model_dump(mode="python")

        existing = await self.db.get(req.job_uuid)
        if existing is not None:
            if existing.status in (JobStatus.COMPLETED,
                                   JobStatus.CANCELLED,
                                   JobStatus.UNKNOWN,
                                   JobStatus.NONE):
                msg = f"Duplicate job. job={req.job_uuid} status={existing.status.value}."
                logger.info(msg)
                return SimpleResponse(status="error", message=msg).model_dump(mode="python")

            if existing.status in (JobStatus.SUBMITTED, JobStatus.RUNNING):
                if existing.submission_state != SubmissionState.SUBMITTED \
                        or any(sub.remote_id is None for sub in existing.sub_jobs):
                    # Accepted previously but not fully submitted (e.g. after a
                    # restart).  Re-drive the same plan instead of duplicating.
                    self._spawn(self._run_submission(req))
                msg = f"Job {req.job_uuid} was already accepted."
                logger.info(msg)
                return SimpleResponse(status="success", message=msg).model_dump(mode="python")

            # FAILED: allow an explicit retry with a clean plan.  Bump the
            # attempt so the new run gets a fresh idempotency key: a crash
            # retry must reuse the key, an explicit retry must not.
            await self.db.delete_sub_jobs(req.job_uuid)
            await self.db.reset_finalize(req.job_uuid)
            next_attempt = existing.attempt + 1
        else:
            next_attempt = 0

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
                                attempt = next_attempt,
                                error = "",
                                submission_state = SubmissionState.PLANNING,
                                finalize_state = FinalizeState.NONE,
                                result_status = None,
                                result_error = None,
                                sub_jobs = [])
        await self.db.save(placeholder)

        self._spawn(self._run_submission(req))
        msg = f"Accepted job. job_uuid = {req.job_uuid}, local_dir = {req.local_dir}. "
        logger.info(msg)
        return SimpleResponse(status="success", message=msg).model_dump(mode="python")

    async def _run_submission(self, req:JobRequest) -> None:
        """Run one submit attempt, guarded against concurrent duplicates."""
        if req.job_uuid in self._inflight:
            logger.info(f"Submission of {req.job_uuid} is already in flight.")
            return
        self._inflight.add(req.job_uuid)
        try:
            await self._submit_job_impl(req)
        finally:
            self._inflight.discard(req.job_uuid)

    async def _submit_job_impl(self, req:JobRequest):
        """执行预处理并把各分组提交到远端。运行在后台任务中。

        Crash safety: the group plan is written before the first provider
        call, and each remote id is persisted immediately.  A retry therefore
        skips groups that were already submitted and only starts the missing
        ones.
        """
        record = await self.db.get(req.job_uuid)
        attempt = record.attempt if record else 0
        # created_at = record.created_at if record else _now()
        try:
            provider = self.providers[req.provider_id]
            remote_root = getattr(provider, "remote_root", None)

            sub_jobs = list(record.sub_jobs) if record else []
            if not sub_jobs:
                for _ in range(3): # 重试三次
                    logger.info(f"Call job agent with preprocessing phase. local_dir: {req.local_dir}.")
                    try:
                        response = await run_job_agent(agent=self.agent,
                                                    phase="preprocessing",
                                                    thread_id=f"thread-{uuid.uuid4()}",
                                                    local_dir=req.local_dir,
                                                    log_msg=True,
                                                    provider_id=req.provider_id,
                                                    job_env_name=req.job_env_name,
                                                    http_client=self.client,
                                                    url=self.front_end_url)
                        if response.get("preprocessing_output"):
                            break 
                    except Exception as e:
                        pass 
                else:
                    raise RuntimeError(f"Agent can not preprocess the job {req}.")

                # 确定作业的分组。批量作业会被拆分成若干组，每组单独提交到远端，
                # 对应一个remote_id和remote_dir。单个作业视为只有一个组的特例。
                preprocessing_output = response.get("preprocessing_output")
                if preprocessing_output["batch_job"]:
                    wildcard = preprocessing_output.get("group_dirs") or ['*']
                    group_local_dirs = []
                    for item in wildcard:
                        if '*' in item:
                            names = [name for name in glob.glob(os.path.join(req.local_dir, item)) if os.path.isdir(name)]
                            if len(names) == 0:
                                raise RuntimeError(f"no group dirs match '{item}' in {req.local_dir}.")    
                            group_local_dirs.extend(names) 
                        else:
                            p = os.path.join(req.local_dir, item)
                            if not os.path.isdir(p):
                                raise NotADirectoryError(f"{p} is not a directory.")
                            group_local_dirs.append(p)
                    
                    group_local_dirs = sorted(group_local_dirs)
                    n_groups = preprocessing_output.get("n_groups")
                    if n_groups and n_groups != len(group_local_dirs):
                        logger.warning(f"Job {req.job_uuid}: agent reported n_groups={n_groups}, "
                                       f"but found {len(group_local_dirs)} group dirs.")
                else:
                    group_local_dirs = [req.local_dir]

                for group_index, group_local_dir in enumerate(group_local_dirs):
                    # 单个作业时remote_dir为remote_root/job_uuid，
                    # 批量作业时每组为remote_root/job_uuid/组目录名
                    if remote_root:
                        if os.path.realpath(group_local_dir) == os.path.realpath(req.local_dir):
                            remote_dir = os.path.join(remote_root, req.job_uuid)
                        else:
                            remote_dir = os.path.join( remote_root, req.job_uuid, os.path.basename(group_local_dir))
                    else:
                        remote_dir = None
                    sub_jobs.append(SubJobRecord(
                        job_uuid=req.job_uuid,
                        group_index=group_index,
                        local_dir=group_local_dir,
                        remote_id=None,
                        remote_dir=remote_dir,
                        status=JobStatus.SUBMITTED,
                        created_at=_now(),
                        updated_at=_now()))

                # Persist the whole plan before submitting anything.  From this
                # point on the submit phase is resumable.
                await self.db.save_plan(req.job_uuid, sub_jobs)

            await self.db.set_submission_state(req.job_uuid, SubmissionState.SUBMITTING)

            for sub in sub_jobs:
                if sub.remote_id is not None:
                    continue  # already submitted before the crash/retry
                remote_id = await self._submit_group(
                    req, provider, sub.local_dir, sub.remote_dir,
                    sub.group_index, attempt)
                sub.remote_id = remote_id
                sub.status = JobStatus.SUBMITTED
                sub.updated_at = _now()
                await self.db.set_sub_job_remote(
                    req.job_uuid, sub.group_index, remote_id, sub.remote_dir,
                    JobStatus.SUBMITTED)

            await self.db.set_submission_state(req.job_uuid, SubmissionState.SUBMITTED)
            fresh = await self.db.get(req.job_uuid)
            if fresh is not None and fresh.status not in (JobStatus.SUBMITTED,
                                                          JobStatus.RUNNING):
                await self.db.update_status(req.job_uuid, JobStatus.SUBMITTED)
            logger.info(f"Submitted job. job_uuid = {req.job_uuid}, local_dir = {req.local_dir}, "
                        f"n_groups = {len(sub_jobs)}. ")
        except Exception as e:
            logger.error(f"Failed to submit job {req.job_uuid}: {e}", exc_info=True)
            await self.db.set_error(req.job_uuid, JobStatus.FAILED, str(e), attempt)
            await self.db.set_submission_state(req.job_uuid, SubmissionState.FAILED)
            await self.db.set_result_ready(
                req.job_uuid, "error", f"Failed to submit job: {e}")
            await self._deliver_callback(req.job_uuid)

    async def _submit_group(self, req:JobRequest, provider:BaseProvider,
                            group_local_dir:str, remote_dir:Optional[str],
                            group_index:int, attempt:int) -> str:
        """提交一个组。组目录下可以有自己的job.json，否则使用作业根目录的job.json。"""
        meta_data = await asyncio.to_thread(read_json, os.path.join(group_local_dir, "job.json"))
        if not meta_data:
            meta_data = await asyncio.to_thread(read_json, os.path.join(req.local_dir, "job.json"))
        # Stable per-attempt key: a provider that implements idempotent submit
        # can return the existing remote job after a crash, while an explicit
        # retry (attempt + 1) is allowed to start a fresh run.
        idempotency_key = f"{req.job_uuid}:g{group_index}:a{attempt}"
        return await provider.submit(JobSetting(
            job_name=meta_data.get("job_name", idempotency_key),
            local_dir=group_local_dir,
            remote_dir=remote_dir,
            result_patterns=req.result_patterns,
            provider_name=req.provider_id,
            job_env_name=req.job_env_name,
            access_remote_dirs=meta_data.get("access_remote_dirs"),
            resource_name=meta_data.get("resource_name"),
            idempotency_key=idempotency_key))


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

    async def set_thread_id(self, job_uuid: str, thread_id:str, interrupt_id:str):
        await self.db.set_thread_id(job_uuid, thread_id, interrupt_id)

    async def list_jobs(self, limit: int = 200) -> list[JobRecord]:
        return await self.db.list_all(limit)


    async def start(self):
        self.polling_task = asyncio.create_task(self._polling_loop())
        # 启动时恢复所有未完成的阶段：提交、终结、callback 投递。
        self._spawn(self._boot_recover())

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
    def _job_request(rec: JobRecord) -> JobRequest:
        return JobRequest(job_uuid=rec.job_uuid,
                          description=rec.description,
                          local_dir=rec.local_dir,
                          provider_id=rec.provider_id,
                          job_env_name=rec.job_env_name,
                          result_patterns=rec.result_patterns)

    async def _boot_recover(self):
        """Resume every durable phase after a restart.

        Order matters: submissions must finish before polling can finalize
        them; finalization must decide a result before the callback outbox can
        deliver it.
        """
        try:
            records = await self.db.list_incomplete_submissions()
        except Exception as e:
            logger.error(f"Failed to list incomplete submissions: {e}", exc_info=True)
            records = []
        for rec in records:
            logger.info(f"Resuming submission of job {rec.job_uuid} "
                        f"(state={rec.submission_state.value}).")
            self._spawn(self._run_submission(self._job_request(rec)))

        try:
            records = await self.db.list_finalize_pending()
        except Exception as e:
            logger.error(f"Failed to list pending finalizations: {e}", exc_info=True)
            records = []
        for rec in records:
            logger.info(f"Resuming finalization of job {rec.job_uuid} "
                        f"(state={rec.finalize_state.value}).")
            self._spawn(self._finalize_job(rec.job_uuid))

        try:
            records = await self.db.list_callback_pending()
        except Exception as e:
            logger.error(f"Failed to list pending callbacks: {e}", exc_info=True)
            records = []
        if records:
            logger.info(f"Re-posting {len(records)} pending callback(s) from previous run.")
        for rec in records:
            self._spawn(self._deliver_callback(rec.job_uuid))

    async def _deliver_callback(self, job_uuid: str):
        """Deliver the persisted result exactly as it was decided.

        `callback_pending` is an outbox flag: the payload is read from the DB
        rather than inferred from the current status, so a crash cannot turn a
        failed download into a success callback.

        Delivery is at-least-once.  If the agent has not registered its
        ``(thread_id, interrupt_id)`` yet (a retry submission's interrupt can
        surface in the outer graph well after the job already finished), or if
        the HTTP call fails, the row stays pending and the polling loop tries
        again.  Do not send a malformed callback with an empty interrupt id:
        the agent endpoint would reject it with HTTP 400 and the result would
        be lost.
        """
        if job_uuid in self._delivering:
            return
        self._delivering.add(job_uuid)
        try:
            record = await self.db.get(job_uuid)
            if record is None:
                logger.error(f"Cannot deliver callback, job {job_uuid} does not exist.")
                return
            if record.finalize_state == FinalizeState.DELIVERED:
                return
            if record.result_status is None:
                logger.error(f"Cannot deliver callback for {job_uuid}: result is not ready.")
                return
            if not record.thread_id or not record.interrupt_id:
                # The agent registers the resume target when the JobSubmission
                # interrupt reaches the outer graph.  That can be later than
                # job completion; keep the result in the outbox.
                logger.warning(f"callback job_uuid={job_uuid} is waiting for the agent "
                               f"to register thread_id/interrupt_id.")
                return

            job_result = JobResult(job_uuid=job_uuid,
                                   thread_id=record.thread_id,
                                   interrupt_id=record.interrupt_id,
                                   local_dir=record.local_dir,
                                   status=record.result_status,
                                   error_msg=record.result_error or "")
            body = job_result.model_dump_json()
            for delay in self.retry_callback:
                try:
                    response = await self.client.post(
                        self.callback_url, content=body,
                        headers={"Content-Type": "application/json"})
                    if 200 <= response.status_code < 300:
                        await self.db.mark_callback_delivered(job_uuid)
                        logger.info(f"callback ok job_uuid = {job_uuid}.")
                        return
                    # 404 means the target/thread is gone; treating it as
                    # delivered would silently lose the result.  Retry and
                    # leave it pending for the periodic outbox sweep.
                    logger.warning(f"callback job_uuid={job_uuid} http={response.status_code}, retrying.")
                except Exception as e:
                    logger.warning(f"callback job_uuid={job_uuid} post failed:{e}.")
                await asyncio.sleep(delay)
            logger.error(f"callback job_uuid={job_uuid} exhausted retries (left pending)")
        finally:
            self._delivering.discard(job_uuid)

    async def _finalize_job(self, job_uuid: str):
        """Download, verify and decide the result, then enqueue the callback.

        The task is idempotent: `claim_finalize` moves PENDING -> FINALIZING
        atomically and the result is persisted before delivery.  If the process
        dies mid-download the state stays FINALIZING and is retried at boot
        instead of being swept as a success.
        """
        if job_uuid in self._finalizing:
            return
        self._finalizing.add(job_uuid)
        try:
            record = await self.db.get(job_uuid)
            if record is None:
                return
            if record.finalize_state in (FinalizeState.READY,
                                         FinalizeState.DELIVERED):
                return
            if not await self.db.claim_finalize(job_uuid):
                return

            decision, message = await self._decide_result(record)

            if decision == "download_retry":
                # The calculation itself may be fine; only the transfer failed.
                # Keep FINALIZING so the polling loop retries the download
                # instead of reporting "results lost" and forcing the agent to
                # submit a duplicate job.
                attempt = await self.db.increment_finalize_attempt(job_uuid)
                if attempt <= self.max_finalize_retries:
                    logger.warning(f"Download/finalize of job {job_uuid} failed "
                                   f"(attempt {attempt}/{self.max_finalize_retries}): "
                                   f"{message} Retrying on the next poll.")
                    return
                final = (f"Failed to obtain the job results after {attempt} attempts. "
                         f"The remote job is not the problem; downloading its result "
                         f"files kept failing. Last error: {message}")
                await self.db.set_result_ready(job_uuid, "error", final)
                await self._deliver_callback(job_uuid)
                return

            if decision == "retry":
                error_msg = ((record.error or "")
                             + f"\n Retry {record.attempt}:" + message)
                await self.db.set_error(job_uuid, JobStatus.FAILED,
                                        error_msg, record.attempt + 1)
                if record.attempt >= self.n_retry_job_submit:
                    final = (f"Job failed {record.attempt} times. Stop retrying. "
                             f"Below is the error messages\n\n {error_msg}")
                    await self.db.set_result_ready(job_uuid, "error", final)
                    await self._deliver_callback(job_uuid)
                    return

                # Retry from a clean plan: preprocessing may fix the input.
                await self.db.reset_finalize(job_uuid)
                await self.db.delete_sub_jobs(job_uuid)
                await self.db.set_submission_state(
                    job_uuid, SubmissionState.PLANNING)
                await self._run_submission(self._job_request(record))
                return

            await self.db.set_result_ready(job_uuid, decision, message)
            await self._deliver_callback(job_uuid)
        except Exception as e:
            # Leave the row in FINALIZING; the polling loop / next boot will
            # retry the idempotent download instead of reporting a result.
            logger.error(f"Finalize failed for job {job_uuid}: {e}", exc_info=True)
        finally:
            self._finalizing.discard(job_uuid)

    async def _decide_result(
            self, record: JobRecord
    ) -> tuple[Literal["success", "error", "retry", "download_retry"], str]:
        job_status = record.status

        if job_status == JobStatus.CANCELLED:
            return "error", "Job is cancelled."
        if job_status == JobStatus.NONE:
            return "error", "Failed to get the status of job"

        # 剩下的状态是COMPLETED, FAILED, UNKNOWN
        # 现在尝试下载任务。批量作业需要下载每一个子作业（组）的结果。
        provider = self.providers[record.provider_id]
        failed_groups = []
        failure_details = []
        for sub in record.sub_jobs:
            group_error = ""
            for delay in self.retry_job_download:
                try:
                    flag = await provider.download(
                        sub.remote_id, sub.remote_dir, sub.local_dir)
                except Exception as e:
                    group_error = str(e) or type(e).__name__
                    logger.warning(f"Failed to download sub job. "
                                   f"job_uuid={record.job_uuid}, "
                                   f"group_index={sub.group_index}, {group_error}.")
                    flag = False
                if flag:
                    group_error = ""
                    break
                await asyncio.sleep(delay)
            else:
                failed_groups.append(sub.group_index)
                failure_details.append(
                    f"group {sub.group_index}: {group_error or 'download returned False'}")

        if failed_groups:
            # A transfer error says nothing about the calculation.  Retry the
            # idempotent download on later polls instead of declaring the
            # results permanently lost.
            return "download_retry", (
                f"Failed to download the results of {len(failed_groups)} "
                f"group(s) ({', '.join(str(g) for g in failed_groups)}). "
                + "; ".join(failure_details) + ".")

        # Deterministic gate: a completed job whose declared result files are
        # absent must never be reported as success.  Provider.download has
        # already verified the archive integrity, so a missing declared file
        # is a result error rather than a transient transfer error.
        if job_status == JobStatus.COMPLETED:
            missing = self._missing_artifacts(record)
            if missing:
                return "error", (f"Declared result files are missing after "
                                 f"download: {', '.join(missing)}.")

        # 这里使用agent 检查一下是否有任务提交上的错误。
        response = None
        last_error: Exception | None = None
        for _ in range(3):
            logger.info(f"Call job agent with checking phase. local_dir: {record.local_dir}.")
            try:
                response = await run_job_agent(agent=self.agent,
                                            phase="checking",
                                            thread_id=f"thread-{uuid.uuid4()}",
                                            local_dir=record.local_dir,
                                            log_msg=True,
                                            provider_id=record.provider_id,
                                            job_env_name=record.job_env_name,
                                            http_client=self.client,
                                            url=self.front_end_url)
                if response.get("checking_output"):
                    break
            except Exception as e:
                last_error = e

        checking_output = (response or {}).get("checking_output", None)
        if not isinstance(checking_output, dict) or (
                checking_output.get("job_result_status") is None):
            # Fail closed: an unavailable analysis is not evidence of success.
            detail = (f": {last_error}" if last_error is not None
                      else "; the analysis agent returned no usable output")
            return "error", (f"Failed to analyze the status of job results{detail}. "
                             f"The result cannot be confirmed.")

        # 成功或输入文件错误则直接返回
        if checking_output.get("job_result_status") == "Success":
            return "success", ""
        if checking_output.get("job_result_status") == "InputError":
            return "error", (checking_output.get("message")
                             or "Input file error.")

        return "retry", (checking_output.get("message")
                         or "Job did not finish successfully.")

    @staticmethod
    def _missing_artifacts(record: JobRecord) -> list[str]:
        """Declared `result_patterns` that produced no file for a group.

        Every declared pattern must match in every group.  Using `any()` here
        would accept a job that produced only one of several declared outputs
        (for example `relaxed.json` but not `relax.log`), which is exactly the
        false-success case this gate is meant to prevent.  An empty pattern list
        means "download everything"; without a declaration there is nothing
        deterministic to check.
        """
        return []
        patterns = [p for p in (record.result_patterns or []) if p]
        if not patterns:
            return []
        missing = []
        for sub in record.sub_jobs:
            for pattern in patterns:
                if not glob.glob(os.path.join(sub.local_dir, pattern)):
                    missing.append(f"group {sub.group_index}: {pattern}")
        return missing


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
            # A finalize task that raised unexpectedly leaves the row in
            # FINALIZING.  Retry it here so a transient failure does not need a
            # process restart.  Claiming is atomic, so a duplicate spawn is safe.
            await self._retry_pending_finalizations()
            # The callback outbox is at-least-once.  A result may become ready
            # before the agent has registered its resume target, and a
            # transient HTTP failure must not strand it until the next restart.
            await self._retry_pending_callbacks()
            await asyncio.sleep(self.poll_interval)

    async def _retry_pending_finalizations(self):
        try:
            records = await self.db.list_finalize_pending()
        except Exception as e:
            logger.error(f"Failed to list pending finalizations: {e}", exc_info=True)
            return
        for rec in records:
            if rec.job_uuid in self._finalizing:
                continue
            if rec.status in (JobStatus.SUBMITTED, JobStatus.RUNNING):
                continue
            self._spawn(self._finalize_job(rec.job_uuid))

    async def _retry_pending_callbacks(self):
        try:
            records = await self.db.list_callback_pending()
        except Exception as e:
            logger.error(f"Failed to list pending callbacks: {e}", exc_info=True)
            return
        for rec in records:
            if rec.job_uuid in self._delivering:
                continue
            self._spawn(self._deliver_callback(rec.job_uuid))

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
            if state == JobStatus.NONE:
                # The status query failed (SSH error, timeout, or an
                # unrecognized provider response).  That is not a terminal
                # state: keep the previous status and ask again next cycle.
                logger.warning(f"Could not determine the state of job {remote_id}; "
                               f"keeping its current state.")
                continue
            await self.db.update_sub_job_status(remote_id, state)

        # 在这里分发任务：只有当一个作业的所有子作业都到达终止状态时
        # 才finalize，保证下载和callback对整个作业只做一次。
        finalized_jobs = set()
        for remote_id, state in remote_job_states.items():
            if state in (JobStatus.SUBMITTED, JobStatus.RUNNING, JobStatus.NONE):
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
            if fresh.finalize_state in (FinalizeState.READY,
                                        FinalizeState.DELIVERED):
                continue
            if fresh.job_uuid in self._finalizing:
                continue
            finalized_jobs.add(job.job_uuid)
            self._spawn(self._finalize_job(fresh.job_uuid))
