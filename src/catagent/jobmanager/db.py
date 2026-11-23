from typing import Optional, Literal

import asyncio, contextlib, aiosqlite, json, sqlite3
from catagent.jobmanager.models import (
    JobRecord, SubJobRecord, JobStatus, SubmissionState, FinalizeState,
)

ACTIVE_STATUS = (JobStatus.SUBMITTED.value, JobStatus.RUNNING.value)
TERMINAL_STATUS = (
    JobStatus.COMPLETED.value,
    JobStatus.FAILED.value,
    JobStatus.CANCELLED.value,
    JobStatus.UNKNOWN.value,
    JobStatus.NONE.value,
)
ERROR_STATUS_PRIORITY = (
    JobStatus.CANCELLED.value,
    JobStatus.FAILED.value,
    JobStatus.NONE.value,
    JobStatus.UNKNOWN.value,
)


def _aggregate_status(sub_statuses: list) -> JobStatus:
    """把所有子作业的状态聚合成作业的总状态。

    - 只要有子作业还在排队或运行，作业保持活跃状态（running/submitted）；
    - 所有子作业都到达终止状态后，任意子作业出错则作业为出错状态，否则为completed。
    """
    if not sub_statuses:
        return JobStatus.SUBMITTED
    if any(s in ACTIVE_STATUS for s in sub_statuses):
        if JobStatus.RUNNING.value in sub_statuses:
            return JobStatus.RUNNING
        return JobStatus.SUBMITTED
    for status in ERROR_STATUS_PRIORITY:
        if status in sub_statuses:
            return JobStatus(status)
    return JobStatus.COMPLETED

class JobDB:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        # 单连接上串行化写操作，避免多个协程的事务相互穿插。
        self._write_lock = asyncio.Lock()

    @contextlib.asynccontextmanager
    async def _transaction(self):
        """显式事务。需要调用方先持有 _write_lock，保证事务之间不交错。"""
        await self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            await self._db.rollback()
            raise
        else:
            await self._db.commit()

    async def connect(self):
        # isolation_level=None 关闭sqlite3的隐式事务管理，改由 _transaction 显式控制。
        self._db = await aiosqlite.connect(self._db_path, isolation_level=None)
        self._db.row_factory = sqlite3.Row

        # Write-Ahead Logging，预写日志
        # 读取和写入可以同时进行互不阻塞
        await self._db.execute("PRAGMA journal_mode=WAL")

        # 使用外键约束检查
        # 有时候数据有一对多或多对多的关系，例如一个用户可以有多个订单。订单存放在另一个表中，此时user_id就是外键。外键检查防止插入不存在的user_id
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.commit()
        await self._create_tables()

        
    async def _create_tables(self):

        await self._db.execute('''
        CREATE TABLE IF NOT EXISTS jobs (
            job_uuid         TEXT PRIMARY KEY,
            thread_id        TEXT,
            interrupt_id     TEXT,
            description      TEXT,
            provider_id      TEXT NOT NULL,
            job_env_name     TEXT NOT NULL,
            local_dir        TEXT NOT NULL,
            result_patterns  TEXT NOT NULL DEFAULT '[]',
            status           TEXT NOT NULL DEFAULT 'pending',
            created_at       TEXT NOT NULL,
            updated_at       TEXT NOT NULL,
            callback_pending INTEGER NOT NULL DEFAULT 0,
            attempt          INTEGER NOT NULL DEFAULT 0,
            error            TEXT DEFAULT '',
            submission_state TEXT NOT NULL DEFAULT 'submitted',
            finalize_state   TEXT NOT NULL DEFAULT 'none',
            finalize_attempt INTEGER NOT NULL DEFAULT 0,
            result_status    TEXT,
            result_error     TEXT
        )
        ''')

        # UNIQUE(job_uuid, group_index) 是复合唯一约束
        # job_uuid 和 group_index 合在一起必须唯一，但它们单独可以不唯一。
        # FOREIGN 建立外键、实现级联删除。
        await self._db.execute('''
        CREATE TABLE IF NOT EXISTS sub_jobs (
            sub_job_id  INTEGER PRIMARY KEY AUTOINCREMENT,
            job_uuid    TEXT NOT NULL,
            group_index INTEGER NOT NULL,

            local_dir   TEXT NOT NULL,
            remote_id   TEXT,
            remote_dir  TEXT,
            status      TEXT NOT NULL DEFAULT 'pending',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            error       TEXT NOT NULL DEFAULT '',

            UNIQUE(job_uuid, group_index),
            FOREIGN KEY(job_uuid) REFERENCES jobs(job_uuid) ON DELETE CASCADE
        )
        ''')


        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_callback_pending "
            "ON jobs(callback_pending) WHERE callback_pending = 1"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sub_jobs_job_uuid ON sub_jobs(job_uuid)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sub_jobs_status ON sub_jobs(status)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sub_jobs_remote_id ON sub_jobs(remote_id)"
        )
        # Migrate before creating indexes on the new columns; an existing
        # database does not have `finalize_state` / `submission_state` yet.
        await self._migrate_schema()
        await self._db.commit()
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_finalize_state "
            "ON jobs(finalize_state)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_submission_state "
            "ON jobs(submission_state)"
        )
        await self._db.commit()

    async def _job_columns(self) -> set[str]:
        cur = await self._db.execute("PRAGMA table_info(jobs)")
        rows = await cur.fetchall()
        await cur.close()
        return {row[1] for row in rows}

    async def _migrate_schema(self) -> None:
        """Add columns introduced after the first database layout.

        Existing terminal rows are marked `pending` for finalization: re-running
        finalize is safer than trusting the old infer-from-status callback,
        which could report success even when no result files were downloaded.
        """
        columns = await self._job_columns()

        added_finalize = False
        if "submission_state" not in columns:
            await self._db.execute(
                "ALTER TABLE jobs ADD COLUMN submission_state TEXT "
                "NOT NULL DEFAULT 'submitted'")
        if "finalize_state" not in columns:
            await self._db.execute(
                "ALTER TABLE jobs ADD COLUMN finalize_state TEXT "
                "NOT NULL DEFAULT 'none'")
            added_finalize = True
        if "finalize_attempt" not in columns:
            await self._db.execute(
                "ALTER TABLE jobs ADD COLUMN finalize_attempt INTEGER "
                "NOT NULL DEFAULT 0")
        if "result_status" not in columns:
            await self._db.execute(
                "ALTER TABLE jobs ADD COLUMN result_status TEXT")
        if "result_error" not in columns:
            await self._db.execute(
                "ALTER TABLE jobs ADD COLUMN result_error TEXT")

        if added_finalize:
            placeholders = ",".join("?" for _ in TERMINAL_STATUS)
            await self._db.execute(
                f"UPDATE jobs SET finalize_state = 'pending' "
                f"WHERE status IN ({placeholders})", TERMINAL_STATUS)

    async def close(self):
        if self._db is not None:
            await self._db.close()
            self._db = None 

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> JobRecord:
        return JobRecord(
            job_uuid=row["job_uuid"],
            thread_id=row["thread_id"] or "",
            interrupt_id=row["interrupt_id"] or "",
            description=row["description"],
            provider_id=row["provider_id"],
            job_env_name=row["job_env_name"],
            local_dir=row["local_dir"],
            result_patterns=json.loads(row["result_patterns"] or "[]"),
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            callback_pending=bool(row["callback_pending"]),
            attempt=row["attempt"],
            error=row["error"],
            submission_state=SubmissionState(
                row["submission_state"] or SubmissionState.SUBMITTED.value),
            finalize_state=FinalizeState(
                row["finalize_state"] or FinalizeState.NONE.value),
            finalize_attempt=row["finalize_attempt"] or 0,
            result_status=row["result_status"],
            result_error=row["result_error"])
    
    @staticmethod
    def _row_to_sub_job(row: aiosqlite.Row) -> SubJobRecord:
        return SubJobRecord(
            job_uuid=row["job_uuid"],
            group_index=row["group_index"],
            local_dir=row["local_dir"],
            remote_id=row["remote_id"],
            remote_dir=row["remote_dir"],
            status=JobStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            error=row["error"])
    
    async def insert(self, rec: JobRecord) -> None:
        await self._upsert(rec, replace=False)

    async def save(self, rec: JobRecord) -> None:
        await self._upsert(rec, replace=True)

    async def _upsert(self, rec: JobRecord, replace: bool) -> None:
        # 用 ON CONFLICT DO UPDATE 而不是 INSERT OR REPLACE：
        # REPLACE 会先删除再插入jobs行，从而触发 sub_jobs 的 ON DELETE CASCADE。
        if replace:
            sql = """INSERT INTO jobs
                   ( job_uuid, description, provider_id, job_env_name, local_dir,
                   result_patterns, status, created_at, updated_at,
                   callback_pending, attempt, error,
                   submission_state, finalize_state, finalize_attempt,
                   result_status, result_error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(job_uuid) DO UPDATE SET
                       description=excluded.description,
                       provider_id=excluded.provider_id,
                       job_env_name=excluded.job_env_name,
                       local_dir=excluded.local_dir,
                       result_patterns=excluded.result_patterns,
                       status=excluded.status,
                       updated_at=excluded.updated_at,
                       callback_pending=excluded.callback_pending,
                       attempt=excluded.attempt,
                       error=excluded.error,
                       submission_state=excluded.submission_state,
                       finalize_state=excluded.finalize_state,
                       finalize_attempt=excluded.finalize_attempt,
                       result_status=excluded.result_status,
                       result_error=excluded.result_error"""
        else:
            sql = """INSERT INTO jobs
                   ( job_uuid, description, provider_id, job_env_name, local_dir,
                   result_patterns, status, created_at, updated_at,
                   callback_pending, attempt, error,
                   submission_state, finalize_state, finalize_attempt,
                   result_status, result_error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""
        params = (
            rec.job_uuid, rec.description, rec.provider_id, rec.job_env_name, rec.local_dir,
            json.dumps(rec.result_patterns), rec.status.value, rec.created_at, rec.updated_at,
            int(rec.callback_pending), rec.attempt, rec.error or "",
            rec.submission_state.value, rec.finalize_state.value, rec.finalize_attempt,
            rec.result_status, rec.result_error,
        )
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(sql, params)
                # 子作业随作业一起写入
                for sub in rec.sub_jobs:
                    await self._upsert_sub_job(sub, replace=replace)

    async def get(self, job_uuid: str) -> Optional[JobRecord]:
        cur = await self._db.execute("SELECT * FROM jobs WHERE job_uuid = ?", (job_uuid,))
        row = await cur.fetchone()
        await cur.close()
        if not row:
            return None
        rec = self._row_to_job(row)
        rec.sub_jobs = await self.get_sub_jobs(job_uuid)
        return rec


    async def list_active_jobs(self, provider_id: str = None) -> list[JobRecord]:
        placeholders = ",".join("?" for _ in ACTIVE_STATUS)
        if provider_id:
            sql = (f"SELECT * FROM jobs WHERE status IN ({placeholders}) "
                   f"AND provider_id = ? ORDER BY created_at ASC")
            params: tuple = tuple(s for s in ACTIVE_STATUS) + (provider_id,)
        else:
            sql = f"SELECT * FROM jobs WHERE status IN ({placeholders}) ORDER BY created_at ASC"
            params = tuple(s for s in ACTIVE_STATUS)
        cur = await self._db.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()
        records = [self._row_to_job(r) for r in rows]
        for rec in records:
            rec.sub_jobs = await self.get_sub_jobs(rec.job_uuid)
        return records

    async def list_callback_pending(self) -> list[JobRecord]:
        """Rows whose final result is persisted but whose callback was not
        acknowledged. Boot sweep re-POSTs these for at-least-once delivery."""
        cur = await self._db.execute(
            "SELECT * FROM jobs WHERE finalize_state = ? AND callback_pending = 1 "
            "ORDER BY updated_at ASC", (FinalizeState.READY.value,)
        )
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_job(r) for r in rows]

    async def list_finalize_pending(self) -> list[JobRecord]:
        """Terminal rows whose final result has not been decided yet.

        Included states are PENDING (never started) and FINALIZING (crashed
        while downloading/checking).  A crash during download must be retried,
        never turned into a success callback by the boot sweep.
        """
        cur = await self._db.execute(
            "SELECT * FROM jobs WHERE finalize_state IN (?, ?) "
            "ORDER BY updated_at ASC",
            (FinalizeState.PENDING.value, FinalizeState.FINALIZING.value),
        )
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_job(r) for r in rows]

    async def list_incomplete_submissions(self) -> list[JobRecord]:
        """Active rows whose plan was not fully submitted.

        This covers a crash after the HTTP request was accepted (no plan yet),
        a crash after the plan was persisted but before every remote id was
        written, and legacy placeholders that have no sub_jobs at all.
        """
        active = ",".join("?" for _ in ACTIVE_STATUS)
        incomplete = ",".join("?" for _ in (
            SubmissionState.PLANNING.value,
            SubmissionState.PLANNED.value,
            SubmissionState.SUBMITTING.value,
        ))
        cur = await self._db.execute(
            f"""SELECT * FROM jobs
                WHERE status IN ({active})
                  AND (submission_state IN ({incomplete})
                       OR NOT EXISTS (
                           SELECT 1 FROM sub_jobs
                           WHERE sub_jobs.job_uuid = jobs.job_uuid))
                ORDER BY created_at ASC""",
            ACTIVE_STATUS + (
                SubmissionState.PLANNING.value,
                SubmissionState.PLANNED.value,
                SubmissionState.SUBMITTING.value,
            ),
        )
        rows = await cur.fetchall()
        await cur.close()
        records = [self._row_to_job(r) for r in rows]
        for rec in records:
            rec.sub_jobs = await self.get_sub_jobs(rec.job_uuid)
        return records
    
    async def list_all(self, limit: int = 200) -> list[JobRecord]:
        cur = await self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        )
        rows = await cur.fetchall()
        await cur.close()
        records = [self._row_to_job(r) for r in rows]
        for rec in records:
            rec.sub_jobs = await self.get_sub_jobs(rec.job_uuid)
        return records

    # async def set_remote(self, job_uuid: str, remote_id: str,
    #                          remote_dir: Optional[str], status: JobStatus) -> None:
    #     await self._db.execute(
    #         """UPDATE jobs SET remote_id = ?, remote_path = ?, status = ?,
    #             updated_at = ? WHERE job_uuid = ?""",
    #         (remote_id, remote_dir, status.value, _now(), job_uuid),
    #     )
    #     await self._db.commit()
    # thread_id        TEXT,
    #             interrupt_id     TEXT,

    async def set_thread_id(self, job_uuid:str, thread_id:str, interrupt_id:str) -> None:
        # Registration is best-effort and can be repeated when the outer graph
        # re-emits a bundle of pending interrupts.  An empty value means the
        # caller could not identify the resume target yet; never let it erase a
        # valid registration.
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    """UPDATE jobs SET
                           thread_id = CASE WHEN ? != '' THEN ? ELSE thread_id END,
                           interrupt_id = CASE WHEN ? != '' THEN ? ELSE interrupt_id END
                       WHERE job_uuid = ?""",
                    (thread_id or "", thread_id or "",
                     interrupt_id or "", interrupt_id or "", job_uuid))

    async def update_status(self, job_uuid: str, status: JobStatus) -> None:
        terminal = status.value in TERMINAL_STATUS
        finalize_sql = (
            ", finalize_state = CASE WHEN finalize_state = 'none' "
            "THEN 'pending' ELSE finalize_state END" if terminal else "")
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    f"""UPDATE jobs SET status = ?, updated_at = ?{finalize_sql}
                        WHERE job_uuid = ?""",
                    (status.value, _now(), job_uuid),
                )

    async def set_error(self, job_uuid: str, status: JobStatus, error: str, attempt: int) -> None:
        now = _now()
        terminal = status.value in TERMINAL_STATUS
        finalize_sql = (
            ", finalize_state = CASE WHEN finalize_state = 'none' "
            "THEN 'pending' ELSE finalize_state END" if terminal else "")
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    f"""UPDATE jobs SET status = ?, error = ?, attempt = ?,
                        updated_at = ?{finalize_sql}
                        WHERE job_uuid = ?""",
                    (status.value, error, attempt, now, job_uuid),
                )
                # 错误状态传播到所有还在排队或运行的子作业，避免轮询继续查询已终止的子作业
                await db.execute(
                    """UPDATE sub_jobs SET status = ?, error = ?, updated_at = ?
                       WHERE job_uuid = ? AND status IN (?, ?)""",
                    (status.value, error, now, job_uuid,
                     JobStatus.SUBMITTED.value, JobStatus.RUNNING.value),
                )

    async def set_callback_pending(self, job_uuid: str, flag:bool) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    "UPDATE jobs SET callback_pending = ?, updated_at = ? WHERE job_uuid = ?",
                    ((1 if flag else 0), _now(), job_uuid),
                )

    async def set_submission_state(self, job_uuid: str,
                                   state: SubmissionState) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    "UPDATE jobs SET submission_state = ?, updated_at = ? "
                    "WHERE job_uuid = ?",
                    (state.value, _now(), job_uuid),
                )

    async def save_plan(self, job_uuid: str, subs: list[SubJobRecord]) -> None:
        """Persist the group plan before any remote job is started."""
        async with self._write_lock:
            async with self._transaction() as db:
                for sub in subs:
                    await self._upsert_sub_job(sub, replace=True)
                await db.execute(
                    "UPDATE jobs SET submission_state = ?, updated_at = ? "
                    "WHERE job_uuid = ?",
                    (SubmissionState.PLANNED.value, _now(), job_uuid),
                )

    async def claim_finalize(self, job_uuid: str) -> bool:
        """Atomically move a terminal row into FINALIZING.

        Returns False when the row is already READY/DELIVERED (another worker
        already produced the result) so finalize never runs twice.
        """
        async with self._write_lock:
            async with self._transaction() as db:
                cur = await db.execute(
                    "UPDATE jobs SET finalize_state = ?, updated_at = ? "
                    "WHERE job_uuid = ? AND finalize_state IN (?, ?)",
                    (FinalizeState.FINALIZING.value, _now(), job_uuid,
                     FinalizeState.PENDING.value, FinalizeState.FINALIZING.value),
                )
                claimed = cur.rowcount > 0
                await cur.close()
                return claimed

    async def set_result_ready(self, job_uuid: str,
                               status: Literal["success", "error"],
                               error: Optional[str]) -> None:
        """Persist the final result and mark the callback outbox row ready."""
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    """UPDATE jobs SET finalize_state = ?, result_status = ?,
                       result_error = ?, callback_pending = 1, updated_at = ?
                       WHERE job_uuid = ?""",
                    (FinalizeState.READY.value, status, error or "",
                     _now(), job_uuid),
                )

    async def mark_callback_delivered(self, job_uuid: str) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    """UPDATE jobs SET finalize_state = ?, callback_pending = 0,
                       updated_at = ? WHERE job_uuid = ?""",
                    (FinalizeState.DELIVERED.value, _now(), job_uuid),
                )

    async def reset_finalize(self, job_uuid: str) -> None:
        """Return a terminal row to the pre-finalize state for a resubmission."""
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    """UPDATE jobs SET finalize_state = ?, callback_pending = 0,
                       finalize_attempt = 0, result_status = NULL,
                       result_error = NULL, updated_at = ?
                       WHERE job_uuid = ?""",
                    (FinalizeState.NONE.value, _now(), job_uuid),
                )

    async def increment_finalize_attempt(self, job_uuid: str) -> int:
        """Count one failed download/finalize pass and return the new count."""
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    """UPDATE jobs SET finalize_attempt = finalize_attempt + 1,
                       updated_at = ? WHERE job_uuid = ?""",
                    (_now(), job_uuid),
                )
                cur = await db.execute(
                    "SELECT finalize_attempt FROM jobs WHERE job_uuid = ?",
                    (job_uuid,))
                row = await cur.fetchone()
                await cur.close()
                return int(row["finalize_attempt"]) if row else 0

    async def delete_job(self, job_uuid: str) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute("DELETE FROM sub_jobs WHERE job_uuid = ?", (job_uuid,))
                await db.execute("DELETE FROM jobs WHERE job_uuid = ?", (job_uuid,))

    async def insert_sub_job(self, sub: SubJobRecord) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await self._upsert_sub_job(sub, replace=False)

    async def save_sub_job(self, sub: SubJobRecord) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await self._upsert_sub_job(sub, replace=True)

    async def _upsert_sub_job(self, sub: SubJobRecord, replace: bool) -> None:
        if replace:
            sql = """INSERT INTO sub_jobs
                   ( job_uuid, group_index, local_dir, remote_id, remote_dir,
                   status, created_at, updated_at, error)
                   VALUES (?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(job_uuid, group_index) DO UPDATE SET
                       local_dir=excluded.local_dir,
                       remote_id=excluded.remote_id,
                       remote_dir=excluded.remote_dir,
                       status=excluded.status,
                       updated_at=excluded.updated_at,
                       error=excluded.error"""
        else:
            sql = """INSERT INTO sub_jobs
                   ( job_uuid, group_index, local_dir, remote_id, remote_dir,
                   status, created_at, updated_at, error)
                   VALUES (?,?,?,?,?,?,?,?,?)"""
        await self._db.execute(
            sql,
            (
                sub.job_uuid, sub.group_index, sub.local_dir, sub.remote_id, sub.remote_dir,
                sub.status.value, sub.created_at or _now(), sub.updated_at or _now(),
                sub.error or ""
            ),
        )

    async def get_sub_jobs(self, job_uuid: str) -> list[SubJobRecord]:
        cur = await self._db.execute(
            "SELECT * FROM sub_jobs WHERE job_uuid = ? ORDER BY group_index ASC", (job_uuid,))
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_sub_job(r) for r in rows]

    async def get_sub_job(self, job_uuid: str, group_index: int) -> Optional[SubJobRecord]:
        cur = await self._db.execute(
            "SELECT * FROM sub_jobs WHERE job_uuid = ? AND group_index = ?",
            (job_uuid, group_index))
        row = await cur.fetchone()
        await cur.close()
        return self._row_to_sub_job(row) if row else None
    
    async def get_sub_job_by_remote_id(self, remote_id: str) -> Optional[SubJobRecord]:
        cur = await self._db.execute(
            "SELECT * FROM sub_jobs WHERE remote_id = ?", (remote_id,))
        row = await cur.fetchone()
        await cur.close()
        return self._row_to_sub_job(row) if row else None

    async def list_active_sub_jobs(self, provider_id: str = None) -> list[tuple[JobRecord, SubJobRecord]]:
        """列出所有还在远端排队或运行的子作业，返回(作业, 子作业)对。
        批量作业被拆分成多个子作业后，轮询以子作业为单位进行。"""
        placeholders = ",".join("?" for _ in ACTIVE_STATUS)
        if provider_id:
            sql = (f"""SELECT jobs.*,
                              sub_jobs.group_index AS sub_group_index,
                              sub_jobs.local_dir   AS sub_local_dir,
                              sub_jobs.remote_id   AS sub_remote_id,
                              sub_jobs.remote_dir  AS sub_remote_dir,
                              sub_jobs.status      AS sub_status,
                              sub_jobs.created_at  AS sub_created_at,
                              sub_jobs.updated_at  AS sub_updated_at,
                              sub_jobs.error       AS sub_error
                       FROM sub_jobs JOIN jobs ON sub_jobs.job_uuid = jobs.job_uuid
                       WHERE sub_jobs.status IN ({placeholders}) AND jobs.provider_id = ?
                       ORDER BY jobs.created_at ASC, sub_jobs.group_index ASC""")
            params: tuple = tuple(s for s in ACTIVE_STATUS) + (provider_id,)
        else:
            sql = f"""SELECT jobs.*,
                             sub_jobs.group_index AS sub_group_index,
                             sub_jobs.local_dir   AS sub_local_dir,
                             sub_jobs.remote_id   AS sub_remote_id,
                             sub_jobs.remote_dir  AS sub_remote_dir,
                             sub_jobs.status      AS sub_status,
                             sub_jobs.created_at  AS sub_created_at,
                             sub_jobs.updated_at  AS sub_updated_at,
                             sub_jobs.error       AS sub_error
                      FROM sub_jobs JOIN jobs ON sub_jobs.job_uuid = jobs.job_uuid
                      WHERE sub_jobs.status IN ({placeholders})
                      ORDER BY jobs.created_at ASC, sub_jobs.group_index ASC"""
            params = tuple(s for s in ACTIVE_STATUS)
        cur = await self._db.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()

        pairs = []
        for row in rows:
            job = self._row_to_job(row)
            sub = SubJobRecord(
                job_uuid=row["job_uuid"],
                group_index=row["sub_group_index"],
                local_dir=row["sub_local_dir"],
                remote_id=row["sub_remote_id"],
                remote_dir=row["sub_remote_dir"],
                status=JobStatus(row["sub_status"]),
                created_at=row["sub_created_at"],
                updated_at=row["sub_updated_at"],
                error=row["sub_error"])
            pairs.append((job, sub))
        return pairs

    async def set_sub_job_remote(self, job_uuid: str, group_index: int,
                                 remote_id: str, remote_dir: Optional[str],
                                 status: JobStatus) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    """UPDATE sub_jobs SET remote_id = ?, remote_dir = ?, status = ?,
                        updated_at = ? WHERE job_uuid = ? AND group_index = ?""",
                    (remote_id, remote_dir, status.value, _now(),
                     job_uuid, group_index),
                )

    async def update_sub_job_status(self, remote_id: str, status: JobStatus) -> None:
        """更新子作业状态，并重新聚合作业（jobs表）的总状态。"""
        now = _now()
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    "UPDATE sub_jobs SET status = ?, updated_at = ? WHERE remote_id = ?",
                    (status.value, now, remote_id),
                )
                cur = await db.execute(
                    "SELECT job_uuid FROM sub_jobs WHERE remote_id = ?", (remote_id,))
                row = await cur.fetchone()
                await cur.close()
                if row:
                    await self._recompute_job_status(row["job_uuid"])

    async def _recompute_job_status(self, job_uuid: str) -> JobStatus:
        """根据所有子作业的状态聚合出作业的总状态并写回jobs表。"""
        cur = await self._db.execute(
            "SELECT status FROM sub_jobs WHERE job_uuid = ?", (job_uuid,))
        rows = await cur.fetchall()
        await cur.close()
        if not rows:
            return JobStatus.SUBMITTED
        new_status = _aggregate_status([r["status"] for r in rows])
        if new_status.value in TERMINAL_STATUS:
            # Enter the finalize state machine exactly once.  Once READY or
            # DELIVERED, later updates must not move it back to PENDING.
            await self._db.execute(
                """UPDATE jobs SET status = ?, updated_at = ?,
                   finalize_state = CASE WHEN finalize_state = 'none'
                       THEN 'pending' ELSE finalize_state END
                   WHERE job_uuid = ?""",
                (new_status.value, _now(), job_uuid),
            )
        else:
            await self._db.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_uuid = ?",
                (new_status.value, _now(), job_uuid),
            )
        return new_status

    async def delete_sub_jobs(self, job_uuid: str) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    "DELETE FROM sub_jobs WHERE job_uuid = ?", (job_uuid,))

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()