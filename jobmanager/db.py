from typing import Optional

import asyncio, contextlib, aiosqlite, json, sqlite3
from jobmanager.models import JobRecord, SubJobRecord, JobStatus

ACTIVE_STATUS = (JobStatus.SUBMITTED.value, JobStatus.RUNNING.value)
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
            error            TEXT DEFAULT ''
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
        await self._db.commit()

    async def close(self):
        if self._db is not None:
            await self._db.close()
            self._db = None 

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> JobRecord:
        return JobRecord(
            job_uuid=row["job_uuid"],
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
            error=row["error"])
    
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
                   callback_pending, attempt, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
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
                       error=excluded.error"""
        else:
            sql = """INSERT INTO jobs
                   ( job_uuid, description, provider_id, job_env_name, local_dir,
                   result_patterns, status, created_at, updated_at,
                   callback_pending, attempt, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)"""
        params = (
            rec.job_uuid, rec.description, rec.provider_id, rec.job_env_name, rec.local_dir,
            json.dumps(rec.result_patterns), rec.status.value, rec.created_at, rec.updated_at,
            int(rec.callback_pending), rec.attempt, rec.error or ""
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
        """Terminal rows whose callback has not yet succeeded (or was lost to
        a crash). Boot sweep re-POSTs these for at-least-once delivery."""
        
        cur = await self._db.execute(
            "SELECT * FROM jobs WHERE callback_pending = 1 ORDER BY updated_at ASC"
        )
        rows = await cur.fetchall()
        await cur.close()
        return [self._row_to_job(r) for r in rows]
    
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

    async def update_status(self, job_uuid: str, status: JobStatus) -> None:
        await self._db.execute(
            """UPDATE jobs SET status = ?, 
               updated_at = ? WHERE job_uuid = ?""",
            (status.value,  _now(), job_uuid),
        )
        await self._db.commit()

    async def set_error(self, job_uuid: str, status: JobStatus, error: str, attempt: int) -> None:
        now = _now()
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute(
                    """UPDATE jobs SET status = ?, error = ?, attempt = ?, updated_at = ?
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
        await self._db.execute(
             "UPDATE jobs SET callback_pending = ?, updated_at = ? WHERE job_uuid = ?",
            ((1 if flag else 0), _now(), job_uuid),
        )
        await self._db.commit()

    async def delete_job(self, job_uuid: str) -> None:
        async with self._write_lock:
            async with self._transaction() as db:
                await db.execute("DELETE FROM sub_jobs WHERE job_uuid = ?", (job_uuid,))
                await db.execute("DELETE FROM jobs WHERE job_uuid = ?", (job_uuid,))

    async def insert_sub_job(self, sub: SubJobRecord) -> None:
        await self._upsert_sub_job(sub, replace=False)
        await self._db.commit()

    async def save_sub_job(self, sub: SubJobRecord) -> None:
        await self._upsert_sub_job(sub, replace=True)
        await self._db.commit()

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
        await self._db.execute(
            """UPDATE sub_jobs SET remote_id = ?, remote_dir = ?, status = ?,
                updated_at = ? WHERE job_uuid = ? AND group_index = ?""",
            (remote_id, remote_dir, status.value, _now(), job_uuid, group_index),
        )
        await self._db.commit()

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
        await self._db.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE job_uuid = ?",
            (new_status.value, _now(), job_uuid),
        )
        return new_status

    async def delete_sub_jobs(self, job_uuid: str) -> None:
        await self._db.execute("DELETE FROM sub_jobs WHERE job_uuid = ?", (job_uuid,))
        await self._db.commit()

def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()