"""SQLite checkpointer with working prune support.

``langgraph``'s :class:`AsyncSqliteSaver` inherits ``aprune`` from
``BaseCheckpointSaver``, but the SQLite implementation is still a
``NotImplementedError``.  There is nothing in the SQLite schema that prevents
pruning: checkpoints are keyed by ``(thread_id, checkpoint_ns,
checkpoint_id)`` and writes carry the same key, so individual snapshots can be
deleted exactly like the PostgreSQL saver does.

This module implements the two strategies from the public API:

``strategy="keep_latest"``
    Keep the newest checkpoint of every namespace, plus its pending writes.
    Older checkpoints and their writes are deleted, and a kept checkpoint
    whose parent was deleted gets ``parent_checkpoint_id = NULL``.

``strategy="delete"``
    Delete every checkpoint and write belonging to the given threads.

The implementation follows ``AsyncPostgresSaver.aprune``, except that it does
not try to detect ``DeltaChannel`` usage.  The graphs in this repository do not
use ``DeltaChannel``; if one is added, ``keep_latest`` would need the
DeltaChannel-aware ancestor walk described in ``BaseCheckpointSaver.prune``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Channel used by LangGraph for pending `Send` objects in checkpoint versions
# older than 4.  Keeping that checkpoint's parent preserves those sends.
_TASKS_CHANNEL = "__pregel_tasks"

PRUNE_STRATEGIES = ("keep_latest", "delete")

# Stay below SQLite's historical 999-variable limit for `IN (...)` queries.
_SQL_CHUNK_SIZE = 500


def _chunks(values: Sequence[str], size: int = _SQL_CHUNK_SIZE) -> list[Sequence[str]]:
    return [values[i:i + size] for i in range(0, len(values), size)]


class PrunableAsyncSqliteSaver(AsyncSqliteSaver):
    """``AsyncSqliteSaver`` with an implemented ``aprune`` method."""

    @asynccontextmanager
    async def _immediate_transaction(self) -> AsyncIterator[None]:
        """Serialize with the saver's other operations and take a write lock.

        ``BEGIN IMMEDIATE`` acquires SQLite's write lock at the start, so a
        concurrent writer cannot insert new checkpoints between the rows we
        read and the rows we delete.
        """
        async with self.lock:
            await self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield
            except BaseException:
                await self.conn.rollback()
                raise
            else:
                await self.conn.commit()

    async def aprune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Asynchronously prune checkpoints for the given threads.

        Args:
            thread_ids: The thread IDs to prune.
            strategy: ``"keep_latest"`` keeps only the newest checkpoint of
                each namespace; ``"delete"`` removes all checkpoints and
                writes of the threads.

        Raises:
            TypeError: If ``thread_ids`` is a single string.
            ValueError: If ``strategy`` is unknown.
        """
        if isinstance(thread_ids, str):
            raise TypeError("thread_ids must be a sequence of thread IDs, not a str")
        if strategy not in PRUNE_STRATEGIES:
            raise ValueError(f"Unknown prune strategy: {strategy!r}")

        thread_ids = [str(thread_id) for thread_id in thread_ids]
        if not thread_ids:
            return

        await self.setup()
        if strategy == "delete":
            async with self._immediate_transaction():
                for chunk in _chunks(thread_ids):
                    placeholders = ",".join("?" * len(chunk))
                    await self.conn.execute(
                        f"DELETE FROM writes WHERE thread_id IN ({placeholders})",
                        chunk,
                    )
                    await self.conn.execute(
                        f"DELETE FROM checkpoints WHERE thread_id IN ({placeholders})",
                        chunk,
                    )
            return

        # One transaction per thread, like the PostgreSQL implementation, so a
        # failure on a later thread does not roll back threads already pruned.
        for thread_id in thread_ids:
            async with self._immediate_transaction():
                await self._keep_latest_thread(thread_id)

    async def _keep_latest_thread(self, thread_id: str) -> None:
        """Keep the newest checkpoint of each namespace for one thread."""
        cursor = await self.conn.execute(
            "SELECT checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint "
            "FROM checkpoints WHERE thread_id = ? "
            "ORDER BY checkpoint_ns ASC, checkpoint_id DESC",
            (thread_id,),
        )
        rows = await cursor.fetchall()

        by_namespace: dict[str, list[tuple[str, str | None, str | None, Any]]] = {}
        for checkpoint_ns, checkpoint_id, parent_id, type_tag, checkpoint in rows:
            by_namespace.setdefault(checkpoint_ns, []).append(
                (checkpoint_id, parent_id, type_tag, checkpoint)
            )

        for checkpoint_ns, checkpoints in by_namespace.items():
            head_id, parent_id, type_tag, checkpoint = checkpoints[0]
            keep = [head_id]

            if parent_id and await self._parent_is_needed(
                    checkpoint, type_tag, thread_id, checkpoint_ns, parent_id,
                    checkpoints):
                keep.append(parent_id)

            unkept = [
                checkpoint_id
                for checkpoint_id, _, _, _ in checkpoints
                if checkpoint_id not in keep
            ]
            for chunk in _chunks(unkept):
                placeholders = ",".join("?" * len(chunk))
                await self.conn.execute(
                    "DELETE FROM writes WHERE thread_id = ? AND checkpoint_ns = ? "
                    f"AND checkpoint_id IN ({placeholders})",
                    (thread_id, checkpoint_ns, *chunk),
                )
                await self.conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = ? "
                    f"AND checkpoint_id IN ({placeholders})",
                    (thread_id, checkpoint_ns, *chunk),
                )

            # Do not leave the kept checkpoint pointing at a deleted parent.
            placeholders = ",".join("?" * len(keep))
            await self.conn.execute(
                "UPDATE checkpoints SET parent_checkpoint_id = NULL "
                "WHERE thread_id = ? AND checkpoint_ns = ? "
                f"AND checkpoint_id IN ({placeholders}) "
                "AND parent_checkpoint_id IS NOT NULL "
                f"AND parent_checkpoint_id NOT IN ({placeholders})",
                (thread_id, checkpoint_ns, *keep, *keep),
            )

    async def _parent_is_needed(
        self,
        checkpoint: Any,
        type_tag: str | None,
        thread_id: str,
        checkpoint_ns: str,
        parent_id: str,
        checkpoints: list[tuple[str, str | None, str | None, Any]],
    ) -> bool:
        """Whether the head's parent must be kept for pending old-format sends.

        LangGraph before checkpoint format v4 stores ``Send`` objects as writes
        on the parent checkpoint.  Dropping that parent would make the kept
        checkpoint unable to resume those sends.  Newer formats do not need it.
        """
        if not any(checkpoint_id == parent_id
                   for checkpoint_id, _, _, _ in checkpoints):
            return False
        try:
            version = int(self.serde.loads_typed((type_tag, checkpoint)).get("v", 1))
        except Exception:
            # If the format cannot be read, keeping one extra checkpoint is
            # safer than deleting a possible pending-sends parent.
            version = 1
        if version >= 4:
            return False
        return await self._has_tasks_write(thread_id, checkpoint_ns, parent_id)

    async def _has_tasks_write(
        self,
        thread_id: str,
        checkpoint_ns: str,
        checkpoint_id: str,
    ) -> bool:
        cursor = await self.conn.execute(
            "SELECT 1 FROM writes "
            "WHERE thread_id = ? AND checkpoint_ns = ? AND checkpoint_id = ? "
            "AND channel = ? LIMIT 1",
            (thread_id, checkpoint_ns, checkpoint_id, _TASKS_CHANNEL),
        )
        return await cursor.fetchone() is not None

    def prune(
        self,
        thread_ids: Sequence[str],
        *,
        strategy: str = "keep_latest",
    ) -> None:
        """Synchronous bridge to :meth:`aprune`.

        As with the other synchronous methods on ``AsyncSqliteSaver``, this
        must not be called from the event-loop thread that owns the
        connection.
        """
        try:
            if asyncio.get_running_loop() is self.loop:
                raise asyncio.InvalidStateError(
                    "Synchronous prune() is only allowed from a different "
                    "thread. From the main thread, use `await "
                    "checkpointer.aprune(...)`."
                )
        except RuntimeError:
            pass
        return asyncio.run_coroutine_threadsafe(
            self.aprune(thread_ids, strategy=strategy), self.loop
        ).result()
