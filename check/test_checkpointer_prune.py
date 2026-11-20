"""Deterministic tests for ``catagent.app.checkpointer``.

Run from the repository root:

    python -m check.test_checkpointer_prune

No graph or model is needed.  The tests write checkpoints and writes directly
through the saver, then inspect the SQLite tables.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import aiosqlite

from catagent.app.checkpointer import PrunableAsyncSqliteSaver


def _checkpoint(checkpoint_id: str, value: str, *, version: int = 4) -> dict:
    return {
        "v": version,
        "id": checkpoint_id,
        "ts": "2026-11-20T00:00:00+00:00",
        "channel_values": {"messages": value},
        "channel_versions": {"messages": "1"},
        "versions_seen": {},
        "updated_channels": ["messages"],
    }


async def _put(
    saver: PrunableAsyncSqliteSaver,
    thread_id: str,
    checkpoint_id: str,
    *,
    namespace: str = "",
    parent: dict | None = None,
    value: str | None = None,
    version: int = 4,
) -> dict:
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": namespace,
        }
    }
    if parent is not None:
        config["configurable"]["checkpoint_id"] = parent["configurable"][
            "checkpoint_id"
        ]
    return await saver.aput(
        config,
        _checkpoint(checkpoint_id, value or checkpoint_id, version=version),
        {"source": "loop", "step": 0, "parents": {}},
        {"messages": "1"},
    )


async def _write(
    saver: PrunableAsyncSqliteSaver,
    config: dict,
    value: str,
    *,
    channel: str = "messages",
    task_id: str | None = None,
) -> None:
    await saver.aput_writes(
        config,
        [(channel, value)],
        task_id=task_id or f"task-{config['configurable']['checkpoint_id']}",
    )


async def _checkpoint_ids(conn, thread_id: str, namespace: str = "") -> list[str]:
    cur = await conn.execute(
        "SELECT checkpoint_id FROM checkpoints "
        "WHERE thread_id = ? AND checkpoint_ns = ? ORDER BY checkpoint_id",
        (thread_id, namespace),
    )
    return [row[0] for row in await cur.fetchall()]


async def _write_checkpoint_ids(conn, thread_id: str, namespace: str = "") -> set[str]:
    cur = await conn.execute(
        "SELECT DISTINCT checkpoint_id FROM writes "
        "WHERE thread_id = ? AND checkpoint_ns = ?",
        (thread_id, namespace),
    )
    return {row[0] for row in await cur.fetchall()}


async def _parent_id(conn, thread_id: str, checkpoint_id: str) -> str | None:
    cur = await conn.execute(
        "SELECT parent_checkpoint_id FROM checkpoints "
        "WHERE thread_id = ? AND checkpoint_id = ?",
        (thread_id, checkpoint_id),
    )
    row = await cur.fetchone()
    return row[0] if row else None


async def test_keep_latest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await aiosqlite.connect(str(Path(tmp) / "cp.db"))
        saver = PrunableAsyncSqliteSaver(conn)
        try:
            c1 = await _put(saver, "t1", "c1")
            await _write(saver, c1, "v1")
            c2 = await _put(saver, "t1", "c2", parent=c1)
            await _write(saver, c2, "v2")
            c3 = await _put(saver, "t1", "c3", parent=c2)
            await _write(saver, c3, "v3")

            other = await _put(saver, "other", "o1")
            await _write(saver, other, "other-value")

            await saver.aprune(["t1"])
            assert await _checkpoint_ids(conn, "t1") == ["c3"]
            assert await _write_checkpoint_ids(conn, "t1") == {"c3"}
            assert await _parent_id(conn, "t1", "c3") is None

            # The other thread is untouched.
            assert await _checkpoint_ids(conn, "other") == ["o1"]
            assert await _write_checkpoint_ids(conn, "other") == {"o1"}

            state = await saver.aget_tuple(
                {"configurable": {"thread_id": "t1"}}
            )
            assert state is not None
            assert state.checkpoint["id"] == "c3"
            assert state.checkpoint["channel_values"] == {"messages": "c3"}
            assert state.pending_writes == [
                ("task-c3", "messages", "v3")
            ]

            # Pruning is idempotent.
            await saver.aprune(["t1"])
            assert await _checkpoint_ids(conn, "t1") == ["c3"]
        finally:
            await conn.close()


async def test_keep_latest_per_namespace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await aiosqlite.connect(str(Path(tmp) / "cp.db"))
        saver = PrunableAsyncSqliteSaver(conn)
        try:
            root1 = await _put(saver, "t1", "r1")
            await _write(saver, root1, "r1")
            root2 = await _put(saver, "t1", "r2", parent=root1)
            await _write(saver, root2, "r2")

            sub1 = await _put(saver, "t1", "s1", namespace="sub")
            await _write(saver, sub1, "s1")
            sub2 = await _put(
                saver, "t1", "s2", namespace="sub", parent=sub1
            )
            await _write(saver, sub2, "s2")

            await saver.aprune(["t1"])
            assert await _checkpoint_ids(conn, "t1") == ["r2"]
            assert await _checkpoint_ids(conn, "t1", "sub") == ["s2"]
            assert await _write_checkpoint_ids(conn, "t1") == {"r2"}
            assert await _write_checkpoint_ids(conn, "t1", "sub") == {"s2"}
        finally:
            await conn.close()


async def test_old_format_sends_parent_is_kept() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await aiosqlite.connect(str(Path(tmp) / "cp.db"))
        saver = PrunableAsyncSqliteSaver(conn)
        try:
            old = await _put(saver, "t1", "c1", version=3)
            await _write(saver, old, "send", channel="__pregel_tasks")
            head = await _put(saver, "t1", "c2", parent=old, version=3)
            await _write(saver, head, "head")

            await saver.aprune(["t1"])
            # c1 carries the pending Send for c2, so both survive.
            assert await _checkpoint_ids(conn, "t1") == ["c1", "c2"]
        finally:
            await conn.close()


async def test_new_format_does_not_keep_sends_parent() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await aiosqlite.connect(str(Path(tmp) / "cp.db"))
        saver = PrunableAsyncSqliteSaver(conn)
        try:
            old = await _put(saver, "t1", "c1", version=4)
            await _write(saver, old, "send", channel="__pregel_tasks")
            head = await _put(saver, "t1", "c2", parent=old, version=4)
            await _write(saver, head, "head")

            await saver.aprune(["t1"])
            assert await _checkpoint_ids(conn, "t1") == ["c2"]
        finally:
            await conn.close()


async def test_inflight_writes_are_not_deleted() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await aiosqlite.connect(str(Path(tmp) / "cp.db"))
        saver = PrunableAsyncSqliteSaver(conn)
        try:
            head = await _put(saver, "t1", "c1")
            await _write(saver, head, "head")
            # A run may commit writes before its checkpoint row.  Pruning must
            # not delete writes for a checkpoint that is still in flight.
            inflight = {"configurable": {
                "thread_id": "t1", "checkpoint_ns": "", "checkpoint_id": "future"}}
            await _write(saver, inflight, "inflight")

            await saver.aprune(["t1"])
            assert await _write_checkpoint_ids(conn, "t1") == {"c1", "future"}
        finally:
            await conn.close()


async def test_delete_strategy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await aiosqlite.connect(str(Path(tmp) / "cp.db"))
        saver = PrunableAsyncSqliteSaver(conn)
        try:
            c1 = await _put(saver, "t1", "c1")
            await _write(saver, c1, "v1")
            c2 = await _put(saver, "t1", "c2", parent=c1)
            await _write(saver, c2, "v2")
            o1 = await _put(saver, "other", "o1")
            await _write(saver, o1, "other")

            await saver.aprune(["t1"], strategy="delete")
            assert await _checkpoint_ids(conn, "t1") == []
            assert await _write_checkpoint_ids(conn, "t1") == set()
            assert await _checkpoint_ids(conn, "other") == ["o1"]

            await saver.aprune(["t1", "other"], strategy="delete")
            assert await _checkpoint_ids(conn, "other") == []
            assert await _write_checkpoint_ids(conn, "other") == set()
        finally:
            await conn.close()


async def test_validation() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        conn = await aiosqlite.connect(str(Path(tmp) / "cp.db"))
        saver = PrunableAsyncSqliteSaver(conn)
        try:
            for bad in ("t1",):
                try:
                    await saver.aprune(bad)
                except TypeError:
                    pass
                else:
                    raise AssertionError("a single string must raise TypeError")
            try:
                await saver.aprune(["t1"], strategy="unknown")
            except ValueError:
                pass
            else:
                raise AssertionError("an unknown strategy must raise ValueError")
            # Empty input is a no-op.
            await saver.aprune([])
        finally:
            await conn.close()


async def main() -> None:
    await test_keep_latest()
    await test_keep_latest_per_namespace()
    await test_old_format_sends_parent_is_kept()
    await test_new_format_does_not_keep_sends_parent()
    await test_inflight_writes_are_not_deleted()
    await test_delete_strategy()
    await test_validation()
    print("CHECKPOINTER PRUNE TESTS: OK")


if __name__ == "__main__":
    asyncio.run(main())
