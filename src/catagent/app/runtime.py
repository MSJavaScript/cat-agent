"""Runtime construction: model, checkpoint database and compiled graph."""

from __future__ import annotations

import os
import sqlite3
from typing import Literal

import aiosqlite
import dotenv
import httpx

from ..agent import create_catagent
from .checkpointer import PrunableAsyncSqliteSaver
from .skills import load_skill_index


def create_llm_model(
    model_type: Literal["glm", "deepseek", "qwen"],
    model_name: str,
    base_url: str,
    api_key: str,
    **kwargs,
):
    """Create the LangChain chat model selected by ``model_type``."""
    if model_type == "glm":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    if model_type == "deepseek":
        from langchain_deepseek import ChatDeepSeek

        return ChatDeepSeek(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    if model_type == "qwen":
        from langchain_qwq import ChatQwen

        return ChatQwen(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
    raise ValueError(f"Unsupported model type {model_type!r}.")


def _infer_model_type(model_name: str) -> Literal["glm", "deepseek", "qwen"]:
    lowered = model_name.lower()
    if "qwen" in lowered:
        return "qwen"
    if "glm" in lowered:
        return "glm"
    if "deepseek" in lowered:
        return "deepseek"
    raise ValueError(f"Cannot infer model type from MODEL_NAME={model_name!r}.")


def _ensure_parent_dir(path: str) -> None:
    """Create the directory that will contain ``path`` when it is missing.

    ``sqlite3`` creates the database file on demand but not intermediate
    directories, so a fresh deployment whose ``--db`` points at a new folder
    fails with the misleading ``unable to open database file``.  ``:memory:``
    and SQLite URI names have no filesystem parent to create.
    """
    if not path or path == ":memory:" or path.startswith("file:"):
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


async def build_runtime(
    dotenv_path: str,
    db_path: str,
    job_manager_url: str,
    skill_index_json: str | None = None,
):
    """Create the model, checkpointer and compiled workflow.

    This is shared by the HTTP service and the debug CLI.
    ``skill_index_json`` is loaded before the graph is built, so every new
    thread starts with those skills in ``loaded_skills``.
    The caller owns the returned httpx client and SQLite connection and must
    close them when done.
    """
    loaded_skills = load_skill_index(skill_index_json) if skill_index_json else {}
    httpx_client = httpx.AsyncClient(timeout=30)
    db_conn = None
    try:
        dotenv.load_dotenv(dotenv_path)
        model_name = os.getenv("MODEL_NAME")
        if not model_name:
            raise RuntimeError("MODEL_NAME is not set (check the dotenv file).")

        model = create_llm_model(
            model_type=_infer_model_type(model_name),
            model_name=model_name,
            base_url=os.getenv("BASE_URL"),
            api_key=os.getenv("API_KEY"),
        )

        _ensure_parent_dir(db_path)
        try:
            db_conn = await aiosqlite.connect(db_path, isolation_level=None)
        except sqlite3.OperationalError as exc:
            # Preserve the original exception (and its type) for callers, but
            # leave a note in the traceback that points at the common causes:
            # a missing/unwritable parent directory, or another process that
            # still has the old file open (which on Windows/WSL blocks
            # recreating a deleted database under the same name).
            exc.add_note(
                f"Checkpoint database: {db_path!r}. Ensure its parent "
                "directory exists and is writable, and that no other process "
                "still holds the file open."
            )
            raise
        checkpointer = PrunableAsyncSqliteSaver(db_conn)

        create_kwargs = {
            "model": model,
            "job_manager_url": job_manager_url,
            "http_client": httpx_client,
            "checkpointer": checkpointer,
        }
        if loaded_skills:
            create_kwargs["loaded_skills"] = loaded_skills
        catagent = create_catagent(**create_kwargs)
        return catagent, httpx_client, db_conn
    except BaseException:
        if db_conn is not None:
            await db_conn.close()
        await httpx_client.aclose()
        raise
