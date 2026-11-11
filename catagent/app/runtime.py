"""Runtime construction: model, checkpoint database and compiled graph."""

from __future__ import annotations

import os
from typing import Literal

import aiosqlite
import dotenv
import httpx
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from ..agent import create_catagent
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

        db_conn = await aiosqlite.connect(db_path, isolation_level=None)
        checkpointer = AsyncSqliteSaver(db_conn)

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
