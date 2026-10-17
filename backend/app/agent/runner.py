"""그래프를 백그라운드로 실행하면서 이벤트를 순서대로 내보낸다."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..config import get_settings
from ..schemas import ChatRequest
from . import graph as agent_graph
from .context import RunContext
from .documents import load_library, parse_document
from .usage import pricing_summary


async def run_agent(req: ChatRequest) -> AsyncIterator[dict]:
    settings = get_settings()
    attachment = parse_document("attachment", req.attachment.content, title=req.attachment.filename) \
        if req.attachment else None
    ctx = RunContext(settings, load_library(settings.data_dir), attachment, asyncio.Queue())

    ctx.emit("run_start", question=req.question, model=settings.model, law_api=settings.has_law_api,
             attachment=attachment.title if attachment else None,
             pricing=pricing_summary(ctx.pricing, settings.model))

    async def _run() -> None:
        try:
            if not settings.has_openai:
                raise RuntimeError("OpenAI API 키가 없습니다. 백엔드를 실행한 터미널에 환경변수 GPT_API 를 설정한 뒤 백엔드를 다시 실행하세요.")
            state = {"question": req.question, "history": [h.model_dump() for h in req.history]}
            await agent_graph.GRAPH.ainvoke(state, config={"configurable": {"ctx": ctx}, "recursion_limit": 60})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            ctx.emit("error", message=f"{type(e).__name__}: {e}")
        finally:
            ctx.emit("done", elapsed_ms=ctx.elapsed_ms())

    task = asyncio.create_task(_run())
    try:
        while True:
            event = await ctx.queue.get()
            yield event
            if event["type"] == "done":
                break
    finally:
        if not task.done():  # 클라이언트가 연결을 끊으면 실행도 멈춘다
            task.cancel()
