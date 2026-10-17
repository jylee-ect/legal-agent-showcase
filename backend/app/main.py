"""계약서 검토 AI 에이전트 API 서버 (port 7000).

    uvicorn app.main:app --port 7000
"""

from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .agent.documents import load_library, parse_document
from .agent.runner import run_agent
from . import library_admin
from .config import get_settings
from .schemas import Attachment, ChatRequest, CommitRequest, ProposeRequest

app = FastAPI(title="계약서 검토 AI 에이전트", version="1.0")


@app.get("/health")
def health() -> dict:
    s = get_settings()
    return {"status": "ok", "model": s.model, "openai_key": s.has_openai, "law_api": s.has_law_api,
            "reasoning_effort": s.reasoning_effort, "max_tool_calls": s.max_tool_calls}


@app.get("/api/library")
def library() -> dict:
    lib = load_library(get_settings().data_dir)
    return {"index": lib.index_text, "docs": lib.catalog(), "missing": lib.missing}


@app.get("/api/library/{doc_id}")
def library_doc(doc_id: str) -> dict:
    doc = load_library(get_settings().data_dir).docs.get(doc_id)
    if not doc:
        raise HTTPException(404, "없는 문서입니다.")
    return {"id": doc.doc_id, "title": doc.title, "category": doc.category, "path": doc.path,
            "description": doc.description, "outline": doc.outline(),
            "sections": [{"id": s.id, "title": s.title, "text": s.text} for s in doc.sections]}


@app.post("/api/library/propose")
async def library_propose(req: ProposeRequest) -> dict:
    """GPT 모델이 새 문서를 분석하고 llms.txt 수정안을 만든다 (아직 저장하지 않음)."""
    try:
        return await library_admin.propose(get_settings(), req.category, req.filename, req.content, req.note)
    except library_admin.LibraryError as e:
        raise HTTPException(400, str(e))


@app.post("/api/library/commit")
def library_commit(req: CommitRequest) -> dict:
    try:
        return library_admin.commit(get_settings(), req.proposal_id, req.llms_txt)
    except library_admin.LibraryError as e:
        raise HTTPException(400, str(e))


@app.delete("/api/library/{doc_id}")
def library_delete(doc_id: str) -> dict:
    try:
        return library_admin.delete(get_settings(), doc_id)
    except library_admin.LibraryError as e:
        raise HTTPException(400, str(e))


@app.post("/api/attachment/outline")
def attachment_outline(att: Attachment) -> dict:
    """첨부 파일을 에이전트가 보는 방식(조항 목차)으로 나눠 보여준다."""
    doc = parse_document("attachment", att.content, title=att.filename)
    return {"title": doc.title, "outline": doc.outline(), "chars": len(att.content)}


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    async def lines():
        async for event in run_agent(req):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(lines(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
