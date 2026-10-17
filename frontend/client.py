"""백엔드(FastAPI :7000) 호출과 녹화 파일 관리."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterator

import httpx
import streamlit as st

# localhost 는 IPv6(::1)로 먼저 연결을 시도해 느려질 수 있어 127.0.0.1 을 쓴다 (서버는 IPv4 0.0.0.0 에서 실행).
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:7000").rstrip("/")
ROOT_DIR = Path(__file__).resolve().parents[1]
RECORDINGS_DIR = ROOT_DIR / "recordings"
SAMPLES_DIR = ROOT_DIR / "samples"


def _get(path: str) -> dict | None:
    try:
        r = httpx.get(BACKEND_URL + path, timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


def health() -> dict | None:
    return _get("/health")


def library() -> dict | None:
    return _get("/api/library")


@st.cache_data(ttl=30, show_spinner=False)
def cached_library() -> dict | None:
    """페이지마다 다시 부르지 않도록 30초 캐시. 문서를 등록·삭제하면 cached_library.clear() 로 비운다."""
    return library()


def library_doc(doc_id: str) -> dict | None:
    return _get(f"/api/library/{doc_id}")


class ApiError(RuntimeError):
    pass


def _send(method: str, path: str, timeout: float, **kwargs) -> dict:
    try:
        r = httpx.request(method, BACKEND_URL + path, timeout=timeout, **kwargs)
    except httpx.HTTPError as e:
        raise ApiError(f"백엔드({BACKEND_URL})에 연결할 수 없습니다: {e}") from e
    if r.status_code != 200:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text[:300]
        raise ApiError(str(detail))
    return r.json()


def propose_document(category: str, filename: str, content: str, note: str) -> dict:
    """GPT 모델이 문서를 분석하고 llms.txt 수정안을 만든다 (저장 전)."""
    return _send("POST", "/api/library/propose", 180,
                 json={"category": category, "filename": filename, "content": content, "note": note})


def commit_document(proposal_id: str, llms_txt: str) -> dict:
    return _send("POST", "/api/library/commit", 30, json={"proposal_id": proposal_id, "llms_txt": llms_txt})


def delete_document(doc_id: str) -> dict:
    return _send("DELETE", f"/api/library/{doc_id}", 30)


def attachment_outline(filename: str, content: str) -> dict | None:
    try:
        r = httpx.post(BACKEND_URL + "/api/attachment/outline", json={"filename": filename, "content": content},
                       timeout=10)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None


def _local_event(type_: str, **data) -> dict:
    """백엔드에 닿지 못했을 때 화면에서 만드는 이벤트 (로그 형식을 맞추려고 ts 포함)."""
    return {"type": type_, "t": 0, "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"), **data}


def stream_chat(question: str, history: list[dict], attachment: dict | None) -> Iterator[dict]:
    payload = {"question": question, "history": history,
               "attachment": {"filename": attachment["filename"], "content": attachment["content"]}
               if attachment else None}
    try:
        with httpx.stream("POST", BACKEND_URL + "/api/chat/stream", json=payload,
                          timeout=httpx.Timeout(10.0, read=300.0)) as r:
            if r.status_code != 200:
                r.read()
                yield _local_event("error", message=f"백엔드 오류 {r.status_code}: {r.text[:300]}")
                yield _local_event("done")
                return
            for line in r.iter_lines():
                if line.strip():
                    yield json.loads(line)
    except httpx.HTTPError as e:
        yield _local_event("error", message=f"백엔드({BACKEND_URL})에 연결할 수 없습니다: {e}")
        yield _local_event("done")


def decode_text(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def list_samples() -> list[Path]:
    return sorted(p for p in SAMPLES_DIR.glob("*") if p.suffix in (".md", ".txt"))


# ---------------------------------------------------------------- 녹화 / 재생

def save_recording(events: list[dict]) -> Path:
    RECORDINGS_DIR.mkdir(exist_ok=True)
    start = next((e for e in events if e.get("type") == "run_start"), {})
    slug = re.sub(r"[^\w가-힣]+", "_", start.get("question", "run"))[:30].strip("_")
    path = RECORDINGS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return path


def list_recordings() -> list[Path]:
    if not RECORDINGS_DIR.exists():
        return []
    return sorted(RECORDINGS_DIR.glob("*.jsonl"), reverse=True)


def load_recording(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
