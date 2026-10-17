"""문서 라이브러리 관리: 표준계약서·회사 참고 문서 등록과 llms.txt 갱신.

등록 흐름 (2단계):
1. propose: 설정된 GPT 모델이 새 문서를 읽고 ID·제목·설명을 정한 뒤, 전체 llms.txt 를 다시 쓴다
   (겹치는 기존 문서의 설명을 다듬을 수 있다). 코드가 결과를 검증·보정하고 diff 를 만든다.
   아직 파일은 바꾸지 않는다.
2. commit: 운영자가 확인(필요하면 직접 수정)한 llms.txt 를 검증한 뒤 문서 파일과 함께 저장한다.
   이전 llms.txt 는 data/.history/ 에 백업한다.

모델이 기존 문서를 빠뜨리거나 경로를 망가뜨리지 못하게, 경로는 코드가 doc_id 와 분류로 만든다.
"""

from __future__ import annotations

import difflib
import re
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .agent import graph as agent_graph
from .agent.documents import (CATEGORY_DIRS, CATEGORY_LABEL, LLMS_ENTRY_RE, category_of, load_library,
                              parse_document, parse_llms_entries)
from .agent.usage import cost_usd, from_usage_metadata, load_pricing
from .config import Settings

LLMS_TITLE = "# 계약 검토 참고 문서 라이브러리"
MAX_DOC_CHARS_FOR_LLM = 12_000
PROPOSAL_TTL_SEC = 3600

_lock = threading.Lock()
_proposals: dict[str, dict] = {}


class LibraryError(ValueError):
    pass


# ---------------------------------------------------------------- 모델 출력 스키마

class Entry(BaseModel):
    doc_id: str = Field(description="영문 소문자·숫자·밑줄로 된 짧은 ID (파일 이름이 됨)")
    title: str = Field(description="문서의 정식 제목")
    description: str = Field(description="어떤 질문에 이 문서를 열어야 하는지: 핵심 주제를 쉼표로 나열하고 '~에 관한 질문'으로 끝냄")


class LlmsTxtDraft(BaseModel):
    new_doc: Entry
    intro: str = Field(description="llms.txt 머리말 2~3문장 (라이브러리 구성과 사용법)")
    standard: list[Entry] = Field(description="'표준계약서' 분류의 전체 문서 목록 (순서대로)")
    company: list[Entry] = Field(description="'회사 참고 문서' 분류의 전체 문서 목록 (순서대로)")
    change_summary: str = Field(description="무엇을 추가·수정했고 왜 그렇게 했는지 2~4문장")


LIBRARIAN_SYSTEM = """당신은 '계약서 검토 AI 에이전트'가 쓰는 문서 라이브러리의 사서입니다.
llms.txt 는 에이전트가 질문을 받을 때마다 가장 먼저 읽고, 어떤 문서를 열어볼지 결정하는 목차입니다.
새 문서를 등록하면서 llms.txt 를 갱신하세요.

1. new_doc: 새 문서를 읽고 작성합니다.
   - doc_id: 내용을 나타내는 짧은 영문 소문자 ID (예: contract_review_checklist). 기존 ID 와 겹치면 안 됩니다.
   - title: 문서의 정식 제목.
   - description: 에이전트가 '이 질문이면 이 문서를 열어야겠다'고 판단할 수 있도록 다루는 주제를 쉼표로 나열하고,
     기존 항목과 같은 문체로 '~에 관한 질문'으로 끝내세요. 150자 이내.
2. standard / company: 분류별 전체 목록을 다시 작성합니다.
   - 기존 문서는 하나도 빠뜨리지 말고 doc_id 와 title 을 그대로 두세요. 분류도 바꾸지 마세요.
   - 새 문서는 지정된 분류에 넣으세요.
   - 새 문서와 주제가 겹치는 기존 문서가 있으면, 에이전트가 둘을 구분할 수 있도록 설명을 꼭 필요한 만큼만 다듬을 수 있습니다.
3. intro: 라이브러리 전체를 설명하는 머리말 2~3문장. 분류 구성(표준계약서는 시연용 모의 문서, 회사 참고 문서는 회사 내부 기준)을 반영하세요.
4. change_summary: 무엇을 추가·수정했고 왜 그렇게 했는지 비전문가도 이해할 수 있게 2~4문장.
"""


# ---------------------------------------------------------------- 보조 함수

def _slug(raw: str) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    return s[:48] or "doc"


def _unique_id(base: str, taken: set[str]) -> str:
    if base not in taken:
        return base
    n = 2
    while f"{base}_{n}" in taken:
        n += 1
    return f"{base}_{n}"


def _one_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _entry_line(category: str, doc_id: str, title: str, desc: str) -> str:
    title = _one_line(title).replace("[", "(").replace("]", ")")
    return f"- [{title}]({CATEGORY_DIRS[category]}/{doc_id}.md): {_one_line(desc)}"


def render_llms_txt(intro: str, by_category: dict[str, list[tuple[str, str, str]]]) -> str:
    lines = [LLMS_TITLE, ""]
    lines += [f"> {_one_line(l)}" for l in re.split(r"\n+", intro.strip()) if l.strip()] or ["> 참고 문서 목록입니다."]
    for category in ("standard", "company"):
        lines += ["", f"## {CATEGORY_LABEL[category]}", ""]
        entries = by_category.get(category, [])
        lines += [_entry_line(category, *e) for e in entries] or ["_(등록된 문서 없음)_"]
    return "\n".join(lines) + "\n"


def _existing(data_dir: Path) -> dict[str, dict]:
    """doc_id → {category, title, description, path}"""
    index = (data_dir / "llms.txt").read_text(encoding="utf-8")
    out = {}
    for title, rel_path, desc in parse_llms_entries(index):
        out[Path(rel_path).stem] = {"category": category_of(rel_path), "title": title, "description": desc,
                                    "path": rel_path}
    return out


def assemble(draft: LlmsTxtDraft, existing: dict[str, dict], new_id: str, category: str) -> tuple[str, list[str]]:
    """모델 초안을 검증·보정해 llms.txt 텍스트를 만든다. (텍스트, 보정 내역)"""
    warnings: list[str] = []
    by_cat: dict[str, list[tuple[str, str, str]]] = {"standard": [], "company": []}
    placed: set[str] = set()
    model_new_id = draft.new_doc.doc_id.strip()

    for listed_cat, entries in (("standard", draft.standard), ("company", draft.company)):
        for e in entries:
            eid = e.doc_id.strip().removesuffix(".md")
            if eid in (model_new_id, new_id) and new_id not in placed:
                by_cat[category].append((new_id, draft.new_doc.title, e.description or draft.new_doc.description))
                placed.add(new_id)
            elif eid in existing and eid not in placed:
                old = existing[eid]
                if _one_line(e.title) != _one_line(old["title"]):
                    warnings.append(f"'{old['title']}'의 제목을 모델이 바꾸려 해서 원래 제목을 유지했습니다.")
                if listed_cat != old["category"]:
                    warnings.append(f"'{old['title']}'의 분류를 모델이 바꾸려 해서 원래 분류를 유지했습니다.")
                by_cat[old["category"]].append((eid, old["title"], e.description or old["description"]))
                placed.add(eid)
            elif eid not in placed:
                warnings.append(f"목록에 없는 문서 ID '{eid}'를 모델이 추가하려 해서 제외했습니다.")

    for eid, old in existing.items():
        if eid not in placed:
            warnings.append(f"모델이 빠뜨린 기존 문서 '{old['title']}'를 원래 설명으로 다시 넣었습니다.")
            by_cat[old["category"]].append((eid, old["title"], old["description"]))
    if new_id not in placed:
        warnings.append("모델이 새 문서를 목록에 넣지 않아 자동으로 추가했습니다.")
        by_cat[category].append((new_id, draft.new_doc.title, draft.new_doc.description))
    return render_llms_txt(draft.intro, by_cat), warnings


def _diff(before: str, after: str) -> str:
    return "\n".join(difflib.unified_diff(before.splitlines(), after.splitlines(), "llms.txt (현재)",
                                          "llms.txt (수정안)", lineterm=""))


def _cleanup() -> None:
    now = time.time()
    for pid in [p for p, v in _proposals.items() if now - v["created"] > PROPOSAL_TTL_SEC]:
        _proposals.pop(pid, None)


# ---------------------------------------------------------------- 1단계: 수정안 만들기

async def propose(settings: Settings, category: str, filename: str, content: str, note: str = "") -> dict:
    if category not in CATEGORY_DIRS:
        raise LibraryError("분류는 standard(표준계약서) 또는 company(회사 참고 문서)여야 합니다.")
    if not content.strip():
        raise LibraryError("빈 문서입니다.")
    if not settings.has_openai:
        raise LibraryError("OpenAI API 키가 없습니다. 환경변수 GPT_API 를 설정한 뒤 백엔드를 다시 실행하세요.")

    data_dir = settings.data_dir
    before = (data_dir / "llms.txt").read_text(encoding="utf-8")
    existing = _existing(data_dir)
    parsed = parse_document("new", content, title=Path(filename).stem)

    excerpt = content[:MAX_DOC_CHARS_FOR_LLM]
    truncated = len(content) > MAX_DOC_CHARS_FOR_LLM
    listing = "\n".join(f"- {i} ({CATEGORY_LABEL[v['category']]}): {v['title']}" for i, v in existing.items())
    user = (
        f"[새 문서 분류]\n{CATEGORY_LABEL[category]}\n\n"
        f"[현재 llms.txt]\n{before}\n\n"
        f"[기존 문서 ID]\n{listing or '(없음)'}\n\n"
        f"[새 문서 파일명]\n{filename}\n\n"
        f"[운영자 메모]\n{note.strip() or '(없음)'}\n\n"
        f"[새 문서 목차]\n{parsed.outline_text()}\n\n"
        f"[새 문서 본문{' (앞부분만)' if truncated else ''}]\n{excerpt}"
    )

    llm = agent_graph.llm_factory(settings).with_structured_output(LlmsTxtDraft, include_raw=True)
    started = time.monotonic()
    out = await llm.ainvoke([SystemMessage(LIBRARIAN_SYSTEM), HumanMessage(user)])
    elapsed_ms = int((time.monotonic() - started) * 1000)
    draft: LlmsTxtDraft | None = out.get("parsed")
    if draft is None:
        raise LibraryError(f"모델 출력 해석 실패: {out.get('parsing_error')}")

    raw = out.get("raw")
    meta = getattr(raw, "response_metadata", None) or {}
    model = meta.get("model_name") or settings.model
    usage = from_usage_metadata(getattr(raw, "usage_metadata", None))

    new_id = _unique_id(_slug(draft.new_doc.doc_id), set(existing))
    after, warnings = assemble(draft, existing, new_id, category)
    if new_id != draft.new_doc.doc_id:
        warnings.insert(0, f"모델이 제안한 ID '{draft.new_doc.doc_id}'를 규칙에 맞게 '{new_id}'로 바꿨습니다.")

    structure = "조항(제N조)" if parsed.sections and parsed.sections[0].id.startswith("제") else (
        "제목(##)" if parsed.sections and parsed.sections[0].id.startswith("섹션") else "문단 묶음")
    _cleanup()
    proposal_id = uuid.uuid4().hex
    rel_path = f"{CATEGORY_DIRS[category]}/{new_id}.md"
    _proposals[proposal_id] = {"created": time.time(), "rel_path": rel_path, "content": content,
                               "llms_before": before}
    return {
        "proposal_id": proposal_id,
        "doc": {"id": new_id, "title": _one_line(draft.new_doc.title), "description": draft.new_doc.description,
                "category": category, "category_label": CATEGORY_LABEL[category], "path": rel_path},
        "outline": parsed.outline(),
        "structure": structure,
        "truncated_for_llm": truncated,
        "change_summary": draft.change_summary,
        "warnings": warnings,
        "llms_before": before,
        "llms_after": after,
        "diff": _diff(before, after),
        "model": model,
        "usage": usage.to_dict(),
        "cost_usd": cost_usd(pricing := load_pricing(), model, usage),
        "usd_krw": pricing.get("usd_krw"),
        "elapsed_ms": elapsed_ms,
    }


# ---------------------------------------------------------------- 2단계: 반영

def validate_llms_txt(text: str, data_dir: Path, new_rel_path: str | None = None) -> list[str]:
    """반영 전 검사. 문제 목록을 돌려준다 (비어 있으면 통과)."""
    problems = []
    entries = parse_llms_entries(text)
    paths = [p for _, p, _ in entries]
    if not entries:
        problems.append("문서 항목이 하나도 없습니다. '- [제목](경로): 설명' 형식을 확인하세요.")
    for p in sorted({p for p in paths if paths.count(p) > 1}):
        problems.append(f"같은 문서가 두 번 등록되어 있습니다: {p}")
    for p in paths:
        if p.split("/")[0] not in CATEGORY_DIRS.values() or not p.endswith(".md"):
            problems.append(f"경로는 contracts/ 또는 company/ 아래 .md 여야 합니다: {p}")
        elif p != new_rel_path and not (data_dir / p).exists():
            problems.append(f"파일이 없는 경로입니다: {p}")
    if new_rel_path and new_rel_path not in paths:
        problems.append(f"새 문서({new_rel_path})가 목록에 없습니다.")
    for line in text.splitlines():
        if line.lstrip().startswith("- [") and not LLMS_ENTRY_RE.match(line.strip()):
            problems.append(f"형식이 잘못된 줄입니다: {line.strip()[:80]}")
    return problems


def _backup(data_dir: Path, tag: str) -> Path:
    hist = data_dir / ".history"
    hist.mkdir(exist_ok=True)
    dest = hist / f"llms_{datetime.now():%Y%m%d_%H%M%S}_{tag}.txt"
    shutil.copy2(data_dir / "llms.txt", dest)
    return dest


def commit(settings: Settings, proposal_id: str, llms_txt: str) -> dict:
    data_dir = settings.data_dir
    with _lock:
        prop = _proposals.get(proposal_id)
        if not prop:
            raise LibraryError("수정안이 없거나 만료되었습니다. 다시 분석해 주세요.")
        current = (data_dir / "llms.txt").read_text(encoding="utf-8")
        if current != prop["llms_before"]:
            raise LibraryError("그 사이 llms.txt 가 바뀌었습니다. 다시 분석해 주세요.")
        problems = validate_llms_txt(llms_txt, data_dir, prop["rel_path"])
        if problems:
            raise LibraryError("llms.txt 검사 실패: " + " / ".join(problems))

        doc_path = data_dir / prop["rel_path"]
        if doc_path.exists():
            raise LibraryError(f"이미 같은 이름의 파일이 있습니다: {prop['rel_path']}")
        backup = _backup(data_dir, "before_add")
        doc_path.parent.mkdir(parents=True, exist_ok=True)
        doc_path.write_text(prop["content"].replace("\r\n", "\n"), encoding="utf-8")
        (data_dir / "llms.txt").write_text(llms_txt.replace("\r\n", "\n"), encoding="utf-8")
        _proposals.pop(proposal_id, None)
        load_library.cache_clear()
    return {"saved": prop["rel_path"], "backup": backup.name}


def delete(settings: Settings, doc_id: str) -> dict:
    """문서 삭제: 해당 줄만 코드로 지우고, 파일은 .history/deleted/ 로 옮긴다 (되돌릴 수 있게)."""
    data_dir = settings.data_dir
    with _lock:
        index = (data_dir / "llms.txt").read_text(encoding="utf-8")
        entry = next(((t, p, d) for t, p, d in parse_llms_entries(index) if Path(p).stem == doc_id), None)
        if not entry:
            raise LibraryError(f"llms.txt 에 없는 문서입니다: {doc_id}")
        rel_path = entry[1]
        lines = [l for l in index.splitlines() if not (LLMS_ENTRY_RE.match(l.strip()) and f"]({rel_path})" in l)]
        new_index = "\n".join(lines) + "\n"
        category_header = f"## {CATEGORY_LABEL[category_of(rel_path)]}"
        # 분류에 남은 문서가 없으면 빈 표시를 넣는다
        sections = re.split(r"(?m)^(## .+)$", new_index)
        for i, part in enumerate(sections):
            if part.strip() == category_header and i + 1 < len(sections) and not sections[i + 1].strip():
                sections[i + 1] = "\n\n_(등록된 문서 없음)_\n\n"
        new_index = "".join(sections).rstrip() + "\n"

        backup = _backup(data_dir, f"before_delete_{doc_id}")
        src = data_dir / rel_path
        if src.exists():
            dest_dir = data_dir / ".history" / "deleted"
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dest_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{src.name}")
        (data_dir / "llms.txt").write_text(new_index, encoding="utf-8")
        load_library.cache_clear()
    return {"deleted": rel_path, "backup": backup.name}
