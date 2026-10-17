"""에이전트가 쓸 수 있는 도구들.

모든 도구는 `reason`(왜 이 도구를 쓰는지)과 `plan_step`(계획의 몇 번째 단계인지)을 먼저 받는다.
모델이 행동하기 전에 이유를 말하게 하고, 그 이유를 화면에 그대로 보여주기 위해서다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from pydantic import BaseModel, Field

from . import law_api
from .context import RunContext
from .documents import CATEGORY_LABEL, Document, Section
from .web_search import web_search


class ToolArgs(BaseModel):
    reason: str = Field(description="이 도구를 왜 사용하는지 사용자에게 보여줄 한 문장 (쉬운 한국어)")
    plan_step: int = Field(description="이 호출이 해당하는 조사 계획의 단계 번호")


class ReadUploadedArgs(ToolArgs):
    sections: list[str] = Field(
        description="읽을 조항 ID 목록. 예: ['제4조', '제6조']. 계약서 전체를 검토해야 하면 ['전체']"
    )


class DocOutlineArgs(ToolArgs):
    doc_id: str = Field(description="llms.txt 에 있는 문서 ID (파일 이름). 예: labor_standard")


class DocSectionsArgs(ToolArgs):
    doc_id: str = Field(description="llms.txt 에 있는 문서 ID. 예: software_service")
    sections: list[str] = Field(description="읽을 조항·섹션 ID 목록 (목차에 나온 그대로). 예: ['제7조', '제13조'], ['섹션2']")


class QueryArgs(ToolArgs):
    query: str = Field(description="검색어 (한국어)")


class LawArticleArgs(ToolArgs):
    law_name: str = Field(description="법령의 정식 명칭. 예: 근로기준법, 주택임대차보호법, 최저임금법 시행령")
    article: str = Field(description="조문 번호. 예: 제20조, 제3조의2")


class PrecedentArgs(ToolArgs):
    precedent_id: str = Field(description="search_precedents 결과의 판례ID(판례일련번호)")


@dataclass
class ToolOutput:
    text: str  # 모델에게 돌려줄 내용
    summary: str  # 화면용 한 줄 요약
    source_ids: list[str] = field(default_factory=list)
    preview: str = ""  # 화면의 '원문 보기'
    ok: bool = True


@dataclass
class ToolSpec:
    name: str
    label: str  # 비전문가용 이름
    description: str
    args_model: type[ToolArgs]
    run: Callable[[RunContext, ToolArgs], Awaitable[ToolOutput]]
    available: Callable[[RunContext], bool] = lambda ctx: True

    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


def _sections_text(ctx: RunContext, doc: Document, sections: list[Section], kind: str) -> tuple[str, list[str]]:
    blocks, ids = [], []
    for s in sections:
        title = f"{s.id}({s.title})" if s.title else s.id
        if kind == "uploaded":
            src = ctx.registry.add("uploaded", s.id, f"첨부 계약서 '{doc.title}' {title}", s.text,
                                   locator=f"{doc.title} {s.id}")
        else:  # 표준계약서(C#) 또는 회사 참고 문서(K#)
            src = ctx.registry.add(doc.category, f"{doc.doc_id}#{s.id}", f"{doc.title} {title}", s.text,
                                   locator=f"{doc.path}#{s.id}")
        blocks.append(f"[{src.id}] {s.text}")
        ids.append(src.id)
    return "\n\n".join(blocks), ids


def _pick_sections(doc: Document, wanted: list[str]) -> tuple[list[Section], list[str]]:
    if any(w.strip() in ("전체", "all", "ALL", "*") for w in wanted):
        return list(doc.sections), []
    found, missing = [], []
    for w in wanted:
        s = doc.get(w)
        if s and s not in found:
            found.append(s)
        elif not s:
            missing.append(w)
    return found, missing


def _section_names(sections: list[Section]) -> str:
    names = [f"{s.id}({s.title})" if s.title else s.id for s in sections]
    return ", ".join(names[:6]) + (f" 외 {len(names) - 6}개" if len(names) > 6 else "")


# ---------------------------------------------------------------- 첨부 계약서

async def read_uploaded_contract(ctx: RunContext, a: ReadUploadedArgs) -> ToolOutput:
    doc = ctx.attachment
    sections, missing = _pick_sections(doc, a.sections)
    if not sections:
        return ToolOutput(f"해당 조항이 없습니다: {missing}. 목차:\n{doc.outline_text()}",
                          "요청한 조항을 찾지 못함", ok=False)
    text, ids = _sections_text(ctx, doc, sections, "uploaded")
    if missing:
        text += f"\n\n(찾지 못한 조항: {', '.join(missing)})"
    return ToolOutput(text, f"첨부 계약서 {len(sections)}개 조항 읽음: {_section_names(sections)}", ids, text)


# ---------------------------------------------------------------- 참고 문서 라이브러리 (llms.txt)

def _get_doc(ctx: RunContext, doc_id: str) -> Document | None:
    return ctx.library.docs.get(doc_id.strip().removesuffix(".md"))


def _no_doc(ctx: RunContext) -> ToolOutput:
    valid = ", ".join(ctx.library.docs)
    return ToolOutput(f"없는 문서 ID 입니다. 사용 가능: {valid}", "문서 ID 오류", ok=False)


async def read_reference_outline(ctx: RunContext, a: DocOutlineArgs) -> ToolOutput:
    doc = _get_doc(ctx, a.doc_id)
    if not doc:
        return _no_doc(ctx)
    outline = doc.outline_text()
    return ToolOutput(f"{doc.title} ({CATEGORY_LABEL[doc.category]}) 목차:\n{outline}",
                      f"'{doc.title}' 목차 {len(doc.sections)}개 항목 확인", preview=outline)


async def read_reference_sections(ctx: RunContext, a: DocSectionsArgs) -> ToolOutput:
    doc = _get_doc(ctx, a.doc_id)
    if not doc:
        return _no_doc(ctx)
    sections, missing = _pick_sections(doc, a.sections)
    if not sections:
        return ToolOutput(f"해당 조항이 없습니다: {missing}. 목차:\n{doc.outline_text()}",
                          "요청한 조항을 찾지 못함", ok=False)
    text, ids = _sections_text(ctx, doc, sections, doc.category)
    return ToolOutput(text, f"'{doc.title}' {_section_names(sections)} 읽음", ids, text)


# ---------------------------------------------------------------- 국가법령정보센터

async def search_laws(ctx: RunContext, a: QueryArgs) -> ToolOutput:
    laws = await law_api.search_laws(ctx.settings.law_api_oc, a.query)
    if not laws:
        return ToolOutput("검색 결과가 없습니다. 법령 정식 명칭으로 다시 검색하세요.", "검색 결과 없음")
    lines = [f"- {l['name']} ({l['kind']}, {l['ministry']}, 시행 {l['effective']})" for l in laws]
    text = "\n".join(lines)
    return ToolOutput(text, f"법령 {len(laws)}건 찾음: " + ", ".join(l["name"] for l in laws[:3]), preview=text)


async def get_law_article(ctx: RunContext, a: LawArticleArgs) -> ToolOutput:
    r = await law_api.get_article(ctx.settings.law_api_oc, a.law_name, a.article)
    title = f"{r['law_name']} {r['article']}" + (f"({r['title']})" if r["title"] else "")
    src = ctx.registry.add("law", f"{r['law_name']}#{r['article']}", title, r["text"], url=r["url"])
    text = f"[{src.id}] {title} (시행 {r['effective']})\n{r['text']}"
    return ToolOutput(text, f"{title} 원문 확인", [src.id], text)


async def search_precedents(ctx: RunContext, a: QueryArgs) -> ToolOutput:
    items = await law_api.search_precedents(ctx.settings.law_api_oc, a.query)
    if not items:
        return ToolOutput("판례 검색 결과가 없습니다. 더 짧은 핵심 키워드로 다시 검색하세요.", "판례 검색 결과 없음")
    lines = [f"- 판례ID {p['id']} | {p['court']} {p['date']} {p['case_no']} '{p['case_name']}' ({p['kind']})"
             for p in items]
    text = "\n".join(lines) + "\n\n(내용을 확인하려면 read_precedent 로 판례ID 를 읽으세요. 목록만으로는 인용할 수 없습니다.)"
    return ToolOutput(text, f"판례 {len(items)}건 발견", preview="\n".join(lines))


async def read_precedent(ctx: RunContext, a: PrecedentArgs) -> ToolOutput:
    p = await law_api.get_precedent(ctx.settings.law_api_oc, a.precedent_id)
    title = f"{p['court']} {p['date']} 선고 {p['case_no']} 판결 ({p['case_name']})"
    gist = p["summary"] or p["holding"] or p["body_excerpt"][:600]
    parts = []
    for label, key in (("판시사항", "holding"), ("판결요지", "summary"), ("참조조문", "ref_articles")):
        if p[key]:
            parts.append(f"■ {label}\n{p[key]}")
    if not p["summary"]:
        parts.append(f"■ 판결문 발췌\n{p['body_excerpt']}")
    body = "\n\n".join(parts)
    src = ctx.registry.add("precedent", p["id"], title, gist, url=p["url"], content=body)
    text = f"[{src.id}] {title}\n\n{body}"
    return ToolOutput(text, f"{p['court']} {p['case_no']} 판결 읽음", [src.id], text)


# ---------------------------------------------------------------- 웹 검색 (OpenAI 내장)

async def search_web(ctx: RunContext, a: QueryArgs) -> ToolOutput:
    started = time.monotonic()
    r = await web_search(ctx.settings, a.query)
    # 웹검색은 내부적으로 LLM 을 한 번 더 호출하므로 토큰·비용을 따로 기록한다.
    ctx.record_llm_call("research", "웹검색 내부 호출", r.model, r.usage,
                        int((time.monotonic() - started) * 1000), web_search_calls=r.search_calls)
    ids, lines = [], []
    for c in r.citations:
        src = ctx.registry.add("web", c.url, c.title, c.snippet, url=c.url)
        ids.append(src.id)
        lines.append(f"[{src.id}] {c.title} - {c.url}")
    queries = ", ".join(r.queries) or a.query
    text = f"실제 검색어: {queries}\n\n요약:\n{r.summary}\n\n출처:\n" + ("\n".join(lines) or "(인용된 출처 없음)")
    return ToolOutput(text, f"웹 자료 {len(ids)}건 (검색어: {queries})", ids, text)


TOOLS: list[ToolSpec] = [
    ToolSpec(
        "read_uploaded_contract", "📎 첨부 계약서 읽기",
        "사용자가 첨부한 계약서의 조항을 읽는다. 목차는 이미 알고 있으므로 필요한 조항만 고르거나, 전체 검토가 필요하면 ['전체'].",
        ReadUploadedArgs, read_uploaded_contract, available=lambda ctx: ctx.attachment is not None,
    ),
    ToolSpec(
        "read_reference_outline", "📚 참고 문서 목차 보기",
        "llms.txt 에 있는 문서(표준계약서 또는 회사 참고 문서) 하나의 목차를 본다. 어떤 조항을 읽을지 고를 때 쓴다.",
        DocOutlineArgs, read_reference_outline,
    ),
    ToolSpec(
        "read_reference_sections", "📄 참고 문서 조항 읽기",
        "llms.txt 에 있는 문서의 특정 조항·섹션 원문을 읽는다. 표준계약서는 비교 기준으로, 회사 참고 문서는 회사 내부 기준으로 쓴다.",
        DocSectionsArgs, read_reference_sections,
    ),
    ToolSpec(
        "search_laws", "🏛️ 법령 찾기",
        "국가법령정보센터에서 법령 이름을 검색한다. 정확한 법령 명칭을 모를 때만 쓴다.",
        QueryArgs, search_laws, available=lambda ctx: ctx.settings.has_law_api,
    ),
    ToolSpec(
        "get_law_article", "📜 법조문 원문 확인",
        "국가법령정보센터에서 현행 법령의 특정 조문 원문을 가져온다. 법령을 근거로 들 때는 반드시 이 도구로 원문을 확인한다.",
        LawArticleArgs, get_law_article, available=lambda ctx: ctx.settings.has_law_api,
    ),
    ToolSpec(
        "search_precedents", "⚖️ 판례 검색",
        "국가법령정보센터에서 판례를 키워드로 검색한다(판결문 본문 검색). 2~4개의 핵심 단어로 검색한다. 예: '위약금 교육비 근로자'",
        QueryArgs, search_precedents, available=lambda ctx: ctx.settings.has_law_api,
    ),
    ToolSpec(
        "read_precedent", "📖 판례 읽기",
        "search_precedents 로 찾은 판례의 판시사항·판결요지를 읽는다. 판례를 인용하려면 반드시 읽어야 한다.",
        PrecedentArgs, read_precedent, available=lambda ctx: ctx.settings.has_law_api,
    ),
    ToolSpec(
        "web_search", "🌐 웹 검색",
        "웹에서 정부 기관 해석, 최신 제도 변경, 실무 가이드 등을 검색한다. 법령·판례 원문은 법령 도구를 우선 사용한다.",
        QueryArgs, search_web,
    ),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def available_tools(ctx: RunContext) -> list[ToolSpec]:
    return [t for t in TOOLS if t.available(ctx)]
