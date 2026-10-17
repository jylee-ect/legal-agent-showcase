"""OpenAI 내장 웹검색(web_search tool) 래퍼.

에이전트 입장에서는 하나의 '도구'로 보이게 하려고 Responses API 를 별도로 호출한다.
그래야 화면에 검색 단계가 따로 보이고, url_citation 을 출처 레지스트리에 넣을 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import AsyncOpenAI

from ..config import Settings
from .usage import Usage, from_openai

SEARCH_INSTRUCTIONS = (
    "당신은 한국 법률 리서치 보조원입니다. 웹 검색으로 아래 질문에 관한 근거를 찾아 한국어로 5~8문장 요약하세요. "
    "법령, 판례, 정부 기관(고용노동부, 법무부, 국토교통부 등)의 해석, 공신력 있는 기관 자료를 우선하고, "
    "개인 블로그는 가급적 피하세요. 확인되지 않은 내용은 추측하지 마세요."
)


@dataclass
class WebCitation:
    url: str
    title: str
    snippet: str


@dataclass
class WebSearchResult:
    summary: str
    queries: list[str] = field(default_factory=list)
    citations: list[WebCitation] = field(default_factory=list)
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    search_calls: int = 0  # 웹검색 도구 호출 수 (호출당 별도 과금)


async def web_search(settings: Settings, query: str) -> WebSearchResult:
    client = AsyncOpenAI(api_key=settings.openai_api_key)
    kwargs = {}
    if settings.reasoning_effort:
        kwargs["reasoning"] = {"effort": settings.reasoning_effort}
    resp = await client.responses.create(
        model=settings.web_search_model,
        instructions=SEARCH_INSTRUCTIONS,
        input=query,
        tools=[{"type": "web_search", "user_location": {"type": "approximate", "country": "KR"}}],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        **kwargs,
    )

    result = WebSearchResult(summary=resp.output_text or "", model=resp.model or settings.web_search_model,
                             usage=from_openai(resp.usage))
    seen: set[str] = set()
    for item in resp.output:
        if item.type == "web_search_call":
            result.search_calls += 1
            action = getattr(item, "action", None)
            if action is not None and getattr(action, "type", "") == "search":
                result.queries += [q for q in (action.queries or [action.query]) if q]
        elif item.type == "message":
            for part in item.content:
                if getattr(part, "type", "") != "output_text":
                    continue
                for ann in part.annotations or []:
                    if getattr(ann, "type", "") != "url_citation" or ann.url in seen:
                        continue
                    seen.add(ann.url)
                    result.citations.append(WebCitation(ann.url, ann.title or ann.url,
                                                        _sentence_before(part.text, ann.start_index)))
    return result


def _sentence_before(text: str, index: int | None, limit: int = 220) -> str:
    """url_citation 의 인덱스는 삽입된 링크 자체를 가리키므로, 그 앞 문장을 근거 발췌로 쓴다."""
    if not index:
        return ""
    before = text[:index].rstrip(" (（")
    line = before.rsplit("\n", 1)[-1].strip()
    return line[-limit:]
