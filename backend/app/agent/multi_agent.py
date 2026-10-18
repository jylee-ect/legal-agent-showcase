"""멀티 에이전트 — 팀장(supervisor) + 전문가 4명, 모두 LangChain `create_agent` 기반 ReAct.

    트리거 → 🧑‍💼 팀장 ⇄ ask_*(지시) ─┬─ 📜 법령조사 에이전트   (법령 API, 웹검색)
                                    ├─ ⚖️ 판례조사 에이전트   (판례 API)
                                    ├─ 📑 계약서분석 에이전트 (첨부 계약서, 표준계약서)
                                    └─ 🏢 회사기준 에이전트   (회사 참고 문서)
            팀장: 보고서를 모아 최종 답변 → 근거 검증(VerifyMiddleware, 에이전트 안) → 최종 안전장치(에이전트 밖)

- '에이전트를 도구로' 패턴: 팀장에게 전문가는 ask_<이름>(지시) 도구 하나다. 그 도구 안에서 전문가 에이전트가
  자기 프롬프트·도구·한도로 ReAct 루프를 돌고, 보고서를 도구 결과로 돌려준다.
- 팀장이 ask_* 를 한 번에 여러 개 부르면 전문가들이 동시에(병렬로) 일한다.
- 출처 레지스트리는 공유한다. 전문가가 읽은 자료의 [L1], [P1] 을 팀장이 그대로 인용하고, 근거 검증도 그대로 동작한다.
- 모든 이벤트에 agent(누가) / run(어느 지시에 대한 작업인지) 이름표가 붙어 화면에서 에이전트별로 보인다.
"""

from __future__ import annotations

import asyncio
import contextvars
import time

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware, TodoListMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from . import graph as agent_graph
from . import prompts
from .context import CURRENT_AGENT, RunContext
from .react_agent import (_CURRENT_CALL, ObserverMiddleware, VerifyMiddleware, _as_langchain_tool,
                          stream_agent)
from .tools import available_tools

SUPERVISOR_MODEL_CALLS = 12
SPECIALIST_MODEL_CALLS = 6
SPECIALIST_TOOL_LIMIT = 5  # 전문가 한 번 실행에 쓸 수 있는 도구 수

SPECIALISTS = {
    "law": {"name": "법령조사 에이전트", "icon": "📜", "tools": ["search_laws", "get_law_article", "web_search"]},
    "precedent": {"name": "판례조사 에이전트", "icon": "⚖️", "tools": ["search_precedents", "read_precedent"]},
    "contract": {"name": "계약서분석 에이전트", "icon": "📑",
                 "tools": ["read_uploaded_contract", "read_reference_outline", "read_reference_sections"]},
    "company": {"name": "회사기준 에이전트", "icon": "🏢", "tools": ["read_reference_outline", "read_reference_sections"]},
}
MIDDLEWARE_INFO = [
    {"name": "ObserverMiddleware", "desc": "팀장·전문가 모두에 붙어 모델·도구 호출과 지시·보고를 화면에 보여줌"},
    {"name": "VerifyMiddleware", "desc": "팀장의 최종 답변 시도를 근거 검증하고, 문제가 있으면 팀장에게 되돌림"},
    {"name": "TodoListMiddleware", "desc": "팀장의 업무 분배 계획 (write_todos)"},
    {"name": "ModelCallLimitMiddleware",
     "desc": f"팀장 {SUPERVISOR_MODEL_CALLS}회 · 전문가 {SPECIALIST_MODEL_CALLS}회 모델 호출 한도"},
]


class AskArgs(BaseModel):
    reason: str = Field(description="왜 이 전문가에게 맡기는지 사용자에게 보여줄 한 문장 (쉬운 한국어)")
    task: str = Field(description="전문가에게 줄 구체적인 지시: 무엇을 확인해 무엇을 보고할지, 관련 조항·쟁점")


def _specialist_tools(ctx: RunContext, key: str) -> list:
    allowed = SPECIALISTS[key]["tools"]
    specs = [t for t in available_tools(ctx) if t.name in allowed]
    if key == "precedent" and not specs:  # 법령 API 키가 없으면 웹검색으로 대신한다
        specs = [t for t in available_tools(ctx) if t.name == "web_search"]
    return specs


def _company_docs(ctx: RunContext) -> list:
    return [d for d in ctx.library.docs.values() if d.category == "company"]


def _team_text(ctx: RunContext) -> str:
    lines = []
    for key, s in SPECIALISTS.items():
        tools = ", ".join(t.name for t in _specialist_tools(ctx, key)) or "(사용 가능한 도구 없음)"
        note = ""
        if key == "company" and not _company_docs(ctx):
            note = " — 현재 등록된 회사 문서가 없으므로 맡길 필요 없음"
        if key == "contract" and not ctx.attachment:
            note = " — 첨부 계약서 없음 (표준계약서 조항 확인만 가능)"
        lines.append(f"- ask_{key}: {s['icon']} {s['name']} — {prompts.SPECIALIST_ROLES[key]} (도구: {tools}){note}")
    return "\n".join(lines)


async def _run_specialist(ctx: RunContext, key: str, task: str, run_id: str, state: dict) -> str:
    """전문가 에이전트 한 번 실행. 팀장의 실행 흐름과 섞이지 않도록 별도 컨텍스트에서 돈다."""
    CURRENT_AGENT.set(key)
    spec = SPECIALISTS[key]
    tools = _specialist_tools(ctx, key)
    observer = ObserverMiddleware(ctx, agent=key, run_id=run_id, tool_limit=SPECIALIST_TOOL_LIMIT, top_level=False)
    agent = create_agent(
        model=agent_graph.llm_factory(ctx.settings),
        tools=[_as_langchain_tool(ctx, t) for t in tools],
        system_prompt=prompts.SPECIALIST_SYSTEM.format(name=spec["name"], role=prompts.SPECIALIST_ROLES[key],
                                                       budget=SPECIALIST_TOOL_LIMIT, llms=ctx.library.index_text),
        middleware=[observer, ModelCallLimitMiddleware(run_limit=SPECIALIST_MODEL_CALLS, exit_behavior="end")],
        name=f"{key}_agent",
    )
    ctx.emit("agent_start", agent=key, run=run_id, name=spec["name"], icon=spec["icon"],
             tools=[t.name for t in tools])
    brief = (f"[팀장의 지시]\n{task}\n\n[트리거]\n{agent_graph._trigger_text(state)}\n\n"
             f"[첨부 계약서]\n{agent_graph._attachment_brief(ctx, with_preview=False)}")
    started = time.monotonic()
    try:
        result = await agent.ainvoke({"messages": [HumanMessage(brief)]}, config={"recursion_limit": 60})
        last = next((m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None)
        report = (last.text.strip() if last is not None else "") or "(보고서를 만들지 못했습니다)"
        ok = True
    except Exception as e:  # 전문가 실패는 팀장이 보고 대처하게 한다
        report, ok = f"(작업 실패: {type(e).__name__}: {e})", False
    ctx.emit("agent_report", agent=key, run=run_id, ok=ok, report=report, source_ids=observer.source_ids,
             model_calls=observer.turn, tool_calls=observer.tools_used,
             elapsed_ms=int((time.monotonic() - started) * 1000))
    sources = "\n".join(f"[{s.id}] {s.title}" for s in ctx.registry.all() if s.id in observer.source_ids)
    return f"[{spec['icon']} {spec['name']} 보고서]\n{report}\n\n[이 전문가가 확인한 자료]\n{sources or '(없음)'}"


def _ask_tool(ctx: RunContext, key: str, state: dict) -> StructuredTool:
    spec = SPECIALISTS[key]

    async def run(reason: str, task: str) -> str:
        run_id = _CURRENT_CALL.get()
        # 빈 컨텍스트에서 실행해야 팀장 스트림에 전문가의 글자가 섞이지 않는다
        return await asyncio.create_task(_run_specialist(ctx, key, task, run_id, state),
                                         context=contextvars.Context())

    return StructuredTool.from_function(
        coroutine=run, name=f"ask_{key}", args_schema=AskArgs,
        description=f"{spec['icon']} {spec['name']}에게 일을 맡기고 보고서를 받는다. {prompts.SPECIALIST_ROLES[key]}")


async def run_multi(ctx: RunContext, state: dict) -> None:
    observer = ObserverMiddleware(ctx, agent="supervisor")
    supervisor = create_agent(
        model=agent_graph.llm_factory(ctx.settings),
        tools=[_ask_tool(ctx, key, state) for key in SPECIALISTS],
        system_prompt=prompts.SUPERVISOR_SYSTEM.format(team=_team_text(ctx), llms=ctx.library.index_text),
        middleware=[observer, VerifyMiddleware(ctx, observer), TodoListMiddleware(),
                    ModelCallLimitMiddleware(run_limit=SUPERVISOR_MODEL_CALLS, exit_behavior="end")],
        name="supervisor",
    )
    team = [{"key": "supervisor", "name": "팀장", "icon": "🧑‍💼", "tools": [f"ask_{k}" for k in SPECIALISTS]}]
    team += [{"key": k, "name": s["name"], "icon": s["icon"], "tools": [t.name for t in _specialist_tools(ctx, k)]}
             for k, s in SPECIALISTS.items()]
    ctx.stage("multi", "start")
    ctx.emit("team", agents=team, company_docs=len(_company_docs(ctx)))
    ctx.emit("context", items=[
        {"label": "트리거", "detail": agent_graph._trigger_text(state).replace("\n", " · ")},
        {"label": "팀 구성", "detail": "팀장 1 + 전문가 4 (" + ", ".join(s["name"] for s in SPECIALISTS.values()) + ")"},
        {"label": "팀장의 도구", "detail": ", ".join(f"ask_{k}" for k in SPECIALISTS) + ", write_todos (자료는 직접 안 읽음)"},
    ])
    answer = await stream_agent(ctx, supervisor, state, observer)
    ctx.stage("multi", "end")
    agent_graph._finish(ctx, answer, None, [])
