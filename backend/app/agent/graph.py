"""LangGraph 에이전트.

    START → analyze ─┬─(direct)──→ direct_answer → END
                     └─(research)→ plan → research ⇄ tools
                                              └──→ write → verify → END

각 노드는 RunContext.emit() 으로 화면에 보여줄 이벤트를 보낸다.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from ..config import Settings
from . import prompts
from .context import RunContext
from .sources import check_citations, normalize_citations, strip_unknown
from .tools import TOOLS_BY_NAME, ToolOutput, available_tools
from .usage import from_usage_metadata

MAX_RESEARCH_ROUNDS = 8


# ---------------------------------------------------------------- 구조화 출력 스키마

class Analysis(BaseModel):
    question_type: Literal["contract_review", "legal_question", "general_chat", "off_topic"]
    summary: str = Field(description="질문의 요지를 쉬운 말로 한 문장")
    issues: list[str] = Field(description="검토할 법적 쟁점 목록 (없으면 빈 목록)")
    use_attachment: bool = Field(description="첨부 계약서를 읽어야 하는지")
    attachment_reason: str = Field(description="첨부 계약서를 읽거나 읽지 않는 이유 (첨부가 없으면 빈 문자열)")
    candidate_docs: list[str] = Field(description="참고할 llms.txt 문서 ID 목록 (표준계약서·회사 참고 문서)")
    needs_law: bool
    needs_precedent: bool
    needs_web: bool
    route: Literal["research", "direct"]
    thought: str = Field(description="이렇게 판단한 이유 1~2문장")


class PlanStep(BaseModel):
    no: int
    goal: str = Field(description="이 단계에서 확인할 것 (한 문장)")
    tool: str = Field(description="주로 사용할 도구 이름")


class Plan(BaseModel):
    strategy: str = Field(description="전체 조사 방향 1~2문장")
    steps: list[PlanStep]


class AgentState(TypedDict, total=False):
    question: str
    history: list[dict]
    analysis: dict
    plan: dict
    messages: Annotated[list[AnyMessage], add_messages]
    research_rounds: int
    research_note: str
    answer: str


# ---------------------------------------------------------------- LLM

def make_llm(settings: Settings) -> ChatOpenAI:
    kwargs: dict = {}
    if settings.reasoning_effort:
        # 추론 요약(summary)을 받으면 화면에 'AI가 밝힌 판단 근거'로 보여줄 수 있다.
        kwargs["reasoning"] = {"effort": settings.reasoning_effort, "summary": "auto"}
    return ChatOpenAI(
        model=settings.model,
        api_key=settings.openai_api_key,
        use_responses_api=True,
        timeout=180,
        max_retries=2,
        stream_usage=True,  # 스트리밍 답변도 토큰 사용량을 받는다
        **kwargs,
    )


# 테스트에서 가짜 LLM 으로 바꿔 끼울 수 있게 모듈 변수로 둔다.
llm_factory = make_llm


def _ctx(config: RunnableConfig) -> RunContext:
    return config["configurable"]["ctx"]


def _reasoning_text(msg) -> str:
    if msg is None:
        return ""
    try:
        blocks = msg.content_blocks
    except Exception:
        return ""
    return "\n".join(b.get("reasoning", "") for b in blocks if b.get("type") == "reasoning").strip()


def _emit_reasoning(ctx: RunContext, stage: str, msg) -> None:
    text = _reasoning_text(msg)
    if text:
        ctx.emit("reasoning", stage=stage, text=text)


def _record(ctx: RunContext, stage: str, purpose: str, msg, started: float) -> None:
    meta = getattr(msg, "response_metadata", None) or {}
    ctx.record_llm_call(stage, purpose, meta.get("model_name") or ctx.settings.model,
                        from_usage_metadata(getattr(msg, "usage_metadata", None)),
                        int((time.monotonic() - started) * 1000))


async def _structured(ctx: RunContext, stage: str, purpose: str, schema: type[BaseModel], messages: list):
    llm = llm_factory(ctx.settings).with_structured_output(schema, include_raw=True)
    started = time.monotonic()
    out = await llm.ainvoke(messages)
    _record(ctx, stage, purpose, out.get("raw"), started)
    _emit_reasoning(ctx, stage, out.get("raw"))
    if out.get("parsed") is None:
        raise RuntimeError(f"{stage} 단계 구조화 출력 실패: {out.get('parsing_error')}")
    return out["parsed"]


def _history_text(history: list[dict], limit: int = 6) -> str:
    if not history:
        return "(없음)"
    lines = []
    for h in history[-limit:]:
        who = "사용자" if h.get("role") == "user" else "에이전트"
        lines.append(f"{who}: {h.get('content', '')[:800]}")
    return "\n".join(lines)


def _attachment_brief(ctx: RunContext, with_preview: bool = True) -> str:
    doc = ctx.attachment
    if not doc:
        return "(첨부된 계약서 없음)"
    text = f"파일명: {doc.title}\n조항 수: {len(doc.sections)}\n목차:\n{doc.outline_text()}"
    if with_preview and doc.preamble:
        text += f"\n\n머리말(앞부분):\n{doc.preamble[:400]}"
    return text


def _tools_brief(ctx: RunContext) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in available_tools(ctx))


# ---------------------------------------------------------------- 노드

async def analyze(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("analyze", "start")
    seen = [
        {"label": "사용자 질문", "detail": state["question"]},
        {"label": "이전 대화", "detail": f"{len(state.get('history', []))}개 메시지"},
        {"label": "참고 문서 목록 (llms.txt)", "detail": ", ".join(d.title for d in ctx.library.docs.values())},
    ]
    if ctx.attachment:
        seen.append({"label": "첨부 계약서 목차 (본문은 아직 안 읽음)",
                     "detail": f"{ctx.attachment.title} · {len(ctx.attachment.sections)}개 조항"})
    ctx.emit("context", items=seen)

    user = (
        f"[참고 문서 목록 llms.txt]\n{ctx.library.index_text}\n\n"
        f"[첨부 계약서]\n{_attachment_brief(ctx)}\n\n"
        f"[이전 대화]\n{_history_text(state.get('history', []))}\n\n"
        f"[사용자 질문]\n{state['question']}"
    )
    analysis: Analysis = await _structured(
        ctx, "analyze", "질문 분석", Analysis, [SystemMessage(prompts.ANALYZE_SYSTEM), HumanMessage(user)]
    )
    if not ctx.attachment:
        analysis.use_attachment = False
        analysis.attachment_reason = ""
    analysis.candidate_docs = [c for c in analysis.candidate_docs if c in ctx.library.docs]
    ctx.emit("analysis", data=analysis.model_dump())
    ctx.emit("route", path=analysis.route,
             label="자료 조사가 필요한 질문 → 계획 수립" if analysis.route == "research" else "조사 없이 바로 답변")
    ctx.stage("analyze", "end")
    return {"analysis": analysis.model_dump()}


def route_after_analyze(state: AgentState) -> str:
    return "plan" if state["analysis"]["route"] == "research" else "direct_answer"


async def plan(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("plan", "start")
    user = (
        f"[질문]\n{state['question']}\n\n"
        f"[질문 분석]\n{json.dumps(state['analysis'], ensure_ascii=False, indent=1)}\n\n"
        f"[첨부 계약서]\n{_attachment_brief(ctx, with_preview=False)}\n\n"
        f"[참고 문서 목록 llms.txt]\n{ctx.library.index_text}\n\n"
        f"[사용 가능한 도구]\n{_tools_brief(ctx)}"
    )
    system = prompts.PLAN_SYSTEM.format(budget=ctx.settings.max_tool_calls)
    result: Plan = await _structured(ctx, "plan", "조사 계획 수립", Plan, [SystemMessage(system), HumanMessage(user)])
    steps = [s.model_copy(update={"no": i}) for i, s in enumerate(result.steps, start=1)]
    ctx.plan_status = {s.no: "pending" for s in steps}
    data = {"strategy": result.strategy, "steps": [s.model_dump() for s in steps]}
    ctx.emit("plan", **data)
    ctx.stage("plan", "end")
    return {"plan": data}


async def research(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    rounds = state.get("research_rounds", 0)
    if rounds == 0:
        ctx.stage("research", "start")

    messages = list(state.get("messages", []))
    new_messages: list[AnyMessage] = []
    if not messages:
        steps = "\n".join(f"{s['no']}. {s['goal']} (도구: {s['tool']})" for s in state["plan"]["steps"])
        new_messages = [
            SystemMessage(prompts.RESEARCH_SYSTEM.format(budget=ctx.settings.max_tool_calls)),
            HumanMessage(
                f"[질문]\n{state['question']}\n\n"
                f"[이전 대화]\n{_history_text(state.get('history', []))}\n\n"
                f"[질문 분석]\n{json.dumps(state['analysis'], ensure_ascii=False, indent=1)}\n\n"
                f"[첨부 계약서]\n{_attachment_brief(ctx, with_preview=False)}\n\n"
                f"[조사 계획]\n{state['plan']['strategy']}\n{steps}"
            ),
        ]
        messages += new_messages

    tool_schemas = [t.openai_schema() for t in available_tools(ctx)]
    budget_left = ctx.settings.max_tool_calls - ctx.tool_calls
    if budget_left <= 0 or rounds >= MAX_RESEARCH_ROUNDS:
        # 이전 대화에 도구 호출이 있으므로 도구 정의는 유지하되 호출은 막는다.
        llm = llm_factory(ctx.settings).bind_tools(tool_schemas, tool_choice="none")
        extra = HumanMessage(prompts.RESEARCH_BUDGET_EXHAUSTED)
        messages.append(extra)
        new_messages.append(extra)
    else:
        llm = llm_factory(ctx.settings).bind_tools(tool_schemas)

    # 에이전트 루프 1회 = LLM 이 다음 행동을 판단하는 호출 1번 (+ 그 결과로 실행되는 도구들)
    ctx.agent_loops += 1
    ctx.emit("loop", n=ctx.agent_loops)
    started = time.monotonic()
    response: AIMessage = await llm.ainvoke(messages)
    _record(ctx, "research", f"루프 {ctx.agent_loops}: 다음 행동 판단", response, started)
    _emit_reasoning(ctx, "research", response)
    new_messages.append(response)

    update: dict = {"messages": new_messages, "research_rounds": rounds + 1}
    if response.tool_calls:
        if response.text.strip():
            ctx.emit("thought", text=response.text.strip())
        return update

    # 도구를 더 부르지 않음 → 조사 종료
    note = response.text.strip()
    ctx.emit("research_note", text=note)
    for no, status in list(ctx.plan_status.items()):
        ctx.set_step(no, "done" if status == "in_progress" else ("skipped" if status == "pending" else status))
    ctx.stage("research", "end")
    update["research_note"] = note
    return update


def route_after_research(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "write"


async def _execute(ctx: RunContext, call: dict) -> ToolOutput:
    spec = TOOLS_BY_NAME.get(call["name"])
    started = time.monotonic()
    if spec is None or not spec.available(ctx):
        out = ToolOutput(f"사용할 수 없는 도구입니다: {call['name']}", "사용할 수 없는 도구", ok=False)
    else:
        try:
            args = spec.args_model.model_validate(call["args"])
            out = await spec.run(ctx, args)
        except Exception as e:  # 도구 실패는 에이전트가 보고 대처하도록 결과로 돌려준다
            out = ToolOutput(f"도구 실행 실패: {type(e).__name__}: {e}", f"실패: {e}", ok=False)
    ctx.emit("tool_result", id=call["id"], tool=call["name"], ok=out.ok, summary=out.summary,
             source_ids=out.source_ids, preview=out.preview[:4000],
             elapsed_ms=int((time.monotonic() - started) * 1000))
    return out


async def tools(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    calls = state["messages"][-1].tool_calls
    budget_left = max(ctx.settings.max_tool_calls - ctx.tool_calls, 0)
    to_run, skipped = calls[:budget_left], calls[budget_left:]

    for call in to_run:
        args = dict(call["args"])
        reason = args.pop("reason", "")
        step = args.pop("plan_step", None)
        spec = TOOLS_BY_NAME.get(call["name"])
        if isinstance(step, int):
            for no, status in ctx.plan_status.items():
                if status == "in_progress" and no < step:
                    ctx.set_step(no, "done")
            ctx.set_step(step, "in_progress")
        ctx.emit("tool_call", id=call["id"], tool=call["name"], label=spec.label if spec else call["name"],
                 args=args, reason=reason, plan_step=step)
    ctx.tool_calls += len(to_run)

    outputs = await asyncio.gather(*(_execute(ctx, c) for c in to_run))
    left = ctx.settings.max_tool_calls - ctx.tool_calls
    messages = [
        ToolMessage(f"{out.text}\n\n(남은 도구 호출 예산: {left}회)", tool_call_id=c["id"], name=c["name"])
        for c, out in zip(to_run, outputs)
    ]
    messages += [
        ToolMessage("도구 호출 예산을 초과하여 실행하지 않았습니다.", tool_call_id=c["id"], name=c["name"])
        for c in skipped
    ]
    return {"messages": messages}


async def _stream_answer(ctx: RunContext, stage: str, purpose: str, messages: list) -> str:
    llm = llm_factory(ctx.settings)
    started = time.monotonic()
    answer = ""
    full = None
    async for chunk in llm.astream(messages):
        full = chunk if full is None else full + chunk
        text = chunk.text
        if text:
            answer += text
            ctx.emit("token", text=text)
    _record(ctx, stage, purpose, full, started)
    _emit_reasoning(ctx, stage, full)
    return answer


async def write(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("write", "start")
    sources = ctx.registry.all()
    source_text = "\n\n".join(f"[{s.id}] ({s.kind}) {s.title}\n{s.content}" for s in sources) or "(수집된 자료 없음)"
    user = (
        f"[질문]\n{state['question']}\n\n"
        f"[질문 분석]\n요지: {state['analysis']['summary']}\n쟁점: {', '.join(state['analysis']['issues'])}\n\n"
        f"[조사 메모]\n{state.get('research_note') or '(없음)'}\n\n"
        f"[수집된 자료] (이 ID 만 인용 가능)\n{source_text}"
    )
    answer = await _stream_answer(ctx, "write", "답변 작성", [SystemMessage(prompts.WRITE_SYSTEM), HumanMessage(user)])
    ctx.stage("write", "end")
    return {"answer": answer}


async def verify(state: AgentState, config: RunnableConfig) -> dict:
    """코드로 인용을 검증한다: 존재하지 않는 ID 제거, 인용 없는 답변 경고."""
    ctx = _ctx(config)
    ctx.stage("verify", "start")
    answer = normalize_citations(state["answer"])
    check = check_citations(answer, ctx.registry)
    answer = strip_unknown(answer, check["unknown"])

    problems = []
    if check["unknown"]:
        problems.append(f"자료 목록에 없는 인용 {len(check['unknown'])}개를 제거했습니다: {', '.join(check['unknown'])}")
    if len(ctx.registry) and not check["cited"]:
        problems.append("답변에 근거 인용이 하나도 없습니다.")
    ctx.emit("verify", ok=not problems, cited=check["cited"], unknown=check["unknown"],
             unused=check["unused"], problems=problems)
    ctx.stage("verify", "end")
    _finish(ctx, answer, check["cited"])
    return {"answer": answer}


async def direct_answer(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("direct", "start")
    user = f"[이전 대화]\n{_history_text(state.get('history', []))}\n\n[사용자 질문]\n{state['question']}"
    answer = await _stream_answer(ctx, "direct", "바로 답변", [SystemMessage(prompts.DIRECT_SYSTEM), HumanMessage(user)])
    ctx.stage("direct", "end")
    _finish(ctx, answer, [])
    return {"answer": answer}


def _finish(ctx: RunContext, answer: str, cited: list[str]) -> None:
    ctx.emit("sources", items=[s.to_dict() | {"cited": s.id in cited} for s in ctx.registry.all()])
    ctx.emit("final", answer_md=answer, cited_ids=cited, stats=ctx.stats() | {"cited": len(cited)})


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze", analyze)
    g.add_node("direct_answer", direct_answer)
    g.add_node("plan", plan)
    g.add_node("research", research)
    g.add_node("tools", tools)
    g.add_node("write", write)
    g.add_node("verify", verify)

    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", route_after_analyze, ["plan", "direct_answer"])
    g.add_edge("plan", "research")
    g.add_conditional_edges("research", route_after_research, ["tools", "write"])
    g.add_edge("tools", "research")
    g.add_edge("write", "verify")
    g.add_edge("verify", END)
    g.add_edge("direct_answer", END)
    return g.compile()


GRAPH = build_graph()
