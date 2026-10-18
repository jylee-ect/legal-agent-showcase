"""LangGraph 에이전트 — 일반적인 에이전트 루프.

    트리거 → define_goal ─┬─(direct)──→ direct_answer → END
                          └─(research)→ plan → pick_step ─→ act ⇄ tools
                                              ↑     │          └→ check ─┬─ 계속/계획 수정 → pick_step
                                              │     └(남은 단계 없음)      └─ 목표 충족 → write
                                              │                                          ↓
                                              └──── 미달: 단계 추가 ──── evaluate ← verify ⇄ write
                                                                  (근거 문제: 피드백과 함께 다시 작성)
                                                                            └─ 달성/한도 → END
    END 뒤: 최종 안전장치(에이전트 밖, 코드) — 그래도 남은 가짜 인용 제거

- 목표 정의: 트리거(질문·파일 도착·정기 일정)에서 목표와 '완료 조건'을 만든다.
- 계획: 단계마다 어떤 완료 조건을 채우는지 연결한다.
- 실행 → 점검: 한 단계를 실행하고, 결과를 관찰해 작업 메모(기억)에 남기고, 완료 조건 현황을 갱신하고,
  계속 / 같은 단계 한 번 더 / 계획 수정 / 실행 종료 중 하나를 결정한다.
- 목표 달성 판단: 답변이 완료 조건을 충족하는지 판정하고, 미달이면 단계를 추가해 다시 실행한다.

각 노드는 RunContext.emit() 으로 화면에 보여줄 이벤트를 보낸다.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..config import Settings
from . import prompts
from .context import RunContext
from .sources import cited_ids, normalize_citations
from .tools import TOOLS_BY_NAME, ToolOutput, available_tools
from .usage import from_usage_metadata
from .verification import MAX_VERIFY_RETRIES, check_grounding, feedback_message, guardrail

MAX_STEP_RUNS = 10  # 단계 실행 총 횟수 한도 (무한 루프 방지)
MAX_STEP_ROUNDS = 2  # 한 단계 안에서 '행동 → 점검'을 반복할 수 있는 횟수
MAX_REVISIONS = 4  # 점검 단계의 계획 수정 한도
MAX_EVAL_RETRIES = 1  # 목표 미달 시 추가 조사로 되돌아가는 횟수

CRITERION_STATUS = Literal["met", "partial", "unmet"]


# ---------------------------------------------------------------- 구조화 출력 스키마

class Goal(BaseModel):
    trigger_summary: str = Field(description="무엇이 이 작업을 시작시켰는지 한 문장")
    question_type: Literal["contract_review", "legal_question", "monitoring", "general_chat", "off_topic"]
    summary: str = Field(description="요청의 요지를 쉬운 말로 한 문장")
    goal: str = Field(description="이번 작업의 목표 한 문장")
    completion_criteria: list[str] = Field(description="목표 달성을 판정할 수 있는 완료 조건 2~5개 (direct 면 빈 목록)")
    issues: list[str] = Field(description="검토할 법적 쟁점 목록 (없으면 빈 목록)")
    use_attachment: bool = Field(description="첨부 계약서를 읽어야 하는지")
    attachment_reason: str = Field(description="첨부 계약서를 읽거나 읽지 않는 이유 (첨부가 없으면 빈 문자열)")
    candidate_docs: list[str] = Field(description="참고할 llms.txt 문서 ID 목록")
    route: Literal["research", "direct"]
    thought: str = Field(description="목표와 완료 조건을 이렇게 정한 이유 1~2문장")


class NewStep(BaseModel):
    goal: str = Field(description="이 단계에서 할 일 한 문장")
    tool: str = Field(description="주로 사용할 도구 이름")
    criteria: list[int] = Field(description="이 단계가 채우는 완료 조건 번호 목록")


class Plan(BaseModel):
    strategy: str = Field(description="전체 진행 방향 1~2문장")
    steps: list[NewStep]


class CriterionCheck(BaseModel):
    id: int
    status: CRITERION_STATUS
    note: str = Field(description="판단 근거 한 문장 (근거 ID 포함)")


class Check(BaseModel):
    observation: str = Field(description="이번 단계에서 알게 된 핵심 1~3문장 (근거 ID 포함). 작업 메모로 저장됨")
    step_done: bool
    criteria: list[CriterionCheck]
    decision: Literal["continue", "revise", "finish"]
    reason: str = Field(description="이렇게 판단한 이유 1~2문장")
    drop_steps: list[int] = Field(description="취소할 대기 단계 번호 (revise 일 때)")
    add_steps: list[NewStep] = Field(description="추가할 단계 (revise 일 때)")
    insert_before: int | None = Field(description="새 단계를 넣을 위치(대기 단계 번호). 없으면 맨 뒤")


class Evaluation(BaseModel):
    criteria: list[CriterionCheck]
    goal_met: bool
    missing: list[str]
    feedback: str
    add_steps: list[NewStep] = Field(description="추가 조사로 해결 가능하면 필요한 단계 (최대 2개), 아니면 빈 목록")


class AgentState(TypedDict, total=False):
    question: str
    trigger: dict  # {type, label, detail}
    history: list[dict]
    goal: dict
    criteria: list[dict]  # [{id, text, status, note}]
    strategy: str
    steps: list[dict]  # [{no, goal, tool, criteria, status, added_by}] status: pending|in_progress|done|skipped|dropped
    current: int | None
    step_round: int
    step_messages: list  # 현재 단계의 행동·도구 메시지 (단계마다 새로 시작)
    notes: list[dict]  # 작업 메모(기억): [{step, text}]
    tool_log: list[dict]  # 도구 결과 기록(기억): [{step, tool, args, text}] — 다음 단계가 이전 결과를 참고
    executing: bool
    steps_run: int
    revisions: int
    eval_attempts: int
    answer: str
    verify_round: int  # 근거 검증 횟수 (현재 답변 사이클)
    verify_feedback: str  # 근거 검증이 돌려준 고칠 점 → 답변 다시 작성
    next: str


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


def _record(ctx: RunContext, stage: str, purpose: str, msg, started: float, agent: str | None = None) -> None:
    meta = getattr(msg, "response_metadata", None) or {}
    ctx.record_llm_call(stage, purpose, meta.get("model_name") or ctx.settings.model,
                        from_usage_metadata(getattr(msg, "usage_metadata", None)),
                        int((time.monotonic() - started) * 1000), agent=agent)


async def _structured(ctx: RunContext, stage: str, purpose: str, schema: type[BaseModel], messages: list):
    llm = llm_factory(ctx.settings).with_structured_output(schema, include_raw=True)
    started = time.monotonic()
    out = await llm.ainvoke(messages)
    _record(ctx, stage, purpose, out.get("raw"), started)
    _emit_reasoning(ctx, stage, out.get("raw"))
    if out.get("parsed") is None:
        raise RuntimeError(f"{purpose} 단계 구조화 출력 실패: {out.get('parsing_error')}")
    return out["parsed"]


# ---------------------------------------------------------------- 프롬프트 재료

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


def _trigger_text(state: AgentState) -> str:
    t = state.get("trigger") or {}
    text = f"{t.get('label', '사용자 질문')}"
    if t.get("detail"):
        text += f": {t['detail']}"
    if state.get("question"):
        text += f"\n질문: {state['question']}"
    return text


STATUS_KO = {"met": "충족", "partial": "일부 충족", "unmet": "미충족"}
STEP_KO = {"pending": "대기", "in_progress": "진행 중", "done": "완료", "skipped": "건너뜀", "dropped": "취소"}


def _criteria_text(criteria: list[dict]) -> str:
    return "\n".join(f"{c['id']}. {c['text']} — {STATUS_KO[c['status']]}" + (f" ({c['note']})" if c["note"] else "")
                     for c in criteria) or "(없음)"


def _steps_text(steps: list[dict]) -> str:
    return "\n".join(f"{s['no']}. [{STEP_KO[s['status']]}] {s['goal']} (도구: {s['tool']}, 조건: {s['criteria']})"
                     for s in steps)


def _notes_text(notes: list[dict]) -> str:
    return "\n".join(f"- (단계 {n['step']}) {n['text']}" for n in notes) or "(아직 없음)"


def _tool_log_text(log: list[dict], last: int = 6, limit: int = 700) -> str:
    if not log:
        return "(아직 없음)"
    lines = []
    for t in log[-last:]:
        args = json.dumps(t["args"], ensure_ascii=False)
        text = t["text"][:limit] + ("…" if len(t["text"]) > limit else "")
        lines.append(f"- (단계 {t['step']}) {t['tool']} {args}\n{text}")
    return "\n".join(lines)


def _sources_brief(ctx: RunContext) -> str:
    return "\n".join(f"[{s.id}] {s.title}" for s in ctx.registry.all()) or "(아직 없음)"


def _goal_block(state: AgentState) -> str:
    return (f"[트리거]\n{_trigger_text(state)}\n\n[목표]\n{state['goal']['goal']}\n\n"
            f"[완료 조건]\n{_criteria_text(state['criteria'])}")


def _budget_left(ctx: RunContext) -> int:
    return max(ctx.settings.max_tool_calls - ctx.tool_calls, 0)


# ---------------------------------------------------------------- 계획 조작 (코드)

def _set_status(ctx: RunContext, steps: list[dict], no: int, status: str) -> None:
    for s in steps:
        if s["no"] == no and s["status"] != status:
            s["status"] = status
            ctx.emit("plan_update", step_no=no, status=status)


def _new_steps(steps: list[dict], new: list[NewStep], valid_criteria: set[int], added_by: str) -> list[dict]:
    next_no = max((s["no"] for s in steps), default=0) + 1
    out = []
    for i, n in enumerate(new):
        out.append({"no": next_no + i, "goal": n.goal, "tool": n.tool,
                    "criteria": [c for c in n.criteria if c in valid_criteria], "status": "pending",
                    "added_by": added_by})
    return out


def _revise(ctx: RunContext, steps: list[dict], drop: list[int], add: list[NewStep], insert_before: int | None,
            criteria: list[dict], reason: str, by: str) -> list[dict]:
    """대기 단계를 취소하고 새 단계를 끼워 넣는다. 화면에는 무엇이 바뀌었는지 보여준다."""
    steps = [dict(s) for s in steps]
    pending = {s["no"] for s in steps if s["status"] == "pending"}
    dropped = [n for n in drop if n in pending]
    for no in dropped:
        _set_status(ctx, steps, no, "dropped")
    added = _new_steps(steps, add, {c["id"] for c in criteria}, by)
    if added:
        pos = next((i for i, s in enumerate(steps) if s["no"] == insert_before and s["status"] == "pending"),
                   len(steps))
        steps[pos:pos] = added
    if dropped or added:
        ctx.emit("plan_revised", by=by, reason=reason, dropped=dropped, added=added, steps=steps)
    return steps


def _apply_criteria(criteria: list[dict], checks: list[CriterionCheck]) -> list[dict]:
    by_id = {c.id: c for c in checks}
    return [c | ({"status": by_id[c["id"]].status, "note": by_id[c["id"]].note} if c["id"] in by_id else {})
            for c in criteria]


# ---------------------------------------------------------------- 노드

async def define_goal(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("goal", "start")
    trigger = state.get("trigger") or {}
    seen = [{"label": "트리거", "detail": _trigger_text(state).replace("\n", " · ")},
            {"label": "이전 대화", "detail": f"{len(state.get('history', []))}개 메시지"},
            {"label": "참고 문서 목록 (llms.txt)", "detail": ", ".join(d.title for d in ctx.library.docs.values())}]
    if ctx.attachment:
        seen.append({"label": "첨부 계약서 목차 (본문은 아직 안 읽음)",
                     "detail": f"{ctx.attachment.title} · {len(ctx.attachment.sections)}개 조항"})
    ctx.emit("context", items=seen)

    user = (
        f"[트리거]\n{_trigger_text(state)}\n\n"
        f"[참고 문서 목록 llms.txt]\n{ctx.library.index_text}\n\n"
        f"[첨부 계약서]\n{_attachment_brief(ctx)}\n\n"
        f"[이전 대화]\n{_history_text(state.get('history', []))}"
    )
    goal: Goal = await _structured(ctx, "goal", "목표 정의", Goal,
                                   [SystemMessage(prompts.GOAL_SYSTEM), HumanMessage(user)])
    if trigger.get("type", "question") != "question":
        goal.route = "research"  # 질문이 아닌 트리거는 항상 작업을 수행한다
    if not ctx.attachment:
        goal.use_attachment, goal.attachment_reason = False, ""
    goal.candidate_docs = [c for c in goal.candidate_docs if c in ctx.library.docs]
    if goal.route == "research" and not goal.completion_criteria:
        goal.completion_criteria = [goal.goal]
    criteria = [{"id": i, "text": t, "status": "unmet", "note": ""}
                for i, t in enumerate(goal.completion_criteria, start=1)] if goal.route == "research" else []

    ctx.emit("goal", data=goal.model_dump(), criteria=criteria)
    ctx.emit("route", path=goal.route,
             label="작업이 필요함 → 계획 수립" if goal.route == "research" else "조사 없이 바로 답변")
    ctx.stage("goal", "end")
    return {"goal": goal.model_dump(), "criteria": criteria, "notes": [], "steps_run": 0, "revisions": 0,
            "eval_attempts": 0, "executing": False}


def route_after_goal(state: AgentState) -> str:
    return "plan" if state["goal"]["route"] == "research" else "direct_answer"


async def plan(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("plan", "start")
    user = (
        f"{_goal_block(state)}\n\n"
        f"[요지·쟁점]\n{state['goal']['summary']} / {', '.join(state['goal']['issues'])}\n\n"
        f"[첨부 계약서]\n{_attachment_brief(ctx, with_preview=False)}\n\n"
        f"[참고 문서 목록 llms.txt]\n{ctx.library.index_text}\n\n"
        f"[사용 가능한 도구]\n{_tools_brief(ctx)}"
    )
    system = prompts.PLAN_SYSTEM.format(budget=ctx.settings.max_tool_calls)
    result: Plan = await _structured(ctx, "plan", "계획 수립", Plan, [SystemMessage(system), HumanMessage(user)])
    steps = _new_steps([], result.steps, {c["id"] for c in state["criteria"]}, "plan")
    ctx.emit("plan", strategy=result.strategy, steps=steps)
    ctx.stage("plan", "end")
    return {"strategy": result.strategy, "steps": steps}


def pick_step(state: AgentState, config: RunnableConfig) -> dict:
    """다음 대기 단계를 고른다. 없거나 한도에 닿으면 실행을 마치고 답변 작성으로 간다."""
    ctx = _ctx(config)
    steps = [dict(s) for s in state["steps"]]
    nxt = next((s for s in steps if s["status"] == "pending"), None)
    limit = state.get("steps_run", 0) >= MAX_STEP_RUNS
    if nxt is None or limit or _budget_left(ctx) == 0:
        return _close_execution(ctx, state, steps, "실행 한도에 도달" if limit else "")

    update: dict = {"steps": steps, "current": nxt["no"], "step_round": 0, "step_messages": [],
                    "steps_run": state.get("steps_run", 0) + 1, "next": "act", "executing": True}
    if not state.get("executing"):
        ctx.stage("execute", "start")
    _set_status(ctx, steps, nxt["no"], "in_progress")
    ctx.agent_loops += 1
    ctx.emit("loop", n=ctx.agent_loops)
    ctx.emit("step_start", no=nxt["no"], goal=nxt["goal"], tool=nxt["tool"], criteria=nxt["criteria"])
    return update


def _close_execution(ctx: RunContext, state: AgentState, steps: list[dict], note: str) -> dict:
    for s in steps:
        if s["status"] == "pending":
            _set_status(ctx, steps, s["no"], "skipped")
    if note:
        ctx.emit("thought", text=f"{note}: 남은 단계는 건너뛰고 답변을 작성합니다.")
    if state.get("executing"):
        ctx.stage("execute", "end")
    return {"steps": steps, "current": None, "executing": False, "next": "write"}


def route_next(state: AgentState) -> str:
    return state["next"]


def _current(state: AgentState) -> dict:
    return next(s for s in state["steps"] if s["no"] == state["current"])


async def act(state: AgentState, config: RunnableConfig) -> dict:
    """현재 단계를 실행: LLM 이 어떤 도구를 쓸지 정한다 (행동)."""
    ctx = _ctx(config)
    step = _current(state)
    messages = list(state.get("step_messages") or [])
    if not messages:
        messages = [
            SystemMessage(prompts.EXECUTOR_SYSTEM),
            HumanMessage(
                f"{_goal_block(state)}\n\n"
                f"[전체 계획]\n{_steps_text(state['steps'])}\n\n"
                f"[현재 단계]\n{step['no']}. {step['goal']} (추천 도구: {step['tool']})\n\n"
                f"[작업 메모 — 지금까지 알게 된 것]\n{_notes_text(state.get('notes', []))}\n\n"
                f"[이전 단계의 도구 결과 — 최근 것]\n{_tool_log_text(state.get('tool_log', []))}\n\n"
                f"[첨부 계약서]\n{_attachment_brief(ctx, with_preview=False)}\n\n"
                f"[참고 문서 목록 llms.txt]\n{ctx.library.index_text}\n\n"
                f"(남은 도구 호출 예산: {_budget_left(ctx)}회)"
            ),
        ]
    schemas = [t.openai_schema() for t in available_tools(ctx)]
    llm = llm_factory(ctx.settings).bind_tools(schemas, **({"tool_choice": "none"} if _budget_left(ctx) == 0 else {}))
    started = time.monotonic()
    response: AIMessage = await llm.ainvoke(messages)
    _record(ctx, "execute", f"단계 {step['no']}: 행동 결정", response, started)
    _emit_reasoning(ctx, "execute", response)
    if response.text.strip():
        ctx.emit("thought", text=response.text.strip())
    return {"step_messages": messages + [response], "step_round": state.get("step_round", 0) + 1}


def route_after_act(state: AgentState) -> str:
    last = state["step_messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else "check"


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
    calls = state["step_messages"][-1].tool_calls
    to_run, skipped = calls[:_budget_left(ctx)], calls[_budget_left(ctx):]
    for call in to_run:
        args = dict(call["args"])
        reason = args.pop("reason", "")
        args.pop("plan_step", None)
        spec = TOOLS_BY_NAME.get(call["name"])
        ctx.emit("tool_call", id=call["id"], tool=call["name"], label=spec.label if spec else call["name"],
                 args=args, reason=reason, step=state["current"])
    ctx.tool_calls += len(to_run)

    outputs = await asyncio.gather(*(_execute(ctx, c) for c in to_run))
    messages = [ToolMessage(f"{out.text}\n\n(남은 도구 호출 예산: {_budget_left(ctx)}회)", tool_call_id=c["id"],
                            name=c["name"]) for c, out in zip(to_run, outputs)]
    messages += [ToolMessage("도구 호출 예산을 초과하여 실행하지 않았습니다.", tool_call_id=c["id"], name=c["name"])
                 for c in skipped]
    log = state.get("tool_log", []) + [
        {"step": state["current"], "tool": c["name"], "args": {k: v for k, v in c["args"].items() if k != "reason"},
         "text": out.text} for c, out in zip(to_run, outputs)]
    return {"step_messages": state["step_messages"] + messages, "tool_log": log}


def _step_observations(messages: list) -> str:
    """현재 단계에서 실제로 관찰한 것: 도구 결과와 실행 담당의 메모."""
    parts = []
    for m in messages:
        if isinstance(m, ToolMessage):
            parts.append(f"[도구 결과: {m.name}]\n{str(m.content)[:2500]}")
        elif isinstance(m, AIMessage) and m.text.strip():
            parts.append(f"[실행 메모]\n{m.text.strip()}")
    return "\n\n".join(parts) or "(관찰한 결과 없음)"


async def check(state: AgentState, config: RunnableConfig) -> dict:
    """관찰 → 작업 메모 기록 → 완료 조건 갱신 → 다음 행동 결정 (계속 / 한 번 더 / 계획 수정 / 종료)."""
    ctx = _ctx(config)
    step = _current(state)
    ctx.emit("check_start", step=step["no"])
    user = (
        f"{_goal_block(state)}\n\n"
        f"[전체 계획]\n{_steps_text(state['steps'])}\n\n"
        f"[방금 실행한 단계]\n{step['no']}. {step['goal']} (이 단계 {state['step_round']}회차 시도)\n\n"
        f"[이번 단계에서 관찰한 것]\n{_step_observations(state['step_messages'][2:])}\n\n"
        f"[작업 메모 — 이전 단계까지]\n{_notes_text(state.get('notes', []))}\n\n"
        f"[지금까지 모은 자료]\n{_sources_brief(ctx)}"
    )
    system = prompts.CHECK_SYSTEM.format(left=_budget_left(ctx))
    result: Check = await _structured(ctx, "execute", f"단계 {step['no']}: 점검",
                                      Check, [SystemMessage(system), HumanMessage(user)])

    criteria = _apply_criteria(state["criteria"], result.criteria)
    notes = state.get("notes", []) + [{"step": step["no"], "text": result.observation}]
    steps = [dict(s) for s in state["steps"]]
    all_met = bool(criteria) and all(c["status"] == "met" for c in criteria)
    decision = "finish" if all_met else result.decision  # 완료 조건을 모두 채웠으면 남은 단계와 상관없이 끝낸다
    if decision == "revise" and state.get("revisions", 0) >= MAX_REVISIONS:
        decision = "continue"
    retry_same = (decision == "continue" and not result.step_done
                  and state["step_round"] < MAX_STEP_ROUNDS and _budget_left(ctx) > 0)

    ctx.emit("check", step=step["no"], observation=result.observation, step_done=result.step_done,
             decision="retry_step" if retry_same else decision, reason=result.reason, criteria=criteria,
             source_ids=cited_ids(normalize_citations(result.observation)))

    update: dict = {"criteria": criteria, "notes": notes}
    if retry_same:
        hint = HumanMessage(f"점검 결과: {result.reason}\n이 단계({step['goal']})에서 부족한 부분을 한 번 더 시도하세요.")
        return update | {"step_messages": state["step_messages"] + [hint], "next": "act", "steps": steps}

    _set_status(ctx, steps, step["no"], "done")
    if decision == "revise":
        steps = _revise(ctx, steps, result.drop_steps, result.add_steps, result.insert_before, criteria,
                        result.reason, "check")
        update["revisions"] = state.get("revisions", 0) + 1
    update["steps"] = steps
    if decision == "finish":
        return update | _close_execution(ctx, state | {"executing": True}, steps, "")
    return update | {"next": "pick_step"}


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


def _write_input(state: AgentState, ctx: RunContext) -> str:
    sources = ctx.registry.all()
    source_text = "\n\n".join(f"[{s.id}] ({s.kind}) {s.title}\n{s.content}" for s in sources) or "(수집된 자료 없음)"
    return (f"{_goal_block(state)}\n\n"
            f"[요지·쟁점]\n{state['goal']['summary']} / {', '.join(state['goal']['issues'])}\n\n"
            f"[작업 메모]\n{_notes_text(state.get('notes', []))}\n\n"
            f"[수집된 자료] (이 ID 만 인용 가능)\n{source_text}")


async def write(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("write", "start")
    attempt = state.get("eval_attempts", 0) + 1
    user = _write_input(state, ctx)
    purpose = f"답변 작성 ({attempt}차)" if attempt > 1 else "답변 작성"
    if state.get("verify_feedback"):  # 근거 검증에서 돌려받은 답변을 고쳐 쓴다
        user += f"\n\n[이전에 쓴 답변]\n{state['answer']}\n\n{state['verify_feedback']}"
        purpose = "답변 다시 작성 (근거 검증 피드백 반영)"
    answer = await _stream_answer(ctx, "write", purpose,
                                  [SystemMessage(prompts.WRITE_SYSTEM), HumanMessage(user)])
    ctx.stage("write", "end")
    return {"answer": answer, "verify_feedback": ""}


async def verify(state: AgentState, config: RunnableConfig) -> dict:
    """근거 검증 (에이전트 안): 인용을 코드로 확인하고, 문제가 있으면 피드백과 함께 답변 작성으로 되돌린다."""
    ctx = _ctx(config)
    ctx.stage("verify", "start")
    result = check_grounding(ctx, state["answer"])
    rounds = state.get("verify_round", 0)
    retry = bool(result["problems"]) and rounds < MAX_VERIFY_RETRIES
    ctx.emit("verify", ok=not result["problems"], cited=result["cited"], unknown=result["unknown"],
             unused=result["unused"], problems=result["problems"], retry=retry, attempt=rounds + 1, where="loop")
    ctx.stage("verify", "end")
    update = {"answer": result["answer"], "verify_round": rounds + 1}
    if retry:
        return update | {"verify_feedback": feedback_message(result["problems"]), "next": "write"}
    return update | {"next": "evaluate"}


async def evaluate(state: AgentState, config: RunnableConfig) -> dict:
    """답변이 목표·완료 조건을 충족하는지 판정한다. 미달이면 단계를 추가해 다시 실행한다."""
    ctx = _ctx(config)
    ctx.stage("evaluate", "start")
    attempt = state.get("eval_attempts", 0)
    user = (f"{_goal_block(state)}\n\n[작성된 답변]\n{state['answer']}\n\n"
            f"[수집된 자료]\n{_sources_brief(ctx)}\n\n[현재 계획]\n{_steps_text(state['steps'])}")
    result: Evaluation = await _structured(ctx, "evaluate", "목표 달성 판단", Evaluation,
                                           [SystemMessage(prompts.EVALUATE_SYSTEM), HumanMessage(user)])
    criteria = _apply_criteria(state["criteria"], result.criteria)
    goal_met = result.goal_met and all(c["status"] == "met" for c in criteria)
    can_retry = (not goal_met and result.add_steps and attempt < MAX_EVAL_RETRIES
                 and _budget_left(ctx) > 0 and state.get("steps_run", 0) < MAX_STEP_RUNS)
    ctx.emit("evaluate", goal_met=goal_met, criteria=criteria, missing=result.missing, feedback=result.feedback,
             retry=bool(can_retry), attempt=attempt + 1)

    if can_retry:
        steps = _revise(ctx, state["steps"], [], result.add_steps[:2], None, criteria, result.feedback, "evaluate")
        ctx.stage("evaluate", "end")
        return {"criteria": criteria, "steps": steps, "eval_attempts": attempt + 1, "verify_round": 0,
                "next": "pick_step"}

    answer = state["answer"]
    if not goal_met:
        unmet = [c["text"] for c in criteria if c["status"] != "met"] or result.missing
        answer += ("\n\n---\n\n> ⚠️ **목표 중 확인하지 못한 부분**: " + "; ".join(unmet)
                   + (f"\n>\n> {result.feedback}" if result.feedback else ""))
    ctx.stage("evaluate", "end")
    _finish(ctx, answer, goal_met, criteria)
    return {"criteria": criteria, "answer": answer, "next": "end"}


def route_after_evaluate(state: AgentState) -> str:
    return "pick_step" if state["next"] == "pick_step" else END


async def direct_answer(state: AgentState, config: RunnableConfig) -> dict:
    ctx = _ctx(config)
    ctx.stage("direct", "start")
    user = f"[이전 대화]\n{_history_text(state.get('history', []))}\n\n[사용자 질문]\n{state['question']}"
    answer = await _stream_answer(ctx, "direct", "바로 답변", [SystemMessage(prompts.DIRECT_SYSTEM), HumanMessage(user)])
    ctx.stage("direct", "end")
    _finish(ctx, answer, None, [])
    return {"answer": answer}


def _finish(ctx: RunContext, answer: str, goal_met: bool | None, criteria: list[dict]) -> None:
    """에이전트가 끝난 뒤: 최종 안전장치(에이전트 밖)를 거쳐 결과를 내보낸다."""
    answer, cited = guardrail(ctx, answer)
    ctx.emit("sources", items=[s.to_dict() | {"cited": s.id in cited} for s in ctx.registry.all()])
    ctx.emit("final", answer_md=answer, cited_ids=cited, goal_met=goal_met, criteria=criteria,
             stats=ctx.stats() | {"cited": len(cited)})


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [("define_goal", define_goal), ("direct_answer", direct_answer), ("plan", plan),
                     ("pick_step", pick_step), ("act", act), ("tools", tools), ("check", check),
                     ("write", write), ("verify", verify), ("evaluate", evaluate)]:
        g.add_node(name, fn)

    g.add_edge(START, "define_goal")
    g.add_conditional_edges("define_goal", route_after_goal, ["plan", "direct_answer"])
    g.add_edge("plan", "pick_step")
    g.add_conditional_edges("pick_step", route_next, ["act", "write"])
    g.add_conditional_edges("act", route_after_act, ["tools", "check"])
    g.add_edge("tools", "check")
    g.add_conditional_edges("check", route_next, ["act", "pick_step", "write"])
    g.add_edge("write", "verify")
    g.add_conditional_edges("verify", route_next, ["write", "evaluate"])
    g.add_conditional_edges("evaluate", route_after_evaluate, ["pick_step", END])
    g.add_edge("direct_answer", END)
    return g.compile()


GRAPH = build_graph()
