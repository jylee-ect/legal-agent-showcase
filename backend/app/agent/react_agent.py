"""실전형 ReAct 에이전트 — LangChain `create_agent`.

    트리거 → [ model ⇄ tools → (최종 답변 시도) → 근거 검증 ─문제→ model ] → 최종 안전장치(에이전트 밖)

교과서형 루프(graph.py)와 달리 목표·계획·점검·종료 판단이 그래프에 없다. 모두 모델이 스스로 한다.
계획은 모델이 `write_todos` 도구(TodoListMiddleware)로 직접 쓰고 고친다.

이 에이전트의 '안'을 보는 방법:
- ObserverMiddleware: 모델 호출(awrap_model_call)과 도구 호출(awrap_tool_call)을 감싸서
  호출마다 추론 요약, 결정(도구 호출 / 최종 답변), 도구 사용 이유·결과, 토큰·비용을 이벤트로 내보낸다.
- stream_mode="messages": 모델이 쓰는 글자를 실시간으로 받는다.
모델 내부의 원본 사고(chain-of-thought)는 API 가 공개하지 않으므로, 추론 '요약'까지만 볼 수 있다.

멀티 에이전트(multi_agent.py)도 이 파일의 ObserverMiddleware·VerifyMiddleware 를 그대로 쓴다.
"""

from __future__ import annotations

import time
from contextvars import ContextVar

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware, TodoListMiddleware, hook_config
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from . import graph as agent_graph
from . import prompts
from .context import CURRENT_AGENT, RunContext
from .tools import TOOLS_BY_NAME, ToolOutput, ToolSpec, available_tools
from .verification import MAX_VERIFY_RETRIES, check_grounding, check_research, feedback_message

MAX_MODEL_CALLS = 25  # 무한 반복 방지 (ModelCallLimitMiddleware)
MIDDLEWARE_INFO = [
    {"name": "ObserverMiddleware", "desc": "모델·도구 호출을 감싸 화면에 보여줌 (이 시연용)"},
    {"name": "VerifyMiddleware", "desc": "최종 답변 시도를 근거 검증하고, 문제가 있으면 모델로 되돌려 고치게 함"},
    {"name": "TodoListMiddleware", "desc": "write_todos 도구 제공 — 모델이 할 일 목록을 직접 관리"},
    {"name": "ModelCallLimitMiddleware", "desc": f"모델 호출 {MAX_MODEL_CALLS}회 한도"},
]

# 지금 실행 중인 도구 호출 ID — 도구 함수가 결과(요약·출처)를 이 ID 로 남긴다
_CURRENT_CALL: ContextVar[str | None] = ContextVar("current_tool_call", default=None)


def _as_langchain_tool(ctx: RunContext, spec: ToolSpec) -> StructuredTool:
    async def run(**kwargs) -> str:
        try:
            out = await spec.run(ctx, spec.args_model.model_validate(kwargs))
        except Exception as e:  # 도구 실패는 모델이 보고 대처하도록 결과로 돌려준다
            out = ToolOutput(f"도구 실행 실패: {type(e).__name__}: {e}", f"실패: {e}", ok=False)
        ctx.tool_outputs[_CURRENT_CALL.get()] = out
        left = max(ctx.settings.max_tool_calls - ctx.tool_calls, 0)
        return f"{out.text}\n\n(남은 도구 호출 예산: {left}회)"

    return StructuredTool.from_function(coroutine=run, name=spec.name, description=spec.description,
                                        args_schema=spec.args_model)


class ObserverMiddleware(AgentMiddleware):
    """ReAct 루프의 매 모델 호출과 도구 호출을 화면 이벤트로 내보낸다. 도구 예산도 여기서 지킨다.

    agent: 멀티 에이전트에서 이 에이전트의 이름 (이벤트에 이름표로 붙음). 단일 ReAct 는 None.
    run_id: 전문가 에이전트가 어느 지시(팀장의 ask_* 호출)를 수행 중인지.
    tool_limit: 이 에이전트 한 번 실행에 쓸 수 있는 도구 수 (전체 예산과 별도).
    """

    def __init__(self, ctx: RunContext, agent: str | None = None, run_id: str | None = None,
                 tool_limit: int | None = None, top_level: bool = True):
        super().__init__()
        self.ctx = ctx
        self.agent = agent
        self.tag = {k: v for k, v in (("agent", agent), ("run", run_id)) if v}
        self.tool_limit = tool_limit
        self.top_level = top_level
        self.turn = 0
        self.tools_used = 0
        self.source_ids: list[str] = []

    async def awrap_model_call(self, request, handler):
        ctx = self.ctx
        self.turn += 1
        if self.top_level:
            ctx.agent_loops = self.turn
        ctx.emit("react_model_start", n=self.turn, messages=len(request.messages), **self.tag)
        started = time.monotonic()
        response = await handler(request)
        msg = next((m for m in reversed(response.result) if isinstance(m, AIMessage)), None)
        if msg is not None:
            agent_graph._record(ctx, "react", f"{self.agent + ' ' if self.agent else ''}모델 호출 #{self.turn}",
                                msg, started, agent=self.agent)
            text = agent_graph._reasoning_text(msg)
            if text:
                ctx.emit("reasoning", stage="react", text=text, **self.tag)
            calls = [c["name"] for c in msg.tool_calls]
            ctx.emit("react_model_end", n=self.turn, tool_calls=calls, final=not calls, text=msg.text.strip(),
                     **self.tag)
        return response

    async def awrap_tool_call(self, request, handler):
        ctx = self.ctx
        call = request.tool_call
        name, args = call["name"], dict(call["args"])
        if name == "write_todos":  # 모델이 스스로 세우고 고치는 계획
            ctx.emit("todos", turn=self.turn, todos=args.get("todos", []), **self.tag)
            return await handler(request)

        reason = args.pop("reason", "")
        if name.startswith("ask_"):  # 멀티 에이전트: 팀장이 전문가에게 일을 맡김
            ctx.emit("handoff", run=call["id"], to=name.removeprefix("ask_"), task=args.get("task", ""),
                     reason=reason, turn=self.turn, **{k: v for k, v in self.tag.items() if k != "run"})
            started = time.monotonic()
            token = _CURRENT_CALL.set(call["id"])
            try:
                result = await handler(request)
            finally:
                _CURRENT_CALL.reset(token)
            ctx.emit("handoff_done", run=call["id"], to=name.removeprefix("ask_"),
                     elapsed_ms=int((time.monotonic() - started) * 1000))
            return result

        spec = TOOLS_BY_NAME.get(name)
        ctx.emit("tool_call", id=call["id"], tool=name, label=spec.label if spec else name, args=args,
                 reason=reason, step=self.turn, **self.tag)
        started = time.monotonic()
        over_total = ctx.tool_calls >= ctx.settings.max_tool_calls
        over_agent = self.tool_limit is not None and self.tools_used >= self.tool_limit
        if over_total or over_agent:
            out = ToolOutput("도구 호출 예산을 모두 써서 실행하지 않았습니다. 지금까지 모은 자료로 답하세요.",
                             "예산 초과로 실행하지 않음", ok=False)
            result = ToolMessage(out.text, tool_call_id=call["id"], name=name)
        else:
            ctx.count_tool_call(self.agent)
            self.tools_used += 1
            call_token = _CURRENT_CALL.set(call["id"])
            agent_token = CURRENT_AGENT.set(self.agent) if self.agent else None
            try:
                result = await handler(request)
            finally:
                _CURRENT_CALL.reset(call_token)
                if agent_token is not None:
                    CURRENT_AGENT.reset(agent_token)
            out = ctx.tool_outputs.pop(call["id"], None) or ToolOutput(
                str(getattr(result, "content", ""))[:500], "도구 오류", ok=False)
        self.source_ids += [i for i in out.source_ids if i not in self.source_ids]
        ctx.emit("tool_result", id=call["id"], tool=name, ok=out.ok, summary=out.summary,
                 source_ids=out.source_ids, preview=out.preview[:4000],
                 elapsed_ms=int((time.monotonic() - started) * 1000), **self.tag)
        return result


class VerifyMiddleware(AgentMiddleware):
    """근거 검증 (에이전트 안): 모델이 도구 없이 최종 답변을 내려는 순간 코드로 검증하고,
    문제가 있으면 피드백 메시지를 넣어 모델로 되돌린다(jump_to). 최대 MAX_VERIFY_RETRIES 회."""

    def __init__(self, ctx: RunContext, observer: ObserverMiddleware):
        super().__init__()
        self.ctx = ctx
        self.observer = observer
        self.rounds = 0

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        last = state["messages"][-1] if state.get("messages") else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None  # 도구를 부르는 중이면 아직 답변이 아니다
        result = check_grounding(self.ctx, last.text)
        problems = result["problems"] + check_research(self.ctx)
        retry = bool(problems) and self.rounds < MAX_VERIFY_RETRIES
        self.ctx.emit("verify", ok=not problems, cited=result["cited"], unknown=result["unknown"],
                      unused=result["unused"], problems=problems, retry=retry, attempt=self.rounds + 1,
                      where="react", turn=self.observer.turn, **self.observer.tag)
        if not retry:
            return None
        self.rounds += 1
        return {"jump_to": "model", "messages": [HumanMessage(feedback_message(problems))]}


def _user_message(ctx: RunContext, state: dict) -> str:
    return (f"[트리거]\n{agent_graph._trigger_text(state)}\n\n"
            f"[첨부 계약서]\n{agent_graph._attachment_brief(ctx)}\n\n"
            f"[이전 대화]\n{agent_graph._history_text(state.get('history', []))}")


async def stream_agent(ctx: RunContext, agent, state: dict, observer: ObserverMiddleware) -> str:
    """에이전트를 실행하며 최상위 모델의 글자를 실시간으로 내보내고, 마지막 답변을 돌려준다."""
    final_state: dict = {}
    async for mode, data in agent.astream({"messages": [HumanMessage(_user_message(ctx, state))]},
                                          stream_mode=["messages", "values"], config={"recursion_limit": 120}):
        if mode == "messages":
            chunk, meta = data
            if meta.get("langgraph_node") == "model" and isinstance(chunk, AIMessageChunk) and chunk.text:
                ctx.emit("token", text=chunk.text, turn=observer.turn)
        else:
            final_state = data
    last = next((m for m in reversed(final_state.get("messages", [])) if isinstance(m, AIMessage)), None)
    answer = last.text.strip() if last is not None else ""
    return answer or "_(모델 호출 한도에 도달해 최종 답변을 만들지 못했습니다.)_"


async def run_react(ctx: RunContext, state: dict) -> None:
    observer = ObserverMiddleware(ctx)
    agent = create_agent(
        model=agent_graph.llm_factory(ctx.settings),
        tools=[_as_langchain_tool(ctx, spec) for spec in available_tools(ctx)],
        system_prompt=prompts.REACT_SYSTEM.format(budget=ctx.settings.max_tool_calls, llms=ctx.library.index_text),
        middleware=[observer, VerifyMiddleware(ctx, observer), TodoListMiddleware(),
                    ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS, exit_behavior="end")],
        name="react_agent",
    )
    seen = [{"label": "트리거", "detail": agent_graph._trigger_text(state).replace("\n", " · ")},
            {"label": "시스템 프롬프트", "detail": "역할·도구 사용 규칙·참고 문서 목록(llms.txt)·답변 규칙"},
            {"label": "도구", "detail": ", ".join(t.name for t in available_tools(ctx)) + ", write_todos"}]
    if ctx.attachment:
        seen.append({"label": "첨부 계약서 목차", "detail": f"{ctx.attachment.title} · {len(ctx.attachment.sections)}개 조항"})

    ctx.stage("react", "start")
    ctx.emit("context", items=seen)
    answer = await stream_agent(ctx, agent, state, observer)
    ctx.stage("react", "end")
    agent_graph._finish(ctx, answer, None, [])  # 최종 안전장치(에이전트 밖)를 거쳐 결과를 낸다
