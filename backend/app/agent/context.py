"""요청 1건 동안 노드와 도구가 함께 쓰는 실행 컨텍스트."""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime

from ..config import Settings
from .documents import Document, Library
from .sources import SourceRegistry
from .usage import Usage, cost_usd, load_pricing

# 멀티 에이전트: 지금 이 코드를 실행 중인 에이전트 (웹검색 내부 호출처럼 에이전트를 직접 넘기기 어려운 곳에서 쓴다)
CURRENT_AGENT: ContextVar[str | None] = ContextVar("current_agent", default=None)


@dataclass
class RunContext:
    settings: Settings
    library: Library
    attachment: Document | None
    queue: asyncio.Queue
    registry: SourceRegistry = field(default_factory=SourceRegistry)
    plan_status: dict[int, str] = field(default_factory=dict)  # 단계번호 → pending|in_progress|done|skipped
    started_at: float = field(default_factory=time.monotonic)
    pricing: dict = field(default_factory=load_pricing)
    llm_calls: int = 0
    tool_calls: int = 0
    agent_loops: int = 0  # 교과서형: 계획 단계 실행 횟수 / 실전형·멀티: (최상위) 모델 호출 횟수
    tool_outputs: dict = field(default_factory=dict)  # 실전형 ReAct: 도구 호출 ID → ToolOutput (화면용 요약·출처)
    precedent_searches: int = 0  # 판례 검색 횟수 (실전형 근거 검증: '검색만 하고 안 읽음' 확인용)
    web_search_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    cost_known: bool = True  # 단가를 모르는 모델이 한 번이라도 쓰이면 False
    by_agent: dict = field(default_factory=dict)  # 멀티 에이전트: 에이전트별 LLM·도구 호출, 토큰, 비용

    def emit(self, type_: str, **data) -> None:
        # t: 실행 시작부터 경과(ms), ts: 실제 시각
        self.queue.put_nowait({"type": type_, "t": self.elapsed_ms(),
                               "ts": datetime.now().astimezone().isoformat(timespec="milliseconds"), **data})

    def stage(self, stage: str, status: str) -> None:
        self.emit("stage", stage=stage, status=status)

    def set_step(self, step_no: int, status: str) -> None:
        if step_no not in self.plan_status or self.plan_status[step_no] == status:
            return
        self.plan_status[step_no] = status
        self.emit("plan_update", step_no=step_no, status=status)

    def _agent_stats(self, agent: str) -> dict:
        return self.by_agent.setdefault(agent, {"llm_calls": 0, "tool_calls": 0, "usage": Usage(), "cost": 0.0})

    def count_tool_call(self, agent: str | None = None) -> None:
        self.tool_calls += 1
        if agent := agent or CURRENT_AGENT.get():
            self._agent_stats(agent)["tool_calls"] += 1

    def record_llm_call(self, stage: str, purpose: str, model: str, usage: Usage, elapsed_ms: int,
                        web_search_calls: int = 0, agent: str | None = None) -> None:
        """LLM 호출 1회의 토큰·비용을 누적하고 화면에 알린다."""
        agent = agent or CURRENT_AGENT.get()
        self.llm_calls += 1
        self.web_search_calls += web_search_calls
        self.usage = self.usage + usage
        cost = cost_usd(self.pricing, model, usage, web_search_calls)
        if cost is None:
            self.cost_known = False
        else:
            self.cost += cost
        if agent:
            a = self._agent_stats(agent)
            a["llm_calls"] += 1
            a["usage"] = a["usage"] + usage
            a["cost"] += cost or 0.0
        extra = {"agent": agent} if agent else {}
        self.emit("llm_call", call_no=self.llm_calls, stage=stage, purpose=purpose, model=model,
                  usage=usage.to_dict(), web_search_calls=web_search_calls, cost_usd=cost, elapsed_ms=elapsed_ms,
                  total_usage=self.usage.to_dict(), total_cost_usd=self.cost if self.cost_known else None, **extra)

    def stats(self) -> dict:
        out = {"elapsed_ms": self.elapsed_ms(), "llm_calls": self.llm_calls, "agent_loops": self.agent_loops,
               "tool_calls": self.tool_calls, "web_search_calls": self.web_search_calls,
               "sources": len(self.registry), "usage": self.usage.to_dict(),
               "cost_usd": self.cost if self.cost_known else None}
        if self.by_agent:
            out["by_agent"] = {k: {"llm_calls": v["llm_calls"], "tool_calls": v["tool_calls"],
                                   "usage": v["usage"].to_dict(), "cost_usd": v["cost"]}
                               for k, v in self.by_agent.items()}
        return out

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)
