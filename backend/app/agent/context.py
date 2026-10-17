"""요청 1건 동안 노드와 도구가 함께 쓰는 실행 컨텍스트."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime

from ..config import Settings
from .documents import Document, Library
from .sources import SourceRegistry
from .usage import Usage, cost_usd, load_pricing


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
    agent_loops: int = 0  # 조사 단계의 '판단 → 도구 → 관찰' 반복 횟수
    web_search_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    cost: float = 0.0
    cost_known: bool = True  # 단가를 모르는 모델이 한 번이라도 쓰이면 False

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

    def record_llm_call(self, stage: str, purpose: str, model: str, usage: Usage, elapsed_ms: int,
                        web_search_calls: int = 0) -> None:
        """LLM 호출 1회의 토큰·비용을 누적하고 화면에 알린다."""
        self.llm_calls += 1
        self.web_search_calls += web_search_calls
        self.usage = self.usage + usage
        cost = cost_usd(self.pricing, model, usage, web_search_calls)
        if cost is None:
            self.cost_known = False
        else:
            self.cost += cost
        self.emit("llm_call", call_no=self.llm_calls, stage=stage, purpose=purpose, model=model,
                  usage=usage.to_dict(), web_search_calls=web_search_calls, cost_usd=cost, elapsed_ms=elapsed_ms,
                  total_usage=self.usage.to_dict(), total_cost_usd=self.cost if self.cost_known else None)

    def stats(self) -> dict:
        return {"elapsed_ms": self.elapsed_ms(), "llm_calls": self.llm_calls, "agent_loops": self.agent_loops,
                "tool_calls": self.tool_calls, "web_search_calls": self.web_search_calls,
                "sources": len(self.registry), "usage": self.usage.to_dict(),
                "cost_usd": self.cost if self.cost_known else None}

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.started_at) * 1000)
