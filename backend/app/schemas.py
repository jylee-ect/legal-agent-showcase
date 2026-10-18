from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Attachment(BaseModel):
    filename: str = Field(max_length=200)
    content: str = Field(max_length=100_000)


MODE_LABELS = {"loop": "🎓 교과서형 루프 (LangGraph)", "react": "⚡ 실전형 ReAct (create_agent)",
               "multi": "👥 멀티 에이전트 (팀장 + 전문가 4)"}
TRIGGER_LABELS = {"question": "💬 사용자 질문", "file": "📂 파일 도착", "schedule": "⏰ 정기 일정"}


class Trigger(BaseModel):
    """에이전트를 깨우는 사건. 질문만이 아니라 파일 도착, 정기 일정 등도 트리거가 된다."""
    type: Literal["question", "file", "schedule"] = "question"
    detail: str = Field(default="", max_length=1000)  # 예: "검토함 폴더에 '카페_근로계약서.txt' 도착"


class ChatRequest(BaseModel):
    question: str = Field(default="", max_length=4000)
    history: list[HistoryItem] = Field(default_factory=list)
    attachment: Attachment | None = None
    trigger: Trigger = Field(default_factory=Trigger)
    # loop: 교과서형 루프(LangGraph, 목표·계획·점검·판단이 단계로 보임) / react: 실전형 ReAct(create_agent)
    mode: Literal["loop", "react", "multi"] = "loop"

    @model_validator(mode="after")
    def _needs_question_or_event(self) -> "ChatRequest":
        if self.trigger.type == "question" and not self.question.strip():
            raise ValueError("질문이 비어 있습니다.")
        if self.trigger.type != "question" and not (self.trigger.detail.strip() or self.question.strip()):
            raise ValueError("트리거 내용(detail)이 비어 있습니다.")
        return self


class ProposeRequest(BaseModel):
    category: Literal["standard", "company"]
    filename: str = Field(max_length=200)
    content: str = Field(min_length=1, max_length=200_000)
    note: str = Field(default="", max_length=1000)


class CommitRequest(BaseModel):
    proposal_id: str
    llms_txt: str = Field(min_length=1, max_length=100_000)
