from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class Attachment(BaseModel):
    filename: str = Field(max_length=200)
    content: str = Field(max_length=100_000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[HistoryItem] = Field(default_factory=list)
    attachment: Attachment | None = None


class ProposeRequest(BaseModel):
    category: Literal["standard", "company"]
    filename: str = Field(max_length=200)
    content: str = Field(min_length=1, max_length=200_000)
    note: str = Field(default="", max_length=1000)


class CommitRequest(BaseModel):
    proposal_id: str
    llms_txt: str = Field(min_length=1, max_length=100_000)
