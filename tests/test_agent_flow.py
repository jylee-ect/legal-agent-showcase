"""가짜 LLM 으로 그래프 전체를 돌려 이벤트 순서와 출처 검증을 확인한다 (API 키·네트워크 불필요)."""

import asyncio
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda

from app.agent import graph, runner
from app.config import get_settings
from app.schemas import Attachment, ChatRequest

CONTRACT = "제1조(목적) 앱 개발.\n제8조(지식재산권) 기존 보유 기술을 포함한 모든 권리는 갑에게 귀속된다.\n"


class FakeLLM:
    def __init__(self, analysis, plan=None, research=(), answer=""):
        self.parsed = {"Analysis": analysis, "Plan": plan}
        self.research = list(research)
        self.answer = answer

    def with_structured_output(self, schema, include_raw=False):
        return RunnableLambda(lambda _: {"raw": AIMessage(content=""), "parsed": self.parsed[schema.__name__],
                                         "parsing_error": None})

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages):
        return self.research.pop(0)

    async def astream(self, messages):
        for i in range(0, len(self.answer), 8):
            yield AIMessageChunk(content=self.answer[i:i + 8])


def _analysis(route="research", use_attachment=True):
    return graph.Analysis(
        question_type="contract_review" if route == "research" else "off_topic", summary="요지", issues=["지식재산권"],
        use_attachment=use_attachment, attachment_reason="질문이 계약서 조항을 묻는다", candidate_docs=["software_service", "bogus"],
        needs_law=False, needs_precedent=False, needs_web=False, route=route, thought="판단 근거",
    )


def _run(monkeypatch, fake, req):
    settings = replace(get_settings(), openai_api_key="sk-test", law_api_oc=None)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(graph, "llm_factory", lambda s: fake)

    async def collect():
        return [e async for e in runner.run_agent(req)]

    return asyncio.run(collect())


def test_research_flow_emits_steps_and_verifies_citations(monkeypatch):
    plan = graph.Plan(strategy="첨부 조항과 표준계약서를 비교", steps=[
        graph.PlanStep(no=1, goal="첨부 계약서 지식재산권 조항 읽기", tool="read_uploaded_contract"),
        graph.PlanStep(no=2, goal="표준계약서 지식재산권 조항 읽기", tool="read_reference_sections"),
        graph.PlanStep(no=3, goal="웹 검색", tool="web_search"),
    ])
    research = [
        AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "read_uploaded_contract", "args": {"reason": "문제 조항 확인", "plan_step": 1, "sections": ["제8조"]}},
            {"id": "c2", "name": "read_reference_sections",
             "args": {"reason": "기준 비교", "plan_step": 2, "doc_id": "software_service", "sections": ["제11조"]}},
        ]),
        AIMessage(content="조사 메모: [U1] 은 [C1] 보다 불리하다."),
    ]
    answer = "## 한눈에 보기\n기존 보유 기술까지 넘기는 조항은 불리합니다 [U1, C1]. 없는 근거 [L7]."
    fake = FakeLLM(_analysis(), plan, research, answer)
    req = ChatRequest(question="지식재산권 조항 괜찮나요?", attachment=Attachment(filename="계약.txt", content=CONTRACT))

    events = _run(monkeypatch, fake, req)
    types = [e["type"] for e in events]

    assert types[0] == "run_start" and types[-1] == "done" and "error" not in types
    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    assert stages == [("analyze", "start"), ("analyze", "end"), ("plan", "start"), ("plan", "end"),
                      ("research", "start"), ("research", "end"), ("write", "start"), ("write", "end"),
                      ("verify", "start"), ("verify", "end")]

    analysis = next(e for e in events if e["type"] == "analysis")["data"]
    assert analysis["candidate_docs"] == ["software_service"]  # 없는 ID 는 걸러진다

    calls = [e for e in events if e["type"] == "tool_call"]
    assert [c["reason"] for c in calls] == ["문제 조항 확인", "기준 비교"]
    assert "reason" not in calls[0]["args"]
    results = {e["id"]: e for e in events if e["type"] == "tool_result"}
    assert results["c1"]["source_ids"] == ["U1"] and results["c2"]["source_ids"] == ["C1"]

    steps = [(e["step_no"], e["status"]) for e in events if e["type"] == "plan_update"]
    assert (1, "done") in steps and (2, "done") in steps and (3, "skipped") in steps

    verify = next(e for e in events if e["type"] == "verify")
    assert verify["cited"] == ["U1", "C1"] and verify["unknown"] == ["L7"] and not verify["ok"]
    final = next(e for e in events if e["type"] == "final")
    assert "[U1][C1]" in final["answer_md"] and "[L7]" not in final["answer_md"]
    assert final["stats"]["tool_calls"] == 2
    assert "".join(e["text"] for e in events if e["type"] == "token") == answer


def test_off_topic_question_skips_research(monkeypatch):
    fake = FakeLLM(_analysis(route="direct", use_attachment=False), answer="저는 계약서 검토 에이전트예요.")
    req = ChatRequest(question="오늘 점심 뭐 먹지?", attachment=Attachment(filename="계약.txt", content=CONTRACT))
    events = _run(monkeypatch, fake, req)
    types = [e["type"] for e in events]
    assert "plan" not in types and "tool_call" not in types and "error" not in types
    assert next(e for e in events if e["type"] == "route")["path"] == "direct"
    assert next(e for e in events if e["type"] == "final")["answer_md"] == "저는 계약서 검토 에이전트예요."


def test_missing_openai_key_reports_error(monkeypatch):
    settings = replace(get_settings(), openai_api_key=None)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)

    async def collect():
        return [e async for e in runner.run_agent(ChatRequest(question="안녕"))]

    events = asyncio.run(collect())
    assert [e["type"] for e in events] == ["run_start", "error", "done"]
    assert "GPT_API" in events[1]["message"]
