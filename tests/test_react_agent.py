"""실전형 ReAct(create_agent) 모드: 가짜 채팅 모델로 이벤트를 확인한다 (API 키·네트워크 불필요)."""

import asyncio
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import graph as g
from app.agent import runner
from app.config import get_settings
from app.schemas import Attachment, ChatRequest

CONTRACT = "제1조(목적) 앱 개발.\n제8조(지식재산권) 기존 보유 기술을 포함한 모든 권리는 갑에게 귀속된다.\n"


class FakeToolModel(BaseChatModel):
    """create_agent 가 쓰는 bind_tools 를 지원하는 가짜 모델. responses 를 차례로 돌려준다."""
    responses: list
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _call(call_id, name, **args):
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def _run(monkeypatch, responses, req, budget=None):
    settings = replace(get_settings(), openai_api_key="sk-test", law_api_oc=None,
                       max_tool_calls=budget or get_settings().max_tool_calls)
    model = FakeToolModel(responses=responses)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(g, "llm_factory", lambda s: model)

    async def collect():
        return [e async for e in runner.run_agent(req)]

    return asyncio.run(collect()), model


def _req(question="지식재산권 조항 괜찮나요?"):
    return ChatRequest(question=question, mode="react", attachment=Attachment(filename="계약.txt", content=CONTRACT))


def test_react_loop_emits_model_turns_todos_tools_and_verified_answer(monkeypatch):
    todos = [{"content": "첨부 계약서 제8조 읽기", "status": "in_progress"},
             {"content": "표준계약서와 비교", "status": "pending"}]
    responses = [
        AIMessage(content="", tool_calls=[_call("t1", "write_todos", todos=todos),
                                          _call("c1", "read_uploaded_contract", reason="조항 확인", sections=["제8조"])]),
        AIMessage(content="표준계약서와 비교하겠습니다.", tool_calls=[
            _call("c2", "read_reference_sections", reason="기준 비교", doc_id="software_service", sections=["제11조"])]),
        AIMessage(content="## 한눈에 보기\n불리합니다 [U1][C1]. 없는 근거 [L9]."),  # 근거 검증에서 되돌아감
        AIMessage(content="## 한눈에 보기\n불리합니다 [U1][C1]."),                  # 피드백 반영해 다시 씀
    ]
    events, model = _run(monkeypatch, responses, _req())
    types = [e["type"] for e in events]
    assert "error" not in types and types[-1] == "done"

    start = events[0]
    assert start["mode"] == "react" and start["mode_label"].startswith("⚡")
    assert [m["name"] for m in start["middleware"]][:3] == ["ObserverMiddleware", "VerifyMiddleware",
                                                            "TodoListMiddleware"]

    # 근거 검증은 에이전트(react) 안에서, 최종 안전장치는 에이전트 밖에서
    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    assert stages == [("react", "start"), ("react", "end"), ("guardrail", "start"), ("guardrail", "end")]
    verifies = [e for e in events if e["type"] == "verify"]
    assert [(v["where"], v["retry"], v["unknown"], v["turn"]) for v in verifies] == [
        ("react", True, ["L9"], 3), ("react", False, [], 4)]

    ends = [e for e in events if e["type"] == "react_model_end"]
    assert [(e["n"], e["tool_calls"], e["final"]) for e in ends] == [
        (1, ["write_todos", "read_uploaded_contract"], False),
        (2, ["read_reference_sections"], False),
        (3, [], True),
        (4, [], True)]
    assert ends[1]["text"] == "표준계약서와 비교하겠습니다."
    assert [e["n"] for e in events if e["type"] == "react_model_start"] == [1, 2, 3, 4]

    assert next(e for e in events if e["type"] == "todos")["todos"] == todos
    calls = [e for e in events if e["type"] == "tool_call"]
    assert [(c["tool"], c["step"], c["reason"]) for c in calls] == [
        ("read_uploaded_contract", 1, "조항 확인"), ("read_reference_sections", 2, "기준 비교")]
    assert "reason" not in calls[0]["args"]
    results = {e["id"]: e for e in events if e["type"] == "tool_result"}
    assert results["c1"]["source_ids"] == ["U1"] and results["c2"]["source_ids"] == ["C1"]

    final = next(e for e in events if e["type"] == "final")
    assert "[U1][C1]" in final["answer_md"] and "[L9]" not in final["answer_md"]
    assert final["goal_met"] is None
    assert final["stats"]["agent_loops"] == 4 and final["stats"]["tool_calls"] == 2
    assert model.calls == 4
    assert next(e for e in events if e["type"] == "guardrail")["removed"] == []


def test_react_guardrail_removes_what_the_agent_failed_to_fix(monkeypatch):
    responses = [
        AIMessage(content="", tool_calls=[_call("c1", "read_uploaded_contract", reason="확인", sections=["제8조"])]),
        AIMessage(content="불리합니다 [U1][L9]."),  # 검증에서 되돌아감
        AIMessage(content="그래도 불리합니다 [U1][L9]."),  # 또 못 고침 → 재시도 한도 → 안전장치가 제거
    ]
    events, _ = _run(monkeypatch, responses, _req())
    assert [v["retry"] for v in events if v["type"] == "verify"] == [True, False]
    assert next(e for e in events if e["type"] == "guardrail")["removed"] == ["L9"]
    assert "[L9]" not in next(e for e in events if e["type"] == "final")["answer_md"]


def test_react_verify_catches_precedent_searched_but_not_read(monkeypatch):
    from app.agent import law_api

    async def fake_search(oc, query, display=6):
        return [{"id": "118858", "case_no": "88가합46941", "case_name": "대여금", "court": "서울민사지법",
                 "date": "1990.06.29", "kind": "민사"}]

    monkeypatch.setattr(law_api, "search_precedents", fake_search)
    responses = [
        AIMessage(content="", tool_calls=[_call("c1", "search_precedents", reason="판례 찾기", query="교육비 반환")]),
        AIMessage(content="교육비는 내지 않아도 됩니다."),
        AIMessage(content="판례는 읽지 못했습니다. 일반적으로 교육비 배상은 제한됩니다."),
    ]
    settings_law = {"law_api_oc": "test-oc"}
    settings = replace(get_settings(), openai_api_key="sk-test", **settings_law)
    model = FakeToolModel(responses=responses)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(g, "llm_factory", lambda s: model)

    async def collect():
        return [e async for e in runner.run_agent(_req())]

    events = asyncio.run(collect())
    first = next(e for e in events if e["type"] == "verify")
    assert first["retry"] and any("판례를 검색했지만" in p for p in first["problems"])
    assert model.calls == 3


def test_react_tool_budget_is_enforced(monkeypatch):
    responses = [
        AIMessage(content="", tool_calls=[_call("c1", "read_uploaded_contract", reason="1", sections=["제1조"]),
                                          _call("c2", "read_uploaded_contract", reason="2", sections=["제8조"])]),
        AIMessage(content="답변 [U1]"),
    ]
    events, _ = _run(monkeypatch, responses, _req(), budget=1)
    results = [e for e in events if e["type"] == "tool_result"]
    assert sorted(r["ok"] for r in results) == [False, True]
    assert any("예산" in r["summary"] for r in results)
    assert next(e for e in events if e["type"] == "final")["stats"]["tool_calls"] == 1


def test_react_answers_small_talk_without_tools(monkeypatch):
    events, _ = _run(monkeypatch, [AIMessage(content="안녕하세요! 계약서를 첨부해 주세요.")], _req("안녕?"))
    assert "tool_call" not in [e["type"] for e in events]
    ends = [e for e in events if e["type"] == "react_model_end"]
    assert len(ends) == 1 and ends[0]["final"]
    assert next(e for e in events if e["type"] == "final")["answer_md"].startswith("안녕하세요")
