"""가짜 LLM 으로 에이전트 루프 전체를 돌려 이벤트 순서를 확인한다 (API 키·네트워크 불필요).

트리거 → 목표 정의 → 계획 → (단계 실행 → 점검·계획 수정)* → 답변 → 근거 검증 → 목표 달성 판단
"""

import asyncio
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from app.agent import graph as g
from app.agent import runner
from app.config import get_settings
from app.schemas import Attachment, ChatRequest, Trigger

CONTRACT = "제1조(목적) 앱 개발.\n제8조(지식재산권) 기존 보유 기술을 포함한 모든 권리는 갑에게 귀속된다.\n"


class FakeLLM:
    """구조화 출력은 스키마 이름별 대기열, 도구 결정은 actions, 답변은 answers 에서 차례로 꺼낸다."""

    def __init__(self, structured: dict, actions=(), answers=()):
        self.structured = {k: list(v) if isinstance(v, list) else [v] for k, v in structured.items()}
        self.actions = list(actions)
        self.answers = list(answers)

    def with_structured_output(self, schema, include_raw=False):
        def run(_):
            queue = self.structured[schema.__name__]
            parsed = queue.pop(0) if len(queue) > 1 else queue[0]
            return {"raw": AIMessage(content=""), "parsed": parsed, "parsing_error": None}
        return RunnableLambda(run)

    def bind_tools(self, tools, **kwargs):
        return self

    async def ainvoke(self, messages):
        return self.actions.pop(0)

    async def astream(self, messages):
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        for i in range(0, len(answer), 8):
            yield AIMessageChunk(content=answer[i:i + 8])


def _goal(route="research", criteria=("첨부 계약서 조항 확인", "표준계약서와 비교")):
    return g.Goal(trigger_summary="사용자 질문", question_type="contract_review" if route == "research" else "off_topic",
                  summary="요지", goal="지식재산권 조항이 불리한지 판단한다",
                  completion_criteria=list(criteria) if route == "research" else [], issues=["지식재산권"],
                  use_attachment=True, attachment_reason="질문이 조항을 묻는다",
                  candidate_docs=["software_service", "bogus"], route=route, thought="판단 근거")


def _check(decision, met=(), step_done=True, drop=(), add=(), note="관찰 [U1]"):
    crit = [g.CriterionCheck(id=i, status="met" if i in met else "unmet", note="") for i in (1, 2)]
    return g.Check(observation=note, step_done=step_done, criteria=crit, decision=decision, reason="이유",
                   drop_steps=list(drop), add_steps=list(add), insert_before=None)


def _eval(met, add=()):
    crit = [g.CriterionCheck(id=i, status="met" if met else ("met" if i == 1 else "partial"), note="") for i in (1, 2)]
    return g.Evaluation(criteria=crit, goal_met=met, missing=[] if met else ["비교 부족"], feedback="표준계약서를 더 확인",
                        add_steps=list(add))


def _tool(call_id, name, **args):
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {"reason": f"{name} 이유", **args}}])


def _run(monkeypatch, fake, req):
    settings = replace(get_settings(), openai_api_key="sk-test", law_api_oc=None)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(g, "llm_factory", lambda s: fake)

    async def collect():
        return [e async for e in runner.run_agent(req)]

    return asyncio.run(collect())


def _req(question="지식재산권 조항 괜찮나요?", **kw):
    return ChatRequest(question=question, attachment=Attachment(filename="계약.txt", content=CONTRACT), **kw)


def test_full_loop_with_plan_revision_and_goal_retry(monkeypatch):
    plan = g.Plan(strategy="조항 확인 후 비교", steps=[
        g.NewStep(goal="첨부 계약서 제8조 읽기", tool="read_uploaded_contract", criteria=[1]),
        g.NewStep(goal="쟁점 정리", tool="없음", criteria=[1]),
        g.NewStep(goal="웹 검색", tool="web_search", criteria=[2]),
    ])
    add_compare = g.NewStep(goal="표준계약서 제11조와 비교", tool="read_reference_sections", criteria=[2, 9])
    add_more = g.NewStep(goal="표준계약서 목차 재확인", tool="read_reference_outline", criteria=[2])
    fake = FakeLLM(
        structured={
            "Goal": _goal(), "Plan": plan,
            "Check": [
                _check("revise", met=(1,), drop=(3, 99), add=(add_compare,)),  # 1단계 후: 웹 검색 취소, 비교 단계 추가
                _check("continue", met=(1,)),                                  # 2단계(도구 없음) 후: 계속
                _check("continue", met=(1, 2)),                                # 4단계 후: 모두 충족 → 코드가 종료로 바꿈
                _check("continue", met=(1, 2)),                                # 추가 조사 단계 후
            ],
            "Evaluation": [_eval(False, add=(add_more,)), _eval(True)],
        },
        actions=[
            _tool("c1", "read_uploaded_contract", sections=["제8조"]),
            AIMessage(content="작업 메모로 충분해 도구 없이 정리합니다."),
            _tool("c2", "read_reference_sections", doc_id="software_service", sections=["제11조"]),
            _tool("c3", "read_reference_outline", doc_id="software_service"),
        ],
        answers=["첫 답변 [U1][L7]", "## 한눈에 보기\n불리합니다 [U1, C1]."],
    )
    events = _run(monkeypatch, fake, _req())
    types = [e["type"] for e in events]
    assert types[0] == "run_start" and types[-1] == "done" and "error" not in types

    # 첫 답변의 가짜 인용 [L7] → 근거 검증이 되돌려 보내 다시 작성 → 목표 미달 → 추가 조사 → 다시 작성 → 안전장치
    order = [e["stage"] for e in events if e["type"] == "stage" and e["status"] == "start"]
    assert order == ["goal", "plan", "execute", "write", "verify", "write", "verify", "evaluate",
                     "execute", "write", "verify", "evaluate", "guardrail"]

    start = events[0]
    assert start["trigger"]["type"] == "question" and start["trigger"]["label"] == "💬 사용자 질문"
    goal = next(e for e in events if e["type"] == "goal")
    assert [c["text"] for c in goal["criteria"]] == ["첨부 계약서 조항 확인", "표준계약서와 비교"]
    assert goal["data"]["candidate_docs"] == ["software_service"]

    # 계획 수정: 점검(check)에서 3번 취소 + 4번 추가, 목표 판단(evaluate)에서 5번 추가
    revisions = [e for e in events if e["type"] == "plan_revised"]
    assert [(r["by"], r["dropped"], [a["no"] for a in r["added"]]) for r in revisions] == \
        [("check", [3], [4]), ("evaluate", [], [5])]
    assert revisions[0]["added"][0]["criteria"] == [2]  # 없는 조건 번호(9)는 걸러진다

    assert [e["no"] for e in events if e["type"] == "step_start"] == [1, 2, 4, 5]
    checks = [e for e in events if e["type"] == "check"]
    assert [c["decision"] for c in checks] == ["revise", "continue", "finish", "finish"]
    assert checks[0]["source_ids"] == ["U1"]

    calls = [e for e in events if e["type"] == "tool_call"]
    assert [c["step"] for c in calls] == [1, 4, 5] and "reason" not in calls[0]["args"]
    assert {e["id"]: e["source_ids"] for e in events if e["type"] == "tool_result"}["c2"] == ["C1"]

    evals = [e for e in events if e["type"] == "evaluate"]
    assert [(e["goal_met"], e["retry"]) for e in evals] == [(False, True), (True, False)]
    verifies = [e for e in events if e["type"] == "verify"]
    assert [(v["unknown"], v["retry"], v["where"]) for v in verifies] == [
        (["L7"], True, "loop"), ([], False, "loop"), ([], False, "loop")]
    assert "L7" in verifies[0]["problems"][0]
    writes = [e["purpose"] for e in events if e["type"] == "llm_call" and e["stage"] == "write"]
    assert writes[1] == "답변 다시 작성 (근거 검증 피드백 반영)"
    assert next(e for e in events if e["type"] == "guardrail")["removed"] == []

    final = next(e for e in events if e["type"] == "final")
    assert final["goal_met"] is True and "[U1][C1]" in final["answer_md"]
    assert all(c["status"] == "met" for c in final["criteria"])
    assert final["stats"]["agent_loops"] == 4 and final["stats"]["tool_calls"] == 3


def test_goal_not_met_without_more_research_is_reported(monkeypatch):
    plan = g.Plan(strategy="조항 확인", steps=[g.NewStep(goal="제8조 읽기", tool="read_uploaded_contract", criteria=[1, 2])])
    fake = FakeLLM(structured={"Goal": _goal(), "Plan": plan, "Check": _check("finish", met=(1,)),
                               "Evaluation": _eval(False)},
                   actions=[_tool("c1", "read_uploaded_contract", sections=["제8조"])], answers=["답변 [U1]"])
    events = _run(monkeypatch, fake, _req())
    final = next(e for e in events if e["type"] == "final")
    assert final["goal_met"] is False
    assert "목표 중 확인하지 못한 부분" in final["answer_md"] and "표준계약서와 비교" in final["answer_md"]
    assert [e["stage"] for e in events if e["type"] == "stage" and e["status"] == "start"].count("execute") == 1


def test_step_retry_then_continue(monkeypatch):
    plan = g.Plan(strategy="조항 확인", steps=[g.NewStep(goal="제8조 읽기", tool="read_uploaded_contract", criteria=[1]),
                                              g.NewStep(goal="비교", tool="read_reference_sections", criteria=[2])])
    fake = FakeLLM(structured={"Goal": _goal(), "Plan": plan,
                               "Check": [_check("continue", step_done=False), _check("continue", met=(1,)),
                                         _check("finish", met=(1, 2))],
                               "Evaluation": _eval(True)},
                   actions=[_tool("c1", "read_uploaded_contract", sections=["제1조"]),
                            _tool("c2", "read_uploaded_contract", sections=["제8조"]),
                            _tool("c3", "read_reference_sections", doc_id="software_service", sections=["제11조"])],
                   answers=["답변 [U1][U2][C1]"])
    events = _run(monkeypatch, fake, _req())
    checks = [e for e in events if e["type"] == "check"]
    assert [c["decision"] for c in checks] == ["retry_step", "continue", "finish"]
    assert [e["no"] for e in events if e["type"] == "step_start"] == [1, 2]  # 같은 단계 재시도는 새 단계가 아님
    assert [c["step"] for c in events if c["type"] == "tool_call"] == [1, 1, 2]


def test_off_topic_question_goes_direct(monkeypatch):
    fake = FakeLLM(structured={"Goal": _goal(route="direct")}, answers=["저는 계약서 검토 에이전트예요."])
    events = _run(monkeypatch, fake, _req(question="오늘 점심 뭐 먹지?"))
    types = [e["type"] for e in events]
    assert "plan" not in types and "tool_call" not in types and "error" not in types
    assert next(e for e in events if e["type"] == "route")["path"] == "direct"
    assert next(e for e in events if e["type"] == "final")["answer_md"] == "저는 계약서 검토 에이전트예요."


def test_file_trigger_without_question_always_does_the_work(monkeypatch):
    plan = g.Plan(strategy="검토", steps=[g.NewStep(goal="전체 읽기", tool="read_uploaded_contract", criteria=[1, 2])])
    fake = FakeLLM(structured={"Goal": _goal(route="direct"), "Plan": plan, "Check": _check("finish", met=(1, 2)),
                               "Evaluation": _eval(True)},
                   actions=[_tool("c1", "read_uploaded_contract", sections=["전체"])], answers=["보고서 [U1]"])
    req = _req(question="", trigger=Trigger(type="file", detail="검토함 폴더에 '계약.txt' 도착"))
    events = _run(monkeypatch, fake, req)
    assert events[0]["trigger"] == {"type": "file", "label": "📂 파일 도착", "detail": "검토함 폴더에 '계약.txt' 도착"}
    assert next(e for e in events if e["type"] == "route")["path"] == "research"  # 모델이 direct 라 해도 작업 수행
    context = next(e for e in events if e["type"] == "context")["items"][0]
    assert context["label"] == "트리거" and "파일 도착" in context["detail"]
    assert next(e for e in events if e["type"] == "final")["goal_met"] is True


def test_request_validation():
    with pytest.raises(ValidationError):
        ChatRequest(question="  ")
    with pytest.raises(ValidationError):
        ChatRequest(trigger=Trigger(type="schedule"))
    assert ChatRequest(trigger=Trigger(type="schedule", detail="매월 정기 점검")).question == ""


def test_missing_openai_key_reports_error(monkeypatch):
    settings = replace(get_settings(), openai_api_key=None)
    monkeypatch.setattr(runner, "get_settings", lambda: settings)

    async def collect():
        return [e async for e in runner.run_agent(ChatRequest(question="안녕"))]

    events = asyncio.run(collect())
    assert [e["type"] for e in events] == ["run_start", "error", "done"]
    assert "GPT_API" in events[1]["message"]
