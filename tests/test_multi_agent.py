"""멀티 에이전트(팀장 + 전문가 4): 시스템 프롬프트로 역할을 구분하는 가짜 모델로 이벤트를 확인한다."""

import asyncio
import shutil
import threading
from dataclasses import replace

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.agent import graph as g
from app.agent import law_api, runner
from app.agent.documents import load_library
from app.config import DATA_DIR, get_settings
from app.schemas import Attachment, ChatRequest

CONTRACT = "제1조(목적) 앱 개발.\n제8조(지식재산권) 기존 보유 기술을 포함한 모든 권리는 갑에게 귀속된다.\n"
ROLE_MARKERS = {"supervisor": "팀장 에이전트", "law": "법령조사 에이전트", "precedent": "판례조사 에이전트",
                "contract": "계약서분석 에이전트", "company": "회사기준 에이전트"}


class RoleModel(BaseChatModel):
    """시스템 프롬프트를 보고 어느 에이전트인지 알아낸 뒤, 그 역할의 각본에서 다음 응답을 꺼낸다."""
    scripts: dict
    calls: dict = {}

    @property
    def _llm_type(self) -> str:
        return "role-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        system = str(messages[0].content) if messages else ""
        role = next(r for r, m in ROLE_MARKERS.items() if m in system and (r == "supervisor") == ("팀장" in m))
        with _lock:
            n = self.calls.get(role, 0)
            self.calls[role] = n + 1
        script = self.scripts[role]
        return ChatResult(generations=[ChatGeneration(message=script[min(n, len(script) - 1)])])


_lock = threading.Lock()


def _call(call_id, name, **args):
    return {"id": call_id, "name": name, "args": args, "type": "tool_call"}


def _patch_law(monkeypatch):
    async def get_article(oc, law_name, article):
        return {"law_name": law_name, "article": article, "title": "저작재산권의 양도", "text": f"{article} 본문",
                "effective": "20260101", "url": "https://www.law.go.kr/법령/저작권법/제45조"}

    async def search_precedents(oc, query, display=6):
        return [{"id": "118858", "case_no": "88가합46941", "case_name": "대여금", "court": "서울민사지법",
                 "date": "1990.06.29", "kind": "민사"}]

    async def get_precedent(oc, prec_id, max_body=2500):
        return {"id": prec_id, "case_no": "88가합46941", "case_name": "대여금", "court": "서울민사지법",
                "date": "1990.06.29", "holding": "교육비 반환 약정의 효력", "summary": "위약금 예정은 무효", "ref_articles": "",
                "body_excerpt": "", "url": "https://www.law.go.kr/LSW/precInfoP.do?precSeq=118858"}

    monkeypatch.setattr(law_api, "get_article", get_article)
    monkeypatch.setattr(law_api, "search_precedents", search_precedents)
    monkeypatch.setattr(law_api, "get_precedent", get_precedent)


def _run(monkeypatch, scripts, req, **settings_kw):
    settings = replace(get_settings(), openai_api_key="sk-test", law_api_oc="test-oc", **settings_kw)
    model = RoleModel(scripts=scripts, calls={})
    monkeypatch.setattr(runner, "get_settings", lambda: settings)
    monkeypatch.setattr(g, "llm_factory", lambda s: model)
    _patch_law(monkeypatch)

    async def collect():
        return [e async for e in runner.run_agent(req)]

    return asyncio.run(collect()), model


def _req(question="지식재산권 조항 괜찮나요?"):
    return ChatRequest(question=question, mode="multi", attachment=Attachment(filename="계약.txt", content=CONTRACT))


def test_supervisor_delegates_in_parallel_and_cites_specialist_sources(monkeypatch):
    scripts = {
        "supervisor": [
            AIMessage(content="", tool_calls=[
                _call("h1", "ask_contract", reason="계약서 비교 필요", task="제8조 지식재산권 조항을 표준계약서와 비교"),
                _call("h2", "ask_law", reason="법령 확인 필요", task="저작권법상 저작재산권 양도 규정 확인")]),
            AIMessage(content="", tool_calls=[
                _call("h3", "ask_precedent", reason="판례 확인", task="기존 기술 귀속 관련 판례 확인")]),
            AIMessage(content="## 한눈에 보기\n기존 기술까지 넘기는 조항은 불리합니다 [U1][L1][P1]."),
        ],
        "contract": [AIMessage(content="", tool_calls=[_call("c1", "read_uploaded_contract", reason="조항 읽기",
                                                             sections=["제8조"])]),
                     AIMessage(content="- 기존 보유 기술까지 갑에게 귀속 [U1]")],
        "law": [AIMessage(content="", tool_calls=[_call("l1", "get_law_article", reason="원문 확인",
                                                        law_name="저작권법", article="제45조")]),
                AIMessage(content="- 저작재산권은 양도할 수 있음 [L1]")],
        "precedent": [AIMessage(content="", tool_calls=[_call("p1", "search_precedents", reason="찾기", query="교육비")]),
                      AIMessage(content="", tool_calls=[_call("p2", "read_precedent", reason="읽기",
                                                              precedent_id="118858")]),
                      AIMessage(content="- 위약금 예정은 무효 [P1]")],
        "company": [AIMessage(content="- 등록된 회사 문서 없음")],
    }
    events, model = _run(monkeypatch, scripts, _req())
    types = [e["type"] for e in events]
    assert "error" not in types and types[-1] == "done"

    start = events[0]
    assert start["mode"] == "multi" and start["mode_label"].startswith("👥")
    stages = [(e["stage"], e["status"]) for e in events if e["type"] == "stage"]
    assert stages == [("multi", "start"), ("multi", "end"), ("guardrail", "start"), ("guardrail", "end")]
    team = next(e for e in events if e["type"] == "team")
    assert [a["key"] for a in team["agents"]] == ["supervisor", "law", "precedent", "contract", "company"]

    handoffs = [e for e in events if e["type"] == "handoff"]
    assert [(h["to"], h["run"], h["turn"], h["agent"]) for h in handoffs] == [
        ("contract", "h1", 1, "supervisor"), ("law", "h2", 1, "supervisor"), ("precedent", "h3", 2, "supervisor")]
    reports = {e["run"]: e for e in events if e["type"] == "agent_report"}
    assert reports["h1"]["agent"] == "contract" and reports["h1"]["source_ids"] == ["U1"]
    assert reports["h2"]["source_ids"] == ["L1"] and reports["h3"]["source_ids"] == ["P1"]
    assert reports["h3"]["model_calls"] == 3 and reports["h3"]["tool_calls"] == 2

    # 전문가의 도구 호출에는 누가·어느 지시인지 이름표가 붙는다
    calls = {e["id"]: e for e in events if e["type"] == "tool_call"}
    assert (calls["c1"]["agent"], calls["c1"]["run"]) == ("contract", "h1")
    assert (calls["p2"]["agent"], calls["p2"]["run"]) == ("precedent", "h3")
    sup_ends = [e for e in events if e["type"] == "react_model_end" and e.get("agent") == "supervisor"]
    assert [e["tool_calls"] for e in sup_ends] == [["ask_contract", "ask_law"], ["ask_precedent"], []]

    verify = next(e for e in events if e["type"] == "verify")
    assert verify["ok"] and verify["agent"] == "supervisor"
    final = next(e for e in events if e["type"] == "final")
    assert "[U1][L1][P1]" in final["answer_md"]
    by_agent = final["stats"]["by_agent"]
    assert set(by_agent) == {"supervisor", "contract", "law", "precedent"}
    assert by_agent["supervisor"]["llm_calls"] == 3 and by_agent["supervisor"]["tool_calls"] == 0
    assert by_agent["precedent"]["tool_calls"] == 2 and final["stats"]["tool_calls"] == 4
    assert final["stats"]["agent_loops"] == 3  # 팀장의 모델 호출 수
    assert model.calls == {"supervisor": 3, "contract": 2, "law": 2, "precedent": 3}


def test_company_agent_reads_company_standard(monkeypatch, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data, ignore=shutil.ignore_patterns(".history"))
    (data / "company").mkdir(exist_ok=True)
    (data / "company" / "review_rules.md").write_text(
        "# 외주 계약 검토 기준\n\n## 제3조(지식재산권)\n기존 보유 기술 이전 조항은 수용하지 않는다.\n\n## 제8조(승인)\n3억 이상은 대표 승인.\n",
        encoding="utf-8")
    index = (data / "llms.txt").read_text(encoding="utf-8").replace(
        "_(등록된 문서 없음)_", "- [외주 계약 검토 기준](company/review_rules.md): 외주 계약의 지식재산권·승인 기준에 관한 질문")
    (data / "llms.txt").write_text(index, encoding="utf-8")
    load_library.cache_clear()
    scripts = {
        "supervisor": [AIMessage(content="", tool_calls=[_call("h1", "ask_company", reason="사내 기준 확인",
                                                               task="지식재산권 관련 사내 기준")]),
                       AIMessage(content="사내 기준상 수용 불가입니다 [K1].")],
        "company": [AIMessage(content="", tool_calls=[_call("k1", "read_reference_sections", reason="기준 읽기",
                                                            doc_id="review_rules", sections=["제3조"])]),
                    AIMessage(content="- 기존 보유 기술 이전 조항은 수용 불가 [K1]")],
    }
    try:
        events, _ = _run(monkeypatch, scripts, _req(), data_dir=data)
    finally:
        load_library.cache_clear()
    team_text = next(e for e in events if e["type"] == "team")
    assert team_text["company_docs"] == 1
    report = next(e for e in events if e["type"] == "agent_report")
    assert report["agent"] == "company" and report["source_ids"] == ["K1"]
    final = next(e for e in events if e["type"] == "final")
    assert "[K1]" in final["answer_md"]
    assert next(s for s in next(e for e in events if e["type"] == "sources")["items"] if s["id"] == "K1")["kind"] == "company"
