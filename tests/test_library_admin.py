"""문서 등록(llms.txt 갱신) 흐름: 가짜 모델 + 임시 data 폴더."""

import asyncio
import shutil
from dataclasses import replace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app import library_admin as la
from app.agent import graph
from app.agent.documents import load_library
from app.config import DATA_DIR, get_settings

CHECKLIST = "# 계약 검토 체크리스트\n\n## 1. 지식재산권\n기존 보유 기술 귀속 여부 확인.\n\n## 2. 손해배상\n한도 확인.\n"


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data"
    shutil.copytree(DATA_DIR, data, ignore=shutil.ignore_patterns(".history"))
    load_library.cache_clear()
    yield replace(get_settings(), openai_api_key="sk-test", data_dir=data)
    load_library.cache_clear()


def _fake(monkeypatch, draft):
    class Fake:
        def with_structured_output(self, schema, include_raw=False):
            raw = AIMessage(content="", response_metadata={"model_name": "gpt-6-luna"},
                            usage_metadata={"input_tokens": 2000, "output_tokens": 300, "total_tokens": 2300})
            return RunnableLambda(lambda _: {"raw": raw, "parsed": draft, "parsing_error": None})

    monkeypatch.setattr(graph, "llm_factory", lambda s: Fake())


def _existing_entries(settings):
    return [la.Entry(doc_id=i, title=v["title"], description=v["description"])
            for i, v in la._existing(settings.data_dir).items()]


def test_propose_repairs_bad_model_output_then_commit(monkeypatch, settings):
    existing = _existing_entries(settings)
    labor = existing[0]
    draft = la.LlmsTxtDraft(
        new_doc=la.Entry(doc_id="Contract Review-Checklist!", title="계약 검토 체크리스트",
                         description="지식재산권, 손해배상 한도 등 회사 계약 검토 기준에 관한 질문"),
        intro="표준계약서와 회사 내부 기준 문서 목록입니다.",
        # 모델 실수: 근로계약서 제목 변경, 비밀유지계약서 누락, 없는 문서 추가
        standard=[labor.model_copy(update={"title": "근로계약서(수정)", "description": "근로 관련 질문"}),
                  *existing[1:3], la.Entry(doc_id="ghost", title="유령", description="x")],
        company=[la.Entry(doc_id="Contract Review-Checklist!", title="계약 검토 체크리스트", description="")],
        change_summary="회사 체크리스트를 회사 참고 문서에 추가했습니다.",
    )
    _fake(monkeypatch, draft)
    prop = asyncio.run(la.propose(settings, "company", "체크리스트.md", CHECKLIST))

    assert prop["doc"]["id"] == "contract_review_checklist"
    assert prop["doc"]["path"] == "company/contract_review_checklist.md"
    after = prop["llms_after"]
    assert "- [표준근로계약서](contracts/labor_standard.md): 근로 관련 질문" in after  # 제목 유지, 설명은 반영
    assert "contracts/nda.md" in after and "ghost" not in after
    assert "## 회사 참고 문서\n\n- [계약 검토 체크리스트](company/contract_review_checklist.md)" in after
    assert len(prop["warnings"]) >= 3 and prop["structure"] == "제목(##)"
    assert prop["cost_usd"] == pytest.approx((2000 * 0.10 + 300 * 0.50) / 1e6)
    assert "+- [계약 검토 체크리스트]" in prop["diff"]
    assert not (settings.data_dir / prop["doc"]["path"]).exists()  # 아직 저장 전

    result = la.commit(settings, prop["proposal_id"], after)
    assert (settings.data_dir / "company/contract_review_checklist.md").exists()
    assert (settings.data_dir / ".history" / result["backup"]).exists()
    lib = load_library(settings.data_dir)
    doc = lib.docs["contract_review_checklist"]
    assert doc.category == "company" and len(doc.sections) == 2 and len(lib.docs) == 5


def test_commit_rejects_broken_edit_and_stale_proposal(monkeypatch, settings):
    existing = _existing_entries(settings)
    draft = la.LlmsTxtDraft(new_doc=la.Entry(doc_id="nda_guide", title="NDA 가이드", description="NDA 에 관한 질문"),
                            intro="목록", standard=existing, company=[], change_summary="추가")
    _fake(monkeypatch, draft)
    prop = asyncio.run(la.propose(settings, "standard", "nda_guide.md", "제1조(목적) 가\n제2조(정의) 나\n"))
    assert prop["doc"]["path"] == "contracts/nda_guide.md" and prop["structure"] == "조항(제N조)"

    broken = prop["llms_after"].replace("contracts/nda_guide.md", "contracts/typo.md")
    with pytest.raises(la.LibraryError, match="파일이 없는 경로|새 문서"):
        la.commit(settings, prop["proposal_id"], broken)

    (settings.data_dir / "llms.txt").write_text(prop["llms_before"] + "\n", encoding="utf-8")
    with pytest.raises(la.LibraryError, match="바뀌었습니다"):
        la.commit(settings, prop["proposal_id"], prop["llms_after"])


def test_delete_moves_file_and_marks_empty_category(monkeypatch, settings):
    existing = _existing_entries(settings)
    draft = la.LlmsTxtDraft(new_doc=la.Entry(doc_id="kb", title="사내 규정", description="규정에 관한 질문"),
                            intro="목록", standard=existing, company=[], change_summary="추가")
    _fake(monkeypatch, draft)
    prop = asyncio.run(la.propose(settings, "company", "kb.md", "규정 본문"))
    la.commit(settings, prop["proposal_id"], prop["llms_after"])

    la.delete(settings, "kb")
    index = (settings.data_dir / "llms.txt").read_text(encoding="utf-8")
    assert "company/kb.md" not in index and "_(등록된 문서 없음)_" in index
    assert not (settings.data_dir / "company/kb.md").exists()
    assert list((settings.data_dir / ".history" / "deleted").glob("*_kb.md"))
    assert len(load_library(settings.data_dir).docs) == 4
