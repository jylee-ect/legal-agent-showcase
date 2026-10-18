from pathlib import Path

from app.agent.documents import decode_upload, load_library, normalize_section_id, parse_document
from app.agent.sources import SourceRegistry, check_citations, normalize_citations, strip_unknown
from app.config import DATA_DIR, get_settings, load_key_file

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def test_library_loads_four_contracts_from_llms_txt():
    lib = load_library(DATA_DIR)
    assert set(lib.docs) == {"labor_standard", "housing_lease", "software_service", "nda"}
    labor = lib.docs["labor_standard"]
    assert labor.title == "표준근로계약서"
    assert labor.get("7").title == "수습기간"
    assert "최저임금의 100분의 90" in labor.get("제7조").text


def test_normalize_section_id():
    assert normalize_section_id("5") == "제5조"
    assert normalize_section_id("제 3 조의 2") == "제3조의2"
    assert normalize_section_id("섹션2") == "섹션2"
    assert normalize_section_id("제13조(위약 예정의 금지)") == "제13조"
    assert normalize_section_id("제3조의2（임차권등기명령）") == "제3조의2"


def test_inline_article_reference_is_not_a_new_section():
    doc = parse_document("x", "제1조(목적) 목적.\n제2조(해지) 제1조에 따라 해지한다.\n제5조에 따라 처리한다.")
    assert [s.id for s in doc.sections] == ["제1조", "제2조"]


def test_sample_uploads_parse_by_article():
    md = parse_document("a", decode_upload((SAMPLES / "프리랜서_앱개발_용역계약서.md").read_bytes()))
    assert len(md.sections) == 12 and md.get("제8조").title == "지식재산권"
    txt = parse_document("b", decode_upload((SAMPLES / "카페_근로계약서.txt").read_bytes()))
    assert txt.get("제6조").title == "퇴사" and "200만원" in txt.get("제6조").text


def test_cp949_upload_and_plain_text_fallback():
    assert decode_upload("제1조(목적) 테스트".encode("cp949")).startswith("제1조")
    doc = parse_document("c", "조항 없는 메모입니다.\n\n두 번째 문단.")
    assert doc.sections[0].id == "부분1"


def test_registry_dedupes_and_citations_are_checked():
    reg = SourceRegistry()
    a = reg.add("standard", "labor#제7조", "표준근로계약서 제7조", "본문")
    b = reg.add("standard", "labor#제7조", "표준근로계약서 제7조", "본문")
    law = reg.add("law", "근로기준법#제20조", "근로기준법 제20조", "본문", url="https://www.law.go.kr")
    assert a.id == b.id == "C1" and law.id == "L1"

    text = normalize_citations("수습 감액은 제한됩니다 [C1, L1]. 근거 없음 [P9].")
    assert "[C1][L1]" in text
    check = check_citations(text, reg)
    assert check == {"cited": ["C1", "L1"], "unknown": ["P9"], "unused": []}
    assert "[P9]" not in strip_unknown(text, check["unknown"])


def test_gpt_key_comes_only_from_gpt_api_env(monkeypatch):
    monkeypatch.setenv("GPT_API", " sk-from-env ")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ignored")
    assert get_settings().openai_api_key == "sk-from-env"
    monkeypatch.delenv("GPT_API")
    assert get_settings().openai_api_key is None and not get_settings().has_openai


def test_key_file_parsing(tmp_path):
    f = tmp_path / "keys.txt"
    f.write_text("gpt:sk-abc:def\nlaw: myoc \n\nbroken line\n", encoding="utf-8")
    assert load_key_file(f) == {"gpt": "sk-abc:def", "law": "myoc"}
