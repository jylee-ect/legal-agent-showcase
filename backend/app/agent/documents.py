"""계약서 문서 파싱 (DB 없이 llms.txt + Markdown 파일).

- 표준계약서: data/llms.txt 가 목록(index), data/contracts/*.md 가 본문.
- 첨부 계약서: 사용자가 올린 .md/.txt 를 같은 방식으로 '제N조' 단위로 나눈다.
  조항 구조가 없으면 Markdown 제목, 그것도 없으면 일정 길이로 나눈다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

# 줄 맨 앞의 '제5조', '## 제3조의2(제목)', '제1조(목적) 본문...' 형태를 조항 시작으로 본다.
# '제5조에 따라' 처럼 본문 중 참조는 조 다음에 공백/괄호/줄끝이 오지 않으므로 제외된다.
ARTICLE_RE = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]*)?(제[ \t]*\d+[ \t]*조(?:[ \t]*의[ \t]*\d+)?)(?=[ \t(（]|$)[ \t]*(?:[(（]([^)）\n]*)[)）])?",
    re.M,
)
HEADING_RE = re.compile(r"^(#{1,3})[ \t]+(.+)$", re.M)
LLMS_ENTRY_RE = re.compile(r"^-\s*\[([^\]]+)\]\(([^)]+)\)\s*:\s*(.+)$", re.M)


@dataclass
class Section:
    id: str  # '제5조', '제3조의2', '섹션2', '부분3'
    title: str
    text: str


@dataclass
class Document:
    doc_id: str
    title: str
    sections: list[Section]
    preamble: str = ""
    description: str = ""
    category: str = "standard"  # standard(표준계약서) | company(회사 참고 문서)
    path: str = ""  # data/ 기준 상대경로
    _by_id: dict[str, Section] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {s.id: s for s in self.sections}

    def get(self, section_id: str) -> Section | None:
        return self._by_id.get(normalize_section_id(section_id))

    def outline(self) -> list[dict]:
        return [{"id": s.id, "title": s.title, "chars": len(s.text)} for s in self.sections]

    def outline_text(self) -> str:
        return "\n".join(f"- {s.id}" + (f"({s.title})" if s.title else "") for s in self.sections)


def normalize_section_id(raw: str) -> str:
    """'5', '5조', '제 5 조', '제3조의 2' → '제5조', '제3조의2'. 그 외('섹션2' 등)는 공백만 제거."""
    s = re.sub(r"\s+", "", str(raw))
    m = re.fullmatch(r"(?:제)?(\d+)(?:조)?(?:의(\d+))?", s)
    if m:
        return f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
    return s


def _first_heading_title(text: str) -> str:
    m = HEADING_RE.search(text)
    return m.group(2).strip() if m else ""


def parse_document(doc_id: str, text: str, title: str = "") -> Document:
    text = text.replace("\r\n", "\n").strip()
    title = title or _first_heading_title(text) or doc_id

    matches = list(ARTICLE_RE.finditer(text))
    if len(matches) >= 2:
        sections = []
        for i, m in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.start():end].strip()
            sections.append(Section(normalize_section_id(m.group(1)), (m.group(2) or "").strip(), body))
        return Document(doc_id, title, _dedupe(sections), preamble=text[: matches[0].start()].strip())

    headings = [h for h in HEADING_RE.finditer(text) if len(h.group(1)) >= 2]
    if len(headings) >= 2:
        sections = []
        for i, h in enumerate(headings):
            end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
            sections.append(Section(f"섹션{i + 1}", h.group(2).strip(), text[h.start():end].strip()))
        return Document(doc_id, title, sections, preamble=text[: headings[0].start()].strip())

    return Document(doc_id, title, _chunk(text))


def _dedupe(sections: list[Section]) -> list[Section]:
    """같은 조 번호가 두 번 나오면(오타 등) 두 번째부터 '-2' 를 붙인다."""
    seen: dict[str, int] = {}
    for s in sections:
        if s.id in seen:
            seen[s.id] += 1
            s.id = f"{s.id}-{seen[s.id]}"
        else:
            seen[s.id] = 1
    return sections


def _chunk(text: str, size: int = 1200) -> list[Section]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > size:
            chunks.append(buf)
            buf = ""
        buf = f"{buf}\n\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return [Section(f"부분{i + 1}", c.splitlines()[0][:30], c) for i, c in enumerate(chunks)]


def decode_upload(raw: bytes) -> str:
    """한국어 .txt 는 UTF-8 이 아닐 수 있다(CP949)."""
    for enc in ("utf-8-sig", "cp949"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# llms.txt 의 링크 경로 첫 폴더가 문서 분류를 정한다.
CATEGORY_DIRS = {"standard": "contracts", "company": "company"}
CATEGORY_LABEL = {"standard": "표준계약서", "company": "회사 참고 문서"}


def category_of(rel_path: str) -> str:
    return "company" if rel_path.replace("\\", "/").startswith(CATEGORY_DIRS["company"] + "/") else "standard"


@dataclass
class Library:
    index_text: str  # llms.txt 원문
    docs: dict[str, Document]
    missing: list[str] = field(default_factory=list)  # llms.txt 에는 있는데 파일이 없는 경로

    def catalog(self) -> list[dict]:
        return [
            {"id": d.doc_id, "title": d.title, "description": d.description, "category": d.category,
             "category_label": CATEGORY_LABEL[d.category], "sections": len(d.sections), "path": d.path}
            for d in self.docs.values()
        ]


def parse_llms_entries(index_text: str) -> list[tuple[str, str, str]]:
    """[(제목, 상대경로, 설명), ...]"""
    return [(t.strip(), p.strip(), d.strip()) for t, p, d in LLMS_ENTRY_RE.findall(index_text)]


@lru_cache(maxsize=4)
def load_library(data_dir: Path) -> Library:
    index_text = (data_dir / "llms.txt").read_text(encoding="utf-8")
    docs: dict[str, Document] = {}
    missing: list[str] = []
    for title, rel_path, desc in parse_llms_entries(index_text):
        path = data_dir / rel_path
        if not path.exists():
            missing.append(rel_path)
            continue
        doc = parse_document(path.stem, path.read_text(encoding="utf-8"), title=title)
        doc.description = desc
        doc.category = category_of(rel_path)
        doc.path = rel_path
        docs[doc.doc_id] = doc
    return Library(index_text=index_text, docs=docs, missing=missing)
