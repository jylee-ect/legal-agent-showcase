"""출처 레지스트리.

도구가 자료를 가져올 때마다 여기에 등록하고 [C1], [L2] 같은 ID 를 붙인다.
답변은 이 ID 로만 인용하고, 최종 '출처' 목록은 코드가 레지스트리에서 만든다.
그래서 모델이 존재하지 않는 URL 이나 조항을 지어낼 수 없다.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

PREFIX = {"uploaded": "U", "standard": "C", "company": "K", "law": "L", "precedent": "P", "web": "W"}
KIND_LABEL = {
    "uploaded": "첨부 계약서",
    "standard": "표준계약서",
    "company": "회사 문서",
    "law": "법령",
    "precedent": "판례",
    "web": "웹 자료",
}

ID_PATTERN = r"[UCKLPW]\d+"
# [C1], [C1, L2], [C1][L2] 모두 인식
CITE_GROUP_RE = re.compile(rf"\[\s*({ID_PATTERN}(?:\s*[,，]\s*{ID_PATTERN})*)\s*\]")


@dataclass
class Source:
    id: str
    kind: str
    title: str
    excerpt: str
    url: str | None = None
    locator: str | None = None
    content: str = ""  # 답변 작성 단계에 넘길 원문 (화면에는 excerpt 만)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("content")
        return d | {"kind_label": KIND_LABEL[self.kind]}


class SourceRegistry:
    def __init__(self) -> None:
        self._items: dict[str, Source] = {}
        self._keys: dict[tuple[str, str], str] = {}
        self._counters = {k: 0 for k in PREFIX}

    def add(self, kind: str, key: str, title: str, excerpt: str, url: str | None = None,
            locator: str | None = None, content: str = "") -> Source:
        """같은 자료(kind, key)를 다시 읽으면 기존 ID 를 그대로 쓴다."""
        existing = self._keys.get((kind, key))
        if existing:
            return self._items[existing]
        self._counters[kind] += 1
        src = Source(f"{PREFIX[kind]}{self._counters[kind]}", kind, title, excerpt.strip()[:600], url, locator,
                     (content or excerpt).strip()[:3000])
        self._items[src.id] = src
        self._keys[(kind, key)] = src.id
        return src

    def get(self, source_id: str) -> Source | None:
        return self._items.get(source_id)

    def all(self) -> list[Source]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)


def normalize_citations(text: str) -> str:
    """[C1, L2] → [C1][L2]"""
    return CITE_GROUP_RE.sub(lambda m: "".join(f"[{i.strip()}]" for i in re.split(r"[,，]", m.group(1))), text)


def cited_ids(text: str) -> list[str]:
    seen: list[str] = []
    for m in re.finditer(rf"\[({ID_PATTERN})\]", text):
        if m.group(1) not in seen:
            seen.append(m.group(1))
    return seen


def check_citations(text: str, registry: SourceRegistry) -> dict:
    """답변 속 인용을 레지스트리와 대조한다."""
    cited = cited_ids(text)
    known = [c for c in cited if registry.get(c)]
    unknown = [c for c in cited if not registry.get(c)]
    unused = [s.id for s in registry.all() if s.id not in cited]
    return {"cited": known, "unknown": unknown, "unused": unused}


def strip_unknown(text: str, unknown: list[str]) -> str:
    for c in unknown:
        text = text.replace(f"[{c}]", "")
    return text


def render_sources_md(sources: list[Source]) -> str:
    lines = []
    for s in sources:
        label = KIND_LABEL[s.kind]
        if s.url:
            lines.append(f"- **[{s.id}]** {label} · [{s.title}]({s.url})")
        else:
            lines.append(f"- **[{s.id}]** {label} · {s.title}")
    return "\n".join(lines)
