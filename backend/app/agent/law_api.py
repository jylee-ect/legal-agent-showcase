"""국가법령정보센터 Open API (https://open.law.go.kr) 클라이언트.

- 법령 검색:   lawSearch.do?target=law
- 조문 조회:   lawService.do?target=law&MST=...&JO=002300  (제23조 → '0023' + 가지번호 '00')
- 판례 검색:   lawSearch.do?target=prec&search=2 (본문 검색)
- 판례 조회:   lawService.do?target=prec&ID=...

응답의 상세 링크에는 OC(인증값)가 들어 있으므로 화면에 보여줄 URL 은 공개 주소로 새로 만든다.
"""

from __future__ import annotations

import html
import re
from urllib.parse import quote

import httpx

BASE = "https://www.law.go.kr/DRF/"
TIMEOUT = httpx.Timeout(20.0)


class LawApiError(RuntimeError):
    pass


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


async def _get(oc: str, endpoint: str, **params) -> dict:
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(BASE + endpoint, params={"OC": oc, "type": "JSON", **params})
    r.raise_for_status()
    try:
        return r.json()
    except ValueError as e:
        raise LawApiError("법령 API 가 JSON 이 아닌 응답을 반환했습니다 (인증값 또는 등록 IP 확인 필요).") from e


def article_jo(article: str) -> str:
    """'제3조의2' → '000302', '23' → '002300'"""
    m = re.search(r"(\d+)\s*조?(?:\s*의\s*(\d+))?", article)
    if not m:
        raise LawApiError(f"조문 번호를 해석할 수 없습니다: {article}")
    return f"{int(m.group(1)):04d}{int(m.group(2) or 0):02d}"


def law_public_url(law_name: str, article: str | None = None) -> str:
    path = f"https://www.law.go.kr/법령/{quote(law_name)}"
    return f"{path}/{quote(article)}" if article else path


def precedent_public_url(prec_id: str) -> str:
    return f"https://www.law.go.kr/LSW/precInfoP.do?precSeq={prec_id}"


def _fmt_date(d: str) -> str:
    return f"{d[:4]}.{d[4:6]}.{d[6:]}" if re.fullmatch(r"\d{8}", d or "") else d


def _clean_html(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text or "", flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"[ \t　]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def _flatten_article(unit: dict) -> str:
    """조문단위 → 조문내용 + 항 + 호 + 목 텍스트."""
    lines: list[str] = []

    def add(value) -> None:
        for v in _as_list(value):
            if isinstance(v, list):
                add(v)
            elif v and str(v).strip():
                lines.append(str(v).strip())

    add(unit.get("조문내용"))
    for hang in _as_list(unit.get("항")):
        add(hang.get("항내용"))
        for ho in _as_list(hang.get("호")):
            add(ho.get("호내용"))
            for mok in _as_list(ho.get("목")):
                add(mok.get("목내용"))
    return "\n".join(lines)


async def search_laws(oc: str, query: str, display: int = 5) -> list[dict]:
    data = await _get(oc, "lawSearch.do", target="law", query=query, display=display)
    items = _as_list(data.get("LawSearch", {}).get("law"))
    return [
        {
            "name": it.get("법령명한글", ""),
            "mst": it.get("법령일련번호", ""),
            "kind": it.get("법령구분명", ""),
            "ministry": it.get("소관부처명", ""),
            "effective": it.get("시행일자", ""),
        }
        for it in items
    ]


async def get_article(oc: str, law_name: str, article: str) -> dict:
    laws = await search_laws(oc, law_name, display=10)
    if not laws:
        raise LawApiError(f"'{law_name}' 법령을 찾지 못했습니다.")
    target = next((l for l in laws if l["name"] == law_name), None) or next(
        (l for l in laws if law_name in l["name"]), laws[0]
    )
    data = await _get(oc, "lawService.do", target="law", MST=target["mst"], JO=article_jo(article))
    units = _as_list(data.get("법령", {}).get("조문", {}).get("조문단위"))
    unit = next((u for u in units if u.get("조문여부") == "조문"), units[0] if units else None)
    if not unit:
        raise LawApiError(f"{target['name']} {article} 조문을 찾지 못했습니다.")
    article_id = f"제{unit.get('조문번호')}조" + (f"의{unit['조문가지번호']}" if unit.get("조문가지번호") else "")
    return {
        "law_name": target["name"],
        "article": article_id,
        "title": unit.get("조문제목", ""),
        "text": _flatten_article(unit),
        "effective": target["effective"],
        "url": law_public_url(target["name"], article_id),
    }


async def search_precedents(oc: str, query: str, display: int = 6) -> list[dict]:
    # 본문 검색(search=2)이 쟁점 키워드에 잘 맞는다. 결과가 없으면 사건명 검색으로 한 번 더.
    items: list = []
    for search in ("2", "1"):
        data = await _get(oc, "lawSearch.do", target="prec", query=query, search=search, display=display)
        items = _as_list(data.get("PrecSearch", {}).get("prec"))
        if items:
            break
    return [
        {
            "id": it.get("판례일련번호", ""),
            "case_no": it.get("사건번호", ""),
            "case_name": it.get("사건명", ""),
            "court": it.get("법원명", ""),
            "date": it.get("선고일자", ""),
            "kind": it.get("사건종류명", ""),
        }
        for it in items
    ]


async def get_precedent(oc: str, prec_id: str, max_body: int = 2500) -> dict:
    data = await _get(oc, "lawService.do", target="prec", ID=prec_id)
    p = data.get("PrecService")
    if not p:
        raise LawApiError(f"판례(일련번호 {prec_id})를 찾지 못했습니다.")
    body = _clean_html(p.get("판례내용", ""))
    return {
        "id": prec_id,
        "case_no": p.get("사건번호", ""),
        "case_name": p.get("사건명", ""),
        "court": p.get("법원명", ""),
        "date": _fmt_date(p.get("선고일자", "")),
        "holding": _clean_html(p.get("판시사항", "")),
        "summary": _clean_html(p.get("판결요지", "")),
        "ref_articles": _clean_html(p.get("참조조문", "")),
        "body_excerpt": body[:max_body] + ("…" if len(body) > max_body else ""),
        "url": precedent_public_url(prec_id),
    }
