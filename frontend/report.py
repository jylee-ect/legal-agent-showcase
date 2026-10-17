"""실행 이벤트 → 다운로드용 Markdown 보고서.

화면과 같은 이벤트 목록으로 만들기 때문에 실시간 실행, 지난 대화, 녹화 재생 모두 같은 결과가 나온다.
"""

from __future__ import annotations

import re
from datetime import datetime

from labels import DISCLAIMER, QUESTION_TYPES, STEP_ICON, STEP_TEXT

KIND_LABEL = {"uploaded": "첨부 계약서", "standard": "표준계약서", "company": "회사 문서",
              "law": "법령", "precedent": "판례", "web": "웹 자료"}


def _first(events: list[dict], type_: str) -> dict:
    return next((e for e in events if e.get("type") == type_), {})


def _when(events: list[dict]) -> datetime | None:
    ts = _first(events, "run_start").get("ts")
    try:
        return datetime.fromisoformat(ts) if ts else None
    except ValueError:
        return None


def _one_line(text: str, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


def _demote(markdown: str, levels: int) -> str:
    """모델이 쓴 제목(##)이 보고서 제목 단계와 겹치지 않게 한 단계씩 낮춘다."""
    return re.sub(r"(?m)^(#{1,6})(?=\s)", lambda m: "#" * min(len(m.group(1)) + levels, 6), markdown)


def _cost(usd: float | None, usd_krw: float | None) -> str:
    if usd is None:
        return "단가 미설정"
    text = f"${usd:.5f}" if usd < 0.01 else f"${usd:.4f}"
    return text + (f" (약 {usd * usd_krw:,.1f}원)" if usd_krw else "")


def file_name(events: list[dict], include_process: bool) -> str:
    start = _first(events, "run_start")
    when = _when(events) or datetime.now()
    slug = re.sub(r"[^\w가-힣]+", "_", start.get("question", "결과"))[:24].strip("_") or "결과"
    suffix = "_작업과정포함" if include_process else ""
    return f"계약검토_{when:%Y%m%d_%H%M}_{slug}{suffix}.md"


def build_report(events: list[dict], include_process: bool = False) -> str:
    start = _first(events, "run_start")
    final = _first(events, "final")
    sources = _first(events, "sources").get("items", [])
    pricing = start.get("pricing") or {}
    when = _when(events)

    lines = ["# 계약서 검토 결과", ""]
    lines.append(f"- **질문**: {start.get('question', '')}")
    if start.get("attachment"):
        lines.append(f"- **첨부 계약서**: {start['attachment']}")
    if when:
        lines.append(f"- **일시**: {when:%Y-%m-%d %H:%M:%S}")
    lines.append(f"- **모델**: {start.get('model', '')}")
    lines += ["", "## 답변", "", _demote(final.get("answer_md", "(답변 없음)").strip(), 1), ""]

    cited = [s for s in sources if s.get("cited")]
    if cited:
        lines += ["## 출처", ""]
        for s in cited:
            title = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            lines.append(f"- **[{s['id']}]** {KIND_LABEL.get(s['kind'], s['kind'])} · {title}")
            if s.get("excerpt"):
                lines.append(f"  > {_one_line(s['excerpt'])}")
        lines.append("")

    if include_process:
        lines += _process_section(events, sources)

    stats = final.get("stats") or {}
    if stats:
        usage = stats.get("usage") or {}
        lines += ["## 사용량", "",
                  "| 소요 시간 | 에이전트 루프 | LLM 호출 | 도구 호출 | 총 토큰 | 비용 |",
                  "|---|---|---|---|---|---|",
                  f"| {stats.get('elapsed_ms', 0) / 1000:.1f}초 | {stats.get('agent_loops', '-')} | "
                  f"{stats.get('llm_calls', '-')} | {stats.get('tool_calls', '-')} | {usage.get('total', 0):,} | "
                  f"{_cost(stats.get('cost_usd'), pricing.get('usd_krw'))} |", ""]
        if usage:
            lines.append(f"토큰 상세: 입력 {usage.get('input', 0):,} (캐시 읽기 {usage.get('cache_read', 0):,} / "
                         f"캐시 쓰기 {usage.get('cache_write', 0):,}) · 출력 {usage.get('output', 0):,} "
                         f"(추론 {usage.get('reasoning', 0):,})")
            lines.append("")

    lines += ["---", "", DISCLAIMER, ""]
    return "\n".join(lines)


def _process_section(events: list[dict], sources: list[dict]) -> list[str]:
    lines = ["## 에이전트 작업 과정", ""]

    analysis = _first(events, "analysis").get("data")
    if analysis:
        lines += ["### 1. 질문 분석", ""]
        lines.append(f"- 질문 유형: {QUESTION_TYPES.get(analysis.get('question_type'), analysis.get('question_type'))}")
        lines.append(f"- 요지: {analysis.get('summary', '')}")
        if analysis.get("issues"):
            lines.append(f"- 쟁점: {', '.join(analysis['issues'])}")
        if analysis.get("attachment_reason"):
            verdict = "읽음" if analysis.get("use_attachment") else "읽지 않음"
            lines.append(f"- 첨부 계약서: {verdict} — {analysis['attachment_reason']}")
        lines.append(f"- 판단 근거: {analysis.get('thought', '')}")
        route = _first(events, "route")
        if route:
            lines.append(f"- 분기: {route.get('label', '')}")
        lines.append("")

    plan = _first(events, "plan")
    if plan:
        status = {s["no"]: "pending" for s in plan.get("steps", [])}
        for e in events:
            if e.get("type") == "plan_update":
                status[e["step_no"]] = e["status"]
        lines += ["### 2. 조사 계획", "", f"전략: {plan.get('strategy', '')}", ""]
        for s in plan.get("steps", []):
            st_ = status.get(s["no"], "pending")
            lines.append(f"{s['no']}. {STEP_ICON[st_]} {s['goal']} (`{s['tool']}`) — {STEP_TEXT[st_]}")
        lines.append("")

    calls = [e for e in events if e.get("type") == "tool_call"]
    if calls:
        results = {e["id"]: e for e in events if e.get("type") == "tool_result"}
        loops = sum(1 for e in events if e.get("type") == "loop")
        lines += ["### 3. 자료 조사", "", f"에이전트 루프 {loops}회, 도구 호출 {len(calls)}회", "",
                  "| # | 도구 | 사용 이유 | 결과 | 등록된 출처 |", "|---|---|---|---|---|"]
        for i, c in enumerate(calls, 1):
            r = results.get(c["id"], {})
            args = ", ".join(f"{k}={', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in c["args"].items())
            label = f"{c.get('label', c['tool'])} ({args})" if args else c.get("label", c["tool"])
            ok = "" if r.get("ok", True) else "⚠️ "
            lines.append(f"| {i} | {_cell(label)} | {_cell(c.get('reason', ''))} | {ok}{_cell(r.get('summary', ''))} | "
                         f"{', '.join(r.get('source_ids', []))} |")
        lines.append("")
        note = _first(events, "research_note").get("text")
        if note:
            lines += ["**조사 메모**", "", _demote(note.strip(), 2), ""]

    verify = _first(events, "verify")
    if verify:
        lines += ["### 4. 근거 검증", ""]
        lines.append(f"- 확인된 인용: {', '.join(verify.get('cited', [])) or '없음'}")
        for p in verify.get("problems", []):
            lines.append(f"- ⚠️ {p}")
        if verify.get("unused"):
            lines.append(f"- 조사했지만 인용하지 않은 자료: {', '.join(verify['unused'])}")
        lines.append("")

    unused = [s for s in sources if not s.get("cited")]
    if unused:
        lines += ["### 참고: 조사했지만 인용하지 않은 자료", ""]
        for s in unused:
            title = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            lines.append(f"- [{s['id']}] {KIND_LABEL.get(s['kind'], s['kind'])} · {title}")
        lines.append("")
    return lines


def _cell(text: str) -> str:
    """표 칸 안에서 줄바꿈과 | 를 정리"""
    return _one_line(text, 200).replace("|", "\\|")
