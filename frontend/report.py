"""실행 이벤트 → 다운로드용 Markdown 보고서.

화면과 같은 이벤트 목록으로 만들기 때문에 실시간 실행, 지난 대화, 녹화 재생 모두 같은 결과가 나온다.
"""

from __future__ import annotations

import re
from datetime import datetime

from labels import CRITERION_ICON, DECISION_LABEL, DISCLAIMER, QUESTION_TYPES, STEP_ICON, STEP_TEXT

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

    trigger = start.get("trigger") or {"type": "question"}
    lines = ["# 계약서 검토 결과", ""]
    if trigger.get("type", "question") != "question":
        lines.append(f"- **트리거**: {trigger.get('label', '')} — {trigger.get('detail', '')}")
    if start.get("question"):
        lines.append(f"- **질문**: {start['question']}")
    if start.get("attachment"):
        lines.append(f"- **첨부 계약서**: {start['attachment']}")
    if when:
        lines.append(f"- **일시**: {when:%Y-%m-%d %H:%M:%S}")
    lines.append(f"- **모델**: {start.get('model', '')}")
    if start.get("mode_label"):
        lines.append(f"- **에이전트 방식**: {start['mode_label']}")
    lines += ["", "## 답변", "", _demote(final.get("answer_md", "(답변 없음)").strip(), 1), ""]

    goal = _first(events, "goal").get("data")
    if goal and final.get("criteria"):
        met = final.get("goal_met")
        lines += ["## 목표 달성 현황", "", f"- **목표**: {goal['goal']}",
                  f"- **결과**: {'✅ 목표 달성' if met else '⚠️ 일부 미달'}", ""]
        lines += [f"{CRITERION_ICON[c['status']]} {c['id']}. {c['text']}" + (f" — {c['note']}" if c.get("note") else "")
                  + "  " for c in final["criteria"]]
        lines.append("")

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


def _verify_line(e: dict) -> str:
    """근거 검증 (에이전트 안) 한 줄 요약."""
    where = "middleware" if e.get("where") == "react" else "근거 검증 단계"
    text = (f"- ✅ 근거 검증 ({where}, {e.get('attempt', 1)}차): 확인된 인용 {', '.join(e.get('cited', [])) or '없음'}"
            + "".join(f" / ⚠️ {p}" for p in e.get("problems", [])))
    if e.get("retry"):
        text += " → ↩️ 피드백과 함께 돌려보내 다시 쓰게 함"
    elif e.get("problems"):
        text += " → 재작성 한도 도달, 그대로 넘김"
    else:
        text += " → 통과"
    return text


def _guardrail_line(e: dict) -> str:
    removed = e.get("removed") or []
    return ("- 🛡️ 최종 안전장치 (에이전트 밖, 코드): "
            + (f"남은 가짜 인용 {', '.join(removed)} 제거" if removed else "남은 문제 없음"))


def _react_process_section(events: list[dict], sources: list[dict]) -> list[str]:
    """실전형 ReAct: 모델 호출마다 무엇을 결정했는지 시간 순으로."""
    start = _first(events, "run_start")
    lines = ["## 에이전트 작업 과정 (실전형 ReAct: 모델 ⇄ 도구)", ""]
    if start.get("middleware"):
        lines += ["사용한 middleware:"] + [f"- `{m['name']}` — {m['desc']}" for m in start["middleware"]] + [""]
    results = {e["id"]: e for e in events if e.get("type") == "tool_result"}
    for e in events:
        t = e.get("type")
        if t == "react_model_start":
            lines += ["", f"#### 🧠 모델 호출 #{e['n']}", ""]
        elif t == "react_model_end":
            if e.get("text") and e["tool_calls"]:
                lines.append(f"- 💬 모델이 쓴 글: {_one_line(e['text'], 300)}")
            lines.append(f"- ➡️ 결정: 도구 {len(e['tool_calls'])}개 호출 ({', '.join(e['tool_calls'])})" if e["tool_calls"]
                         else "- 🏁 결정: 도구를 더 부르지 않고 최종 답변 (모델이 스스로 멈춤)")
        elif t == "todos":
            done = sum(x.get("status") == "completed" for x in e["todos"])
            lines.append(f"- 📝 할 일 목록 갱신 ({done}/{len(e['todos'])} 완료): "
                         + " / ".join(f"[{x.get('status')}] {x.get('content')}" for x in e["todos"]))
        elif t == "tool_call":
            r = results.get(e["id"], {})
            args = ", ".join(f"{k}={', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in e["args"].items())
            ids = f" [{', '.join(r.get('source_ids', []))}]" if r.get("source_ids") else ""
            ok = "" if r.get("ok", True) else "⚠️ "
            lines.append(f"- 🔧 {e.get('label', e['tool'])} ({args}) — 이유: {_one_line(e.get('reason', ''), 150)}"
                         f" → {ok}{_one_line(r.get('summary', ''), 150)}{ids}")
        elif t == "verify":
            lines.append(_verify_line(e))
        elif t == "guardrail":
            lines += ["", _guardrail_line(e)]
    lines.append("")
    unused = [s for s in sources if not s.get("cited")]
    if unused:
        lines += ["### 참고: 조사했지만 인용하지 않은 자료", ""]
        lines += [f"- [{s['id']}] {KIND_LABEL.get(s['kind'], s['kind'])} · {s['title']}" for s in unused] + [""]
    return lines


def _multi_process_section(events: list[dict], sources: list[dict]) -> list[str]:
    """멀티 에이전트: 팀장의 결정, 지시(handoff), 전문가의 도구 사용과 보고서를 시간 순으로."""
    start = _first(events, "run_start")
    team = {a["key"]: f"{a['icon']} {a['name']}" for a in _first(events, "team").get("agents", [])}
    lines = ["## 에이전트 작업 과정 (멀티 에이전트: 팀장 + 전문가)", ""]
    if team:
        lines += ["팀 구성: " + " · ".join(team.values()), ""]
    if start.get("middleware"):
        lines += ["사용한 middleware:"] + [f"- `{m['name']}` — {m['desc']}" for m in start["middleware"]] + [""]
    results = {e["id"]: e for e in events if e.get("type") == "tool_result"}
    for e in events:
        t, who = e.get("type"), e.get("agent")
        specialist = who not in (None, "supervisor")
        if t == "react_model_start" and not specialist:
            lines += ["", f"#### 🧑‍💼 팀장 모델 호출 #{e['n']}", ""]
        elif t == "react_model_end" and not specialist:
            asks = [n.removeprefix("ask_") for n in e["tool_calls"] if n.startswith("ask_")]
            if asks:
                lines.append(f"- ➡️ 결정: 전문가 {len(asks)}명에게 맡김" + (" (동시에)" if len(asks) > 1 else "")
                             + ": " + ", ".join(team.get(a, a) for a in asks))
            elif not e["tool_calls"]:
                lines.append("- 🏁 결정: 보고서를 모아 최종 답변 작성")
        elif t == "handoff":
            lines.append(f"- 📨 팀장 → {team.get(e['to'], e['to'])}: {_one_line(e['task'], 200)} "
                         f"(이유: {_one_line(e.get('reason', ''), 120)})")
        elif t == "tool_call" and specialist:
            r = results.get(e["id"], {})
            args = ", ".join(f"{k}={', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in e["args"].items())
            ids = f" [{', '.join(r.get('source_ids', []))}]" if r.get("source_ids") else ""
            lines.append(f"  - {team.get(who, who)} 🔧 {e.get('label', e['tool'])} ({args}) → "
                         f"{_one_line(r.get('summary', ''), 120)}{ids}")
        elif t == "agent_report":
            lines.append(f"  - 📋 {team.get(who, who)} 보고서 ({e['model_calls']}회 생각 · 도구 {e['tool_calls']}회 · "
                         f"{e['elapsed_ms'] / 1000:.1f}초): {_one_line(e['report'], 400)}")
        elif t == "todos" and not specialist:
            done = sum(x.get("status") == "completed" for x in e["todos"])
            lines.append(f"- 📝 업무 분배 목록 ({done}/{len(e['todos'])} 완료): "
                         + " / ".join(f"[{x.get('status')}] {x.get('content')}" for x in e["todos"]))
        elif t == "verify":
            lines += ["", _verify_line(e)]
        elif t == "guardrail":
            lines += ["", _guardrail_line(e)]
    lines.append("")
    by_agent = (_first(events, "final").get("stats") or {}).get("by_agent")
    if by_agent:
        lines += ["### 에이전트별 사용량", "", "| 에이전트 | LLM 호출 | 도구 호출 | 토큰 | 비용 |", "|---|---|---|---|---|"]
        lines += [f"| {team.get(k, k)} | {v['llm_calls']} | {v['tool_calls']} | {v['usage']['total']:,} | "
                  f"${v['cost_usd']:.5f} |" for k, v in by_agent.items()] + [""]
    unused = [s for s in sources if not s.get("cited")]
    if unused:
        lines += ["### 참고: 조사했지만 인용하지 않은 자료", ""]
        lines += [f"- [{s['id']}] {KIND_LABEL.get(s['kind'], s['kind'])} · {s['title']}" for s in unused] + [""]
    return lines


def _process_section(events: list[dict], sources: list[dict]) -> list[str]:
    mode = _first(events, "run_start").get("mode")
    if mode == "multi":
        return _multi_process_section(events, sources)
    if mode == "react":
        return _react_process_section(events, sources)
    if not _first(events, "goal"):
        return _legacy_process_section(events, sources)
    lines = ["## 에이전트 작업 과정", ""]
    goal_ev = _first(events, "goal")
    g = goal_ev["data"]
    lines += ["### 1. 🎯 목표 정의", "", f"- 트리거: {g.get('trigger_summary', '')}", f"- 목표: {g['goal']}",
              f"- 유형: {QUESTION_TYPES.get(g.get('question_type'), g.get('question_type'))} · 요지: {g.get('summary', '')}"]
    if g.get("attachment_reason"):
        lines.append(f"- 첨부 계약서: {'읽음' if g.get('use_attachment') else '읽지 않음'} — {g['attachment_reason']}")
    lines.append(f"- 판단 근거: {g.get('thought', '')}")
    route = _first(events, "route")
    if route:
        lines.append(f"- 분기: {route.get('label', '')}")
    if goal_ev.get("criteria"):
        lines += ["", "완료 조건:"] + [f"{c['id']}. {c['text']}" for c in goal_ev["criteria"]]
    lines.append("")

    plan = _first(events, "plan")
    if plan:
        lines += ["### 2. 🧭 계획 수립", "", f"전략: {plan.get('strategy', '')}", "", "처음 세운 계획:"]
        lines += [f"{s['no']}. {s['goal']} (`{s['tool']}`, 조건 {', '.join(map(str, s.get('criteria', [])))})"
                  for s in plan.get("steps", [])]
        lines.append("")

    lines += ["### 3. 🔁 계획 실행과 점검", ""]
    results = {e["id"]: e for e in events if e.get("type") == "tool_result"}
    loop_n, evals, writes = 0, 0, 0
    for e in events:
        t = e.get("type")
        if t == "loop":
            loop_n = e["n"]
        elif t == "step_start":
            lines += ["", f"#### ▶ 단계 {e['no']}. {e['goal']} (에이전트 루프 {loop_n}회차)", ""]
        elif t == "tool_call":
            r = results.get(e["id"], {})
            args = ", ".join(f"{k}={', '.join(map(str, v)) if isinstance(v, list) else v}" for k, v in e["args"].items())
            ids = f" [{', '.join(r.get('source_ids', []))}]" if r.get("source_ids") else ""
            ok = "" if r.get("ok", True) else "⚠️ "
            lines.append(f"- 🔧 {e.get('label', e['tool'])} ({args}) — 이유: {_one_line(e.get('reason', ''), 150)}"
                         f" → {ok}{_one_line(r.get('summary', ''), 150)}{ids}")
        elif t == "check":
            met = sum(c["status"] == "met" for c in e["criteria"])
            lines += [f"- 🔎 점검 — 📝 작업 메모: {_one_line(e['observation'], 400)}",
                      f"  - 판단: {DECISION_LABEL.get(e['decision'], e['decision'])} — {e['reason']} "
                      f"(완료 조건 {met}/{len(e['criteria'])} 충족)"]
        elif t == "plan_revised":
            by = "점검" if e["by"] == "check" else "목표 달성 판단"
            lines.append(f"- 🔄 **계획 수정** ({by}): {e['reason']}")
            dropped = {s["no"]: s["goal"] for s in e["steps"]}
            lines += [f"  - 🚫 취소: {n}. {dropped.get(n, '')}" for n in e["dropped"]]
            lines += [f"  - 🆕 추가: {s['no']}. {s['goal']}" for s in e["added"]]
        elif t == "stage" and e["stage"] == "write" and e["status"] == "start":
            writes += 1
            lines += ["", f"#### ✍️ 답변 작성{f' ({writes}차)' if writes > 1 else ''}", ""]
        elif t == "verify":
            lines.append(_verify_line(e))
        elif t == "guardrail":
            lines += ["", "#### 🛡️ 최종 안전장치 (에이전트 밖)", "", _guardrail_line(e)]
        elif t == "evaluate":
            evals += 1
            met = sum(c["status"] == "met" for c in e["criteria"])
            lines += ["", f"#### 🏁 목표 달성 판단{f' ({evals}차)' if evals > 1 else ''}", "",
                      f"- 결과: {'✅ 목표 달성' if e['goal_met'] else '⚠️ 미달'} (완료 조건 {met}/{len(e['criteria'])} 충족)"]
            if not e["goal_met"]:
                lines.append(f"- 부족한 점: {'; '.join(e.get('missing', [])) or '-'}")
                lines.append(f"- 다음 행동: {'추가 조사로 돌아감' if e.get('retry') else '확인하지 못한 부분을 밝히고 종료'}"
                             f" — {e.get('feedback', '')}")
    lines.append("")

    steps = {s["no"]: dict(s) for s in plan.get("steps", [])} if plan else {}
    for e in events:
        if e.get("type") == "plan_revised":
            steps = {s["no"]: dict(s) for s in e["steps"]}
        elif e.get("type") == "plan_update" and e["step_no"] in steps:
            steps[e["step_no"]]["status"] = e["status"]
    if steps:
        lines += ["### 최종 계획 상태", ""]
        lines += [f"{s['no']}. {STEP_ICON[s.get('status', 'pending')]} {s['goal']} — {STEP_TEXT[s.get('status', 'pending')]}"
                  for s in steps.values()]
        lines.append("")

    unused = [s for s in sources if not s.get("cited")]
    if unused:
        lines += ["### 참고: 조사했지만 인용하지 않은 자료", ""]
        for s in unused:
            title = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            lines.append(f"- [{s['id']}] {KIND_LABEL.get(s['kind'], s['kind'])} · {title}")
        lines.append("")
    return lines


def _legacy_process_section(events: list[dict], sources: list[dict]) -> list[str]:
    """예전 녹화(질문 분석 → 조사 계획 → 자료 조사) 형식."""
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
