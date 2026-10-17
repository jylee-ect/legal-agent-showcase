"""에이전트 이벤트 → Streamlit 화면.

실시간 실행, 대화 기록 다시 그리기, 녹화 재생이 모두 같은 TurnRenderer 를 쓴다.
"""

from __future__ import annotations

import json
import re

import streamlit as st

from labels import (ARG_NAMES, BADGE_COLOR, DISCLAIMER, KIND_ICON, QUESTION_TYPES, STAGES, STEP_ICON,
                    STEP_TEXT)
from report import build_report, file_name

CITE_RE = re.compile(r"\[([UCKLPW])(\d+)\]")


def badges(text: str) -> str:
    """[C1] → 색 배지"""
    return CITE_RE.sub(lambda m: f" :{BADGE_COLOR[m.group(1)]}-badge[{m.group(1)}{m.group(2)}]", text)


def id_badges(ids: list[str]) -> str:
    return " ".join(f":{BADGE_COLOR[i[0]]}-badge[{i}]" for i in ids if i and i[0] in BADGE_COLOR)


# ---------------------------------------------------------------- 흐름도

FLOW_NODES = [
    ("analyze", "🔍 질문 분석"),
    ("direct", "💬 바로 답변"),
    ("plan", "🧭 조사 계획"),
    ("research", "🧠 자료 조사\n(다음 행동 판단)"),
    ("tools", "🔧 도구 실행"),
    ("write", "✍️ 답변 작성"),
    ("verify", "✅ 근거 검증"),
]
FLOW_EDGES = [
    ("START", "analyze", ""), ("analyze", "plan", "조사 필요"), ("analyze", "direct", "조사 불필요"),
    ("plan", "research", ""), ("research", "tools", "도구 호출"), ("tools", "research", "결과 관찰"),
    ("research", "write", "근거 충분"), ("write", "verify", ""), ("verify", "END", ""), ("direct", "END", ""),
]
FLOW_STYLE = {
    "idle": 'fillcolor="#f2f4f7" color="#98a2b3" fontcolor="#475467"',
    "active": 'fillcolor="#fef0c7" color="#f79009" penwidth=3 fontcolor="#101828"',
    "done": 'fillcolor="#d1fadf" color="#12b76a" fontcolor="#101828"',
}


def flow_dot(states: dict[str, str]) -> str:
    lines = [
        "digraph G {", 'bgcolor="transparent"; rankdir=TB; nodesep=0.3; ranksep=0.35;',
        'node [shape=box style="rounded,filled" fontname="Malgun Gothic" fontsize=11 margin="0.15,0.06"];',
        'edge [fontname="Malgun Gothic" fontsize=9 color="#98a2b3" fontcolor="#667085"];',
        'START [shape=circle label="" width=0.15 style=filled fillcolor="#667085" color="#667085"];',
        'END [shape=doublecircle label="" width=0.12 style=filled fillcolor="#667085" color="#667085"];',
    ]
    for node, label in FLOW_NODES:
        lines.append(f'{node} [label="{label}" {FLOW_STYLE[states.get(node, "idle")]}];')
    for a, b, label in FLOW_EDGES:
        lines.append(f'{a} -> {b} [label="{label}"];' if label else f"{a} -> {b};")
    lines.append("{rank=same; plan; direct;}")
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 한 턴 렌더러

def fmt_cost(usd: float | None, usd_krw: float | None = None) -> str:
    if usd is None:
        return "단가 미설정"
    text = f"${usd:.5f}" if usd < 0.01 else f"${usd:.4f}"
    if usd_krw:
        krw = usd * usd_krw
        text += f" (약 {krw:.2f}원)" if krw < 10 else f" (약 {krw:,.0f}원)"
    return text


class TurnRenderer:
    """작업 과정과 각 단계는 기본으로 펼쳐 두고, 사용자가 제목을 클릭해 접을 수 있다.
    auto_collapse=True 면 끝난 단계(와 끝난 작업 과정 전체)를 자동으로 접는다."""

    def __init__(self, detail: str, contract_titles: dict[str, str] | None = None, flow_ph=None,
                 auto_collapse: bool = False, key: str = "turn"):
        self.key = key  # 대화 안에서 이 턴을 구분하는 값 (위젯 key 용)
        self.detail = detail
        self.more = detail in ("자세히", "개발자")
        self.dev = detail == "개발자"
        self.titles = contract_titles or {}
        self.flow_ph = flow_ph
        self.flow: dict[str, str] = {}
        self.auto_collapse = auto_collapse

        self.proc = st.status("🧠 에이전트 작업 과정", expanded=True, state="running")
        with self.proc:
            self.meta_ph = st.empty()
        self.answer_ph = st.empty()
        self.after = st.container()

        self.pricing: dict = {}
        self.llm_calls: list[dict] = []
        self.total_usage: dict = {}
        self.total_cost: float | None = 0.0
        self.stage_box: dict[str, object] = {}
        self.stage_body: dict[str, object] = {}
        self.stage_numbers: dict[str, int] = {}
        self.current_stage: str | None = None
        self.plan_ph = None
        self.plan: dict | None = None
        self.plan_status: dict[int, str] = {}
        self.tool_slots: dict[str, tuple] = {}
        self.running_tools: set[str] = set()
        self.answer = ""
        self.sources: list[dict] = []
        self.events: list[dict] = []
        self.final: dict | None = None
        self.error: str | None = None

    # ------------------------------------------------------------ 공통

    def handle(self, ev: dict) -> None:
        self.events.append(ev)
        handler = getattr(self, f"_on_{ev['type']}", None)
        if handler:
            handler(ev)

    def _body(self, stage: str | None = None):
        stage = stage or self.current_stage
        return self.stage_body.get(stage, self.proc)

    def _draw_flow(self) -> None:
        if self.flow_ph is not None:
            self.flow_ph.graphviz_chart(flow_dot(self.flow), width="stretch")

    def _stage_label(self, stage: str) -> str:
        icon, name, _ = STAGES.get(stage, ("•", stage, ""))
        return f"{self.stage_numbers[stage]}. {icon} {name}"

    def _usage_text(self) -> str:
        total = self.total_usage.get("total", 0)
        return f"🪙 {total:,} 토큰 · {fmt_cost(self.total_cost, self.pricing.get('usd_krw'))}"

    def _update_proc_label(self) -> None:
        if self.final:
            s = self.final["stats"]
            label = (f"🧠 에이전트 작업 과정 · 완료 {s['elapsed_ms'] / 1000:.1f}초 · "
                     f"LLM {s['llm_calls']}회 · 도구 {s['tool_calls']}회 · {self._usage_text()}")
            self.proc.update(label=label, state="complete", expanded=not self.auto_collapse)
        elif self.error:
            self.proc.update(label="🧠 에이전트 작업 과정 · 오류로 중단", state="error", expanded=True)
        else:
            current = STAGES.get(self.current_stage, ("", "준비", ""))[1] if self.current_stage else "준비"
            self.proc.update(label=f"🧠 에이전트 작업 과정 · {current} 중… · {self._usage_text()}", expanded=True)

    # ------------------------------------------------------------ 이벤트별 처리

    def _on_run_start(self, ev: dict) -> None:
        self.pricing = ev.get("pricing") or {}
        parts = [f"모델 `{ev.get('model')}`"]
        parts.append(f"첨부 `{ev['attachment']}`" if ev.get("attachment") else "첨부 없음")
        parts.append("법령 API 연결됨" if ev.get("law_api") else "법령 API 없음(웹 검색만 사용)")
        if ev.get("ts"):
            parts.append(f"시작 {ev['ts'][11:19]}")
        self.meta_ph.caption(" · ".join(parts))

    def _on_stage(self, ev: dict) -> None:
        stage, status = ev["stage"], ev["status"]
        if status == "start":
            self.stage_numbers[stage] = len(self.stage_numbers) + 1
            with self.proc:
                box = st.status(self._stage_label(stage), state="running", expanded=True)
            with box:
                st.caption(STAGES.get(stage, ("", "", ""))[2])
                body = st.container()
            self.stage_box[stage], self.stage_body[stage] = box, body
            self.current_stage = stage
            self.flow[stage] = "active"
            self._update_proc_label()
        else:
            self.flow[stage] = "done"
            self.stage_box[stage].update(state="complete", expanded=not self.auto_collapse)
        self._draw_flow()

    def _on_loop(self, ev: dict) -> None:
        with self._body("research"):
            st.markdown(f":gray[🔁 **에이전트 루프 {ev['n']}회차** — 생각(다음 행동 판단) → 행동(도구) → 관찰]")

    def _on_llm_call(self, ev: dict) -> None:
        self.llm_calls.append(ev)
        self.total_usage = ev["total_usage"]
        self.total_cost = ev["total_cost_usd"]
        if self.more:
            u = ev["usage"]
            web = f" · 웹검색 {ev['web_search_calls']}회" if ev.get("web_search_calls") else ""
            with self._body(ev["stage"]):
                st.caption(
                    f"🪙 LLM 호출 #{ev['call_no']} ({ev['purpose']}) · 입력 {u['input']:,} "
                    f"(캐시 읽기 {u['cache_read']:,} / 캐시 쓰기 {u['cache_write']:,}) · 출력 {u['output']:,} "
                    f"(추론 {u['reasoning']:,}){web} · {fmt_cost(ev['cost_usd'], self.pricing.get('usd_krw'))} · "
                    f"{ev['elapsed_ms'] / 1000:.1f}초")
        self._update_proc_label()

    def _on_context(self, ev: dict) -> None:
        with self._body("analyze"):
            lines = "\n".join(f"- **{i['label']}**: {i['detail']}" for i in ev["items"])
            st.markdown(f"📥 **에이전트가 받은 정보**\n{lines}")

    def _on_analysis(self, ev: dict) -> None:
        a = ev["data"]
        with self._body("analyze"):
            st.markdown(f"**질문 유형** {QUESTION_TYPES.get(a['question_type'], a['question_type'])} · "
                        f"**요지** {a['summary']}")
            if a["issues"]:
                st.markdown("**쟁점** " + " ".join(f":blue-badge[{i}]" for i in a["issues"]))
            if a.get("attachment_reason"):
                verdict = "📎 첨부 계약서를 **읽겠습니다**" if a["use_attachment"] else "📎 첨부 계약서는 **읽지 않겠습니다**"
                st.markdown(f"{verdict} — {a['attachment_reason']}")
            if self.more:
                cands = [self.titles.get(c, c) for c in a.get("candidate_docs", a.get("candidate_contracts", []))]
                need = [name for key, name in (("needs_law", "법령"), ("needs_precedent", "판례"), ("needs_web", "웹 검색"))
                        if a.get(key)]
                st.markdown(f"**참고할 문서** {', '.join(cands) or '없음'} · **필요한 조사** {', '.join(need) or '없음'}")
            st.info(f"💭 **판단 근거**: {a['thought']}")

    def _on_route(self, ev: dict) -> None:
        with self._body("analyze"):
            st.markdown(f"🔀 **분기 결정**: {ev['label']}")

    def _on_reasoning(self, ev: dict) -> None:
        if not self.more:
            return
        with self._body(ev.get("stage")):
            with st.expander("🧠 모델의 추론 요약 (OpenAI reasoning summary)"):
                st.markdown(ev["text"])

    def _on_plan(self, ev: dict) -> None:
        self.plan = ev
        self.plan_status = {s["no"]: "pending" for s in ev["steps"]}
        with self._body("plan"):
            st.markdown(f"🗺️ **전략**: {ev['strategy']}")
            self.plan_ph = st.empty()
        self._draw_plan()

    def _draw_plan(self) -> None:
        if not self.plan or self.plan_ph is None:
            return
        rows = []
        for s in self.plan["steps"]:
            status = self.plan_status.get(s["no"], "pending")
            tool = f" :gray[`{s['tool']}`]" if self.more else ""
            text = f"~~{s['goal']}~~" if status == "skipped" else s["goal"]
            rows.append(f"{STEP_ICON[status]} **{s['no']}.** {text}{tool} :gray[— {STEP_TEXT[status]}]")
        self.plan_ph.markdown("  \n".join(rows))

    def _on_plan_update(self, ev: dict) -> None:
        self.plan_status[ev["step_no"]] = ev["status"]
        self._draw_plan()

    def _on_thought(self, ev: dict) -> None:
        if self.more:
            with self._body("research"):
                st.markdown(f"💭 _{ev['text']}_")

    def _arg_text(self, args: dict) -> str:
        parts = []
        for k, v in args.items():
            if k in ("doc_id", "contract_id"):
                v = self.titles.get(v, v)
            if isinstance(v, list):
                v = ", ".join(map(str, v))
            parts.append(f"{ARG_NAMES.get(k, k)}: **{v}**")
        return " · ".join(parts)

    def _on_tool_call(self, ev: dict) -> None:
        with self._body("research"):
            box = st.container(border=True)
        with box:
            step = f" :gray[· 계획 {ev['plan_step']}단계]" if ev.get("plan_step") else ""
            st.markdown(f"**{ev['label']}**{step}  \n{self._arg_text(ev['args'])}")
            st.markdown(f"💬 _{ev['reason']}_")
            if self.dev:
                st.caption(f"tool = `{ev['tool']}`")
                st.json(ev["args"], expanded=False)
            result_ph = st.empty()
            result_ph.caption("⏳ 실행 중…")
            extra = st.container()
        self.tool_slots[ev["id"]] = (result_ph, extra)
        self.running_tools.add(ev["id"])
        self.flow["research"], self.flow["tools"] = "done", "active"
        self._draw_flow()

    def _on_tool_result(self, ev: dict) -> None:
        slot = self.tool_slots.get(ev["id"])
        if not slot:
            return
        result_ph, extra = slot
        mark = "✅" if ev["ok"] else "⚠️"
        ms = f" :gray[({ev['elapsed_ms'] / 1000:.1f}초)]" if self.dev else ""
        sources = f"  \n출처 등록 {id_badges(ev['source_ids'])}" if ev["source_ids"] else ""
        result_ph.markdown(f"👀 **관찰** {mark} {ev['summary']}{ms}{sources}")
        if self.more and ev.get("preview"):
            with extra:
                with st.expander("원문 보기"):
                    st.text(ev["preview"])
        self.running_tools.discard(ev["id"])
        if not self.running_tools:
            self.flow["tools"], self.flow["research"] = "done", "active"
            self._draw_flow()

    def _on_research_note(self, ev: dict) -> None:
        if self.more and ev.get("text"):
            with self._body("research"):
                with st.expander("📝 조사 메모 (답변 작성 단계로 넘기는 요약)"):
                    st.markdown(badges(ev["text"]))

    def _on_token(self, ev: dict) -> None:
        self.answer += ev["text"]
        self.answer_ph.markdown(badges(self.answer) + " ▌")

    def _on_verify(self, ev: dict) -> None:
        with self._body("verify"):
            if ev["cited"]:
                st.markdown(f"인용된 근거 {len(ev['cited'])}개 모두 실제로 읽은 자료와 일치합니다 {id_badges(ev['cited'])}"
                            if not ev["unknown"] else f"확인된 인용 {id_badges(ev['cited'])}")
            for p in ev["problems"]:
                st.warning(p)
            if ev["unused"]:
                st.caption(f"조사했지만 답변에는 쓰지 않은 자료 {len(ev['unused'])}개: {', '.join(ev['unused'])}")

    def _on_sources(self, ev: dict) -> None:
        self.sources = ev["items"]

    def _on_final(self, ev: dict) -> None:
        self.final = ev
        self.answer = ev["answer_md"]
        self.answer_ph.markdown(badges(self.answer))
        with self.after:
            cited = [s for s in self.sources if s["cited"]]
            unused = [s for s in self.sources if not s["cited"]]
            if cited:
                st.markdown("##### 📎 출처")
                cols = st.columns(2)
                for i, s in enumerate(cited):
                    with cols[i % 2]:
                        self._source_card(s)
            if unused and self.more:
                with st.expander(f"조사했지만 인용하지 않은 자료 {len(unused)}개"):
                    for s in unused:
                        self._source_card(s)
            if self.sources or cited:
                st.caption(DISCLAIMER)
            self._downloads()
            self._stats(ev["stats"])
        self._update_proc_label()

    def _downloads(self) -> None:
        """결과를 Markdown 파일로 내려받기. 클릭해도 화면을 다시 실행하지 않는다(on_click='ignore')."""
        key = self.key
        c1, c2, _ = st.columns([1, 1, 2])
        for col, include, label in ((c1, False, "📥 답변 다운로드 (.md)"),
                                    (c2, True, "📥 작업 과정 포함 (.md)")):
            col.download_button(label, build_report(self.events, include).encode("utf-8"),
                                file_name=file_name(self.events, include), mime="text/markdown",
                                on_click="ignore", key=f"dl_{include}_{key}", width="stretch")

    def _stats(self, stats: dict) -> None:
        usage = stats.get("usage") or {}
        cols = st.columns(6)
        cols[0].metric("소요 시간", f"{stats['elapsed_ms'] / 1000:.1f}초")
        cols[1].metric("에이전트 루프", stats.get("agent_loops", 0),
                       help="조사 단계에서 '생각 → 행동(도구) → 관찰'을 반복한 횟수")
        cols[2].metric("LLM 호출", stats["llm_calls"],
                       help="분석·계획·루프·답변 작성·웹검색 내부 호출을 모두 합친 모델 호출 수")
        cols[3].metric("도구 호출", stats["tool_calls"])
        cols[4].metric("총 토큰", f"{usage.get('total', 0):,}")
        cols[5].metric("비용", fmt_cost(stats.get("cost_usd")).split(" ")[0],
                       help=fmt_cost(stats.get("cost_usd"), self.pricing.get("usd_krw")))

        with st.expander(f"🪙 토큰 사용량 · 비용 상세 — {fmt_cost(stats.get('cost_usd'), self.pricing.get('usd_krw'))}"):
            rows = []
            for c in self.llm_calls:
                u = c["usage"]
                rows.append({
                    "#": str(c["call_no"]), "단계": STAGES.get(c["stage"], ("", c["stage"], ""))[1], "목적": c["purpose"],
                    "모델": c["model"], "입력 전체": u["input"], "└ 캐시 안 됨": u["uncached_input"],
                    "└ 캐시 읽기": u["cache_read"], "└ 캐시 쓰기": u["cache_write"], "출력 전체": u["output"],
                    "└ 추론": u["reasoning"], "웹검색": c.get("web_search_calls", 0),
                    "비용(USD)": c["cost_usd"], "시간(초)": round(c["elapsed_ms"] / 1000, 2),
                })
            if rows:
                rows.append({
                    "#": "합계", "단계": "", "목적": "", "모델": "", "입력 전체": usage.get("input", 0),
                    "└ 캐시 안 됨": usage.get("uncached_input", 0), "└ 캐시 읽기": usage.get("cache_read", 0),
                    "└ 캐시 쓰기": usage.get("cache_write", 0), "출력 전체": usage.get("output", 0),
                    "└ 추론": usage.get("reasoning", 0), "웹검색": stats.get("web_search_calls", 0),
                    "비용(USD)": stats.get("cost_usd"), "시간(초)": round(stats["elapsed_ms"] / 1000, 2),
                })
                st.dataframe(rows, hide_index=True, width="stretch",
                             column_config={"비용(USD)": st.column_config.NumberColumn(format="$%.6f")})
            p = self.pricing
            if p.get("rates"):
                r = p["rates"]
                st.caption(
                    f"단가 기준 `{p['model']}` (USD / 1M 토큰): 입력 ${r['input']} · 캐시 읽기 ${r['cached_input']} · "
                    f"캐시 쓰기 ${r.get('cache_write', r['input'])} · 출력 ${r['output']} (추론 토큰 포함) · "
                    f"웹검색 ${p.get('web_search_per_1k_calls')} / 1,000회. "
                    f"출처: {p.get('source')} ({p.get('checked_at')} 확인) · 환율 {p.get('usd_krw'):,}원/$ (설정값)")
            else:
                st.caption("이 모델의 단가가 pricing.json 에 없어 비용을 계산하지 못했습니다.")

    def _source_card(self, s: dict) -> None:
        with st.container(border=True):
            title = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
            st.markdown(f":{BADGE_COLOR[s['id'][0]]}-badge[{s['id']}] {KIND_ICON.get(s['kind'], '')} "
                        f"**{s['kind_label']}**  \n{title}")
            if s.get("excerpt"):
                excerpt = s["excerpt"].replace("\n", " ")
                st.caption(excerpt[:220] + ("…" if len(excerpt) > 220 else ""))

    def _on_error(self, ev: dict) -> None:
        self.error = ev["message"]
        with self.proc:
            st.error(f"오류: {ev['message']}")
        if self.current_stage in self.stage_box and self.flow.get(self.current_stage) == "active":
            self.stage_box[self.current_stage].update(state="error", expanded=True)
        self._update_proc_label()

    def _on_done(self, ev: dict) -> None:
        if not self.final and self.answer:
            self.answer_ph.markdown(badges(self.answer))
        if self.error and not self.final:
            self.answer_ph.error(f"오류: {self.error}")
        if self.dev:
            with self.after:
                with st.expander(f"🛠️ 원본 이벤트 로그 ({len(self.events)}개, 토큰 이벤트 제외)"):
                    st.code(event_log(self.events), language="text", wrap_lines=True)


def event_log(events: list[dict]) -> str:
    """'14:03:21.482  +1.234s  tool_call  {...}' 형식. ts 는 실제 시각, t 는 실행 시작 후 경과 시간."""
    lines = []
    for e in events:
        if e["type"] == "token":
            continue
        body = {k: v for k, v in e.items() if k not in ("type", "t", "ts")}
        ts = e.get("ts", "")[11:23] or "--:--:--.---"
        lines.append(f"{ts}  +{e.get('t', 0) / 1000:7.3f}s  {e['type']:<12} {json.dumps(body, ensure_ascii=False)}")
    return "\n".join(lines)
