"""에이전트 이벤트 → Streamlit 화면.

실시간 실행, 대화 기록 다시 그리기, 녹화 재생이 모두 같은 TurnRenderer 를 쓴다.
에이전트 루프: 트리거 → 목표 정의 → 계획 → (단계 실행 → 점검·계획 수정)* → 답변 → 근거 검증 → 목표 달성 판단
"""

from __future__ import annotations

import json
import re

import streamlit as st

from labels import (ARG_NAMES, BADGE_COLOR, CRITERION_ICON, DECISION_LABEL, DISCLAIMER, KIND_ICON, QUESTION_TYPES,
                    STAGES, STEP_ICON, STEP_TEXT)
from report import build_report, file_name

CITE_RE = re.compile(r"\[([UCKLPW])(\d+)\]")


def badges(text: str) -> str:
    """[C1] → 색 배지"""
    return CITE_RE.sub(lambda m: f" :{BADGE_COLOR[m.group(1)]}-badge[{m.group(1)}{m.group(2)}]", text)


def id_badges(ids: list[str]) -> str:
    return " ".join(f":{BADGE_COLOR[i[0]]}-badge[{i}]" for i in ids if i and i[0] in BADGE_COLOR)


# ---------------------------------------------------------------- 흐름도

# 교과서형 루프 (LangGraph): 점선 상자 = 에이전트. 근거 검증은 안, 최종 안전장치는 밖.
FLOW_NODES = [
    ("trigger", "🔔 트리거"),
    ("goal", "🎯 목표 정의"),
    ("direct", "💬 바로 답변"),
    ("plan", "🧭 계획 수립"),
    ("step", "▶ 단계 실행\n(행동 결정)"),
    ("tools", "🔧 도구 사용"),
    ("check", "🔎 점검\n(관찰·계획 수정)"),
    ("write", "✍️ 답변 작성"),
    ("verify", "✅ 근거 검증"),
    ("evaluate", "🏁 목표 달성 판단"),
    ("guardrail", "🛡️ 최종 안전장치\n(에이전트 밖·코드)"),
]
FLOW_AGENT = ["goal", "direct", "plan", "step", "tools", "check", "write", "verify", "evaluate"]
FLOW_EDGES = [
    ("trigger", "goal", ""), ("goal", "plan", "작업 필요"), ("goal", "direct", "바로 답변"),
    ("plan", "step", ""), ("step", "tools", "도구 호출"), ("tools", "check", "결과 관찰"),
    ("step", "check", "도구 없음"), ("check", "step", "계속·계획 수정"), ("check", "write", "목표 충족"),
    ("write", "verify", ""), ("verify", "write", "근거 문제:\n고쳐 쓰기"), ("verify", "evaluate", "통과"),
    ("evaluate", "step", "미달: 추가 조사"), ("evaluate", "guardrail", "달성·종료"), ("direct", "guardrail", ""),
    ("guardrail", "END", ""),
]
# 이벤트의 stage 이름 → 흐름도 노드 (analyze·research 는 예전 녹화 호환)
STAGE_NODE = {"goal": "goal", "analyze": "goal", "plan": "plan", "research": "step", "write": "write",
              "verify": "verify", "evaluate": "evaluate", "direct": "direct", "react": "model",
              "guardrail": "guardrail", "multi": "supervisor"}
FLOW_STYLE = {
    "idle": 'fillcolor="#f2f4f7" color="#98a2b3" fontcolor="#475467"',
    "active": 'fillcolor="#fef0c7" color="#f79009" penwidth=3 fontcolor="#101828"',
    "done": 'fillcolor="#d1fadf" color="#12b76a" fontcolor="#101828"',
}

# 실전형 ReAct (create_agent): 모델 ⇄ 도구 + middleware. 근거 검증은 middleware(에이전트 안)에서.
REACT_NODES = [
    ("trigger", "🔔 트리거"),
    ("model", "🧠 모델\n(생각 → 다음 행동 결정)"),
    ("tools", "🔧 도구 사용\n(write_todos 포함)"),
    ("verify", "✅ 근거 검증\n(after_model middleware)"),
    ("guardrail", "🛡️ 최종 안전장치\n(에이전트 밖·코드)"),
]
REACT_AGENT = ["model", "tools", "verify"]
REACT_EDGES = [
    ("trigger", "model", ""), ("model", "tools", "도구 호출"), ("tools", "model", "결과 관찰"),
    ("model", "verify", "최종 답변 시도"), ("verify", "model", "근거 문제:\n고쳐 쓰기"),
    ("verify", "guardrail", "통과 = 종료"), ("guardrail", "END", ""),
]

# 멀티 에이전트: 팀장(supervisor) + 전문가 4. 전문가 각자도 모델 ⇄ 도구 ReAct 다.
SPECIALIST_NODES = ["law", "precedent", "contract", "company"]
MULTI_NODES = [
    ("trigger", "🔔 트리거"),
    ("supervisor", "🧑‍💼 팀장\n(일 나누기 · 보고서 모으기 · 답변)"),
    ("law", "📜 법령조사\n(ReAct)"),
    ("precedent", "⚖️ 판례조사\n(ReAct)"),
    ("contract", "📑 계약서분석\n(ReAct)"),
    ("company", "🏢 회사기준\n(ReAct)"),
    ("verify", "✅ 근거 검증\n(after_model middleware)"),
    ("guardrail", "🛡️ 최종 안전장치\n(에이전트 밖·코드)"),
]
MULTI_AGENT = ["supervisor", *SPECIALIST_NODES, "verify"]
MULTI_EDGES = [
    ("trigger", "supervisor", ""),
    *[("supervisor", s, "지시 ⇄ 보고") for s in SPECIALIST_NODES],
    ("supervisor", "verify", "최종 답변 시도"), ("verify", "supervisor", "근거 문제:\n고쳐 쓰기"),
    ("verify", "guardrail", "통과 = 종료"), ("guardrail", "END", ""),
]

FLOWS = {
    "loop": (FLOW_NODES, FLOW_EDGES, FLOW_AGENT, "🤖 에이전트 (LangGraph)"),
    "react": (REACT_NODES, REACT_EDGES, REACT_AGENT,
              "🤖 에이전트 (create_agent + middleware: 관찰·근거 검증·할 일 목록·호출 한도)"),
    "multi": (MULTI_NODES, MULTI_EDGES, MULTI_AGENT, "👥 에이전트 팀 (팀장 + 전문가 4, 모두 create_agent)"),
}


def flow_dot(states: dict[str, str], mode: str = "loop") -> str:
    nodes, edges, inside, title = FLOWS.get(mode, FLOWS["loop"])
    lines = [
        "digraph G {", 'bgcolor="transparent"; rankdir=TB; nodesep=0.3; ranksep=0.3;',
        'node [shape=box style="rounded,filled" fontname="Malgun Gothic" fontsize=11 margin="0.15,0.06"];',
        'edge [fontname="Malgun Gothic" fontsize=9 color="#98a2b3" fontcolor="#667085"];',
        'END [shape=doublecircle label="" width=0.12 style=filled fillcolor="#667085" color="#667085"];',
        f'subgraph cluster_agent {{ label="{title}"; fontname="Malgun Gothic"; fontsize=9; '
        f'fontcolor="#667085"; style="rounded,dashed"; color="#98a2b3"; {"; ".join(inside)}; }}',
    ]
    for node, label in nodes:
        lines.append(f'{node} [label="{label}" {FLOW_STYLE[states.get(node, "idle")]}];')
    for a, b, label in edges:
        both = " dir=both" if mode == "multi" and b in SPECIALIST_NODES else ""
        lines.append(f'{a} -> {b} [label="{label}"{both}];' if label else f"{a} -> {b};")
    if mode == "loop":
        lines += ["{rank=same; plan; direct;}", "{rank=same; tools; check;}"]
    if mode == "multi":
        lines.append("{rank=same; " + "; ".join(SPECIALIST_NODES) + ";}")
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
            self.pinned = st.container()  # 📌 목표와 계획 (실시간) — 목표가 정해지면 채운다
        self.answer_ph = st.empty()
        self.after = st.container()

        self.pricing: dict = {}
        self.llm_calls: list[dict] = []
        self.total_usage: dict = {}
        self.total_cost: float | None = 0.0
        self.stage_box: dict[str, object] = {}
        self.stage_body: dict[str, object] = {}
        self.stage_numbers: dict[str, int] = {}
        self.stage_counts: dict[str, int] = {}
        self.stage_seq = 0
        self.current_stage: str | None = None
        self.goal_text = ""
        self.criteria: list[dict] = []
        self.criteria_ph = None
        self.steps: list[dict] = []
        self.plan_ph = None
        self.loop_n = 0
        self.check_ph = None
        self.tool_slots: dict[str, tuple] = {}
        self.running_tools: set[str] = set()
        self.answer = ""
        self.sources: list[dict] = []
        self.events: list[dict] = []
        self.final: dict | None = None
        self.error: str | None = None
        self.mode = "loop"  # loop: 교과서형 루프 / react: 실전형 ReAct (run_start 에서 정해짐)
        self.todos_ph = None
        self.turn_box = None
        self.last_draft = ""  # 실전형: 근거 검증 전 모델이 쓴 답변 (돌려보낸 초안 보기용)
        # 멀티 에이전트: 전문가별 작업 레인
        self.lanes: dict[str, object] = {}  # run(팀장의 ask_* 호출 ID) → 레인 컨테이너
        self.agent_lane: dict[str, object] = {}  # 에이전트 → 가장 최근 레인
        self.pending_lanes: list[tuple[str, object]] = []  # 팀장이 방금 지시한 전문가들의 빈 열
        self.team: dict[str, dict] = {}
        self.team_ph = None

    # ------------------------------------------------------------ 공통

    def handle(self, ev: dict) -> None:
        self.events.append(ev)
        handler = getattr(self, f"_on_{ev['type']}", None)
        if handler:
            handler(ev)

    def _body(self, stage: str | None = None):
        stage = stage or self.current_stage
        if stage not in self.stage_body:  # 예전 녹화의 stage 이름
            stage = {"goal": "analyze", "execute": "research"}.get(stage, stage)
        return self.stage_body.get(stage, self.proc)

    def _set_flow(self, **states: str) -> None:
        self.flow.update(states)
        if self.flow_ph is not None:
            self.flow_ph.graphviz_chart(flow_dot(self.flow, self.mode), width="stretch")

    def _stage_label(self, stage: str) -> str:
        icon, name, _ = STAGES.get(stage, ("•", stage, ""))
        n = self.stage_counts.get(stage, 1)
        return f"{self.stage_numbers[stage]}. {icon} {name}" + (f" ({n}차)" if n > 1 else "")

    def _usage_text(self) -> str:
        total = self.total_usage.get("total", 0)
        return f"🪙 {total:,} 토큰 · {fmt_cost(self.total_cost, self.pricing.get('usd_krw'))}"

    def _update_proc_label(self) -> None:
        if self.final:
            s = self.final["stats"]
            met = self.final.get("goal_met")
            goal = "" if met is None else (" · 목표 달성 ✅" if met else " · 목표 일부 미달 ⚠️")
            loops = f"모델 호출 {s.get('agent_loops', 0)}회" if self.mode == "react" else f"루프 {s.get('agent_loops', 0)}회"
            label = (f"🧠 에이전트 작업 과정 · 완료 {s['elapsed_ms'] / 1000:.1f}초{goal} · "
                     f"{loops} · 도구 {s['tool_calls']}회 · {self._usage_text()}")
            self.proc.update(label=label, state="complete", expanded=not self.auto_collapse)
        elif self.error:
            self.proc.update(label="🧠 에이전트 작업 과정 · 오류로 중단", state="error", expanded=True)
        else:
            current = STAGES.get(self.current_stage, ("", "준비", ""))[1] if self.current_stage else "준비"
            self.proc.update(label=f"🧠 에이전트 작업 과정 · {current} 중… · {self._usage_text()}", expanded=True)

    # ------------------------------------------------------------ 📌 목표와 계획 (실시간)

    def _ensure_pinned(self) -> None:
        if self.criteria_ph is not None:
            return
        with self.pinned:
            with st.container(border=True):
                st.markdown("##### 📌 목표와 계획 (실시간)")
                left, right = st.columns(2)
                self.criteria_ph = left.empty()
                self.plan_ph = right.empty()

    def _draw_criteria(self) -> None:
        if self.criteria_ph is None:
            return
        met = sum(c["status"] == "met" for c in self.criteria)
        rows = [f"**🎯 목표** — {self.goal_text}", "",
                f"**완료 조건** :gray[({met}/{len(self.criteria)} 충족)]"]
        for c in self.criteria:
            note = f"  \n:gray[{badges(c['note'])}]" if self.more and c.get("note") else ""
            rows.append(f"{CRITERION_ICON[c['status']]} {c['id']}. {c['text']}{note}")
        self.criteria_ph.markdown("  \n".join(rows))

    def _draw_plan(self) -> None:
        if self.plan_ph is None or not self.steps:
            return
        rows = ["**🧭 계획**"]
        for s in self.steps:
            status = s.get("status", "pending")
            text = f"~~{s['goal']}~~" if status in ("skipped", "dropped") else s["goal"]
            new = "🆕 " if s.get("added_by") in ("check", "evaluate") else ""
            crit = f" :gray[→ 조건 {', '.join(map(str, s['criteria']))}]" if s.get("criteria") else ""
            tool = f" :gray[`{s['tool']}`]" if self.more else ""
            rows.append(f"{STEP_ICON[status]} {new}**{s['no']}.** {text}{crit}{tool} :gray[— {STEP_TEXT[status]}]")
        self.plan_ph.markdown("  \n".join(rows))

    # ------------------------------------------------------------ 이벤트별 처리

    def _on_run_start(self, ev: dict) -> None:
        self.pricing = ev.get("pricing") or {}
        self.mode = ev.get("mode", "loop")
        trigger = ev.get("trigger") or {"label": "💬 사용자 질문"}
        parts = [f"**{ev.get('mode_label', '🎓 교과서형 루프 (LangGraph)')}**", f"트리거 {trigger['label']}",
                 f"모델 `{ev.get('model')}`"]
        parts.append(f"첨부 `{ev['attachment']}`" if ev.get("attachment") else "첨부 없음")
        parts.append("법령 API 연결됨" if ev.get("law_api") else "법령 API 없음(웹 검색만 사용)")
        if ev.get("ts"):
            parts.append(f"시작 {ev['ts'][11:19]}")
        if ev.get("middleware"):
            parts.append("middleware: " + ", ".join(f"`{m['name']}`" for m in ev["middleware"]))
        self.meta_ph.caption(" · ".join(parts))
        self._set_flow(trigger="done")

    def _on_stage(self, ev: dict) -> None:
        stage, status = ev["stage"], ev["status"]
        node = STAGE_NODE.get(stage)
        if status == "start":
            self.stage_seq += 1
            self.stage_numbers[stage] = self.stage_seq
            self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
            with self.proc:
                box = st.status(self._stage_label(stage), state="running", expanded=True)
            with box:
                st.caption(STAGES.get(stage, ("", "", ""))[2])
                body = st.container()
            self.stage_box[stage], self.stage_body[stage] = box, body
            self.current_stage = stage
            if stage == "write" and self.stage_counts[stage] > 1:  # 근거 검증 피드백 또는 추가 조사 뒤 다시 쓴다
                self.answer = ""
                self.answer_ph.caption("✍️ 답변을 다시 쓰는 중…")
            if node:
                self._set_flow(**{node: "active"})
            self._update_proc_label()
        else:
            self.stage_box[stage].update(state="complete", expanded=not self.auto_collapse)
            if node:
                self._set_flow(**{node: "done"})

    def _on_llm_call(self, ev: dict) -> None:
        self.llm_calls.append(ev)
        self.total_usage = ev["total_usage"]
        self.total_cost = ev["total_cost_usd"]
        if self.more:
            u = ev["usage"]
            web = f" · 웹검색 {ev['web_search_calls']}회" if ev.get("web_search_calls") else ""
            if self.mode == "multi":
                target = self._lane(ev) if self._is_specialist(ev) else self._top_box()
            elif self.mode == "react":
                target = self._top_box()
            else:
                target = self._body(ev["stage"])
            with target:
                st.caption(
                    f"🪙 LLM 호출 #{ev['call_no']} ({ev['purpose']}) · 입력 {u['input']:,} "
                    f"(캐시 읽기 {u['cache_read']:,} / 캐시 쓰기 {u['cache_write']:,}) · 출력 {u['output']:,} "
                    f"(추론 {u['reasoning']:,}){web} · {fmt_cost(ev['cost_usd'], self.pricing.get('usd_krw'))} · "
                    f"{ev['elapsed_ms'] / 1000:.1f}초")
        self._update_proc_label()

    def _on_context(self, ev: dict) -> None:
        with self._body("goal"):
            lines = "\n".join(f"- **{i['label']}**: {i['detail']}" for i in ev["items"])
            st.markdown(f"📥 **에이전트가 받은 정보**\n{lines}")

    def _on_goal(self, ev: dict) -> None:
        g = ev["data"]
        self.goal_text = g["goal"]
        self.criteria = ev.get("criteria", [])
        with self._body("goal"):
            st.markdown(f"🔔 **트리거**: {g['trigger_summary']}")
            if g["route"] == "research":
                st.success(f"🎯 **목표**: {g['goal']}")
                st.markdown("✅ **완료 조건** — 이 조건을 모두 채우면 일이 끝난 것으로 봅니다  \n"
                            + "  \n".join(f"{c['id']}. {c['text']}" for c in self.criteria))
            st.markdown(f"**유형** {QUESTION_TYPES.get(g['question_type'], g['question_type'])} · **요지** {g['summary']}")
            if g["issues"]:
                st.markdown("**쟁점** " + " ".join(f":blue-badge[{i}]" for i in g["issues"]))
            if g.get("attachment_reason"):
                verdict = "📎 첨부 계약서를 **읽겠습니다**" if g["use_attachment"] else "📎 첨부 계약서는 **읽지 않겠습니다**"
                st.markdown(f"{verdict} — {g['attachment_reason']}")
            if self.more and g.get("candidate_docs"):
                st.markdown(f"**참고할 문서** {', '.join(self.titles.get(c, c) for c in g['candidate_docs'])}")
            st.info(f"💭 **판단 근거**: {g['thought']}")
        if self.criteria:
            self._ensure_pinned()
            self._draw_criteria()

    def _on_analysis(self, ev: dict) -> None:  # 예전 녹화 호환
        a = ev["data"]
        with self._body("analyze"):
            st.markdown(f"**질문 유형** {QUESTION_TYPES.get(a['question_type'], a['question_type'])} · "
                        f"**요지** {a['summary']}")
            if a.get("issues"):
                st.markdown("**쟁점** " + " ".join(f":blue-badge[{i}]" for i in a["issues"]))
            st.info(f"💭 **판단 근거**: {a['thought']}")

    def _on_route(self, ev: dict) -> None:
        with self._body("goal"):
            st.markdown(f"🔀 **분기 결정**: {ev['label']}")

    def _on_reasoning(self, ev: dict) -> None:
        if not self.more:
            return
        if self._is_specialist(ev):
            target = self._lane(ev)
        elif self.mode in ("react", "multi"):
            target = self._top_box()
        else:
            target = self._body(ev.get("stage"))
        with target:
            with st.expander("🧠 모델의 추론 요약 (OpenAI reasoning summary)"):
                st.markdown(ev["text"])

    def _on_plan(self, ev: dict) -> None:
        self.steps = [s | {"status": s.get("status", "pending")} for s in ev["steps"]]
        with self._body("plan"):
            st.markdown(f"🗺️ **전략**: {ev['strategy']}")
            st.markdown("**처음 세운 계획**  \n" + "  \n".join(
                f"{s['no']}. {s['goal']}" + (f" :gray[→ 조건 {', '.join(map(str, s['criteria']))}]"
                                             if s.get("criteria") else "") for s in self.steps))
            if not self.criteria:  # 예전 녹화: 실시간 계획을 여기에 그린다
                self.plan_ph = st.empty()
        if self.criteria:
            self._ensure_pinned()
        self._draw_plan()

    def _on_plan_update(self, ev: dict) -> None:
        for s in self.steps:
            if s["no"] == ev["step_no"]:
                s["status"] = ev["status"]
        self._draw_plan()

    def _on_plan_revised(self, ev: dict) -> None:
        self.steps = ev["steps"]
        by = "점검 결과" if ev["by"] == "check" else "목표 달성 판단 결과"
        with self._body("execute" if ev["by"] == "check" else "evaluate"):
            with st.container(border=True):
                st.markdown(f"🔄 **계획 수정** :gray[({by})] — {ev['reason']}")
                dropped = [s for s in self.steps if s["no"] in ev["dropped"]]
                if dropped:
                    st.markdown("  \n".join(f"🚫 취소: ~~{s['no']}. {s['goal']}~~" for s in dropped))
                if ev["added"]:
                    st.markdown("  \n".join(f"🆕 추가: **{s['no']}. {s['goal']}**" for s in ev["added"]))
        self._draw_plan()

    def _on_loop(self, ev: dict) -> None:
        self.loop_n = ev["n"]
        if self.current_stage == "research":  # 예전 녹화
            with self._body("research"):
                st.markdown(f":gray[🔁 **에이전트 루프 {ev['n']}회차** — 생각 → 행동(도구) → 관찰]")

    def _on_step_start(self, ev: dict) -> None:
        crit = f" · 채울 완료 조건: {', '.join(map(str, ev['criteria']))}" if ev.get("criteria") else ""
        with self._body("execute"):
            st.markdown(f"##### ▶ 단계 {ev['no']}. {ev['goal']}")
            st.caption(f"🔁 에이전트 루프 {self.loop_n}회차 — 행동(도구) → 관찰 → 점검{crit}")
        self._set_flow(step="active", check="done" if self.flow.get("check") else "idle")

    def _on_thought(self, ev: dict) -> None:
        if self.more:
            with self._body():
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
        if self._is_specialist(ev):
            target = self._lane(ev)
        elif self.mode in ("react", "multi") and self.turn_box is not None:
            target = self.turn_box
        else:
            target = self._body("execute")
        with target:
            box = st.container(border=True)
        with box:
            step = ev.get("step") or ev.get("plan_step")
            step = f" :gray[· 단계 {step}]" if step else ""
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
        if self.mode != "multi":
            self._set_flow(**({"model": "done"} if self.mode == "react" else {"step": "done"}), tools="active")

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
        if not self.running_tools and self.mode != "multi":
            self._set_flow(tools="done")

    def _on_check_start(self, ev: dict) -> None:
        with self._body("execute"):
            self.check_ph = st.empty()
        self.check_ph.caption("🔎 결과를 점검하는 중…")
        self._set_flow(step="done", check="active")

    def _on_check(self, ev: dict) -> None:
        self.criteria = ev["criteria"]
        if self.check_ph is None:
            with self._body("execute"):
                self.check_ph = st.empty()
        met = sum(c["status"] == "met" for c in self.criteria)
        with self.check_ph.container(border=True):
            st.markdown(f"🔎 **점검** :gray[(단계 {ev['step']} 결과)]")
            st.markdown(f"📝 **작업 메모에 기록**: {badges(ev['observation'])}")
            st.markdown(f"**판단**: {DECISION_LABEL.get(ev['decision'], ev['decision'])} — {ev['reason']}")
            st.caption(f"완료 조건 {met}/{len(self.criteria)} 충족: " + " ".join(
                f"{CRITERION_ICON[c['status']]}{c['id']}" for c in self.criteria))
        self.check_ph = None
        self._draw_criteria()
        self._set_flow(check="done")

    def _on_evaluate(self, ev: dict) -> None:
        self.criteria = ev["criteria"]
        with self._body("evaluate"):
            st.markdown("  \n".join(f"{CRITERION_ICON[c['status']]} {c['id']}. {c['text']}"
                                    + (f" :gray[— {badges(c['note'])}]" if c.get("note") else "")
                                    for c in self.criteria))
            if ev["goal_met"]:
                st.success("🏁 **목표 달성** — 완료 조건을 모두 채웠습니다. 작업을 끝냅니다.")
            else:
                st.warning("**목표 미달** — 부족한 점: " + ("; ".join(ev["missing"]) or "일부 조건 미충족"))
                st.markdown(f"💭 {ev['feedback']}")
                if ev["retry"]:
                    st.markdown("↩️ **추가 조사로 돌아갑니다.** 계획에 단계를 더하고 다시 실행합니다.")
                else:
                    st.markdown("⏹️ 더 조사해도 해결하기 어렵거나 한도에 도달해, 확인하지 못한 부분을 답변에 밝히고 끝냅니다.")
        self._draw_criteria()
        if ev["retry"]:
            self._set_flow(evaluate="done", step="active")

    def _on_research_note(self, ev: dict) -> None:  # 예전 녹화 호환
        if self.more and ev.get("text"):
            with self._body("research"):
                with st.expander("📝 조사 메모"):
                    st.markdown(badges(ev["text"]))

    # ------------------------------------------------------------ 실전형 ReAct (create_agent)

    def _ensure_todos(self) -> None:
        if self.todos_ph is None:
            with self.pinned:
                with st.container(border=True):
                    st.markdown("##### 📝 모델이 직접 관리하는 할 일 목록 :gray[(write_todos 도구 · 실시간)]")
                    self.todos_ph = st.empty()

    # ------------------------------------------------------------ 멀티 에이전트: 팀 패널·레인

    def _is_specialist(self, ev: dict) -> bool:
        return self.mode == "multi" and ev.get("agent") not in (None, "supervisor")

    def _top_body(self):
        return self._body("multi" if self.mode == "multi" else "react")

    def _top_box(self):
        return self.turn_box or self._top_body()

    def _lane(self, ev: dict):
        return self.lanes.get(ev.get("run")) or self.agent_lane.get(ev.get("agent")) or self._top_box()

    def _agent_label(self, key: str) -> str:
        a = self.team.get(key, {})
        return f"{a.get('icon', '🤖')} {a.get('name', key)}"

    def _draw_team(self) -> None:
        if self.team_ph is None:
            return
        icon = {"idle": "⬜ 대기", "thinking": "🧠 생각 중", "working": "⏳ 작업 중", "done": "✅ 보고 완료"}
        rows = []
        for key, a in self.team.items():
            extra = f" :gray[({a['reports']}건 보고)]" if a.get("reports") else ""
            tools = f" :gray[— {', '.join(a['tools'])}]" if self.more and a.get("tools") else ""
            rows.append(f"{a['icon']} **{a['name']}** · {icon[a['status']]}{extra}{tools}")
        self.team_ph.markdown("  \n".join(rows))

    def _set_member(self, key: str, status: str) -> None:
        if key in self.team:
            self.team[key]["status"] = status
            self._draw_team()

    def _on_team(self, ev: dict) -> None:
        self.team = {a["key"]: a | {"status": "idle", "reports": 0} for a in ev["agents"]}
        with self.pinned:
            with st.container(border=True):
                st.markdown("##### 👥 에이전트 팀 :gray[(실시간)]")
                self.team_ph = st.empty()
                if not ev.get("company_docs"):
                    st.caption("🏢 회사기준 에이전트: 등록된 회사 문서가 없어 팀장이 맡기지 않을 수 있습니다 "
                               "(📚 문서 관리에서 회사 참고 문서를 등록하세요).")
        self._draw_team()

    def _on_handoff(self, ev: dict) -> None:
        """팀장 → 전문가 지시. 같은 호출에서 지시한 전문가들은 나란히 놓인 열에서 동시에 일한다."""
        idx = next((i for i, (to, _) in enumerate(self.pending_lanes) if to == ev["to"]), None)
        holder = self.pending_lanes.pop(idx)[1] if idx is not None else self._top_box()
        with holder:
            lane = st.container(border=True)
        with lane:
            st.markdown(f"📨 **팀장 → {self._agent_label(ev['to'])}**")
            st.markdown(f"**지시**: {ev['task']}")
            st.caption(f"💬 {ev['reason']}")
        self.lanes[ev["run"]] = lane
        self.agent_lane[ev["to"]] = lane
        self._set_member(ev["to"], "working")
        self._set_flow(supervisor="done", **{ev["to"]: "active"})

    def _on_agent_start(self, ev: dict) -> None:
        if self.more:
            with self._lane(ev):
                st.caption(f"🔧 쓸 수 있는 도구: {', '.join(ev['tools']) or '없음'}")

    def _on_agent_report(self, ev: dict) -> None:
        with self._lane(ev):
            st.markdown("📋 **보고서**" + ("" if ev.get("ok", True) else " ⚠️"))
            st.markdown(badges(ev["report"]))
            st.caption(f"모델 {ev['model_calls']}회 · 도구 {ev['tool_calls']}회 · {ev['elapsed_ms'] / 1000:.1f}초"
                       + (f" · 확인한 자료 {id_badges(ev['source_ids'])}" if ev["source_ids"] else ""))
        if ev["agent"] in self.team:
            self.team[ev["agent"]]["reports"] += 1
        self._set_member(ev["agent"], "done")
        self._set_flow(**{ev["agent"]: "done"})

    def _on_react_model_start(self, ev: dict) -> None:
        if self._is_specialist(ev):
            with self._lane(ev):
                st.caption(f"🧠 생각 #{ev['n']} — 다음 행동을 정합니다")
            return
        with self._top_body():
            self.turn_box = st.container(border=True)
        with self.turn_box:
            who = "🧑‍💼 팀장 " if self.mode == "multi" else ""
            st.markdown(f"**🧠 {who}모델 호출 #{ev['n']}** — 생각하고 다음 행동을 정합니다")
            if self.more:
                st.caption(f"모델이 보는 것: 시스템 프롬프트 + 대화·도구 결과 메시지 {ev['messages']}개")
        if self.mode == "multi":
            self._set_member("supervisor", "thinking")
            self._set_flow(supervisor="active")
        else:
            self._set_flow(model="active", tools="done" if self.flow.get("tools") else "idle")

    def _on_react_model_end(self, ev: dict) -> None:
        if self._is_specialist(ev):
            with self._lane(ev):
                if ev["tool_calls"]:
                    st.markdown(f"➡️ 도구 {len(ev['tool_calls'])}개: " + ", ".join(f"`{n}`" for n in ev["tool_calls"]))
                else:
                    st.markdown("📝 조사를 마치고 보고서를 씁니다")
            return
        asks = [n for n in ev["tool_calls"] if n.startswith("ask_")]
        with self._top_box():
            if ev["tool_calls"]:
                text = self.answer.strip() or ev.get("text", "")
                if text:  # 도구를 부르기 전에 모델이 쓴 글 = 겉으로 드러난 생각
                    st.markdown(f"💬 _{badges(text)}_")
                if asks:
                    st.markdown(f"➡️ **결정**: 전문가 {len(asks)}명에게 일을 맡김"
                                + (" — **동시에(병렬로)** 진행" if len(asks) > 1 else "")
                                + " (" + ", ".join(self._agent_label(n.removeprefix("ask_")) for n in asks) + ")")
                    cols = st.columns(len(asks))
                    self.pending_lanes += [(n.removeprefix("ask_"), c) for n, c in zip(asks, cols)]
                others = [n for n in ev["tool_calls"] if not n.startswith("ask_")]
                if others:
                    st.markdown(f"➡️ **결정**: 도구 {len(others)}개 호출 (" + ", ".join(f"`{n}`" for n in others) + ")")
            else:
                who = "팀장이 보고서를 모아 " if self.mode == "multi" else ""
                st.markdown(f"🏁 **결정**: {who}도구를 더 부르지 않고 답변을 씁니다 — 스스로 멈추려 함. "
                            "이제 근거 검증 middleware 가 이 답변을 확인합니다.")
        if not ev["tool_calls"]:
            self.last_draft = ev.get("text", "")
        if ev["tool_calls"] and self.answer:
            self.answer = ""
            self.answer_ph.empty()
        if self.mode == "multi":
            self._set_member("supervisor", "idle" if ev["tool_calls"] else "thinking")
        else:
            self._set_flow(model="done")

    def _on_todos(self, ev: dict) -> None:
        icon = {"pending": "⬜", "in_progress": "⏳", "completed": "✅"}
        rows = [f"{icon.get(t.get('status'), '⬜')} {t.get('content', '')}" for t in ev["todos"]]
        done = sum(t.get("status") == "completed" for t in ev["todos"])
        self._ensure_todos()
        self.todos_ph.markdown(f":gray[모델 호출 #{ev['turn']}에서 갱신 · {done}/{len(rows)} 완료]  \n"
                               + "  \n".join(rows))
        with self._top_box():
            st.markdown(f"📝 **할 일 목록 갱신** (`write_todos`) — {done}/{len(rows)} 완료")

    def _on_token(self, ev: dict) -> None:
        self.answer += ev["text"]
        self.answer_ph.markdown(badges(self.answer) + " ▌")

    def _on_verify(self, ev: dict) -> None:
        """근거 검증 (에이전트 안). 문제가 있으면 피드백과 함께 되돌려 보내 다시 쓰게 한다."""
        react = ev.get("where") == "react"
        box = self._top_box() if react else self._body("verify")
        with box:
            where = "after_model middleware" if react else "근거 검증 단계"
            with st.container(border=True):
                st.markdown(f"✅ **근거 검증** :gray[(에이전트 안 · {where} · {ev.get('attempt', 1)}차)]")
                if ev["cited"]:
                    st.markdown(f"실제로 읽은 자료와 일치하는 인용 {len(ev['cited'])}개 {id_badges(ev['cited'])}")
                for p in ev["problems"]:
                    st.warning(p)
                if ev.get("retry"):
                    st.markdown("↩️ **피드백과 함께 돌려보냄** — " +
                                ("모델이 이 지적을 읽고 답변을 다시 씁니다." if react else "답변 작성 단계가 고쳐 씁니다."))
                    draft = self.answer.strip() or self.last_draft
                    if draft:
                        with st.expander("📝 돌려보낸 답변 초안"):
                            st.markdown(badges(draft))
                elif ev["problems"]:
                    st.markdown("⏭️ 재작성 한도에 도달해 그대로 넘깁니다. 남은 가짜 인용은 최종 안전장치가 지웁니다.")
                else:
                    st.markdown("👍 **통과** — 문제가 없어 다음으로 넘어갑니다.")
                if ev["unused"] and self.more:
                    st.caption(f"조사했지만 답변에는 쓰지 않은 자료 {len(ev['unused'])}개: {', '.join(ev['unused'])}")
        if ev.get("retry"):
            self.answer = ""
            self.answer_ph.caption("✍️ 근거 검증 피드백을 반영해 답변을 다시 쓰는 중…")
        if react:
            top = "supervisor" if self.mode == "multi" else "model"
            self._set_flow(verify="done", **{top: "active" if ev.get("retry") else self.flow.get(top, "done")})

    def _on_guardrail(self, ev: dict) -> None:
        """최종 안전장치 (에이전트 밖): 에이전트가 끝난 뒤 남은 가짜 인용을 코드로 지운다."""
        with self._body("guardrail"):
            if ev["removed"]:
                st.warning(f"에이전트가 끝까지 고치지 못한 가짜 인용 {', '.join(ev['removed'])} 를 지웠습니다.")
            else:
                st.markdown("👍 남은 문제 없음 — 그대로 내보냅니다.")
            st.caption("에이전트에게 다시 고치게 하지 않고, 무조건 지켜야 할 규칙만 코드로 강제하는 마지막 필터입니다.")

    def _on_sources(self, ev: dict) -> None:
        self.sources = ev["items"]

    def _on_final(self, ev: dict) -> None:
        self.final = ev
        if self.mode == "multi":
            self._set_member("supervisor", "done")
        if ev.get("criteria"):
            self.criteria = ev["criteria"]
            self._draw_criteria()
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

        if stats.get("by_agent"):  # 멀티 에이전트: 누가 얼마나 썼나
            st.markdown("##### 👥 에이전트별 사용량")
            st.dataframe([{
                "에이전트": self._agent_label(k), "LLM 호출": v["llm_calls"], "도구 호출": v["tool_calls"],
                "총 토큰": v["usage"]["total"], "비용(USD)": v["cost_usd"],
                "비중": f"{v['usage']['total'] / max(usage.get('total', 1), 1):.0%}",
            } for k, v in stats["by_agent"].items()], hide_index=True, width="stretch",
                column_config={"비용(USD)": st.column_config.NumberColumn(format="$%.5f")})

        with st.expander(f"🪙 토큰 사용량 · 비용 상세 — {fmt_cost(stats.get('cost_usd'), self.pricing.get('usd_krw'))}"):
            rows = []
            for c in self.llm_calls:
                u = c["usage"]
                rows.append({
                    "#": str(c["call_no"]),
                    **({"에이전트": self._agent_label(c["agent"]) if c.get("agent") else ""} if self.mode == "multi" else {}),
                    "단계": STAGES.get(c["stage"], ("", c["stage"], ""))[1], "목적": c["purpose"],
                    "모델": c["model"], "입력 전체": u["input"], "└ 캐시 안 됨": u["uncached_input"],
                    "└ 캐시 읽기": u["cache_read"], "└ 캐시 쓰기": u["cache_write"], "출력 전체": u["output"],
                    "└ 추론": u["reasoning"], "웹검색": c.get("web_search_calls", 0),
                    "비용(USD)": c["cost_usd"], "시간(초)": round(c["elapsed_ms"] / 1000, 2),
                })
            if rows:
                rows.append({
                    "#": "합계", **({"에이전트": ""} if self.mode == "multi" else {}),
                    "단계": "", "목적": "", "모델": "", "입력 전체": usage.get("input", 0),
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
