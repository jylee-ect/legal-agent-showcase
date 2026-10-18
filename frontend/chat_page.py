"""에이전트 시연 페이지 (app.py 의 st.navigation 에서 실행)."""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

import client
from labels import AGENT_EXPLAINER, DETAIL_LEVELS, EXAMPLES, MODES, TRIGGER_EXAMPLES
from render import TurnRenderer, flow_dot

ss = st.session_state
ss.setdefault("messages", [])  # {"role": "user", "content", "attachment"} | {"role": "assistant", "events", "answer"}
ss.setdefault("attachment", None)  # {"filename", "content", "outline"}
ss.setdefault("pending", None)  # 예시 버튼으로 들어온 질문
ss.setdefault("replay", None)  # 재생할 녹화 파일 경로
ss.setdefault("pending_trigger", None)  # 트리거 시뮬레이션 버튼으로 들어온 사건


def attach(filename: str, content: str) -> None:
    info = client.attachment_outline(filename, content) or {}
    ss.attachment = {"filename": filename, "content": content, "outline": info.get("outline", [])}


def fire_trigger(tr: dict) -> None:
    """질문이 아닌 사건(파일 도착·정기 일정)으로 에이전트를 깨운다."""
    if tr["sample"]:
        attach_sample(tr["sample"])
    else:
        ss.attachment = None
    labels = {"file": "📂 파일 도착", "schedule": "⏰ 정기 일정"}
    ss.pending_trigger = {"type": tr["type"], "label": labels[tr["type"]],
                          "detail": tr["detail"].format(sample=tr["sample"] or "")}
    st.rerun()


def attach_sample(name: str) -> None:
    path = client.SAMPLES_DIR / name
    attach(path.name, client.decode_text(path.read_bytes()))


# ---------------------------------------------------------------- 사이드바

health = client.health()
library = client.cached_library() or {"index": "", "docs": []}
titles = {d["id"]: d["title"] for d in library["docs"]}

with st.sidebar:
    st.markdown("## ⚖️ 계약서 검토 AI 에이전트")
    if health:
        ok = lambda v: "✅" if v else "❌"
        st.caption(f"백엔드 연결됨 · 모델 `{health['model']}`  \n"
                   f"OpenAI 키 {ok(health['openai_key'])} · 법령 API {ok(health['law_api'])}")
    else:
        st.error(f"백엔드({client.BACKEND_URL})에 연결할 수 없습니다. 녹화 재생만 가능합니다.")

    st.markdown("#### 🤖 에이전트 방식")
    agent_mode = st.segmented_control("에이전트 방식", list(MODES), format_func=lambda m: MODES[m]["label"],
                                      default="loop", key="agent_mode", label_visibility="collapsed") or "loop"
    st.caption(f"`{MODES[agent_mode]['engine']}`  \n{MODES[agent_mode]['desc']}")

    st.markdown("#### 🗺️ 에이전트 흐름")
    flow_ph = st.empty()
    flow_ph.graphviz_chart(flow_dot({}, agent_mode), width="stretch")

    with st.expander("🔔 트리거 시뮬레이션 (질문이 아닌 사건)"):
        for tr in TRIGGER_EXAMPLES:
            if st.button(f"{tr['icon']} {tr['title']}", key=f"side_tr_{tr['type']}", width="stretch"):
                fire_trigger(tr)

    with st.expander("🧠 AI 에이전트란?"):
        st.markdown(AGENT_EXPLAINER)

    with st.expander("📚 참고 문서 라이브러리 (llms.txt)"):
        st.caption("DB 없이, 에이전트는 이 목록을 먼저 보고 필요한 문서의 조항만 골라 읽습니다.")
        st.page_link("library_page.py", label="문서 등록·관리 페이지로", icon="📚")
        st.code(library["index"] or "(백엔드 연결 필요)", language="markdown", wrap_lines=True)
        if library["docs"]:
            pick = st.selectbox("문서 원문 보기", list(titles), format_func=lambda c: titles[c])
            detail = client.library_doc(pick)
            if detail:
                for s in detail["sections"]:
                    st.markdown(f"**{s['id']}({s['title']})**")
                    st.text(s["text"].split("\n", 1)[-1])

    st.markdown("#### ⚙️ 보기 설정")
    detail_level = st.segmented_control("상세도", DETAIL_LEVELS, default="자세히") or "자세히"
    delay = st.slider("발표 모드: 단계 사이 지연(초)", 0.0, 2.0, 0.0, 0.25,
                      help="청중이 따라올 수 있게 각 단계 사이에 잠깐 멈춥니다.")
    auto_collapse = st.toggle("완료된 단계 자동 접기", value=False,
                              help="기본은 모든 단계를 펼쳐 둡니다. 켜면 끝난 단계는 접고 진행 중인 단계만 펼칩니다. "
                                   "어느 쪽이든 단계 제목을 클릭하면 접거나 펼칠 수 있습니다.")
    record = st.toggle("이번 실행 녹화하기", help="recordings/ 폴더에 저장해 두면 네트워크 없이도 재생할 수 있습니다.")

    st.markdown("#### 🎬 녹화 재생")
    recordings = client.list_recordings()
    if recordings:
        chosen = st.selectbox("녹화 파일", recordings, format_func=lambda p: p.stem)
        if st.button("▶️ 재생", width="stretch"):
            ss.replay = str(chosen)
    else:
        st.caption("저장된 녹화가 없습니다.")

    st.divider()
    if st.button("🗑️ 대화 초기화", width="stretch"):
        ss.messages, ss.attachment = [], None
        st.rerun()


# ---------------------------------------------------------------- 메인

st.title("⚖️ 계약서 검토 AI 에이전트")
st.caption("질문을 입력하면 AI 에이전트가 **어떻게 판단하고, 계획하고, 도구를 쓰는지** 실시간으로 보여줍니다. "
           "계약서(.md, .txt)를 첨부하면 그 내용도 검토합니다.")

if ss.attachment:
    att = ss.attachment
    c1, c2 = st.columns([5, 1])
    c1.info(f"📎 첨부된 계약서: **{att['filename']}** · {len(att['outline'])}개 조항 "
            "(질문할 때마다 에이전트에게 함께 전달됩니다)")
    if c2.button("첨부 해제", width="stretch"):
        ss.attachment = None
        st.rerun()
    with st.expander("에이전트가 처음 보는 첨부 계약서 목차"):
        st.markdown("\n".join(f"- {o['id']}" + (f"({o['title']})" if o["title"] else "") for o in att["outline"]))

if not ss.messages and not ss.replay:
    st.markdown("##### 예시 질문")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{ex['icon']} {ex['title']}**")
            st.caption(ex["question"] + (f"  \n📎 {ex['sample']}" if ex["sample"] else ""))
            if st.button("실행", key=f"ex_{ex['title']}", width="stretch"):
                if ex["sample"]:
                    attach_sample(ex["sample"])
                ss.pending = ex["question"]
                st.rerun()
    st.markdown("##### 🔔 질문이 아닌 트리거로 시작하기 :gray[(시뮬레이션)]")
    st.caption("에이전트는 질문뿐 아니라 '파일이 도착했다', '점검 일정이 됐다' 같은 사건으로도 일을 시작합니다.")
    cols = st.columns(len(TRIGGER_EXAMPLES) + 1)
    for col, tr in zip(cols, TRIGGER_EXAMPLES):
        with col.container(border=True, height="stretch"):
            detail = tr["detail"].format(sample=tr["sample"] or "")
            st.markdown(f"**{tr['icon']} {tr['title']}**")
            st.caption(detail)
            if st.button("트리거 발생", key=f"tr_{tr['type']}", width="stretch"):
                fire_trigger(tr)
    with st.expander("📎 샘플 계약서 첨부하기"):
        for p in client.list_samples():
            if st.button(f"{p.name} 첨부", key=f"sample_{p.name}"):
                attach_sample(p.name)
                st.rerun()

def show_request(content: str, attachment_name: str | None, trigger: dict | None) -> None:
    """사용자 질문이면 말풍선, 질문이 아닌 트리거면 🔔 사건으로 보여준다."""
    if trigger and trigger.get("type", "question") != "question":
        with st.chat_message("user", avatar="🔔"):
            st.markdown(f"**{trigger.get('label', '트리거')}** :gray[(시뮬레이션)]  \n{trigger.get('detail', '')}")
            if attachment_name:
                st.caption(f"📎 {attachment_name}")
        return
    with st.chat_message("user"):
        st.markdown(content)
        if attachment_name:
            st.caption(f"📎 {attachment_name}")


# 지난 대화 다시 그리기
for turn_no, msg in enumerate(ss.messages):
    if msg["role"] == "user":
        show_request(msg["content"], msg.get("attachment"), msg.get("trigger"))
    else:
        with st.chat_message("assistant"):
            r = TurnRenderer(detail_level, titles, auto_collapse=auto_collapse, key=f"turn{turn_no}")
            for ev in msg["events"]:
                r.handle(ev)


def history_payload() -> list[dict]:
    out = []
    for m in ss.messages[-6:]:
        if m["role"] == "user":
            content = m["content"] or (m.get("trigger") or {}).get("detail", "")
        else:
            content = m.get("answer", "")
        out.append({"role": m["role"], "content": content})
    return out


def play(question: str, attachment_name: str | None, events, trigger: dict | None = None) -> list[dict]:
    """요청 하나를 실행(또는 재생)하며 화면에 그린다."""
    show_request(question, attachment_name, trigger)
    with st.chat_message("assistant"):
        r = TurnRenderer(detail_level, titles, flow_ph, auto_collapse=auto_collapse, key=f"turn{len(ss.messages) + 1}")
        for ev in events:
            r.handle(ev)
            if delay and ev["type"] not in ("token", "plan_update", "llm_call"):
                time.sleep(delay)
    ss.messages.append({"role": "user", "content": question, "attachment": attachment_name, "trigger": trigger})
    ss.messages.append({"role": "assistant", "events": r.events, "answer": r.answer})
    return r.events


def replay_events(events: list[dict]):
    for ev in events:
        yield ev
        if ev["type"] == "token":
            time.sleep(0.01)
        elif not delay:
            time.sleep(0.15)


submitted = st.chat_input("계약서에 대해 무엇이든 물어보세요 (파일 첨부: .md, .txt)",
                          accept_file=True, file_type=["md", "txt"])
question = None
if submitted:
    if submitted.files:
        f = submitted.files[0]
        attach(f.name, client.decode_text(f.getvalue()))
    question = submitted.text.strip() or (
        "첨부한 계약서에서 불리하거나 위험한 조항이 있는지 검토해 주세요." if submitted.files else None)
elif ss.pending:
    question, ss.pending = ss.pending, None

trigger = None
if ss.pending_trigger:
    trigger, ss.pending_trigger = ss.pending_trigger, None
    question = ""

if ss.replay:
    path, ss.replay = Path(ss.replay), None
    recorded = client.load_recording(path)
    start = next((e for e in recorded if e["type"] == "run_start"), {})
    rec_trigger = start.get("trigger")
    if rec_trigger and rec_trigger.get("type") != "question":
        rec_trigger = rec_trigger | {"detail": "🎬 (녹화 재생) " + rec_trigger.get("detail", "")}
    play(f"🎬 (녹화 재생) {start.get('question') or path.stem}", start.get("attachment"), replay_events(recorded),
         rec_trigger)
elif question is not None and (question or trigger):
    att = ss.attachment
    history = history_payload()
    events = play(question, att["filename"] if att else None, client.stream_chat(question, history, att, trigger, agent_mode),
                  trigger)
    if record and any(e["type"] == "final" for e in events):
        saved = client.save_recording(events)
        st.toast(f"녹화 저장됨: {saved.name}")
