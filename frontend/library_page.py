"""문서 관리 페이지: 표준계약서·회사 참고 문서 등록과 llms.txt 갱신."""

from __future__ import annotations

import streamlit as st

import client
from render import fmt_cost

CATEGORIES = {"standard": "📚 표준계약서", "company": "🏢 회사 참고 문서"}

ss = st.session_state
ss.setdefault("proposal", None)
ss.setdefault("upload_key", 0)  # 등록 후 업로더를 비우기 위한 키

health = client.health()
library = client.cached_library() or {"index": "", "docs": [], "missing": []}

st.title("📚 문서 관리")
st.caption("에이전트가 참고하는 **표준계약서**와 **회사 참고 문서**를 등록합니다. "
           f"문서를 올리면 GPT 모델(`{health['model'] if health else '?'}`)이 내용을 읽고 "
           "문서 목록인 llms.txt 의 수정안을 만듭니다. 확인 후 반영하면 다음 질문부터 에이전트가 이 문서를 씁니다.")
if not health:
    st.error(f"백엔드({client.BACKEND_URL})에 연결할 수 없습니다.")
elif not health["openai_key"]:
    st.warning("OpenAI API 키가 없어 문서 분석을 할 수 없습니다. 백엔드 터미널에 환경변수 GPT_API 를 설정하고 백엔드를 다시 실행하세요.")
if library.get("missing"):
    st.warning("llms.txt 에는 있지만 파일이 없는 문서: " + ", ".join(library["missing"]))

tab_add, tab_list, tab_index = st.tabs(["➕ 문서 등록", f"📋 등록된 문서 ({len(library['docs'])})", "📄 llms.txt"])


# ---------------------------------------------------------------- 등록

def show_proposal(p: dict) -> None:
    doc = p["doc"]
    st.markdown("#### 2. GPT 수정안 확인")
    with st.container(border=True):
        st.markdown(f"**{CATEGORIES[doc['category']]}** · **{doc['title']}**  \n"
                    f"ID `{doc['id']}` · 저장 경로 `data/{doc['path']}` · 문서 구조: {p['structure']} "
                    f"{len(p['outline'])}개 항목")
        st.markdown(f"**목록 설명**: {doc['description']}")
        if p["truncated_for_llm"]:
            st.caption("문서가 길어 앞부분만 모델에 보냈습니다. 문서 전체는 그대로 저장됩니다.")
    st.info(f"💭 **GPT가 설명한 변경 내용**: {p['change_summary']}")
    for w in p["warnings"]:
        st.warning(f"🛡️ 코드가 보정함: {w}")
    u = p["usage"]
    st.caption(f"🪙 `{p['model']}` · 입력 {u['input']:,} (캐시 읽기 {u['cache_read']:,} / 캐시 쓰기 {u['cache_write']:,}) · "
               f"출력 {u['output']:,} (추론 {u['reasoning']:,}) · {fmt_cost(p['cost_usd'], p.get('usd_krw'))} · "
               f"{p['elapsed_ms'] / 1000:.1f}초")

    diff_tab, edit_tab, outline_tab = st.tabs(["변경 사항 (diff)", "✏️ 수정안 직접 고치기", "에이전트가 보게 될 목차"])
    with diff_tab:
        st.code(p["diff"] or "(변경 없음)", language="diff", wrap_lines=True)
    with edit_tab:
        st.caption("반영 전에 직접 고칠 수 있습니다. 저장할 때 형식과 경로를 다시 검사합니다.")
        edited = st.text_area("llms.txt 수정안", p["llms_after"], height=360, key=f"edit_{p['proposal_id']}",
                              label_visibility="collapsed")
    with outline_tab:
        st.markdown("\n".join(f"- {o['id']}" + (f"({o['title']})" if o["title"] else "") + f" :gray[{o['chars']}자]"
                              for o in p["outline"]))

    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("✅ 반영하기", type="primary", width="stretch"):
        try:
            result = client.commit_document(p["proposal_id"], edited)
        except client.ApiError as e:
            st.error(f"반영하지 못했습니다: {e}")
        else:
            ss.proposal = None
            ss.upload_key += 1
            client.cached_library.clear()
            st.toast(f"등록 완료: data/{result['saved']} (이전 llms.txt 백업: {result['backup']})", icon="✅")
            st.rerun()
    if c2.button("✖️ 취소", width="stretch"):
        ss.proposal = None
        st.rerun()


with tab_add:
    st.markdown("#### 1. 문서 올리기")
    category = st.radio("분류", list(CATEGORIES), format_func=CATEGORIES.get, horizontal=True,
                        help="표준계약서는 비교 기준(출처 C#), 회사 참고 문서는 회사 내부 기준(출처 K#)으로 쓰입니다.")
    uploaded = st.file_uploader("문서 파일 (.md, .txt)", type=["md", "txt"], key=f"upload_{ss.upload_key}")
    note = st.text_input("운영자 메모 (선택)", placeholder="예: 외주 개발 계약을 검토할 때 반드시 확인하는 사내 기준",
                         help="GPT가 llms.txt 설명을 쓸 때 참고합니다.")

    if uploaded:
        content = client.decode_text(uploaded.getvalue())
        outline = client.attachment_outline(uploaded.name, content)
        if outline:
            st.caption(f"미리보기: {len(content):,}자 · {len(outline['outline'])}개 항목 — "
                       + ", ".join(o["id"] + (f"({o['title']})" if o["title"] else "") for o in outline["outline"][:8])
                       + (" …" if len(outline["outline"]) > 8 else ""))
        if st.button("🤖 GPT로 분석하고 llms.txt 수정안 만들기", type="primary",
                     disabled=not (health and health["openai_key"])):
            with st.status("GPT 모델이 문서를 읽고 llms.txt 수정안을 만드는 중…", expanded=True) as status:
                st.write("1) 문서 내용과 현재 llms.txt 를 모델에 전달")
                st.write("2) 모델이 문서 ID·제목·설명을 정하고 전체 목록을 다시 작성")
                st.write("3) 코드가 결과를 검사·보정하고 변경 사항(diff)을 만듦")
                try:
                    ss.proposal = client.propose_document(category, uploaded.name, content, note)
                except client.ApiError as e:
                    status.update(label=f"실패: {e}", state="error")
                else:
                    status.update(label="수정안 완성 — 아래에서 확인 후 반영하세요", state="complete", expanded=False)

    if ss.proposal:
        st.divider()
        show_proposal(ss.proposal)


# ---------------------------------------------------------------- 목록

with tab_list:
    for cat, label in CATEGORIES.items():
        docs = [d for d in library["docs"] if d["category"] == cat]
        st.markdown(f"#### {label} ({len(docs)})")
        if not docs:
            st.caption("등록된 문서가 없습니다.")
        for d in docs:
            with st.container(border=True):
                c1, c2 = st.columns([6, 1])
                c1.markdown(f"**{d['title']}** :gray[`{d['id']}` · data/{d['path']} · {d['sections']}개 항목]  \n"
                            f"{d['description']}")
                with c2.popover("🗑️ 삭제", width="stretch"):
                    st.caption("llms.txt 에서 빼고 파일은 data/.history/deleted/ 로 옮깁니다.")
                    if st.button("삭제 확인", key=f"del_{d['id']}", type="primary"):
                        try:
                            client.delete_document(d["id"])
                        except client.ApiError as e:
                            st.error(str(e))
                        else:
                            client.cached_library.clear()
                            st.rerun()
                with st.expander("내용 보기"):
                    detail = client.library_doc(d["id"])
                    if detail:
                        for s in detail["sections"]:
                            st.markdown(f"**{s['id']}" + (f"({s['title']})" if s["title"] else "") + "**")
                            st.text(s["text"].split("\n", 1)[-1] if s["text"].startswith(("#", "제")) else s["text"])


# ---------------------------------------------------------------- llms.txt

with tab_index:
    st.caption("에이전트가 질문마다 가장 먼저 읽는 문서 목록입니다. 문서를 등록하면 GPT가 이 파일을 갱신합니다.")
    st.code(library["index"] or "(백엔드 연결 필요)", language="markdown", wrap_lines=True)
