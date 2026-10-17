"""계약서 검토 AI 에이전트 (Streamlit, 0.0.0.0:3000).

    streamlit run app.py
"""

import streamlit as st

st.set_page_config(page_title="계약서 검토 AI 에이전트", page_icon="⚖️", layout="wide")

page = st.navigation([
    st.Page("chat_page.py", title="에이전트 시연", icon="⚖️", default=True),
    st.Page("library_page.py", title="문서 관리", icon="📚"),
])
page.run()
