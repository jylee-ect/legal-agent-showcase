# 계약서 검토 AI 에이전트 쇼케이스 — Development Plan

> Goal: show non-experts **how an AI agent works**, not just what it answers.
> A user asks a contract question in Korean and may attach a contract (.md/.txt). The app shows each step live: how the agent reads the question, decides whether the attachment matters, makes a plan, picks clauses, checks official law and precedent text, searches the web, writes an answer, and checks its sources.

**Status (2026-10-18):** Phases 0–5 and token/cost tracking are implemented. The first live run on `gpt-6-luna` (cafe contract + 교육비 question) succeeded: 41 s, 4 agent loops, 8 LLM calls, 8 tool calls, 34K tokens, $0.0138, and all 4 citations were verified. The multi-agent extension is designed in §10 but not built yet.

---

## 1. Decisions

| Topic | Decision |
|---|---|
| Framework | LangGraph `StateGraph` with LangChain `ChatOpenAI` (Responses API) |
| Model | `gpt-6-luna` for testing → `gpt-6.1-sol` later (`OPENAI_MODEL`) |
| Web search | OpenAI built-in `web_search` tool, wrapped as an agent tool so it shows up as a separate visible step |
| Laws and precedents | 국가법령정보센터 Open API (statute article text, precedent search and detail) |
| Contract data | `data/llms.txt` index + 4 mock standard contracts in Markdown, no DB |
| User contracts | `.md`/`.txt` attachment through Streamlit `chat_input(accept_file=True)`, parsed into 제N조 sections |
| Keys | OpenAI key only from env var `GPT_API` (never in a file); law OC from `api_keys.txt` (`law:...`) or `LAW_API_OC`, re-read on every request |
| Serving | FastAPI on **:7000** (NDJSON event stream), Streamlit on **:3000** |
| UI language | Korean |

---

## 2. Architecture

```
┌──────────────────────────┐  POST /api/chat/stream   ┌──────────────────────────────────┐
│ Streamlit UI (:3000)     │ ───────────────────────▶ │ FastAPI agent server (:7000)     │
│  chat + file attachment  │ ◀── NDJSON event stream  │  LangGraph agent                 │
│  live step timeline      │                          │  tools: 첨부 계약서 / 표준계약서    │
│  flow diagram, sources   │  GET /api/contracts      │         법령·판례 API / 웹검색     │
│  record / replay         │  POST /api/attachment/.. │  source registry, citation check │
└──────────────────────────┘                          └──────────────────────────────────┘
                                                        │            │              │
                                                     OpenAI     data/llms.txt   law.go.kr DRF
                                                  (LLM + web)   contracts/*.md
```

The request carries `question`, the last few turns of `history`, and the optional `attachment {filename, content}`. The backend is stateless, and the UI keeps the attachment for follow-up questions.

---

## 3. Agent loop (updated 2026-10-18: full, general agent loop)

The demo teaches how agents work, so the graph follows the complete, general loop rather than a simplified pipeline:

```
🔔 trigger (question / file arrived / schedule)
 → define_goal ─┬─ direct → direct_answer → END                       (off-topic chit-chat only)
                └─ plan → pick_step → act ⇄ tools → check ─┬─ continue / retry step / revise plan → pick_step
                                                            └─ all completion criteria met → write
                   write → verify ─┬─ citation problems → write again with feedback   (max 1 retry)
                                   └─ pass → evaluate ─┬─ goal not met + fixable → add steps → pick_step   (max 1 retry)
                                                       └─ met, or limits reached → end of agent
 → 🛡️ guardrail (outside the agent, code): strip any fake citations still left
```

| Node | What it does | What the audience sees |
|---|---|---|
| `define_goal` | Reads the trigger, history, `llms.txt`, and **only the outline** of the attachment. Writes a one-line **goal** and 2–5 checkable **completion criteria**, plus route, issues, whether to read the attachment, and candidate docs. Non-question triggers always route to work | 🔔 trigger, 🎯 goal, ✅ completion criteria, "💭 판단 근거", "🔀 분기" |
| `plan` | 2–6 tool-using steps; each step lists which criteria it serves. No "종합/답변" steps | "처음 세운 계획" + live plan in the 📌 panel |
| `pick_step` | Code: picks the next pending step; closes execution when none are left or limits are hit | ▶ 단계 N, 🔁 루프 N회차 |
| `act` | LLM decides the tool calls for **the current step only** (max 4 in parallel). It sees the 작업 메모 and recent tool results from earlier steps (memory) | 💭 thought, tool cards with **reason** |
| `tools` | Runs the calls concurrently, registers sources, enforces the budget, appends to `tool_log` | 👀 observation + source badges |
| `check` | Observes the step result → writes a **작업 메모** (memory) → updates criteria status → decides `continue` / `retry_step` / `revise` (drop/add steps) / `finish`. If every criterion is met, code forces `finish` | 🔎 점검 box, 🔄 계획 수정 box (🚫 cancelled, 🆕 added), live criteria checklist |
| `write` | Streams the answer from the 작업 메모 + collected sources, citing registry IDs only | answer typing out with `[C1]` badges |
| `verify` | **Code-based** citation check **inside the agent**. On problems (unknown IDs, no citations) it routes back to `write` with feedback (`verify_feedback`), max 1 retry | ✅ 근거 검증 box: problems, "↩️ 피드백과 함께 돌려보냄", the returned draft |
| `guardrail` (in `_finish`) | **Outside the agent**: after the agent ends, removes any fake citations still left. Never asks the agent to fix anything | 🛡️ 최종 안전장치 stage |
| `evaluate` | Judges the **answer** against the completion criteria. If not met and fixable, adds up to 2 steps and loops back (once). Otherwise ends and appends "확인하지 못한 부분" to the answer | 🏁 목표 달성 / 미달 + next action |
| `direct_answer` | Short answer without tools | the agent choosing *not* to work |

Limits (in `graph.py`): `MAX_STEP_RUNS=10`, `MAX_STEP_ROUNDS=2` (tries per step), `MAX_REVISIONS=4`, `MAX_EVAL_RETRIES=1`, plus the tool budget `MAX_TOOL_CALLS` (default 15).

**Triggers:** `ChatRequest.trigger = {type: question|file|schedule, detail}`. The UI simulates two non-question triggers: "📂 계약서 파일 도착" (attaches a sample contract, no question) and "⏰ 정기 점검 일정".

**Live results (gpt-6-luna, 2026-10-18):**
- cafe contract + 교육비 question: 77 s, 5 loops, 16 LLM calls, $0.0070, goal met. This was after the memory fix; the first run took 127 s with 9 loops and 4 plan revisions caused by a lost precedent ID.
- 📂 file trigger: the goal was derived from the event alone. The tool budget ran out because one step made 7 parallel calls, and the run honestly ended with "목표 미달: 판례 미확인". After that run the per-step limit was set to 4 and the budget raised to 15.
- ⏰ schedule trigger: 128 s, 6 loops, 1 plan revision, goal met.

### 3.0 Second mode: 실전형 ReAct (LangChain `create_agent`) — added 2026-10-18

The sidebar switches between **🎓 교과서형 루프** (the LangGraph graph above) and **⚡ 실전형 ReAct** (`backend/app/agent/react_agent.py`). `ChatRequest.mode` is `loop` or `react`.

```
trigger → [ model ⇄ tools → (answer attempt) → VerifyMiddleware.after_model ─problems→ jump_to model ] → 🛡️ guardrail (outside, code)
           middleware: ObserverMiddleware · VerifyMiddleware · TodoListMiddleware (write_todos) · ModelCallLimitMiddleware
```

- **Same tools, sources, citation check and cost tracking** as the loop. Only the control flow differs: there's no goal, plan, check or evaluate node. The model plans with `write_todos`, decides each next action, and decides when to stop.
- **Seeing inside the ReAct agent:**
  - `ObserverMiddleware.awrap_model_call` emits `react_model_start`/`react_model_end` per model call. The events carry the decision (tool names or final answer), any text written before tool calls, the reasoning summary, and tokens/cost.
  - `awrap_tool_call` emits `tool_call`/`tool_result` (with `reason`), turns `write_todos` calls into `todos` events, and enforces the tool budget.
  - Token streaming comes from `agent.astream(stream_mode=["messages","values"])`.
  - Raw chain-of-thought is not exposed by the API, so only the reasoning summary is visible.
- **Live comparison** (same cafe question, gpt-6-luna):

| | 교과서형 루프 | 실전형 ReAct |
|---|---|---|
| time | 77 s | 20.6 s |
| LLM calls | 16 | 5 |
| cost | $0.0070 | $0.0013 |
| precedent | searched **and read**, cited | found 6, **read none**, not cited |
| guarantee | goal/criteria checked by code and an evaluator | depends on the model's own judgment |

Teaching point: the explicit loop trades speed and cost for guaranteed checks. Production agents mostly use the ReAct form and push planning into tools.

### 3.0.1 Verification inside vs. guardrail outside (updated 2026-10-18)

| | Inside the agent: 근거 검증 | Outside the agent: 🛡️ 최종 안전장치 |
|---|---|---|
| when | every answer attempt | once, after the agent has finished |
| what | code checks citations (unknown IDs, no citations); ReAct also flags "precedents searched but none read" | strips fake citations still left |
| on failure | **feedback goes back to the agent**, which rewrites (loop: `verify → write`; ReAct: `after_model` → `jump_to: model`), max 1 retry | removes them; the agent is never asked to fix anything |
| code | `agent/verification.py: check_grounding, check_research, feedback_message` | `agent/verification.py: guardrail` (called from `graph._finish`) |

Live ReAct run after the change (same cafe question): 23.5 s, 6 model calls, $0.0017. This time the model read 3 precedents on its own, so verification passed first time. The retry paths are covered by tests (`[L7]`/`[L9]` sent back and fixed; failure to fix stripped by the guardrail; precedents searched but not read).

### 3.1 Tools

Every tool's first argument is `reason` (one plain-Korean sentence). That makes the "why" visible without exposing hidden chain-of-thought. Code knows the current step, so there's no `plan_step` argument.

| Tool | Label | Source ID |
|---|---|---|
| `read_uploaded_contract(sections)` | 📎 첨부 계약서 읽기 | `U#` |
| `read_reference_outline(doc_id)` | 📚 참고 문서 목차 보기 | — |
| `read_reference_sections(doc_id, sections)` | 📄 참고 문서 조항 읽기 | `C#` (표준계약서) / `K#` (회사 문서) |
| `search_laws(query)` | 🏛️ 법령 찾기 | — |
| `get_law_article(law_name, article)` | 📜 법조문 원문 확인 (`lawService.do?target=law&JO=`) | `L#` |
| `search_precedents(query)` | ⚖️ 판례 검색 (body-text search) | — (a list alone can't be cited) |
| `read_precedent(precedent_id)` | 📖 판례 읽기 (numeric 판례ID only) | `P#` |
| `web_search(query)` | 🌐 웹 검색 (OpenAI `web_search`, KR location, `url_citation`s) | `W#` |

Law tools are only offered when a `law` key exists. Public law.go.kr URLs are rebuilt, so the OC key never reaches the UI. Section IDs copied with their titles (`제13조(위약 예정의 금지)`) are accepted.

### 3.2 Sources and citations

- Each tool result is registered as `Source{id, kind, title, excerpt, url, content}`. Reading the same clause again reuses its ID.
- The writer receives every collected source's content and may only cite those IDs.
- The answer is checked against the registry. The 출처 cards and the disclaimer are built **by the UI from the registry**, so the model can't invent a URL.

---

## 4. Event stream (NDJSON, one JSON per line)

Every event carries `t` (ms since the run started) and `ts` (wall-clock ISO time with ms and timezone).

`run_start{question,trigger,mode,mode_label,middleware,…,pricing}` · `stage{stage,status}` (goal/plan/execute/write/verify/evaluate/direct; execute/write/verify/evaluate can repeat) · `context{items}` · `goal{data,criteria}` · `route{path,label}` · `plan{strategy,steps[no,goal,tool,criteria,status,added_by]}` · `loop{n}` · `step_start{no,goal,tool,criteria}` · `thought{text}` · `tool_call{id,tool,label,args,reason,step}` · `tool_result{id,ok,summary,source_ids,preview,elapsed_ms}` · `check_start{step}` · `check{step,observation,step_done,decision,reason,criteria,source_ids}` · `plan_revised{by,reason,dropped,added,steps}` · `plan_update{step_no,status}` (pending/in_progress/done/skipped/dropped) · `llm_call{…}` · `reasoning{stage,text}` · `token{text}` · `verify{ok,cited,unknown,unused,problems}` · `evaluate{goal_met,criteria,missing,feedback,retry,attempt}` · `sources{items}` · `verify{ok,cited,unknown,unused,problems,retry,attempt,where[,turn]}` · `guardrail{removed,cited}` · ReAct only: `react_model_start{n,messages}` · `react_model_end{n,tool_calls,final,text}` · `todos{turn,todos}` · `final{answer_md,cited_ids,goal_met,criteria,stats}` · `error{message}` · `done`

Old recordings (`analysis`, `research`, `research_note`) still replay.

### 4.1 Token usage and cost

- Every LLM call reports `usage = {input, cache_read, cache_write, uncached_input, output, reasoning, total}`. This includes the hidden call inside `web_search`.
  - `input` includes cache reads and writes.
  - `output` includes reasoning tokens, which are billed at the output rate.
- Cost = uncached × input rate + cache_read × cached rate + cache_write × cache-write rate + output × output rate, plus $10 per 1,000 web-search calls. Requests over 272K input tokens use the long-context rates.
- Rates live in `pricing.json`, taken from the official OpenAI pricing page on 2026-10-18. The file is re-read on every request; a model missing from it shows "단가 미설정". The KRW display uses the rate set in that file.

### 4.2 Terminology: turn vs loop vs LLM call

| Term | Meaning | Live run example |
|---|---|---|
| 대화 턴 (conversation turn) | one user question → one answer | 1 |
| 에이전트 루프 (agent loop; called a "turn" in some agent SDKs) | one plan step: act (decide tools) → tools → check (observe, update memory/criteria, decide next) | 5 (cafe run) |
| LLM 호출 | every model API call: goal, plan, act + check per loop (+ retries), write, evaluate, plus hidden web-search calls | 16 (cafe run) |
| 도구 호출 | individual tool executions; one loop can issue several in parallel (max 4) | 7 (cafe run) |

Live runs, history re-rendering and recorded replays all use the same renderer (`frontend/render.py`).

---

## 5. UI (Korean)

- **Sidebar:**
  - backend, key and law API status
  - **live flow diagram** with the current node highlighted
  - "AI 에이전트란?" explainer
  - `llms.txt` viewer and contract browser
  - detail level (간단 / 자세히 / 개발자)
  - presenter delay
  - record toggle and replay
- **Main:**
  - 5 example cards: 2 plain questions, 2 with sample attachments, 1 off-topic routing demo
  - attachment banner showing the outline the agent sees
  - chat with a step timeline, streamed answer, source cards and stats (time, agent loops, LLM calls, tool calls, total tokens, cost)
  - the whole "에이전트 작업 과정" is a collapsible `st.status`. Its label shows live progress, tokens and cost.
  - each stage is its own collapsible status. With "완료된 단계 자동 접기" on, only the running stage stays open.
  - past turns re-render collapsed
  - 🔁 loop markers in the research stage
  - per-LLM-call token captions (자세히/개발자)
  - a "토큰 사용량 · 비용 상세" table per call with totals and the pricing basis
  - the developer log shows `HH:MM:SS.mmm  +elapsed  type  {json}`
- **Samples:** `samples/프리랜서_앱개발_용역계약서.md` and `samples/카페_근로계약서.txt` contain deliberately problematic clauses (all IP including prior tech transferred, unlimited free warranty, 1%/day delay penalty with no cap, 80% probation wage, 2,000,000-won training-fee penalty, non-compete).

---

## 6. Project structure

```
agent_showcase/
├── api_keys.txt              # law:<OC> only (GPT key lives in env var GPT_API)
├── requirements.txt          # pip install -r requirements.txt (includes backend/ + frontend/ pinned deps)
├── requirements-dev.txt      # + pytest
├── data/llms.txt, data/contracts/*.md
├── samples/                  # contracts to attach in the demo
├── backend/app/
│   ├── main.py               # /health, /api/contracts, /api/attachment/outline, /api/chat/stream
│   ├── config.py, schemas.py
│   └── agent/  graph.py prompts.py tools.py law_api.py web_search.py documents.py sources.py context.py runner.py
├── frontend/  app.py render.py client.py labels.py .streamlit/config.toml
├── recordings/               # *.jsonl saved runs (created on first recording)
└── tests/                    # parsing, citations, full graph flow with a fake LLM
```

---

## 7. Milestones

| Phase | Scope | Status |
|---|---|---|
| 0 | venv, pinned requirements, servers on :7000/:3000 | ✅ |
| 1 | mock contracts, `llms.txt`, parser, attachment parsing (제N조 → headings → chunks, UTF-8/CP949) | ✅ |
| 2 | LangGraph nodes, Korean prompts, tools, source registry, law API | ✅ (law API verified live) |
| 3 | NDJSON streaming, event schema | ✅ |
| 4 | Streamlit UI, attachment, flow diagram, source cards | ✅ (verified with AppTest at all detail levels) |
| 5 | Detail levels, presenter mode, record/replay, examples, errors, disclaimer | ✅ |
| 6 | Token/cost tracking, timestamps, collapsible stages | ✅ |
| 7 | **Live GPT runs** of all 5 examples, prompt tuning, record demo runs | 🟡 4 live recordings exist; the 3 from 2026-10-17 predate cost tracking and should be re-recorded |
| 8 | Multi-agent mode (§10) | 📝 designed |
| 9 (optional) | `gpt-6.1-sol` switch-over and time/cost comparison | — |

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Network or API failure on stage | Record each demo scenario during rehearsal; replay mode |
| `gpt-6-luna` rejects `reasoning` params or structured output | Set `REASONING_EFFORT=` (empty) to drop reasoning params; structured output can switch to `method="function_calling"` in `graph._structured` |
| Slow runs (> 30–40 s) | Tool budget, parallel tool calls, every step streams so waiting feels active; lower `MAX_TOOL_CALLS` |
| Web results from blogs | Search instructions prefer government and legal sources; law API is the primary source for statutes and precedents |
| Hallucinated citations | Registry-only IDs, code verification, UI-built source list |
| Audience reads "thinking" literally | UI labels it "AI가 밝힌 판단 근거", explainer in the sidebar |
| Mistaken for real legal advice | Disclaimer under every cited answer; contracts labeled as mock |

---

## 9. Next steps once the GPT key is added

1. Set env var `GPT_API` in the backend terminal; the sidebar should show `OpenAI 키 ✅`.
2. Run the 5 example cards and check: route decisions, plan quality, whether the agent uses `get_law_article` and `read_precedent` before citing, and answer length.
3. Tune `prompts.py` if needed, then record each scenario with "이번 실행 녹화하기".
4. Switch to `gpt-6.1-sol` with `OPENAI_MODEL` and re-record.

---

## 9a. Document intake (📚 문서 관리 page) — implemented 2026-10-18

- Library = `data/llms.txt` with two sections. A document's category comes from its path: `contracts/` = 표준계약서 (source `C#`), `company/` = 회사 참고 문서 (source `K#`).
- The agent reads both categories with the same tools: `read_reference_outline(doc_id)` and `read_reference_sections(doc_id, sections)`.
- Registration is two-step, implemented in `backend/app/library_admin.py`:
  1. `POST /api/library/propose`: the configured GPT model (structured output `LlmsTxtDraft`) picks `doc_id`, title and description, then rewrites the whole list. It may refine overlapping descriptions and must explain its changes. Code then repairs the draft: it sanitizes and de-duplicates the ID, restores dropped documents and original titles/categories, drops unknown IDs, and builds every path itself. It returns a diff plus token cost. Nothing is written yet.
  2. `POST /api/library/commit`: the operator may edit the text first. Code validates the format, paths, duplicates and the presence of the new doc, and checks that llms.txt didn't change since the proposal. It then backs up the old llms.txt to `data/.history/`, writes the files and clears the cache.
- `DELETE /api/library/{doc_id}` removes the entry in code (no LLM) and moves the file to `data/.history/deleted/`.
- Live test on a scratch copy: a company checklist was registered in 7.8 s for $0.00068. The model also rewrote the software-service description to separate "contract clauses" from "internal procedure".
- Both servers bind 0.0.0.0. There is **no authentication**, so restrict the ports to the demo network.

## 10. Multi-agent mode — implemented 2026-10-18 (`backend/app/agent/multi_agent.py`)

Third mode in the sidebar: **👥 멀티 에이전트** (`ChatRequest.mode = "multi"`). It is built only on `create_agent` (no extra packages), using the **agent-as-tool supervisor** pattern:

- 🧑‍💼 **Supervisor** (`create_agent`). Tools: `ask_law`, `ask_precedent`, `ask_contract`, `ask_company` (args `reason`, `task`) plus `write_todos`. It never reads sources directly. Middleware: Observer (`agent="supervisor"`), VerifyMiddleware (in-agent verification of the final answer), TodoList, and ModelCallLimit 12.
- **Specialists**, each its own `create_agent` with its own prompt (`SPECIALIST_SYSTEM` + `SPECIALIST_ROLES`), tool subset, `ModelCallLimit 6` and a 5-tool limit:

| key | agent | tools |
|---|---|---|
| `law` | 📜 법령조사 | search_laws, get_law_article, web_search |
| `precedent` | ⚖️ 판례조사 | search_precedents, read_precedent (web_search if no law key) |
| `contract` | 📑 계약서분석 | read_uploaded_contract, read_reference_outline/sections (standard contracts) |
| `company` | 🏢 회사기준 | read_reference_outline/sections (company docs; reports "none" if no company doc) |

- **Parallelism:** independent `ask_*` calls in one supervisor turn run concurrently. Each specialist runs in a fresh `contextvars.Context`, so its tokens don't leak into the supervisor's stream.
- **Shared source registry:** specialists' `[U*]/[C*]/[K*]/[L*]/[P*]` are cited by the supervisor. Verification inside the agent and the guardrail outside it work unchanged.
- **Events:** every event is tagged `agent` (who) and `run` (which `ask_*` call). New events: `team`, `handoff{run,to,task,reason,turn}`, `agent_start`, `agent_report{agent,run,report,source_ids,model_calls,tool_calls,elapsed_ms}`, `handoff_done`. `llm_call` carries `agent`, and `final.stats.by_agent` holds per-agent calls, tokens and cost.
- **UI:** 👥 team panel (live status per agent). For each supervisor turn that delegates, one **lane per specialist side by side** shows the instruction, the specialist's thinking and tools, and its 📋 report. The flow diagram puts the supervisor at the centre with ⇄ edges to the 4 specialists. There is a per-agent usage table, and the downloadable report gets a multi-agent section.
- **Live run** (📂 file trigger, freelancer contract, test copy of `data/` with the company doc; `DATA_DIR` env override added for this): 51.7 s, $0.0038. The supervisor wrote todos and delegated **contract + company in parallel**: company read the 사내 기준 (K1–K7), contract compared the attachment with the standard contract (U1–U12, C1–C10). It chose **not** to call law/precedent. Per agent: supervisor $0.0021, contract $0.0012, company $0.0006.
- Tests: `tests/test_multi_agent.py`. A role-routing fake model checks parallel delegation, agent/run tags, per-agent stats, and the company agent citing `[K1]`.

## 10-old. Multi-agent extension (original design)

### 10.1 Target architecture: supervisor + specialists

```
                         ┌───────────────────────────────┐
  질문 + 첨부 ─────────▶ │ 🧑‍💼 오케스트레이터 (팀장)          │  analyze → plan → delegate → merge → write → verify
                         └──────┬───────┬───────┬────────┘
                    지시문 ↓     │       │       │     ↑ 보고서 + 근거 ID
          ┌─────────────────────┘       │       └──────────────────────┐
          ▼                             ▼                              ▼
 📜 법령조사 에이전트             ⚖️ 판례조사 에이전트              🏢 사내 지식베이스 에이전트
 search_laws, get_law_article    search_precedents, read_precedent  list_kb, read_kb_doc
 (+ web_search, gov domains)                                         (data/company_kb/llms.txt)
          └──────────────── shared SourceRegistry (L#, P#, K#, …) ─────────────────┘
 (+ 📑 계약서 분석 에이전트: read_uploaded_contract, standard contracts — today's contract tools)
```

- **Pattern: agents as tools.** The supervisor's tools are `ask_law_agent(task)`, `ask_precedent_agent(task)`, `ask_kb_agent(task)`, `ask_contract_agent(task)`. Each specialist is its own small LangGraph ReAct subgraph with its own system prompt, tool subset, tool budget and context window. It returns a short report whose claims cite registry IDs.
- **Parallelism:** the supervisor issues independent delegations in one turn (parallel tool calls, or LangGraph `Send` fan-out). The audience sees three agents working at once.
- **Why this pattern for a demo:** it maps to an org chart (팀장 + 전문가), it's deterministic enough to rehearse, and it reuses today's nodes. The alternative, handoff/swarm (agents pass control to each other), is harder to follow on screen.
- **Company knowledge base (mock):** `data/company_kb/llms.txt` plus Markdown docs such as 계약 검토 체크리스트, 위임전결 규정, 표준 특약 문구집 and 과거 분쟁 사례. It uses the same llms.txt approach as the standard contracts, with a new source kind `K#` (🏢 사내 기준).

### 10.2 What changes in the code

| Area | Change |
|---|---|
| `RunContext` | `ctx.child(agent_name, budget)` returns a child context that shares the queue, registry and pricing but tags events and tracks its own usage/cost |
| Events | every event gets `agent` (+ `parent_agent`, `task_id`); new `agent_start{agent,task}`, `handoff{from,to,instruction}`, `agent_report{agent,summary,source_ids,usage,cost}`, `agent_end` |
| Tools | `ToolSpec.owner` assigns each tool to a specialist; the supervisor only gets `ask_*` tools |
| Graph | factor `research`/`tools` into `build_specialist(prompt, tools, budget)`; supervisor graph = analyze → plan → delegate ⇄ specialists → write → verify |
| Usage | per-agent token/cost breakdown (the multi-agent cost trade-off is a teaching point) |
| UI | sidebar toggle **단일 에이전트 / 멀티 에이전트**; in the research stage, one nested `st.status` per specialist laid out in `st.columns` so parallel work is visible; handoff bubbles ("팀장 → 판례조사 에이전트: …"); flow diagram with the supervisor in the centre; cost table grouped by agent |
| Demo | run the same question in both modes and compare time, cost and answer quality side by side |

### 10.3 Expectations to state on stage

- Multi-agent typically uses **more tokens** (each specialist re-reads its own instructions and the task), often 1.5–3× the single-agent run. Parallelism can still make it **faster** in wall-clock time.
- Benefits: specialization (focused prompts and tools), context isolation (a precedent agent isn't distracted by contract text), parallel work, and clear ownership of evidence.
- Risks: coordination overhead, duplicated searches, and a supervisor that over-delegates. Mitigate with per-agent budgets and a "delegate only what the plan needs" rule.

Estimated effort: about 1.5–2 days, including the mock company KB and UI lanes.
