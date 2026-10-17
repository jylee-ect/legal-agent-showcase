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

## 3. Agent graph

```
START → analyze ─┬─ direct ──→ direct_answer → END          (off-topic / chit-chat)
                 └─ research → plan → research ⇄ tools
                                         └──→ write → verify → END
```

| Node | What it does | What the audience sees |
|---|---|---|
| `analyze` | Sees the question, history, `llms.txt`, and **only the outline** of the attachment. Decides the question type, the issues, whether to read the attachment (with a reason), candidate standard contracts, and the route | "📥 에이전트가 받은 정보", the verdict, "💭 판단 근거", "🔀 분기 결정" |
| `plan` | 2–6 step plan within the tool budget | A checklist whose items turn ⏳ → ✅ / ⏭️ (skipped) live |
| `research` | LLM with tools bound decides the next action. Parallel tool calls are allowed. Stops when it has enough evidence and writes a research memo | 💭 thoughts, optional reasoning summary, 📝 memo |
| `tools` | Runs the tool calls concurrently, registers sources, and enforces the budget (`MAX_TOOL_CALLS`) | One card per call: label, args, **reason**, then 👀 observation + source badges |
| `write` | Streams the answer, citing registry IDs only | Answer typing out, with colored `[C1]` badges |
| `verify` | **Code-based** check: removes unknown IDs, warns if nothing is cited, lists collected-but-unused sources | "인용 N개 모두 실제 자료와 일치" |
| `direct_answer` | Short answer without tools | Shows the agent choosing *not* to use tools |

**Irrelevant questions:** if a contract is attached but the question has nothing to do with it (e.g. "점심 메뉴 추천"), `analyze` routes to `direct_answer`. If the question is legal but unrelated to the attachment, it sets `use_attachment=false`, and the plan won't read the attachment.

### 3.1 Tools

Every tool's first two arguments are `reason` (one plain-Korean sentence) and `plan_step`. This is how the "why" becomes visible without exposing hidden chain-of-thought.

| Tool | Label | Source ID |
|---|---|---|
| `read_uploaded_contract(sections)` | 📎 첨부 계약서 읽기 | `U#` |
| `read_standard_contract_outline(contract_id)` | 📚 표준계약서 목차 보기 | — |
| `read_standard_contract_articles(contract_id, articles)` | 📄 표준계약서 조항 읽기 | `C#` |
| `search_laws(query)` | 🏛️ 법령 찾기 | — |
| `get_law_article(law_name, article)` | 📜 법조문 원문 확인 (`lawService.do?target=law&JO=`) | `L#` |
| `search_precedents(query)` | ⚖️ 판례 검색 (body-text search) | — (a list alone can't be cited) |
| `read_precedent(precedent_id)` | 📖 판례 읽기 (판시사항/판결요지/참조조문) | `P#` |
| `web_search(query)` | 🌐 웹 검색 (OpenAI `web_search`, KR location, `url_citation`s) | `W#` |

Law tools are only offered when a `law` key exists. Public law.go.kr URLs are rebuilt, so the OC key never reaches the UI.

### 3.2 Sources and citations

- Each tool result is registered as `Source{id, kind, title, excerpt, url, content}`. Reading the same clause again reuses its ID.
- The writer receives every collected source's content and may only cite those IDs.
- The answer is checked against the registry. The 출처 cards and the disclaimer are built **by the UI from the registry**, so the model can't invent a URL.

---

## 4. Event stream (NDJSON, one JSON per line)

Every event carries `t` (ms since the run started) and `ts` (wall-clock ISO time with ms and timezone).

`run_start{…,pricing}` · `stage{stage,status}` · `context{items}` · `analysis{data}` · `route{path,label}` · `plan{strategy,steps}` · `plan_update{step_no,status}` · `loop{n}` · `llm_call{call_no,stage,purpose,model,usage,web_search_calls,cost_usd,elapsed_ms,total_usage,total_cost_usd}` · `reasoning{stage,text}` · `thought{text}` · `tool_call{id,tool,label,args,reason,plan_step}` · `tool_result{id,ok,summary,source_ids,preview,elapsed_ms}` · `research_note{text}` · `token{text}` · `verify{ok,cited,unknown,unused,problems}` · `sources{items}` · `final{answer_md,cited_ids,stats}` · `error{message}` · `done`

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
| 에이전트 루프 (agent loop / step; called a "turn" in some agent SDKs) | one "decide next action" LLM call + the tools it triggers + observing results | 4 |
| LLM 호출 | every model API call: analyze, plan, each loop, write, plus hidden web-search calls | 8 = 1+1+4+1+1 |
| 도구 호출 | individual tool executions; one loop can issue several in parallel | 8 |

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

## 10. Multi-agent extension (design)

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
