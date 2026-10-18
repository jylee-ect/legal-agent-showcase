# ⚖️ 계약서 검토 AI 에이전트 (시연용)

AI 에이전트가 **질문 분석 → 계획 → 도구 사용 → 답변 → 근거 검증**을 어떻게 수행하는지 실시간으로 보여주는 시연 서비스입니다.

## 빠른 시작

준비물: **Python 3.11 이상** (3.12 권장). Windows 는 [python.org](https://www.python.org/downloads/) 설치 화면에서 'Add python.exe to PATH' 를 체크하세요.

### Windows (PowerShell)

**① 설치 (처음 한 번만)**

```powershell
cd C:\경로\agent_showcase
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

> 가상환경을 activate 하지 않고 `.venv` 안의 python 을 직접 부르므로, PowerShell 실행 정책을 바꿀 필요가 없습니다.

**② GPT 키를 환경변수 `GPT_API` 에 설정 (처음 한 번만)**

```powershell
setx GPT_API "sk-여기에_OpenAI_API_키"
```

> `setx` 는 이 PC 사용자 계정에 저장합니다. **저장한 뒤 PowerShell 창을 새로 열어야** 적용됩니다.
> 지금 창에서만 쓰려면 `$env:GPT_API = "sk-..."`

**③ 국가법령정보센터 OC 입력 (선택, 없으면 법령·판례 대신 웹 검색만 사용)**

```powershell
copy api_keys.example.txt api_keys.txt
notepad api_keys.txt
```

`law:여기에_국가법령정보센터_OC_입력` 줄의 값을 바꾸고 저장합니다.

**④ 실행: PowerShell 창 2개**

창 1 — 백엔드 (에이전트 API, 포트 7000)

```powershell
cd C:\경로\agent_showcase\backend
..\.venv\Scripts\python -m uvicorn app.main:app --host 0.0.0.0 --port 7000
```

창 2 — 화면 (Streamlit, 포트 3000)

```powershell
cd C:\경로\agent_showcase\frontend
..\.venv\Scripts\python -m streamlit run app.py
```

브라우저에서 **http://localhost:3000** 을 엽니다. 종료는 각 창에서 `Ctrl+C`.

### macOS / Linux (터미널)

```bash
# ① 설치 (처음 한 번만)
cd /경로/agent_showcase
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# ② GPT 키 (영구 적용하려면 이 줄을 ~/.zshrc 또는 ~/.bashrc 에 추가)
export GPT_API="sk-여기에_OpenAI_API_키"

# ③ 국가법령정보센터 OC (선택)
cp api_keys.example.txt api_keys.txt     # 편집기로 law: 값 입력

# ④ 실행 — 터미널 1: 백엔드 (② 의 GPT_API 가 설정된 터미널이어야 함)
cd backend
../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 7000

# ④ 실행 — 터미널 2: 화면
cd /경로/agent_showcase/frontend
../.venv/bin/python -m streamlit run app.py
```

### 확인

- http://localhost:7000/health 에서 `"openai_key": true` 이면 GPT 키가 적용된 것입니다.
- 화면 왼쪽 사이드바에 `OpenAI 키 ✅ · 법령 API ✅` 가 보이면 준비 완료입니다.

### 자주 막히는 곳

| 증상 | 해결 |
|---|---|
| `python` 을 찾을 수 없음 | Python 설치 시 PATH 체크. 또는 `py -3.12 -m venv .venv` |
| 사이드바에 `OpenAI 키 ❌` | 백엔드 창에 `GPT_API` 가 없습니다. `setx` 후 **새 창**에서 백엔드를 다시 실행하세요 |
| 화면이 3000 이 아닌 8501 포트로 뜸 | `frontend` 폴더 안에서 실행해야 `frontend/.streamlit/config.toml`(0.0.0.0:3000)이 적용됩니다 |
| 포트가 이미 사용 중 | 백엔드 `--port 7001`, 화면 `--server.port 3001` 로 바꾸고, 화면 창에서 `$env:BACKEND_URL = "http://127.0.0.1:7001"` 설정 후 실행 |
| 다른 PC 에서 접속 안 됨 | Windows 방화벽에서 3000(필요하면 7000) 포트 허용 |

> 다른 PC 로 옮길 때는 `.venv` 폴더와 `api_keys.txt` 를 빼고 복사한 뒤, 위 순서를 그대로 따라 하세요.

## 구성

- 백엔드: FastAPI + LangGraph + OpenAI (`0.0.0.0:7000`, API 문서 http://localhost:7000/docs)
- 화면: Streamlit (`0.0.0.0:3000`) — 페이지 2개: ⚖️ 에이전트 시연, 📚 문서 관리
- 자료: `data/llms.txt` + 표준계약서(`data/contracts/`)·회사 참고 문서(`data/company/`) (DB 없음), 국가법령정보센터 Open API, OpenAI 내장 웹검색

> 두 서버 모두 0.0.0.0 에서 실행되므로 같은 네트워크의 다른 PC 에서 `http://<이 PC IP>:3000` 으로 접속할 수 있습니다.
> **인증 기능이 없으므로** 인터넷에 노출되지 않게 방화벽으로 시연용 네트워크만 허용하세요.

## 설정

| 항목 | 위치 | 설명 |
|---|---|---|
| `GPT_API` | 환경변수 (**필수**) | OpenAI API 키. 파일에 저장하지 않습니다 |
| `law:` | `api_keys.txt` (또는 환경변수 `LAW_API_OC`) | 국가법령정보센터 OC. 파일은 요청마다 다시 읽으므로 고친 뒤 재시작 불필요 |
| `OPENAI_MODEL` | 환경변수 (기본 `gpt-6-luna`) | 에이전트 모델 (운영 시 `gpt-6.1-sol`) |
| `WEB_SEARCH_MODEL` | 환경변수 (기본 `OPENAI_MODEL`) | 웹검색 도구가 쓰는 모델 |
| `REASONING_EFFORT` | 환경변수 (기본 `low`) | 추론 강도. 빈 값이면 reasoning 옵션을 보내지 않음 (추론 모델이 아닐 때) |
| `MAX_TOOL_CALLS` | 환경변수 (기본 `15`) | 한 작업당 도구 호출 예산 |
| `BACKEND_URL` | 환경변수, 화면 창 (기본 `http://127.0.0.1:7000`) | 화면이 호출할 백엔드 주소 |

환경변수는 **서버를 띄운 창의 값**이 쓰이므로, 바꾼 뒤에는 그 서버를 다시 실행하세요.

## 에이전트 방식 (사이드바에서 전환)

| 방식 | 엔진 | 특징 |
|---|---|---|
| 🎓 교과서형 루프 | LangGraph `StateGraph` (직접 설계) | 목표·완료 조건 → 계획 → 실행·점검·계획 수정 → 목표 달성 판단이 **항상 단계로 보임** |
| ⚡ 실전형 ReAct | LangChain `create_agent` + middleware | 그래프는 모델 ⇄ 도구뿐. 계획(`write_todos`)·다음 행동·종료를 **모델이 스스로** 정함. 빠르고 저렴하지만 점검이 보장되지 않음 |
| 👥 멀티 에이전트 | `create_agent` × 5 (팀장 + 전문가 4) | 🧑‍💼 팀장이 📜 법령조사 · ⚖️ 판례조사 · 📑 계약서분석 · 🏢 회사기준 에이전트에게 일을 맡기고(독립적인 일은 **동시에**), 보고서를 모아 답변. 전문가 각자도 ReAct. 에이전트별 토큰·비용 표시 |

두 방식 모두 **근거 검증은 에이전트 안**에서 합니다. 답변의 인용을 코드로 확인하고, 문제가 있으면 피드백과 함께 되돌려 다시 쓰게 합니다 (교과서형: 검증 → 답변 작성, 실전형: `VerifyMiddleware` → 모델). 에이전트가 끝난 뒤에는 **🛡️ 최종 안전장치(에이전트 밖)**가 그래도 남은 가짜 인용을 지웁니다.

실전형 ReAct 의 중간 과정은 middleware 가 모델 호출·도구 호출을 감싸서 보여줍니다: 모델 호출마다 결정(도구 호출/최종 답변), 도구를 부르기 전에 쓴 글, 추론 요약, 도구 사용 이유와 결과, 할 일 목록, 토큰·비용.
🏢 회사기준 에이전트는 📚 문서 관리에서 **회사 참고 문서**를 등록해야 제 역할을 합니다 (예: `samples/회사_외주개발_계약검토_기준.md`).

같은 질문을 여러 방식으로 돌려 비교하면 "에이전트를 어떻게 설계하느냐"의 차이를 보여줄 수 있습니다.

## 문서 등록 (📚 문서 관리 페이지)

1. 분류(표준계약서 / 회사 참고 문서)를 고르고 .md/.txt 파일을 올립니다. 운영자 메모는 선택입니다.
2. **GPT로 분석하고 llms.txt 수정안 만들기**: 설정된 모델이 문서를 읽고 ID·제목·설명을 정한 뒤 llms.txt 전체를 다시 씁니다.
   겹치는 기존 문서가 있으면 구분되도록 그 설명을 다듬기도 합니다.
3. 변경 이유, diff, 코드 보정 내역, 토큰·비용을 확인하고(필요하면 직접 수정) **반영하기**를 누르면 저장됩니다.
   - 모델이 기존 문서를 빠뜨리거나 제목·분류를 바꾸거나 경로를 망가뜨리면 코드가 원래대로 되돌립니다.
   - 반영 전 llms.txt 는 `data/.history/` 에 백업되고, 삭제한 문서는 `data/.history/deleted/` 로 옮겨집니다.
4. 에이전트는 다음 질문부터 새 문서를 사용합니다. 회사 참고 문서의 출처는 `[K1]` 처럼 표시됩니다.

시연용 회사 문서 예시: `samples/회사_외주개발_계약검토_기준.md`

## 토큰 · 비용 단가 (`pricing.json`)

- 모델별 단가(USD / 1M 토큰: 입력, 캐시 읽기, 캐시 쓰기, 출력)와 웹검색 호출 단가, 원화 환율을 설정합니다.
- 2026-10-18 OpenAI 공식 가격표 기준입니다. 단가가 바뀌면 이 파일만 고치면 됩니다(서버 재시작 불필요).
- 파일에 없는 모델은 화면에 '단가 미설정'으로 표시됩니다.

## 시연 팁

- **예시 질문 버튼 5개**: 일반 질문 2개, 계약서 첨부 2개, 엉뚱한 질문 1개(조사 없이 바로 답변하는 분기를 보여줌).
- **상세도**: 간단(청중용) / 자세히(원문·추론 요약) / 개발자(도구 인자 JSON, 이벤트 로그).
- **발표 모드**: 단계 사이에 지연을 넣어 천천히 보여줍니다.
- **녹화 / 재생**: 리허설 때 '이번 실행 녹화하기'를 켜 두면 `recordings/`에 저장됩니다. 현장 네트워크가 불안하면 녹화 재생을 쓰세요.
- 샘플 계약서는 `samples/`에 있습니다. 채팅 입력창에서 직접 .md/.txt 파일을 첨부할 수도 있습니다.

## 테스트

```powershell
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python -m pytest tests -q
```

가짜 LLM 으로 그래프 전체 흐름(분석 → 계획 → 도구 → 작성 → 검증, 엉뚱한 질문 분기)을 검증하므로 API 키가 없어도 실행됩니다.
