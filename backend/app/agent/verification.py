"""답변 검증 — 에이전트 안의 '근거 검증'과 에이전트 밖의 '최종 안전장치'.

1. 근거 검증 (에이전트 안): 문제를 찾으면 에이전트에게 피드백을 돌려주고 답변을 다시 쓰게 한다.
   - 교과서형 루프: verify 노드 → 문제 있으면 write 노드로 되돌아감
   - 실전형 ReAct: VerifyMiddleware.after_model → 문제 있으면 모델로 되돌아감(jump_to)
2. 최종 안전장치 (에이전트 밖): 에이전트가 끝난 뒤에도 남은 가짜 인용을 코드로 지운다.
   에이전트를 믿지 않고 무조건 지켜야 할 규칙을 강제하는 마지막 필터다. 다시 고치게 하지는 않는다.
"""

from __future__ import annotations

from .context import RunContext
from .sources import check_citations, normalize_citations, strip_unknown

MAX_VERIFY_RETRIES = 1  # 근거 검증 실패 시 다시 쓰게 하는 횟수


def check_grounding(ctx: RunContext, answer: str) -> dict:
    """답변의 인용이 실제로 읽은 자료(출처 레지스트리)와 맞는지 코드로 확인한다."""
    answer = normalize_citations(answer)
    result = check_citations(answer, ctx.registry)
    problems = []
    if result["unknown"]:
        problems.append(f"자료 목록에 없는 근거 ID {', '.join(result['unknown'])} 를 인용했습니다. "
                        "실제로 읽은 자료의 ID 로 바꾸거나, 근거가 없으면 그 내용을 '일반적으로'라고 밝히고 인용을 빼세요.")
    if len(ctx.registry) and not result["cited"]:
        problems.append("답변에 근거 인용이 하나도 없습니다. 사실이나 법적 판단을 말할 때마다 근거 ID 를 붙이세요.")
    return result | {"answer": answer, "problems": problems}


def check_research(ctx: RunContext) -> list[str]:
    """실전형 ReAct 전용: 조사 과정의 빈틈. (교과서형은 '목표 달성 판단'이 이 역할을 한다)"""
    problems = []
    if ctx.precedent_searches and not any(s.kind == "precedent" for s in ctx.registry.all()):
        problems.append("판례를 검색했지만 하나도 읽지 않았습니다. 관련 판례를 read_precedent 로 확인하거나, "
                        "확인하지 않은 이유를 답변에 밝히세요.")
    return problems


def feedback_message(problems: list[str]) -> str:
    return ("[근거 검증 결과] 이 답변은 그대로 내보낼 수 없습니다. 아래를 고친 전체 답변을 다시 작성하세요.\n"
            + "\n".join(f"- {p}" for p in problems))


def guardrail(ctx: RunContext, answer: str) -> tuple[str, list[str]]:
    """최종 안전장치 (에이전트 밖): 남은 가짜 인용을 지운다. (정리된 답변, 인용 ID 목록)"""
    answer = normalize_citations(answer)
    result = check_citations(answer, ctx.registry)
    if len(ctx.registry) or result["unknown"] or result["cited"]:
        ctx.stage("guardrail", "start")
        ctx.emit("guardrail", removed=result["unknown"], cited=result["cited"])
        ctx.stage("guardrail", "end")
    return strip_unknown(answer, result["unknown"]), result["cited"]
