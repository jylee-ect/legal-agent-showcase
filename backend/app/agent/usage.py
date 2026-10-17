"""토큰 사용량과 비용 계산.

OpenAI Responses API 의 usage 구성:
- input_tokens 에는 캐시에서 읽은 토큰(cached_tokens)과 캐시에 쓴 토큰(cache_write_tokens)이 포함된다.
- output_tokens 에는 추론 토큰(reasoning_tokens)이 포함된다. 추론 토큰도 출력 단가로 과금된다.
LangChain 은 이를 usage_metadata 의 input_token_details.cache_read / cache_creation,
output_token_details.reasoning 으로 옮겨 준다.

단가는 프로젝트 루트의 pricing.json 에서 요청마다 읽는다.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from ..config import ROOT_DIR

PRICING_FILE = ROOT_DIR / "pricing.json"


@dataclass
class Usage:
    input: int = 0  # 전체 입력 (캐시 읽기·쓰기 포함)
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0  # 전체 출력 (추론 포함)
    reasoning: int = 0

    @property
    def uncached_input(self) -> int:
        return max(self.input - self.cache_read - self.cache_write, 0)

    @property
    def total(self) -> int:
        return self.input + self.output

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(a + b for a, b in zip(asdict(self).values(), asdict(other).values())))

    def to_dict(self) -> dict:
        return asdict(self) | {"uncached_input": self.uncached_input, "total": self.total}


def from_usage_metadata(um: dict | None) -> Usage:
    """LangChain AIMessage.usage_metadata → Usage"""
    if not um:
        return Usage()
    inp = um.get("input_token_details") or {}
    out = um.get("output_token_details") or {}
    return Usage(
        input=um.get("input_tokens") or 0,
        cache_read=inp.get("cache_read") or 0,
        cache_write=inp.get("cache_creation") or 0,
        output=um.get("output_tokens") or 0,
        reasoning=out.get("reasoning") or 0,
    )


def from_openai(usage) -> Usage:
    """openai SDK ResponseUsage → Usage"""
    if usage is None:
        return Usage()
    inp = getattr(usage, "input_tokens_details", None)
    out = getattr(usage, "output_tokens_details", None)
    return Usage(
        input=usage.input_tokens or 0,
        cache_read=getattr(inp, "cached_tokens", 0) or 0,
        cache_write=getattr(inp, "cache_write_tokens", 0) or 0,
        output=usage.output_tokens or 0,
        reasoning=getattr(out, "reasoning_tokens", 0) or 0,
    )


def load_pricing() -> dict:
    try:
        return json.loads(PRICING_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"models": {}}


def model_rates(pricing: dict, model: str) -> dict | None:
    """'gpt-6-luna-2026-09-01' 처럼 날짜가 붙은 이름도 가장 긴 접두어로 찾는다."""
    models = pricing.get("models", {})
    matches = [name for name in models if model == name or model.startswith(name + "-")]
    return models[max(matches, key=len)] if matches else None


def cost_usd(pricing: dict, model: str, usage: Usage, web_search_calls: int = 0) -> float | None:
    """단가를 모르는 모델이면 None."""
    rates = model_rates(pricing, model)
    if rates is None:
        return None
    long_ctx = usage.input > pricing.get("long_context_threshold", 272_000)
    r = rates["long" if long_ctx and "long" in rates else "short"]
    tokens = (usage.uncached_input * r["input"] + usage.cache_read * r["cached_input"]
              + usage.cache_write * r.get("cache_write", r["input"]) + usage.output * r["output"]) / 1_000_000
    return tokens + web_search_calls * pricing.get("web_search_per_1k_calls", 0) / 1000


def pricing_summary(pricing: dict, model: str) -> dict:
    """화면에 '단가 기준'으로 보여줄 정보."""
    rates = model_rates(pricing, model)
    return {
        "model": model,
        "rates": rates["short"] if rates else None,
        "web_search_per_1k_calls": pricing.get("web_search_per_1k_calls"),
        "usd_krw": pricing.get("usd_krw"),
        "source": pricing.get("source"),
        "checked_at": pricing.get("checked_at"),
    }
