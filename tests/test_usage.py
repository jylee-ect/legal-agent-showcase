from types import SimpleNamespace

import pytest

from app.agent.usage import Usage, cost_usd, from_openai, from_usage_metadata, load_pricing, model_rates

PRICING = {
    "long_context_threshold": 272_000,
    "web_search_per_1k_calls": 10.0,
    "models": {"gpt-6-luna": {"short": {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50},
                              "long": {"input": 0.20, "cached_input": 0.02, "cache_write": 0.25, "output": 0.75}}},
}


def test_langchain_usage_metadata_mapping():
    u = from_usage_metadata({"input_tokens": 1000, "output_tokens": 300, "total_tokens": 1300,
                             "input_token_details": {"cache_read": 600, "cache_creation": 100},
                             "output_token_details": {"reasoning": 200}})
    assert (u.input, u.cache_read, u.cache_write, u.output, u.reasoning) == (1000, 600, 100, 300, 200)
    assert u.uncached_input == 300 and u.total == 1300


def test_openai_sdk_usage_mapping():
    usage = SimpleNamespace(input_tokens=50, output_tokens=20,
                            input_tokens_details=SimpleNamespace(cached_tokens=10, cache_write_tokens=5),
                            output_tokens_details=SimpleNamespace(reasoning_tokens=7))
    assert from_openai(usage) == Usage(50, 10, 5, 20, 7)


def test_cost_splits_cached_and_cache_write_and_adds_web_search_fee():
    u = Usage(input=100_000, cache_read=40_000, cache_write=10_000, output=20_000, reasoning=5_000)
    # 미캐시 5만×0.10 + 캐시읽기 4만×0.01 + 캐시쓰기 1만×0.125 + 출력 2만×0.50  (USD per 1M)
    expected = (50_000 * 0.10 + 40_000 * 0.01 + 10_000 * 0.125 + 20_000 * 0.50) / 1e6
    assert cost_usd(PRICING, "gpt-6-luna-2026-09-01", u) == pytest.approx(expected)  # 날짜 붙은 이름도 매칭
    assert cost_usd(PRICING, "gpt-6-luna", u, web_search_calls=3) == pytest.approx(expected + 3 * 10.0 / 1000)


def test_long_context_rate_and_unknown_model():
    u = Usage(input=300_000, output=1000)
    assert cost_usd(PRICING, "gpt-6-luna", u) == pytest.approx(300_000 * 0.20 / 1e6 + 1000 * 0.75 / 1e6)
    assert cost_usd(PRICING, "some-other-model", u) is None


def test_project_pricing_file_has_configured_models():
    pricing = load_pricing()
    assert model_rates(pricing, "gpt-6-luna") and model_rates(pricing, "gpt-6.1-sol")
