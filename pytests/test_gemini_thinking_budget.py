"""Gemini 客户端思考预算裁剪逻辑测试。

覆盖：Gemma 等开放模型不支持思考预算时返回 None（省略该参数），
Gemini 系列保持原有自动/钳制行为。
"""

from typing import Any, Dict, Optional

from src.llm_models.model_client.gemini_client import (
    GeminiClient,
    THINKING_BUDGET_AUTO,
    THINKING_BUDGET_LIMITS,
)


def _clamp(extra_params: Optional[Dict[str, Any]], model_id: str) -> Optional[int]:
    return GeminiClient.clamp_thinking_budget(extra_params, model_id)


def test_gemma_model_omits_thinking_budget():
    """Gemma 等开放模型不支持思考预算，返回 None 以省略该参数。"""

    assert _clamp(None, "gemma-4-31b-it") is None
    assert _clamp({"thinking_budget": 100}, "gemma-4-31b-it") is None
    assert _clamp(None, "gemma-3-27b-it") is None


def test_gemini_model_keeps_auto_budget():
    """Gemini 模型未显式配置时保持自动模式（-1）。"""

    assert _clamp(None, "gemini-2.5-flash") == THINKING_BUDGET_AUTO
    assert _clamp(None, "gemini-2.0-flash") == THINKING_BUDGET_AUTO


def test_gemini_model_clamps_budget_to_limits():
    """Gemini 模型的思考预算被钳制到允许范围内。"""

    limits = THINKING_BUDGET_LIMITS["gemini-2.5-pro"]
    assert _clamp({"thinking_budget": 1}, "gemini-2.5-pro") == int(limits["min"])
    assert _clamp({"thinking_budget": 10**9}, "gemini-2.5-pro") == int(limits["max"])
    mid_value = (int(limits["min"]) + int(limits["max"])) // 2
    assert _clamp({"thinking_budget": mid_value}, "gemini-2.5-pro") == mid_value


def test_supports_thinking_budget_classification():
    """支持性判定：Gemini 系列返回 True，Gemma 等返回 False。"""

    assert GeminiClient._model_supports_thinking_budget("gemini-2.5-flash") is True
    assert GeminiClient._model_supports_thinking_budget("gemini-2.5-flash-lite") is True
    assert GeminiClient._model_supports_thinking_budget("gemma-4-31b-it") is False
    assert GeminiClient._model_supports_thinking_budget("embed-001") is False
