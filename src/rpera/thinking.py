from __future__ import annotations

from typing import Literal

from .gemini_thinking import GeminiThinkingLevel, validate_gemini_thinking_level


DeepSeekThinkingLevel = Literal["none", "low", "high", "max"]
ThinkingLevel = GeminiThinkingLevel | DeepSeekThinkingLevel

_DEEPSEEK_LEVELS: dict[str, tuple[DeepSeekThinkingLevel, ...]] = {
    "deepseek-flash": ("none", "low", "high", "max"),
    "deepseek-v4-pro": ("none", "low", "high", "max"),
    "deepseek-v4-flash": ("none", "low", "high", "max"),
    "deepseek-v4-flash-vision-exp": ("none", "low", "high", "max"),
}


def supported_deepseek_thinking_levels() -> dict[str, list[DeepSeekThinkingLevel]]:
    return {model: list(levels) for model, levels in _DEEPSEEK_LEVELS.items()}


def validate_thinking_level(provider: str, model: str, level: ThinkingLevel | None) -> None:
    if level is None:
        return
    if provider == "google_gemini":
        validate_gemini_thinking_level(provider, model, level)
        return
    if provider != "deepseek":
        raise ValueError("思考强度只能用于 Google Gemini 或 DeepSeek Preset")
    if level not in _DEEPSEEK_LEVELS.get(model, ()):
        raise ValueError(f"模型 {model} 不支持思考强度 {level}；请选择自动或该模型支持的档位")
