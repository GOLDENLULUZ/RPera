from __future__ import annotations

from typing import Literal


GeminiThinkingLevel = Literal["minimal", "low", "medium", "high"]

_SUPPORTED_LEVELS: dict[str, tuple[GeminiThinkingLevel, ...]] = {
    "gemini-3-flash-preview": ("minimal", "low", "medium", "high"),
    "gemini-3.1-pro-preview": ("low", "medium", "high"),
    "gemini-3.1-flash-lite": ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.5-flash-lite": ("minimal", "low", "medium", "high"),
    "gemini-3.6-flash": ("minimal", "low", "medium", "high"),
    "gemini-3.7-flash": ("low", "medium", "high"),
    "gemini-3.8-flash": ("low", "medium", "high"),
}


def supported_gemini_thinking_levels() -> dict[str, list[GeminiThinkingLevel]]:
    return {model: list(levels) for model, levels in _SUPPORTED_LEVELS.items()}


def validate_gemini_thinking_level(provider: str, model: str, level: str | None) -> None:
    if level is None:
        return
    if provider != "google_gemini":
        raise ValueError("思考强度只能用于 Google Gemini Preset")
    supported = _SUPPORTED_LEVELS.get(model.removeprefix("models/"), ())
    if level not in supported:
        raise ValueError(f"模型 {model} 不支持思考强度 {level}；请选择自动或该模型支持的档位")
