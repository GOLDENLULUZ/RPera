from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.models import AiPreset
from rpera.providers import BoundModelClient, ConfiguredModelClient, FallbackModelClient
from rpera.thinking import ThinkingLevel
from tests.helpers import make_data_dir


def preset_data(**changes: object) -> dict[str, Any]:
    return {
        "name": "DeepSeek", "provider": "deepseek",
        "base_url": "https://provider.example/v1", "api_key": "test-key",
        "model": "deepseek-flash", **changes,
    }


def test_deepseek_preset_levels_save_copy_reload_and_freeze(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        initial = client.get("/api/ai-presets").json()
        preset_id = initial["main_preset_id"]
        assert initial["presets"][0]["model"] == "deepseek-flash"
        assert initial["presets"][0]["thinking_level"] is None
        assert initial["deepseek_thinking_levels"] == {
            model: ["none", "low", "high", "max"] for model in (
                "deepseek-flash", "deepseek-v4-pro", "deepseek-v4-flash", "deepseek-v4-flash-vision-exp",
            )
        }
        for model in initial["deepseek_thinking_levels"]:
            for level in ("none", "low", "high", "max"):
                response = client.put(f"/api/ai-presets/{preset_id}", json=preset_data(model=model, thinking_level=level))
                assert response.status_code == 200
                assert response.json()["presets"][0]["thinking_level"] == level
        copied = client.post(f"/api/ai-presets/{preset_id}/duplicate", json={"name": "备用"})
        assert copied.status_code == 200
        fallback_id = copied.json()["presets"][1]["id"]
        assert client.put("/api/ai-fallback-settings", json={"enabled": True, "preset_id": fallback_id}).status_code == 200
        bound = ConfiguredModelClient(app.state.presets).bind_for_agents(("coordinator",))["coordinator"]
        assert isinstance(bound, FallbackModelClient)
        assert bound.primary.metadata()["thinking_level"] == "max"
        assert bound.fallback.metadata()["thinking_level"] == "max"

        changed = client.put(f"/api/ai-presets/{preset_id}", json=preset_data(thinking_level="none"))
        assert changed.status_code == 200
        assert bound.primary.metadata()["thinking_level"] == "max"
        assert app.state.presets.get(preset_id).thinking_level == "none"
        stored = json.loads((data_dir / "config" / "ai_presets.json").read_text(encoding="utf-8"))
        assert [item["thinking_level"] for item in stored["presets"]] == ["none", "max"]
        assert "deepseek_thinking_levels" not in stored
        assert all("api_key" not in item for item in changed.json()["presets"])
    with TestClient(create_app(data_dir=data_dir)) as client:
        reloaded = client.get("/api/ai-presets").json()
        assert [item["thinking_level"] for item in reloaded["presets"]] == ["none", "max"]
        assert reloaded["deepseek_thinking_levels"] == initial["deepseek_thinking_levels"]


def test_invalid_levels_are_rejected_without_overwriting_the_preset(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    with TestClient(create_app(data_dir=data_dir)) as client:
        preset_id = client.get("/api/ai-presets").json()["main_preset_id"]
        assert client.put(f"/api/ai-presets/{preset_id}", json=preset_data(thinking_level="high")).status_code == 200
        config_path = data_dir / "config" / "ai_presets.json"
        saved = config_path.read_bytes()
        for changes in (
            {"thinking_level": "medium"}, {"thinking_level": "minimal"}, {"thinking_level": "ultra"},
            {"thinking_level": "xhigh"}, {"model": "deepseek-chat", "thinking_level": "high"},
            {"model": "deepseek-reasoner", "thinking_level": "high"},
            {"model": "deepseek-unknown", "thinking_level": "max"},
            {"provider": "openai_compatible", "thinking_level": "high"},
            {"provider": "google_gemini", "model": "gemini-3.8-flash", "thinking_level": "max"},
            {"provider": "google_gemini", "model": "gemini-3.8-flash", "thinking_level": "none"},
        ):
            assert client.put(f"/api/ai-presets/{preset_id}", json=preset_data(**changes)).status_code == 422
            assert config_path.read_bytes() == saved
        for model in ("deepseek-chat", "deepseek-reasoner", "deepseek-unknown"):
            assert client.put(f"/api/ai-presets/{preset_id}", json=preset_data(model=model)).status_code == 200


@pytest.mark.parametrize("level", [None, "none", "low", "high", "max"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_deepseek_effort_reaches_requests_and_preserves_tool_reasoning(
    level: ThinkingLevel | None, streaming: bool,
) -> None:
    sent: list[dict[str, Any]] = []
    response_message: dict[str, Any] = {
        "content": "", "reasoning_content": "继续读取。",
        "tool_calls": [{"id": "read-next", "type": "function", "function": {"name": "file_read", "arguments": "{}"}}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        sent.append(json.loads(request.content))
        if streaming:
            delta = {**response_message, "tool_calls": [{"index": 0, **response_message["tool_calls"][0]}]}
            chunk = {"choices": [{"delta": delta}], "usage": {"total_tokens": 10}}
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                 text=f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"choices": [{"message": response_message}], "usage": {"total_tokens": 10}})

    client = BoundModelClient(AiPreset(id="ds", **preset_data(thinking_level=level)),
                              transport=httpx.MockTransport(handler), streaming=streaming)
    tools = [{"type": "function", "function": {"name": "file_read", "parameters": {"type": "object", "properties": {}}}}]
    result = await client.complete("system", [
        {"role": "user", "content": "继续"},
        {"role": "assistant", "content": None, "_reasoning_content": "先读取。", "tool_calls": [
            {"id": "read-previous", "type": "function", "function": {"name": "file_read", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "read-previous", "content": "{}"},
    ], tools)
    payload = sent[0]
    assert payload["stream"] is streaming
    if level is None:
        assert "reasoning_effort" not in payload
        assert "thinking_level" not in client.metadata()
    else:
        assert payload["reasoning_effort"] == level
        assert client.metadata()["thinking_level"] == level
    assert "thinking" not in payload
    assert "thinking_level" not in payload
    assert payload["messages"][2]["reasoning_content"] == "先读取。"
    assert payload["tools"] == tools
    assert result.reasoning == "继续读取。"
    assert result.tool_calls[0].id == "read-next"
    assert result.tool_calls[0].name == "file_read"
    assert result.usage == {"total_tokens": 10}


async def test_automatic_legacy_and_other_provider_do_not_send_effort() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    for provider, model in (("deepseek", "deepseek-chat"), ("deepseek", "deepseek-reasoner"), ("openai_compatible", "deepseek-flash")):
        client = BoundModelClient(AiPreset(id="auto", **preset_data(provider=provider, model=model)),
                                  transport=httpx.MockTransport(handler))
        await client.complete("system", [{"role": "user", "content": "请求"}])
    assert all("reasoning_effort" not in payload and "thinking" not in payload for payload in sent)


async def test_connection_test_retains_effort_and_gives_reasoning_room() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})

    for level, limit, expected in ((None, 1000, 256), ("max", 1000, 256), ("none", 1000, 8), ("low", 100, 100)):
        preset = AiPreset(id="connection", **preset_data(thinking_level=level, max_tokens=limit))
        result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).test_connection()
        assert result.content == "OK"
        assert sent[-1]["max_tokens"] == expected
        assert sent[-1].get("reasoning_effort") == level
        assert preset.max_tokens == limit
