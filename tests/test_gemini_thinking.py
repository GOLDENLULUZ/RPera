from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rpera.app import create_app
from rpera.gemini import request_payload
from rpera.gemini_thinking import supported_gemini_thinking_levels
from rpera.models import AiPreset, PresetWrite
from rpera.providers import BoundModelClient
from tests.helpers import make_data_dir


def preset_data(**changes: object) -> dict[str, object]:
    return {
        "name": "Gemini 3", "provider": "google_gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "test-key", "model": "gemini-3.8-flash",
        **changes,
    }


def test_gemini_3_thinking_levels_are_model_specific() -> None:
    supported = supported_gemini_thinking_levels()
    assert supported["gemini-3.8-flash"] == ["low", "medium", "high"]
    assert supported["gemini-3.7-flash"] == ["low", "medium", "high"]
    assert supported["gemini-3.6-flash"] == ["minimal", "low", "medium", "high"]
    assert supported["gemini-3.1-pro-preview"] == ["low", "medium", "high"]
    assert supported["gemini-3-flash-preview"] == ["minimal", "low", "medium", "high"]
    assert PresetWrite.model_validate(preset_data(thinking_level="low")).thinking_level == "low"
    assert PresetWrite.model_validate(preset_data(model="models/gemini-3.1-pro-preview", thinking_level="medium")).thinking_level == "medium"
    assert PresetWrite.model_validate(preset_data(model="gemini-3.6-flash", thinking_level="minimal")).thinking_level == "minimal"
    assert AiPreset(id="legacy", name="旧配置").thinking_level is None
    for changes in (
        {"model": "gemini-3.8-flash", "thinking_level": "minimal"},
        {"model": "gemini-3-pro-preview", "thinking_level": "high"},
        {"model": "gemini-2.5-flash", "thinking_level": "low"},
        {"model": "gemini-3-unknown-preview", "thinking_level": "high"},
        {"provider": "deepseek", "model": "deepseek-chat", "thinking_level": "high"},
    ):
        with pytest.raises(ValidationError, match="思考强度"):
            PresetWrite.model_validate(preset_data(**changes))
    with pytest.raises(ValidationError, match="思考强度"):
        AiPreset(id="bad", **preset_data(model="gemini-2.5-flash", thinking_level="high"))


def test_preset_thinking_level_saves_duplicates_and_loads_old_config(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        initial = client.get("/api/ai-presets").json()
        original_id = initial["main_preset_id"]
        assert initial["gemini_thinking_levels"] == supported_gemini_thinking_levels()
        assert initial["gemini_thinking_levels"]["gemini-3.8-flash"] == ["low", "medium", "high"]
        assert "gemini-3.8-flash-tts" not in initial["gemini_thinking_levels"]
        invalid = client.put(f"/api/ai-presets/{original_id}", json=preset_data(model="gemini-2.5-flash", thinking_level="low"))
        assert invalid.status_code == 422
        assert client.put(f"/api/ai-presets/{original_id}", json=preset_data(thinking_level="minimal")).status_code == 422
        result = client.put(f"/api/ai-presets/{original_id}", json=preset_data(thinking_level="medium"))
        assert result.status_code == 200
        assert result.json()["presets"][0]["thinking_level"] == "medium"
        assert result.json()["gemini_thinking_levels"] == initial["gemini_thinking_levels"]
        copied = client.post(f"/api/ai-presets/{original_id}/duplicate", json={"name": "副本"})
        assert copied.status_code == 200
        assert [item["thinking_level"] for item in copied.json()["presets"]] == ["medium", "medium"]
        assert app.state.presets.resolve(("coordinator",))["coordinator"].thinking_level == "medium"

        config_file = data_dir / "config" / "ai_presets.json"
        stored = json.loads(config_file.read_text(encoding="utf-8"))
        assert "gemini_thinking_levels" not in stored
        for item in stored["presets"]:
            item.pop("thinking_level")
        config_file.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
        assert all(item["thinking_level"] is None for item in client.get("/api/ai-presets").json()["presets"])
        assert client.put(f"/api/ai-presets/{original_id}", json=preset_data(model="gemini-2.5-flash")).status_code == 200


@pytest.mark.asyncio
async def test_gemini_thinking_level_reaches_native_request() -> None:
    sent: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}}]})

    preset = AiPreset(id="gemini", **preset_data(model="models/gemini-3-flash-preview", thinking_level="medium"))
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    result = await client.complete("system", [{"role": "user", "content": "请求"}])
    assert result.content == "OK"
    assert sent[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MEDIUM"}
    assert client.metadata()["thinking_level"] == "medium"

    automatic = BoundModelClient(preset.model_copy(update={"thinking_level": None}), transport=httpx.MockTransport(handler))
    await automatic.complete("system", [{"role": "user", "content": "请求"}])
    assert "thinkingConfig" not in sent[1]["generationConfig"]
    assert "thinking_level" not in automatic.metadata()

    assert "thinkingConfig" not in request_payload("system", [], None, temperature=1, top_p=1, max_tokens=100)["generationConfig"]
    with pytest.raises(ValueError, match="思考强度"):
        request_payload("system", [], None, temperature=1, top_p=1, max_tokens=100,
                        model="gemini-2.5-flash", thinking_level="low")
    with pytest.raises(ValueError, match="思考强度"):
        request_payload("system", [], None, temperature=1, top_p=1, max_tokens=100,
                        model="gemini-3.8-flash", thinking_level="minimal")
