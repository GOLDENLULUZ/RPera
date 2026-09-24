from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from rpera.agents import AGENTS
from rpera.app import create_app
from rpera.capabilities import CapabilityContext, CapabilityError, CapabilityExecutor
from rpera.drawing_providers import GeneratedImage
from rpera.models import CharacterPortraitGenerationSettings, DrawingPreset, SaveCreate
from rpera.runtime import AgentRunner
from tests.helpers import make_data_dir


def tiny_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), (17, 34, 51)).save(output, format="PNG")
    return output.getvalue()


def tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {tool["function"]["name"] for tool in tools}


def test_generated_image_artifact_writes_png_metadata_and_reads_back(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="生成立绘"))
    image = tiny_png()

    metadata = app.state.saves.write_generated_image_artifact(save.id, image)

    assert metadata == {
        "path": metadata["path"],
        "mime_type": "image/png",
        "size": len(image),
        "width": 3,
        "height": 2,
        "sha256": hashlib.sha256(image).hexdigest(),
    }
    assert metadata["path"].startswith("artifacts/character-portrait-")
    assert metadata["path"].endswith(".png")
    assert (app.state.saves.saves_dir / save.id / "current" / metadata["path"]).read_bytes() == image
    assert app.state.saves.read_image_artifact(save.id, metadata["path"]) == {
        "path": metadata["path"],
        "mime_type": "image/png",
        "size": len(image),
        "sha256": metadata["sha256"],
        "data": base64.b64encode(image).decode("ascii"),
    }


def test_generated_image_artifact_removes_published_file_when_directory_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="立绘写入失败"))
    artifacts = app.state.saves.saves_dir / save.id / "current" / "artifacts"

    def fail_sync(_path: Path) -> None:
        raise OSError("模拟目录同步失败")

    monkeypatch.setattr(app.state.saves, "_sync_directory", fail_sync)
    with pytest.raises(OSError, match="模拟目录同步失败"):
        app.state.saves.write_generated_image_artifact(save.id, tiny_png())

    assert list(artifacts.iterdir()) == []


def test_character_portrait_capability_schema_has_only_required_description(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    spec = app.state.runtime.capability_registry.get("character_portrait_generate")

    schema = spec.input_model.model_json_schema()
    assert set(schema["properties"]) == {"description"}
    assert schema["required"] == ["description"]
    assert "default" not in schema["properties"]["description"]


@pytest.mark.asyncio
async def test_character_portrait_handler_generates_persists_and_emits_without_base64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="立绘处理器"))
    image = tiny_png()
    events: list[tuple[str, dict[str, Any]]] = []
    observed: dict[str, Any] = {}
    preset = DrawingPreset(id="portrait", name="本地绘图")
    settings = CharacterPortraitGenerationSettings(enabled=True, width=64, height=64)

    class FakeStableDiffusionWebUIClient:
        def __init__(self, drawing_preset: DrawingPreset, **kwargs: Any) -> None:
            observed["preset"] = drawing_preset
            observed.update(kwargs)

        async def generate(
            self,
            description: str,
            portrait_settings: CharacterPortraitGenerationSettings,
        ) -> GeneratedImage:
            observed["description"] = description
            observed["settings"] = portrait_settings
            return GeneratedImage(image, "image/png", 3, 2, 8675309)

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    monkeypatch.setattr("rpera.capabilities.StableDiffusionWebUIClient", FakeStableDiffusionWebUIClient)
    executor = CapabilityExecutor(app.state.runtime.capability_registry)
    context = CapabilityContext(
        save.id,
        "turn",
        "character_designer",
        emit,
        drawing_preset=preset,
        portrait_settings=settings,
    )

    result = await executor.execute(
        "character_portrait_generate",
        {"description": "正面站立的港区档案员"},
        context,
        {"character_portrait_generate"},
    )

    assert observed == {
        "preset": preset,
        "network_settings": None,
        "description": "正面站立的港区档案员",
        "settings": settings,
    }
    assert result["seed"] == 8675309
    assert result["width"] == 3 and result["height"] == 2
    assert "data" not in result
    assert base64.b64encode(image).decode("ascii") not in json.dumps(result)
    assert events == [("character.portrait_generated", result)]
    readback = app.state.saves.read_image_artifact(save.id, result["path"])
    assert base64.b64decode(readback["data"]) == image


@pytest.mark.asyncio
async def test_character_portrait_uses_novelai_default_preset_and_persists_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="NovelAI 立绘"))
    preset = DrawingPreset(id="nai", name="NovelAI", provider="novelai", base_url="https://api.novelai.net", api_key="token")
    settings = CharacterPortraitGenerationSettings(enabled=True)
    image = tiny_png()

    class FakeNovelAIClient:
        def __init__(self, drawing_preset: DrawingPreset, **_kwargs: Any) -> None:
            assert drawing_preset == preset

        async def generate(self, description: str, portrait_settings: CharacterPortraitGenerationSettings) -> GeneratedImage:
            assert description == "角色肖像"
            assert portrait_settings == settings
            return GeneratedImage(image, "image/png", 3, 2, 13)

    async def emit(_event_type: str, _payload: dict[str, Any]) -> None:
        pass

    monkeypatch.setattr("rpera.capabilities.NovelAIClient", FakeNovelAIClient)
    result = await CapabilityExecutor(app.state.runtime.capability_registry).execute(
        "character_portrait_generate", {"description": "角色肖像"},
        CapabilityContext(save.id, "turn", "character_designer", emit, drawing_preset=preset, portrait_settings=settings),
        {"character_portrait_generate"},
    )
    assert result["seed"] == 13
    assert (app.state.saves.saves_dir / save.id / "current" / result["path"]).read_bytes() == image


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "preset", "message"),
    [
        (None, DrawingPreset(id="portrait", name="本地绘图"), "未启用"),
        (CharacterPortraitGenerationSettings(), DrawingPreset(id="portrait", name="本地绘图"), "未启用"),
        (CharacterPortraitGenerationSettings(enabled=True), None, "缺少默认绘画 Preset"),
    ],
)
async def test_character_portrait_handler_rejects_missing_or_disabled_context(
    tmp_path: Path,
    settings: CharacterPortraitGenerationSettings | None,
    preset: DrawingPreset | None,
    message: str,
) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="立绘禁用"))

    async def emit(_event_type: str, _payload: dict[str, Any]) -> None:
        raise AssertionError("rejected generation must not emit an event")

    context = CapabilityContext(
        save.id,
        "turn",
        "character_designer",
        emit,
        drawing_preset=preset,
        portrait_settings=settings,
    )
    executor = CapabilityExecutor(app.state.runtime.capability_registry)

    with pytest.raises(CapabilityError, match=message):
        await executor.execute(
            "character_portrait_generate",
            {"description": "不应生成"},
            context,
            {"character_portrait_generate"},
        )


def test_completed_portrait_result_reconstructs_visual_block_and_rejects_changed_hash() -> None:
    image = tiny_png()
    encoded = base64.b64encode(image).decode("ascii")
    metadata = {
        "path": "artifacts/character-portrait-fixed.png",
        "mime_type": "image/png",
        "size": len(image),
        "width": 3,
        "height": 2,
        "sha256": hashlib.sha256(image).hexdigest(),
        "seed": 42,
    }
    messages = [{
        "role": "assistant",
        "model": "{}",
        "parts": [{
            "type": "tool",
            "tool_name": "character_portrait_generate",
            "state": "completed",
            "provider_call_id": "portrait-call",
            "input": '{"description":"角色"}',
            "output": json.dumps(metadata, ensure_ascii=False),
        }],
    }]

    def read_artifact(save_id: str, path: str) -> dict[str, Any]:
        assert save_id == "save"
        assert path == metadata["path"]
        return {**metadata, "data": encoded}

    story = {"story": {"opening": "", "turns": []}}
    rebuilt = AgentRunner._provider_messages(messages, story, "save", artifact_reader=read_artifact)
    tool_message = next(message for message in rebuilt if message["role"] == "tool")
    assert tool_message["content"] == [
        {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]

    def read_changed_artifact(_save_id: str, _path: str) -> dict[str, Any]:
        return {**metadata, "sha256": "0" * 64, "data": encoded}

    with pytest.raises(RuntimeError, match="内容与已完成 Tool Result 不一致"):
        AgentRunner._provider_messages(messages, story, "save", artifact_reader=read_changed_artifact)


def test_portrait_tool_is_exposed_only_to_enabled_character_designer(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    runner = app.state.runtime.runner
    descriptions = app.state.runtime.prompts.load(
        frozenset(spec.name for spec in AGENTS.all())
    ).capability_descriptions

    for agent in AGENTS.all():
        disabled = tool_names(runner._tools_for(agent, descriptions, enabled_agents=frozenset()))
        enabled = tool_names(
            runner._tools_for(
                agent,
                descriptions,
                enabled_agents=frozenset(),
                enabled_configurable_capabilities=frozenset({"character_portrait_generate"}),
            )
        )
        assert "character_portrait_generate" not in disabled
        assert ("character_portrait_generate" in enabled) is (agent.name == "character_designer")
