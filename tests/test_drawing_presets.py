from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rpera.app import create_app
from rpera.drawing_providers import NovelAIClient, StableDiffusionWebUIClient
from rpera.models import CharacterPortraitGenerationSettings, DrawingPreset
from tests.helpers import make_data_dir
from tests.test_prototype import ScriptedModelClient


def drawing_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "本地 SD",
        "base_url": "http://127.0.0.1:7860",
        "auth_username": "user",
        "auth_password": "password",
        "timeout_seconds": 300,
    }
    payload.update(overrides)
    return payload


def portrait_settings_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "enabled": True,
        "checkpoint": "model.safetensors [12345678]",
        "sampler_name": "Euler",
        "scheduler": "Automatic",
        "steps": 20,
        "cfg_scale": 7,
        "width": 512,
        "height": 512,
        "positive_prompt": "masterpiece, cinematic",
        "negative_prompt": "lowres, blurry",
    }
    payload.update(overrides)
    return payload


def test_drawing_preset_crud_default_and_password_rules(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        assert client.get("/api/drawing-presets").json() == {"default_preset_id": None, "presets": []}
        created = client.post("/api/drawing-presets", json=drawing_payload())
        assert created.status_code == 201
        collection = created.json()
        preset = collection["presets"][0]
        preset_id = preset["id"]
        assert collection["default_preset_id"] == preset_id
        assert preset["has_auth_password"] is True
        assert '"auth_password"' not in created.text
        assert set(preset) == {
            "id", "name", "provider", "base_url", "auth_username", "has_auth_password", "has_api_key", "model", "timeout_seconds"
        }
        assert not set(portrait_settings_payload()) & set(preset)
        stale = client.post(
            "/api/drawing-presets",
            json={**drawing_payload(name="旧客户端"), "checkpoint": "不应被静默忽略"},
        )
        assert stale.status_code == 422

        settings = {"character_portrait_generation": portrait_settings_payload(novelai_sampler="k_euler_ancestral")}
        updated_settings = client.put("/api/capability-settings", json=settings)
        assert updated_settings.status_code == 200
        assert updated_settings.json() == settings
        assert client.get("/api/capability-settings").json() == settings

        unchanged = client.put(f"/api/drawing-presets/{preset_id}", json=drawing_payload(auth_password="")).json()
        assert unchanged["presets"][0]["has_auth_password"] is True
        changed = client.put(f"/api/drawing-presets/{preset_id}", json=drawing_payload(base_url="http://127.0.0.1:7861", auth_password="")).json()
        assert changed["presets"][0]["has_auth_password"] is False
        duplicated = client.post(f"/api/drawing-presets/{preset_id}/duplicate", json={"name": "副本"}).json()
        assert len(duplicated["presets"]) == 2
        assert client.delete(f"/api/drawing-presets/{preset_id}").json()["default_preset_id"] != preset_id

    assert json.loads((data_dir / "config" / "capability_settings.json").read_text(encoding="utf-8")) == settings


@pytest.mark.asyncio
async def test_webui_resources_and_test_generation() -> None:
    requests: list[httpx.Request] = []
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (512).to_bytes(4, "big") + (768).to_bytes(4, "big")

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"].startswith("Basic ")
        match request.url.path:
            case "/sdapi/v1/options":
                return httpx.Response(200, json={"sd_model_checkpoint": "active"})
            case "/sdapi/v1/sd-models":
                return httpx.Response(200, json=[{"title": "model-a"}])
            case "/sdapi/v1/samplers":
                return httpx.Response(200, json=[{"name": "Euler"}])
            case "/sdapi/v1/schedulers":
                return httpx.Response(200, json=[{"label": "Automatic"}])
            case "/sdapi/v1/txt2img":
                payload = json.loads(request.content)
                assert payload["prompt"] == "masterpiece, cinematic, a lighthouse"
                assert payload["negative_prompt"] == "lowres, blurry"
                assert payload["override_settings"] == {"sd_model_checkpoint": "model.safetensors [12345678]"}
                return httpx.Response(200, json={"images": [base64.b64encode(png).decode()], "info": '{"seed": 42}'})
        raise AssertionError(request.url.path)

    preset = DrawingPreset(id="test", **drawing_payload())
    client = StableDiffusionWebUIClient(preset, transport=httpx.MockTransport(handler))
    resources = await client.resources()
    settings = CharacterPortraitGenerationSettings.model_validate(portrait_settings_payload())
    result = await client.generate_test("a lighthouse", settings)
    assert resources.active_checkpoint == "active"
    assert [item.id for item in resources.checkpoints] == ["model-a"]
    assert result.width == 512 and result.height == 768 and result.seed == 42
    assert result.image_data_url.startswith("data:image/png;base64,")
    request_paths = [request.url.path for request in requests]
    assert request_paths
    assert "/sdapi/v1/txt2img" in request_paths


@pytest.mark.asyncio
async def test_webui_resources_remain_available_without_scheduler_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/schedulers"):
            return httpx.Response(404, text="not found")
        if request.url.path.endswith("/options"):
            return httpx.Response(200, json={"sd_model_checkpoint": "active"})
        if request.url.path.endswith("/sd-models"):
            return httpx.Response(200, json=[{"title": "model-a"}])
        if request.url.path.endswith("/samplers"):
            return httpx.Response(200, json=[{"name": "Euler"}])
        raise AssertionError(request.url.path)

    client = StableDiffusionWebUIClient(DrawingPreset(id="test", **drawing_payload()), transport=httpx.MockTransport(handler))
    resources = await client.resources()
    assert [item.id for item in resources.checkpoints] == ["model-a"]
    assert resources.schedulers == []


@pytest.mark.parametrize("model", ["nai-diffusion-5-full", "nai-diffusion-5-curated", "nai-diffusion-4-5-full", "nai-diffusion-4-5-curated"])
def test_novelai_preset_connection_and_test_generation_from_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str,
) -> None:
    data_dir = make_data_dir(tmp_path)
    image = BytesIO()
    Image.new("RGB", (64, 64)).save(image, format="PNG")
    archive = BytesIO()
    with ZipFile(archive, "w") as zip_file:
        zip_file.writestr("image_0.png", image.getvalue())
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer secret-token"
        if request.url.path == "/user/information":
            return httpx.Response(200, json={"tier": 3})
        assert request.url.path == "/ai/generate-image"
        payload = json.loads(request.content)
        assert payload["model"] == model and payload["action"] == "generate"
        assert payload["input"] == "quality, a lighthouse"
        assert payload["parameters"]["v4_prompt"]["caption"]["base_caption"] == payload["input"]
        assert payload["parameters"]["v4_negative_prompt"]["caption"]["base_caption"] == "blurry"
        assert payload["parameters"]["sampler"] == "k_dpmpp_2m"
        assert payload["parameters"]["n_samples"] == 1
        assert payload["parameters"]["params_version"] == (4 if model.startswith("nai-diffusion-5-") else 3)
        return httpx.Response(201, content=archive.getvalue(), headers={"content-type": "application/zip"})

    class MockNovelAIClient(NovelAIClient):
        def __init__(self, preset: DrawingPreset, **kwargs: Any) -> None:
            super().__init__(preset, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("rpera.app.NovelAIClient", MockNovelAIClient)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        created = client.post("/api/drawing-presets", json={
            "name": "NAI", "provider": "novelai", "base_url": "https://api.novelai.net",
            "api_key": "secret-token", "model": model,
        })
        assert created.status_code == 201
        preset_id = created.json()["presets"][0]["id"]
        assert created.json()["presets"][0]["has_api_key"] is True
        assert "secret-token" not in created.text
        assert client.post(f"/api/drawing-presets/{preset_id}/test").json()["active_checkpoint"] == model
        assert client.get(f"/api/drawing-presets/{preset_id}/resources").status_code == 400
        settings = client.get("/api/capability-settings").json()
        settings["character_portrait_generation"].update({
            "positive_prompt": "quality", "negative_prompt": "blurry", "novelai_sampler": "k_dpmpp_2m",
        })
        assert client.put("/api/capability-settings", json=settings).status_code == 200
        result = client.post(f"/api/drawing-presets/{preset_id}/test-generation", json={"content_prompt": "a lighthouse"})
        assert result.status_code == 200
        assert base64.b64decode(result.json()["image_data_url"].split(",", 1)[1]) == image.getvalue()
        assert result.json()["seed"] > 0
        assert [request.url.path for request in requests] == ["/user/information", "/ai/generate-image"]

        unchanged = client.put(f"/api/drawing-presets/{preset_id}", json={
            "name": "NAI", "provider": "novelai", "base_url": "https://api.novelai.net", "model": model,
        })
        assert unchanged.json()["presets"][0]["has_api_key"] is True
        switched = client.put(f"/api/drawing-presets/{preset_id}", json={
            "name": "SD", "provider": "stable_diffusion_webui", "base_url": "http://127.0.0.1:7860",
        })
        assert switched.json()["presets"][0]["has_api_key"] is False
        assert "secret-token" not in (data_dir / "config" / "drawing_presets.json").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_novelai_errors_do_not_leak_token_or_accept_broken_zip() -> None:
    preset = DrawingPreset(id="nai", name="NAI", provider="novelai", base_url="https://api.novelai.net", api_key="secret-token")
    settings = CharacterPortraitGenerationSettings()

    def reject(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid secret-token")

    with pytest.raises(RuntimeError, match="invalid \\*\\*\\*") as error:
        await NovelAIClient(preset, transport=httpx.MockTransport(reject)).generate("portrait", settings)
    assert "secret-token" not in str(error.value)

    def broken(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, content=b"not-a-zip")

    with pytest.raises(RuntimeError, match="压缩包"):
        await NovelAIClient(preset, transport=httpx.MockTransport(broken)).generate("portrait", settings)
