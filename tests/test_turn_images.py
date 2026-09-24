from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rpera.app import create_app
from rpera.models import ModelResult, SaveCreate
from rpera.runtime import AgentRunner
from tests.helpers import make_data_dir, wait_for_turn


def png_bytes(color: tuple[int, int, int]) -> bytes:
    output = BytesIO()
    Image.new("RGB", (3, 2), color).save(output, format="PNG")
    return output.getvalue()


class CapturingModelClient:
    def __init__(self, provider: str = "deepseek") -> None:
        self.requests: list[list[dict[str, Any]]] = []
        self.provider = provider

    def bind(self):
        return self

    def bind_for_agents(self, names: tuple[str, ...]):
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": self.provider, "model": "visual-test", "base_url": "local", "timeout_seconds": 1}

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        del system_prompt, tools
        self.requests.append(messages)
        return ModelResult(content="", raw_response={}, tool_calls=[])


@pytest.mark.parametrize("provider", ["openai_compatible", "deepseek", "google_gemini", "anthropic", "xai"])
def test_multipart_turn_persists_images_and_sends_provider_independent_blocks(tmp_path: Path, provider: str) -> None:
    data_dir = make_data_dir(tmp_path)
    model = CapturingModelClient(provider)
    app = create_app(data_dir=data_dir, model_client=model)
    uploaded = [png_bytes((index * 40, 0, 255 - index * 40)) for index in range(5)]

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "图片存档"}).json()
        response = client.post(
            f"/api/saves/{save['id']}/turns",
            data={"content": ""},
            files=[
                ("images", (f"第{index + 1}张.png", data, "image/png"))
                for index, data in enumerate(uploaded)
            ],
        )
        assert response.status_code == 202
        created = response.json()
        assert [image["original_name"] for image in created["images"]] == [f"第{index + 1}张.png" for index in range(5)]
        turn = wait_for_turn(client, save["id"])
        assert turn["player_input"] == ""
        assert [image["position"] for image in turn["images"]] == list(range(5))

        image_response = client.get(turn["images"][0]["content_url"])
        assert image_response.status_code == 200
        assert image_response.headers["content-type"] == "image/png"
        assert image_response.content == uploaded[0]
        assert client.get(
            f"/api/saves/{save['id']}/turns/{turn['id']}/images/{turn['images'][1]['id']}/content"
        ).content == uploaded[1]
        assert client.get(
            f"/api/saves/{save['id']}/turns/00000000-0000-0000-0000-000000000000/images/{turn['images'][0]['id']}/content"
        ).status_code == 404

    user_content = next(
        message["content"]
        for message in model.requests[0]
        if message["role"] == "user" and isinstance(message["content"], list)
    )
    assert [block["type"] for block in user_content] == ["image_url"] * 5
    for block, data in zip(user_content, uploaded, strict=True):
        assert block["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(data).decode('ascii')}"


def test_invalid_upload_creates_neither_turn_nor_artifact(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=CapturingModelClient())

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "拒绝伪图"}).json()
        response = client.post(
            f"/api/saves/{save['id']}/turns",
            data={"content": "查看图片"},
            files=[("images", ("伪造.png", b"not an image", "image/png"))],
        )
        assert response.status_code == 400
        assert client.get(f"/api/saves/{save['id']}/turns").json() == []
        assert list((data_dir / "saves" / save["id"] / "current" / "artifacts").iterdir()) == []


def test_completed_turn_images_follow_the_existing_history_window(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=CapturingModelClient())
    saves = app.state.saves
    save = saves.create_save(SaveCreate(world_name="雾港", name="图片历史"))
    old_image = png_bytes((10, 20, 30))
    current_image = png_bytes((40, 50, 60))

    first = saves.create_turn(save.id, "先看这张", 4, [("旧图.png", BytesIO(old_image))])
    with saves.connect(save.id) as db:
        db.execute(
            "UPDATE turns SET status = 'completed', narrative = ?, completed_at = created_at WHERE id = ?",
            ("第一段故事", first.id),
        )
    second = saves.create_turn(save.id, "再看这张", 4, [("新图.png", BytesIO(current_image))])
    messages = saves.load_session(save.id, saves.root_session_id(save.id, second.id))

    rebuilt = app.state.runtime.runner._provider_messages(
        messages,
        saves.load_story(save.id, second.id),
        save.id,
        coordinator=True,
        image_reader=saves.read_turn_image_data,
    )
    users = [message for message in rebuilt if message["role"] == "user"]
    assert len(users) == 2
    assert users[0]["content"][0] == {"type": "text", "text": "先看这张"}
    assert users[0]["content"][1]["image_url"]["url"].endswith(base64.b64encode(old_image).decode("ascii"))
    assert users[1]["content"][0] == {"type": "text", "text": "再看这张"}
    assert users[1]["content"][1]["image_url"]["url"].endswith(base64.b64encode(current_image).decode("ascii"))
    assert any(message == {"role": "assistant", "content": "第一段故事"} for message in rebuilt)

    child_rebuilt = app.state.runtime.runner._provider_messages(
        messages,
        saves.load_story(save.id, second.id),
        save.id,
        coordinator=False,
        image_reader=saves.read_turn_image_data,
    )
    child_users = [message for message in child_rebuilt if message["role"] == "user"]
    child_story = json.loads(child_users[0]["content"][0]["text"])["story"]
    assert [turn["turn_number"] for turn in child_story["turns"]] == [1, 2]
    assert child_users[0]["content"][2]["image_url"]["url"].endswith(base64.b64encode(old_image).decode("ascii"))
    assert child_users[0]["content"][4]["image_url"]["url"].endswith(base64.b64encode(current_image).decode("ascii"))

    saves.fail_turn(save.id, second.id, "turn.failed", {"message": "测试失败"})
    saves.retry_turn(save.id, second.id, 4)
    retried = app.state.runtime.runner._provider_messages(
        saves.load_session(save.id, saves.root_session_id(save.id, second.id)),
        saves.load_story(save.id, second.id),
        save.id,
        coordinator=True,
        image_reader=saves.read_turn_image_data,
    )
    retried_users = [message for message in retried if message["role"] == "user"]
    assert retried_users[1]["content"][1]["image_url"]["url"].endswith(base64.b64encode(current_image).decode("ascii"))


@pytest.mark.parametrize("image_format", ["GIF", "WEBP"])
def test_upload_rejects_formats_not_supported_by_every_provider(tmp_path: Path, image_format: str) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=CapturingModelClient())
    output = BytesIO()
    Image.new("RGB", (3, 2), "red").save(output, format=image_format)

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "格式边界"}).json()
        response = client.post(
            f"/api/saves/{save['id']}/turns",
            data={"content": "查看图片"},
            files=[("images", (f"图片.{image_format.lower()}", output.getvalue(), f"image/{image_format.lower()}"))],
        )

    assert response.status_code == 400
    assert "只支持 PNG 或 JPEG" in response.json()["detail"]


def test_invalid_or_unsupported_persisted_image_reference_fails_explicitly() -> None:
    with pytest.raises(RuntimeError, match="缺少合法 ID"):
        AgentRunner._content_with_images("save", "", [{}], lambda _save, _image: {})

    with pytest.raises(RuntimeError, match="JPEG 或 PNG"):
        AgentRunner._content_with_images(
            "save",
            "",
            [{"id": "old-image"}],
            lambda _save, _image: {"mime_type": "image/gif", "data": "aGVsbG8="},
        )
