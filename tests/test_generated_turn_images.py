from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rpera.anthropic import request_payload as anthropic_request_payload
from rpera.app import create_app
from rpera.gemini import request_payload as gemini_request_payload
from rpera.models import SaveCreate
from rpera.runtime import AgentRunner
from rpera.storage import SaveStore
from rpera.xai import request_payload as xai_request_payload
from tests.helpers import make_data_dir
from tests.test_turn_images import png_bytes


def delegate(saves: SaveStore, save_id: str, turn_id: str, task_id: str | None = None) -> str:
    root = saves.root_session_id(save_id, turn_id)
    message_id = saves.create_assistant_message(save_id, root, {"provider": "test"})
    part = saves.record_model_result(
        save_id, message_id, "", None,
        [{"id": "task-call", "name": "task", "arguments": "{}"}], None, None,
    )[0]
    saves.set_tool_running(save_id, part["id"])
    return saves.prepare_child_task(
        save_id, turn_id, root, part["id"], "character_designer", {"task": "生成人物"}, task_id,
    )


def generate(saves: SaveStore, save_id: str, session_id: str, data: bytes, *, success: bool = True) -> str:
    metadata = saves.write_generated_image_artifact(save_id, data)
    message_id = saves.create_assistant_message(save_id, session_id, {"provider": "test"})
    part = saves.record_model_result(
        save_id, message_id, "", None,
        [{"id": "portrait-call", "name": "character_portrait_generate", "arguments": "{}"}], None, None,
    )[0]
    saves.set_tool_running(save_id, part["id"])
    if success:
        saves.complete_tool(save_id, part["id"], metadata)
    else:
        saves.error_tool(save_id, part["id"], {"ok": False, "error": {"message": "生成失败"}})
    return part["id"]


def test_each_independent_agent_session_shows_only_its_last_successful_image(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    saves = app.state.saves
    save = saves.create_save(SaveCreate(world_name="雾港", name="回合生成图"))
    uploaded = png_bytes((2, 3, 4))
    turn = saves.create_turn(save.id, "人物", 4, [("玩家图.png", BytesIO(uploaded))])
    first = delegate(saves, save.id, turn.id)
    discarded = generate(saves, save.id, first, png_bytes((10, 20, 30)))
    chosen_first_data = png_bytes((40, 50, 60))
    chosen_first = generate(saves, save.id, first, chosen_first_data)
    # Continuing the same session must replace its selection, not create another slot.
    assert delegate(saves, save.id, turn.id, first) == first
    generate(saves, save.id, first, png_bytes((99, 99, 99)), success=False)
    second = delegate(saves, save.id, turn.id)
    assert second != first
    generate(saves, save.id, second, png_bytes((1, 2, 3)))
    chosen_second_data = png_bytes((70, 80, 90))
    chosen_second = generate(saves, save.id, second, chosen_second_data)
    saves.fail_turn(save.id, turn.id, "turn.failed", {"message": "测试失败"})

    with TestClient(app) as client:
        path = f"/api/saves/{save.id}/turns/{turn.id}"
        listed = client.get(f"/api/saves/{save.id}/turns").json()[0]
        fetched = client.get(path).json()
        assert fetched == listed
        assert fetched["status"] == "failed"
        assert len(fetched["images"]) == 1
        assert [image["id"] for image in fetched["generated_images"]] == [chosen_first, chosen_second]
        assert [client.get(image["content_url"]).content for image in fetched["generated_images"]] == [
            chosen_first_data, chosen_second_data,
        ]
        assert all(image["agent"] == "character_designer" for image in fetched["generated_images"])
        assert client.get(f"{path}/generated-images/{discarded}/content").status_code == 200
        assert client.get(f"/api/saves/{save.id}/turns/not-this-turn/generated-images/{chosen_first}/content").status_code == 404
        assert client.get(f"{path}/generated-images/{fetched['images'][0]['id']}/content").status_code == 404

        assert [image["id"] for image in client.get(path).json()["generated_images"]] == [chosen_first, chosen_second]
        saves.retry_turn(save.id, turn.id, 4)
        assert [image["id"] for image in client.get(path).json()["generated_images"]] == [chosen_first, chosen_second]
        newer_data = png_bytes((100, 110, 120))
        newer = generate(saves, save.id, first, newer_data)
        assert [image["id"] for image in client.get(path).json()["generated_images"]] == [newer, chosen_second]
        assert client.get(f"{path}/generated-images/{newer}/content").content == newer_data


def test_generated_image_url_is_bound_to_save_and_validated_against_tool_result(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    saves = app.state.saves
    source = saves.create_save(SaveCreate(world_name="雾港", name="来源"))
    other = saves.create_save(SaveCreate(world_name="雾港", name="其他存档"))
    turn = saves.create_turn(source.id, "生成", 4)
    session = delegate(saves, source.id, turn.id)
    part_id = generate(saves, source.id, session, png_bytes((3, 5, 7)))
    url = f"/api/saves/{source.id}/turns/{turn.id}/generated-images/{part_id}/content"

    with TestClient(app) as client:
        assert client.get(url).status_code == 200
        assert client.get(url.replace(source.id, other.id)).status_code == 404
        assert client.get(f"/api/saves/{source.id}/turns/{turn.id}/generated-images/no-part/content").status_code == 404
        with saves.connect(source.id) as db:
            output = db.execute("SELECT output FROM parts WHERE id = ?", (part_id,)).fetchone()[0]
        metadata = json.loads(output)
        artifact = saves.saves_dir / source.id / "current" / metadata["path"]
        artifact.write_bytes(png_bytes((9, 8, 7)))
        assert client.get(url).status_code == 400


def test_completed_turn_branch_retains_generated_images(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    saves = app.state.saves
    source = saves.create_save(SaveCreate(world_name="雾港", name="原存档"))
    turn = saves.create_turn(source.id, "生成", 4)
    session = delegate(saves, source.id, turn.id)
    image = png_bytes((20, 40, 60))
    part_id = generate(saves, source.id, session, image)
    with saves.connect(source.id) as db:
        db.execute("UPDATE turns SET status = 'completed', narrative = '完成' WHERE id = ?", (turn.id,))
    branch = saves.branch_save(source.id, turn.id, "分支")

    with TestClient(app) as client:
        result = client.get(f"/api/saves/{branch.id}/turns/{turn.id}").json()
        assert [item["id"] for item in result["generated_images"]] == [part_id]
        assert client.get(result["generated_images"][0]["content_url"]).content == image


def test_generated_images_follow_uploaded_images_into_frozen_story_and_provider_requests(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    saves = app.state.saves
    save = saves.create_save(SaveCreate(world_name="雾港", name="图片上下文"))
    upload = png_bytes((1, 2, 3))
    first = saves.create_turn(save.id, "第一回合", 4, [("玩家.png", BytesIO(upload))])
    child_a = delegate(saves, save.id, first.id)
    generate(saves, save.id, child_a, png_bytes((4, 5, 6)))
    generated_a = png_bytes((7, 8, 9))
    chosen_a = generate(saves, save.id, child_a, generated_a)
    child_b = delegate(saves, save.id, first.id)
    generated_b = png_bytes((10, 11, 12))
    chosen_b = generate(saves, save.id, child_b, generated_b)
    with saves.connect(save.id) as db:
        db.execute("UPDATE turns SET status = 'completed', narrative = '第一段故事' WHERE id = ?", (first.id,))

    second = saves.create_turn(save.id, "第二回合", 4)
    frozen = saves.load_story(save.id, second.id)
    prior = frozen["story"]["turns"][0]
    assert prior["player"]["images"] == [{"id": first.images[0].id}]
    assert prior["generated_images"] == [{"id": chosen_a}, {"id": chosen_b}]
    assert prior["narrative"] == "第一段故事"
    assert "generated_images" not in frozen["story"]["turns"][1]

    session = saves.load_session(save.id, saves.root_session_id(save.id, second.id))
    arguments = {"image_reader": saves.read_turn_image_data, "generated_reader": saves.read_generated_image_data}
    main = AgentRunner._provider_messages(session, frozen, save.id, coordinator=True, **arguments)
    assert main[2]["role"] == "user"
    assert main[2]["content"][2]["image_url"]["url"].endswith(base64.b64encode(upload).decode("ascii"))
    assert main[3] == {"role": "assistant", "content": "第一段故事"}
    assert main[4]["role"] == "user"
    assert "非玩家上传" in main[4]["content"][0]["text"]
    assert [block["image_url"]["url"].split(",", 1)[1] for block in main[4]["content"][1:]] == [
        base64.b64encode(data).decode("ascii") for data in (generated_a, generated_b)
    ]
    assert main[5]["content"] == "第二回合"

    child = AgentRunner._provider_messages(session, frozen, save.id, **arguments)
    content = child[0]["content"]
    assert json.loads(content[0]["text"]) == frozen
    assert json.loads(content[3]["text"]) == {"turn_number": 1, "source": "generated", "image_id": chosen_a}
    assert content[4]["image_url"]["url"].endswith(base64.b64encode(generated_a).decode("ascii"))
    assert json.loads(content[5]["text"]) == {"turn_number": 1, "source": "generated", "image_id": chosen_b}

    anthropic = anthropic_request_payload("system", main, None, model="claude", temperature=1, top_p=1, max_tokens=100, stream=False)
    assert any(block["type"] == "image" and block["source"]["data"] == base64.b64encode(generated_a).decode("ascii")
               for message in anthropic["messages"] if message["role"] == "user" and isinstance(message["content"], list)
               for block in message["content"])
    gemini = gemini_request_payload("system", main, None, temperature=1, top_p=1, max_tokens=100)
    assert any(part.get("inlineData", {}).get("data") == base64.b64encode(generated_b).decode("ascii")
               for message in gemini["contents"] for part in message["parts"])
    xai = xai_request_payload("system", main, None, model="grok", temperature=1, top_p=1, max_tokens=100, stream=False)
    assert any(block.get("type") == "input_image" and block["image_url"].endswith(base64.b64encode(generated_a).decode("ascii"))
               for message in xai["input"] if message.get("role") == "user" and isinstance(message["content"], list)
               for block in message["content"])


def test_failed_turn_image_is_frozen_for_new_turn_and_retry_and_history_read(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    saves = app.state.saves
    save = saves.create_save(SaveCreate(world_name="雾港", name="无故事的图片"))
    first = saves.create_turn(save.id, "失败行动", 2)
    child_id = delegate(saves, save.id, first.id)
    data = png_bytes((23, 24, 25))
    image_id = generate(saves, save.id, child_id, data)
    assert "generated_images" not in saves.load_story(save.id, first.id)["story"]["turns"][-1]
    saves.fail_turn(save.id, first.id, "turn.interrupted", {"message": "中止"})
    saves.retry_turn(save.id, first.id, 2)
    retry_story = saves.load_story(save.id, first.id)["story"]["turns"][-1]
    assert retry_story["has_ai_output"] is False
    assert retry_story["generated_images"] == [{"id": image_id}]
    saves.fail_turn(save.id, first.id, "turn.failed", {"message": "失败"})

    second = saves.create_turn(save.id, "继续行动", 2)
    frozen = saves.load_story(save.id, second.id)
    assert frozen["story"]["turns"][0]["generated_images"] == [{"id": image_id}]
    args = {"generated_reader": saves.read_generated_image_data}
    main = AgentRunner._provider_messages(
        saves.load_session(save.id, saves.root_session_id(save.id, second.id)), frozen,
        save.id, coordinator=True, **args,
    )
    assert [message["role"] for message in main] == ["system", "assistant", "user", "user", "user", "system"]
    assert main[3]["content"][1]["image_url"]["url"].endswith(base64.b64encode(data).decode("ascii"))

    history = saves.read_story_history(save.id, second.id, 0)
    assert history["turns"][0]["generated_images"] == [{"id": image_id}]
    third_child = delegate(saves, save.id, second.id)
    message_id = saves.create_assistant_message(save.id, third_child, {"provider": "test"})
    part = saves.record_model_result(
        save.id, message_id, "", None,
        [{"id": "history-call", "name": "story_history_read", "arguments": "{}"}], None, None,
    )[0]
    saves.set_tool_running(save.id, part["id"])
    saves.complete_tool(save.id, part["id"], history)
    rebuilt = AgentRunner._provider_messages(
        saves.load_session(save.id, third_child), frozen, save.id, **args,
    )
    tool_result = next(message for message in rebuilt if message["role"] == "tool")
    assert json.loads(tool_result["content"][1]["text"]) == {
        "turn_number": 1, "source": "generated", "image_id": image_id,
    }
    assert tool_result["content"][2]["image_url"]["url"].endswith(base64.b64encode(data).decode("ascii"))

    with saves.connect(save.id) as db:
        metadata = json.loads(db.execute("SELECT output FROM parts WHERE id = ?", (image_id,)).fetchone()[0])
    artifact = saves.saves_dir / save.id / "current" / metadata["path"]
    artifact.write_bytes(png_bytes((90, 91, 92)))
    with pytest.raises(ValueError, match="内容与 Tool Result 不一致"):
        AgentRunner._provider_messages(
            saves.load_session(save.id, saves.root_session_id(save.id, second.id)), frozen,
            save.id, coordinator=True, **args,
        )
