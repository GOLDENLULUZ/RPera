from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from rpera.app import create_app
from rpera.content import markdown_frontmatter
from rpera.models import ModelResult, SaveCreate, ToolCall
from rpera.runtime import AgentRunner, _redact_image_data
from tests.prompt_fixtures import CHARACTER_DESIGNER_PROMPT, COORDINATOR_PROMPT, NARRATOR_PROMPT
from tests.helpers import make_data_dir, make_symlink, wait_for_turn


PROFILE = {
    "type": "character",
    "gender": "女",
    "age": "27",
    "background": "港区账房家庭出身。",
    "goal": {"short-term": "找到失踪账册", "long-term": "摆脱议会控制"},
    "personality": "INTJ，谨慎而执着。",
    "personal_traits": {},
    "appearance": {},
    "attire": {},
    "relationship": {},
    "sex_related_traits": {},
    "模型自行扩展": {"保留": True},
}

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP4z8BAEiJN9aiGUQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII="
)


def test_save_character_operations_use_real_snapshot_files_without_migrating_rename(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    source_character = data_dir / "worlds" / "雾港" / "entities" / "character" / "伊蕾" / "ENTITY.md"
    original_source = source_character.read_text(encoding="utf-8")
    mod_character = data_dir / "mods" / "读心大师" / "entities" / "character" / "模组角色"
    mod_character.mkdir(parents=True)
    mod_character_file = mod_character / "ENTITY.md"
    mod_character_file.write_text(original_source, encoding="utf-8")
    app = create_app(data_dir=data_dir)
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="角色文件", mod_names=["读心大师"]))

    created = app.state.saves.create_character(
        save.id,
        "沈青",
        "议会档案员",
        ["沈档案员"],
        PROFILE,
    )
    assert created["path"] == "entities/character/沈青/ENTITY.md"
    saved_path = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / created["path"]
    front, body = markdown_frontmatter(saved_path.read_text(encoding="utf-8"), "角色")
    assert front == {
        "description": "议会档案员",
        "aliases": ["沈档案员"],
        "visibility": "world_truth",
        "required": False,
    }
    assert json.loads(body) == PROFILE
    assert all(key not in json.loads(body) for key in ("Chinese_name", "English_name", "Japanese_name", "keyword", "short_description"))

    with pytest.raises(ValueError, match="实体名称已存在"):
        app.state.saves.create_character(save.id, "暮潮旅店", "冲突", [], PROFILE)

    existing_path = "entities/character/伊蕾/ENTITY.md"
    snapshot_file = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / existing_path
    original_snapshot = snapshot_file.read_text(encoding="utf-8")
    renamed = app.state.saves.rename_character(save.id, existing_path, "伊蕾新名")
    assert renamed["path"] == "entities/character/伊蕾新名/ENTITY.md"
    renamed_file = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / renamed["path"]
    assert renamed_file.read_text(encoding="utf-8") == original_snapshot

    edited = app.state.saves.edit_character(
        save.id,
        renamed["path"],
        "description: 暮潮旅店老板娘，寡言而敏锐，与守灯人奥伦失踪事件有隐秘关联\naliases: [老板娘, 旅店老板]",
        "description: 更新后的老板娘\naliases: [老板娘, 伊蕾]",
    )
    assert edited["name"] == "伊蕾新名"
    assert edited["description"] == "更新后的老板娘"
    assert edited["aliases"] == ["老板娘", "伊蕾"]
    assert renamed_file.read_text(encoding="utf-8") == original_snapshot.replace(
        "description: 暮潮旅店老板娘，寡言而敏锐，与守灯人奥伦失踪事件有隐秘关联\naliases: [老板娘, 旅店老板]",
        "description: 更新后的老板娘\naliases: [老板娘, 伊蕾]",
        1,
    )

    mod_path = "mods/读心大师/entities/character/模组角色/ENTITY.md"
    mod_edited = app.state.saves.edit_character(
        save.id,
        mod_path,
        "description: 暮潮旅店老板娘，寡言而敏锐，与守灯人奥伦失踪事件有隐秘关联",
        "description: 模组内角色",
    )
    assert mod_edited["description"] == "模组内角色"
    mod_renamed = app.state.saves.rename_character(save.id, mod_path, "模组新名")
    assert mod_renamed["path"] == "mods/读心大师/entities/character/模组新名/ENTITY.md"
    assert mod_character_file.read_text(encoding="utf-8") == original_source
    assert source_character.read_text(encoding="utf-8") == original_source


def test_character_file_replace_survives_real_process_exit_before_and_after_publish(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="角色强杀"))
    path = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / "entities" / "character" / "伊蕾" / "ENTITY.md"
    old_content = path.read_text(encoding="utf-8")
    new_content = app.state.saves._character_document("新简介", ["老板娘"], PROFILE, required=False)
    temporary = path.parent / ".ENTITY.md.kill.tmp"

    before = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys\n"
                "from pathlib import Path\n"
                "from rpera.storage import SaveStore\n"
                "target, temporary, content = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]\n"
                "SaveStore._write_new_file(temporary, content.encode('utf-8'))\n"
                "os._exit(9)\n"
            ),
            str(path),
            str(temporary),
            new_content,
        ],
        check=False,
    )
    assert before.returncode == 9
    assert path.read_text(encoding="utf-8") == old_content
    temporary.unlink()

    after = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sys\n"
                "from pathlib import Path\n"
                "from rpera.storage import SaveStore\n"
                "target, temporary, content = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]\n"
                "SaveStore._write_new_file(temporary, content.encode('utf-8'))\n"
                "from rpera.filesystem import atomic_replace\n"
                "atomic_replace(temporary, target)\n"
                "os._exit(9)\n"
            ),
            str(path),
            str(temporary),
            new_content,
        ],
        check=False,
    )
    assert after.returncode == 9
    assert path.read_text(encoding="utf-8") == new_content
    assert json.loads(app.state.saves.read_entities_exact(save.id, ["entities/character/伊蕾/ENTITY.md"])[0]["content"]) == PROFILE


def test_character_writes_reject_non_standard_json_and_snapshot_root_symlink(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="角色边界"))

    with pytest.raises(ValueError, match="Out of range float values"):
        app.state.saves.create_character(save.id, "异常角色", "异常", [], {"value": float("nan")})
    assert not any(entity["name"] == "异常角色" for entity in app.state.saves.list_entities(save.id))

    save_dir = app.state.saves.saves_dir / save.id
    snapshot = save_dir / "current" / "world_snapshot"
    mods = snapshot / "mods"
    moved_mods = snapshot / "moved-mods"
    mods.rename(moved_mods)
    make_symlink(mods, moved_mods, target_is_directory=True)
    with pytest.raises(ValueError, match="模组快照根路径必须是非符号链接目录"):
        app.state.saves.list_entities(save.id)
    mods.unlink()
    moved_mods.rename(mods)

    moved = save_dir / "moved-snapshot"
    snapshot.rename(moved)
    make_symlink(snapshot, moved, target_is_directory=True)
    original = (moved / "entities" / "character" / "伊蕾" / "ENTITY.md").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="快照根路径必须是非符号链接目录"):
        app.state.saves.edit_character(
            save.id,
            "entities/character/伊蕾/ENTITY.md",
            "暮潮旅店老板娘",
            "不应写入",
        )
    assert (moved / "entities" / "character" / "伊蕾" / "ENTITY.md").read_text(encoding="utf-8") == original


class CharacterFlowModel:
    def __init__(self) -> None:
        self.narrator_input: dict[str, Any] | None = None
        self.character_input: dict[str, Any] | None = None

    def bind(self):
        return self

    def bind_for_agents(self, names: tuple[str, ...]):
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": "test", "model": "character-flow", "base_url": "local", "timeout_seconds": 1}

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        del tools
        completed = [message for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            if not completed:
                call = ToolCall(id="character-task", name="task", arguments={"agent": "character_designer", "task": "创建一位长期参与调查的议会档案员"})
            elif len(completed) == 1:
                result = json.loads(completed[0]["content"])
                call = ToolCall(
                    id="narrator-task",
                    name="task",
                    arguments={
                        "agent": "narrator",
                        "task": "让新角色进入场景",
                        "related_entities": result["result"]["related_entities"],
                        "report_refs": [result["report_ref"]],
                    },
                )
            else:
                call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
            return self._result(call)
        if system_prompt == CHARACTER_DESIGNER_PROMPT:
            if not completed:
                self.character_input = json.loads(messages[1]["content"])
                return self._result(ToolCall(
                    id="create-character",
                    name="character_create",
                    arguments={"name": "沈青", "description": "议会档案员", "aliases": ["沈档案员"], "profile": PROFILE},
                ))
            if len(completed) == 1:
                created = json.loads(completed[0]["content"])["character"]
                assert "content" not in created
                assert json.loads(markdown_frontmatter(created["document"], "角色")[1]) == PROFILE
                return self._result(ToolCall(
                    id="read-character",
                    name="entity_read",
                    arguments={"paths": ["entities/character/沈青/ENTITY.md"]},
                ))
            if len(completed) == 2:
                read_character = json.loads(completed[1]["content"])["documents"][0]
                assert "content" not in read_character
                assert read_character["document"] == json.loads(completed[0]["content"])["character"]["document"]
                return self._result(ToolCall(
                    id="rename-character",
                    name="character_rename",
                    arguments={"path": "entities/character/沈青/ENTITY.md", "new_name": "沈清"},
                ))
            if len(completed) == 3:
                return self._result(ToolCall(
                    id="edit-character",
                    name="character_edit",
                    arguments={
                        "path": "entities/character/沈清/ENTITY.md",
                        "old_text": "港区账房家庭出身。",
                        "new_text": "港区秘密账房家庭出身。",
                    },
                ))
            return self._result(ToolCall(
                id="report-character",
                name="character_report",
                arguments={"report": "创建并完善了长期角色沈清。", "related_entities": ["entities/character/沈清/ENTITY.md"]},
            ))
        if system_prompt == NARRATOR_PROMPT:
            if not completed:
                self.narrator_input = json.loads(messages[1]["content"])
                return self._result(ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "沈青合上账册，抬眼看向你。"}))
            return ModelResult(content="草稿完成。", raw_response={}, tool_calls=[])
        raise AssertionError(f"unexpected prompt: {system_prompt[:80]}")

    @staticmethod
    def _result(call: ToolCall) -> ModelResult:
        return ModelResult(content="", raw_response={}, tool_calls=[call])


def test_character_report_is_brief_while_runtime_hydrates_real_entity_for_narrator(tmp_path: Path) -> None:
    model = CharacterFlowModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "角色闭环", "mod_names": ["读心大师"]}).json()
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "我需要一位档案员协助调查。"})
        assert response.status_code == 202
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert model.character_input is not None
    assert model.character_input["related_entity_documents"][0]["name"] == "读心能力"
    assert model.narrator_input is not None
    assert model.narrator_input["report_documents"][0]["content"] == "创建并完善了长期角色沈清。"
    entity = model.narrator_input["related_entity_documents"][0]
    assert entity["name"] == "沈清"
    assert entity["aliases"] == ["沈档案员"]
    assert "document" not in entity
    assert json.loads(entity["content"])["background"] == "港区秘密账房家庭出身。"


def test_character_edit_replaces_only_unique_text_and_preserves_json_formatting(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="局部角色编辑"))
    profile = {
        **PROFILE,
        "background": "港区账房家庭出身。",
        "保真字段": {"'7-11'": "第一行\n第二行"},
    }
    created = app.state.saves.create_character(save.id, "沈青", "议会档案员", ["沈档案员"], profile)
    path = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / created["path"]
    before = path.read_text(encoding="utf-8")

    edited = app.state.saves.edit_character(save.id, created["path"], "港区账房家庭出身。", "港区秘密账房家庭出身。")

    after = path.read_text(encoding="utf-8")
    assert after == before.replace("港区账房家庭出身。", "港区秘密账房家庭出身。", 1)
    assert json.loads(edited["content"])["保真字段"] == {"'7-11'": "第一行\n第二行"}

    with pytest.raises(ValueError, match="唯一出现一次"):
        app.state.saves.edit_character(save.id, created["path"], "  ", " ")
    assert path.read_text(encoding="utf-8") == after

    with pytest.raises(ValueError, match="合法 JSON 对象"):
        app.state.saves.edit_character(save.id, created["path"], '"age": "27"', '"age":')
    assert path.read_text(encoding="utf-8") == after

    with pytest.raises(ValueError, match="不允许修改 visibility 或 required"):
        app.state.saves.edit_character(save.id, created["path"], "required: false", "required: true")
    assert path.read_text(encoding="utf-8") == after


def test_image_artifact_read_is_scoped_and_validates_real_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="图片边界"))
    artifacts = app.state.saves.saves_dir / save.id / "current" / "artifacts"
    image = artifacts / "角色.png"
    image.write_bytes(PNG_BYTES)

    observed = app.state.saves.read_image_artifact(save.id, "artifacts/角色.png")

    assert observed == {
        "path": "artifacts/角色.png",
        "mime_type": "image/png",
        "size": len(PNG_BYTES),
        "sha256": hashlib.sha256(PNG_BYTES).hexdigest(),
        "data": base64.b64encode(PNG_BYTES).decode("ascii"),
    }

    with pytest.raises(ValueError, match="顶层文件"):
        app.state.saves.read_image_artifact(save.id, "artifacts/nested/角色.png")
    with pytest.raises(ValueError, match="顶层文件"):
        app.state.saves.read_image_artifact(save.id, "artifacts/../save.json")

    fake = artifacts / "伪造.png"
    fake.write_bytes(b"\x89PNG\r\n\x1a\nnot a complete image")
    with pytest.raises(ValueError, match="不是完整有效"):
        app.state.saves.read_image_artifact(save.id, "artifacts/伪造.png")

    jpeg = BytesIO()
    Image.new("RGB", (64, 64), "red").save(jpeg, format="JPEG")
    truncated = artifacts / "截断.jpg"
    truncated.write_bytes(jpeg.getvalue()[:-1])
    with pytest.raises(ValueError, match="不是完整有效"):
        app.state.saves.read_image_artifact(save.id, "artifacts/截断.jpg")

    gif = BytesIO()
    Image.new("RGB", (4, 4), "blue").save(gif, format="GIF")
    (artifacts / "旧格式.gif").write_bytes(gif.getvalue())
    with pytest.raises(ValueError, match="只支持 PNG 或 JPEG"):
        app.state.saves.read_image_artifact(save.id, "artifacts/旧格式.gif")

    link = artifacts / "链接.png"
    make_symlink(link, image)
    with pytest.raises(ValueError, match="不存在或不是普通文件"):
        app.state.saves.read_image_artifact(save.id, "artifacts/链接.png")

    monkeypatch.setattr("rpera.storage.IMAGE_ARTIFACT_MAX_BYTES", len(PNG_BYTES) - 1)
    with pytest.raises(ValueError, match="超过 0 MiB 上限"):
        app.state.saves.read_image_artifact(save.id, "artifacts/角色.png")


class ImageCharacterFlowModel:
    def __init__(self, provider: str = "deepseek") -> None:
        self.image_observations = 0
        self.provider = provider
        self.image_path = "artifacts/参考.png"
        self.image_data = PNG_BYTES

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
        del tools
        completed = [message for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            if not completed:
                return self._result(ToolCall(
                    id="image-character-task",
                    name="task",
                    arguments={
                        "agent": "character_designer",
                        "task": f"读取 {self.image_path}，并据此创建长期角色",
                    },
                ))
            if len(completed) == 1:
                return self._result(ToolCall(
                    id="image-narrator-task",
                    name="task",
                    arguments={"agent": "narrator", "task": "简短确认角色已经建立"},
                ))
            return self._result(ToolCall(id="image-publish", name="narrative_publish", arguments={"path": "narrative.md"}))
        if system_prompt == CHARACTER_DESIGNER_PROMPT:
            image_results = [message for message in completed if isinstance(message["content"], list)]
            if not completed:
                return self._result(ToolCall(
                    id="read-image",
                    name="image_read",
                    arguments={"path": self.image_path},
                ))
            assert len(image_results) == 1
            blocks = image_results[0]["content"]
            assert json.loads(blocks[0]["text"])["path"] == self.image_path
            assert blocks[1]["image_url"]["url"].split(",", 1)[1] == base64.b64encode(self.image_data).decode("ascii")
            self.image_observations += 1
            if len(completed) == 1:
                return self._result(ToolCall(id="search-before-create", name="entity_search", arguments={"query": "图片参考角色"}))
            if len(completed) == 2:
                return self._result(ToolCall(
                    id="create-from-image",
                    name="character_create",
                    arguments={"name": "丹朱", "description": "依据参考图创建的角色", "aliases": [], "profile": PROFILE},
                ))
            return self._result(ToolCall(
                id="report-from-image",
                name="character_report",
                arguments={"report": "读取参考图并创建了丹朱。", "related_entities": ["entities/character/丹朱/ENTITY.md"]},
            ))
        if system_prompt == NARRATOR_PROMPT:
            if not completed:
                return self._result(ToolCall(id="image-story", name="file_write", arguments={"path": "narrative.md", "content": "新角色已经建立。"}))
            return ModelResult(content="草稿完成。", raw_response={}, tool_calls=[])
        raise AssertionError(f"unexpected prompt: {system_prompt[:80]}")

    @staticmethod
    def _result(call: ToolCall) -> ModelResult:
        return ModelResult(content="", raw_response={}, tool_calls=[call])


@pytest.mark.parametrize("provider", ["deepseek", "anthropic"])
def test_image_read_persists_visual_context_and_redacts_observation_events(tmp_path: Path, provider: str) -> None:
    model = ImageCharacterFlowModel(provider)
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "图片建角"}).json()
        artifacts = app.state.saves.saves_dir / save["id"] / "current" / "artifacts"
        (artifacts / "参考.png").write_bytes(PNG_BYTES)
        response = client.post(
            f"/api/saves/{save['id']}/turns",
            json={"content": "请让角色设计师读取 artifacts/参考.png 并创建角色。"},
        )
        assert response.status_code == 202
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert model.image_observations == 3
    assert (app.state.saves.saves_dir / save["id"] / "current" / "world_snapshot" / "entities" / "character" / "丹朱" / "ENTITY.md").is_file()

    events = app.state.saves.list_events(save["id"])
    character_task = next(event for event in events if event.type == "task.created" and event.payload["agent"] == "character_designer")
    messages = app.state.saves.load_session(save["id"], character_task.payload["task_id"])
    image_part = next(
        part
        for message in messages
        for part in message["parts"]
        if part["type"] == "tool" and part["tool_name"] == "image_read"
    )
    persisted = json.loads(image_part["output"])
    assert persisted["data"] == base64.b64encode(PNG_BYTES).decode("ascii")
    (artifacts / "参考.png").unlink()
    story = app.state.saves.load_story(save["id"], turn["id"])
    rebuilt = AgentRunner._provider_messages(messages, story)
    rebuilt_image = next(message for message in rebuilt if isinstance(message.get("content"), list))
    assert rebuilt_image["content"][1]["image_url"]["url"].endswith(persisted["data"])
    image_part["output"] = "{}"
    with pytest.raises(RuntimeError, match="缺少图片数据"):
        AgentRunner._provider_messages(messages, story)
    assert _redact_image_data(f"provider echoed data:image/png;base64,{persisted['data']}") == (
        "provider echoed data:image/png;base64,[omitted]"
    )

    serialized_events = json.dumps([event.model_dump() for event in events], ensure_ascii=False)
    assert persisted["data"] not in serialized_events
    image_completed = next(event for event in events if event.type == "tool.completed" and event.payload.get("tool") == "image_read")
    assert "data" not in image_completed.payload["result"]
    image_requests = [event for event in events if event.type == "model.request" and event.payload["agent"] == "character_designer"]
    assert any("data:image/png;base64,[omitted]" in json.dumps(event.payload) for event in image_requests)


class UploadedPortraitFlowModel(ImageCharacterFlowModel):
    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        if system_prompt in {COORDINATOR_PROMPT, CHARACTER_DESIGNER_PROMPT}:
            references = [
                json.loads(block["text"])
                for message in messages
                if message["role"] == "user" and isinstance(message["content"], list)
                for block in message["content"]
                if block["type"] == "text" and block["text"].startswith("{")
            ]
            reference = next(item for item in references if item.get("source") == "player")
            self.image_path = reference["path"]
            assert reference["image_id"] in self.image_path
            assert "character_portrait_generate" not in {tool["function"]["name"] for tool in tools or []}
        result = await super().complete(system_prompt, messages, tools)
        for call in result.tool_calls:
            if call.name == "character_create":
                call.arguments["profile"] = {**PROFILE, "portrait": self.image_path}
        return result


@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_uploaded_image_can_be_reused_as_portrait_without_generation(tmp_path: Path, image_format: str) -> None:
    model = UploadedPortraitFlowModel()
    output = BytesIO()
    with Image.open(BytesIO(PNG_BYTES)) as image:
        image.save(output, format=image_format)
    model.image_data = output.getvalue()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "原图立绘"}).json()
        response = client.post(
            f"/api/saves/{save['id']}/turns",
            data={"content": "请直接用上传的图片作为丹朱的立绘，不要重新绘制。"},
            files=[("images", ("玩家原图", model.image_data, f"image/{image_format.lower()}"))],
        )
        assert response.status_code == 202
        turn = wait_for_turn(client, save["id"])
        assert turn["status"] == "completed"
        assert turn["generated_images"] == []
        assert client.get(turn["images"][0]["content_url"]).content == model.image_data

    state_dir = app.state.saves.saves_dir / save["id"] / "current"
    _, body = markdown_frontmatter(
        (state_dir / "world_snapshot" / "entities" / "character" / "丹朱" / "ENTITY.md").read_text(encoding="utf-8"),
        "角色",
    )
    portrait = json.loads(body)["portrait"]
    assert portrait == model.image_path
    assert (state_dir / portrait).read_bytes() == model.image_data
    assert list((state_dir / "artifacts").iterdir()) == [state_dir / portrait]
    assert model.image_observations == 3
    assert not any(
        event.type == "tool.completed" and event.payload.get("tool") == "character_portrait_generate"
        for event in app.state.saves.list_events(save["id"])
    )


class ImageRetryFlowModel:
    def __init__(self) -> None:
        self.failed_after_image = False
        self.failed_root = False
        self.image_observations = 0
        self.provider = "deepseek"

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
        del tools
        completed = [message for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            if not completed:
                return self._result(ToolCall(
                    id="retry-character-task",
                    name="task",
                    arguments={"agent": "character_designer", "task": "读取 artifacts/重试.png 创建角色"},
                ))
            last = json.loads(completed[-1]["content"])
            if completed[-1].get("_tool_error"):
                if not self.failed_root:
                    self.failed_root = True
                    raise RuntimeError("模拟 root 在 child 失败后中断")
                return self._result(ToolCall(
                    id="continue-character-task",
                    name="task",
                    arguments={
                        "agent": "character_designer",
                        "task_id": last["task_id"],
                        "task": "继续完成同一图片建角任务",
                    },
                ))
            if last.get("agent") == "character_designer":
                return self._result(ToolCall(
                    id="retry-narrator-task",
                    name="task",
                    arguments={"agent": "narrator", "task": "确认角色建立"},
                ))
            return self._result(ToolCall(id="retry-publish", name="narrative_publish", arguments={"path": "narrative.md"}))
        if system_prompt == CHARACTER_DESIGNER_PROMPT:
            image_results = [message for message in completed if isinstance(message["content"], list)]
            if not completed:
                return self._result(ToolCall(id="retry-image-read", name="image_read", arguments={"path": "artifacts/重试.png"}))
            assert len(image_results) == 1
            self.image_observations += 1
            if not self.failed_after_image:
                self.failed_after_image = True
                raise RuntimeError(
                    "模拟图片读取后的模型失败 data:image/png;base64,"
                    + base64.b64encode(PNG_BYTES).decode("ascii")
                )
            if len(completed) == 1:
                return self._result(ToolCall(
                    id="retry-create-character",
                    name="character_create",
                    arguments={"name": "赭羽", "description": "从持久化图片观察创建", "aliases": [], "profile": PROFILE},
                ))
            return self._result(ToolCall(
                id="retry-character-report",
                name="character_report",
                arguments={"report": "从恢复的图片观察创建了赭羽。", "related_entities": ["entities/character/赭羽/ENTITY.md"]},
            ))
        if system_prompt == NARRATOR_PROMPT:
            if not completed:
                return self._result(ToolCall(id="retry-story", name="file_write", arguments={"path": "narrative.md", "content": "赭羽已经建立。"}))
            return ModelResult(content="草稿完成。", raw_response={}, tool_calls=[])
        raise AssertionError(f"unexpected prompt: {system_prompt[:80]}")

    @staticmethod
    def _result(call: ToolCall) -> ModelResult:
        return ModelResult(content="", raw_response={}, tool_calls=[call])


def test_image_observation_survives_failed_turn_retry_without_source_file(tmp_path: Path) -> None:
    model = ImageRetryFlowModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "图片恢复"}).json()
        image = app.state.saves.saves_dir / save["id"] / "current" / "artifacts" / "重试.png"
        image.write_bytes(PNG_BYTES)
        response = client.post(
            f"/api/saves/{save['id']}/turns",
            json={"content": "根据 artifacts/重试.png 创建角色。"},
        )
        assert response.status_code == 202
        failed = wait_for_turn(client, save["id"])
        assert failed["status"] == "failed"
        encoded = base64.b64encode(PNG_BYTES).decode("ascii")
        root_messages = app.state.saves.load_session(
            save["id"], app.state.saves.root_session_id(save["id"], failed["id"])
        )
        assert encoded not in json.dumps(root_messages, ensure_ascii=False)
        image.unlink()
        model.provider = "anthropic"

        retry = client.post(f"/api/saves/{save['id']}/turns/{failed['id']}/retry")
        assert retry.status_code == 202
        completed = wait_for_turn(client, save["id"])

    assert completed["status"] == "completed"
    assert completed["id"] == failed["id"]
    assert model.image_observations == 3
    assert (app.state.saves.saves_dir / save["id"] / "current" / "world_snapshot" / "entities" / "character" / "赭羽" / "ENTITY.md").is_file()
