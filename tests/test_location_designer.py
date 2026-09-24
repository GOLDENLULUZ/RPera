from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.content import markdown_frontmatter
from rpera.models import ModelResult, SaveCreate, ToolCall
from tests.prompt_fixtures import COORDINATOR_PROMPT, LOCATION_DESIGNER_PROMPT, NARRATOR_PROMPT
from tests.helpers import make_data_dir, wait_for_turn


PROFILE = {
    "type": "宫殿",
    "parent_location": "王都",
    "feature": {
        "fixed_decorations": ["正门上方嵌着金色日轮纹章"],
        "furniture": ["觐见厅中央铺着深红长毯"],
        "unnamed_npcs": ["正门常驻四名王家卫兵"],
    },
    "模型自行扩展": {"保留": True},
}


def test_save_location_operations_use_real_snapshot_files_and_preserve_markdown(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    source_location = data_dir / "worlds" / "雾港" / "entities" / "location" / "暮潮旅店" / "ENTITY.md"
    original_source = source_location.read_text(encoding="utf-8")
    mod_location = data_dir / "mods" / "读心大师" / "entities" / "location" / "静思室"
    mod_location.mkdir(parents=True)
    mod_location_file = mod_location / "ENTITY.md"
    mod_location_file.write_text(original_source, encoding="utf-8")
    app = create_app(data_dir=data_dir)
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="地点文件", mod_names=["读心大师"]))

    created = app.state.saves.create_location(save.id, "日轮宫", "王都中心的王室宫殿", ["王宫"], PROFILE)
    assert created["path"] == "entities/location/日轮宫/ENTITY.md"
    saved_path = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / created["path"]
    front, body = markdown_frontmatter(saved_path.read_text(encoding="utf-8"), "地点")
    assert front == {
        "description": "王都中心的王室宫殿",
        "aliases": ["王宫"],
        "visibility": "world_truth",
        "required": False,
    }
    assert json.loads(body) == PROFILE

    with pytest.raises(ValueError, match="实体名称已存在"):
        app.state.saves.create_location(save.id, "伊蕾", "冲突", [], PROFILE)

    existing_path = "entities/location/暮潮旅店/ENTITY.md"
    snapshot_file = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / existing_path
    before = snapshot_file.read_text(encoding="utf-8")
    edited = app.state.saves.edit_location(save.id, existing_path, "几张盐渍木桌", "六张盐渍木桌")
    assert edited["content"].startswith("# 暮潮旅店")
    assert snapshot_file.read_text(encoding="utf-8") == before.replace("几张盐渍木桌", "六张盐渍木桌", 1)
    renamed = app.state.saves.rename_location(save.id, existing_path, "暮潮酒馆")
    assert renamed["path"] == "entities/location/暮潮酒馆/ENTITY.md"

    mod_path = "mods/读心大师/entities/location/静思室/ENTITY.md"
    mod_edited = app.state.saves.edit_location(save.id, mod_path, "老板娘伊蕾", "静思室看守人")
    assert mod_edited["description"].startswith("靠近旧码头")
    mod_renamed = app.state.saves.rename_location(save.id, mod_path, "内心静室")
    assert mod_renamed["path"] == "mods/读心大师/entities/location/内心静室/ENTITY.md"
    assert mod_location_file.read_text(encoding="utf-8") == original_source
    assert source_location.read_text(encoding="utf-8") == original_source


def test_location_json_edit_is_local_and_must_remain_an_object(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="地点局部编辑"))
    created = app.state.saves.create_location(save.id, "日轮宫", "王都中心的王室宫殿", [], PROFILE)
    path = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / created["path"]
    before = path.read_text(encoding="utf-8")

    edited = app.state.saves.edit_location(save.id, created["path"], "正门常驻四名王家卫兵", "正门常驻六名王家卫兵")
    after = path.read_text(encoding="utf-8")
    assert after == before.replace("正门常驻四名王家卫兵", "正门常驻六名王家卫兵", 1)
    assert json.loads(edited["content"])["模型自行扩展"] == {"保留": True}

    with pytest.raises(ValueError, match="唯一出现一次"):
        app.state.saves.edit_location(save.id, created["path"], "  ", " ")
    with pytest.raises(ValueError, match="合法 JSON 对象"):
        app.state.saves.edit_location(save.id, created["path"], '"type": "宫殿"', '"type":')
    with pytest.raises(ValueError, match="不允许修改 visibility 或 required"):
        app.state.saves.edit_location(save.id, created["path"], "required: false", "required: true")
    assert path.read_text(encoding="utf-8") == after


class LocationFlowModel:
    def __init__(self) -> None:
        self.location_input: dict[str, Any] | None = None
        self.narrator_input: dict[str, Any] | None = None

    def bind(self):
        return self

    def bind_for_agents(self, names: tuple[str, ...]):
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": "test", "model": "location-flow", "base_url": "local", "timeout_seconds": 1}

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
                call = ToolCall(id="location-task", name="task", arguments={"agent": "location_designer", "task": "创建一座长期使用的王室宫殿"})
            elif len(completed) == 1:
                result = json.loads(completed[0]["content"])
                call = ToolCall(
                    id="narrator-task",
                    name="task",
                    arguments={
                        "agent": "narrator",
                        "task": "让玩家抵达新宫殿",
                        "related_entities": result["result"]["related_entities"],
                        "report_refs": [result["report_ref"]],
                    },
                )
            else:
                call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
            return self._result(call)
        if system_prompt == LOCATION_DESIGNER_PROMPT:
            if not completed:
                self.location_input = json.loads(messages[1]["content"])
                return self._result(ToolCall(
                    id="create-location",
                    name="location_create",
                    arguments={"name": "日轮宫", "description": "王都中心的王室宫殿", "aliases": ["王宫"], "profile": PROFILE},
                ))
            if len(completed) == 1:
                location = json.loads(completed[0]["content"])["location"]
                assert "content" not in location
                assert json.loads(markdown_frontmatter(location["document"], "地点")[1]) == PROFILE
                return self._result(ToolCall(
                    id="rename-location",
                    name="location_rename",
                    arguments={"path": "entities/location/日轮宫/ENTITY.md", "new_name": "曜日宫"},
                ))
            if len(completed) == 2:
                return self._result(ToolCall(
                    id="edit-location",
                    name="location_edit",
                    arguments={
                        "path": "entities/location/曜日宫/ENTITY.md",
                        "old_text": "正门常驻四名王家卫兵",
                        "new_text": "正门常驻六名王家卫兵",
                    },
                ))
            return self._result(ToolCall(
                id="report-location",
                name="location_report",
                arguments={"report": "创建并完善了长期地点曜日宫。", "related_entities": ["entities/location/曜日宫/ENTITY.md"]},
            ))
        if system_prompt == NARRATOR_PROMPT:
            if not completed:
                self.narrator_input = json.loads(messages[1]["content"])
                return self._result(ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "曜日宫的金色纹章在你面前亮起。"}))
            return ModelResult(content="草稿完成。", raw_response={}, tool_calls=[])
        raise AssertionError(f"unexpected prompt: {system_prompt[:80]}")

    @staticmethod
    def _result(call: ToolCall) -> ModelResult:
        return ModelResult(content="", raw_response={}, tool_calls=[call])


def test_location_report_is_brief_while_runtime_hydrates_real_entity_for_narrator(tmp_path: Path) -> None:
    model = LocationFlowModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "地点闭环", "mod_names": ["读心大师"]}).json()
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "我前往王都的宫殿。"})
        assert response.status_code == 202
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert model.location_input is not None
    assert model.location_input["related_entity_documents"][0]["name"] == "读心能力"
    assert model.narrator_input is not None
    assert model.narrator_input["report_documents"][0]["content"] == "创建并完善了长期地点曜日宫。"
    entity = model.narrator_input["related_entity_documents"][0]
    assert entity["name"] == "曜日宫"
    assert entity["aliases"] == ["王宫"]
    assert "document" not in entity
    assert json.loads(entity["content"])["feature"]["unnamed_npcs"] == ["正门常驻六名王家卫兵"]
