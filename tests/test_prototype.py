from __future__ import annotations

import json
import shutil
import uuid

import pytest
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rpera.anthropic import request_payload as anthropic_request_payload
from rpera.app import PROJECT_ROOT, create_app
from rpera.config import PresetStore, RuntimeSettingsStore
from rpera.content import content_name, content_name_key, entity_snapshot_path
from rpera.entities import EntityEntry, EntityStore, search_entry_list
from rpera.gemini import request_payload as gemini_request_payload
from rpera.models import AgentPresetSettingsWrite, AiPreset, ModelListRequest, ModelResult, PresetWrite, RuntimeSettings, SaveCreate, ToolCall
from rpera.model_errors import ModelFallbackError
from rpera.providers import BoundModelClient, ConfiguredModelClient, FallbackModelClient
from rpera.runtime import AgentRunner
from tests.prompt_fixtures import COORDINATOR_PROMPT, NARRATOR_PROMPT, TASK_DESCRIPTIONS, WORLD_RESEARCHER_PROMPT
from rpera.storage import WorldLibrary
from rpera.xai import request_payload as xai_request_payload
from tests.helpers import make_data_dir, make_symlink, wait_for_turn


class ScriptedModelClient:
    def __init__(self) -> None:
        self.coordinator_calls = 0
        self.research_calls = 0
        self.narrator_payload: dict[str, Any] | None = None

    def bind(self):
        return self

    def bind_for_agents(self, names: tuple[str, ...]):
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": "test", "model": "scripted", "base_url": "local", "timeout_seconds": 1}

    def research_action(self, research_calls: int) -> ToolCall | None:
        if research_calls == 1:
            return ToolCall(id="research-search", name="entity_search", arguments={"query": "暮潮旅店 老板 异乡人"})
        if research_calls == 2:
            return ToolCall(
                id="research-read",
                name="entity_read",
                arguments={
                    "paths": [
                        "entities/location/暮潮旅店/ENTITY.md",
                        "entities/character/伊蕾/ENTITY.md",
                    ]
                },
            )
        return None

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        del tools
        if system_prompt == COORDINATOR_PROMPT:
            self.coordinator_calls += 1
            completed = [message for message in messages if message["role"] == "tool"]
            if not completed:
                calls = [ToolCall(id="delegate-research", name="task", arguments={"agent": "world_researcher", "task": "查明旅店中与灯塔失踪有关的线索", "reason": "玩家行动依赖当地人物和已知事实"})]
                content = ""
            elif len(completed) == 1:
                report = json.loads(completed[0]["content"])["result"]
                calls = [ToolCall(id="delegate-narrator", name="task", arguments={"agent": "narrator", "task": "描写玩家向伊蕾打听守灯人的交锋，让伊蕾先观察玩家的真实意图。", "reason": "伊蕾寡言而警惕；不得透露奥伦仍然活着或封蜡信。", "related_entities": report["related_entities"]})]
                content = ""
            elif len(completed) == 2:
                calls = [ToolCall(id="read-narrative", name="file_read", arguments={"path": "narrative.md"})]
                content = ""
            else:
                calls = [ToolCall(id="publish-narrative", name="narrative_publish", arguments={"path": "narrative.md"})]
                content = ""
        elif system_prompt == WORLD_RESEARCHER_PROMPT:
            self.research_calls += 1
            call = self.research_action(self.research_calls)
            if call is None:
                initial = json.loads(messages[1]["content"])
                paths = [item["path"] for item in initial.get("required_world_entities", [])]
                for message in messages:
                    if message["role"] != "tool":
                        continue
                    result = json.loads(message["content"])
                    paths.extend(item["path"] for item in result.get("documents", []))
                call = ToolCall(id="research-report", name="research_report", arguments={"report": "调查资料已经整理。", "related_entities": list(dict.fromkeys(paths))})
            calls = [call]
            content = ""
        elif system_prompt == NARRATOR_PROMPT:
            self.narrator_payload = json.loads(messages[1]["content"])
            completed = [message for message in messages if message["role"] == "tool"]
            if not completed:
                content = ""
                calls = [ToolCall(id="write-narrative", name="file_write", arguments={"path": "narrative.md", "content": "伊蕾擦拭着一只空杯，没有立刻回答。她抬眼打量你：‘你为什么关心那个守灯人？’"})]
            else:
                content = "叙事草稿已完成。"
                calls = []
        else:
            raise AssertionError(f"Unexpected prompt: {system_prompt[:80]}")
        return ModelResult(
            content=content,
            reasoning="test reasoning",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
            raw_response={"choices": [{"message": {"content": content}}]},
            tool_calls=calls,
        )


class FailOnceNarratorClient(ScriptedModelClient):
    def __init__(self) -> None:
        super().__init__()
        self.narrator_attempts = 0

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        if system_prompt == NARRATOR_PROMPT:
            self.narrator_attempts += 1
            if self.narrator_attempts == 1:
                raise RuntimeError("temporary narrator failure")
        return await super().complete(system_prompt, messages, tools)


class FailOnceResearchClient(ScriptedModelClient):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        if system_prompt == WORLD_RESEARCHER_PROMPT and not self.failed_once and self.research_calls >= 1:
            self.failed_once = True
            raise RuntimeError("temporary research failure")
        return await super().complete(system_prompt, messages, tools)


def test_default_user_data_is_in_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("RPERA_DATA_DIR", raising=False)
    monkeypatch.delenv("RPERA_USER_DATA_DIR", raising=False)

    app = create_app(model_client=ScriptedModelClient())

    assert app.state.content_data_dir == PROJECT_ROOT / "data"
    assert app.state.user_data_dir == PROJECT_ROOT
    assert app.state.presets.path == PROJECT_ROOT / "config" / "ai_presets.json"
    assert app.state.saves.saves_dir == PROJECT_ROOT / "saves"

    custom_user_data = tmp_path / "custom-user-data"
    custom_user_data.mkdir(mode=0o750)
    custom_user_data.chmod(0o750)
    permissions = custom_user_data.stat().st_mode
    monkeypatch.setenv("RPERA_USER_DATA_DIR", str(custom_user_data))
    overridden = create_app(model_client=ScriptedModelClient())
    assert overridden.state.user_data_dir == custom_user_data
    assert overridden.state.saves.saves_dir == custom_user_data / "saves"
    assert custom_user_data.stat().st_mode == permissions


def test_complete_agent_flow_and_save_isolation(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        worlds = client.get("/api/worlds").json()
        assert worlds[0]["name"] == "雾港"
        assert worlds[0]["description"].startswith("一座被长雾")
        assert "language" not in worlds[0]
        assert "id" not in worlds[0]
        assert "version" not in worlds[0]

        first = client.post(
            "/api/saves",
            json={"world_name": "雾港", "name": "第一条时间线"},
        ).json()
        second = client.post(
            "/api/saves",
            json={"world_name": "雾港", "name": "第二条时间线"},
        ).json()
        assert first["id"] != second["id"]
        assert "source_world_version" not in first
        assert "source_world_version" not in json.loads(
            (data_dir / "saves" / first["id"] / "save.json").read_text(encoding="utf-8")
        )
        assert first["opening"].startswith("黄昏的夜钟")
        assert not (data_dir / "saves" / first["id"] / "current" / "world_snapshot" / "manifest.json").exists()
        assert (data_dir / "saves" / first["id"] / "current" / "world_snapshot" / "scenarios" / "失踪的守灯人.md").is_file()
        assert (data_dir / "saves" / first["id"] / "current" / "world_snapshot" / "entities" / "character" / "伊蕾" / "ENTITY.md").is_file()
        entities = client.get(f"/api/saves/{first['id']}/entities").json()
        assert {entry["path"] for entry in entities} == {
            "entities/character/伊蕾/ENTITY.md",
            "entities/character/奥伦/ENTITY.md",
            "entities/event/失踪的守灯人/ENTITY.md",
            "entities/location/暮潮旅店/ENTITY.md",
        }
        assert all("type" not in entry and "source_kind" not in entry and "source_name" not in entry for entry in entities)

        response = client.post(
            f"/api/saves/{first['id']}/turns",
            json={"content": "我问老板娘是否见过守灯人。"},
        )
        assert response.status_code == 202
        turn = wait_for_turn(client, first["id"])

        assert turn["status"] == "completed"
        assert "为什么关心那个守灯人" in turn["narrative"]
        assert client.get(f"/api/saves/{second['id']}/turns").json() == []

        events = app.state.saves.list_events(first["id"])
        event_types = [event.type for event in events]
        assert "task.created" in event_types
        assert "tool.completed" in event_types
        assert "entity.search" in event_types
        assert "entity.read" in event_types
        assert "model.request" in event_types
        assert "model.response" in event_types
        assert "turn.completed" in event_types
        model_response = next(event for event in events if event.type == "model.response")
        assert model_response.payload["reasoning"] == "test reasoning"
        assert model_response.payload["raw_response"]["choices"]
        started = next(event for event in events if event.type == "turn.started")
        assert started.payload["delegation_budget"] == {"max_delegations": 20}
        research_task = next(event for event in events if event.type == "task.created")
        assert research_task.payload["initiator_agent"] == "coordinator"
        assert research_task.payload["target_agent"] == "world_researcher"
        assert research_task.payload["reason"] == "玩家行动依赖当地人物和已知事实"
        assert research_task.payload["instruction_blocked"] is False
        read_started = next(
            event for event in events
            if event.type == "tool.started" and event.payload.get("tool") == "entity_read"
        )
        assert json.loads(read_started.payload["input"])["paths"] == [
            "entities/location/暮潮旅店/ENTITY.md",
            "entities/character/伊蕾/ENTITY.md",
        ]
        trace_events = app.state.saves.list_trace_page(first["id"]).events
        read_reply = next(
            event for event in trace_events
            if event.type == "tool.completed" and event.payload.get("tool") == "entity_read"
        )
        assert [document["path"] for document in read_reply.payload["result"]["documents"]] == [
            "entities/location/暮潮旅店/ENTITY.md",
            "entities/character/伊蕾/ENTITY.md",
        ]
        narrator_task = next(
            event for event in trace_events
            if event.type == "task.created" and event.payload["target_agent"] == "narrator"
        )
        assert [entity["path"] for entity in narrator_task.payload["related_entities"]] == [
            "entities/location/暮潮旅店/ENTITY.md",
            "entities/character/伊蕾/ENTITY.md",
        ]
        assert all("content" not in entity for entity in narrator_task.payload["related_entities"])
        coordinator_request = next(
            event for event in events if event.type == "model.request" and event.payload["agent"] == "coordinator"
        )
        task_tool = next(tool["function"] for tool in coordinator_request.payload["tools"] if tool["function"]["name"] == "task")
        child_names = [agent.name for agent in app.state.runtime.runner.registry.children()]
        assert task_tool["parameters"]["properties"]["agent"]["enum"] == child_names
        assert "继续同一子代理会话" in task_tool["parameters"]["properties"]["task_id"]["description"]
        assert all(
            agent.name in task_tool["description"]
            and TASK_DESCRIPTIONS[agent.name] in task_tool["description"]
            for agent in app.state.runtime.runner.registry.children()
        )
        assert "coordinator" not in task_tool["description"]
        expected_story = {
            "story": {
                "opening": first["opening"],
                "turns": [{
                    "turn_number": 1,
                    "player": {"content": "我问老板娘是否见过守灯人。", "images": []},
                    "has_ai_output": False,
                }],
            }
        }
        assert coordinator_request.payload["messages"] == [
            {"role": "system", "content": "<recent_story>"},
            {"role": "assistant", "content": first["opening"]},
            {"role": "user", "content": "我问老板娘是否见过守灯人。"},
            {"role": "system", "content": "</recent_story>"},
        ]
        researcher_request = next(
            event for event in events if event.type == "model.request" and event.payload["agent"] == "world_researcher"
        )
        assert {tool["function"]["name"] for tool in researcher_request.payload["tools"]} == {"entity_search", "entity_read", "report_read", "research_report", "story_summary_read"}
        researcher_story = json.loads(researcher_request.payload["messages"][0]["content"])
        assert researcher_story == expected_story
        assert any(message["role"] == "tool" for message in researcher_request.payload["messages"][1:]) is False
        coordinator_requests = [event for event in events if event.type == "model.request" and event.payload["agent"] == "coordinator"]
        assert coordinator_requests[1].payload["messages"][-1]["role"] == "tool"
        scripted_model = app.state.runtime.model
        assert scripted_model.narrator_payload["task"]["task"].startswith("描写玩家")
        assert [item["path"] for item in scripted_model.narrator_payload["related_entity_documents"]] == [
            "entities/location/暮潮旅店/ENTITY.md",
            "entities/character/伊蕾/ENTITY.md",
        ]
        read_event = next(event for event in events if event.type == "entity.read")
        assert [doc["path"] for doc in read_event.payload["documents"]] == [
            "entities/location/暮潮旅店/ENTITY.md",
            "entities/character/伊蕾/ENTITY.md",
        ]
        assert all("id" not in doc and "source_id" not in doc for doc in read_event.payload["documents"])
        assert read_event.payload["retrieval_counter"] == {
            "total_searchable_entities": 4,
            "read_entities": 2,
        }
        assert "禁止将100%阅读作为参考指标" in read_event.payload["retrieval_counter_warning"]


def test_directory_and_markdown_are_content_fact_sources(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    scenario_dir = data_dir / "worlds" / "空白世界" / "scenarios"
    scenario_dir.mkdir(parents=True)
    (scenario_dir / "直接开始.md").write_text("直接从这里开始。\n", encoding="utf-8")

    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        assert client.get("/api/worlds").json() == [
            {
                "name": "空白世界",
                "description": "",
                "scenarios": [{"name": "直接开始", "description": ""}],
            }
        ]
        save = client.post(
            "/api/saves",
            json={
                "world_name": "空白世界",
                "name": "直接开始",
                "scenario": {"source_kind": "world", "source_name": "空白世界", "name": "直接开始"},
            },
        ).json()

    assert save["opening"] == "直接从这里开始。"


def test_selecting_second_scenario_uses_its_opening_as_first_context(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        world = client.get("/api/worlds").json()[0]
        assert [scenario["name"] for scenario in world["scenarios"]] == ["失踪的守灯人", "雾中空艇"]
        assert all("id" not in scenario for scenario in world["scenarios"])
        save = client.post(
            "/api/saves",
            json={
                "world_name": "雾港",
                "name": "空艇开局",
                "scenario": {"source_kind": "world", "source_name": "雾港", "name": "雾中空艇"},
            },
        ).json()
        assert save["scenario"] == {"source_kind": "world", "source_name": "雾港", "name": "雾中空艇"}
        assert save["opening"].startswith("晨雾压在旧码头")

        client.post(f"/api/saves/{save['id']}/turns", json={"content": "我查看那艘小艇。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    events = app.state.saves.list_events(save["id"])
    coordinator_request = next(
        event for event in events if event.type == "model.request" and event.payload["agent"] == "coordinator"
    )
    assert coordinator_request.payload["messages"] == [
        {"role": "system", "content": "<recent_story>"},
        {"role": "assistant", "content": save["opening"]},
        {"role": "user", "content": "我查看那艘小艇。"},
        {"role": "system", "content": "</recent_story>"},
    ]


def test_empty_scenario_starts_without_assistant_history(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    model = ScriptedModelClient()
    app = create_app(data_dir=data_dir, model_client=model)

    with TestClient(app) as client:
        assert client.post("/api/content/worlds", json={"name": "空序章世界"}).status_code == 201
        assert client.post("/api/content/worlds/空序章世界/scenarios", json={"name": "直接开始"}).status_code == 201
        save = client.post(
            "/api/saves",
            json={
                "world_name": "空序章世界",
                "name": "无序章冒险",
                "scenario": {"source_kind": "world", "source_name": "空序章世界", "name": "直接开始"},
            },
        ).json()
        assert save["opening"] == ""
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "迈出第一步。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    coordinator_request = next(
        event
        for event in app.state.saves.list_events(save["id"])
        if event.type == "model.request" and event.payload["agent"] == "coordinator"
    )
    assert coordinator_request.payload["messages"] == [
        {"role": "system", "content": "<recent_story>"},
        {"role": "user", "content": "迈出第一步。"},
        {"role": "system", "content": "</recent_story>"},
    ]
    assert model.narrator_payload is not None
    narrator_request = next(
        event
        for event in app.state.saves.list_events(save["id"])
        if event.type == "model.request" and event.payload["agent"] == "narrator"
    )
    assert json.loads(narrator_request.payload["messages"][0]["content"])["story"] == {
        "opening": "",
        "turns": [{
            "turn_number": 1,
            "player": {"content": "迈出第一步。", "images": []},
            "has_ai_output": False,
        }],
    }


def test_mod_snapshot_combines_scenarios_and_required_entities(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    model = RequiredEntityClient()
    app = create_app(data_dir=data_dir, model_client=model)
    with TestClient(app) as client:
        mod_page = client.get("/api/mods").json()
        assert [mod["name"] for mod in mod_page["mods"]] == ["读心大师"]
        assert "id" not in mod_page["mods"][0]
        assert "version" not in mod_page["mods"][0]
        scenarios = client.get("/api/worlds/雾港/scenarios?mod_name=读心大师").json()
        assert [(scenario["source_kind"], scenario["source_name"], scenario["name"]) for scenario in scenarios] == [
            ("world", "雾港", "失踪的守灯人"),
            ("world", "雾港", "雾中空艇"),
            ("mod", "读心大师", "第一道心声"),
        ]
        assert all("id" not in scenario and "source_id" not in scenario for scenario in scenarios)
        save = client.post(
            "/api/saves",
            json={
                "world_name": "雾港",
                "name": "读心者",
                "mod_names": ["读心大师"],
                "scenario": {"source_kind": "mod", "source_name": "读心大师", "name": "第一道心声"},
            },
        ).json()
        assert save["mods"][0]["name"] == "读心大师"
        assert "id" not in save["mods"][0]
        assert "version" not in save["mods"][0]
        assert save["scenario"] == {"source_kind": "mod", "source_name": "读心大师", "name": "第一道心声"}
        assert save["opening"].startswith("清晨的雾")

        mod_entity_path = "mods/读心大师/entities/player_trait/读心能力/ENTITY.md"
        assert (data_dir / "saves" / save["id"] / "current" / "world_snapshot" / "mods" / "读心大师" / "description.md").is_file()
        assert app.state.saves.required_entity_paths(save["id"]) == [mod_entity_path]
        assert mod_entity_path not in [
            entry["path"]
            for entry in app.state.saves.list_entities(save["id"])
            if not entry["required"]
        ]
        shutil.rmtree(data_dir / "mods" / "读心大师")
        assert app.state.saves.required_entity_paths(save["id"]) == [mod_entity_path]

        client.post(f"/api/saves/{save['id']}/turns", json={"content": "我凝神倾听四周的心声。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    events = app.state.saves.list_events(save["id"])
    coordinator_request = next(
        event for event in events if event.type == "model.request" and event.payload["agent"] == "coordinator"
    )
    coordinator_context = json.dumps(coordinator_request.payload["messages"], ensure_ascii=False)
    assert "required_world_entities_instruction" not in coordinator_context
    assert "required_world_entities" not in coordinator_context
    assert model.researcher_payload is not None
    assert [entity["path"] for entity in model.researcher_payload["required_world_entities"]] == [mod_entity_path]


def test_entity_search_keeps_name_alias_and_description_scores() -> None:
    entries = [
        EntityEntry(path="entities/character/伊蕾/ENTITY.md", name="伊蕾"),
        EntityEntry(path="entities/character/老板娘/ENTITY.md", name="老板娘", aliases=["伊蕾"]),
        EntityEntry(path="entities/character/伊蕾的姐姐/ENTITY.md", name="伊蕾的姐姐"),
        EntityEntry(
            path="entities/character/走私者/ENTITY.md",
            name="走私者",
            description="伊蕾的秘密联系人",
        ),
    ]

    assert [entry.path for entry in search_entry_list(entries, "伊蕾")] == [
        "entities/character/伊蕾/ENTITY.md",
        "entities/character/老板娘/ENTITY.md",
        "entities/character/伊蕾的姐姐/ENTITY.md",
        "entities/character/走私者/ENTITY.md",
    ]


def test_same_named_entities_from_different_sources_stay_independent(tmp_path: Path) -> None:
    data_dir, world_name = make_required_world(tmp_path)
    crime_mod = data_dir / "mods" / "犯罪安娜"
    (crime_mod / "entities" / "character" / "安娜").mkdir(parents=True)
    (crime_mod / "entities" / "character" / "安娜" / "ENTITY.md").write_text(
        "---\n"
        "description: 安娜暗中从事走私犯罪活动。\n"
        "---\n"
        "# 安娜的犯罪活动\n"
        "她在夜间安排走私货物。\n",
        encoding="utf-8",
    )
    class ModEntityClient(ScriptedModelClient):
        def research_action(self, research_calls: int) -> ToolCall | None:
            if research_calls == 1:
                return ToolCall(id="research-search", name="entity_search", arguments={"query": "安娜"})
            if research_calls == 2:
                return ToolCall(id="research-read", name="entity_read", arguments={"paths": ["mods/犯罪安娜/entities/character/安娜/ENTITY.md"]})
            return None

    app = create_app(data_dir=data_dir, model_client=ModEntityClient())
    save = app.state.saves.create_save(SaveCreate(world_name=world_name, name="犯罪资料", mod_names=["犯罪安娜"]))

    matches = app.state.saves.search_entities(save.id, "安娜")
    assert {entry["path"] for entry in matches if entry["name"] == "安娜"} == {
        "entities/character/安娜/ENTITY.md",
        "mods/犯罪安娜/entities/character/安娜/ENTITY.md",
    }
    assert all("id" not in entry and "source_id" not in entry for entry in matches)
    documents = app.state.saves.read_entities(
        save.id,
        ["mods/犯罪安娜/entities/character/安娜/ENTITY.md"],
    )
    assert len(documents) == 1
    assert documents[0]["content"].startswith("# 安娜的犯罪活动")
    assert "id" not in documents[0] and "source_id" not in documents[0]

    with TestClient(app) as client:
        client.post(f"/api/saves/{save.id}/turns", json={"content": "我调查安娜。"})
        assert wait_for_turn(client, save.id)["status"] == "completed"
    events = app.state.saves.list_events(save.id)
    search_event = next(event for event in events if event.type == "entity.search")
    read_event = next(event for event in events if event.type == "entity.read")
    assert search_event.payload["retrieval_counter"] == {
        "total_searchable_entities": 2,
        "read_entities": 0,
    }
    assert read_event.payload["retrieval_counter"] == {
        "total_searchable_entities": 2,
        "read_entities": 1,
    }


def test_saves_are_ordered_by_recent_activity_and_paginated(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        created = [
            client.post("/api/saves", json={"world_name": "雾港", "name": f"存档 {number}"}).json()
            for number in range(1, 12)
        ]
        for number, save in enumerate(created, start=1):
            save_path = data_dir / "saves" / save["id"] / "save.json"
            save_data = json.loads(save_path.read_text(encoding="utf-8"))
            save_data["last_played_at"] = f"2026-01-01T00:{number:02}:00+00:00"
            save_path.write_text(json.dumps(save_data, ensure_ascii=False), encoding="utf-8")

        first_page = client.get("/api/saves").json()
        assert first_page["page"] == 1
        assert first_page["page_size"] == 10
        assert first_page["total"] == 11
        assert first_page["total_pages"] == 2
        assert [save["name"] for save in first_page["saves"]] == [f"存档 {number}" for number in range(11, 1, -1)]

        second_page = client.get("/api/saves?page=2").json()
        assert second_page["page"] == 2
        assert [save["name"] for save in second_page["saves"]] == ["存档 1"]

        assert client.get("/api/saves?page=99").json()["page"] == 2
        assert client.get("/api/saves?page=0").status_code == 422


def test_save_rename_updates_only_name_and_rejects_running_turn(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        original = client.post("/api/saves", json={"world_name": "雾港", "name": "旧名称"}).json()
        other = client.post("/api/saves", json={"world_name": "雾港", "name": "另一个存档"}).json()
        path = f"/api/saves/{original['id']}"

        for name in ["", "   ", "x" * 81]:
            assert client.post(f"{path}/rename", json={"name": name}).status_code == 422
        assert client.post("/api/saves/not-a-uuid/rename", json={"name": "新名称"}).status_code == 404
        assert client.get(path).json() == original

        turn = app.state.saves.create_turn(original["id"], "开始冒险", 4)
        before = client.get(path).json()
        blocked = client.post(f"{path}/rename", json={"name": "运行中改名"})
        assert blocked.status_code == 409
        assert "回合正在执行" in blocked.json()["detail"]
        assert client.get(path).json() == before

        app.state.saves.fail_turn(original["id"], turn.id, "turn.interrupted", {"message": "测试中止"})
        response = client.post(f"{path}/rename", json={"name": "  新名称  "})
        assert response.status_code == 200
        assert response.json() == {**before, "name": "新名称"}
        assert client.get(path).json() == response.json()
        assert any(save["id"] == original["id"] and save["name"] == "新名称" for save in client.get("/api/saves").json()["saves"])
        assert json.loads((data_dir / "saves" / original["id"] / "save.json").read_text(encoding="utf-8")) == response.json()
        assert client.get(f"/api/saves/{other['id']}").json() == other
        assert client.get(f"{path}/turns").json()[0]["id"] == turn.id


def test_event_history_is_paginated_and_defaults_to_the_last_page(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "事件分页"}).json()
        for number in range(1, 122):
            app.state.saves.add_event(save["id"], None, "test.event", {"number": number})

        last_page = client.get(f"/api/saves/{save['id']}/events/history").json()
        assert last_page["page"] == 3
        assert last_page["page_size"] == 50
        assert last_page["total"] == 121
        assert last_page["total_pages"] == 3
        assert [event["payload"]["number"] for event in last_page["events"]] == list(range(101, 122))

        first_page = client.get(f"/api/saves/{save['id']}/events/history?page=1").json()
        assert [event["payload"]["number"] for event in first_page["events"]] == list(range(1, 51))
        second_page = client.get(f"/api/saves/{save['id']}/events/history?page=2").json()
        assert [event["payload"]["number"] for event in second_page["events"]] == list(range(51, 101))
        assert client.get(f"/api/saves/{save['id']}/events/history?page=999").json()["page"] == 3
        assert client.get(f"/api/saves/{save['id']}/events/history?page=0").status_code == 422

        empty = client.post("/api/saves", json={"world_name": "雾港", "name": "空事件"}).json()
        empty_page = client.get(f"/api/saves/{empty['id']}/events/history").json()
        assert empty_page == {"events": [], "page": 1, "page_size": 50, "total": 0, "total_pages": 0}
        assert client.get("/api/saves/unknown/events/history").status_code == 404


def test_trace_history_only_pages_visible_actions(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "可读轨迹"}).json()
        for number in range(1, 52):
            app.state.saves.add_event(
                save["id"],
                None,
                "tool.started",
                {"agent": "world_researcher", "tool": "entity_search", "input": json.dumps({"query": str(number)})},
            )
        app.state.saves.add_event(save["id"], None, "model.request", {"agent": "coordinator", "messages": ["hidden"]})
        app.state.saves.add_event(save["id"], None, "tool.started", {"agent": "coordinator", "tool": "task", "input": "{}"})
        app.state.saves.add_event(
            save["id"],
            None,
            "model.response",
            {"agent": "coordinator", "content": "", "tool_calls": [{"name": "task"}], "raw_response": {"hidden": True}},
        )
        app.state.saves.add_event(
            save["id"],
            None,
            "tool.failed",
            {
                "agent": "coordinator",
                "tool": "task",
                "input": json.dumps({"agent": "narrator", "task": "无效委派"}),
                "result": {"ok": False, "error": {"code": "validation_error", "message": "委派失败"}},
            },
        )
        last_raw = app.state.saves.add_event(save["id"], None, "entity.search", {"results": ["hidden"]})

        last_page = client.get(f"/api/saves/{save['id']}/trace/history").json()
        assert last_page["page"] == 2
        assert last_page["total"] == 52
        assert last_page["total_pages"] == 2
        assert len(last_page["events"]) == 2
        assert last_page["events"][0]["payload"] == {
            "agent": "world_researcher",
            "tool": "entity_search",
            "input": json.dumps({"query": "51"}),
        }
        assert last_page["events"][1]["type"] == "tool.failed"
        assert last_page["events"][1]["payload"]["tool"] == "task"
        assert last_page["cursor"] == last_raw.id

        first_page = client.get(f"/api/saves/{save['id']}/trace/history?page=1").json()
        assert len(first_page["events"]) == 50
        assert client.get("/api/saves/unknown/trace/history").status_code == 404


def test_delete_save_removes_it_and_clamps_pagination(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        created = [
            client.post("/api/saves", json={"world_name": "雾港", "name": f"存档 {number}"}).json()
            for number in range(1, 12)
        ]
        deleted = created[-1]
        save_path = data_dir / "saves" / deleted["id"] / "save.json"
        save_data = json.loads(save_path.read_text(encoding="utf-8"))
        save_data["last_played_at"] = "2020-01-01T00:00:00+00:00"
        save_path.write_text(json.dumps(save_data, ensure_ascii=False), encoding="utf-8")

        response = client.delete(f"/api/saves/{deleted['id']}")

        assert response.status_code == 200
        assert response.json() == {"id": deleted["id"], "deleted": True}
        assert not (data_dir / "saves" / deleted["id"]).exists()
        assert client.get(f"/api/saves/{deleted['id']}").status_code == 404
        assert client.get(f"/api/saves/{deleted['id']}/turns").status_code == 404
        saves = client.get("/api/saves?page=2").json()
        assert saves["page"] == 1
        assert saves["total"] == 10


def test_delete_save_rejects_invalid_and_running_but_closes_subscribers(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        assert client.delete("/api/saves/not-a-uuid").status_code == 404
        assert client.delete(f"/api/saves/{uuid.uuid4()}").status_code == 404
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "删除测试"}).json()
        app.state.saves.create_turn(save["id"], "保持运行", 4)

        running = client.delete(f"/api/saves/{save['id']}")

        assert running.status_code == 409
        assert (data_dir / "saves" / save["id"]).is_dir()

        app.state.saves.fail_turn(
            save["id"],
            app.state.saves.list_turns(save["id"])[0].id,
            "turn.interrupted",
            {"message": "测试结束"},
        )
        subscription = app.state.events.subscribe(save["id"])
        deleted = client.delete(f"/api/saves/{save['id']}")

        assert deleted.status_code == 200
        assert not (data_dir / "saves" / save["id"]).exists()
        assert subscription.get_nowait() is True


def preset_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "name": "DeepSeek 主配置",
        "provider": "deepseek",
        "base_url": "https://ignored.example",
        "api_key": "secret-value",
        "model": "deepseek-chat",
        "timeout_seconds": 90,
        "temperature": 0.8,
        "top_p": 0.9,
        "max_tokens": 4096,
    }
    payload.update(overrides)
    return payload


def test_modern_large_output_limits_are_accepted() -> None:
    deepseek_sized = PresetWrite.model_validate(preset_payload(max_tokens=393_216))
    assert deepseek_sized.max_tokens == 393_216
    assert AiPreset(id="default", name="默认").max_tokens == 65_536
    xai = PresetWrite.model_validate(preset_payload(provider="xai", xai_protocol="chat_completions"))
    assert xai.xai_protocol == "chat_completions"
    assert AiPreset(id="xai", name="Grok", provider="xai").xai_protocol == "responses"


def test_runtime_settings_are_global_and_unbounded_above(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        assert client.get("/api/runtime-settings").json() == {"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "prefill_agents": [], "blocked_instruction_agents": [], "always_attach_report_agents": [], "force_publish_narrative": False, "force_start_delegation": False, "block_coordinator_narrative_read": False, "use_compliance_fixed_response": False}
        assert client.put("/api/runtime-settings", json={"max_delegations": 3, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": []}).status_code == 422
        updated = client.put("/api/runtime-settings", json={"max_delegations": 100_000, "context_turns": 7, "disabled_agents": ["role_player"], "prefill_agents": ["coordinator", "role_player"], "blocked_instruction_agents": ["role_player", "narrator"], "always_attach_report_agents": ["role_player", "narrator"], "force_publish_narrative": True, "force_start_delegation": True, "block_coordinator_narrative_read": True, "use_compliance_fixed_response": True}).json()
        assert updated == {"max_delegations": 100_000, "context_turns": 7, "disabled_agents": ["role_player"], "prefill_agents": ["coordinator", "role_player"], "blocked_instruction_agents": ["role_player", "narrator"], "always_attach_report_agents": ["role_player", "narrator"], "force_publish_narrative": True, "force_start_delegation": True, "block_coordinator_narrative_read": True, "use_compliance_fixed_response": True}
    settings = RuntimeSettings.model_validate_json(
        (data_dir / "config" / "runtime_settings.json").read_text(encoding="utf-8")
    )
    assert settings.max_delegations == 100_000
    assert settings.context_turns == 7
    assert settings.disabled_agents == ["role_player"]
    assert settings.prefill_agents == ["coordinator", "role_player"]
    assert settings.blocked_instruction_agents == ["role_player", "narrator"]
    assert settings.always_attach_report_agents == ["role_player", "narrator"]
    assert settings.force_publish_narrative is True
    assert settings.force_start_delegation is True
    assert settings.block_coordinator_narrative_read is True
    assert settings.use_compliance_fixed_response is True


def test_runtime_settings_require_integer_context_turns_at_least_two(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ScriptedModelClient())
    payload = {"max_delegations": 20, "disabled_agents": [], "prefill_agents": [], "blocked_instruction_agents": []}
    with TestClient(app) as client:
        for value in (None, 1, 2.5, "2"):
            request = dict(payload)
            if value is not None:
                request["context_turns"] = value
            assert client.put("/api/runtime-settings", json=request).status_code == 422

        response = client.put("/api/runtime-settings", json={**payload, "context_turns": 2})
        assert response.status_code == 200
        assert response.json()["context_turns"] == 2


def test_runtime_settings_reject_agents_that_cannot_be_disabled(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ScriptedModelClient())
    with TestClient(app) as client:
        for agent in ("coordinator", "narrator", "missing"):
            response = client.put(
                "/api/runtime-settings",
                json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [agent], "blocked_instruction_agents": []},
            )
            assert response.status_code == 422
            assert "不存在或不能禁用" in response.json()["detail"]

        duplicate = client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": ["role_player", "role_player"], "blocked_instruction_agents": []},
        )
        assert duplicate.status_code == 422
        assert "不能重复" in duplicate.json()["detail"]

        invalid_prefill = client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "prefill_agents": ["missing"], "blocked_instruction_agents": []},
        )
        assert invalid_prefill.status_code == 422
        assert "尾部续写 Agent 不存在" in invalid_prefill.json()["detail"]

        duplicate_prefill = client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "prefill_agents": ["coordinator", "coordinator"], "blocked_instruction_agents": []},
        )
        assert duplicate_prefill.status_code == 422
        assert "尾部续写 Agent 不能重复" in duplicate_prefill.json()["detail"]

        for blocked_agents, message in ((["coordinator"], "不存在"), (["missing"], "不存在"), (["narrator", "narrator"], "不能重复")):
            blocked = client.put(
                "/api/runtime-settings",
                json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": blocked_agents},
            )
            assert blocked.status_code == 422
            assert message in blocked.json()["detail"]

        for report_agents, message in ((["coordinator"], "不存在或不生成报告"), (["missing"], "不存在或不生成报告"), (["narrator", "narrator"], "不能重复")):
            forced = client.put(
                "/api/runtime-settings",
                json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": [], "always_attach_report_agents": report_agents},
            )
            assert forced.status_code == 422
            assert message in forced.json()["detail"]


def test_runtime_settings_do_not_migrate_old_shape(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    config_dir = data_dir / "config"
    config_dir.mkdir()
    path = config_dir / "runtime_settings.json"
    old_content = json.dumps({"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "prefill_agents": []})
    path.write_text(old_content, encoding="utf-8")

    with pytest.raises(ValidationError, match="blocked_instruction_agents"):
        RuntimeSettingsStore(data_dir).get()

    assert path.read_text(encoding="utf-8") == old_content


def test_runtime_settings_default_missing_report_attachment_list(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    config_dir = data_dir / "config"
    config_dir.mkdir()
    path = config_dir / "runtime_settings.json"
    existing_content = json.dumps({
        "max_delegations": 20,
        "context_turns": 4,
        "disabled_agents": [],
        "prefill_agents": [],
        "blocked_instruction_agents": [],
    })
    path.write_text(existing_content, encoding="utf-8")

    settings = RuntimeSettingsStore(data_dir).get()

    assert settings.always_attach_report_agents == []
    assert settings.force_publish_narrative is False
    assert settings.force_start_delegation is False
    assert settings.block_coordinator_narrative_read is False
    assert settings.use_compliance_fixed_response is False
    assert path.read_text(encoding="utf-8") == existing_content


def test_ai_preset_crud_and_secret_rules(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        initial = client.get("/api/ai-presets").json()
        default_id = initial["main_preset_id"]
        response = client.put(f"/api/ai-presets/{default_id}", json=preset_payload())
        assert response.status_code == 200
        public = response.json()["presets"][0]
        assert public["base_url"] == "https://ignored.example"
        assert public["xai_protocol"] == "responses"
        assert public["temperature"] == 0.8
        assert public["has_api_key"] is True
        assert public["masked_api_key"] == "sec***lue"
        assert "secret-value" not in response.text
        preset_file = data_dir / "config" / "ai_presets.json"
        assert "secret-value" in preset_file.read_text(encoding="utf-8")

        created = client.post(
            "/api/ai-presets",
            json=preset_payload(
                name="兼容接口",
                provider="openai_compatible",
                base_url="https://provider.example/v1",
                api_key="second-secret",
                model="custom-model",
            ),
        ).json()
        second_id = next(preset["id"] for preset in created["presets"] if preset["name"] == "兼容接口")
        assigned = client.put(
            "/api/agent-preset-settings",
            json={"main_preset_id": second_id, "agent_preset_overrides": {}},
        ).json()
        assert assigned["main_preset_id"] == second_id

        duplicated = client.post(
            f"/api/ai-presets/{second_id}/duplicate",
            json={"name": "兼容接口副本"},
        ).json()
        duplicate_id = next(preset["id"] for preset in duplicated["presets"] if preset["name"] == "兼容接口副本")
        assert client.delete(f"/api/ai-presets/{duplicate_id}").status_code == 200

        changed = client.put(
            f"/api/ai-presets/{second_id}",
            json=preset_payload(
                name="新端点",
                provider="openai_compatible",
                base_url="https://new-provider.example/v1",
                api_key="",
                model="other-model",
            ),
        ).json()
        changed_preset = next(preset for preset in changed["presets"] if preset["id"] == second_id)
        assert changed_preset["has_api_key"] is True
        assert "second-secret" in preset_file.read_text(encoding="utf-8")

        switched = client.put(
            f"/api/ai-presets/{second_id}",
            json=preset_payload(name="切换供应商", provider="deepseek", api_key=""),
        ).json()
        switched_preset = next(preset for preset in switched["presets"] if preset["id"] == second_id)
        assert switched_preset["has_api_key"] is False
        assert "second-secret" not in preset_file.read_text(encoding="utf-8")


def test_model_list_preset_uses_unsaved_settings_and_preserves_matching_key(tmp_path: Path) -> None:
    store = PresetStore(make_data_dir(tmp_path))
    preset_id = store.public().main_preset_id
    store.update(preset_id, PresetWrite.model_validate(preset_payload(api_key="saved-secret")))

    unsaved = store.model_list_preset(ModelListRequest(
        provider="openai_compatible",
        base_url="https://new.example/v1/",
        api_key="new-secret",
        timeout_seconds=45,
    ))
    existing = store.model_list_preset(ModelListRequest(
        preset_id=preset_id,
        provider="deepseek",
        base_url="https://edited.example/",
        timeout_seconds=60,
    ))
    switched = store.model_list_preset(ModelListRequest(
        preset_id=preset_id,
        provider="anthropic",
        base_url="https://api.anthropic.com/v1",
    ))

    assert unsaved.id == "unsaved"
    assert unsaved.base_url == "https://new.example/v1"
    assert unsaved.api_key == "new-secret"
    assert unsaved.timeout_seconds == 45
    assert existing.api_key == "saved-secret"
    assert existing.base_url == "https://edited.example"
    assert switched.api_key == ""


def test_unsaved_preset_can_list_models_without_being_persisted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[AiPreset] = []

    class ModelListingClient:
        def __init__(self, preset: AiPreset, **_kwargs: Any) -> None:
            captured.append(preset)

        async def list_models(self) -> list[dict[str, str]]:
            return [{"id": "available-model", "owned_by": "local"}]

    data_dir = make_data_dir(tmp_path)
    monkeypatch.setattr("rpera.app.BoundModelClient", ModelListingClient)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        response = client.post("/api/ai-presets/models", json={
            "preset_id": None,
            "provider": "openai_compatible",
            "base_url": "https://provider.example/v1",
            "api_key": "draft-secret",
            "xai_protocol": "responses",
            "timeout_seconds": 30,
        })

    assert response.status_code == 200
    assert response.json() == [{"id": "available-model", "owned_by": "local"}]
    assert captured[0].model == ""
    assert captured[0].api_key == "draft-secret"
    assert not (data_dir / "config" / "ai_presets.json").exists()


def test_agent_preset_settings_inherit_override_and_remap_deleted_preset(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        initial = client.get("/api/ai-presets").json()
        first_id = initial["main_preset_id"]
        created = client.post("/api/ai-presets", json=preset_payload(name="研究模型")).json()
        second_id = next(preset["id"] for preset in created["presets"] if preset["name"] == "研究模型")

        catalog = client.get("/api/agent-preset-settings").json()
        assert [agent["name"] for agent in catalog["agents"]] == [
            "coordinator", "compliance_reviewer", "story_summarizer", "world_researcher", "character_designer", "location_designer", "EroticOrNot", "goal_keeper", "role_player", "style_planner", "narrator", "consistency_checker"
        ]
        assert {agent["name"] for agent in catalog["agents"]} == {
            "coordinator", "story_summarizer", "world_researcher", "character_designer", "location_designer", "compliance_reviewer", "EroticOrNot", "goal_keeper", "role_player", "style_planner", "narrator", "consistency_checker"
        }
        assert {agent["name"] for agent in catalog["agents"] if agent["can_disable"]} == {
            "story_summarizer", "world_researcher", "character_designer", "location_designer", "compliance_reviewer", "EroticOrNot", "goal_keeper", "role_player", "style_planner", "consistency_checker"
        }
        assert {agent["name"] for agent in catalog["agents"] if agent["supports_fixed_response"]} == {"compliance_reviewer"}
        assert {agent["name"] for agent in catalog["agents"] if agent["produces_reports"]} == {
            "story_summarizer", "world_researcher", "character_designer", "location_designer", "compliance_reviewer", "EroticOrNot", "role_player", "style_planner", "narrator", "consistency_checker"
        }
        assert all("template_order" not in agent for agent in catalog["agents"])
        assert catalog["agent_preset_overrides"] == {}

        updated = client.put(
            "/api/agent-preset-settings",
            json={
                "main_preset_id": first_id,
                "agent_preset_overrides": {"world_researcher": second_id, "compliance_reviewer": second_id},
                "agent_streaming": {"compliance_reviewer": True},
            },
        ).json()
        assert updated["agent_preset_overrides"] == {"world_researcher": second_id, "compliance_reviewer": second_id}
        assert updated["agent_streaming"] == {"compliance_reviewer": True}
        resolved = app.state.presets.resolve(("coordinator", "world_researcher", "compliance_reviewer", "narrator"))
        assert resolved["coordinator"].id == first_id
        assert resolved["world_researcher"].id == second_id
        assert resolved["compliance_reviewer"].id == second_id
        assert resolved["narrator"].id == first_id

        deleted = client.delete(f"/api/ai-presets/{second_id}").json()
        assert deleted["agent_preset_overrides"] == {"world_researcher": first_id, "compliance_reviewer": first_id}
        assert deleted["agent_streaming"] == {"compliance_reviewer": True}


def test_shared_ai_fallback_settings_and_deleted_preset(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ScriptedModelClient())
    with TestClient(app) as client:
        first_id = client.get("/api/ai-presets").json()["main_preset_id"]
        assert client.get("/api/ai-fallback-settings").json() == {"enabled": False, "preset_id": None}
        assert client.put("/api/ai-fallback-settings", json={"enabled": True, "preset_id": None}).status_code == 422
        # A fallback may intentionally be the same preset as the preferred model.
        assert client.put("/api/ai-fallback-settings", json={"enabled": True, "preset_id": first_id}).json() == {
            "enabled": True, "preset_id": first_id,
        }
        bound = app.state.presets.resolve_with_fallback(("coordinator", "narrator"))
        assert all(primary.id == fallback.id == first_id for primary, fallback, _streaming in bound.values())
        clients = ConfiguredModelClient(app.state.presets).bind_for_agents(("coordinator", "narrator"))
        assert all(isinstance(model, FallbackModelClient) for model in clients.values())
        created = client.post("/api/ai-presets", json=preset_payload(name="备用模型")).json()
        backup_id = next(preset["id"] for preset in created["presets"] if preset["name"] == "备用模型")
        assert client.put("/api/ai-fallback-settings", json={"enabled": True, "preset_id": backup_id}).status_code == 200
        assert client.delete(f"/api/ai-presets/{backup_id}").json()["fallback_enabled"] is False
        assert client.get("/api/ai-fallback-settings").json() == {"enabled": False, "preset_id": None}
        assert all(not isinstance(model, FallbackModelClient) for model in ConfiguredModelClient(app.state.presets).bind_for_agents(("coordinator", "narrator")).values())


def test_model_fallback_keeps_completed_tools_and_actual_model_metadata(tmp_path: Path) -> None:
    class Primary(ScriptedModelClient):
        def __init__(self) -> None:
            super().__init__()
            self.blocked = False

        def metadata(self):
            return {"provider": "test", "model": "primary", "preset_name": "首选"}

        async def complete(self, system_prompt, messages, tools=None):
            if system_prompt == WORLD_RESEARCHER_PROMPT and self.research_calls == 1 and not self.blocked:
                self.blocked = True
                raise ModelFallbackError("content_blocked", "PROHIBITED_CONTENT")
            return await super().complete(system_prompt, messages, tools)

    class Backup:
        def __init__(self, primary):
            self.primary = primary
            self.calls = 0

        def metadata(self):
            return {"provider": "test", "model": "backup", "preset_name": "备用"}

        async def complete(self, system_prompt, messages, tools=None):
            self.calls += 1
            return await ScriptedModelClient.complete(self.primary, system_prompt, messages, tools)

    class Routed:
        def __init__(self):
            self.primary = Primary()
            self.backup = Backup(self.primary)

        def bind_for_agents(self, names):
            return {name: FallbackModelClient(self.primary, self.backup) for name in names}

    model = Routed()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "备用测试"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "打听灯塔"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"
        assert model.backup.calls == 1
        assert model.primary.research_calls == 3
        events = app.state.saves.list_events(save["id"])
        fallback = next(event for event in events if event.type == "model.fallback")
        assert fallback.payload["reason"] == "content_blocked"
        requests = [event for event in events if event.type == "model.request" and event.payload["message_id"] == fallback.payload["message_id"]]
        assert [event.payload["model"]["model"] for event in requests] == ["primary", "backup"]
        with app.state.saves.connect(save["id"]) as db:
            row = db.execute("SELECT model FROM messages WHERE id = ?", (fallback.payload["message_id"],)).fetchone()
        assert json.loads(row["model"])["model"] == "backup"
        assert len([event for event in events if event.type == "tool.completed" and event.payload["tool"] == "entity_search"]) == 1


def test_cross_provider_fallback_does_not_replay_foreign_native_messages() -> None:
    story = {"story": {"opening": "", "turns": [{"turn_number": 1, "player": {"content": "继续", "images": []}, "has_ai_output": False}]}}
    messages = [
        {"role": "user", "model": None, "parts": [{"type": "text", "content": json.dumps({"task": {"task": "任务"}}, ensure_ascii=False)}]},
        {"role": "assistant", "model": json.dumps({
            "provider": "google_gemini", "xai_protocol": "responses",
            "provider_content": {"role": "model", "parts": [{"functionCall": {"name": "entity_search", "args": {"query": "伊蕾"}}, "thoughtSignature": "native-signature"}]},
        }), "parts": [{"type": "tool", "content": None, "provider_call_id": "gemini-1", "tool_name": "entity_search", "input": '{"query":"伊蕾"}', "state": "completed", "output": '{"documents":[]}'}]},
    ]
    primary = AgentRunner._provider_messages(messages, story, target_model={"provider": "google_gemini", "xai_protocol": "responses"})
    backup = AgentRunner._provider_messages(messages, story, target_model={"provider": "anthropic", "xai_protocol": "responses"})
    assert primary[2]["_provider_content"]["parts"][0]["thoughtSignature"] == "native-signature"
    assert "_provider_content" not in backup[2]
    assert backup[2]["tool_calls"][0]["function"]["name"] == "entity_search"
    assert backup[3]["role"] == "tool"


def test_failed_fallback_stops_after_one_additional_request(tmp_path: Path) -> None:
    class Refusing(ScriptedModelClient):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        async def complete(self, system_prompt, messages, tools=None):
            self.calls += 1
            raise ModelFallbackError("content_blocked", "PROHIBITED_CONTENT")

    class Unavailable(Refusing):
        async def complete(self, system_prompt, messages, tools=None):
            self.calls += 1
            raise RuntimeError("备用模型故障")

    class Routed:
        def __init__(self):
            self.primary = Refusing()
            self.backup = Unavailable()

        def bind_for_agents(self, names):
            return {name: FallbackModelClient(self.primary, self.backup) for name in names}

    model = Routed()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "单次备用"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续"})
        assert wait_for_turn(client, save["id"])["status"] == "failed"
    assert model.primary.calls == model.backup.calls == 1
    events = app.state.saves.list_events(save["id"])
    failure = next(event for event in events if event.type == "turn.failed")
    assert "备用模型故障" in failure.payload["message"]
    assert len([event for event in events if event.type == "model.fallback"]) == 1


def test_preset_collection_rejects_stale_v1_schema(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    config_dir = data_dir / "config"
    config_dir.mkdir()
    preset_id = str(uuid.uuid4())
    path = config_dir / "ai_presets.json"
    stale_content = json.dumps({"schema_version": 1, "active_preset_id": preset_id, "presets": [AiPreset(id=preset_id, name="旧配置").model_dump()]})
    path.write_text(stale_content, encoding="utf-8")

    with pytest.raises(ValidationError):
        PresetStore(data_dir).public()

    assert path.read_text(encoding="utf-8") == stale_content


def test_home_page_exposes_main_views_and_static_assets(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ScriptedModelClient())
    with TestClient(app) as client:
        response = client.get("/")
        javascript = client.get("/static/app.js").text
        trace_markdown = client.get("/static/trace-markdown.js")
        stylesheet = client.get("/static/app.css").text
    assert response.status_code == 200
    assert "[ 前台 ]" in response.text
    assert "[ 后台 ]" in response.text
    assert "[ 创作 ]" in response.text
    assert "[ 设置 ]" in response.text
    assert "/static/app.css?v=" in response.text
    assert "/static/app.js?v=" in response.text
    assert "/static/trace-markdown.js?v=" in response.text
    assert trace_markdown.status_code == 200
    assert "RPeraTraceMarkdown" in trace_markdown.text
    assert 'id="image-input"' in response.text
    assert 'id="pending-images"' in response.text
    assert 'id="image-preview-dialog"' in response.text
    assert 'id="toggle-style-enabled-button"' in response.text
    assert response.text.count('value="cancel" type="submit" formnovalidate') == 8
    assert 'api("/api/ai-presets/models"' in javascript
    assert "请先保存 Preset，再拉取模型列表" not in javascript
    assert "new FormData()" in javascript
    assert 'body.append("images"' in javascript
    assert '"文风已禁用"' in javascript
    assert '"文风已启用"' in javascript
    assert "object-fit: contain" in stylesheet
    assert "width: 160px; height: 160px" in stylesheet
    assert "width: 112px; height: 112px" in stylesheet
    assert response.headers["cache-control"] == "no-cache"


def test_legacy_settings_are_not_migrated_and_bound_preset_is_frozen(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    config_dir = data_dir / "config"
    config_dir.mkdir()
    legacy = {
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com",
        "api_key": "legacy-key",
        "model": "deepseek-chat",
        "timeout_seconds": 75,
    }
    legacy_path = config_dir / "settings.json"
    legacy_content = json.dumps(legacy)
    legacy_path.write_text(legacy_content, encoding="utf-8")
    store = PresetStore(data_dir)
    collection = store.public()
    assert collection.presets[0].name == "默认配置"
    assert collection.presets[0].has_api_key is False
    assert collection.presets[0].masked_api_key == ""
    assert legacy_path.read_text(encoding="utf-8") == legacy_content
    preset_content = (config_dir / "ai_presets.json").read_text(encoding="utf-8")
    assert "legacy-key" not in preset_content
    assert "schema_version" not in preset_content

    client = ConfiguredModelClient(store)
    bound = client.bind_for_agents(("coordinator",))["coordinator"]
    original_metadata = bound.metadata()
    store.create(PresetWrite.model_validate(preset_payload(name="第二配置", api_key="new-key")))
    second_id = next(item.id for item in store.public().presets if item.name == "第二配置")
    store.update_agent_settings(AgentPresetSettingsWrite(main_preset_id=second_id))
    assert bound.metadata() == original_metadata
    assert client.bind_for_agents(("coordinator",))["coordinator"].metadata()["preset_name"] == "第二配置"


async def test_model_listing_and_connection_test() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["authorization"] == "Bearer test-key"
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200,
                json={"data": [{"id": "model-b"}, {"id": "model-a", "owned_by": "local"}]},
            )
        payload = json.loads(request.content)
        content = "test-key" if payload["max_tokens"] > 8 else "OK"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content, "reasoning_content": content}}],
                "usage": {"total_tokens": 2},
                "echo": content,
            },
        )

    preset = AiPreset(
        id="preset-id",
        name="测试",
        provider="openai_compatible",
        base_url="https://provider.example/v1",
        api_key="test-key",
        model="model-a",
        max_tokens=1024,
    )
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    models = await client.list_models()
    result = await client.test_connection()
    completed = await client.complete("system", [{"role": "user", "content": "request"}])
    assert [model.id for model in models] == ["model-a", "model-b"]
    assert result.content == "OK"
    assert completed.content == "test-key"
    assert completed.reasoning == "test-key"
    assert completed.raw_response["echo"] == "test-key"
    assert [request.url.path for request in requests] == [
        "/v1/models",
        "/v1/chat/completions",
        "/v1/chat/completions",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "deepseek"])
async def test_openai_protocol_prefill_and_empty_reasoning(
    provider: Literal["openai_compatible", "deepseek"],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][-1] == {"role": "assistant", "content": "预填开头："}
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "续写内容", "reasoning_content": ""}}]},
        )

    preset = AiPreset(
        id=provider,
        name="预填测试",
        provider=provider,
        base_url="https://provider.example/v1",
        api_key="key",
        model="model",
    )
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
        "system",
        [
            {"role": "user", "content": "请求"},
            {"role": "assistant", "content": None, "_reasoning_content": "不应发送"},
            {"role": "assistant", "content": "预填开头：", "_prefix": True},
        ],
    )

    assert result.content == "续写内容"
    assert result.reasoning is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "deepseek"])
async def test_openai_protocol_preserves_inline_story_system_messages(
    provider: Literal["openai_compatible", "deepseek"],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["messages"] == [
            {"role": "system", "content": "coordinator"},
            {"role": "system", "content": "<recent_story>"},
            {"role": "assistant", "content": "序章"},
            {"role": "user", "content": "行动"},
            {"role": "system", "content": "</recent_story>"},
        ]
        return httpx.Response(200, json={"choices": [{"message": {"content": "完成"}}]})

    preset = AiPreset(
        id=provider,
        name="上下文结构",
        provider=provider,
        base_url="https://provider.example/v1",
        api_key="key",
        model="model",
    )
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
        "coordinator",
        [
            {"role": "system", "content": "<recent_story>"},
            {"role": "assistant", "content": "序章"},
            {"role": "user", "content": "行动"},
            {"role": "system", "content": "</recent_story>"},
        ],
    )

    assert result.content == "完成"


def test_native_providers_downgrade_and_merge_inline_story_system_messages() -> None:
    messages = [
        {"role": "system", "content": "<recent_story>"},
        {"role": "assistant", "content": "序章"},
        {"role": "user", "content": "行动"},
        {"role": "system", "content": "</recent_story>"},
    ]
    anthropic = anthropic_request_payload(
        "coordinator",
        messages,
        None,
        model="claude-test",
        temperature=1,
        top_p=1,
        max_tokens=100,
        stream=False,
    )
    gemini = gemini_request_payload(
        "coordinator",
        messages,
        None,
        temperature=1,
        top_p=1,
        max_tokens=100,
    )
    xai = xai_request_payload(
        "coordinator",
        messages,
        None,
        model="grok-test",
        temperature=1,
        top_p=1,
        max_tokens=100,
        stream=False,
    )

    assert anthropic["messages"] == [
        {"role": "user", "content": "<recent_story>"},
        {"role": "assistant", "content": [{"type": "text", "text": "序章"}]},
        {"role": "user", "content": "行动\n</recent_story>"},
    ]
    assert gemini["contents"] == [
        {"role": "user", "parts": [{"text": "<recent_story>"}]},
        {"role": "model", "parts": [{"text": "序章"}]},
        {"role": "user", "parts": [{"text": "行动\n</recent_story>"}]},
    ]
    assert xai["input"] == [
        {"role": "user", "content": "<recent_story>"},
        {"role": "assistant", "content": "序章"},
        {"role": "user", "content": "行动\n</recent_story>"},
    ]


def test_native_providers_preserve_child_story_and_task_as_separate_text_blocks() -> None:
    messages = [
        {"role": "user", "content": '{"story":{"opening":"","turns":[]}}', "_separate_next_user": True},
        {"role": "user", "content": '{"task":{"task":"开始工作","reason":""}}'},
    ]
    anthropic = anthropic_request_payload(
        "child",
        messages,
        None,
        model="claude-test",
        temperature=1,
        top_p=1,
        max_tokens=100,
        stream=False,
    )
    gemini = gemini_request_payload(
        "child",
        messages,
        None,
        temperature=1,
        top_p=1,
        max_tokens=100,
    )
    xai = xai_request_payload(
        "child",
        messages,
        None,
        model="grok-test",
        temperature=1,
        top_p=1,
        max_tokens=100,
        stream=False,
    )

    assert anthropic["messages"] == [{
        "role": "user",
        "content": [
            {"type": "text", "text": messages[0]["content"]},
            {"type": "text", "text": "\n"},
            {"type": "text", "text": messages[1]["content"]},
        ],
    }]
    assert gemini["contents"] == [{
        "role": "user",
        "parts": [
            {"text": messages[0]["content"]},
            {"text": "\n"},
            {"text": messages[1]["content"]},
        ],
    }]
    assert xai["input"] == [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": messages[0]["content"]},
            {"type": "input_text", "text": "\n"},
            {"type": "input_text", "text": messages[1]["content"]},
        ],
    }]


def test_native_providers_preserve_consecutive_image_only_player_turn_boundaries() -> None:
    first_url = "data:image/png;base64,YQ=="
    second_url = "data:image/png;base64,Yg=="
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": first_url}}],
            "_separate_next_user": True,
        },
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": second_url}}],
        },
    ]
    anthropic = anthropic_request_payload(
        "coordinator",
        messages,
        None,
        model="claude-test",
        temperature=1,
        top_p=1,
        max_tokens=100,
        stream=False,
    )
    gemini = gemini_request_payload(
        "coordinator",
        messages,
        None,
        temperature=1,
        top_p=1,
        max_tokens=100,
    )
    xai = xai_request_payload(
        "coordinator",
        messages,
        None,
        model="grok-test",
        temperature=1,
        top_p=1,
        max_tokens=100,
        stream=False,
    )

    assert [block["type"] for block in anthropic["messages"][0]["content"]] == ["image", "text", "image"]
    assert [next(iter(part)) for part in gemini["contents"][0]["parts"]] == ["inlineData", "text", "inlineData"]
    assert [block["type"] for block in xai["input"][0]["content"]] == ["input_image", "input_text", "input_image"]


def test_empty_opening_native_projection_preserves_each_user_only_turn() -> None:
    story = {
        "story": {
            "opening": "",
            "turns": [
                {
                    "turn_number": number,
                    "player": {"content": f"输入 {number}", "images": []},
                    "has_ai_output": False,
                }
                for number in range(1, 4)
            ],
        }
    }
    messages = AgentRunner._story_messages(story, "", coordinator=True, image_reader=None)
    anthropic = anthropic_request_payload(
        "coordinator",
        messages,
        None,
        model="claude-test",
        temperature=1,
        top_p=1,
        max_tokens=100,
        stream=False,
    )

    content = anthropic["messages"][0]["content"]
    assert [block["text"] for block in content if block["type"] == "text"] == [
        "<recent_story>\n输入 1",
        "\n",
        "输入 2",
        "\n",
        "输入 3\n</recent_story>",
    ]


def test_native_providers_convert_user_and_tool_images() -> None:
    url = "data:image/png;base64,aGVsbG8="
    image = {"type": "image_url", "image_url": {"url": url}}
    messages = [
        {"role": "user", "content": [{"type": "text", "text": "看图"}, image]},
        {"role": "system", "content": "</recent_story>"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "image-call", "type": "function", "function": {"name": "image_read", "arguments": {"path": "artifacts/参考.png"}}}],
        },
        {
            "role": "tool",
            "tool_call_id": "image-call",
            "content": [{"type": "text", "text": '{"path":"artifacts/参考.png"}'}, image],
        },
    ]
    tools = [{"type": "function", "function": {"name": "image_read", "parameters": {"type": "object"}}}]

    anthropic = anthropic_request_payload("system", messages, tools, model="claude", temperature=1, top_p=1, max_tokens=100, stream=False)
    assert anthropic["messages"][0]["content"] == [
        {"type": "text", "text": "看图"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
        {"type": "text", "text": "</recent_story>"},
    ]
    assert anthropic["messages"][2]["content"][0]["content"] == [
        {"type": "text", "text": '{"path":"artifacts/参考.png"}'},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}},
    ]

    gemini = gemini_request_payload("system", messages, tools, temperature=1, top_p=1, max_tokens=100)
    assert gemini["contents"][0]["parts"] == [
        {"text": "看图"},
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
        {"text": "</recent_story>"},
    ]
    assert gemini["contents"][2]["parts"] == [
        {"functionResponse": {"name": "image_read", "response": {"path": "artifacts/参考.png"}, "id": "image-call"}},
        {"inlineData": {"mimeType": "image/png", "data": "aGVsbG8="}},
    ]

    xai = xai_request_payload("system", messages, tools, model="grok", temperature=1, top_p=1, max_tokens=100, stream=False)
    assert xai["input"][0]["content"] == [
        {"type": "input_text", "text": "看图"},
        {"type": "input_image", "image_url": url},
        {"type": "input_text", "text": "</recent_story>"},
    ]
    assert xai["input"][2] == {"type": "function_call_output", "call_id": "image-call", "output": '{"path":"artifacts/参考.png"}'}
    assert xai["input"][3] == {"role": "user", "content": [{"type": "input_image", "image_url": url}]}

    pure_image = [{"role": "user", "content": [image]}]
    anthropic_pure = anthropic_request_payload("system", pure_image, None, model="claude", temperature=1, top_p=1, max_tokens=100, stream=False)
    assert anthropic_pure["messages"][0]["content"][0]["type"] == "image"
    gemini_pure = gemini_request_payload("system", pure_image, None, temperature=1, top_p=1, max_tokens=100)
    assert gemini_pure["contents"][0]["parts"][0]["inlineData"]["mimeType"] == "image/png"
    xai_pure = xai_request_payload("system", pure_image, None, model="grok", temperature=1, top_p=1, max_tokens=100, stream=False)
    assert xai_pure["input"][0]["content"] == [{"type": "input_image", "image_url": url}]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai_compatible", "deepseek", "xai"])
async def test_chat_completion_providers_send_tool_images_as_following_user_message(provider: str) -> None:
    url = "data:image/jpeg;base64,aGVsbG8="

    def handler(request: httpx.Request) -> httpx.Response:
        messages = json.loads(request.content)["messages"]
        assert messages[-2] == {"role": "tool", "tool_call_id": "image-call", "content": '{"path":"artifacts/参考.jpg"}'}
        assert messages[-1] == {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": "完成"}}]})

    preset = AiPreset(
        id=provider,
        name=provider,
        provider=provider,
        api_key="key",
        model="vision-test",
        xai_protocol="chat_completions",
    )
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
        "system",
        [{
            "role": "tool",
            "tool_call_id": "image-call",
            "content": [
                {"type": "text", "text": '{"path":"artifacts/参考.jpg"}'},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }],
    )
    assert result.content == "完成"


@pytest.mark.asyncio
async def test_deepseek_tools_replay_only_persisted_reasoning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        history = json.loads(request.content)["messages"]
        assert history[2]["reasoning_content"] == "先读取草稿。"
        assert history[4]["reasoning_content"] == ""
        assert history[6] == {
            "role": "assistant",
            "content": None,
            "reasoning_content": "继续思考。",
        }
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "完成", "reasoning_content": ""}}]},
        )

    preset = AiPreset(
        id="deepseek",
        name="DeepSeek 工具回放",
        provider="deepseek",
        base_url="https://provider.example/beta",
        api_key="key",
        model="deepseek-flash",
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
        "system",
        [
            {"role": "user", "content": "读取草稿"},
            {
                "role": "assistant",
                "content": None,
                "_reasoning_content": "先读取草稿。",
                "tool_calls": [{
                    "id": "read-1",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "read-1", "content": "{}"},
            {
                "role": "assistant",
                "content": None,
                "_reasoning_content": None,
                "tool_calls": [{
                    "id": "read-2",
                    "type": "function",
                    "function": {"name": "file_read", "arguments": "{}"},
                }],
            },
            {"role": "tool", "tool_call_id": "read-2", "content": "{}"},
            {"role": "assistant", "content": None, "_reasoning_content": "继续思考。"},
        ],
        tools,
    )

    assert result.content == "完成"
    assert result.reasoning is None


@pytest.mark.asyncio
async def test_deepseek_sends_prefill_with_tools_and_surfaces_native_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][-1] == {"role": "assistant", "content": "预填"}
        assert "prefix" not in json.dumps(payload)
        return httpx.Response(
            400,
            json={"error": {"message": "Prefix completion is not supported with tools"}},
        )

    preset = AiPreset(
        id="deepseek",
        name="DeepSeek 组合限制",
        provider="deepseek",
        base_url="https://provider.example/beta",
        api_key="key",
        model="deepseek-flash",
    )
    tools = [{
        "type": "function",
        "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}},
    }]

    with pytest.raises(RuntimeError, match="HTTP 400.*Prefix completion is not supported with tools"):
        await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
            "system",
            [{"role": "user", "content": "请求"}, {"role": "assistant", "content": "预填", "_prefix": True}],
            tools,
        )


@pytest.mark.asyncio
async def test_anthropic_prefill_and_native_tool_round_trip() -> None:
    requests: list[dict[str, Any]] = []
    native_content = [
        {"type": "thinking", "thinking": "先读取。", "signature": "signed-thinking"},
        {"type": "tool_use", "id": "toolu_read", "name": "file_read", "input": {"path": "narrative.md"}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-api-key"] == "anthropic-key"
        assert request.headers["anthropic-version"] == "2023-06-01"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["system"] == "system"
        assert payload["model"] == "claude-test"
        assert payload["max_tokens"] == 2048
        if len(requests) == 1:
            assert payload["messages"][-1] == {
                "role": "assistant",
                "content": [{"type": "text", "text": "预填开头："}],
            }
            assert payload["tools"][0]["input_schema"]["properties"]["path"]["type"] == "string"
            assert payload["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
            return httpx.Response(200, json={"content": native_content, "usage": {"input_tokens": 5, "output_tokens": 4}})
        assert payload["messages"][1] == {"role": "assistant", "content": native_content}
        assert payload["messages"][2] == {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_read", "content": '{"ok": true}'}],
        }
        return httpx.Response(200, json={"content": [{"type": "text", "text": "完成"}], "usage": {"output_tokens": 1}})

    preset = AiPreset(
        id="anthropic",
        name="Claude",
        provider="anthropic",
        base_url="https://api.anthropic.test/v1",
        api_key="anthropic-key",
        model="claude-test",
        max_tokens=2048,
    )
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "file_read", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    first = await client.complete(
        "system",
        [{"role": "user", "content": "读取"}, {"role": "assistant", "content": "预填开头：", "_prefix": True}],
        tools,
    )
    assert first.reasoning == "先读取。"
    assert first.provider_content == {"content": native_content}
    assert first.tool_calls == [ToolCall(id="toolu_read", name="file_read", arguments={"path": "narrative.md"})]

    second = await client.complete(
        "system",
        [
            {"role": "user", "content": "读取"},
            {"role": "assistant", "content": None, "_provider_content": first.provider_content, "tool_calls": [{"id": "toolu_read", "type": "function", "function": {"name": "file_read", "arguments": {"path": "narrative.md"}}}]},
            {"role": "tool", "tool_call_id": "toolu_read", "content": '{"ok": true}'},
        ],
        tools,
    )
    assert second.content == "完成"


@pytest.mark.asyncio
async def test_anthropic_marks_tool_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][-1]["content"][0] == {
            "type": "tool_result",
            "tool_use_id": "failed-call",
            "content": '{"error":"failed"}',
            "is_error": True,
        }
        return httpx.Response(200, json={"content": [{"type": "text", "text": "已收到错误"}]})

    preset = AiPreset(id="anthropic", name="Claude", provider="anthropic", base_url="https://api.anthropic.test/v1", api_key="key", model="claude-test")
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
        "system",
        [{"role": "tool", "tool_call_id": "failed-call", "content": '{"error":"failed"}', "_tool_error": True}],
    )
    assert result.content == "已收到错误"


@pytest.mark.asyncio
async def test_anthropic_stream_and_model_listing() -> None:
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "claude-b", "display_name": "Claude B"}, {"id": "claude-a", "display_name": "Claude A"}], "has_more": False})
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 2}}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "完成"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ]
        body = "".join(f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events).encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    preset = AiPreset(id="anthropic", name="Claude", provider="anthropic", base_url="https://api.anthropic.test/v1", api_key="key", model="claude-test")
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler), streaming=True)
    models = await client.list_models()
    result = await client.complete("system", [{"role": "user", "content": "请求"}])
    assert [model.id for model in models] == ["claude-a", "claude-b"]
    assert result.content == "完成"
    assert result.usage == {"input_tokens": 2, "output_tokens": 1}
    assert requests == ["/v1/models", "/v1/messages"]


@pytest.mark.asyncio
async def test_xai_responses_native_tool_round_trip() -> None:
    requests: list[dict[str, Any]] = []
    native_output = [
        {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "先读取。"}], "encrypted_content": "opaque", "status": "completed"},
        {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "file_read", "arguments": '{"path":"narrative.md"}', "status": "completed"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer xai-key"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["store"] is False
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert payload["parallel_tool_calls"] is False
        if len(requests) == 1:
            assert payload["tools"][0]["name"] == "file_read"
            assert payload["input"][-1] == {"role": "assistant", "content": "预填"}
            return httpx.Response(200, json={"id": "resp_1", "output": native_output, "usage": {"total_tokens": 8}})
        assert payload["input"][1:3] == native_output
        assert payload["input"][3] == {"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'}
        return httpx.Response(200, json={"id": "resp_2", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "完成"}]}]})

    preset = AiPreset(id="xai", name="Grok", provider="xai", base_url="https://api.x.ai/v1", api_key="xai-key", model="grok-test")
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "file_read", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
    first = await client.complete("system", [{"role": "user", "content": "读取"}, {"role": "assistant", "content": "预填", "_prefix": True}], tools)
    assert first.reasoning == "先读取。"
    assert first.provider_content == {"output": native_output}
    assert first.tool_calls == [ToolCall(id="call_1", name="file_read", arguments='{"path":"narrative.md"}')]
    second = await client.complete(
        "system",
        [
            {"role": "user", "content": "读取"},
            {"role": "assistant", "content": None, "_provider_content": first.provider_content, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "file_read", "arguments": '{"path":"narrative.md"}'}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
        ],
        tools,
    )
    assert second.content == "完成"


@pytest.mark.asyncio
async def test_xai_protocol_selector_supports_chat_completions_and_responses_stream() -> None:
    paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        payload = json.loads(request.content)
        if request.url.path.endswith("chat/completions"):
            assert payload["messages"][-1] == {"role": "assistant", "content": "预填"}
            assert payload["parallel_tool_calls"] is False
            return httpx.Response(200, json={"choices": [{"message": {"content": "兼容完成"}}]})
        response = {"id": "resp", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "原生完成"}]}], "usage": {"total_tokens": 2}}
        events = [{"type": "response.completed", "response": response}]
        body = "".join(f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events).encode() + b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    preset = AiPreset(id="xai", name="Grok", provider="xai", base_url="https://api.x.ai/v1", api_key="key", model="grok-test", xai_protocol="chat_completions")
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]
    chat = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete("system", [{"role": "user", "content": "请求"}, {"role": "assistant", "content": "预填", "_prefix": True}], tools)
    responses = await BoundModelClient(preset.model_copy(update={"xai_protocol": "responses"}), transport=httpx.MockTransport(handler), streaming=True).complete("system", [{"role": "user", "content": "请求"}])
    assert chat.content == "兼容完成"
    assert responses.content == "原生完成"
    assert responses.usage == {"total_tokens": 2}
    assert paths == ["/v1/chat/completions", "/v1/responses"]


@pytest.mark.asyncio
async def test_xai_surfaces_incomplete_and_refusal_responses() -> None:
    responses = [
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
        {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "request refused"}]}]},
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    preset = AiPreset(id="xai", name="Grok", provider="xai", base_url="https://api.x.ai/v1", api_key="key", model="grok-test")
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="生成未完成.*max_output_tokens"):
        await client.complete("system", [{"role": "user", "content": "请求"}])
    with pytest.raises(RuntimeError, match="拒绝生成.*request refused"):
        await client.complete("system", [{"role": "user", "content": "请求"}])

    chat = BoundModelClient(preset.model_copy(update={"xai_protocol": "chat_completions"}), transport=httpx.MockTransport(lambda _request: httpx.Response(200, json={"choices": [{"message": {"content": None, "refusal": "chat refused"}}]})))
    with pytest.raises(RuntimeError, match="拒绝生成.*chat refused"):
        await chat.complete("system", [{"role": "user", "content": "请求"}])

    async def stream_handler(_request: httpx.Request) -> httpx.Response:
        event = {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode())

    streaming = BoundModelClient(preset, transport=httpx.MockTransport(stream_handler), streaming=True)
    with pytest.raises(RuntimeError, match="xAI 接口返回错误.*max_output_tokens"):
        await streaming.complete("system", [{"role": "user", "content": "请求"}])


@pytest.mark.asyncio
async def test_gemini_prefill_and_missing_reasoning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["contents"][-1] == {"role": "model", "parts": [{"text": "预填开头："}]}
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "续写内容"}]}, "finishReason": "STOP"}]},
        )

    preset = AiPreset(
        id="gemini",
        name="Gemini 预填测试",
        provider="google_gemini",
        base_url="https://provider.example/v1beta",
        api_key="key",
        model="gemini-test",
    )
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
        "system",
        [{"role": "user", "content": "请求"}, {"role": "assistant", "content": "预填开头："}],
    )

    assert result.content == "续写内容"
    assert result.reasoning is None


@pytest.mark.asyncio
async def test_gemini_native_request_response_and_signed_tool_round_trip() -> None:
    requests: list[dict[str, Any]] = []
    signed_content = {
        "role": "model",
        "parts": [
            {"text": "先检查文件。", "thought": True},
            {
                "functionCall": {"name": "file_read", "args": {"path": "narrative.md"}},
                "thoughtSignature": "opaque-signature",
            },
        ],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-goog-api-key"] == "gemini-key"
        assert "authorization" not in request.headers
        payload = json.loads(request.content)
        requests.append(payload)
        assert request.url.path == "/v1beta/models/gemini-test:generateContent"
        if len(requests) == 1:
            assert payload["systemInstruction"] == {"parts": [{"text": "system"}]}
            assert payload["generationConfig"] == {"temperature": 0.7, "topP": 0.8, "maxOutputTokens": 2048}
            declaration = payload["tools"][0]["functionDeclarations"][0]
            assert declaration["parametersJsonSchema"]["properties"]["path"]["type"] == "string"
            return httpx.Response(
                200,
                json={
                    "candidates": [{"content": signed_content, "finishReason": "STOP"}],
                    "usageMetadata": {"promptTokenCount": 4, "totalTokenCount": 7},
                },
            )
        assert payload["contents"][1] == signed_content
        response = payload["contents"][2]["parts"][0]["functionResponse"]
        assert response == {"name": "file_read", "response": {"ok": True, "content": "story"}}
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "完成"}]}, "finishReason": "STOP"}]},
        )

    preset = AiPreset(
        id="gemini",
        name="Gemini",
        provider="google_gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        api_key="gemini-key",
        model="models/gemini-test",
        temperature=0.7,
        top_p=0.8,
        max_tokens=2048,
    )
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    tools = [{"type": "function", "function": {"name": "file_read", "description": "Read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
    first = await client.complete("system", [{"role": "user", "content": "读取"}], tools)
    assert first.reasoning == "先检查文件。"
    assert first.content == ""
    assert first.usage == {"promptTokenCount": 4, "totalTokenCount": 7}
    assert first.provider_content == signed_content
    assert first.tool_calls[0].name == "file_read"
    assert first.tool_calls[0].arguments == {"path": "narrative.md"}

    second = await client.complete(
        "system",
        [
            {"role": "user", "content": "读取"},
            {
                "role": "assistant",
                "content": None,
                "_provider_content": first.provider_content,
                "tool_calls": [{"id": first.tool_calls[0].id, "type": "function", "function": {"name": "file_read", "arguments": {"path": "narrative.md"}}}],
            },
            {"role": "tool", "tool_call_id": first.tool_calls[0].id, "content": json.dumps({"ok": True, "content": "story"})},
        ],
        tools,
    )
    assert second.content == "完成"


@pytest.mark.asyncio
async def test_gemini_replays_foreign_tool_call_with_skip_signature() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["contents"][1] == {
            "role": "model",
            "parts": [{
                "functionCall": {
                    "name": "task",
                    "args": {"agent": "narrator", "task": "继续写作"},
                    "id": "foreign-call",
                },
                "thoughtSignature": "skip_thought_signature_validator",
            }],
        }
        assert payload["contents"][2] == {
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": "task",
                    "response": {"ok": True},
                    "id": "foreign-call",
                },
            }],
        }
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "完成"}]}, "finishReason": "STOP"}]},
        )

    preset = AiPreset(
        id="gemini",
        name="Gemini",
        provider="google_gemini",
        base_url="https://example.test/v1beta",
        api_key="key",
        model="gemini-test",
    )
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete(
        "system",
        [
            {"role": "user", "content": "继续"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "foreign-call",
                    "type": "function",
                    "function": {
                        "name": "task",
                        "arguments": {"agent": "narrator", "task": "继续写作"},
                    },
                }],
            },
            {"role": "tool", "tool_call_id": "foreign-call", "content": json.dumps({"ok": True})},
        ],
        [{"type": "function", "function": {"name": "task", "parameters": {"type": "object"}}}],
    )

    assert result.content == "完成"


def test_gemini_signed_tool_calls_execute_through_persistent_runtime(tmp_path: Path) -> None:
    story = "Gemini 工具调用写成的故事。"
    signatures_seen: list[str] = []

    def function_response_names(payload: dict[str, Any]) -> list[str]:
        names: list[str] = []
        for content in payload["contents"]:
            for part in content["parts"]:
                function_response = part.get("functionResponse")
                if isinstance(function_response, dict):
                    names.append(function_response["name"])
        return names

    def call(name: str, args: dict[str, Any], signature: str) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "candidates": [{
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": name, "args": args}, "thoughtSignature": signature}],
                    },
                    "finishReason": "STOP",
                }]
            },
        )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        system_prompt = payload["systemInstruction"]["parts"][0]["text"]
        responses = function_response_names(payload)
        if responses:
            model_parts = [content["parts"] for content in payload["contents"] if content["role"] == "model"]
            signature = model_parts[-1][-1].get("thoughtSignature")
            assert isinstance(signature, str)
            signatures_seen.append(signature)
        if system_prompt == COORDINATOR_PROMPT:
            if not responses:
                return call("task", {"agent": "narrator", "task": "写故事"}, "coordinator-task-signature")
            assert responses[-1] == "task"
            return call("narrative_publish", {"path": "narrative.md"}, "publish-signature")
        if system_prompt == NARRATOR_PROMPT:
            if not responses:
                return call("file_write", {"path": "narrative.md", "content": story}, "write-signature")
            assert responses[-1] == "file_write"
            return httpx.Response(200, json={"candidates": [{"content": {"role": "model", "parts": [{"text": "完成"}]}, "finishReason": "STOP"}]})
        raise AssertionError(system_prompt)

    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    model = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "Gemini 工具"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == story
    assert signatures_seen == ["write-signature", "coordinator-task-signature"]
    with app.state.saves.connect(save["id"]) as db:
        stored_models = [json.loads(row["model"]) for row in db.execute("SELECT model FROM messages WHERE role = 'assistant' AND model IS NOT NULL")]
    stored_signatures = [part.get("thoughtSignature") for model_data in stored_models for part in model_data.get("provider_content", {}).get("parts", [])]
    assert "coordinator-task-signature" in stored_signatures
    assert "write-signature" in stored_signatures


@pytest.mark.asyncio
async def test_gemini_stream_finishes_at_eof_and_preserves_parts() -> None:
    chunks = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "思考", "thought": True, "thoughtSignature": "sig"}]}}]},
        {"candidates": [{"content": {"role": "model", "parts": [{"text": "正文"}]}, "finishReason": "STOP"}], "usageMetadata": {"totalTokenCount": 3}},
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models/gemini-test:streamGenerateContent")
        assert request.url.params["alt"] == "sse"
        body = "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks).encode()
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler), streaming=True).complete("system", [{"role": "user", "content": "request"}])
    assert result.content == "正文"
    assert result.reasoning == "思考"
    assert result.usage == {"totalTokenCount": 3}
    assert result.provider_content == {"role": "model", "parts": [*chunks[0]["candidates"][0]["content"]["parts"], *chunks[1]["candidates"][0]["content"]["parts"]]}


@pytest.mark.asyncio
async def test_gemini_model_listing_uses_pagination_and_generation_capability() -> None:
    requested_tokens: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_tokens.append(request.url.params.get("pageToken", ""))
        if len(requested_tokens) == 1:
            return httpx.Response(200, json={"models": [{"name": "models/embed", "supportedGenerationMethods": ["embedContent"]}, {"name": "models/gemini-z", "displayName": "Gemini Z", "supportedGenerationMethods": ["generateContent"]}], "nextPageToken": "next/token"})
        assert request.url.params["pageToken"] == "next/token"
        return httpx.Response(200, json={"models": [{"name": "models/gemini-a", "supportedGenerationMethods": ["generateContent"]}]})

    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    models = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).list_models()
    assert requested_tokens == ["", "next/token"]
    assert [(model.id, model.owned_by) for model in models] == [("gemini-a", "Google"), ("gemini-z", "Gemini Z")]


@pytest.mark.asyncio
async def test_gemini_model_listing_stops_on_repeated_page_token_and_deduplicates() -> None:
    requests = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "models": [{"name": "models/gemini-flash", "supportedGenerationMethods": ["generateContent"]}],
                "nextPageToken": "ignored-token",
            },
        )

    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    models = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).list_models()

    assert requests == 2
    assert [model.id for model in models] == ["gemini-flash"]


@pytest.mark.asyncio
async def test_gemini_connection_test_allows_thinking_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["generationConfig"]["maxOutputTokens"] == 256
        return httpx.Response(
            200,
            json={"candidates": [{"content": {"role": "model", "parts": [{"text": "OK"}]}, "finishReason": "STOP"}]},
        )

    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).test_connection()

    assert result.content == "OK"


@pytest.mark.asyncio
async def test_gemini_reports_prompt_and_finish_errors() -> None:
    responses = [
        {"promptFeedback": {"blockReason": "SAFETY"}},
        {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "MISSING_THOUGHT_SIGNATURE", "finishMessage": "signature required"}]},
        {"candidates": [{"content": {"role": "model"}, "finishReason": "MAX_TOKENS"}], "usageMetadata": {"thoughtsTokenCount": 8}},
    ]

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=responses.pop(0))

    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="SAFETY"):
        await client.complete("system", [{"role": "user", "content": "request"}])
    with pytest.raises(RuntimeError, match="MISSING_THOUGHT_SIGNATURE.*signature required"):
        await client.complete("system", [{"role": "user", "content": "request"}])
    with pytest.raises(RuntimeError, match="达到输出 token 上限.*thoughtsTokenCount"):
        await client.complete("system", [{"role": "user", "content": "request"}])


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_gemini_prohibited_content_is_eligible_for_fallback(streaming: bool) -> None:
    response = {"candidates": [{"finishReason": "PROHIBITED_CONTENT", "finishMessage": "blocked"}]}
    if streaming:
        response = {"promptFeedback": {"blockReason": "PROHIBITED_CONTENT"}}

    def handler(_request: httpx.Request) -> httpx.Response:
        if streaming:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(response)}\n\n")
        return httpx.Response(200, json=response)

    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    with pytest.raises(ModelFallbackError, match="PROHIBITED_CONTENT") as raised:
        await BoundModelClient(preset, transport=httpx.MockTransport(handler), streaming=streaming).complete("system", [{"role": "user", "content": "request"}])
    assert raised.value.reason == "content_blocked"


@pytest.mark.asyncio
@pytest.mark.parametrize("status,eligible", [(400, False), (401, False), (403, False), (429, True), (503, True)])
async def test_http_fallback_only_for_temporary_failures(status: int, eligible: bool) -> None:
    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    client = BoundModelClient(preset, transport=httpx.MockTransport(lambda _request: httpx.Response(status, json={"error": {"message": "unavailable"}})))
    with pytest.raises(RuntimeError) as raised:
        await client.complete("system", [{"role": "user", "content": "request"}])
    assert isinstance(raised.value, ModelFallbackError) is eligible


@pytest.mark.asyncio
async def test_http_explicit_prohibited_content_is_eligible_but_auth_error_is_not() -> None:
    preset = AiPreset(id="gemini", name="Gemini", provider="google_gemini", base_url="https://example.test/v1beta", api_key="key", model="gemini-test")
    blocked = BoundModelClient(preset, transport=httpx.MockTransport(lambda _request: httpx.Response(400, json={"error": {"status": "INVALID_ARGUMENT", "message": "PROHIBITED_CONTENT"}})))
    with pytest.raises(ModelFallbackError) as raised:
        await blocked.complete("system", [{"role": "user", "content": "request"}])
    assert raised.value.reason == "content_blocked"
    unauthorized = BoundModelClient(preset, transport=httpx.MockTransport(lambda _request: httpx.Response(403, json={"error": {"status": "PERMISSION_DENIED", "message": "key not allowed"}})))
    with pytest.raises(RuntimeError) as raised:
        await unauthorized.complete("system", [{"role": "user", "content": "request"}])
    assert not isinstance(raised.value, ModelFallbackError)


@pytest.mark.asyncio
async def test_streaming_model_client_supports_deepseek_and_openai_chunks() -> None:
    preset = AiPreset(
        id="preset-id",
        name="测试",
        provider="deepseek",
        base_url="https://provider.example/v1",
        api_key="test-key",
        model="model-a",
    )
    responses = [
        'data: {"choices":[{"delta":{"reasoning_content":"思考"}}],"usage":null}\n\n'.encode(),
        'data: {"choices":[{"delta":{"content":"正文"}}],"usage":{"total_tokens":3}}\n\n'.encode(),
        b"data: [DONE]\n\n",
        b'data: {"choices":[{"delta":{"content":"Open"}}]}\n\n'
        b'data: {"choices":[],"usage":{"total_tokens":4}}\n\n'
        b"data: [DONE]\n\n",
    ]

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["stream"] is True
        body = b"".join(responses[:3] if payload["model"] == "model-a" else responses[3:])
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    client = BoundModelClient(preset, transport=httpx.MockTransport(handler), streaming=True)
    deepseek = await client.complete("system", [{"role": "user", "content": "request"}])
    assert deepseek.content == "正文"
    assert deepseek.reasoning == "思考"
    assert deepseek.usage == {"total_tokens": 3}

    openai_client = BoundModelClient(
        preset.model_copy(update={"provider": "openai_compatible", "model": "model-b"}),
        transport=httpx.MockTransport(handler),
        streaming=True,
    )
    openai = await openai_client.complete("system", [{"role": "user", "content": "request"}])
    assert openai.content == "Open"
    assert openai.usage == {"total_tokens": 4}


def test_streaming_tool_call_fragments_execute_through_runtime(tmp_path: Path) -> None:
    story = "流式工具调用写成的故事。"

    def stream_call(call_id: str, name: str, arguments: dict[str, Any]) -> bytes:
        raw = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        split_arguments = max(1, len(raw) // 2)
        chunks = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": call_id, "function": {"name": name, "arguments": raw[:split_arguments]}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": raw[split_arguments:]}}]}}]},
        ]
        return "".join(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n" for chunk in chunks).encode() + b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        system_prompt = payload["messages"][0]["content"]
        tool_results = [message for message in payload["messages"] if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            if not tool_results:
                body = stream_call("task-call", "task", {"agent": "narrator", "task": "写故事"})
            else:
                assert tool_results[-1]["tool_call_id"] == "task-call"
                body = stream_call("publish-call", "narrative_publish", {"path": "narrative.md"})
        elif system_prompt == NARRATOR_PROMPT:
            if not tool_results:
                body = stream_call("write-call", "file_write", {"path": "narrative.md", "content": story})
            else:
                assert tool_results[-1]["tool_call_id"] == "write-call"
                body = 'data: {"choices":[{"delta":{"content":"完成"}}]}\n\ndata: [DONE]\n\n'.encode()
        else:
            raise AssertionError(system_prompt)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)

    preset = AiPreset(id="stream", name="流式", api_key="test-key", model="model-a")
    model = BoundModelClient(preset, transport=httpx.MockTransport(handler), streaming=True)
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "流式工具"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == story


@pytest.mark.asyncio
async def test_streaming_model_client_reports_non_sse_response() -> None:
    preset = AiPreset(id="preset-id", name="测试", api_key="test-key")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, json={"choices": []})

    client = BoundModelClient(preset, transport=httpx.MockTransport(handler), streaming=True)
    with pytest.raises(RuntimeError, match="不支持流式模式"):
        await client.complete("system", [{"role": "user", "content": "request"}])


async def test_provider_error_messages_are_distinct() -> None:
    preset = AiPreset(
        id="preset-id",
        name="测试",
        provider="openai_compatible",
        base_url="https://provider.example/v1",
        api_key="test-key",
        model="model-a",
        timeout_seconds=120,
        max_tokens=1024,
    )

    async def expect_message(error: BaseException, snippet: str) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise error

        client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
        try:
            await client.complete("system", [{"role": "user", "content": "hello"}])
        except RuntimeError as raised:
            assert snippet in str(raised)
        else:
            raise AssertionError(f"expected RuntimeError for {type(error).__name__}")

    await expect_message(httpx.ReadTimeout(""), "等待模型响应超时（120 秒）")
    await expect_message(httpx.ConnectTimeout(""), "连接模型接口超时（120 秒）")
    await expect_message(httpx.WriteTimeout(""), "发送模型请求超时（120 秒）")
    await expect_message(httpx.PoolTimeout(""), "等待模型连接池超时（120 秒）")
    await expect_message(httpx.ConnectError(""), "无法建立模型连接：ConnectError")
    await expect_message(httpx.ConnectError("connection refused"), "无法建立模型连接：connection refused")

    request = httpx.Request("POST", "https://provider.example/v1/chat/completions")
    response = httpx.Response(524, text="A timeout occurred", request=request)
    await expect_message(
        httpx.HTTPStatusError("Server error", request=request, response=response),
        "模型接口返回 HTTP 524",
    )


def test_rejects_concurrent_turn(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="中断测试"))
    app.state.saves.create_turn(save.id, "第一项行动", 4)
    try:
        app.state.saves.create_turn(save.id, "不应进入历史的第二项行动", 4)
    except RuntimeError as error:
        assert "正在执行" in str(error)
    else:
        raise AssertionError("second running turn should be rejected")


def test_turn_terminal_state_is_compare_and_swap(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="终态测试"))
    turn = app.state.saves.create_turn(save.id, "观察码头", 4)
    interrupted = app.state.saves.fail_turn(
        save.id,
        turn.id,
        "turn.interrupted",
        {"message": "cancelled"},
    )
    failed = app.state.saves.fail_turn(save.id, turn.id, "turn.failed", {"message": "late failure"})
    assert interrupted is not None
    assert failed is None
    assert app.state.saves.get_turn(save.id, turn.id).status == "interrupted"


def test_rejects_world_symlinks_and_blank_input(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")
    make_symlink(data_dir / "worlds" / "雾港" / "entities" / "escape.txt", outside)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        response = client.post(
            "/api/saves",
            json={"world_name": "雾港", "name": "危险世界"},
        )
        assert response.status_code == 400
        assert "符号链接" in response.json()["detail"]

        blank = client.post(
            "/api/saves",
            json={"world_name": "雾港", "name": "   "},
        )
        assert blank.status_code == 422


def test_rejects_legacy_source_id_fields_and_unsafe_scenario_name(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        legacy = client.post(
            "/api/saves",
            json={"world_id": "mist-harbor", "name": "旧身份", "scenario_id": "missing-keeper", "mod_ids": []},
        )
    assert legacy.status_code == 422

    # Leading spaces are invalid application names but can be created on both
    # platforms, unlike '?' which Windows rejects before the scan can run.
    (data_dir / "worlds" / "雾港" / "scenarios" / " bad.md").write_text("危险场景", encoding="utf-8")
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        response = client.post(
            "/api/saves",
            json={"world_name": "雾港", "name": "不安全场景"},
        )
    assert response.status_code == 400
    assert "名称不能包含首尾空白" in response.json()["detail"]


def test_content_name_contract_and_casefold_uniqueness(tmp_path: Path) -> None:
    assert content_name("e\u0301") == "é"
    assert content_name("a" * 80) == "a" * 80
    assert content_name("界" * 80) == "界" * 80
    assert content_name_key("Straße") == content_name_key("STRASSE")

    for invalid in ("", " ", " name", "name ", "a/b", "a\\b", "CON", "con.txt", "name.", "a\u200bb", "a" * 81, "界" * 81):
        with pytest.raises(ValueError):
            content_name(invalid)

    worlds_dir = tmp_path / "worlds"
    for name in ("Straße", "STRASSE"):
        world = worlds_dir / name
        world.mkdir(parents=True)
    with pytest.raises(ValueError, match="世界名称重复"):
        WorldLibrary(tmp_path).list_worlds()

    linked_data = tmp_path / "linked"
    linked_world = tmp_path / "outside-world"
    linked_world.mkdir()
    (linked_data / "worlds").mkdir(parents=True)
    make_symlink(linked_data / "worlds" / "外部世界", linked_world, target_is_directory=True)
    with pytest.raises(ValueError, match="世界根目录不允许使用符号链接"):
        WorldLibrary(linked_data).list_worlds()

    linked_root_data = tmp_path / "linked-root"
    linked_root_data.mkdir()
    make_symlink(linked_root_data / "worlds", linked_world, target_is_directory=True)
    with pytest.raises(ValueError, match="世界库根目录不允许使用符号链接"):
        WorldLibrary(linked_root_data).list_worlds()


@pytest.mark.parametrize(
    "path",
    ("/entities/character/伊蕾/ENTITY.md", "entities/../伊蕾/ENTITY.md", "entities\\character\\伊蕾\\ENTITY.md", "entities/character/伊蕾/OTHER.md"),
)
def test_entity_reference_rejects_noncanonical_paths(path: str) -> None:
    with pytest.raises(ValueError):
        entity_snapshot_path(path)


@pytest.mark.parametrize("field", ("id: character.ilei", "name: 伊蕾", "type: character"))
def test_path_derived_entity_fields_are_rejected(tmp_path: Path, field: str) -> None:
    data_dir = make_data_dir(tmp_path)
    entity_path = data_dir / "worlds" / "雾港" / "entities" / "character" / "伊蕾" / "ENTITY.md"
    entity_path.write_text(entity_path.read_text(encoding="utf-8").replace("---\n", f"---\n{field}\n", 1), encoding="utf-8")
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        response = client.post("/api/saves", json={"world_name": "雾港", "name": "旧实体"})
    assert response.status_code == 400
    assert f"frontmatter 不允许字段：{field.partition(':')[0]}" in response.json()["detail"]


def test_entity_names_are_unique_across_types_within_source(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    duplicate = data_dir / "worlds" / "雾港" / "entities" / "item" / "伊蕾" / "ENTITY.md"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_text("---\n---\n另一份资料。\n", encoding="utf-8")
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        response = client.post("/api/saves", json={"world_name": "雾港", "name": "重复实体"})
    assert response.status_code == 400
    assert "来源内实体名称重复：伊蕾" in response.json()["detail"]


def test_rejects_legacy_scenario_fields(tmp_path: Path) -> None:
    data_dir, world_name = make_required_world(tmp_path)
    scenario_path = data_dir / "worlds" / world_name / "scenarios" / "主场景.md"
    scenario_path.write_text(
        "---\n"
        "description: d\n"
        "starting_location: 暮潮旅店\n"
        "known_to_player: [北岬灯塔连续三个夜晚没有点亮]\n"
        "---\n\n"
        "测试开场。\n",
        encoding="utf-8",
    )
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        response = client.post(
            "/api/saves",
            json={"world_name": world_name, "name": "遗留场景字段"},
        )
    assert response.status_code == 400
    assert "starting_location" in response.json()["detail"]
    assert "known_to_player" in response.json()["detail"]


def make_required_world(tmp_path: Path) -> tuple[Path, str]:
    data_dir = tmp_path / "data"
    world_dir = data_dir / "worlds" / "必选测试世界"
    world_dir.mkdir(parents=True)
    (world_dir / "description.md").write_text("用于验证 required 实体属性\n", encoding="utf-8")
    scenarios_dir = world_dir / "scenarios"
    scenarios_dir.mkdir()
    (scenarios_dir / "主场景.md").write_text(
        "---\ndescription: d\n---\n\n测试开场。\n",
        encoding="utf-8",
    )
    core = (
        world_dir
        / "entities"
        / "event"
        / "核心谜团"
        / "ENTITY.md"
    )
    core.parent.mkdir(parents=True)
    core.write_text(
        "---\n"
        "description: 世界核心\n"
        "required: true\n"
        "---\n"
        "# 核心谜团\n"
        "这是必选实体完整正文，玩家不应过早知晓。\n",
        encoding="utf-8",
    )
    anna = (
        world_dir
        / "entities"
        / "character"
        / "安娜"
        / "ENTITY.md"
    )
    anna.parent.mkdir(parents=True)
    anna.write_text(
        "---\n"
        "description: 外来旅人\n"
        "---\n"
        "# 安娜\n"
        "可选实体正文。\n",
        encoding="utf-8",
    )
    return data_dir, "必选测试世界"


class RequiredEntityClient(ScriptedModelClient):
    def __init__(self) -> None:
        super().__init__()
        self.researcher_payload: dict[str, Any] | None = None

    def research_action(self, research_calls: int) -> ToolCall | None:
        if research_calls == 1:
            return ToolCall(id="research-search", name="entity_search", arguments={"query": "安娜"})
        if research_calls == 2:
            return ToolCall(id="research-read", name="entity_read", arguments={"paths": ["entities/character/安娜/ENTITY.md"]})
        return None

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        if system_prompt == WORLD_RESEARCHER_PROMPT and self.researcher_payload is None:
            self.researcher_payload = json.loads(messages[1]["content"])
        return await super().complete(system_prompt, messages, tools)


def test_required_entity_parsing_and_filtering(tmp_path: Path) -> None:
    data_dir, world_name = make_required_world(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        save = client.post(
            "/api/saves",
            json={"world_name": world_name, "name": "必选解析"},
        ).json()
    optional_entities = [entry for entry in app.state.saves.list_entities(save["id"]) if not entry["required"]]
    assert [entry["path"] for entry in optional_entities] == ["entities/character/安娜/ENTITY.md"]
    assert all("id" not in entry and "source_id" not in entry for entry in optional_entities)
    assert app.state.saves.searchable_entity_count(save["id"]) == 1
    assert app.state.saves.required_entity_paths(save["id"]) == ["entities/event/核心谜团/ENTITY.md"]
    required_docs = app.state.saves.required_entity_documents(save["id"])
    assert [doc["path"] for doc in required_docs] == ["entities/event/核心谜团/ENTITY.md"]
    assert all("id" not in doc and "source_id" not in doc for doc in required_docs)
    required_doc = required_docs[0]
    assert required_doc["required"] is True
    assert "必选实体完整正文" in required_doc["content"]


def test_required_entity_reaches_researcher_and_narrator_context(tmp_path: Path) -> None:
    data_dir, world_name = make_required_world(tmp_path)
    model = RequiredEntityClient()
    app = create_app(data_dir=data_dir, model_client=model)
    with TestClient(app) as client:
        save = client.post(
            "/api/saves",
            json={"world_name": world_name, "name": "必选注入"},
        ).json()
        client.post(
            f"/api/saves/{save['id']}/turns",
            json={"content": "我找安娜打听。"},
        )
        turn = wait_for_turn(client, save["id"])
    assert turn["status"] == "completed"

    researcher_first = model.researcher_payload
    assert researcher_first is not None
    required_at_top = researcher_first["required_world_entities"]
    assert researcher_first["required_world_entities_instruction"] == "这些是对故事长期必要的信息，因此直接提供给你。"
    assert [item["path"] for item in required_at_top] == ["entities/event/核心谜团/ENTITY.md"]
    assert all("id" not in item and "source_id" not in item for item in required_at_top)
    assert required_at_top[0]["required"] is True
    assert researcher_first["task"]["task"] == "查明旅店中与灯塔失踪有关的线索"

    events = app.state.saves.list_events(save["id"])
    coordinator_request = next(
        event for event in events if event.type == "model.request" and event.payload["agent"] == "coordinator"
    )
    coordinator = json.dumps(coordinator_request.payload["messages"], ensure_ascii=False)
    assert "required_world_entities_instruction" not in coordinator
    assert "required_world_entities" not in coordinator

    assert model.narrator_payload is not None
    narrator = model.narrator_payload
    assert "required_world_entities_instruction" not in narrator
    assert "required_world_entities" not in narrator
    assert narrator["task"]["task"].startswith("描写玩家")
    assert [item["path"] for item in narrator["related_entity_documents"]] == [
        "entities/event/核心谜团/ENTITY.md",
        "entities/character/安娜/ENTITY.md",
    ]


def test_read_required_entity_returns_current_document(tmp_path: Path) -> None:
    data_dir, world_name = make_required_world(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        save = client.post(
            "/api/saves",
            json={"world_name": world_name, "name": "必选读取"},
        ).json()

        class ReadAllClient(RequiredEntityClient):
            def research_action(self, research_calls: int) -> ToolCall | None:
                if research_calls == 1:
                    return ToolCall(
                        id="research-read",
                        name="entity_read",
                        arguments={
                            "paths": [
                                "entities/event/核心谜团/ENTITY.md",
                                "entities/character/安娜/ENTITY.md",
                            ]
                        },
                    )
                return None

        model = ReadAllClient()
        app.state.runtime.model = model
        created = client.post(
            f"/api/saves/{save['id']}/turns",
            json={"content": "我打听核心谜团。"},
        ).json()
        wait_for_turn(client, save["id"])
    events = app.state.saves.list_turn_events(save["id"], created["turn_id"])
    read_event = next(event for event in events if event.type == "entity.read")
    assert [doc["path"] for doc in read_event.payload["documents"]] == [
        "entities/event/核心谜团/ENTITY.md", "entities/character/安娜/ENTITY.md",
    ]
    assert read_event.payload["retrieval_counter"] == {
        "total_searchable_entities": 1,
        "read_entities": 1,
    }


def test_save_database_contains_only_current_runtime_state(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="数据库结构测试"))

    with app.state.saves.connect(save.id) as db:
        tables = {
            row["name"]
            for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }

    assert {"world_facts", "entities", "changesets"}.isdisjoint(tables)
    matches = app.state.saves.search_entities(save.id, "伊蕾")
    assert [entry["path"] for entry in matches] == ["entities/character/伊蕾/ENTITY.md"]
    documents = app.state.saves.read_entities(
        save.id,
        ["entities/character/伊蕾/ENTITY.md"],
    )
    assert "伊蕾" in documents[0]["content"]
    assert not (data_dir / "saves" / save.id / "tmp").exists()
