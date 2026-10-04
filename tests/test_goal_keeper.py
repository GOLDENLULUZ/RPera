from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.content import markdown_frontmatter
from rpera.models import ContentEntityCreate, ModelResult, SaveCreate, ToolCall
from tests.helpers import make_data_dir, wait_for_turn
from tests.prompt_fixtures import COORDINATOR_PROMPT, NARRATOR_PROMPT, PROMPT_TEMPLATES, WORLD_RESEARCHER_PROMPT


GOAL_PATH = "entities/goal/目标清单/ENTITY.md"


def test_goal_is_an_ordinary_entity_type_and_agent_creation_is_required(tmp_path: Path) -> None:
    assert ContentEntityCreate(name="自定目标", type="goal").type == "goal"
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="目标实体"))
    manual = app.state.saves.create_save_entity(save.id, "自定目标", "goal")
    assert manual["required"] is False

    created = app.state.saves.create_goal(save.id, "当前可回应的机会", {"long_term": ["找到归途"], "short_term": ["赴约"]})
    assert created["path"] == GOAL_PATH
    assert created["required"] is True
    assert GOAL_PATH in app.state.saves.required_entity_paths(save.id)
    path = app.state.saves.saves_dir / save.id / "current" / "world_snapshot" / GOAL_PATH
    front, body = markdown_frontmatter(path.read_text(encoding="utf-8"), "目标")
    assert front["required"] is True
    assert json.loads(body)["short_term"] == ["赴约"]

    edited = app.state.saves.edit_goal(save.id, '"short_term": [\n    "赴约"\n  ]', '"short_term": []')
    assert json.loads(edited["content"]) == {"long_term": ["找到归途"], "short_term": []}
    assert edited["required"] is True
    with pytest.raises(ValueError, match="不允许修改"):
        app.state.saves.edit_goal(save.id, "required: true", "required: false")
    with pytest.raises(ValueError, match="合法 JSON 对象"):
        app.state.saves.edit_goal(save.id, '"short_term": []', '"short_term":')


class GoalFlowModel:
    def __init__(self) -> None:
        self.goals_returned: list[dict[str, Any]] = []
        self.narrator_goals: list[dict[str, Any]] = []

    def bind(self):
        return self

    def bind_for_agents(self, names: tuple[str, ...]):
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": "test", "model": "goal-flow", "base_url": "local", "timeout_seconds": 1}

    async def complete(
        self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        del tools
        completed = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            if not completed:
                call = ToolCall(id="goal-task", name="task", arguments={"agent": "goal_keeper", "task": "维护玩家当前可回应的目标"})
            elif len(completed) == 1:
                result = completed[0]
                assert result["ok"] is True
                assert "report_ref" not in result
                assert "related_entity_documents" not in result
                self.goals_returned.append(result["goal"])
                call = ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写出接下来的情境"})
            else:
                call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
            return self._call(call)
        if system_prompt == PROMPT_TEMPLATES["goal_keeper"]:
            if not completed:
                return self._call(ToolCall(id="read", name="goal_read", arguments={}))
            if len(completed) == 1:
                current = completed[0]["goal"]
                if current is None:
                    return self._call(ToolCall(id="create", name="goal_create", arguments={
                        "description": "活跃目标", "profile": {"long_term": ["寻找失踪者"], "short_term": ["陪同去图书馆"]},
                    }))
                assert current["path"] == GOAL_PATH
                assert current["document"]
                return self._call(ToolCall(id="edit", name="goal_edit", arguments={
                    "old_text": '"short_term": [\n    "陪同去图书馆"\n  ]',
                    "new_text": '"short_term": []',
                }))
            return ModelResult(content="目标清单已维护。", raw_response={})
        if system_prompt == NARRATOR_PROMPT:
            if not completed:
                initial = json.loads(messages[1]["content"])
                self.narrator_goals.append(next(
                    entity for entity in initial["related_entity_documents"] if entity["path"] == GOAL_PATH
                ))
                return self._call(ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "新的日子开始了。"}))
            return ModelResult(content="完成。", raw_response={})
        raise AssertionError(f"unexpected prompt: {system_prompt[:80]}")

    @staticmethod
    def _call(call: ToolCall) -> ModelResult:
        return ModelResult(content="", raw_response={}, tool_calls=[call])


def test_goal_task_returns_real_entity_without_report_and_cleans_finished_goal(tmp_path: Path) -> None:
    model = GoalFlowModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "目标闭环"}).json()
        for action in ("我遇到新同学。", "我们已经去了图书馆。"):
            response = client.post(f"/api/saves/{save['id']}/turns", json={"content": action})
            assert response.status_code == 202
            turn = wait_for_turn(client, save["id"])
            assert turn["status"] == "completed", [
                event.payload for event in app.state.saves.list_turn_events(save["id"], turn["id"])
                if event.type in {"tool.failed", "turn.failed"}
            ]

    assert len(model.goals_returned) == 2
    assert [json.loads(goal["content"])["short_term"] for goal in model.goals_returned] == [["陪同去图书馆"], []]
    assert all(json.loads(goal["content"])["long_term"] == ["寻找失踪者"] for goal in model.goals_returned)
    assert [entity["content"] for entity in model.narrator_goals] == [goal["content"] for goal in model.goals_returned]


def test_goal_keeper_receives_reported_optional_entity_before_required_entities(tmp_path: Path) -> None:
    character_path = "entities/character/伊蕾/ENTITY.md"
    required_path = "mods/读心大师/entities/player_trait/读心能力/ENTITY.md"

    class ReportedGoalFlowModel(GoalFlowModel):
        def __init__(self) -> None:
            super().__init__()
            self.goal_input: dict[str, Any] | None = None

        async def complete(
            self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
        ) -> ModelResult:
            completed = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not completed:
                    call = ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "调查伊蕾"})
                elif len(completed) == 1:
                    assert completed[0]["result"]["related_entities"] == [character_path]
                    call = ToolCall(id="goal-task", name="task", arguments={
                        "agent": "goal_keeper", "task": "根据伊蕾的设定维护目标",
                        "related_entities": [character_path], "report_refs": [completed[0]["report_ref"]],
                    })
                elif len(completed) == 2:
                    assert completed[1]["ok"] is True
                    call = ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写出接下来的情境"})
                else:
                    call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
                return self._call(call)
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                if not completed:
                    call = ToolCall(id="search", name="entity_search", arguments={"query": "伊蕾"})
                elif len(completed) == 1:
                    call = ToolCall(id="read-entity", name="entity_read", arguments={"paths": [character_path]})
                else:
                    call = ToolCall(id="report", name="research_report", arguments={
                        "report": "伊蕾正在寻找线索。", "related_entities": [character_path],
                    })
                return self._call(call)
            if system_prompt == PROMPT_TEMPLATES["goal_keeper"] and not completed:
                self.goal_input = json.loads(messages[1]["content"])
            return await super().complete(system_prompt, messages, tools)

    model = ReportedGoalFlowModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={
            "world_name": "雾港", "name": "目标相关实体", "mod_names": ["读心大师"],
        }).json()
        assert client.post(f"/api/saves/{save['id']}/turns", json={"content": "我遇到了伊蕾。"}).status_code == 202
        turn = wait_for_turn(client, save["id"], timeout=30)
        assert turn["status"] == "completed", [
            event.payload for event in app.state.saves.list_turn_events(save["id"], turn["id"])
            if event.type in {"tool.failed", "turn.failed"}
        ]

    assert model.goal_input is not None
    assert model.goal_input["task"]["related_entities"] == [character_path, required_path]
    assert [entity["path"] for entity in model.goal_input["related_entity_documents"]] == [character_path, required_path]
    assert model.goal_input["related_entity_documents"][0]["content"]
    assert model.goal_input["report_documents"][0]["source_agent"] == "world_researcher"
