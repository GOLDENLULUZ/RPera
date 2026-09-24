from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpera.agents import AGENTS, coordinator_prompt
from rpera.app import PROJECT_ROOT, create_app
from rpera.config import RuntimeSettingsStore
from rpera.draft_files import DraftFileError, DraftFileStore
from rpera.models import ModelResult, RuntimeSettings, SaveCreate, ToolCall
from rpera.providers import ModelClient
from rpera.runtime import AgentRunner, _thread as runtime_thread
from rpera.storage import TurnRetryError
from tests.helpers import make_data_dir, wait_for_turn
from tests.prompt_fixtures import COMPLIANCE_REVIEWER_PROMPT, CONSISTENCY_CHECKER_PROMPT, COORDINATOR_PROMPT, EROTIC_OR_NOT_PROMPT, NARRATOR_PROMPT, PROMPT_TEMPLATES, ROLE_PLAYER_PROMPT, STYLE_PLANNER_PROMPT, WORLD_RESEARCHER_PROMPT


STORY = "伊蕾放下空杯，抬眼问道：‘你为什么关心那个守灯人？’"


def make_prompt_dir(tmp_path: Path) -> Path:
    prompt_dir = tmp_path / "prompts"
    shutil.copytree(PROJECT_ROOT / "prompts", prompt_dir)
    return prompt_dir


def make_prefill_dir(tmp_path: Path) -> Path:
    prefill_dir = tmp_path / "prefills"
    shutil.copytree(PROJECT_ROOT / "prefills", prefill_dir)
    return prefill_dir


class NarrativeFileClient:
    def bind(self) -> ModelClient:
        return self

    def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, ModelClient]:
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": "test", "model": "batch-one", "base_url": "local", "timeout_seconds": 1}

    @staticmethod
    def result(content: str = "", call: ToolCall | None = None) -> ModelResult:
        return ModelResult(content=content, raw_response={}, tool_calls=[] if call is None else [call])

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        tool_names = {item["function"]["name"] for item in tools or []}
        core_tools = tool_names - {"story_summary_read"}
        results = [message for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            assert core_tools == {"task", "file_read", "narrative_publish"}
            if not results:
                return self.result(call=ToolCall(id="same", name="task", arguments={"agent": "world_researcher", "task": "调查旅店"}))
            if len(results) == 1:
                return self.result(call=ToolCall(id="same", name="task", arguments={"agent": "narrator", "task": "写出酒馆场景"}))
            if len(results) == 2:
                return self.result(call=ToolCall(id="same", name="file_read", arguments={"path": "narrative.md"}))
            return self.result(call=ToolCall(id="same", name="narrative_publish", arguments={"path": "narrative.md"}))
        if system_prompt == WORLD_RESEARCHER_PROMPT:
            assert core_tools == {"entity_search", "entity_read", "research_report"}
            if not results:
                return self.result(call=ToolCall(id="same", name="entity_search", arguments={"query": "暮潮旅店"}))
            return self.result(call=ToolCall(id="report", name="research_report", arguments={"report": "暮潮旅店是当地消息汇集处。", "related_entities": []}))
        if system_prompt == NARRATOR_PROMPT:
            assert core_tools == {"file_read", "file_write", "file_edit"}
            if not results:
                return self.result(call=ToolCall(id="same", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
            return self.result("叙事草稿已完成。")
        raise AssertionError(system_prompt)


def test_user_can_abort_and_resume_same_turn(tmp_path: Path) -> None:
    class AbortOnceClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.started = threading.Event()
            self.block_first_request = True

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if self.block_first_request:
                self.block_first_request = False
                self.started.set()
                await asyncio.Event().wait()
            return await super().complete(system_prompt, messages, tools)

    model = AbortOnceClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "主动中止"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续调查。"}).json()
        assert model.started.wait(timeout=2)
        root_session_id = app.state.saves.root_session_id(save["id"], created["turn_id"])

        aborted = client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/abort")
        assert aborted.status_code == 200
        assert aborted.json() == {"turn_id": created["turn_id"], "status": "interrupted"}
        assert app.state.saves.get_turn(save["id"], created["turn_id"]).status == "interrupted"
        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/abort").status_code == 200

        retried = client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry")
        assert retried.status_code == 202
        turn = wait_for_turn(client, save["id"])
        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/abort").status_code == 409

    assert turn["id"] == created["turn_id"]
    assert turn["status"] == "completed"
    assert app.state.saves.root_session_id(save["id"], created["turn_id"]) == root_session_id
    with app.state.saves.connect(save["id"]) as db:
        events = db.execute("SELECT type, payload FROM events WHERE turn_id = ? ORDER BY id", (created["turn_id"],)).fetchall()
        user_messages = db.execute("SELECT COUNT(*) FROM messages WHERE session_id = ? AND role = 'user'", (root_session_id,)).fetchone()[0]
        stories = db.execute("SELECT COUNT(*) FROM messages WHERE kind = 'story'").fetchone()[0]
    assert [row["type"] for row in events].count("turn.interrupted") == 1
    interrupted = next(row for row in events if row["type"] == "turn.interrupted")
    assert json.loads(interrupted["payload"])["message"] == "用户中止，回合已中止"
    assert "turn.resumed" in [row["type"] for row in events]
    assert user_messages == 1
    assert stories == 1


def test_retry_removes_trailing_coordinator_reply_without_tool_call(tmp_path: Path) -> None:
    rejected_reply = "我先停在这里，等待进一步指示。"

    class RetryReplyClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.coordinator_calls = 0
            self.retry_messages: list[dict[str, Any]] | None = None

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                self.coordinator_calls += 1
                if self.coordinator_calls == 1:
                    return self.result(rejected_reply)
                if self.coordinator_calls == 2:
                    self.retry_messages = messages
            return await super().complete(system_prompt, messages, tools)

    model = RetryReplyClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "清理末尾回复"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续调查。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"
        with app.state.saves.connect(save["id"]) as db:
            rejected_message = db.execute(
                """SELECT messages.id FROM messages
                   JOIN parts ON parts.message_id = messages.id
                   WHERE parts.content = ?""",
                (rejected_reply,),
            ).fetchone()
        assert rejected_message is not None

        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry").status_code == 202
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    assert model.retry_messages is not None
    assert model.retry_messages[-2:] == [
        {"role": "user", "content": "继续调查。"},
        {"role": "system", "content": "</recent_story>"},
    ]
    assert all(message.get("content") != rejected_reply for message in model.retry_messages)
    with app.state.saves.connect(save["id"]) as db:
        assert db.execute("SELECT 1 FROM messages WHERE id = ?", (rejected_message["id"],)).fetchone() is None
        response_events = db.execute("SELECT payload FROM events WHERE type = 'model.response'").fetchall()
    assert any(json.loads(row["payload"]).get("content") == rejected_reply for row in response_events)


@pytest.mark.parametrize(
    ("disabled_agents", "expected_agent"),
    [([], "story_summarizer"), (["story_summarizer"], "world_researcher")],
)
def test_force_start_delegation_removes_plain_reply_from_real_chat(
    tmp_path: Path,
    disabled_agents: list[str],
    expected_agent: str,
) -> None:
    rejected_reply = "我会先说明计划，再开始行动。"
    hidden_reasoning = "先向用户解释一下。"
    native_marker = "native coordinator reply"

    class ForceStartClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.coordinator_calls = 0
            self.second_coordinator_messages: list[dict[str, Any]] | None = None

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if "narrative_publish" in {item["function"]["name"] for item in tools or []}:
                self.coordinator_calls += 1
                results = [message for message in messages if message["role"] == "tool"]
                if self.coordinator_calls == 1:
                    return ModelResult(
                        content=rejected_reply,
                        reasoning=hidden_reasoning,
                        raw_response={},
                        provider_content={"content": native_marker},
                    )
                if self.coordinator_calls == 2:
                    self.second_coordinator_messages = messages
                    assert len(results) == 1
                    return self.result(call=ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写出酒馆场景"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            return await super().complete(system_prompt, messages, tools)

    model = ForceStartClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        settings = client.get("/api/runtime-settings").json()
        settings["force_start_delegation"] = True
        settings["disabled_agents"] = disabled_agents
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "强制开始委派"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert model.second_coordinator_messages is not None
    real_chat = json.dumps(model.second_coordinator_messages, ensure_ascii=False)
    assert rejected_reply not in real_chat
    assert hidden_reasoning not in real_chat
    assert native_marker not in real_chat
    root = app.state.saves.root_session_id(save["id"], created["turn_id"])
    with app.state.saves.connect(save["id"]) as db:
        first_reply = db.execute(
            "SELECT id, model FROM messages WHERE session_id = ? AND role = 'assistant' AND kind = 'model' ORDER BY sequence LIMIT 1",
            (root,),
        ).fetchone()
        parts = db.execute("SELECT type, tool_name, input FROM parts WHERE message_id = ? ORDER BY sequence", (first_reply["id"],)).fetchall()
        response_events = db.execute(
            "SELECT payload FROM events WHERE turn_id = ? AND type = 'model.response'",
            (created["turn_id"],),
        ).fetchall()
        first_task = db.execute(
            "SELECT payload FROM events WHERE turn_id = ? AND type = 'task.created' ORDER BY id LIMIT 1",
            (created["turn_id"],),
        ).fetchone()

    assert len(parts) == 1
    assert parts[0]["type"] == "tool"
    assert parts[0]["tool_name"] == "task"
    assert json.loads(parts[0]["input"]) == {"agent": expected_agent, "task": "请开始工作", "reason": ""}
    assert "provider_content" not in json.loads(first_reply["model"])
    assert any(json.loads(row["payload"])["content"] == rejected_reply for row in response_events)
    assert json.loads(first_task["payload"])["target_agent"] == expected_agent


def test_force_start_delegation_only_checks_first_session_response(tmp_path: Path) -> None:
    class ToolThenTextClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.coordinator_calls = 0

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                self.coordinator_calls += 1
                if self.coordinator_calls == 1:
                    return self.result(call=ToolCall(id="missing", name="file_read", arguments={"path": "narrative.md"}))
                return self.result("第二次响应不应被改成委派。")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ToolThenTextClient())
    with TestClient(app) as client:
        settings = client.get("/api/runtime-settings").json()
        settings["force_start_delegation"] = True
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "只检查首次响应"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"
        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry").status_code == 202
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "failed"
    with app.state.saves.connect(save["id"]) as db:
        assert db.execute("SELECT COUNT(*) FROM parts WHERE tool_name = 'task'").fetchone()[0] == 0


def test_force_start_delegation_runs_on_retry_when_session_has_no_assistant(tmp_path: Path) -> None:
    class PlainReplyClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                return self.result("本次只返回文本。")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=PlainReplyClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "重试强制委派"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"
        settings = client.get("/api/runtime-settings").json()
        settings["force_start_delegation"] = True
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry").status_code == 202
        assert wait_for_turn(client, save["id"])["status"] == "failed"

    with app.state.saves.connect(save["id"]) as db:
        assert db.execute("SELECT COUNT(*) FROM parts WHERE tool_name = 'task'").fetchone()[0] == 1


def test_force_publish_replaces_plain_coordinator_reply(tmp_path: Path) -> None:
    coordinator_reply = "叙事已经完成，可以发布。"

    class AutomaticPublishClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                results = [message for message in messages if message["role"] == "tool"]
                if not results:
                    return self.result(call=ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写出酒馆场景"}))
                return self.result(coordinator_reply)
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=AutomaticPublishClient())
    with TestClient(app) as client:
        settings = client.get("/api/runtime-settings").json()
        settings["force_publish_narrative"] = True
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "自动发布"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == STORY
    root = app.state.saves.root_session_id(save["id"], created["turn_id"])
    with app.state.saves.connect(save["id"]) as db:
        final_text = db.execute(
            """SELECT parts.content FROM messages
               JOIN parts ON parts.message_id = messages.id
               WHERE messages.session_id = ? AND messages.kind = 'model' AND parts.type = 'text'
               ORDER BY messages.sequence DESC LIMIT 1""",
            (root,),
        ).fetchone()["content"]
        publish_parts = db.execute(
            "SELECT state, input, output FROM parts WHERE session_id = ? AND tool_name = 'narrative_publish'",
            (root,),
        ).fetchall()
        response_events = db.execute(
            "SELECT payload FROM events WHERE turn_id = ? AND type = 'model.response'",
            (created["turn_id"],),
        ).fetchall()
        story_count = db.execute("SELECT COUNT(*) FROM messages WHERE kind = 'story'").fetchone()[0]

    assert final_text == "已自动发布 narrative.md。"
    assert len(publish_parts) == 1
    assert publish_parts[0]["state"] == "completed"
    assert json.loads(publish_parts[0]["input"]) == {"path": "narrative.md"}
    assert json.loads(publish_parts[0]["output"])["content"] == STORY
    assert story_count == 1
    assert any(json.loads(row["payload"])["content"] == coordinator_reply for row in response_events)


def test_force_publish_requires_existing_narrative(tmp_path: Path) -> None:
    class NoNarrativeClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                return self.result("还没有叙事草稿。")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NoNarrativeClient())
    with TestClient(app) as client:
        settings = client.get("/api/runtime-settings").json()
        settings["force_publish_narrative"] = True
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "无草稿"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "failed"
    assert turn["narrative"] is None
    with app.state.saves.connect(save["id"]) as db:
        assert db.execute("SELECT COUNT(*) FROM parts WHERE tool_name = 'narrative_publish'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM messages WHERE kind = 'story'").fetchone()[0] == 0


def test_retry_preserves_trailing_coordinator_tool_call(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="保留工具调用"))
    turn = app.state.saves.create_turn(save.id, "继续。", 4)
    root = app.state.saves.root_session_id(save.id, turn.id)
    message = app.state.saves.create_assistant_message(save.id, root, {"model": "test"})
    part = app.state.saves.record_model_result(
        save.id,
        message,
        "",
        None,
        [{"id": "read", "name": "file_read", "arguments": {"path": "narrative.md"}}],
    )[0]
    app.state.saves.set_tool_running(save.id, part["id"])
    app.state.saves.error_tool(save.id, part["id"], {"ok": False, "error": {"code": "test"}})
    app.state.saves.fail_turn(save.id, turn.id, "turn.failed", {"message": "test"})

    app.state.saves.retry_turn(save.id, turn.id, 4)

    with app.state.saves.connect(save.id) as db:
        assert db.execute("SELECT 1 FROM messages WHERE id = ?", (message,)).fetchone() is not None


def test_rejected_retry_does_not_remove_trailing_reply(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="拒绝重试"))
    turn = app.state.saves.create_turn(save.id, "继续。", 4)
    root = app.state.saves.root_session_id(save.id, turn.id)
    message = app.state.saves.create_assistant_message(save.id, root, {"model": "test"})
    app.state.saves.record_model_result(save.id, message, "尚未结束。", None, [])

    with pytest.raises(RuntimeError, match="只有失败或中断回合可以继续"):
        app.state.saves.retry_turn(save.id, turn.id, 4)

    with app.state.saves.connect(save.id) as db:
        assert db.execute("SELECT 1 FROM messages WHERE id = ?", (message,)).fetchone() is not None


async def test_runtime_waits_for_sync_boundary_before_interrupting() -> None:
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def operation() -> None:
        started.set()
        assert release.wait(timeout=2)
        finished.set()

    task = asyncio.create_task(runtime_thread(operation))
    while not started.is_set():
        await asyncio.sleep(0.001)
    task.cancel("stop")
    await asyncio.sleep(0.01)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert finished.is_set()


def test_public_turn_persists_sessions_parts_and_publishes_once(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "持久化故事"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "我询问守灯人。"}).json()
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == STORY
    draft = tmp_path / "data" / "saves" / save["id"] / "current" / "turns" / created["turn_id"] / "drafts" / "narrative.md"
    assert draft.read_text(encoding="utf-8") == STORY
    with app.state.saves.connect(save["id"]) as db:
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE turn_id = ?", (created["turn_id"],)).fetchone()[0] == 3
        assert db.execute("SELECT COUNT(*) FROM messages WHERE session_id IS NOT NULL", ()).fetchone()[0] > 3
        completed = db.execute("SELECT tool_name, output FROM parts WHERE type = 'tool' AND state = 'completed'").fetchall()
        assert [row["tool_name"] for row in completed].count("narrative_publish") == 1
        assert db.execute("SELECT COUNT(*) FROM messages WHERE kind = 'story'").fetchone()[0] == 1


def test_role_report_receives_disclosed_entities_and_remains_distinct_in_narrator_task(tmp_path: Path) -> None:
    character_path = "entities/character/伊蕾/ENTITY.md"

    class RoleFlowClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.role_input: dict[str, Any] | None = None
            self.style_input: dict[str, Any] | None = None
            self.narrator_input: dict[str, Any] | None = None
            self.task_tool_description = ""

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                task_tool = next(item["function"] for item in tools or [] if item["function"]["name"] == "task")
                self.task_tool_description = task_tool["description"]
                if not results:
                    return self.result(call=ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "调查伊蕾"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "扮演伊蕾，判断她如何回应玩家。", "related_entities": [character_path]}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="style", name="task", arguments={"agent": "style_planner", "task": "为酒馆对话选择文风。", "related_entities": [character_path], "report_refs": [results[0]["report_ref"], results[1]["report_ref"]]}))
                if len(results) == 3:
                    return self.result(call=ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写出玩家与伊蕾的交锋，不得替玩家行动。", "reason": "整合角色和文风报告形成正文", "related_entities": [character_path], "report_refs": [result["report_ref"] for result in results]}))
                if len(results) == 4:
                    return self.result(call=ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="search", name="entity_search", arguments={"query": "伊蕾"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="read-entity", name="entity_read", arguments={"paths": [character_path]}))
                return self.result(call=ToolCall(id="research-report", name="research_report", arguments={"report": "伊蕾寡言、警惕并保守秘密。", "related_entities": [character_path]}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                assert {item["function"]["name"] for item in tools or []} == {"story_summary_read"}
                self.role_input = json.loads(messages[1]["content"])
                return self.result("伊蕾会先观察玩家的诚意，准备含糊回应，不主动提及封蜡信。")
            if system_prompt == STYLE_PLANNER_PROMPT:
                self.style_input = json.loads(messages[1]["content"])
                return self.result(call=ToolCall(id="style-report", name="style_report", arguments={"report": "本回合没有合适的全局文风，保持克制。", "style_paths": []}))
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_input = json.loads(messages[1]["content"])
                if not results:
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
                return self.result("叙事草稿已完成。")
            raise AssertionError(system_prompt)

    model = RoleFlowClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": ["narrator"]},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "角色报告链条", "mod_names": ["读心大师"]}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "我试探伊蕾是否隐瞒消息。"})
        assert wait_for_turn(client, save["id"])["narrative"] == STORY

    assert model.role_input is not None
    assert model.style_input is not None
    assert model.narrator_input is not None
    assert [document["path"] for document in model.role_input["related_entity_documents"]] == [
        character_path,
        "mods/读心大师/entities/player_trait/读心能力/ENTITY.md",
    ]
    assert model.style_input["task"]["task"] == "为酒馆对话选择文风。"
    assert [document["content"] for document in model.style_input["report_documents"]] == [
        "伊蕾寡言、警惕并保守秘密。",
        "伊蕾会先观察玩家的诚意，准备含糊回应，不主动提及封蜡信。",
    ]
    assert model.narrator_input["task"]["task"] == "请开始工作"
    assert model.narrator_input["task"]["reason"] == ""
    assert [document["source_agent"] for document in model.narrator_input["report_documents"]] == [
        "world_researcher",
        "role_player",
        "style_planner",
    ]
    assert [document["content"] for document in model.narrator_input["report_documents"]] == [
        "伊蕾寡言、警惕并保守秘密。",
        "伊蕾会先观察玩家的诚意，准备含糊回应，不主动提及封蜡信。",
        "本回合没有合适的全局文风，保持克制。",
    ]
    assert model.narrator_input["related_entity_documents"][0]["path"] == character_path
    assert model.task_tool_description.count("该子代理不接收具体任务文本，只接受实体和报告，请只要求“开始任务”。") == 1
    task_events = [event for event in app.state.saves.list_events(save["id"]) if event.type == "task.created"]
    narrator_event = next(event for event in task_events if event.payload["target_agent"] == "narrator")
    assert narrator_event.payload["task"] == "写出玩家与伊蕾的交锋，不得替玩家行动。"
    assert narrator_event.payload["reason"] == "整合角色和文风报告形成正文"
    assert narrator_event.payload["instruction_blocked"] is True
    assert all(event.payload["instruction_blocked"] is False for event in task_events if event is not narrator_event)
    trace_narrator = next(
        event for event in app.state.saves.list_trace_page(save["id"]).events
        if event.type == "task.created" and event.payload["target_agent"] == "narrator"
    )
    assert trace_narrator.payload["instruction_blocked"] is True


def test_forced_reports_merge_with_explicit_refs_and_skip_same_child_history(tmp_path: Path) -> None:
    class ForcedReportClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.role_initial_inputs: list[dict[str, Any]] = []
            self.role_continuation_input: dict[str, Any] | None = None
            self.narrator_input: dict[str, Any] | None = None
            self.task_tool_description = ""

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                task_tool = next(item["function"] for item in tools or [] if item["function"]["name"] == "task")
                self.task_tool_description = task_tool["description"]
                if not results:
                    return self.result(call=ToolCall(id="role-a", name="task", arguments={"agent": "role_player", "task": "形成角色报告甲"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="role-b", name="task", arguments={"agent": "role_player", "task": "形成角色报告乙"}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="role-a-continue", name="task", arguments={"agent": "role_player", "task": "修订角色报告甲", "task_id": results[0]["task_id"]}))
                if len(results) == 3:
                    return self.result(call=ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "根据报告写故事", "report_refs": [results[1]["report_ref"]]}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                latest_user = json.loads(next(message["content"] for message in reversed(messages) if message["role"] == "user"))
                if latest_user.get("continuation"):
                    self.role_continuation_input = latest_user
                    return self.result("角色报告甲修订版")
                initial = json.loads(messages[1]["content"])
                self.role_initial_inputs.append(initial)
                return self.result("角色报告甲" if initial["task"]["task"].endswith("甲") else "角色报告乙")
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_input = json.loads(messages[1]["content"])
                if not results:
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
                return self.result("叙事草稿已完成。")
            raise AssertionError(system_prompt)

    model = ForcedReportClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": [], "always_attach_report_agents": ["role_player", "narrator"]},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "强制报告"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["narrative"] == STORY

    assert "若尚未取得该来源的报告" in model.task_tool_description
    assert "report_documents" not in model.role_initial_inputs[0]
    assert [document["content"] for document in model.role_initial_inputs[1]["report_documents"]] == ["角色报告甲"]
    assert model.role_continuation_input is not None
    assert [document["content"] for document in model.role_continuation_input["report_documents"]] == ["角色报告乙"]
    assert model.narrator_input is not None
    assert [document["content"] for document in model.narrator_input["report_documents"]] == [
        "角色报告乙",
        "角色报告甲",
        "角色报告甲修订版",
    ]
    report_ids = [ref["id"] for ref in model.narrator_input["task"]["report_refs"]]
    assert len(report_ids) == len(set(report_ids)) == 3
    narrator_event = next(
        event for event in app.state.saves.list_events(save["id"])
        if event.type == "task.created" and event.payload["target_agent"] == "narrator"
    )
    assert [report["source_agent"] for report in narrator_event.payload["reports"]] == ["role_player"] * 3


def test_later_delegation_requires_each_earliest_missing_report(tmp_path: Path) -> None:
    class RequiredReportClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.role_input: dict[str, Any] | None = None
            self.narrator_input: dict[str, Any] | None = None

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="blocked-compliance", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if len(results) == 1:
                    assert results[0] == {
                        "ok": False,
                        "error": {
                            "code": "required_report_missing",
                            "message": "委派 narrator 前必须先取得 compliance_reviewer 的报告",
                        },
                        "blocked_agent": "narrator",
                        "required_delegation": {"agent": "compliance_reviewer"},
                    }
                    return self.result(call=ToolCall(id="compliance", name="task", arguments={"agent": "compliance_reviewer", "task": "审核叙事方向"}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="blocked-role", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if len(results) == 3:
                    assert results[2]["error"]["message"] == "委派 narrator 前必须先取得 role_player 的报告"
                    assert results[2]["required_delegation"] == {"agent": "role_player"}
                    return self.result(call=ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "形成角色报告"}))
                if len(results) == 4:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                self.role_input = json.loads(messages[1]["content"])
                return self.result("角色报告")
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_input = json.loads(messages[1]["content"])
            return await super().complete(system_prompt, messages, tools)

    model = RequiredReportClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": [], "always_attach_report_agents": ["role_player", "compliance_reviewer"]},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "前置报告门禁"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert app.state.saves.task_count(save["id"], turn["id"]) == 3
    assert model.role_input is not None
    assert [document["source_agent"] for document in model.role_input["report_documents"]] == ["compliance_reviewer"]
    assert model.narrator_input is not None
    assert [document["source_agent"] for document in model.narrator_input["report_documents"]] == ["compliance_reviewer", "role_player"]
    with app.state.saves.connect(save["id"]) as db:
        task_parts = db.execute(
            "SELECT state, child_session_id, output FROM parts WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?) AND tool_name = 'task' ORDER BY rowid",
            (created["turn_id"],),
        ).fetchall()
    assert [part["state"] for part in task_parts] == ["error", "completed", "error", "completed", "completed"]
    assert task_parts[0]["child_session_id"] is None
    assert task_parts[2]["child_session_id"] is None
    failed_events = [event for event in app.state.saves.list_events(save["id"]) if event.type == "tool.failed"]
    assert [event.payload["result"]["required_delegation"]["agent"] for event in failed_events] == ["compliance_reviewer", "role_player"]


def test_disabling_report_source_before_retry_stops_attachment_and_gate(tmp_path: Path) -> None:
    class DisabledRetryReportClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.failed_once = False
            self.narrator_input: dict[str, Any] | None = None

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt.startswith(PROMPT_TEMPLATES["coordinator"].split("{recommended_agent_sequence}")[0]):
                if not results:
                    return self.result(call=ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "形成角色报告"}))
                if len(results) == 1 and not self.failed_once:
                    self.failed_once = True
                    return self.result("等待重试")
                if len(results) == 1:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                return self.result("不应在禁用后的重试中附加")
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_input = json.loads(messages[1]["content"])
            return await super().complete(system_prompt, messages, tools)

    model = DisabledRetryReportClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": [], "always_attach_report_agents": ["role_player"]},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "禁用报告来源重试"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": ["role_player"], "blocked_instruction_agents": [], "always_attach_report_agents": ["role_player"]},
        ).status_code == 200
        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry").status_code == 202
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert model.narrator_input is not None
    assert "report_documents" not in model.narrator_input


def test_report_name_error_can_be_corrected_without_creating_a_child(tmp_path: Path) -> None:
    report = "第一段完整意见。\n第二段保留原有标点：甲、乙；丙。"

    class ReportRetryClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.narrator_input: dict[str, Any] | None = None

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "形成完整角色报告"}))
                if len(results) == 1:
                    bad_ref = {**results[0]["report_ref"], "name": "写错的报告名称"}
                    return self.result(call=ToolCall(id="bad-ref", name="task", arguments={"agent": "narrator", "task": "依据随附报告写故事。", "report_refs": [bad_ref]}))
                if len(results) == 2:
                    assert results[1]["error"]["code"] == "report_name_mismatch"
                    return self.result(call=ToolCall(id="correct-ref", name="task", arguments={"agent": "narrator", "task": "依据随附报告写故事。", "report_refs": [results[0]["report_ref"]]}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                return self.result(report)
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_input = json.loads(messages[1]["content"])
                if not results:
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
                return self.result("叙事草稿已完成。")
            raise AssertionError(system_prompt)

    model = ReportRetryClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "报告引用纠错"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["narrative"] == STORY

    assert model.narrator_input is not None
    assert model.narrator_input["report_documents"][0]["content"] == report
    with app.state.saves.connect(save["id"]) as db:
        task_parts = db.execute(
            "SELECT state, child_session_id, output FROM parts WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?) AND tool_name = 'task' ORDER BY rowid",
            (created["turn_id"],),
        ).fetchall()
    assert [part["state"] for part in task_parts] == ["completed", "error", "completed"]
    assert task_parts[1]["child_session_id"] is None
    assert json.loads(task_parts[1]["output"])["error"]["code"] == "report_name_mismatch"


def test_required_entities_automatically_reach_all_transfer_receivers_without_research(tmp_path: Path) -> None:
    required_path = "mods/读心大师/entities/player_trait/读心能力/ENTITY.md"

    class RequiredTransferClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.inputs: dict[str, dict[str, Any]] = {}

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="erotic", name="task", arguments={"agent": "EroticOrNot", "task": "给出文本化意见"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "扮演伊蕾并报告反应"}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="style", name="task", arguments={"agent": "style_planner", "task": f"根据角色意见选择文风：{results[1]['result']}"}))
                if len(results) == 3:
                    task = """【本回合叙事目标】\n写出伊蕾的回应。\n【世界探索 Agent 意见】\n本回合未调用。\n【角色扮演 Agent 意见】\n伊蕾保持警惕。\n【文风 Agent 意见】\n本回合没有合适的全局文风。\n【其他叙事约束】\n不得替玩家行动。"""
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": task}))
                if len(results) == 4:
                    return self.result(call=ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == EROTIC_OR_NOT_PROMPT:
                self.inputs["EroticOrNot"] = json.loads(messages[1]["content"])
                return self.result("文本化意见")
            if system_prompt == ROLE_PLAYER_PROMPT:
                self.inputs["role_player"] = json.loads(messages[1]["content"])
                return self.result("伊蕾保持警惕，准备先观察玩家。")
            if system_prompt == STYLE_PLANNER_PROMPT:
                self.inputs["style_planner"] = json.loads(messages[1]["content"])
                return self.result(call=ToolCall(id="style-report", name="style_report", arguments={"report": "本回合没有合适的全局文风。", "style_paths": []}))
            if system_prompt == NARRATOR_PROMPT:
                self.inputs["narrator"] = json.loads(messages[1]["content"])
                if not results:
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
                return self.result("叙事草稿已完成。")
            raise AssertionError(system_prompt)

    model = RequiredTransferClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "必选实体自动传递", "mod_names": ["读心大师"]}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "我询问伊蕾。"})
        assert wait_for_turn(client, save["id"])["narrative"] == STORY

    assert set(model.inputs) == {"EroticOrNot", "role_player", "style_planner", "narrator"}
    for initial in model.inputs.values():
        assert initial["task"]["related_entities"] == [required_path]
        assert [document["path"] for document in initial["related_entity_documents"]] == [required_path]


def test_single_file_tools_reject_unsafe_files_and_edit_exactly_once(tmp_path: Path) -> None:
    root = tmp_path / "drafts"
    root.mkdir()
    files = DraftFileStore(root)

    files.write("narrative.md", "old old")
    with pytest.raises(DraftFileError, match="已经存在"):
        files.write("narrative.md", "replacement")
    with pytest.raises(DraftFileError, match="唯一"):
        files.edit("narrative.md", "old", "new")
    with pytest.raises(DraftFileError):
        files.read("../narrative.md")

    regular = root / "regular.md"
    regular.write_bytes(b"\xff")
    with pytest.raises(DraftFileError, match="UTF-8"):
        files.read("regular.md")

    target = root / "target.md"
    target.write_text("target", encoding="utf-8")
    (root / "link.md").symlink_to(target)
    with pytest.raises(DraftFileError, match="符号链接"):
        files.read("link.md")

    fifo = root / "pipe.md"
    os.mkfifo(fifo)
    with pytest.raises(DraftFileError, match="普通文件"):
        files.read("pipe.md")


def test_multiple_tool_calls_are_all_rejected_without_execution(tmp_path: Path) -> None:
    class MultipleCallsClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT and not any(message["role"] == "tool" for message in messages):
                calls = [
                    ToolCall(id="one", name="file_read", arguments={"path": "narrative.md"}),
                    ToolCall(id="two", name="narrative_publish", arguments={"path": "narrative.md"}),
                ]
                return ModelResult(content="", raw_response={}, tool_calls=calls)
            raise RuntimeError("stop after observing rejection")

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=MultipleCallsClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "多工具拒绝"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"

    with app.state.saves.connect(save["id"]) as db:
        rows = db.execute("SELECT provider_call_id, state, output FROM parts WHERE message_id IN (SELECT id FROM messages WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?)) AND type = 'tool' ORDER BY sequence", (created["turn_id"],)).fetchall()
    assert [row["provider_call_id"] for row in rows] == ["one", "two"]
    assert all(row["state"] == "error" for row in rows)
    assert all(json.loads(row["output"])["error"]["code"] == "multiple_tool_calls" for row in rows)


def test_eighth_consecutive_tool_failure_stops_the_agent(tmp_path: Path) -> None:
    class FailingToolsClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            assert system_prompt == COORDINATOR_PROMPT
            attempt = sum(message["role"] == "tool" for message in messages)
            return self.result(call=ToolCall(id=f"missing-{attempt}", name="file_read", arguments={"path": "narrative.md"}))

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=FailingToolsClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "连续工具失败"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"

    with app.state.saves.connect(save["id"]) as db:
        rows = db.execute(
            "SELECT state FROM parts WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?) AND tool_name = 'file_read' ORDER BY rowid",
            (created["turn_id"],),
        ).fetchall()
    assert [row["state"] for row in rows] == ["error"] * 8


def test_success_resets_failures_and_multi_tool_rejection_counts_once(tmp_path: Path) -> None:
    class ResettingFailureClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.root_step = 0

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == ROLE_PLAYER_PROMPT:
                return self.result("角色报告")
            if system_prompt != COORDINATOR_PROMPT:
                return await super().complete(system_prompt, messages, tools)
            step = self.root_step
            self.root_step += 1
            if step < 7:
                return ModelResult(content="", raw_response={}, tool_calls=[
                    ToolCall(id=f"read-{step}", name="file_read", arguments={"path": "narrative.md"}),
                    ToolCall(id=f"publish-{step}", name="narrative_publish", arguments={"path": "narrative.md"}),
                ])
            if step == 7:
                return self.result(call=ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "形成角色报告"}))
            if step < 15:
                return self.result(call=ToolCall(id=f"missing-{step}", name="file_read", arguments={"path": "narrative.md"}))
            if step == 15:
                return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
            return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ResettingFailureClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "失败计数归零"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["narrative"] == STORY


def test_more_than_thirty_two_successful_tools_can_complete(tmp_path: Path) -> None:
    class LongSuccessClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.searches = 0

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [message for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "充分调查世界"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                if self.searches < 33:
                    self.searches += 1
                    return self.result(call=ToolCall(id=f"search-{self.searches}", name="entity_search", arguments={"query": "伊蕾"}))
                return self.result(call=ToolCall(id="report", name="research_report", arguments={"report": "调查完成。", "related_entities": []}))
            return await super().complete(system_prompt, messages, tools)

    model = LongSuccessClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "长成功工具链"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["narrative"] == STORY

    assert model.searches == 33


def test_delegation_budget_rejects_only_the_extra_child_task(tmp_path: Path) -> None:
    class BudgetClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.role_calls = 0

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if len(results) < 3:
                    return self.result(call=ToolCall(id=f"role-{len(results)}", name="task", arguments={"agent": "role_player", "task": f"形成角色报告 {len(results)}"}))
                if len(results) == 3:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if len(results) == 4:
                    return self.result(call=ToolCall(id="over-budget", name="task", arguments={"agent": "role_player", "task": "超额任务"}))
                assert results[-1]["error"]["code"] == "validation_error"
                assert "委派预算" in results[-1]["error"]["message"]
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                self.role_calls += 1
                return self.result(content="角色报告")
            return await super().complete(system_prompt, messages, tools)

    model = BudgetClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        assert client.put("/api/runtime-settings", json={"max_delegations": 4, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": []}).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "委派预算"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == STORY
    assert model.role_calls == 3
    assert app.state.saves.task_count(save["id"], turn["id"]) == 4


def test_disabled_agent_is_omitted_rejected_and_does_not_block_narration(tmp_path: Path) -> None:
    expected_coordinator_prompt = coordinator_prompt(PROMPT_TEMPLATES["coordinator"], (AGENTS.get("narrator"),))

    class DisabledAgentClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.task_agents: list[str] = []
            self.role_calls = 0
            self.coordinator_prompts: list[str] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == expected_coordinator_prompt:
                self.coordinator_prompts.append(system_prompt)
                task = next(item["function"] for item in tools or [] if item["function"]["name"] == "task")
                self.task_agents = task["parameters"]["properties"]["agent"]["enum"]
                if not results:
                    return self.result(call=ToolCall(id="disabled", name="task", arguments={"agent": "role_player", "task": "不应执行"}))
                if len(results) == 1:
                    assert results[0]["error"]["code"] == "validation_error"
                    assert results[0]["error"]["message"] == "Agent 已禁用：role_player"
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                self.role_calls += 1
            return await super().complete(system_prompt, messages, tools)

    model = DisabledAgentClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": ["story_summarizer", "world_researcher", "character_designer", "location_designer", "compliance_reviewer", "EroticOrNot", "role_player", "style_planner", "consistency_checker"], "blocked_instruction_agents": [], "always_attach_report_agents": ["role_player"]},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "禁用辅助 Agent"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == STORY
    assert model.task_agents == ["narrator"]
    assert model.role_calls == 0
    recommendation = next(
        line for line in model.coordinator_prompts[0].splitlines()
        if line.startswith("本回合可参考以下专业 Agent 委派序列")
    )
    assert recommendation == "本回合可参考以下专业 Agent 委派序列：narrator。这只是推荐顺序；应根据实际任务自主跳过不需要的 Agent，也可以调整调用顺序。"
    coordinator_request = next(
        event for event in app.state.saves.list_events(save["id"])
        if event.type == "model.request" and event.payload["agent"] == "coordinator"
    )
    assert coordinator_request.payload["system_prompt"] == model.coordinator_prompts[0]


@pytest.mark.parametrize(("approved", "reason"), [
    (True, "固定审核通过"),
    (False, "固定审核不通过但不形成发布门禁"),
])
def test_compliance_reviewer_uses_fixed_response_without_model_call_and_passes_report(
    tmp_path: Path, approved: bool, reason: str
) -> None:
    class ComplianceFlowClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.called_prompts: list[str] = []
            self.role_payload: dict[str, Any] | None = None

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            self.called_prompts.append(system_prompt)
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(
                        id="compliance",
                        name="task",
                        arguments={"agent": "compliance_reviewer", "task": "审核本回合叙事方向"},
                    ))
                if len(results) == 1:
                    assert results[0]["result"] == {
                        "approved": approved,
                        "reason": reason,
                    }
                    return self.result(call=ToolCall(
                        id="role",
                        name="task",
                        arguments={
                            "agent": "role_player",
                            "task": "形成角色报告",
                            "report_refs": [results[0]["report_ref"]],
                        },
                    ))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if len(results) == 3:
                    return self.result(call=ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == ROLE_PLAYER_PROMPT:
                self.role_payload = json.loads(messages[1]["content"])
                return self.result(content="角色报告")
            if system_prompt == COMPLIANCE_REVIEWER_PROMPT:
                raise AssertionError("合规性审核 Agent 不应调用模型客户端")
            return await super().complete(system_prompt, messages, tools)

    model = ComplianceFlowClient()
    mock_response_dir = tmp_path / "mock_responses"
    mock_response_dir.mkdir()
    (mock_response_dir / "compliance_reviewer.json").write_text(
        json.dumps({"approved": approved, "reason": reason}, ensure_ascii=False),
        encoding="utf-8",
    )
    app = create_app(
        data_dir=make_data_dir(tmp_path),
        model_client=model,
        mock_response_dir=mock_response_dir,
    )
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "prefill_agents": ["compliance_reviewer"], "blocked_instruction_agents": []},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "固定审核响应"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert COMPLIANCE_REVIEWER_PROMPT not in model.called_prompts
    assert model.role_payload is not None
    document = model.role_payload["report_documents"][0]
    assert document["source_agent"] == "compliance_reviewer"
    assert json.loads(document["content"]) == {
        "approved": approved,
        "reason": reason,
    }
    events = app.state.saves.list_events(save["id"])
    fixed_event = next(event for event in events if event.type == "agent.fixed_response")
    assert fixed_event.payload["agent"] == "compliance_reviewer"
    assert fixed_event.payload["result"]["approved"] is approved
    assert not any(
        event.payload.get("agent") == "compliance_reviewer"
        for event in events
        if event.type in {"model.request", "model.response"}
    )
    child_messages = app.state.saves.load_session(save["id"], fixed_event.payload["session_id"])
    assistant = next(message for message in child_messages if message["role"] == "assistant")
    assert json.loads(assistant["model"])["response_source"] == "fixed_file"
    fixed_text = next(part["content"] for part in assistant["parts"] if part["type"] == "text")
    assert json.loads(fixed_text) == {"approved": approved, "reason": reason}


def test_retry_uses_current_disabled_agents(tmp_path: Path) -> None:
    expected_coordinator_prompt = coordinator_prompt(PROMPT_TEMPLATES["coordinator"], tuple(
        spec for spec in AGENTS.children() if spec.name != "role_player"
    ))

    class RetrySettingsClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.fail_once = True
            self.retry_task_agents: list[str] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt in {COORDINATOR_PROMPT, expected_coordinator_prompt}:
                if self.fail_once:
                    self.fail_once = False
                    raise RuntimeError("等待 retry")
                task = next(item["function"] for item in tools or [] if item["function"]["name"] == "task")
                self.retry_task_agents = task["parameters"]["properties"]["agent"]["enum"]
                if not results:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            return await super().complete(system_prompt, messages, tools)

    model = RetrySettingsClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "Retry 设置"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": ["role_player"], "blocked_instruction_agents": []},
        ).status_code == 200
        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry").status_code == 202
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert "role_player" not in model.retry_task_agents
    assert "narrator" in model.retry_task_agents


def test_invalid_prompt_blocks_new_turn_without_creating_it(tmp_path: Path) -> None:
    prompt_dir = make_prompt_dir(tmp_path)
    (prompt_dir / "coordinator.md").write_text("无动态占位符", encoding="utf-8")
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient(), prompt_dir=prompt_dir)

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "Prompt 校验"}).json()
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})

        assert response.status_code == 500
        assert "必须且只能包含一个" in response.json()["detail"]
        assert client.get(f"/api/saves/{save['id']}/turns").json() == []


def test_invalid_disabled_agent_files_do_not_block_turn(tmp_path: Path) -> None:
    disabled = {"role_player"}
    enabled_children = tuple(spec for spec in AGENTS.children() if spec.name not in disabled)
    enabled_coordinator_prompt = coordinator_prompt(PROMPT_TEMPLATES["coordinator"], enabled_children)

    class EnabledOnlyClient(NarrativeFileClient):
        bound_names: tuple[str, ...] = ()

        def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, ModelClient]:
            self.bound_names = names
            return super().bind_for_agents(names)

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == enabled_coordinator_prompt:
                system_prompt = COORDINATOR_PROMPT
            return await super().complete(system_prompt, messages, tools)

    prompt_dir = make_prompt_dir(tmp_path)
    (prompt_dir / "role_player.md").unlink()
    (prompt_dir / "task_descriptions" / "role_player.md").unlink()
    model = EnabledOnlyClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model, prompt_dir=prompt_dir)

    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": ["role_player"], "blocked_instruction_agents": []},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "禁用 Agent 文件"}).json()
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})

        assert response.status_code == 202
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    assert "role_player" not in model.bound_names


def test_invalid_prompt_blocks_retry_without_changing_failed_turn(tmp_path: Path) -> None:
    class FailingClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            raise RuntimeError("预期失败")

    prompt_dir = make_prompt_dir(tmp_path)
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=FailingClient(), prompt_dir=prompt_dir)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "Retry Prompt 校验"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"

        (prompt_dir / "coordinator.md").write_text("无动态占位符", encoding="utf-8")
        response = client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry")

        assert response.status_code == 500
        turns = client.get(f"/api/saves/{save['id']}/turns").json()
        assert len(turns) == 1
        assert turns[0]["status"] == "failed"


def test_prompt_files_are_snapshotted_once_per_execution(tmp_path: Path) -> None:
    prompt_dir = make_prompt_dir(tmp_path)

    class EditingClient(NarrativeFileClient):
        changed = False

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT and not self.changed:
                self.changed = True
                (prompt_dir / "narrator.md").write_text("本回合不应读取这次修改。", encoding="utf-8")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=EditingClient(), prompt_dir=prompt_dir)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "Prompt 快照"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == STORY


def test_active_prefill_is_appended_to_every_model_request_and_empty_reasoning_is_safe(tmp_path: Path) -> None:
    prefill_dir = make_prefill_dir(tmp_path)
    prefills = {
        "coordinator": "协调预填：",
        "world_researcher": "检索预填：",
        "narrator": "叙事预填：",
    }
    for agent, content in prefills.items():
        (prefill_dir / f"{agent}.md").write_text(content, encoding="utf-8")

    class PrefillClient(NarrativeFileClient):
        requests: list[tuple[str, dict[str, Any]]] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            agent = {
                COORDINATOR_PROMPT: "coordinator",
                WORLD_RESEARCHER_PROMPT: "world_researcher",
                NARRATOR_PROMPT: "narrator",
            }[system_prompt]
            assert messages[-1] == {"role": "assistant", "content": prefills[agent], "_prefix": True}
            self.requests.append((agent, messages[-1]))
            result = await super().complete(system_prompt, messages, tools)
            return result.model_copy(update={"reasoning": ""})

    model = PrefillClient()
    app = create_app(
        data_dir=make_data_dir(tmp_path),
        model_client=model,
        prefill_dir=prefill_dir,
    )
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={
                "max_delegations": 20,
                "context_turns": 4,
                "disabled_agents": [],
                "prefill_agents": list(prefills),
                "blocked_instruction_agents": [],
            },
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "尾部续写"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert {agent for agent, _message in model.requests} == set(prefills)
    with app.state.saves.connect(save["id"]) as db:
        assert db.execute("SELECT COUNT(*) FROM parts WHERE type = 'reasoning'").fetchone()[0] == 0
        stored_text = [row["content"] for row in db.execute("SELECT content FROM parts WHERE type = 'text'")]
    assert not set(prefills.values()) & set(stored_text)


def test_prefill_file_is_snapshotted_once_per_execution(tmp_path: Path) -> None:
    prefill_dir = make_prefill_dir(tmp_path)
    target = prefill_dir / "narrator.md"
    original = "原始叙事预填："
    replacement = "下一回合叙事预填："
    target.write_text(original, encoding="utf-8")

    class EditingPrefillClient(NarrativeFileClient):
        changed = False
        narrator_prefills: list[str] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT and not self.changed:
                self.changed = True
                target.write_text(replacement, encoding="utf-8")
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_prefills.append(messages[-1]["content"])
            return await super().complete(system_prompt, messages, tools)

    model = EditingPrefillClient()
    app = create_app(
        data_dir=make_data_dir(tmp_path),
        model_client=model,
        prefill_dir=prefill_dir,
    )
    with TestClient(app) as client:
        client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "prefill_agents": ["narrator"], "blocked_instruction_agents": []},
        )
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "预填快照"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"
        first_execution_count = len(model.narrator_prefills)

        client.post(f"/api/saves/{save['id']}/turns", json={"content": "再继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    assert first_execution_count > 0
    assert model.narrator_prefills[:first_execution_count] == [original] * first_execution_count
    assert model.narrator_prefills[first_execution_count:] == [replacement] * (
        len(model.narrator_prefills) - first_execution_count
    )


def test_empty_active_prefill_blocks_new_turn_without_creating_it(tmp_path: Path) -> None:
    prefill_dir = make_prefill_dir(tmp_path)
    (prefill_dir / "coordinator.md").write_text("", encoding="utf-8")
    app = create_app(
        data_dir=make_data_dir(tmp_path),
        model_client=NarrativeFileClient(),
        prefill_dir=prefill_dir,
    )

    with TestClient(app) as client:
        client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "prefill_agents": ["coordinator"], "blocked_instruction_agents": []},
        )
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "空预填校验"}).json()
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})

        assert response.status_code == 500
        assert "尾部续写文件不能为空" in response.json()["detail"]
        assert client.get(f"/api/saves/{save['id']}/turns").json() == []


def test_task_descriptions_are_snapshotted_once_per_execution(tmp_path: Path) -> None:
    prompt_dir = make_prompt_dir(tmp_path)
    target = prompt_dir / "task_descriptions" / "role_player.md"
    original = target.read_text(encoding="utf-8").strip()
    replacement = "下一次执行使用的新角色代理描述。"

    class EditingClient(NarrativeFileClient):
        changed = False
        role_descriptions: list[str] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                task = next(item["function"] for item in tools or [] if item["function"]["name"] == "task")
                role_line = next(line for line in task["description"].splitlines() if line.startswith("- role_player"))
                self.role_descriptions.append(role_line)
                if not self.changed:
                    self.changed = True
                    target.write_text(replacement, encoding="utf-8")
            return await super().complete(system_prompt, messages, tools)

    model = EditingClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model, prompt_dir=prompt_dir)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "task 描述快照"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"
        first_execution_count = len(model.role_descriptions)

        client.post(f"/api/saves/{save['id']}/turns", json={"content": "再继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    assert first_execution_count > 1
    assert all(original in line for line in model.role_descriptions[:first_execution_count])
    assert all(replacement in line for line in model.role_descriptions[first_execution_count:])


def test_capability_descriptions_are_snapshotted_once_per_execution(tmp_path: Path) -> None:
    prompt_dir = make_prompt_dir(tmp_path)
    target = prompt_dir / "capabilities" / "narrative_publish.md"
    original = target.read_text(encoding="utf-8").strip()
    replacement = "发布本回合的新描述。"

    class EditingClient(NarrativeFileClient):
        changed = False
        publish_descriptions: list[str] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                publish = next(item["function"] for item in tools or [] if item["function"]["name"] == "narrative_publish")
                self.publish_descriptions.append(publish["description"])
                if not self.changed:
                    self.changed = True
                    target.write_text(replacement, encoding="utf-8")
            return await super().complete(system_prompt, messages, tools)

    model = EditingClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model, prompt_dir=prompt_dir)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "Capability 描述快照"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"
        first_execution_count = len(model.publish_descriptions)

        client.post(f"/api/saves/{save['id']}/turns", json={"content": "再继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    assert first_execution_count > 1
    assert model.publish_descriptions[:first_execution_count] == [original] * first_execution_count
    assert model.publish_descriptions[first_execution_count:] == [replacement] * (
        len(model.publish_descriptions) - first_execution_count
    )


def test_empty_capability_description_blocks_new_turn_without_creating_it(tmp_path: Path) -> None:
    prompt_dir = make_prompt_dir(tmp_path)
    (prompt_dir / "capabilities" / "narrative_publish.md").write_text("", encoding="utf-8")
    app = create_app(
        data_dir=make_data_dir(tmp_path),
        model_client=NarrativeFileClient(),
        prompt_dir=prompt_dir,
    )

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "空 Capability 描述校验"}).json()
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})

        assert response.status_code == 500
        assert "Capability 描述文件不能为空" in response.json()["detail"]
        assert client.get(f"/api/saves/{save['id']}/turns").json() == []


def test_runtime_settings_are_frozen_during_one_execution(tmp_path: Path) -> None:
    class FrozenSettingsClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.runtime_settings: RuntimeSettingsStore | None = None
            self.task_agent_snapshots: list[list[str]] = []
            self.narrator_tasks: list[str] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                task = next(item["function"] for item in tools or [] if item["function"]["name"] == "task")
                self.task_agent_snapshots.append(task["parameters"]["properties"]["agent"]["enum"])
                if not results:
                    assert self.runtime_settings is not None
                    self.runtime_settings.update(RuntimeSettings(max_delegations=20, context_turns=4, disabled_agents=["role_player"], blocked_instruction_agents=["narrator"]))
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_tasks.append(json.loads(messages[1]["content"])["task"]["task"])
            return await super().complete(system_prompt, messages, tools)

    model = FrozenSettingsClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    model.runtime_settings = app.state.runtime_settings
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "执行设置快照"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert len(model.task_agent_snapshots) >= 2
    assert all("role_player" in agents for agents in model.task_agent_snapshots)
    assert app.state.runtime_settings.get().disabled_agents == ["role_player"]
    assert app.state.runtime_settings.get().blocked_instruction_agents == ["narrator"]
    assert model.narrator_tasks == ["写故事", "写故事"]


def test_failed_child_is_explicitly_continued_with_same_task_id(tmp_path: Path) -> None:
    class BoundGenerationClient:
        def __init__(self, owner: Any, model: str) -> None:
            self.owner = owner
            self.model = model

        def metadata(self) -> dict[str, str | int | float]:
            return {"provider": "test", "model": self.model, "base_url": "local", "timeout_seconds": 1}

        def bind(self) -> ModelClient:
            return self

        def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, ModelClient]:
            return {name: self for name in names}

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            return await self.owner.complete(system_prompt, messages, tools)

    class ContinueClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.narrator_calls = 0
            self.narrator_tasks: list[str] = []
            self.stop_parent_once = True
            self.current_model = "before-retry"

        def metadata(self) -> dict[str, str | int | float]:
            return {"provider": "test", "model": self.current_model, "base_url": "local", "timeout_seconds": 1}

        def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, ModelClient]:
            return {name: BoundGenerationClient(self, self.current_model) for name in names}

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_calls += 1
                latest_user = next(message for message in reversed(messages) if message["role"] == "user")
                self.narrator_tasks.append(json.loads(latest_user["content"])["task"]["task"])
                if self.narrator_calls == 1:
                    raise RuntimeError("narrator model failed")
                if not results:
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
                return self.result("叙事草稿已完成。")
            if system_prompt == COORDINATOR_PROMPT:
                task_errors = [result for result in results if result.get("error", {}).get("code") == "child_failed"]
                task_successes = [result for result in results if result.get("ok") is True and result.get("agent") == "narrator"]
                reads = [result for result in results if result.get("path") == "narrative.md" and "content" in result and "status" not in result]
                if not results:
                    return self.result(call=ToolCall(id="task-1", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                if task_errors and not task_successes:
                    if self.stop_parent_once:
                        self.stop_parent_once = False
                        raise RuntimeError("wait for explicit user continuation")
                    return self.result(call=ToolCall(id="task-2", name="task", arguments={"agent": "narrator", "task": "继续写故事", "task_id": task_errors[-1]["task_id"]}))
                if task_successes and not reads:
                    return self.result(call=ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            raise AssertionError(system_prompt)

    model = ContinueClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "显式续接"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"
        assert model.narrator_calls == 1
        with app.state.saves.connect(save["id"]) as db:
            before = db.execute("SELECT COUNT(*) FROM messages WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?)", (created["turn_id"],)).fetchone()[0]
            db.execute("DELETE FROM events")

        model.current_model = "after-retry"
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": ["narrator"]},
        ).status_code == 200
        assert client.post(f"/api/saves/{save['id']}/turns/{created['turn_id']}/retry").status_code == 202
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    with app.state.saves.connect(save["id"]) as db:
        task_parts = db.execute("SELECT child_session_id, state, output FROM parts WHERE tool_name = 'task' ORDER BY rowid").fetchall()
        after = db.execute("SELECT COUNT(*) FROM messages WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?)", (created["turn_id"],)).fetchone()[0]
        model_rows = db.execute(
            "SELECT sessions.agent, messages.model FROM messages JOIN sessions ON sessions.id = messages.session_id WHERE messages.role = 'assistant' AND messages.model IS NOT NULL ORDER BY messages.rowid"
        ).fetchall()
    models = [json.loads(row["model"])["model"] for row in model_rows]
    latest_models = {agent: next(json.loads(row["model"])["model"] for row in reversed(model_rows) if row["agent"] == agent) for agent in ("coordinator", "narrator")}
    assert model.narrator_calls == 3
    assert model.narrator_tasks == ["写故事", "请开始工作", "请开始工作"]
    assert len({row["child_session_id"] for row in task_parts}) == 1
    assert json.loads(task_parts[0]["output"])["task_id"] == task_parts[1]["child_session_id"]
    assert after > before
    assert "before-retry" in models and "after-retry" in models
    assert latest_models == {"coordinator": "after-retry", "narrator": "after-retry"}


def test_successful_child_can_continue_or_restart_with_full_context(tmp_path: Path) -> None:
    class ReuseClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.research_inputs: list[dict[str, Any]] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="first", name="task", arguments={"agent": "world_researcher", "task": "首次调查", "reason": "需要初始资料"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="continue", name="task", arguments={"agent": "world_researcher", "task": "结合刚才结果继续调查", "reason": "扩大调查范围", "task_id": results[0]["task_id"]}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="fresh", name="task", arguments={"agent": "world_researcher", "task": "用全新视角调查"}))
                if len(results) == 3:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                latest = json.loads(next(message["content"] for message in reversed(messages) if message["role"] == "user"))
                self.research_inputs.append(latest)
                return self.result(call=ToolCall(id=f"report-{len(self.research_inputs)}", name="research_report", arguments={"report": f"已完成：{latest['task']['task']}", "related_entities": []}))
            return await super().complete(system_prompt, messages, tools)

    model = ReuseClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        assert client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 4, "disabled_agents": [], "blocked_instruction_agents": ["world_researcher"]},
        ).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "复用子代理"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    with app.state.saves.connect(save["id"]) as db:
        task_ids = [row["child_session_id"] for row in db.execute("SELECT child_session_id FROM parts WHERE tool_name = 'task' ORDER BY rowid").fetchall()]
        reused_messages = db.execute("SELECT role FROM messages WHERE session_id = ? ORDER BY sequence", (task_ids[0],)).fetchall()
    assert task_ids[0] == task_ids[1]
    assert task_ids[2] != task_ids[0]
    assert [row["role"] for row in reused_messages] == ["user", "assistant", "user", "assistant"]
    assert model.research_inputs[1]["continuation"] is True
    assert [item["task"]["task"] for item in model.research_inputs] == ["请开始工作"] * 3
    assert [item["task"]["reason"] for item in model.research_inputs] == [""] * 3


def test_retrieval_counter_accumulates_per_researcher_session(tmp_path: Path) -> None:
    class RetrievalCounterClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.counters: dict[str, list[dict[str, int]]] = {"first": [], "continued": [], "fresh": []}

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="first", name="task", arguments={"agent": "world_researcher", "task": "首次调查"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="continue", name="task", arguments={"agent": "world_researcher", "task": "继续调查", "task_id": results[0]["task_id"]}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="fresh", name="task", arguments={"agent": "world_researcher", "task": "重新调查"}))
                if len(results) == 3:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                task = json.loads(next(message["content"] for message in reversed(messages) if message["role"] == "user"))["task"]["task"]
                last_user = max(index for index, message in enumerate(messages) if message["role"] == "user")
                current_results = [json.loads(message["content"]) for message in messages[last_user + 1 :] if message["role"] == "tool"]
                if current_results:
                    warning = current_results[-1]["retrieval_counter_warning"]
                    assert "计数器用于帮助你了解小型世界观中当前已阅读资料的规模" in warning
                    assert "禁止将100%阅读作为参考指标" in warning
                if task == "首次调查":
                    if not current_results:
                        return self.result(call=ToolCall(id="search-1", name="entity_search", arguments={"query": "伊蕾 暮潮旅店 近海盐船"}))
                    self.counters["first"].append(current_results[-1]["retrieval_counter"])
                    if "results" in current_results[-1]:
                        return self.result(call=ToolCall(id="read-1", name="entity_read", arguments={"paths": [
                            "entities/character/伊蕾/ENTITY.md",
                            "entities/character/奥伦/ENTITY.md",
                        ]}))
                    return self.result(call=ToolCall(id="report-1", name="research_report", arguments={"report": "首次调查完成", "related_entities": ["entities/character/伊蕾/ENTITY.md", "entities/character/奥伦/ENTITY.md"]}))
                if task == "继续调查":
                    if not current_results:
                        return self.result(call=ToolCall(id="search-2", name="entity_search", arguments={"query": "奥伦 暮潮旅店 守灯人失踪"}))
                    self.counters["continued"].append(current_results[-1]["retrieval_counter"])
                    if "results" in current_results[-1]:
                        return self.result(call=ToolCall(id="read-2", name="entity_read", arguments={"paths": [
                            "entities/character/奥伦/ENTITY.md",
                            "entities/event/失踪的守灯人/ENTITY.md",
                        ]}))
                    return self.result(call=ToolCall(id="report-2", name="research_report", arguments={"report": "继续调查完成", "related_entities": ["entities/character/奥伦/ENTITY.md", "entities/event/失踪的守灯人/ENTITY.md"]}))
                if not current_results:
                    return self.result(call=ToolCall(id="search-fresh", name="entity_search", arguments={"query": "伊蕾"}))
                self.counters["fresh"].append(current_results[-1]["retrieval_counter"])
                return self.result(call=ToolCall(id="report-fresh", name="research_report", arguments={"report": "重新调查完成", "related_entities": []}))
            return await super().complete(system_prompt, messages, tools)

    model = RetrievalCounterClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "检索计数"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "调查。"})
        assert wait_for_turn(client, save["id"], timeout=20)["status"] == "completed"

    assert model.counters == {
        "first": [
            {"total_searchable_entities": 4, "read_entities": 0},
            {"total_searchable_entities": 4, "read_entities": 2},
        ],
        "continued": [
            {"total_searchable_entities": 4, "read_entities": 2},
            {"total_searchable_entities": 4, "read_entities": 3},
        ],
        "fresh": [
            {"total_searchable_entities": 4, "read_entities": 0},
        ],
    }


def test_restart_turns_unfinished_tool_into_visible_interruption(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=NarrativeFileClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="重启"))
    turn = app.state.saves.create_turn(save.id, "继续。", 4)
    session_id = app.state.saves.root_session_id(save.id, turn.id)
    message_id = app.state.saves.create_assistant_message(save.id, session_id, {"model": "test"})
    part = app.state.saves.record_model_result(save.id, message_id, "", None, [{"id": "pending", "name": "file_read", "arguments": {"path": "narrative.md"}}])[0]
    app.state.saves.set_tool_running(save.id, part["id"])

    restarted = create_app(data_dir=data_dir, model_client=NarrativeFileClient())
    with TestClient(restarted):
        pass
    assert restarted.state.saves.get_turn(save.id, turn.id).status == "interrupted"
    loaded = restarted.state.saves.load_session(save.id, session_id)
    tool = next(part for message in loaded for part in message["parts"] if part["type"] == "tool")
    assert tool["state"] == "error"
    assert json.loads(tool["output"])["error"]["code"] == "execution_interrupted"


def test_narrative_publish_transaction_rolls_back_as_one_unit(tmp_path: Path) -> None:
    class RollbackClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
                if results and results[-1].get("error", {}).get("code") == "tool_error":
                    raise RuntimeError("stop after publication failure")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=RollbackClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "发布回滚"}).json()
        with app.state.saves.connect(save["id"]) as db:
            db.execute("CREATE TRIGGER reject_story BEFORE INSERT ON messages WHEN NEW.kind = 'story' BEGIN SELECT RAISE(ABORT, 'reject story'); END")
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["status"] == "failed"

    with app.state.saves.connect(save["id"]) as db:
        turn = db.execute("SELECT narrative FROM turns WHERE id = ?", (created["turn_id"],)).fetchone()
        assert turn["narrative"] is None
        assert db.execute("SELECT COUNT(*) FROM messages WHERE kind = 'story'").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM events WHERE type = 'narrative.delta'").fetchone()[0] == 0


def test_automatic_publish_text_replacement_rolls_back_with_story(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="自动发布回滚"))
    turn = app.state.saves.create_turn(save.id, "继续。", 4)
    root = app.state.saves.root_session_id(save.id, turn.id)
    message = app.state.saves.create_assistant_message(save.id, root, {"model": "test"})
    app.state.saves.record_model_result(save.id, message, "模型原文", None, [])
    part = app.state.saves.create_automatic_publish_part(save.id, message)
    DraftFileStore(app.state.saves.draft_dir(save.id, turn.id)).write("narrative.md", STORY)
    with app.state.saves.connect(save.id) as db:
        db.execute("CREATE TRIGGER reject_story BEFORE INSERT ON messages WHEN NEW.kind = 'story' BEGIN SELECT RAISE(ABORT, 'reject story'); END")

    with pytest.raises(sqlite3.IntegrityError, match="reject story"):
        app.state.saves.publish_narrative(
            save.id,
            turn.id,
            root,
            part["id"],
            1,
            message,
            "已自动发布 narrative.md。",
        )

    with app.state.saves.connect(save.id) as db:
        assert db.execute("SELECT status FROM turns WHERE id = ?", (turn.id,)).fetchone()["status"] == "running"
        assert db.execute("SELECT content FROM parts WHERE message_id = ? AND type = 'text'", (message,)).fetchone()["content"] == "模型原文"
        assert db.execute("SELECT state FROM parts WHERE id = ?", (part["id"],)).fetchone()["state"] == "running"


def test_publish_response_loss_does_not_duplicate_story(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    original = app.state.saves.publish_narrative
    lost = False

    def publish_then_lose(*args: Any, **kwargs: Any):
        nonlocal lost
        result = original(*args, **kwargs)
        if not lost:
            lost = True
            raise RuntimeError("response lost after commit")
        return result

    monkeypatch.setattr(app.state.saves, "publish_narrative", publish_then_lose)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "响应丢失"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == STORY
    with app.state.saves.connect(save["id"]) as db:
        assert db.execute("SELECT COUNT(*) FROM messages WHERE kind = 'story'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM events WHERE type = 'narrative.delta'").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM parts WHERE tool_name = 'narrative_publish' AND state = 'completed'").fetchone()[0] == 1


@pytest.mark.parametrize(("window", "expected"), [("before", "old"), ("after", "new")])
def test_atomic_draft_is_complete_when_process_is_killed(tmp_path: Path, window: str, expected: str) -> None:
    root = tmp_path / "drafts"
    root.mkdir()
    (root / "narrative.md").write_text("old", encoding="utf-8")
    marker = tmp_path / "window"
    script = """
import sys, time
from pathlib import Path
import rpera.draft_files as module

root, marker, window = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
original = module.os.replace
def controlled_replace(source, target):
    if window == "before":
        marker.write_text("ready", encoding="utf-8")
        while True: time.sleep(1)
    original(source, target)
    marker.write_text("ready", encoding="utf-8")
    while True: time.sleep(1)
module.os.replace = controlled_replace
module.DraftFileStore(root).edit("narrative.md", "old", "new")
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(root), str(marker), window])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.01)
        assert marker.exists()
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    assert (root / "narrative.md").read_text(encoding="utf-8") == expected


def test_sqlite_session_history_survives_process_kill(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    marker = tmp_path / "committed"
    script = """
import sys, time
from pathlib import Path
from rpera.app import create_app
from rpera.models import SaveCreate

data, marker = Path(sys.argv[1]), Path(sys.argv[2])
app = create_app(data_dir=data, model_client=object())
save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="kill"))
turn = app.state.saves.create_turn(save.id, "继续。", 4)
marker.write_text(save.id + "\\n" + turn.id, encoding="utf-8")
while True: time.sleep(1)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(data_dir), str(marker)])
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.01)
        assert marker.exists()
        save_id, turn_id = marker.read_text(encoding="utf-8").splitlines()
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    restarted = create_app(data_dir=data_dir, model_client=NarrativeFileClient())
    with TestClient(restarted):
        pass
    with restarted.state.saves.connect(save_id) as db:
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE turn_id = ?", (turn_id,)).fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM messages WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?)", (turn_id,)).fetchone()[0] == 1
    assert restarted.state.saves.get_turn(save_id, turn_id).status == "interrupted"


def test_http_turn_continues_after_kill_between_file_replace_and_tool_result(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    marker = tmp_path / "file-published"
    identifiers = tmp_path / "ids"
    script = """
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
from fastapi.testclient import TestClient
from rpera.app import create_app
from test_runtime_batch_one import NarrativeFileClient, STORY

data, marker, identifiers = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
app = create_app(data_dir=data, model_client=NarrativeFileClient())
original = app.state.saves.complete_tool
def block_before_result(save_id, part_id, output):
    if output.get("path") == "narrative.md" and output.get("content") == STORY:
        marker.write_text("ready", encoding="utf-8")
        while True: time.sleep(1)
    return original(save_id, part_id, output)
app.state.saves.complete_tool = block_before_result
with TestClient(app) as client:
    save = client.post("/api/saves", json={"world_name": "雾港", "name": "kill window"}).json()
    turn = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
    identifiers.write_text(save["id"] + "\\n" + turn["turn_id"], encoding="utf-8")
    while True: time.sleep(1)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(data_dir), str(marker), str(identifiers)], cwd=PROJECT_ROOT)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.01)
        assert marker.exists()
        save_id, turn_id = identifiers.read_text(encoding="utf-8").splitlines()
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    class ResumeClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                resumed = [result for result in results if result.get("ok") is True and result.get("agent") == "narrator"]
                reads = [result for result in results if result.get("path") == "narrative.md" and "status" not in result and "content" in result]
                interrupted = [result for result in results if result.get("error", {}).get("code") == "execution_interrupted" and result.get("task_id")]
                if not resumed:
                    return self.result(call=ToolCall(id="resume-task", name="task", arguments={"agent": "narrator", "task": "检查文件并继续", "task_id": interrupted[-1]["task_id"]}))
                if not reads:
                    return self.result(call=ToolCall(id="root-read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == NARRATOR_PROMPT:
                completed_reads = [result for result in results if result.get("path") == "narrative.md" and result.get("content") == STORY]
                if not completed_reads:
                    return self.result(call=ToolCall(id="child-read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result("叙事草稿已完成。")
            raise AssertionError(system_prompt)

    restarted = create_app(data_dir=data_dir, model_client=ResumeClient())
    with TestClient(restarted) as client:
        assert restarted.state.saves.get_turn(save_id, turn_id).status == "interrupted"
        draft = data_dir / "saves" / save_id / "current" / "turns" / turn_id / "drafts" / "narrative.md"
        assert draft.read_text(encoding="utf-8") == STORY
        assert client.post(f"/api/saves/{save_id}/turns/{turn_id}/retry").status_code == 202
        assert wait_for_turn(client, save_id)["narrative"] == STORY
    with restarted.state.saves.connect(save_id) as db:
        assert db.execute("SELECT COUNT(*) FROM messages WHERE kind = 'story'").fetchone()[0] == 1
        states = db.execute("SELECT parts.tool_name, parts.state FROM parts JOIN sessions ON sessions.id = parts.session_id WHERE parts.type = 'tool' AND sessions.agent = 'narrator' ORDER BY parts.rowid").fetchall()
    assert [(row["tool_name"], row["state"]) for row in states] == [("file_write", "error"), ("file_read", "completed")]


def test_http_turn_continues_after_kill_before_file_replace(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    marker = tmp_path / "before-file"
    identifiers = tmp_path / "ids"
    script = """
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "tests"))
from fastapi.testclient import TestClient
from rpera.app import create_app
from rpera.draft_files import DraftFileStore
from test_runtime_batch_one import NarrativeFileClient

data, marker, identifiers = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
app = create_app(data_dir=data, model_client=NarrativeFileClient())
original = DraftFileStore._replace
def block_before_replace(self, path, content):
    marker.write_text("ready", encoding="utf-8")
    while True: time.sleep(1)
DraftFileStore._replace = block_before_replace
with TestClient(app) as client:
    save = client.post("/api/saves", json={"world_name": "雾港", "name": "kill before file"}).json()
    turn = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
    identifiers.write_text(save["id"] + "\\n" + turn["turn_id"], encoding="utf-8")
    while True: time.sleep(1)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(data_dir), str(marker), str(identifiers)], cwd=PROJECT_ROOT)
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.01)
        assert marker.exists()
        save_id, turn_id = identifiers.read_text(encoding="utf-8").splitlines()
        process.kill()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    class ResumeBeforeClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                resumed = [result for result in results if result.get("ok") is True and result.get("agent") == "narrator"]
                root_reads = [result for result in results if result.get("path") == "narrative.md" and result.get("content") == STORY]
                interrupted = [result for result in results if result.get("error", {}).get("code") == "execution_interrupted" and result.get("task_id")]
                if not resumed:
                    return self.result(call=ToolCall(id="resume-task", name="task", arguments={"agent": "narrator", "task": "检查文件并继续", "task_id": interrupted[-1]["task_id"]}))
                if not root_reads:
                    return self.result(call=ToolCall(id="root-read", name="file_read", arguments={"path": "narrative.md"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == NARRATOR_PROMPT:
                completed = [result for result in results if result.get("path") == "narrative.md" and result.get("content") == STORY]
                failed_read = any(result.get("error", {}).get("message", "").find("不存在") >= 0 for result in results)
                if completed:
                    return self.result("叙事草稿已完成。")
                if failed_read:
                    return self.result(call=ToolCall(id="rewrite", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
                return self.result(call=ToolCall(id="check", name="file_read", arguments={"path": "narrative.md"}))
            raise AssertionError(system_prompt)

    restarted = create_app(data_dir=data_dir, model_client=ResumeBeforeClient())
    with TestClient(restarted) as client:
        draft = data_dir / "saves" / save_id / "current" / "turns" / turn_id / "drafts" / "narrative.md"
        assert not draft.exists()
        assert client.post(f"/api/saves/{save_id}/turns/{turn_id}/retry").status_code == 202
        assert wait_for_turn(client, save_id)["narrative"] == STORY
    assert draft.read_text(encoding="utf-8") == STORY


def test_child_cannot_invoke_task_by_hallucinating_its_name(tmp_path: Path) -> None:
    class ChildTaskClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
                if not results:
                    return self.result(call=ToolCall(id="forbidden", name="task", arguments={"agent": "narrator", "task": "越权"}))
                assert results[-1]["error"]["code"] == "validation_error"
                return self.result(call=ToolCall(id="report", name="research_report", arguments={"report": "越权委派被拒绝。", "related_entities": []}))
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ChildTaskClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "静态权限"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"
    with app.state.saves.connect(save["id"]) as db:
        agents = [row["agent"] for row in db.execute("SELECT agent FROM sessions ORDER BY rowid")]
        forbidden = db.execute("SELECT state FROM parts WHERE provider_call_id = 'forbidden'").fetchone()
    assert agents == ["coordinator", "world_researcher", "narrator"]
    assert forbidden["state"] == "error"


def test_narrator_entities_require_current_structured_research_report(tmp_path: Path) -> None:
    path = "entities/character/伊蕾/ENTITY.md"

    class UnreportedEntityClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "调查"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写故事", "related_entities": [path]}))
                assert results[-1]["error"]["code"] == "validation_error"
                return self.result("停止")
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                return self.result(call=ToolCall(id="report", name="research_report", arguments={"report": "没有使用实体。", "related_entities": []}))
            raise AssertionError("未授权实体不应创建 narrator")

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=UnreportedEntityClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "严格实体来源"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "failed"
    with app.state.saves.connect(save["id"]) as db:
        agents = [row["agent"] for row in db.execute("SELECT agent FROM sessions ORDER BY rowid")]
        rejected = json.loads(db.execute("SELECT output FROM parts WHERE provider_call_id = 'narrate'").fetchone()["output"])
    assert agents == ["coordinator", "world_researcher"]
    assert "未由当前 researcher 报告" in rejected["error"]["message"]


def test_world_researcher_plain_text_is_not_a_structured_report(tmp_path: Path) -> None:
    class PlainResearchClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "调查"}))
                assert results[-1]["error"]["code"] == "child_failed"
                return self.result("停止")
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                return self.result('{"report":"看似结构化","related_entities":[]}')
            raise AssertionError(system_prompt)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=PlainResearchClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "拒绝文本报告"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "failed"
    with app.state.saves.connect(save["id"]) as db:
        report_tools = db.execute("SELECT COUNT(*) FROM parts WHERE tool_name = 'research_report'").fetchone()[0]
    assert report_tools == 0


def test_world_researcher_can_report_only_entities_it_received_or_read(tmp_path: Path) -> None:
    path = "entities/character/伊蕾/ENTITY.md"

    class UnreadEntityClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="invalid-report", name="research_report", arguments={"report": "未经读取。", "related_entities": [path]}))
                assert results[-1]["error"]["code"] == "tool_error"
                return self.result(call=ToolCall(id="valid-report", name="research_report", arguments={"report": "没有引用实体。", "related_entities": []}))
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=UnreadEntityClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "限制报告来源"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    with app.state.saves.connect(save["id"]) as db:
        invalid = json.loads(db.execute("SELECT output FROM parts WHERE provider_call_id = 'invalid-report'").fetchone()["output"])
        valid = json.loads(db.execute("SELECT output FROM parts WHERE provider_call_id = 'valid-report'").fetchone()["output"])
    assert "未获得的实体" in invalid["error"]["message"]
    assert valid == {"report": "没有引用实体。", "related_entities": []}


def test_child_creation_and_parent_attachment_roll_back_together(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="child transaction"))
    turn = app.state.saves.create_turn(save.id, "继续。", 4)
    root = app.state.saves.root_session_id(save.id, turn.id)
    message = app.state.saves.create_assistant_message(save.id, root, {"model": "test"})
    part = app.state.saves.record_model_result(save.id, message, "", None, [{"id": "task", "name": "task", "arguments": {"agent": "narrator", "task": "写故事"}}])[0]
    app.state.saves.set_tool_running(save.id, part["id"])
    with app.state.saves.connect(save.id) as db:
        db.execute("CREATE TRIGGER reject_child_link BEFORE UPDATE OF child_session_id ON parts WHEN NEW.child_session_id IS NOT NULL BEGIN SELECT RAISE(ABORT, 'reject link'); END")

    with pytest.raises(sqlite3.IntegrityError, match="reject link"):
        app.state.saves.prepare_child_task(save.id, turn.id, root, part["id"], "narrator", {"task": "写故事", "reason": ""}, None)

    with app.state.saves.connect(save.id) as db:
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE turn_id = ?", (turn.id,)).fetchone()[0] == 1
        assert db.execute("SELECT child_session_id FROM parts WHERE id = ?", (part["id"],)).fetchone()["child_session_id"] is None


def test_configured_context_turns_apply_to_coordinator_and_child_agents(tmp_path: Path) -> None:
    class HistoryClient(NarrativeFileClient):
        def __init__(self) -> None:
            self.root_inputs: list[list[dict[str, Any]]] = []
            self.narrator_inputs: list[dict[str, Any]] = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == NARRATOR_PROMPT:
                story_context = json.loads(messages[0]["content"])
                self.narrator_inputs.append(story_context)
                if not results:
                    current = story_context["story"]["turns"][-1]["player"]["content"]
                    story = f"故事：{current}"
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": story}))
                return self.result(content="叙事草稿已完成。")
            if system_prompt == COORDINATOR_PROMPT and not any(message["role"] == "tool" for message in messages):
                self.root_inputs.append(messages)
                if messages[-2]["content"] == "失败回合":
                    raise RuntimeError("planned failure")
            return await super().complete(system_prompt, messages, tools)

    model = HistoryClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        response = client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 2, "disabled_agents": [], "blocked_instruction_agents": []},
        )
        assert response.status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "故事历史"}).json()
        for index in range(5):
            client.post(f"/api/saves/{save['id']}/turns", json={"content": f"成功回合 {index}"})
            assert wait_for_turn(client, save["id"])["status"] == "completed"
            if index == 1:
                client.post(f"/api/saves/{save['id']}/turns", json={"content": "失败回合"})
                assert wait_for_turn(client, save["id"])["status"] == "failed"
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "检查历史"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    final_input = next(item for item in model.root_inputs if item[-2]["content"] == "检查历史")
    expected_history = [
        message
        for index in range(3, 5)
        for message in (
            {"speaker": "player", "content": f"成功回合 {index}"},
            {"speaker": "RPera", "content": f"故事：成功回合 {index}"},
        )
    ]
    expected_messages = [
        {"role": "system", "content": "<recent_story>"},
        {"role": "assistant", "content": save["opening"]},
        *[
            {"role": "user" if message["speaker"] == "player" else "assistant", "content": message["content"]}
            for message in expected_history
        ],
        {"role": "user", "content": "检查历史"},
        {"role": "system", "content": "</recent_story>"},
    ]
    assert final_input == expected_messages
    final_narrator_input = next(
        item for item in model.narrator_inputs
        if item["story"]["turns"][-1]["player"]["content"] == "检查历史"
    )
    assert final_narrator_input["story"] == {
        "opening": save["opening"],
        "turns": [
            {
                "turn_number": index + 2,
                "player": {"content": f"成功回合 {index}", "images": []},
                "has_ai_output": True,
                "narrative": f"故事：成功回合 {index}",
            }
            for index in range(3, 5)
        ] + [{
            "turn_number": 7,
            "player": {"content": "检查历史", "images": []},
            "has_ai_output": False,
        }],
    }


def test_context_turns_above_sqlite_integer_limit_remain_unbounded(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    with TestClient(app) as client:
        response = client.put(
            "/api/runtime-settings",
            json={"max_delegations": 20, "context_turns": 10**30, "disabled_agents": [], "blocked_instruction_agents": []},
        )
        assert response.status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "无上限历史"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"


def test_story_keeps_user_only_turns_and_retry_refreezes_current_window(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    saves = app.state.saves
    save = saves.create_save(SaveCreate(world_name="雾港", name="完整正文记录"))

    completed: list[Any] = []
    for number in range(1, 4):
        turn = saves.create_turn(save.id, f"已完成输入 {number}", 2)
        with saves.connect(save.id) as db:
            db.execute(
                "UPDATE turns SET status = 'completed', narrative = ?, completed_at = created_at WHERE id = ?",
                (f"正式正文 {number}", turn.id),
            )
        completed.append(turn)

    failed = saves.create_turn(save.id, "没有 AI 输出的输入", 2)
    saves.fail_turn(save.id, failed.id, "turn.failed", {"message": "测试失败"})
    assert [turn["turn_number"] for turn in saves.load_story(save.id, failed.id)["story"]["turns"]] == [2, 3, 4]

    retried = saves.retry_turn(save.id, failed.id, 3)
    assert retried.id == failed.id
    assert saves.load_story(save.id, failed.id)["story"]["turns"] == [
        {
            "turn_number": number,
            "player": {"content": f"已完成输入 {number}", "images": []},
            "has_ai_output": True,
            "narrative": f"正式正文 {number}",
        }
        for number in range(1, 4)
    ] + [{
        "turn_number": 4,
        "player": {"content": "没有 AI 输出的输入", "images": []},
        "has_ai_output": False,
    }]


def test_new_input_after_failure_preserves_user_turn_and_blocks_older_retry(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    saves = app.state.saves
    save = saves.create_save(SaveCreate(world_name="雾港", name="失败后继续输入"))
    failed = saves.create_turn(save.id, "第一次输入", 2)
    saves.fail_turn(save.id, failed.id, "turn.failed", {"message": "测试失败"})

    following = saves.create_turn(save.id, "直接补充输入", 2)
    story_before = saves.load_story(save.id, failed.id)
    last_played_before = saves.get_save(save.id).last_played_at
    with pytest.raises(TurnRetryError, match="最后一个"):
        saves.retry_turn(save.id, failed.id, 2)
    assert saves.load_story(save.id, failed.id) == story_before
    assert saves.get_turn(save.id, failed.id).status == "failed"
    assert saves.get_save(save.id).last_played_at == last_played_before
    assert saves.load_story(save.id, following.id)["story"]["turns"] == [
        {
            "turn_number": 1,
            "player": {"content": "第一次输入", "images": []},
            "has_ai_output": False,
        },
        {
            "turn_number": 2,
            "player": {"content": "直接补充输入", "images": []},
            "has_ai_output": False,
        },
    ]


def test_retry_touch_failure_keeps_turn_terminal_and_story_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    saves = app.state.saves
    save = saves.create_save(SaveCreate(world_name="雾港", name="重试元数据失败"))
    failed = saves.create_turn(save.id, "等待重试", 2)
    saves.fail_turn(save.id, failed.id, "turn.failed", {"message": "测试失败"})
    story_before = saves.load_story(save.id, failed.id)

    def reject_touch(_save_id: str) -> None:
        raise OSError("touch failed")

    monkeypatch.setattr(saves, "_touch_unlocked", reject_touch)
    with pytest.raises(OSError, match="touch failed"):
        saves.retry_turn(save.id, failed.id, 2)

    assert saves.get_turn(save.id, failed.id).status == "failed"
    assert saves.load_story(save.id, failed.id) == story_before


def test_checker_can_report_narrator_revision_and_recheck_before_publication(tmp_path: Path) -> None:
    class EditClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="draft", name="task", arguments={"agent": "narrator", "task": "写初稿"}))
                if len(results) == 1:
                    return self.result(call=ToolCall(id="check", name="task", arguments={"agent": "consistency_checker", "task": "检查当前正文"}))
                if len(results) == 2:
                    return self.result(call=ToolCall(id="revise", name="task", arguments={"agent": "narrator", "task": "根据随附检查报告修改初稿", "task_id": results[0]["task_id"], "report_refs": [results[1]["report_ref"]]}))
                if len(results) == 3:
                    return self.result(call=ToolCall(id="recheck", name="task", arguments={"agent": "consistency_checker", "task": "复查修改后的正文", "task_id": results[1]["task_id"], "report_refs": [results[2]["report_ref"]]}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == NARRATOR_PROMPT:
                latest_user = json.loads(next(message["content"] for message in reversed(messages) if message["role"] == "user"))
                if latest_user.get("continuation"):
                    assert latest_user["report_documents"][0]["source_agent"] == "consistency_checker"
                    assert latest_user["report_documents"][0]["content"] == "正文 rough 需要改为完整故事。"
                    if messages[-1]["role"] == "user":
                        return self.result(call=ToolCall(id="edit", name="file_edit", arguments={"path": "narrative.md", "old_text": "rough", "new_text": STORY}))
                    return self.result("修改已完成。")
                if not results:
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "rough"}))
                return self.result("初稿已完成。")
            if system_prompt == CONSISTENCY_CHECKER_PROMPT:
                assert {item["function"]["name"] for item in tools or []} == {"file_read", "story_summary_read"}
                latest_user = json.loads(next(message["content"] for message in reversed(messages) if message["role"] == "user"))
                if messages[-1]["role"] == "user":
                    if not latest_user.get("continuation"):
                        story = json.loads(messages[0]["content"])["story"]
                        assert story["turns"][-1]["player"]["content"] == "继续。"
                        assert story["turns"][-1]["has_ai_output"] is False
                    else:
                        assert latest_user["report_documents"][0]["source_agent"] == "narrator"
                        assert latest_user["report_documents"][0]["content"] == "修改已完成。"
                    call_id = "read-revision" if latest_user.get("continuation") else "read-draft"
                    return self.result(call=ToolCall(id=call_id, name="file_read", arguments={"path": "narrative.md"}))
                return self.result("未发现范围内问题。" if len(results) == 2 else "正文 rough 需要改为完整故事。")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=EditClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "编辑草稿"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["narrative"] == STORY

    with app.state.saves.connect(save["id"]) as db:
        child_tasks = db.execute("SELECT child_session_id FROM parts WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?) AND tool_name = 'task' ORDER BY rowid", (created["turn_id"],)).fetchall()
        root_reads = db.execute("SELECT COUNT(*) FROM parts WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?) AND tool_name = 'file_read' AND state = 'completed'", (created["turn_id"],)).fetchone()[0]
    assert child_tasks[0]["child_session_id"] == child_tasks[2]["child_session_id"]
    assert child_tasks[1]["child_session_id"] == child_tasks[3]["child_session_id"]
    assert child_tasks[0]["child_session_id"] != child_tasks[1]["child_session_id"]
    assert root_reads == 0


def test_publish_reads_current_draft_without_prior_coordinator_read(tmp_path: Path) -> None:
    class DirectPublishClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                assert {item["function"]["name"] for item in tools or []} == {"task", "narrative_publish", "story_summary_read"}
                results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
                if not results:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写故事"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=DirectPublishClient())
    with TestClient(app) as client:
        settings = client.get("/api/runtime-settings").json()
        settings["block_coordinator_narrative_read"] = True
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "直接发布"}).json()
        created = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).json()
        assert wait_for_turn(client, save["id"])["narrative"] == STORY

    with app.state.saves.connect(save["id"]) as db:
        root_tools = db.execute("SELECT tool_name FROM parts WHERE session_id = (SELECT root_session_id FROM turns WHERE id = ?) AND type = 'tool' ORDER BY rowid", (created["turn_id"],)).fetchall()
    assert [row["tool_name"] for row in root_tools] == ["task", "narrative_publish"]


def test_blocked_coordinator_narrative_read_rejects_unadvertised_call(tmp_path: Path) -> None:
    class ForbiddenReadClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if system_prompt == COORDINATOR_PROMPT:
                assert "file_read" not in {item["function"]["name"] for item in tools or []}
                results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
                if not results:
                    return self.result(call=ToolCall(id="forbidden", name="file_read", arguments={"path": "narrative.md"}))
                assert results[-1]["error"]["message"] == "主代理已禁止读取 narrative.md"
                raise RuntimeError("stop after observing rejected read")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ForbiddenReadClient())
    with TestClient(app) as client:
        settings = client.get("/api/runtime-settings").json()
        settings["block_coordinator_narrative_read"] = True
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "拒绝主代理读取"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "failed"

    with app.state.saves.connect(save["id"]) as db:
        output = json.loads(db.execute("SELECT output FROM parts WHERE tool_name = 'file_read'").fetchone()["output"])
    assert output["error"]["code"] == "validation_error"


@pytest.mark.parametrize(("draft", "message"), [(None, "不存在"), ("   ", "不能为空")])
def test_publish_rejects_missing_or_blank_current_draft(tmp_path: Path, draft: str | None, message: str) -> None:
    class InvalidDraftClient(NarrativeFileClient):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(item["content"]) for item in messages if item["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if results and results[-1].get("error"):
                    raise RuntimeError("stop after observing publish error")
                if draft is not None and not results:
                    return self.result(call=ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "写草稿"}))
                return self.result(call=ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == NARRATOR_PROMPT:
                if not results:
                    return self.result(call=ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": draft or ""}))
                return self.result("草稿已完成。")
            return await super().complete(system_prompt, messages, tools)

    app = create_app(data_dir=make_data_dir(tmp_path), model_client=InvalidDraftClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "无效草稿"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "failed"

    with app.state.saves.connect(save["id"]) as db:
        result = json.loads(db.execute("SELECT output FROM parts WHERE tool_name = 'narrative_publish'").fetchone()["output"])
    assert message in result["error"]["message"]


def test_task_id_cannot_cross_turns(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="task ownership"))

    first = app.state.saves.create_turn(save.id, "第一回合", 4)
    first_root = app.state.saves.root_session_id(save.id, first.id)
    first_message = app.state.saves.create_assistant_message(save.id, first_root, {"model": "test"})
    first_part = app.state.saves.record_model_result(save.id, first_message, "", None, [{"id": "one", "name": "task", "arguments": {}}])[0]
    app.state.saves.set_tool_running(save.id, first_part["id"])
    child = app.state.saves.prepare_child_task(save.id, first.id, first_root, first_part["id"], "narrator", {"task": "写", "reason": ""}, None)
    app.state.saves.error_tool(save.id, first_part["id"], {"error": {"code": "execution_interrupted"}, "task_id": child})
    app.state.saves.fail_turn(save.id, first.id, "turn.interrupted", {"message": "test"})

    second = app.state.saves.create_turn(save.id, "第二回合", 4)
    second_root = app.state.saves.root_session_id(save.id, second.id)
    second_message = app.state.saves.create_assistant_message(save.id, second_root, {"model": "test"})
    second_part = app.state.saves.record_model_result(save.id, second_message, "", None, [{"id": "two", "name": "task", "arguments": {}}])[0]
    app.state.saves.set_tool_running(save.id, second_part["id"])

    with pytest.raises(ValueError, match="不属于当前回合"):
        app.state.saves.prepare_child_task(save.id, second.id, second_root, second_part["id"], "narrator", {"task": "越界", "reason": ""}, child)
    with app.state.saves.connect(save.id) as db:
        assert db.execute("SELECT child_session_id FROM parts WHERE id = ?", (second_part["id"],)).fetchone()["child_session_id"] is None


def test_provider_messages_rebuild_persisted_reasoning_only_response(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="reasoning replay"))
    turn = app.state.saves.create_turn(save.id, "继续。", 4)
    root = app.state.saves.root_session_id(save.id, turn.id)
    message = app.state.saves.create_assistant_message(save.id, root, {"model": "deepseek-flash"})
    app.state.saves.record_model_result(
        save.id,
        message,
        "",
        "先读取草稿。",
        [],
    )

    rebuilt = AgentRunner._provider_messages(
        app.state.saves.load_session(save.id, root),
        app.state.saves.load_story(save.id, turn.id),
        coordinator=True,
    )

    reasoning = next(message for message in rebuilt if message.get("_reasoning_content"))
    assert reasoning["_reasoning_content"] == "先读取草稿。"
    assert reasoning["content"] is None


def test_failure_after_tool_started_state_exposes_interrupted_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=NarrativeFileClient())
    original = app.state.saves.add_event

    def fail_tool_started(save_id: str, turn_id: str | None, event_type: str, payload: dict[str, Any]):
        if event_type == "tool.started":
            raise RuntimeError("event storage failed")
        return original(save_id, turn_id, event_type, payload)

    monkeypatch.setattr(app.state.saves, "add_event", fail_tool_started)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "事件故障"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "failed"
    with app.state.saves.connect(save["id"]) as db:
        part = db.execute("SELECT state, output FROM parts WHERE type = 'tool'").fetchone()
    assert part["state"] == "error"
    assert json.loads(part["output"])["error"]["code"] == "execution_interrupted"
