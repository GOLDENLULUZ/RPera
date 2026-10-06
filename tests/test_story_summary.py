from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.models import ModelResult, SaveCreate, ToolCall
from rpera.runtime import AgentRunner
from rpera.story_summary import StorySummaryStore
from tests.helpers import make_data_dir, wait_for_revision, wait_for_turn
from tests.prompt_fixtures import COORDINATOR_PROMPT, NARRATOR_PROMPT, PROMPT_TEMPLATES


class SummaryClient:
    def __init__(self, *, skip_first: bool = False) -> None:
        self.skip_first = skip_first
        self.summary_calls: list[list[dict[str, Any]]] = []
        self.summary_tools: list[set[str]] = []
        self.narrator_contexts: list[list[dict[str, Any]]] = []
        self.history_results: list[dict[str, Any]] = []
        self.coordinator_tools: list[set[str]] = []

    def bind(self) -> SummaryClient:
        return self

    def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, SummaryClient]:
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int]:
        return {"provider": "test", "model": "summary", "base_url": "local", "timeout_seconds": 1}

    @staticmethod
    def call(name: str, **kwargs: Any) -> ModelResult:
        return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="reused", name=name, arguments=kwargs)])

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        names = {tool["function"]["name"] for tool in tools or []}
        outputs = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            self.coordinator_tools.append(names)
            story_end = next(index for index, message in enumerate(messages) if message.get("content") == "</recent_story>")
            player_input = next(message["content"] for message in reversed(messages[:story_end]) if message["role"] == "user")
            turn_number = int(player_input.split(" ")[0])
            summarize = turn_number != 1 or not self.skip_first
            if not outputs and summarize:
                return self.call("task", agent="story_summarizer", task="补齐之前的故事摘要")
            if len(outputs) == int(summarize):
                return self.call("task", agent="narrator", task="继续故事")
            return self.call("narrative_publish", path="narrative.md")
        if system_prompt == NARRATOR_PROMPT:
            self.narrator_contexts.append(messages)
            if not outputs:
                if self.skip_first and not self.summary_calls:
                    return self.call("file_write", path="narrative.md", content="第一段故事")
                return self.call("story_summary_read")
            if outputs[-1].get("path") == "story_summary.json":
                return self.call("file_write", path="narrative.md", content="故事继续")
            return ModelResult(content="草稿完成", raw_response={})
        if system_prompt == PROMPT_TEMPLATES["story_summarizer"]:
            self.summary_calls.append(messages)
            self.summary_tools.append(names)
            if not outputs:
                return self.call("story_summary_read")
            if len(outputs) == 1:
                return self.call("story_history_read", after_turn_number=0)
            if len(outputs) == 2:
                history = outputs[-1]
                self.history_results.append(history)
                existing = outputs[0]["content"]
                return self.call("story_summary_edit", old_text=existing, new_text='{"turns": [')
            if len(outputs) == 3:
                assert outputs[-1]["ok"] is False
                existing = outputs[0]["content"]
                entries = [
                    {"turn_number": turn["turn_number"], "summary": turn.get("narrative", turn["player"]["content"])}
                    for turn in self.history_results[-1]["turns"]
                ]
                return self.call("story_summary_edit", old_text=existing, new_text=json.dumps({"turns": entries}, ensure_ascii=False))
            return ModelResult(content="历史摘要已更新", raw_response={})
        raise AssertionError(system_prompt)


def configure(client: TestClient, *, required: bool = False, context_turns: int = 4) -> None:
    settings = client.get("/api/runtime-settings").json()
    settings["context_turns"] = context_turns
    settings["always_attach_report_agents"] = ["story_summarizer"] if required else []
    assert client.put("/api/runtime-settings", json=settings).status_code == 200


def play(client: TestClient, save_id: str, number: int) -> dict[str, Any]:
    response = client.post(f"/api/saves/{save_id}/turns", json={"content": f"{number} 玩家行动"})
    assert response.status_code == 202
    turn = wait_for_turn(client, save_id)
    assert turn["status"] == "completed", turn
    return turn


def test_first_turn_returns_local_report_second_turn_validates_patch_and_others_read(tmp_path: Path) -> None:
    model = SummaryClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        configure(client, required=True)
        save_id = client.post("/api/saves", json={"world_name": "雾港", "name": "摘要"}).json()["id"]
        first = play(client, save_id, 1)
        summary_file = tmp_path / "data" / "saves" / save_id / "current" / "story_summary.json"
        assert json.loads(summary_file.read_text(encoding="utf-8")) == {"turns": []}
        assert model.summary_calls == []
        with app.state.saves.connect(save_id) as db:
            task = db.execute("SELECT output FROM parts WHERE tool_name = 'task' AND session_id = ? ORDER BY rowid LIMIT 1", (app.state.saves.root_session_id(save_id, first["id"]),)).fetchone()
        assert "首个回合" in json.loads(task["output"])["result"]
        assert json.loads(task["output"])["report_ref"]["name"] == "story_summarizer 报告"
        second = play(client, save_id, 2)
        assert second["narrative"] == "故事继续"
        assert json.loads(summary_file.read_text(encoding="utf-8")) == {"turns": [{"turn_number": 1, "summary": first["narrative"]}]}
        assert len(model.summary_calls) == 5
        assert {"story_summary_read", "story_summary_edit", "story_history_read"} <= model.summary_tools[0]
        assert model.history_results[0]["next_after_turn_number"] == 1
        assert "story_summary_read" in model.coordinator_tools[-1]
        assert any('"source_agent": "story_summarizer"' in str(message.get("content")) for message in model.narrator_contexts[-3])
        assert any('"source_agent": "story_summarizer"' in str(message.get("content")) for message in model.narrator_contexts[-1])
        with app.state.saves.connect(save_id) as db:
            failed = db.execute("SELECT output FROM parts WHERE tool_name = 'story_summary_edit' AND state = 'error'").fetchone()
        assert failed is not None and json.loads(failed["output"])["ok"] is False


def test_second_turn_first_call_creates_file_and_backfills_beyond_recent_window(tmp_path: Path) -> None:
    model = SummaryClient(skip_first=True)
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        configure(client, context_turns=2)
        save_id = client.post("/api/saves", json={"world_name": "雾港", "name": "漏记补齐"}).json()["id"]
        summary_file = tmp_path / "data" / "saves" / save_id / "current" / "story_summary.json"
        play(client, save_id, 1)
        assert not summary_file.exists()
        play(client, save_id, 2)
        assert summary_file.exists()
        assert [item["turn_number"] for item in json.loads(summary_file.read_text(encoding="utf-8"))["turns"]] == [1]
        # A missing summary remains recoverable after the corresponding story leaves the recent-story window.
        # Publication precedes checkpoint copying; wait before an external file
        # edit, which otherwise races the copy's read handle on Windows.
        second = client.get(f"/api/saves/{save_id}/turns").json()[-1]
        wait_for_revision(app.state.saves.saves_dir / save_id / "revisions" / second["id"])
        summary_file.write_text('{"turns": []}\n', encoding="utf-8")
        play(client, save_id, 3)
        play(client, save_id, 4)
        assert [item["turn_number"] for item in json.loads(summary_file.read_text(encoding="utf-8"))["turns"]] == [1, 2, 3]
        assert [turn["turn_number"] for turn in json.loads(model.summary_calls[-1][0]["content"])["story"]["turns"]] == [2, 3, 4]
        assert [turn["turn_number"] for turn in model.history_results[-1]["turns"]] == [1, 2, 3]


def test_invalid_or_current_turn_patch_preserves_last_valid_json(tmp_path: Path) -> None:
    store = StorySummaryStore(tmp_path)
    store.ensure()
    original = store.read()["content"]
    for replacement in ('{"turns": [', '{"turns": [{"turn_number": 2, "summary": "还没发生"}]}',
                        '{"turns": [{"turn_number": 1, "summary": "合法"}, {"turn_number": 1, "summary": "重复"}]}'):
        with pytest.raises(ValueError):
            store.edit(original, replacement, current_turn_number=2)
        assert store.read()["content"] == original
    edited = store.edit(original, '{"turns": [{"turn_number": 1, "summary": "已完成"}]}', 2)
    assert edited["summary"]["turns"] == [{"turn_number": 1, "summary": "已完成"}]


def test_history_includes_failed_player_turn_but_not_current_input(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=SummaryClient())
    saves = app.state.saves
    save_id = saves.create_save(SaveCreate(world_name="雾港", name="失败回合")).id
    first = saves.create_turn(save_id, "未发布的玩家行动", 2)
    saves.fail_turn(save_id, first.id, "turn.failed", {"message": "测试"})
    second = saves.create_turn(save_id, "本回合未完成的输入", 2)
    history = saves.read_story_history(save_id, second.id, 0)
    assert history["turns"] == [{"turn_number": 1, "player": {"content": "未发布的玩家行动", "images": []}, "has_ai_output": False}]
    assert history["next_after_turn_number"] == 1


def test_historical_image_is_rebuilt_from_artifact_for_model() -> None:
    record = {"story": {"opening": "", "turns": [{"turn_number": 2, "player": {"content": "现在", "images": []}, "has_ai_output": False}]}}
    history = {"turns": [{"turn_number": 1, "player": {"content": "", "images": [{"id": "picture"}]}, "has_ai_output": False}], "next_after_turn_number": 1}
    session = [{"role": "assistant", "model": "{}", "parts": [{"type": "tool", "tool_name": "story_history_read", "provider_call_id": "same", "state": "completed", "output": json.dumps(history), "input": "{}"}]}]
    data = base64.b64encode(b"picture data").decode("ascii")
    rebuilt = AgentRunner._provider_messages(
        session, record, "save", image_reader=lambda _save_id, _image_id: {
            "path": "artifacts/picture.png", "mime_type": "image/png", "data": data,
        },
    )
    content = rebuilt[-1]["content"]
    assert isinstance(content, list)
    assert content[0]["text"] == json.dumps(history)
    assert json.loads(content[1]["text"]) == {
        "source": "player", "image_id": "picture", "path": "artifacts/picture.png",
    }
    assert any(block.get("type") == "image_url" or block.get("type") == "image" for block in content[1:])


def test_disabling_summarizer_hides_read_tool_and_rejects_unadvertised_call(tmp_path: Path) -> None:
    class DisabledClient(SummaryClient):
        disabled = False
        observed: list[set[str]]

        def __init__(self) -> None:
            super().__init__()
            self.observed = []

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            if not self.disabled:
                return await super().complete(system_prompt, messages, tools)
            names = {item["function"]["name"] for item in tools or []}
            self.observed.append(names)
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if "narrative_publish" in names:
                assert "story_summary_read" not in names
                task = next(item["function"] for item in tools or [] if item["function"]["name"] == "task")
                assert "story_summarizer" not in task["parameters"]["properties"]["agent"]["enum"]
                if not results:
                    return self.call("story_summary_read")
                if len(results) == 1:
                    assert results[-1]["ok"] is False
                    return self.call("task", agent="narrator", task="继续故事")
                return self.call("narrative_publish", path="narrative.md")
            assert "story_summary_read" not in names
            if not results:
                return self.call("file_write", path="narrative.md", content="禁用后继续")
            return ModelResult(content="完成", raw_response={})

    model = DisabledClient()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save_id = client.post("/api/saves", json={"world_name": "雾港", "name": "摘要开关"}).json()["id"]
        play(client, save_id, 1)
        summary_file = tmp_path / "data" / "saves" / save_id / "current" / "story_summary.json"
        assert summary_file.exists()
        settings = client.get("/api/runtime-settings").json()
        settings["disabled_agents"] = ["story_summarizer"]
        assert client.put("/api/runtime-settings", json=settings).status_code == 200
        model.disabled = True
        play(client, save_id, 2)
        assert json.loads(summary_file.read_text(encoding="utf-8")) == {"turns": []}
        assert model.observed
        with app.state.saves.connect(save_id) as db:
            rejected = db.execute("SELECT output FROM parts WHERE tool_name = 'story_summary_read' AND state = 'error'").fetchone()
        assert rejected is not None and "无权" in json.loads(rejected["output"])["error"]["message"]
