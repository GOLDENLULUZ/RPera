from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.models import ModelResult, ToolCall
from tests.helpers import make_data_dir, wait_for_turn
from tests.prompt_fixtures import COORDINATOR_PROMPT, CONSISTENCY_CHECKER_PROMPT, NARRATOR_PROMPT


class SettingsStoryClient:
    def __init__(self) -> None:
        self.fail_next_coordinator = False

    def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, SettingsStoryClient]:
        return {name: self for name in names}

    def metadata(self) -> dict[str, Any]:
        return {"provider": "test", "model": "settings-story"}

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        if self.fail_next_coordinator:
            self.fail_next_coordinator = False
            raise RuntimeError("simulated model interruption")
        completed = [message for message in messages if message["role"] == "tool"]
        call: ToolCall | None = None
        if system_prompt == COORDINATOR_PROMPT:
            if not completed:
                call = ToolCall(id="draft", name="task", arguments={"agent": "narrator", "task": "写故事"})
            elif len(completed) == 1:
                call = ToolCall(id="check", name="task", arguments={"agent": "consistency_checker", "task": "检查故事"})
            else:
                call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
        elif system_prompt.startswith("你是 RPera 的叙事 Agent。"):
            if not completed:
                call = ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "故事已写。"})
        elif system_prompt.startswith("你是 RPera 的一致性检查 Agent。"):
            if not completed:
                call = ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"})
        else:
            raise AssertionError(f"Unexpected prompt: {system_prompt[:80]}")
        return ModelResult(content="" if call else "完成。", raw_response={}, tool_calls=[call] if call else [])


def requests_for_save(app: Any, save_id: str, agent: str) -> list[str]:
    return [
        event.payload["system_prompt"]
        for event in app.state.saves.list_events(save_id)
        if event.type == "model.request" and event.payload["agent"] == agent
    ]


def test_custom_save_settings_reach_both_agents_after_retry_and_do_not_leak(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    model = SettingsStoryClient()
    app = create_app(data_dir=data_dir, model_client=model)
    custom_settings = {
        "play_mode": "director",
        "narrative_person": "first",
        "language": "ja",
        "other_requirements": "由伊蕾自述，不要加入玩家角色。",
    }
    with TestClient(app) as client:
        invalid = client.post("/api/saves", json={"world_name": "雾港", "name": "非法选项", "language": "fr"})
        assert invalid.status_code == 422
        assert client.post("/api/saves", json={"world_name": "雾港", "name": "过长", "other_requirements": "x" * 4_001}).status_code == 422
        custom = client.post("/api/saves", json={"world_name": "雾港", "name": "导演存档", **custom_settings}).json()
        assert {key: custom[key] for key in custom_settings} == custom_settings
        saved = json.loads((data_dir / "saves" / custom["id"] / "save.json").read_text(encoding="utf-8"))
        assert {key: saved[key] for key in custom_settings} == custom_settings

        model.fail_next_coordinator = True
        failed = client.post(f"/api/saves/{custom['id']}/turns", json={"content": "从伊蕾的视角开始。"}).json()
        assert wait_for_turn(client, custom["id"])["status"] == "failed"
        assert client.post(f"/api/saves/{custom['id']}/turns/{failed['turn_id']}/retry").status_code == 202
        assert wait_for_turn(client, custom["id"])["status"] == "completed"

        other = client.post("/api/saves", json={"world_name": "雾港", "name": "默认存档"}).json()
        assert (other["play_mode"], other["narrative_person"], other["language"], other["other_requirements"]) == ("roleplay", "second", "zh", "")
        assert client.post(f"/api/saves/{other['id']}/turns", json={"content": "继续。"}).status_code == 202
        assert wait_for_turn(client, other["id"])["status"] == "completed"

        third = client.post("/api/saves", json={
            "world_name": "雾港", "name": "英文第三人称", "narrative_person": "third", "language": "en",
        }).json()
        assert client.post(f"/api/saves/{third['id']}/turns", json={"content": "继续。"}).status_code == 202
        assert wait_for_turn(client, third["id"])["status"] == "completed"

    for agent in ("narrator", "consistency_checker"):
        prompts = requests_for_save(app, custom["id"], agent)
        assert prompts and all("导演模式" in prompt and "第一人称" in prompt and "日语" in prompt for prompt in prompts)
        assert all(custom_settings["other_requirements"] in prompt and "{save_narrative_settings}" not in prompt for prompt in prompts)
        assert requests_for_save(app, other["id"], agent) == [
            NARRATOR_PROMPT if agent == "narrator" else CONSISTENCY_CHECKER_PROMPT,
            NARRATOR_PROMPT if agent == "narrator" else CONSISTENCY_CHECKER_PROMPT,
        ]
        assert all("第三人称" in prompt and "英语" in prompt and "角色扮演模式" in prompt
                   for prompt in requests_for_save(app, third["id"], agent))


def test_legacy_save_uses_original_defaults_without_rewriting_file(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=SettingsStoryClient())
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "旧存档"}).json()
        path = data_dir / "saves" / save["id"] / "save.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        for key in ("play_mode", "narrative_person", "language", "other_requirements"):
            del saved[key]
        path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
        assert client.get(f"/api/saves/{save['id']}").json()["narrative_person"] == "second"
        assert client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"}).status_code == 202
        assert wait_for_turn(client, save["id"])["status"] == "completed"
        assert "play_mode" not in json.loads(path.read_text(encoding="utf-8"))

    assert requests_for_save(app, save["id"], "narrator") == [NARRATOR_PROMPT, NARRATOR_PROMPT]
    assert requests_for_save(app, save["id"], "consistency_checker") == [CONSISTENCY_CHECKER_PROMPT, CONSISTENCY_CHECKER_PROMPT]
