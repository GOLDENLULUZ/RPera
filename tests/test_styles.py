from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpera.app import PROJECT_ROOT, create_app
from rpera.content import ParsedStyle, parse_style_document, serialize_style_document, style_path
from rpera.content_store import ContentNotFoundError, ContentStore, UnsafeContentError
from rpera.models import ModelResult, ToolCall
from tests.prompt_fixtures import COORDINATOR_PROMPT, NARRATOR_PROMPT, STYLE_PLANNER_PROMPT, WORLD_RESEARCHER_PROMPT


STYLE_ONE = "styles/冷峻悬疑.md"
STYLE_TWO = "styles/细腻感官.md"
STYLE_LATE = "styles/后来新增.md"
STYLE_RENAMED = "styles/改名后.md"
STORY = "伊蕾把杯口转向墙壁。瓷器擦过木桌，发出一声轻响。"


def style_document(description: str, content: str, *, enabled: bool = True) -> str:
    return serialize_style_document(ParsedStyle(description, enabled, content))


def make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    worlds_dir = data_dir / "worlds"
    worlds_dir.mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "data" / "worlds" / "雾港", worlds_dir / "雾港")
    styles_dir = data_dir / "styles"
    styles_dir.mkdir()
    (styles_dir / "冷峻悬疑.md").write_text(style_document("旧简介", "旧版正文"), encoding="utf-8")
    (styles_dir / "细腻感官.md").write_text(style_document("强调感官线索", "保留触觉、气味与声音。"), encoding="utf-8")
    return data_dir


def wait_for_turn(client: TestClient, save_id: str, timeout: float = 8) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turn = client.get(f"/api/saves/{save_id}/turns").json()[-1]
        if turn["status"] != "running":
            return turn
        time.sleep(0.01)
    raise AssertionError("turn did not finish")


class StyleFlowClient:
    def __init__(self, styles_dir: Path) -> None:
        self.styles_dir = styles_dir
        self.planner_input: dict[str, Any] | None = None
        self.narrator_input: dict[str, Any] | None = None

    def bind(self) -> StyleFlowClient:
        return self

    def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, StyleFlowClient]:
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": "test", "model": "styles", "base_url": "local", "timeout_seconds": 1}

    @staticmethod
    def result(call: ToolCall | None = None, content: str = "") -> ModelResult:
        return ModelResult(content=content, raw_response={}, tool_calls=[] if call is None else [call])

    async def complete(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResult:
        results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            if not results:
                return self.result(ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "调查伊蕾"}))
            if len(results) == 1:
                entities = results[0]["result"]["related_entities"]
                return self.result(ToolCall(id="unreported-style", name="task", arguments={"agent": "narrator", "task": "提前写作", "related_entities": entities, "style_paths": [STYLE_ONE]}))
            if len(results) == 2:
                assert results[-1]["error"]["code"] == "validation_error"
                entities = results[0]["result"]["related_entities"]
                return self.result(ToolCall(id="plan-style", name="task", arguments={"agent": "style_planner", "task": "为克制的酒馆盘问选择并融合文风", "related_entities": entities}))
            if len(results) == 3:
                report = results[-1]["result"]
                entities = results[0]["result"]["related_entities"]
                return self.result(ToolCall(id="reordered-style", name="task", arguments={"agent": "narrator", "task": "错误重排", "reason": report["report"], "related_entities": entities, "style_paths": list(reversed(report["style_paths"]))}))
            if len(results) == 4:
                assert results[-1]["error"]["code"] == "validation_error"
                report = results[-2]["result"]
                entities = results[0]["result"]["related_entities"]
                return self.result(ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写出克制的酒馆盘问", "reason": report["report"], "related_entities": entities, "style_paths": report["style_paths"]}))
            if len(results) == 5:
                return self.result(ToolCall(id="read", name="file_read", arguments={"path": "narrative.md"}))
            return self.result(ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
        if system_prompt == WORLD_RESEARCHER_PROMPT:
            if not results:
                return self.result(ToolCall(id="search", name="entity_search", arguments={"query": "伊蕾"}))
            if len(results) == 1:
                path = results[0]["results"][0]["path"]
                return self.result(ToolCall(id="read-entity", name="entity_read", arguments={"paths": [path]}))
            path = results[-1]["documents"][0]["path"]
            return self.result(ToolCall(id="research-report", name="research_report", arguments={"report": "伊蕾寡言而警惕。", "related_entities": [path]}))
        if system_prompt == STYLE_PLANNER_PROMPT:
            self.planner_input = json.loads(messages[1]["content"])
            assert {tool["function"]["name"] for tool in tools or []} == {"style_read", "style_report", "story_summary_read"}
            if not results:
                return self.result(ToolCall(id="invalid-style-report", name="style_report", arguments={"report": "尚未读取。", "style_paths": [STYLE_ONE]}))
            if len(results) == 1:
                assert results[0]["error"]["code"] == "tool_error"
                (self.styles_dir / "后来新增.md").write_text(style_document("后来新增", "不应进入旧会话。"), encoding="utf-8")
                return self.result(ToolCall(id="read-unlisted", name="style_read", arguments={"paths": [STYLE_LATE]}))
            if len(results) == 2:
                assert results[-1]["error"]["code"] == "tool_error"
                path = self.styles_dir / "冷峻悬疑.md"
                parsed = parse_style_document(path.read_text(encoding="utf-8"))
                path.write_text(style_document(parsed.description, parsed.content, enabled=False), encoding="utf-8")
                return self.result(ToolCall(id="read-styles", name="style_read", arguments={"paths": [STYLE_ONE, STYLE_TWO]}))
            if len(results) == 3:
                assert "已禁用" in results[-1]["error"]["message"]
                path = self.styles_dir / "冷峻悬疑.md"
                parsed = parse_style_document(path.read_text(encoding="utf-8"))
                path.write_text(style_document(parsed.description, parsed.content), encoding="utf-8")
                return self.result(ToolCall(id="read-enabled-styles", name="style_read", arguments={"paths": [STYLE_ONE, STYLE_TWO]}))
            return self.result(ToolCall(id="style-report", name="style_report", arguments={"report": "以感官细节为底，冷峻留白为主。", "style_paths": [STYLE_TWO, STYLE_ONE]}))
        if system_prompt == NARRATOR_PROMPT:
            self.narrator_input = json.loads(messages[1]["content"])
            if not results:
                return self.result(ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
            return self.result(content="叙事完成。")
        raise AssertionError(system_prompt)


def test_content_store_uses_safe_deterministic_style_files(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    store = ContentStore(data_dir)

    assert [style.name for style in store.list_styles()] == ["冷峻悬疑", "细腻感官"]
    assert [document.path for document in store.read_styles([STYLE_TWO, STYLE_ONE])] == [STYLE_TWO, STYLE_ONE]
    assert parse_style_document(style_document("简介", "正文")) == ParsedStyle("简介", True, "正文")
    assert style_path(STYLE_ONE) == STYLE_ONE

    (data_dir / "styles" / "冷峻悬疑.md").write_text(style_document("旧简介", "旧版正文", enabled=False), encoding="utf-8")
    assert [style.name for style in store.list_styles()] == ["冷峻悬疑", "细腻感官"]
    assert [style.name for style in store.list_enabled_styles()] == ["细腻感官"]
    with pytest.raises(ContentNotFoundError, match="已禁用"):
        store.read_enabled_styles([STYLE_ONE])

    with pytest.raises(ValueError, match="enabled 为必填字段"):
        parse_style_document("---\ndescription: 缺少状态\n---\n\n正文")
    with pytest.raises(ValueError, match="enabled 必须是布尔值"):
        parse_style_document("---\ndescription: 状态错误\nenabled: 'true'\n---\n\n正文")

    temporary = data_dir / "styles" / ".冷峻悬疑.md.12345678-1234-4234-8234-123456789abc.tmp"
    temporary.write_text("未发布", encoding="utf-8")
    assert [style.name for style in store.list_styles()] == ["冷峻悬疑", "细腻感官"]

    (data_dir / "styles" / "不是文风.txt").write_text("x", encoding="utf-8")
    with pytest.raises(UnsafeContentError, match="只允许"):
        store.list_styles()


def test_style_crud_uses_public_content_api_and_real_markdown(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.get("/api/content/styles").json() == []
        created = client.post("/api/content/styles", json={"name": "清透日常"})
        assert created.status_code == 201
        assert created.json() == {
            "path": "styles/清透日常.md",
            "name": "清透日常",
            "description": "",
            "enabled": True,
            "content": "",
        }
        path = data_dir / "styles" / "清透日常.md"
        assert parse_style_document(path.read_text(encoding="utf-8")) == ParsedStyle("", True, "")

        assert client.put("/api/content/styles/清透日常/enabled", json={"enabled": "false"}).status_code == 422
        disabled = client.put("/api/content/styles/清透日常/enabled", json={"enabled": False})
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        assert client.get("/api/content/styles").json()[0]["enabled"] is False

        saved = client.put(
            "/api/content/styles/清透日常",
            json={"description": "用于平静生活片段", "content": "使用自然短句，保留生活细节。"},
        )
        assert saved.status_code == 200
        assert saved.json()["content"] == "使用自然短句，保留生活细节。"
        assert saved.json()["enabled"] is False
        assert "content" not in client.get("/api/content/styles").json()[0]

        renamed = client.post("/api/content/styles/清透日常/rename", json={"name": "日常留白"})
        assert renamed.status_code == 200
        assert renamed.json()["description"] == "用于平静生活片段"
        assert renamed.json()["enabled"] is False
        assert not path.exists()

    with TestClient(create_app(data_dir=data_dir)) as client:
        persisted = client.get("/api/content/styles/日常留白").json()
        assert persisted["content"] == "使用自然短句，保留生活细节。"
        assert persisted["enabled"] is False
        assert client.put("/api/content/styles/日常留白/enabled", json={"enabled": True}).json()["enabled"] is True
        assert client.post("/api/content/styles", json={"name": "日常留白"}).status_code == 409
        assert client.post("/api/content/styles", json={"name": " bad"}).status_code == 422
        assert client.put(
            "/api/content/styles/日常留白", json={"description": "x" * 4_001, "content": ""}
        ).status_code == 422
        assert client.get("/api/content/styles/不存在").status_code == 404
        assert client.delete("/api/content/styles/日常留白").json() == {
            "name": "日常留白",
            "path": "styles/日常留白.md",
            "deleted": True,
        }
        assert client.get("/api/content/styles").json() == []


def test_style_planner_selects_global_styles_for_narrator(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    (data_dir / "styles" / "禁用文风.md").write_text(style_document("不应被 AI 看到", "禁用正文", enabled=False), encoding="utf-8")
    model = StyleFlowClient(data_dir / "styles")
    app = create_app(data_dir=data_dir, model_client=model)

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "文风测试"}).json()
        snapshot = data_dir / "saves" / save["id"] / "current" / "world_snapshot"
        assert not (snapshot / "styles").exists()
        (data_dir / "styles" / "冷峻悬疑.md").write_text(
            style_document("更新后的克制悬疑", "使用短句、动作与留白。"), encoding="utf-8"
        )
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "我询问守灯人的去向。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert turn["narrative"] == STORY
    assert model.planner_input is not None
    assert model.narrator_input is not None
    assert [style["path"] for style in model.planner_input["available_styles"]] == [STYLE_ONE, STYLE_TWO]
    assert model.planner_input["available_styles"][0]["description"] == "更新后的克制悬疑"
    assert model.planner_input["related_entity_documents"][0]["name"] == "伊蕾"
    assert [style["path"] for style in model.narrator_input["style_documents"]] == [STYLE_TWO, STYLE_ONE]
    assert model.narrator_input["style_documents"][1]["content"] == "使用短句、动作与留白。"

    events = app.state.saves.list_events(save["id"])
    assert [event.type for event in events].count("style.read") == 1
    with app.state.saves.connect(save["id"]) as db:
        invalid = db.execute("SELECT state FROM parts WHERE provider_call_id = 'invalid-style-report'").fetchone()
        unreported = db.execute("SELECT state FROM parts WHERE provider_call_id = 'unreported-style'").fetchone()
        reordered = db.execute("SELECT state FROM parts WHERE provider_call_id = 'reordered-style'").fetchone()
        unlisted = db.execute("SELECT state FROM parts WHERE provider_call_id = 'read-unlisted'").fetchone()
    assert invalid["state"] == "error"
    assert unreported["state"] == "error"
    assert reordered["state"] == "error"
    assert unlisted["state"] == "error"


def test_renamed_style_path_fails_until_a_new_planner_session(tmp_path: Path) -> None:
    class RenamedStyleClient:
        def __init__(self, styles_dir: Path) -> None:
            self.styles_dir = styles_dir
            self.old_path_rejected = False
            self.narrator_input: dict[str, Any] | None = None

        def bind(self) -> RenamedStyleClient:
            return self

        def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, RenamedStyleClient]:
            return {name: self for name in names}

        def metadata(self) -> dict[str, str | int | float]:
            return {"provider": "test", "model": "renamed-style", "base_url": "local", "timeout_seconds": 1}

        @staticmethod
        def result(call: ToolCall | None = None, content: str = "") -> ModelResult:
            return ModelResult(content=content, raw_response={}, tool_calls=[] if call is None else [call])

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                if not results:
                    return self.result(ToolCall(id="old-planner", name="task", arguments={"agent": "style_planner", "task": "使用旧清单规划"}))
                if len(results) == 1:
                    return self.result(ToolCall(id="new-planner", name="task", arguments={"agent": "style_planner", "task": "重新扫描后规划"}))
                if len(results) == 2:
                    report = results[-1]["result"]
                    return self.result(ToolCall(id="narrator", name="task", arguments={"agent": "narrator", "task": "按新文风写故事", "style_paths": report["style_paths"]}))
                return self.result(ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"}))
            if system_prompt == STYLE_PLANNER_PROMPT:
                initial = json.loads(messages[1]["content"])
                paths = [style["path"] for style in initial["available_styles"]]
                if STYLE_ONE in paths:
                    if not results:
                        (self.styles_dir / "冷峻悬疑.md").rename(self.styles_dir / "改名后.md")
                        return self.result(ToolCall(id="read-old", name="style_read", arguments={"paths": [STYLE_ONE]}))
                    error = results[-1]["error"]
                    self.old_path_rejected = (
                        error["code"] == "tool_error"
                        and STYLE_ONE in error["message"]
                        and "不存在" in error["message"]
                    )
                    return self.result(ToolCall(id="empty-report", name="style_report", arguments={"report": "旧路径已失效。", "style_paths": []}))
                assert STYLE_RENAMED in paths
                if not results:
                    return self.result(ToolCall(id="read-new", name="style_read", arguments={"paths": [STYLE_RENAMED]}))
                return self.result(ToolCall(id="new-report", name="style_report", arguments={"report": "使用改名后的文风。", "style_paths": [STYLE_RENAMED]}))
            if system_prompt == NARRATOR_PROMPT:
                self.narrator_input = json.loads(messages[1]["content"])
                if not results:
                    return self.result(ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": STORY}))
                return self.result(content="叙事完成。")
            raise AssertionError(system_prompt)

    data_dir = make_data_dir(tmp_path)
    model = RenamedStyleClient(data_dir / "styles")
    app = create_app(data_dir=data_dir, model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "文风改名"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        turn = wait_for_turn(client, save["id"])

    assert turn["status"] == "completed"
    assert model.old_path_rejected is True
    assert model.narrator_input is not None
    assert [style["path"] for style in model.narrator_input["style_documents"]] == [STYLE_RENAMED]
