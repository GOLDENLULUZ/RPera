from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.model_text import render_model_text
from rpera.models import AiPreset, ModelResult, ToolCall
from rpera.providers import BoundModelClient
from rpera.runtime import AgentRunner
from tests.helpers import make_data_dir, wait_for_turn
from tests.model_text_helpers import decode_model_text
from tests.prompt_fixtures import CHARACTER_DESIGNER_PROMPT, COORDINATOR_PROMPT, NARRATOR_PROMPT


EMPTY_STORY = {"story": {"opening": "", "turns": []}}
PROFILE = {
    "appearance": {
        "height": "168cm",
        "notes": '她说："保持原样"\n第二行',
        "path": r"C:\Users\资料",
        "example": "```json\n{}\n```",
    },
}
RAW = json.dumps(PROFILE, ensure_ascii=False, indent=2) + "\n"
PATH = "entities/character/测试角色/ENTITY.md"


def tool_session(name: str, output: Any, *, state: str = "completed") -> list[dict[str, Any]]:
    return [{"role": "assistant", "model": "{}", "parts": [{
        "type": "tool", "tool_name": name, "state": state, "provider_call_id": "read-call",
        "input": "{}", "output": json.dumps(output, ensure_ascii=False),
    }]}]


def test_readable_text_preserves_literal_characters_boundaries_and_structure() -> None:
    value = {
        "documents": [{"path": PATH, "document": RAW}],
        "strange.key": {"colon: key": '空白 \t\r\n反斜杠 \\ 和真实的 \\" 引号\n`````\n$ — object'},
        "empty": "", "array": [], "object": {}, "boolean": True, "none": None, "number": 42,
    }
    rendered = render_model_text(value)
    assert RAW in rendered
    assert json.dumps(RAW, ensure_ascii=False)[1:-1] not in rendered
    assert "``````text\n" in rendered
    assert decode_model_text(rendered) == value


@pytest.mark.parametrize("name,output", [
    ("entity_read", {"documents": [{"path": PATH, "document": RAW}, {"path": "entities/rule/规则/ENTITY.md", "content": RAW}]}),
    *[(name, {"character": {"path": PATH, "document": RAW}}) for name in ("character_create", "character_edit", "character_rename")],
    *[(name, {"location": {"path": "entities/location/房间/ENTITY.md", "document": RAW}}) for name in ("location_create", "location_edit", "location_rename")],
    *[(name, {"goal": {"path": "entities/goal/目标清单/ENTITY.md", "document": RAW}}) for name in ("goal_read", "goal_create", "goal_edit")],
    *[(name, {"path": "narrative.md", "content": RAW}) for name in ("file_read", "file_write", "file_edit", "narrative_publish")],
    *[(name, {"path": "story_summary.json", "content": RAW, "summary": PROFILE}) for name in ("story_summary_read", "story_summary_edit")],
    ("style_read", {"documents": [{"path": "styles/文风.md", "content": RAW}]}),
    ("report_read", {"report_documents": [{"name": "世界报告", "id": "report-id", "source_agent": "world_researcher", "source_task": RAW, "content": RAW}]}),
    ("task", {"task_id": "child-id", "result": {"report": RAW}, "related_entity_documents": [{"path": PATH, "content": RAW}]}),
    ("task", {"task_id": "child-id", "result": RAW, "goal": {"content": RAW}}),
    ("research_report", {"report": RAW, "related_entities": [PATH]}),
    ("style_report", {"report": RAW, "style_paths": ["styles/文风.md"]}),
    ("compliance_report", {"approved": True, "reason": RAW}),
    ("character_report", {"report": RAW, "related_entities": [PATH]}),
    ("location_report", {"report": RAW, "related_entities": []}),
    ("role_report", {"report": RAW}),
    ("erotic_report", {"report": RAW}),
])
def test_all_textual_tool_results_expose_original_text(name: str, output: dict[str, Any]) -> None:
    session = tool_session(name, output)
    original = session[0]["parts"][0]["output"]
    messages = AgentRunner._provider_messages(session, EMPTY_STORY)
    text = messages[-1]["content"]
    assert RAW in text
    assert json.dumps(RAW, ensure_ascii=False)[1:-1] not in text
    assert decode_model_text(text) == output
    assert session[0]["parts"][0]["output"] == original


def test_initial_and_continued_child_context_and_history_are_verbatim() -> None:
    story = {"story": {"opening": RAW, "turns": [
        {"turn_number": 1, "player": {"content": RAW, "images": []}, "has_ai_output": True, "narrative": RAW},
        {"turn_number": 2, "player": {"content": RAW, "images": []}, "has_ai_output": False},
    ]}}
    initial = {"task": {"task": RAW, "reason": RAW, "related_entities": [PATH]}, "style_documents": [{"path": "styles/文风.md", "content": RAW}]}
    continued = {"task": {"task": RAW}, "continuation": True}
    session = [{"role": "user", "parts": [{"type": "text", "content": json.dumps(value, ensure_ascii=False)}]}
               for value in (initial, continued)]
    history = {"turns": story["story"]["turns"][:1], "next_after_turn_number": 1}
    session.extend(tool_session("story_history_read", history))
    messages = AgentRunner._provider_messages(session, story)
    assert [decode_model_text(message["content"]) for message in messages if message["role"] == "user"] == [story, initial, continued]
    assert decode_model_text(messages[-1]["content"]) == history
    for message in messages:
        if message["role"] in {"user", "tool"}:
            assert RAW in message["content"]


def test_structured_compliance_report_content_is_unfolded_without_changing_storage() -> None:
    review = {"approved": True, "reason": RAW}
    output = {"report_documents": [{
        "name": "审核报告", "id": "report-id", "source_agent": "compliance_reviewer",
        "content": json.dumps(review, ensure_ascii=False),
    }]}
    session = tool_session("report_read", output)
    stored = session[0]["parts"][0]["output"]
    text = AgentRunner._provider_messages(session, EMPTY_STORY)[-1]["content"]
    assert RAW in text
    assert decode_model_text(text)["report_documents"][0]["content"] == review
    assert session[0]["parts"][0]["output"] == stored


def test_fixed_response_assistant_history_preserves_reason_as_original_text() -> None:
    review = {"approved": True, "reason": RAW}
    stored = json.dumps(review, ensure_ascii=False)
    session = [{"role": "assistant", "model": json.dumps({"response_source": "fixed_file"}),
                "parts": [{"type": "text", "content": stored}]}]
    text = AgentRunner._provider_messages(session, EMPTY_STORY)[-1]["content"]
    assert RAW in text
    assert decode_model_text(text) == review
    assert session[0]["parts"][0]["content"] == stored


def test_image_labels_history_and_visual_tool_results_keep_text_and_image_order() -> None:
    encoded = base64.b64encode(b"picture").decode("ascii")
    image = {"path": "artifacts/玩家.png", "mime_type": "image/png", "data": encoded}
    generated = {**image, "path": "artifacts/生成.png", "sha256": hashlib.sha256(b"picture").hexdigest()}
    story = {"story": {"opening": "", "turns": [{
        "turn_number": 1, "player": {"content": RAW, "images": [{"id": "player-id"}]},
        "has_ai_output": False, "generated_images": [{"id": "generated-id"}],
    }]}}
    readers = {
        "image_reader": lambda _save, _id: image,
        "generated_reader": lambda _save, _id: generated,
        "artifact_reader": lambda _save, _path: generated,
    }
    child = AgentRunner._provider_messages([], story, "save", **readers)[0]["content"]
    assert decode_model_text(child[0]["text"]) == story
    assert decode_model_text(child[1]["text"])["path"] == image["path"]
    assert decode_model_text(child[3]["text"])["source"] == "generated"
    assert [block["image_url"]["url"] for block in child if block["type"] == "image_url"] == [f"data:image/png;base64,{encoded}"] * 2
    main = AgentRunner._provider_messages([], story, "save", coordinator=True, **readers)
    player = main[1]["content"]
    assert player[0]["text"] == RAW
    assert decode_model_text(player[1]["text"])["image_id"] == "player-id"

    history = {"turns": story["story"]["turns"], "next_after_turn_number": 1}
    rebuilt = AgentRunner._provider_messages(tool_session("story_history_read", history), EMPTY_STORY, "save", **readers)[-1]["content"]
    assert decode_model_text(rebuilt[0]["text"]) == history
    assert decode_model_text(rebuilt[1]["text"])["path"] == image["path"]
    assert rebuilt[2]["type"] == "image_url"
    assert decode_model_text(rebuilt[3]["text"])["image_id"] == "generated-id"
    assert rebuilt[4]["type"] == "image_url"
    for name, output in (("image_read", image), ("character_portrait_generate", {key: value for key, value in generated.items() if key != "data"})):
        blocks = AgentRunner._provider_messages(tool_session(name, output), EMPTY_STORY, "save", **readers)[-1]["content"]
        assert decode_model_text(blocks[0]["text"])["path"] == output["path"]
        assert encoded not in blocks[0]["text"]
        assert blocks[1]["image_url"]["url"] == f"data:image/png;base64,{encoded}"


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,protocol", [
    ("deepseek", "chat_completions"), ("openai_compatible", "chat_completions"),
    ("openai_compatible", "responses"), ("anthropic", "chat_completions"),
    ("google_gemini", "chat_completions"), ("xai", "responses"), ("xai", "chat_completions"),
])
async def test_native_http_requests_contain_original_tool_text(provider: str, protocol: str) -> None:
    projected = AgentRunner._provider_messages(tool_session("file_read", {"path": "narrative.md", "content": RAW}), EMPTY_STORY)
    expected = projected[-1]["content"]

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if provider == "google_gemini":
            parts = payload["contents"][-1]["parts"]
            assert parts[0]["functionResponse"]["response"] == {"result": "工具结果见后续原文文本。"}
            assert parts[1] == {"text": expected}
            return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "完成"}]}, "finishReason": "STOP"}]})
        if provider == "anthropic":
            assert payload["messages"][-1]["content"][0]["content"] == expected
            return httpx.Response(200, json={"content": [{"type": "text", "text": "完成"}]})
        if protocol == "responses":
            assert payload["input"][-1]["output"] == expected
            return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "完成"}]}]})
        assert payload["messages"][-1]["content"] == expected
        return httpx.Response(200, json={"choices": [{"message": {"content": "完成"}}]})

    preset = AiPreset(id="test", name="test", provider=provider, api_key="test-key", base_url="https://provider.example/v1", model="test", openai_protocol=protocol, xai_protocol=protocol if provider == "xai" else "responses")
    result = await BoundModelClient(preset, transport=httpx.MockTransport(handler)).complete("system", projected)
    assert result.content == "完成"


class ExactEditingModel:
    def __init__(self, interrupt: bool) -> None:
        self.interrupt = interrupt
        self.failed_once = False
        self.retry_started = False
        self.original_document = ""
        self.received_documents: list[str] = []

    def bind_for_agents(self, names: tuple[str, ...]):
        return {name: self for name in names}

    def bind(self):
        return self

    def metadata(self):
        return {"provider": "test", "model": "exact-edit"}

    @staticmethod
    def call(name: str, **arguments: Any) -> ModelResult:
        # Exercise the real JSON decoding boundary of native Tool arguments.
        return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="reused", name=name, arguments=json.dumps(arguments, ensure_ascii=False))])

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools=None) -> ModelResult:
        outputs = [decode_model_text(message["content"]) for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            tasks = [output for output in outputs if "task_id" in output]
            character_tasks = [output for output in tasks if output["agent"] == "character_designer"]
            if not any(output.get("ok") for output in character_tasks):
                if character_tasks and not self.retry_started:
                    return ModelResult(content="等待显式重试。", raw_response={})
                continuation = {"task_id": character_tasks[-1]["task_id"]} if character_tasks else {}
                return self.call("task", agent="character_designer", task='更新身高，保留所有引号和真实反斜杠。\n仅替换 "height"。', **continuation)
            if not any(output.get("agent") == "narrator" and output.get("ok") for output in tasks):
                return self.call("task", agent="narrator", task="确认角色更新。")
            return self.call("narrative_publish", path="narrative.md")
        if system_prompt == CHARACTER_DESIGNER_PROMPT:
            documents = [document for output in outputs for document in output.get("documents", []) if document["path"] == PATH]
            if not documents:
                return self.call("entity_read", paths=[PATH])
            document = documents[-1]["document"]
            visible = next(message["content"] for message in reversed(messages) if message["role"] == "tool" and document in message["content"])
            assert document in visible
            assert document == self.original_document
            self.received_documents.append(document)
            if self.interrupt and not self.failed_once:
                self.failed_once = True
                raise RuntimeError("模拟读取后接口中断")
            if not any(output.get("operation") == "edit" for output in outputs):
                old_text = '  "appearance": {\n    "height": "168cm",'
                assert document.count(old_text) == 1
                return self.call("character_edit", path=PATH, old_text=old_text, new_text=old_text.replace("168cm", "169cm"))
            return self.call("character_report", report="已精确更新。", related_entities=[PATH])
        if system_prompt == NARRATOR_PROMPT:
            if not any(output.get("path") == "narrative.md" for output in outputs):
                return self.call("file_write", path="narrative.md", content='角色说："已更新"。\n保留路径 C:\\Users。')
            return ModelResult(content="完成。", raw_response={})
        raise AssertionError(system_prompt)


@pytest.mark.parametrize("interrupt", [False, True])
def test_verbatim_entity_read_supports_exact_edit_and_explicit_retry(tmp_path: Path, interrupt: bool) -> None:
    model = ExactEditingModel(interrupt)
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "原文编辑"}).json()
        app.state.saves.create_character(save["id"], "测试角色", "精确编辑测试", [], PROFILE)
        model.original_document = app.state.saves.read_entities_exact_for_character_edit(save["id"], [PATH])[0]["document"]
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "更新角色身高。"})
        assert response.status_code == 202
        turn = wait_for_turn(client, save["id"])
        if interrupt:
            assert turn["status"] == "failed"
            model.retry_started = True
            assert client.post(f"/api/saves/{save['id']}/turns/{turn['id']}/retry").status_code == 202
            resumed = wait_for_turn(client, save["id"])
            assert resumed["id"] == turn["id"]
            turn = resumed
        assert turn["status"] == "completed"
    updated = app.state.saves.read_entities_exact(save["id"], [PATH])[0]
    assert json.loads(updated["content"]) == {"appearance": {**PROFILE["appearance"], "height": "169cm"}}
    assert all(document == model.original_document for document in model.received_documents)
    with app.state.saves.connect(save["id"]) as db:
        edits = db.execute("SELECT state,input,output FROM parts WHERE tool_name = 'character_edit'").fetchall()
        assert len(edits) == 1 and edits[0]["state"] == "completed"
        assert json.loads(edits[0]["input"])["old_text"] in model.original_document
        assert '"height": "169cm"' in json.loads(edits[0]["output"])["character"]["document"]
        assert db.execute("SELECT COUNT(*) FROM sessions WHERE agent = 'character_designer'").fetchone()[0] == 1
