from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient

from rpera.anthropic import request_payload as anthropic_payload
from rpera.app import create_app
from rpera.gemini import request_payload as gemini_payload
from rpera.models import ModelResult, ToolCall
from rpera.openai_responses import request_payload as responses_payload
from rpera.providers import ModelClient
from tests.helpers import make_data_dir, wait_for_turn
from tests.prompt_fixtures import CHARACTER_DESIGNER_PROMPT, COMPLIANCE_REVIEWER_PROMPT, COORDINATOR_PROMPT, NARRATOR_PROMPT, ROLE_PLAYER_PROMPT, WORLD_RESEARCHER_PROMPT


PATH = "entities/character/伊蕾/ENTITY.md"
REQUIRED = "mods/读心大师/entities/player_trait/读心能力/ENTITY.md"


class TransferModel:
    def __init__(self) -> None:
        self.received: dict[str, list[list[dict[str, Any]]]] = {}

    def bind(self) -> ModelClient:
        return self

    def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, ModelClient]:
        return {name: self for name in names}

    def metadata(self) -> dict[str, str | int | float]:
        return {"provider": "test", "model": "transfer"}

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        self.received.setdefault(system_prompt, []).append(messages)
        results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
        if system_prompt == COORDINATOR_PROMPT:
            tasks = [result for result in results if "task_id" in result]
            if not tasks:
                call = ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "调查伊蕾"})
            elif len(tasks) == 1:
                call = ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "扮演伊蕾", "related_entities": [PATH], "report_refs": [tasks[0]["report_ref"]]})
            elif len(tasks) == 2:
                call = ToolCall(id="continue", name="task", arguments={"agent": "role_player", "task": "再核对研究报告", "task_id": tasks[1]["task_id"], "report_refs": [tasks[0]["report_ref"]]})
            elif len(tasks) == 3:
                call = ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写故事", "report_refs": [tasks[2]["report_ref"]]})
            else:
                call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
            return ModelResult(content="", tool_calls=[call], raw_response={})
        if system_prompt == WORLD_RESEARCHER_PROMPT:
            assert results[0]["documents"][0]["path"] == REQUIRED
            if len(results) == 1:
                return ModelResult(content="", tool_calls=[ToolCall(id="read", name="entity_read", arguments={"paths": [PATH]})], raw_response={})
            return ModelResult(content="", tool_calls=[ToolCall(id="report", name="research_report", arguments={"report": "伊蕾是老板娘", "related_entities": [PATH]})], raw_response={})
        if system_prompt == ROLE_PLAYER_PROMPT:
            assert [item["path"] for item in results[0]["documents"]] == [PATH, REQUIRED]
            assert [item["content"] for item in results[1]["report_documents"]] == ["伊蕾是老板娘"]
            if len(results) == 2:
                return ModelResult(content="", tool_calls=[ToolCall(id="role-first", name="role_report", arguments={"report": "伊蕾保持警惕。"})], raw_response={})
            assert results[3]["report_documents"][0]["content"] == "伊蕾是老板娘"
            return ModelResult(content="", tool_calls=[ToolCall(id="role-second", name="role_report", arguments={"report": "伊蕾仍然保持警惕。"})], raw_response={})
        if system_prompt == NARRATOR_PROMPT:
            if len(results) == 2:
                assert results[1]["report_documents"][0]["content"] == "伊蕾仍然保持警惕。"
                return ModelResult(content="", tool_calls=[ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "故事完成。"})], raw_response={})
            return ModelResult(content="完成。", raw_response={})
        raise AssertionError(system_prompt)


def test_transfers_are_persisted_as_separate_calls_before_model_request(tmp_path: Any) -> None:
    model = TransferModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "mod_names": ["读心大师"], "name": "自动传递"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "我询问伊蕾。"})
        turn = wait_for_turn(client, save["id"])
    assert turn["status"] == "completed"
    expected = {
        WORLD_RESEARCHER_PROMPT: ["entity_read"],
        ROLE_PLAYER_PROMPT: ["entity_read", "report_read"],
        NARRATOR_PROMPT: ["entity_read", "report_read"],
    }
    for prompt, expected_tools in expected.items():
        messages = model.received[prompt][0]
        user = next(value for item in messages if item["role"] == "user" and isinstance(item["content"], str) and item["content"].startswith("{") if "task" in (value := json.loads(item["content"])))
        assert "report_documents" not in user
        assert "related_entity_documents" not in user
        synthetic = [item for item in messages if item.get("tool_calls") and item["tool_calls"][0]["id"].startswith("call-")]
        assert [item["tool_calls"][0]["function"]["name"] for item in synthetic] == expected_tools
        for item in synthetic:
            call = item["tool_calls"][0]
            reply = next(value for value in messages if value["role"] == "tool" and value["tool_call_id"] == call["id"])
            output = json.loads(reply["content"])
            assert output["target_agent"]
            if call["function"]["name"] == "entity_read":
                assert set(json.loads(call["function"]["arguments"])) == {"paths"}
                assert output["documents"]
            else:
                assert set(json.loads(call["function"]["arguments"])) == {"report_refs"}
                assert "documents" not in output
        if len(synthetic) == 2:
            indexes = [messages.index(item) for item in synthetic]
            assert indexes[1] == indexes[0] + 2

    continued = model.received[ROLE_PLAYER_PROMPT][1]
    assert json.loads([item["content"] for item in continued if item["role"] == "user"][-1])["continuation"] is True
    assert sum(item.get("role") == "tool" for item in continued) == 4
    assert [item["tool_calls"][0]["function"]["name"] for item in continued if item.get("tool_calls")] == ["entity_read", "report_read", "role_report", "report_read"]

    with app.state.saves.connect(save["id"]) as db:
        rows = db.execute("SELECT tool_name, state, input, output FROM parts WHERE provider_call_id LIKE 'call-%'").fetchall()
        assert len(rows) == 6
        assert all(row["state"] == "completed" for row in rows)
        assert not db.execute("SELECT 1 FROM parts WHERE tool_name IN ('context_transfer_read', 'entity_transfer_read', 'report_transfer_read')").fetchone()


def test_synthetic_call_round_trips_through_native_provider_formats() -> None:
    messages = [
        {"role": "user", "content": '{"task": "read"}'},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-123", "type": "function", "function": {"name": "entity_read", "arguments": '{"paths": ["entities/rule/规则/ENTITY.md"]}'}}]},
        {"role": "tool", "tool_call_id": "call-123", "content": '{"documents": [{"content": "规则"}]}'},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call-456", "type": "function", "function": {"name": "report_read", "arguments": '{"report_refs": [{"name": "报告", "id": "123"}]}'}}]},
        {"role": "tool", "tool_call_id": "call-456", "content": '{"report_documents": [{"content": "报告"}]}'},
    ]
    anthropic = anthropic_payload("system", messages, [], model="test", temperature=1, top_p=1, max_tokens=100, stream=False)["messages"]
    assert [item["content"][0]["tool_use_id"] for item in anthropic if item["role"] == "user" and isinstance(item["content"], list) and item["content"][0]["type"] == "tool_result"] == ["call-123", "call-456"]
    gemini = gemini_payload("system", messages, [], temperature=1, top_p=1, max_tokens=100)["contents"]
    assert [item["parts"][0]["functionCall"]["id"] for item in gemini if item["role"] == "model"] == ["call-123", "call-456"]
    assert [item["parts"][0]["functionResponse"]["id"] for item in gemini if item["role"] == "user" and "functionResponse" in item["parts"][0]] == ["call-123", "call-456"]
    responses = responses_payload("system", messages, [], model="test", max_tokens=100, stream=False)["input"]
    assert [item["call_id"] for item in responses if item.get("type") == "function_call"] == ["call-123", "call-456"]
    assert [item["call_id"] for item in responses if item.get("type") == "function_call_output"] == ["call-123", "call-456"]


def test_no_transfer_and_report_only_do_not_create_empty_entity_calls(tmp_path: Any) -> None:
    class ReportOnlyModel(TransferModel):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            self.received.setdefault(system_prompt, []).append(messages)
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                tasks = [result for result in results if "task_id" in result]
                if not tasks:
                    call = ToolCall(id="review", name="task", arguments={"agent": "compliance_reviewer", "task": "审核"})
                elif len(tasks) == 1:
                    call = ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "表演", "report_refs": [tasks[0]["report_ref"]]})
                elif len(tasks) == 2:
                    call = ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写故事"})
                else:
                    call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
                return ModelResult(content="", raw_response={}, tool_calls=[call])
            if system_prompt == COMPLIANCE_REVIEWER_PROMPT:
                return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="approved", name="compliance_report", arguments={"approved": True, "reason": "通过"})])
            if system_prompt == ROLE_PLAYER_PROMPT:
                assert len(results) == 1 and results[0]["report_documents"][0]["source_agent"] == "compliance_reviewer"
                return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="role", name="role_report", arguments={"report": "角色意见"})])
            if system_prompt == NARRATOR_PROMPT:
                if not results:
                    return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "故事完成。"})])
                return ModelResult(content="完成", raw_response={})
            raise AssertionError(system_prompt)

    model = ReportOnlyModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "报告单独传递"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

    def synthetic_names(prompt: str) -> list[str]:
        return [item["tool_calls"][0]["function"]["name"] for item in model.received[prompt][0] if item.get("tool_calls")]

    assert synthetic_names(COMPLIANCE_REVIEWER_PROMPT) == []
    assert synthetic_names(ROLE_PLAYER_PROMPT) == ["report_read"]
    assert synthetic_names(NARRATOR_PROMPT) == []
    with app.state.saves.connect(save["id"]) as db:
        assert [row[0] for row in db.execute("SELECT tool_name FROM parts WHERE provider_call_id LIKE 'call-%'")] == ["report_read"]


def test_child_can_read_unprovided_entity_and_verified_report(tmp_path: Any) -> None:
    class ReadingModel(TransferModel):
        report_ref: dict[str, str] | None = None

        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            self.received.setdefault(system_prompt, []).append(messages)
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                tasks = [item for item in results if "task_id" in item]
                if not tasks:
                    call = ToolCall(id="research", name="task", arguments={"agent": "world_researcher", "task": "调查"})
                elif len(tasks) == 1:
                    self.report_ref = tasks[0]["report_ref"]
                    call = ToolCall(id="role", name="task", arguments={"agent": "role_player", "task": "自行查证资料"})
                elif len(tasks) == 2:
                    call = ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写故事"})
                else:
                    call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
                return ModelResult(content="", raw_response={}, tool_calls=[call])
            if system_prompt == WORLD_RESEARCHER_PROMPT:
                return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="summary", name="research_report", arguments={"report": "查明事实", "related_entities": []})])
            if system_prompt == ROLE_PLAYER_PROMPT:
                assert tools is not None
                assert {"entity_read", "report_read"} <= {item["function"]["name"] for item in tools}
                if len(results) == 1:
                    return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="read-character", name="entity_read", arguments={"paths": [PATH]})])
                if len(results) == 2:
                    assert results[1]["documents"][0]["path"] == PATH
                    assert self.report_ref is not None
                    return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="read-report", name="report_read", arguments={"report_refs": [self.report_ref]})])
                assert results[2]["report_documents"][0]["content"] == "查明事实"
                if len(results) == 3:
                    return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="invalid-report", name="report_read", arguments={"report_refs": [{"name": "world_researcher 报告", "id": "not-a-real-id"}]})])
                assert results[3]["error"]["code"] == "report_not_found"
                return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="role", name="role_report", arguments={"report": "角色意见"})])
            if system_prompt == NARRATOR_PROMPT:
                if len(results) == 1:
                    return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "完成。"})])
                return ModelResult(content="完成", raw_response={})
            raise AssertionError(system_prompt)

    model = ReadingModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "mod_names": ["读心大师"], "name": "主动读取"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"
    role_messages = model.received[ROLE_PLAYER_PROMPT][-1]
    assert [item["tool_calls"][0]["function"]["name"] for item in role_messages if item.get("tool_calls")] == ["entity_read", "entity_read", "report_read", "report_read"]


def test_character_designer_rereads_current_file_after_edit(tmp_path: Any) -> None:
    path = "entities/character/新角色/ENTITY.md"

    class EditingModel(TransferModel):
        async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
            self.received.setdefault(system_prompt, []).append(messages)
            results = [json.loads(message["content"]) for message in messages if message["role"] == "tool"]
            if system_prompt == COORDINATOR_PROMPT:
                tasks = [item for item in results if "task_id" in item]
                if not tasks:
                    call = ToolCall(id="design", name="task", arguments={"agent": "character_designer", "task": "建立角色"})
                elif len(tasks) == 1:
                    call = ToolCall(id="narrate", name="task", arguments={"agent": "narrator", "task": "写故事"})
                else:
                    call = ToolCall(id="publish", name="narrative_publish", arguments={"path": "narrative.md"})
                return ModelResult(content="", raw_response={}, tool_calls=[call])
            if system_prompt == CHARACTER_DESIGNER_PROMPT:
                if not results:
                    call = ToolCall(id="create", name="character_create", arguments={"name": "新角色", "description": "测试角色", "aliases": [], "profile": {"trait": "甲"}})
                elif len(results) == 1:
                    assert results[0]["character"]["path"] == path
                    call = ToolCall(id="edit", name="character_edit", arguments={"path": path, "old_text": '"trait": "甲"', "new_text": '"trait": "乙"'})
                elif len(results) == 2:
                    call = ToolCall(id="read", name="entity_read", arguments={"paths": [path]})
                else:
                    assert '"trait": "乙"' in results[2]["documents"][0]["document"]
                    call = ToolCall(id="report", name="character_report", arguments={"report": "角色已更新", "related_entities": [path]})
                return ModelResult(content="", raw_response={}, tool_calls=[call])
            if system_prompt == NARRATOR_PROMPT:
                if not results:
                    return ModelResult(content="", raw_response={}, tool_calls=[ToolCall(id="write", name="file_write", arguments={"path": "narrative.md", "content": "完成。"})])
                return ModelResult(content="完成", raw_response={})
            raise AssertionError(system_prompt)

    model = EditingModel()
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "角色重读"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"
