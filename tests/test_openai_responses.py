from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.model_errors import ModelFallbackError
from rpera.models import AiPreset, ModelResult, ToolCall
from rpera.providers import BoundModelClient
from rpera.runtime import AgentRunner
from tests.helpers import make_data_dir, wait_for_turn
from tests.prompt_fixtures import COORDINATOR_PROMPT, NARRATOR_PROMPT


def preset(**changes: Any) -> AiPreset:
    return AiPreset(
        id="openai", name="推理模型", provider="openai_compatible",
        base_url="https://provider.example/v1", api_key="test-key", model="gpt-6.1-sol",
        **{"openai_protocol": "responses", **changes},
    )


def test_openai_protocol_defaults_to_chat_and_persists_responses(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        collection = client.get("/api/ai-presets").json()
        existing = collection["main_preset_id"]
        assert collection["presets"][0]["openai_protocol"] == "chat_completions"
        payload = {
            "name": "推理模型", "provider": "openai_compatible",
            "base_url": "https://provider.example/v1", "api_key": "test-key",
            "model": "gpt-6.1-sol", "openai_protocol": "responses",
        }
        response = client.put(f"/api/ai-presets/{existing}", json=payload)
        assert response.status_code == 200
        assert response.json()["presets"][0]["openai_protocol"] == "responses"
        assert 'id="openai-protocol"' in client.get("/").text
        assert app.state.presets.resolve(("coordinator",))["coordinator"].openai_protocol == "responses"
        assert json.loads((data_dir / "config" / "ai_presets.json").read_text())["presets"][0]["openai_protocol"] == "responses"
        # Existing saved presets did not contain this new field.
        old_data = json.loads((data_dir / "config" / "ai_presets.json").read_text())
        old_data["presets"][0].pop("openai_protocol")
        (data_dir / "config" / "ai_presets.json").write_text(json.dumps(old_data))
        assert client.get("/api/ai-presets").json()["presets"][0]["openai_protocol"] == "chat_completions"


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_openai_responses_replays_native_reasoning_tool_and_images(streaming: bool) -> None:
    requests: list[dict[str, Any]] = []
    native_output = [
        {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "先读文件"}], "encrypted_content": "opaque"},
        {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "file_read", "arguments": '{"path":"narrative.md"}'},
    ]
    url = "data:image/png;base64,aGVsbG8="

    def reply(output: list[dict[str, Any]], usage: dict[str, Any] | None = None) -> httpx.Response:
        response = {"status": "completed", "output": output, "usage": usage or {}}
        if not streaming:
            return httpx.Response(200, json=response)
        event = {"type": "response.completed", "response": response}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        assert request.headers["authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == "gpt-6.1-sol"
        assert payload["store"] is False
        assert payload["include"] == ["reasoning.encrypted_content"]
        assert payload["parallel_tool_calls"] is False
        assert payload["max_output_tokens"] == 65_536
        assert "temperature" not in payload and "top_p" not in payload
        if len(requests) == 1:
            assert payload["tools"][0]["name"] == "file_read"
            assert payload["input"][-1] == {"role": "assistant", "content": "预填"}
            assert payload["input"][0]["content"] == [{"type": "input_text", "text": "读取"}, {"type": "input_image", "image_url": url}]
            return reply(native_output, {"total_tokens": 8})
        assert payload["input"][1:3] == native_output
        assert payload["input"][3] == {"type": "function_call_output", "call_id": "call_1", "output": '{"ok":true}'}
        assert payload["input"][4] == {"role": "user", "content": [{"type": "input_image", "image_url": url}]}
        return reply([{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "完成"}]}])

    client = BoundModelClient(preset(), transport=httpx.MockTransport(handler), streaming=streaming)
    tools = [{"type": "function", "function": {"name": "file_read", "description": "Read", "parameters": {"type": "object"}}}]
    first = await client.complete("system", [
        {"role": "user", "content": [{"type": "text", "text": "读取"}, {"type": "image_url", "image_url": {"url": url}}]},
        {"role": "assistant", "content": "预填", "_prefix": True},
    ], tools)
    assert first.reasoning == "先读文件"
    assert first.tool_calls == [ToolCall(id="call_1", name="file_read", arguments='{"path":"narrative.md"}')]
    assert first.provider_content == {"output": native_output}
    second = await client.complete("system", [
        {"role": "user", "content": "读取"},
        {"role": "assistant", "content": None, "_provider_content": first.provider_content, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "file_read", "arguments": '{"path":"narrative.md"}'}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": '{"ok":true}'}, {"type": "image_url", "image_url": {"url": url}}]},
    ], tools)
    assert second.content == "完成"
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_openai_responses_stream_and_connection_test_use_responses_endpoint() -> None:
    requests: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        requests.append(payload)
        response = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}], "usage": {"total_tokens": 4}}
        if payload["stream"]:
            events = [{"type": "response.created", "response": {"id": "resp_1"}}, {"type": "response.completed", "response": response}]
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode())
        return httpx.Response(200, json=response)

    client = BoundModelClient(preset(), transport=httpx.MockTransport(handler), streaming=True)
    streamed = await client.complete("system", [{"role": "user", "content": "只回复 OK"}])
    assert streamed.content == "OK"
    assert streamed.usage == {"total_tokens": 4}
    assert len(streamed.raw_response["events"]) == 2
    tested = await client.test_connection()
    assert tested.content == "OK"
    assert [item["stream"] for item in requests] == [True, False]
    assert requests[1]["max_output_tokens"] == 256


@pytest.mark.asyncio
async def test_openai_chat_rejection_requires_explicit_responses_selection() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("chat/completions"):
            return httpx.Response(400, json={"error": {"message": "Function tools with reasoning_effort are not supported for gpt-6.1-sol"}})
        return httpx.Response(200, json={"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "成功"}]}]})

    transport = httpx.MockTransport(handler)
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object"}}}]
    with pytest.raises(RuntimeError, match="reasoning_effort"):
        await BoundModelClient(preset(openai_protocol="chat_completions"), transport=transport).complete("system", [{"role": "user", "content": "请求"}], tools)
    result = await BoundModelClient(preset(openai_protocol="responses"), transport=transport).complete("system", [{"role": "user", "content": "请求"}], tools)
    assert result.content == "成功"
    assert paths == ["/v1/chat/completions", "/v1/responses"]


@pytest.mark.asyncio
async def test_openai_responses_exposes_refusal_and_incomplete_errors() -> None:
    responses = [
        {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []},
        {"status": "completed", "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "request refused"}]}]},
    ]
    client = BoundModelClient(preset(openai_protocol="responses"), transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=responses.pop(0))))
    with pytest.raises(RuntimeError, match="生成未完成.*max_output_tokens"):
        await client.complete("system", [{"role": "user", "content": "请求"}])
    with pytest.raises(ModelFallbackError, match="拒绝生成.*request refused"):
        await client.complete("system", [{"role": "user", "content": "请求"}])


def test_openai_protocol_switch_drops_only_foreign_native_content() -> None:
    story = {"story": {"opening": "", "turns": [{"turn_number": 1, "player": {"content": "继续", "images": []}, "has_ai_output": False}]}}
    native = {"output": [{"id": "rs_1", "type": "reasoning", "encrypted_content": "opaque"}, {"type": "function_call", "call_id": "call_1", "name": "noop", "arguments": "{}"}]}
    messages = [
        {"role": "user", "model": None, "parts": [{"type": "text", "content": json.dumps({"task": {"task": "任务"}}, ensure_ascii=False)}]},
        {"role": "assistant", "model": json.dumps({"provider": "openai_compatible", "openai_protocol": "responses", "xai_protocol": "responses", "provider_content": native}),
         "parts": [{"type": "tool", "content": None, "provider_call_id": "call_1", "tool_name": "noop", "input": "{}", "state": "completed", "output": '{"ok":true}'}]},
    ]
    response_messages = AgentRunner._provider_messages(messages, story, target_model={"provider": "openai_compatible", "openai_protocol": "responses", "xai_protocol": "responses"})
    chat_messages = AgentRunner._provider_messages(messages, story, target_model={"provider": "openai_compatible", "openai_protocol": "chat_completions", "xai_protocol": "responses"})
    assert response_messages[2]["_provider_content"] == native
    assert "_provider_content" not in chat_messages[2]
    assert chat_messages[2]["tool_calls"][0]["function"]["name"] == "noop"
    assert chat_messages[3]["role"] == "tool"
    other_model = AgentRunner._provider_messages(messages, story, target_model={
        "provider": "openai_compatible", "openai_protocol": "responses", "xai_protocol": "responses", "model": "another-model",
    })
    assert "_provider_content" not in other_model[2]


def test_openai_responses_completes_root_and_child_tool_cycle(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        payload = json.loads(request.content)
        calls.append(payload)
        instructions = payload["instructions"]
        outputs = [item for item in payload["input"] if item.get("type") == "function_call_output"]
        if instructions == COORDINATOR_PROMPT:
            if not outputs:
                item = {"type": "function_call", "call_id": "root-task", "name": "task", "arguments": '{"agent":"narrator","task":"写故事"}'}
            else:
                assert outputs[-1]["call_id"] == "root-task"
                assert payload["input"][-2]["call_id"] == "root-task"
                item = {"type": "function_call", "call_id": "publish", "name": "narrative_publish", "arguments": '{"path":"narrative.md"}'}
        elif instructions == NARRATOR_PROMPT:
            if not outputs:
                item = {"type": "function_call", "call_id": "write", "name": "file_write", "arguments": '{"path":"narrative.md","content":"故事已写。"}'}
            else:
                assert outputs[-1]["call_id"] == "write"
                item = {"type": "message", "content": [{"type": "output_text", "text": "写作完成"}]}
        else:
            raise AssertionError("unexpected prompt")
        return httpx.Response(200, json={"status": "completed", "output": [item]})

    model = BoundModelClient(preset(openai_protocol="responses"), transport=httpx.MockTransport(handler))
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=model)
    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "Responses 工具流程"}).json()
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续故事"})
        turn = wait_for_turn(client, save["id"])
    assert turn["status"] == "completed", turn
    assert turn["narrative"] == "故事已写。"
    assert len(calls) == 4
    assert all(call["parallel_tool_calls"] is False for call in calls)
    assert all(call["store"] is False for call in calls)
    with app.state.saves.connect(save["id"]) as db:
        stored = [json.loads(row["model"]) for row in db.execute("SELECT model FROM messages WHERE role = 'assistant' AND kind = 'model' AND model IS NOT NULL")]
    assert all(item["openai_protocol"] == "responses" for item in stored)
    assert any(item.get("provider_content", {}).get("output", [{}])[0].get("call_id") == "root-task" for item in stored)
