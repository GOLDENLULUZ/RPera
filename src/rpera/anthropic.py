from __future__ import annotations

import json
from typing import Any

from .message_compat import content_blocks, image_data_url, system_messages_as_user
from .models import ModelResult, ToolCall


def request_payload(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "system": system_prompt,
        "messages": _messages(system_messages_as_user(messages)),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }
    declarations = [_tool(tool) for tool in tools or []]
    if declarations:
        payload["tools"] = declarations
        payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
    return payload


def parse_response(raw: dict[str, Any]) -> ModelResult:
    error = raw.get("error")
    if isinstance(error, dict):
        raise RuntimeError(f"Anthropic 接口返回错误：{json.dumps(error, ensure_ascii=False)}")
    blocks = raw.get("content")
    if not isinstance(blocks, list):
        raise RuntimeError("Anthropic 响应缺少 content 数组")
    content, reasoning, calls = _blocks(blocks)
    usage = raw.get("usage")
    return ModelResult(
        content=content,
        reasoning=reasoning,
        usage=usage if isinstance(usage, dict) else {},
        raw_response=raw,
        tool_calls=calls,
        provider_content={"content": blocks},
    )


def parse_stream(events: list[dict[str, Any]]) -> ModelResult:
    blocks: dict[int, dict[str, Any]] = {}
    partial_json: dict[int, str] = {}
    usage: dict[str, Any] = {}
    completed = False
    for event in events:
        event_type = event.get("type")
        if event_type == "error":
            error = event.get("error")
            raise RuntimeError(f"Anthropic 接口返回错误：{json.dumps(error, ensure_ascii=False)}")
        if event_type == "message_start":
            message = event.get("message")
            start_usage = message.get("usage") if isinstance(message, dict) else None
            if isinstance(start_usage, dict):
                usage.update(start_usage)
            continue
        if event_type == "content_block_start" and isinstance(event.get("index"), int):
            block = event.get("content_block")
            if isinstance(block, dict):
                blocks[event["index"]] = dict(block)
            continue
        if event_type == "content_block_delta" and isinstance(event.get("index"), int):
            index = event["index"]
            delta = event.get("delta")
            block = blocks.get(index)
            if not isinstance(delta, dict) or block is None:
                continue
            delta_type = delta.get("type")
            if delta_type == "text_delta" and isinstance(delta.get("text"), str):
                block["text"] = str(block.get("text") or "") + delta["text"]
            elif delta_type == "thinking_delta" and isinstance(delta.get("thinking"), str):
                block["thinking"] = str(block.get("thinking") or "") + delta["thinking"]
            elif delta_type == "signature_delta" and isinstance(delta.get("signature"), str):
                block["signature"] = str(block.get("signature") or "") + delta["signature"]
            elif delta_type == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                partial_json[index] = partial_json.get(index, "") + delta["partial_json"]
            continue
        if event_type == "content_block_stop" and isinstance(event.get("index"), int):
            index = event["index"]
            if index in partial_json and index in blocks:
                try:
                    parsed = json.loads(partial_json[index] or "{}")
                except json.JSONDecodeError as error:
                    raise RuntimeError("Anthropic 接口返回了非法的工具参数") from error
                if not isinstance(parsed, dict):
                    raise RuntimeError("Anthropic 接口返回的工具参数不是对象")
                blocks[index]["input"] = parsed
            continue
        if event_type == "message_delta":
            delta_usage = event.get("usage")
            if isinstance(delta_usage, dict):
                usage.update(delta_usage)
            continue
        if event_type == "message_stop":
            completed = True
    if not completed:
        raise RuntimeError("Anthropic 接口流式响应在完成前中断")
    ordered = [blocks[index] for index in sorted(blocks)]
    result = parse_response({"content": ordered, "usage": usage})
    return result.model_copy(update={"raw_response": {"events": events}})


def _messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            raw_content = message.get("content")
            converted.append({
                "role": "user",
                "content": _content(raw_content) if isinstance(raw_content, list) else str(raw_content or ""),
            })
            continue
        if role == "assistant":
            provider_content = message.get("_provider_content")
            blocks = provider_content.get("content") if isinstance(provider_content, dict) else None
            if isinstance(blocks, list):
                converted.append({"role": "assistant", "content": blocks})
                continue
            content: list[dict[str, Any]] = []
            text = message.get("content")
            if isinstance(text, str) and text:
                content.append({"type": "text", "text": text})
            raw_calls = message.get("tool_calls")
            for call in raw_calls if isinstance(raw_calls, list) else []:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    continue
                function = call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                content.append({
                    "type": "tool_use",
                    "id": call["id"],
                    "name": function["name"],
                    "input": _json_object(function.get("arguments")),
                })
            if content:
                converted.append({"role": "assistant", "content": content})
            continue
        if role == "tool" and isinstance(message.get("tool_call_id"), str):
            raw_content = message.get("content")
            tool_result: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message["tool_call_id"],
                "content": _content(raw_content) if isinstance(raw_content, list) else str(raw_content or ""),
            }
            if message.get("_tool_error") is True:
                tool_result["is_error"] = True
            converted.append({
                "role": "user",
                "content": [tool_result],
            })
    return converted


def _content(value: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for block in content_blocks(value):
        if block["type"] == "text":
            converted.append(block)
            continue
        mime_type, data = image_data_url(block["image_url"]["url"])
        converted.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data},
        })
    return converted


def _tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise RuntimeError("工具声明缺少 function 对象")
    return {
        "name": function["name"],
        "description": str(function.get("description") or ""),
        "input_schema": function.get("parameters", {"type": "object"}),
    }


def _blocks(value: list[Any]) -> tuple[str, str | None, list[ToolCall]]:
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for block in value:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            content.append(block["text"])
        elif block_type == "thinking" and isinstance(block.get("thinking"), str):
            reasoning.append(block["thinking"])
        elif block_type == "tool_use" and isinstance(block.get("id"), str) and isinstance(block.get("name"), str):
            arguments = block.get("input")
            calls.append(ToolCall(
                id=block["id"],
                name=block["name"],
                arguments=arguments if isinstance(arguments, dict) else {},
            ))
    return "".join(content), "".join(reasoning) or None, calls


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}
