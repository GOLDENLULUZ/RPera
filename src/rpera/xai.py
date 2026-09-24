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
        "instructions": system_prompt,
        "input": _input(system_messages_as_user(messages)),
        "stream": stream,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
        "parallel_tool_calls": False,
    }
    declarations = [_tool(tool) for tool in tools or []]
    if declarations:
        payload["tools"] = declarations
    return payload


def parse_response(raw: dict[str, Any]) -> ModelResult:
    error = raw.get("error")
    if isinstance(error, dict):
        raise RuntimeError(f"xAI 接口返回错误：{json.dumps(error, ensure_ascii=False)}")
    if raw.get("status") == "incomplete":
        detail = raw.get("incomplete_details")
        raise RuntimeError(f"xAI Responses 生成未完成：{json.dumps(detail, ensure_ascii=False)}")
    output = raw.get("output")
    if not isinstance(output, list):
        raise RuntimeError("xAI Responses 响应缺少 output 数组")
    refusal = _refusal(output)
    if refusal:
        raise RuntimeError(f"xAI 模型拒绝生成：{refusal}")
    content, reasoning, calls = _output(output)
    usage = raw.get("usage")
    return ModelResult(
        content=content,
        reasoning=reasoning,
        usage=usage if isinstance(usage, dict) else {},
        raw_response=raw,
        tool_calls=calls,
        provider_content={"output": output},
    )


def parse_stream(events: list[dict[str, Any]]) -> ModelResult:
    for event in events:
        event_type = event.get("type")
        if event_type in {"error", "response.failed", "response.incomplete"}:
            detail = event.get("error") or event.get("response") or event
            raise RuntimeError(f"xAI 接口返回错误：{json.dumps(detail, ensure_ascii=False)}")
        if event_type == "response.completed" and isinstance(event.get("response"), dict):
            result = parse_response(event["response"])
            return result.model_copy(update={"raw_response": {"events": events}})
    raise RuntimeError("xAI Responses 流式响应在完成前中断")


def _input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            output = provider_content.get("output") if isinstance(provider_content, dict) else None
            if isinstance(output, list):
                converted.extend(item for item in output if isinstance(item, dict))
                continue
            text = message.get("content")
            if isinstance(text, str) and text:
                converted.append({"role": "assistant", "content": text})
            raw_calls = message.get("tool_calls")
            for call in raw_calls if isinstance(raw_calls, list) else []:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    continue
                function = call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                arguments = function.get("arguments", "{}")
                converted.append({
                    "type": "function_call",
                    "call_id": call["id"],
                    "name": function["name"],
                    "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False),
                })
            continue
        if role == "tool" and isinstance(message.get("tool_call_id"), str):
            raw_content = message.get("content")
            blocks = content_blocks(raw_content) if isinstance(raw_content, list) else []
            text = "".join(block["text"] for block in blocks if block["type"] == "text")
            converted.append({
                "type": "function_call_output",
                "call_id": message["tool_call_id"],
                "output": text if blocks else str(raw_content or ""),
            })
            images = [block for block in blocks if block["type"] == "image_url"]
            if images:
                converted.append({"role": "user", "content": _content(images)})
    return converted


def _content(value: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for block in content_blocks(value):
        if block["type"] == "text":
            converted.append({"type": "input_text", "text": block["text"]})
            continue
        url = block["image_url"]["url"]
        image_data_url(url)
        converted.append({"type": "input_image", "image_url": url})
    return converted


def _tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise RuntimeError("工具声明缺少 function 对象")
    return {
        "type": "function",
        "name": function["name"],
        "description": str(function.get("description") or ""),
        "parameters": function.get("parameters", {"type": "object"}),
    }


def _output(value: list[Any]) -> tuple[str, str | None, list[ToolCall]]:
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            parts = item.get("content")
            for part in parts if isinstance(parts, list) else []:
                if isinstance(part, dict) and part.get("type") in {"output_text", "text"} and isinstance(part.get("text"), str):
                    content.append(part["text"])
        elif item_type == "reasoning":
            summary = item.get("summary")
            for part in summary if isinstance(summary, list) else []:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    reasoning.append(part["text"])
        elif item_type == "function_call" and isinstance(item.get("call_id"), str) and isinstance(item.get("name"), str):
            arguments = item.get("arguments", "{}")
            calls.append(ToolCall(
                id=item["call_id"],
                name=item["name"],
                arguments=arguments if isinstance(arguments, (str, dict)) else "{}",
            ))
    return "".join(content), "".join(reasoning) or None, calls


def _refusal(value: list[Any]) -> str:
    refusals: list[str] = []
    for item in value:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        parts = item.get("content")
        for part in parts if isinstance(parts, list) else []:
            if not isinstance(part, dict) or part.get("type") != "refusal":
                continue
            text = part.get("refusal") or part.get("text")
            if isinstance(text, str):
                refusals.append(text)
    return "".join(refusals)
