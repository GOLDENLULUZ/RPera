from __future__ import annotations

import json
from typing import Any

from .message_compat import content_blocks, image_data_url
from .models import ToolCall


def response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "user":
            raw_content = message.get("content")
            converted.append({
                "role": "user",
                "content": response_content(raw_content) if isinstance(raw_content, list) else str(raw_content or ""),
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
                converted.append({"role": "user", "content": response_content(images)})
    return converted


def response_content(value: Any) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for block in content_blocks(value):
        if block["type"] == "text":
            converted.append({"type": "input_text", "text": block["text"]})
            continue
        url = block["image_url"]["url"]
        image_data_url(url)
        converted.append({"type": "input_image", "image_url": url})
    return converted


def response_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        raise RuntimeError("工具声明缺少 function 对象")
    return {
        "type": "function",
        "name": function["name"],
        "description": str(function.get("description") or ""),
        "parameters": function.get("parameters", {"type": "object"}),
    }


def response_output(value: list[Any]) -> tuple[str, str | None, list[ToolCall]]:
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


def response_refusal(value: list[Any]) -> str:
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
