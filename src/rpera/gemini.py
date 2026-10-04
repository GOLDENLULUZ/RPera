from __future__ import annotations

import json
import uuid
from typing import Any

from .message_compat import content_blocks, image_data_url, system_messages_as_user
from .gemini_thinking import validate_gemini_thinking_level
from .model_errors import CONTENT_BLOCK_REASONS, ModelFallbackError, is_content_block_error
from .models import ModelResult, ToolCall
from .thinking import ThinkingLevel


SKIP_THOUGHT_SIGNATURE = "skip_thought_signature_validator"


def request_payload(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    model: str = "",
    thinking_level: ThinkingLevel | None = None,
) -> dict[str, Any]:
    validate_gemini_thinking_level("google_gemini", model, thinking_level)
    payload: dict[str, Any] = {
        "contents": _contents(system_messages_as_user(messages)),
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {
            "temperature": temperature,
            "topP": top_p,
            "maxOutputTokens": max_tokens,
        },
    }
    if thinking_level is not None:
        payload["generationConfig"]["thinkingConfig"] = {"thinkingLevel": thinking_level.upper()}
    declarations = [_function_declaration(tool) for tool in tools or []]
    if declarations:
        payload["tools"] = [{"functionDeclarations": declarations}]
    return payload


def parse_response(raw: dict[str, Any]) -> ModelResult:
    provider_content = _candidate(raw)
    content, reasoning, calls = _parts(provider_content.get("parts"))
    usage = raw.get("usageMetadata")
    return ModelResult(
        content=content,
        reasoning=reasoning,
        usage=usage if isinstance(usage, dict) else {},
        raw_response=raw,
        tool_calls=calls,
        provider_content=provider_content,
    )


def parse_stream(chunks: list[dict[str, Any]]) -> ModelResult:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    calls: list[ToolCall] = []
    provider_parts: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    saw_candidate = False
    for chunk in chunks:
        _raise_for_error(chunk)
        chunk_usage = chunk.get("usageMetadata")
        if isinstance(chunk_usage, dict):
            usage = chunk_usage
        candidates = chunk.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            continue
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            continue
        _raise_for_finish(candidate)
        provider_content = candidate.get("content")
        if not isinstance(provider_content, dict):
            continue
        parts = provider_content.get("parts")
        if not isinstance(parts, list):
            continue
        saw_candidate = True
        valid_parts = [part for part in parts if isinstance(part, dict)]
        provider_parts.extend(valid_parts)
        text, reasoning, chunk_calls = _parts(valid_parts)
        content_parts.append(text)
        if reasoning:
            reasoning_parts.append(reasoning)
        calls.extend(chunk_calls)
    if not saw_candidate:
        feedback = next((chunk.get("promptFeedback") for chunk in chunks if isinstance(chunk.get("promptFeedback"), dict)), None)
        if isinstance(feedback, dict) and feedback.get("blockReason") in CONTENT_BLOCK_REASONS:
            raise ModelFallbackError("content_blocked", f"Gemini 响应没有候选内容{_detail(feedback)}")
        raise RuntimeError(f"Gemini 响应没有候选内容{_detail(feedback)}")
    return ModelResult(
        content="".join(content_parts),
        reasoning="".join(reasoning_parts) or None,
        usage=usage,
        raw_response={"chunks": chunks},
        tool_calls=calls,
        provider_content={"role": "model", "parts": provider_parts},
    )


def _contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    calls: dict[str, tuple[str, str | None]] = {}
    for message in messages:
        role = message.get("role")
        if role == "user":
            converted.append({"role": "user", "parts": _content_parts(message.get("content"))})
            continue
        if role == "assistant":
            raw_tool_calls = message.get("tool_calls")
            tool_calls: list[Any] = raw_tool_calls if isinstance(raw_tool_calls, list) else []
            provider_content = message.get("_provider_content")
            provider_parts = provider_content.get("parts") if isinstance(provider_content, dict) else None
            if isinstance(provider_content, dict) and isinstance(provider_parts, list):
                converted.append(provider_content)
                function_parts: list[dict[str, Any]] = [part["functionCall"] for part in provider_parts if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)]
                for index, call in enumerate(tool_calls):
                    if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                        continue
                    function = call.get("function")
                    if not isinstance(function, dict):
                        continue
                    provider_call = function_parts[index] if index < len(function_parts) else {}
                    calls[call["id"]] = (str(function.get("name") or ""), provider_call.get("id") if isinstance(provider_call.get("id"), str) else None)
                continue
            parts: list[dict[str, Any]] = []
            if isinstance(message.get("content"), str) and message["content"]:
                parts.append({"text": message["content"]})
            for call in tool_calls:
                if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                    continue
                function = call.get("function")
                if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                    continue
                arguments = _json_object(function.get("arguments"))
                parts.append({
                    "functionCall": {"name": function["name"], "args": arguments, "id": call["id"]},
                    "thoughtSignature": SKIP_THOUGHT_SIGNATURE,
                })
                calls[call["id"]] = (function["name"], call["id"])
            if parts:
                converted.append({"role": "model", "parts": parts})
            continue
        if role == "tool" and isinstance(message.get("tool_call_id"), str):
            name, provider_id = calls.get(message["tool_call_id"], ("", None))
            raw_content = message.get("content")
            blocks = content_blocks(raw_content) if isinstance(raw_content, list) else []
            text = "".join(block["text"] for block in blocks if block["type"] == "text")
            response = _json_object(text if blocks else raw_content)
            function_response: dict[str, Any] = {"name": name, "response": response}
            if provider_id:
                function_response["id"] = provider_id
            parts = [{"functionResponse": function_response}]
            parts.extend(_content_parts([block for block in blocks if block["type"] == "image_url"]))
            converted.append({"role": "user", "parts": parts})
    return converted


def _content_parts(value: Any) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for block in content_blocks(value):
        if block["type"] == "text":
            parts.append({"text": block["text"]})
            continue
        mime_type, data = image_data_url(block["image_url"]["url"])
        parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
    return parts


def _function_declaration(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function")
    if not isinstance(function, dict):
        raise RuntimeError("工具声明缺少 function 对象")
    declaration = {
        "name": function.get("name"),
        "description": function.get("description", ""),
        "parametersJsonSchema": function.get("parameters", {"type": "object"}),
    }
    return declaration


def _candidate(raw: dict[str, Any]) -> dict[str, Any]:
    _raise_for_error(raw)
    candidates = raw.get("candidates")
    if not isinstance(candidates, list) or not candidates or not isinstance(candidates[0], dict):
        feedback = raw.get("promptFeedback")
        if isinstance(feedback, dict) and feedback.get("blockReason") in CONTENT_BLOCK_REASONS:
            raise ModelFallbackError("content_blocked", f"Gemini 响应没有候选内容{_detail(feedback)}")
        raise RuntimeError(f"Gemini 响应没有候选内容{_detail(raw.get('promptFeedback'))}")
    candidate = candidates[0]
    _raise_for_finish(candidate)
    provider_content = candidate.get("content")
    parts = provider_content.get("parts") if isinstance(provider_content, dict) else None
    if candidate.get("finishReason") == "MAX_TOKENS" and (not isinstance(parts, list) or not parts):
        raise RuntimeError(f"Gemini 达到输出 token 上限且未返回内容，请提高最大输出 Tokens{_detail(raw.get('usageMetadata'))}")
    if not isinstance(provider_content, dict) or not isinstance(parts, list):
        raise RuntimeError("Gemini 响应缺少 candidates[0].content.parts")
    return provider_content


def _parts(value: Any) -> tuple[str, str | None, list[ToolCall]]:
    if not isinstance(value, list):
        return "", None, []
    content: list[str] = []
    reasoning: list[str] = []
    calls: list[ToolCall] = []
    for part in value:
        if not isinstance(part, dict):
            continue
        text = part.get("text")
        if isinstance(text, str):
            (reasoning if part.get("thought") is True else content).append(text)
        function = part.get("functionCall")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            continue
        raw_call_id = function.get("id")
        call_id: str = raw_call_id if isinstance(raw_call_id, str) else f"gemini-{uuid.uuid4()}"
        name = str(function["name"])
        arguments = function.get("args")
        calls.append(ToolCall(id=call_id, name=name, arguments=arguments if isinstance(arguments, dict) else {}))
    return "".join(content), "".join(reasoning) or None, calls


def _raise_for_finish(candidate: dict[str, Any]) -> None:
    reason = candidate.get("finishReason")
    if reason in {None, "", "STOP", "MAX_TOKENS"}:
        return
    message = candidate.get("finishMessage")
    suffix = f"：{message}" if isinstance(message, str) and message else ""
    if reason in CONTENT_BLOCK_REASONS:
        raise ModelFallbackError("content_blocked", f"Gemini 生成已停止（{reason}）{suffix}")
    raise RuntimeError(f"Gemini 生成已停止（{reason}）{suffix}")


def _raise_for_error(raw: dict[str, Any]) -> None:
    error = raw.get("error")
    if isinstance(error, dict):
        if is_content_block_error(error):
            raise ModelFallbackError("content_blocked", f"Gemini 接口返回错误：{json.dumps(error, ensure_ascii=False)}")
        raise RuntimeError(f"Gemini 接口返回错误：{json.dumps(error, ensure_ascii=False)}")


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"result": value}
        if isinstance(parsed, dict):
            return parsed
        return {"result": parsed}
    return {"result": value}


def _detail(value: Any) -> str:
    if not isinstance(value, dict) or not value:
        return ""
    return f"：{json.dumps(value, ensure_ascii=False)}"
