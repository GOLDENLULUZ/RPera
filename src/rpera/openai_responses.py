from __future__ import annotations

import json
from typing import Any

from .message_compat import system_messages_as_user
from .model_errors import ModelFallbackError
from .models import ModelResult
from .responses_content import response_input, response_output, response_refusal, response_tool


def request_payload(
    system_prompt: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    *,
    model: str,
    max_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "instructions": system_prompt,
        "input": response_input(system_messages_as_user(messages)),
        "stream": stream,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": max_tokens,
        "parallel_tool_calls": False,
    }
    declarations = [response_tool(tool) for tool in tools or []]
    if declarations:
        payload["tools"] = declarations
    return payload


def parse_response(raw: dict[str, Any]) -> ModelResult:
    error = raw.get("error")
    if error is not None:
        raise RuntimeError(f"OpenAI Responses 接口返回错误：{json.dumps(error, ensure_ascii=False)}")
    status = raw.get("status")
    if status is not None and status != "completed":
        detail = raw.get("incomplete_details") or status
        raise RuntimeError(f"OpenAI Responses 生成未完成：{json.dumps(detail, ensure_ascii=False)}")
    output = raw.get("output")
    if not isinstance(output, list):
        raise RuntimeError("OpenAI Responses 响应缺少 output 数组")
    refusal = response_refusal(output)
    if refusal:
        raise ModelFallbackError("content_blocked", f"OpenAI 模型拒绝生成：{refusal}")
    content, reasoning, calls = response_output(output)
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
            raise RuntimeError(f"OpenAI Responses 接口返回错误：{json.dumps(detail, ensure_ascii=False)}")
        if event_type == "response.completed" and isinstance(event.get("response"), dict):
            result = parse_response(event["response"])
            return result.model_copy(update={"raw_response": {"events": events}})
    raise RuntimeError("OpenAI Responses 流式响应在完成前中断")
