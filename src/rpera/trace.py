from __future__ import annotations

from typing import Any

from .models import RuntimeEvent


TURN_EVENTS = frozenset({"turn.started", "turn.resumed", "turn.completed", "turn.failed", "turn.interrupted"})
TOOL_EVENTS = frozenset({"tool.started", "tool.completed", "tool.failed"})
TRACE_EVENT_TYPES = (*TURN_EVENTS, *TOOL_EVENTS, "task.created", "model.response", "agent.fixed_response")


def project_trace_event(event: RuntimeEvent) -> RuntimeEvent | None:
    payload = event.payload
    if event.type in TOOL_EVENTS:
        if payload.get("tool") == "task" and event.type != "tool.failed":
            return None
        projected = _only(payload, "agent", "tool", "input", "result")
    elif event.type == "model.response":
        content = payload.get("content")
        if payload.get("tool_calls") or not isinstance(content, str) or not content.strip():
            return None
        projected = _only(payload, "agent", "content", "duration_ms")
    elif event.type == "agent.fixed_response":
        projected = _only(payload, "agent", "result")
    elif event.type == "task.created":
        projected = _only(
            payload,
            "initiator_agent",
            "target_agent",
            "agent",
            "task",
            "reason",
            "instruction_blocked",
            "related_entities",
            "reports",
            "styles",
        )
    elif event.type in TURN_EVENTS:
        projected = _only(payload, "player_input", "duration_ms", "message", "error_type")
    else:
        return None
    return event.model_copy(update={"payload": projected})


def _only(payload: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}
