from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import pytest

from rpera.runtime import AgentRunner


@pytest.fixture(autouse=True)
def scripted_model_projection(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep scripted-model tests focused on their existing assertions.

    Context transfer tests exercise the actual provider-visible tool messages.
    """
    if request.node.path.name == "test_context_transfer.py":
        return

    install_scripted_model_projection(monkeypatch)


def install_scripted_model_projection(monkeypatch: pytest.MonkeyPatch) -> None:

    original_messages = AgentRunner._provider_messages
    def is_initial_call(value: str) -> bool:
        if not value.startswith("call-"):
            return False
        try:
            UUID(value[5:])
        except ValueError:
            return False
        return True

    def model_messages(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        messages = original_messages(*args, **kwargs)
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.get("role") == "assistant" and any(
                call.get("function", {}).get("name") in {"entity_read", "report_read"}
                and is_initial_call(str(call.get("id", "")))
                for call in message.get("tool_calls", [])
            ):
                continue
            if message.get("role") == "tool" and is_initial_call(str(message.get("tool_call_id", ""))):
                output = json.loads(message["content"])
                user = next((item for item in reversed(converted) if item["role"] == "user"), None)
                if user is not None:
                    initial = json.loads(user["content"])
                    if output.get("documents"):
                        if output["target_agent"] == "world_researcher":
                            initial["required_world_entities"] = output["documents"]
                            initial["required_world_entities_instruction"] = "这些是对故事长期必要的信息，因此直接提供给你。"
                        else:
                            initial["related_entity_documents"] = output["documents"]
                    if output.get("report_documents"):
                        initial["report_documents"] = output["report_documents"]
                        initial["report_documents_instruction"] = "以下 report_documents 是 Runtime 验真的完整报告原文，请结合当前任务直接使用，不要要求主代理重新转述。"
                    user["content"] = json.dumps(initial, ensure_ascii=False)
                continue
            converted.append(message)
        return converted

    monkeypatch.setattr(AgentRunner, "_provider_messages", staticmethod(model_messages))
