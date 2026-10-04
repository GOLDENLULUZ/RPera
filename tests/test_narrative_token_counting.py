from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpera.agents import AGENTS
from rpera.app import create_app
from rpera.capabilities import CapabilityContext, CapabilityExecutor
from rpera.draft_files import DraftFileStore
from rpera.models import NarrativeTokenCountingSettings, SaveCreate
from rpera.runtime import AgentRunner
from rpera.token_counting import count_narrative_tokens
from tests.helpers import make_data_dir


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("The rain stopped, and the harbor lights returned.", 10),
        ("雨停了，港口的灯火重新亮起。", 13),
        ("雨が止み、港の灯りが戻った。", 13),
        ("<|endoftext|>", 7),
    ],
)
def test_o200k_base_count_is_stable_for_supported_languages(content: str, expected: int) -> None:
    assert count_narrative_tokens(content) == expected


def test_token_counting_settings_api_defaults_and_validates_strict_limit(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    with TestClient(app) as client:
        settings = client.get("/api/capability-settings").json()
        assert settings["narrative_token_counting"] == {"enabled": False, "max_tokens": 4_000}

        settings["narrative_token_counting"] = {"enabled": True, "max_tokens": 2_500}
        assert client.put("/api/capability-settings", json=settings).json()["narrative_token_counting"] == {
            "enabled": True,
            "max_tokens": 2_500,
        }

        settings["narrative_token_counting"]["max_tokens"] = True
        assert client.put("/api/capability-settings", json=settings).status_code == 422


@pytest.mark.asyncio
async def test_token_count_capability_reads_authoritative_draft_and_reports_excess(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    save = app.state.saves.create_save(SaveCreate(world_name="雾港", name="Token 检查"))
    turn = app.state.saves.create_turn(save.id, "继续", 4)
    content = "雨停了，港口的灯火重新亮起。"
    DraftFileStore(app.state.saves.draft_dir(save.id, turn.id)).write("narrative.md", content)

    async def emit(_event_type: str, _payload: dict[str, Any]) -> None:
        pass

    executor = CapabilityExecutor(app.state.runtime.capability_registry)
    result = await executor.execute(
        "narrative_token_count",
        {},
        CapabilityContext(
            save.id,
            turn.id,
            "narrator",
            emit,
            token_counting_settings=NarrativeTokenCountingSettings(enabled=True, max_tokens=10),
        ),
        {"narrative_token_count"},
    )

    assert result == {
        "path": "narrative.md",
        "encoding": "o200k_base",
        "token_count": 13,
        "max_tokens": 10,
        "over_limit": True,
        "excess_tokens": 3,
    }
    assert DraftFileStore(app.state.saves.draft_dir(save.id, turn.id)).read("narrative.md") == content


def test_token_count_tool_is_exposed_only_to_two_agents_when_enabled(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path))
    runner: AgentRunner = app.state.runtime.runner
    descriptions = app.state.runtime.prompts.load(
        frozenset(spec.name for spec in AGENTS.all())
    ).capability_descriptions

    for agent in AGENTS.all():
        disabled = runner._tools_for(agent, descriptions, enabled_agents=frozenset())
        enabled = runner._tools_for(
            agent,
            descriptions,
            enabled_agents=frozenset(),
            enabled_configurable_capabilities=frozenset({"narrative_token_count"}),
            narrative_token_limit=1234,
        )
        disabled_names = {tool["function"]["name"] for tool in disabled}
        enabled_tools = {tool["function"]["name"]: tool for tool in enabled}
        assert "narrative_token_count" not in disabled_names
        assert ("narrative_token_count" in enabled_tools) is (agent.name in {"narrator", "consistency_checker"})
        if "narrative_token_count" in enabled_tools:
            description = enabled_tools["narrative_token_count"]["function"]["description"]
            assert "1234 token" in description
            assert "{max_tokens}" not in description
