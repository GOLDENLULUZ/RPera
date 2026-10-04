from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from rpera.agents import (
    AGENTS,
    AgentRegistry,
    AgentSpec,
    AgentTask,
    CharacterChangeReport,
    ComplianceReviewResult,
    LocationChangeReport,
    ReportRef,
    StylePlanningReport,
    TaskToolInput,
    TextReport,
    WorldResearchReport,
)
from rpera.capabilities import CapabilityRegistry
from rpera.runtime import AgentRunner

from tests.prompt_fixtures import TASK_DESCRIPTIONS


def test_agent_catalog_order_and_capabilities_are_declared_once() -> None:
    assert [agent.name for agent in AGENTS.children()] == [
        "story_summarizer",
        "world_researcher",
        "character_designer",
        "location_designer",
        "compliance_reviewer",
        "EroticOrNot",
        "goal_keeper",
        "role_player",
        "style_planner",
        "narrator",
        "consistency_checker",
    ]
    assert {agent.name for agent in AGENTS.all()} == {
        "coordinator",
        "story_summarizer",
        "world_researcher",
        "character_designer",
        "location_designer",
        "compliance_reviewer",
        "EroticOrNot",
        "goal_keeper",
        "role_player",
        "style_planner",
        "narrator",
        "consistency_checker",
    }
    assert {agent.name for agent in AGENTS.report_producing_children()} == {agent.name for agent in AGENTS.children()} - {"goal_keeper"}
    assert AGENTS.main().produces_reports is False

    assert [agent.name for agent in sorted(AGENTS.children(), key=lambda agent: agent.template_order)] == [
        "compliance_reviewer",
        "story_summarizer",
        "world_researcher",
        "character_designer",
        "location_designer",
        "EroticOrNot",
        "goal_keeper",
        "role_player",
        "style_planner",
        "narrator",
        "consistency_checker",
    ]
    shared_reads = {"entity_read", "report_read"}
    assert AGENTS.get("world_researcher").allowed_capabilities == frozenset(shared_reads | {"entity_search", "research_report"})
    assert AGENTS.get("story_summarizer").allowed_capabilities == frozenset(shared_reads | {"story_summary_read", "story_summary_edit", "story_history_read"})
    assert AGENTS.get("role_player").allowed_capabilities == frozenset(shared_reads | {"role_report"})
    assert AGENTS.get("EroticOrNot").allowed_capabilities == frozenset(shared_reads | {"erotic_report"})
    assert TextReport(report="完整意见").model_dump() == {"report": "完整意见"}
    with pytest.raises(ValidationError):
        TextReport(report=" ")
    with pytest.raises(ValidationError):
        TextReport(report="意见", category="不需要第二个字段")  # type: ignore[call-arg]
    assert AGENTS.get("goal_keeper").allowed_capabilities == frozenset(shared_reads | {"goal_read", "goal_create", "goal_edit"})
    assert AGENTS.get("goal_keeper").produces_reports is False
    assert AGENTS.get("compliance_reviewer").allowed_capabilities == frozenset(shared_reads | {"compliance_report"})
    assert AGENTS.get("consistency_checker").allowed_capabilities == frozenset(shared_reads | {"file_read", "narrative_token_count"})
    assert all(shared_reads <= spec.allowed_capabilities for spec in AGENTS.children())
    assert "image_read" in AGENTS.get("character_designer").allowed_capabilities
    assert all(
        "image_read" not in agent.allowed_capabilities
        for agent in AGENTS.all()
        if agent.name != "character_designer"
    )

    compliance = AGENTS.get("compliance_reviewer")
    assert compliance.can_disable is True
    assert compliance.fixed_response_file == "compliance_reviewer.json"
    assert ComplianceReviewResult(approved=True, reason="通过").approved is True
    with pytest.raises(ValidationError):
        ComplianceReviewResult(approved="true", reason="通过")  # type: ignore[arg-type]


def test_task_tool_discovers_children_from_runner_registry() -> None:
    registry = AgentRegistry((
        AgentSpec("main", "Main", "prompt", None, AgentTask, frozenset(), 0, True, False),
        AgentSpec("specialist", "Specialist", "prompt", "task_descriptions/specialist.md", AgentTask, frozenset(), 10),
    ))
    runner = AgentRunner(cast(Any, None), registry, CapabilityRegistry([]))

    task = runner._tools_for(
        registry.main(),
        {},
        task_descriptions={"specialist": "Use when specialist work is needed"},
    )[0]["function"]

    assert task["parameters"]["properties"]["agent"]["enum"] == ["specialist"]
    assert "specialist（Specialist）：Use when specialist work is needed" in task["description"]
    assert "report_refs" in task["description"]
    assert "main" not in task["description"]


def test_task_tool_only_lists_enabled_children() -> None:
    runner = AgentRunner(cast(Any, None), AGENTS, CapabilityRegistry([]))

    task = runner._tools_for(
        AGENTS.main(),
        {},
        frozenset({"coordinator", "narrator"}),
        TASK_DESCRIPTIONS,
    )[0]["function"]

    assert task["parameters"]["properties"]["agent"]["enum"] == ["narrator"]
    assert "narrator" in task["description"]
    assert "world_researcher" not in task["description"]


def test_related_entity_schemas_reject_ambiguous_transfers() -> None:
    path = "entities/character/伊蕾/ENTITY.md"
    assert WorldResearchReport(report="事实", related_entities=[path]).related_entities == [path]
    assert TaskToolInput(agent="role_player", task="扮演伊蕾", related_entities=[path]).related_entities == [path]
    assert TaskToolInput(agent="EroticOrNot", task="给出意见", related_entities=[path]).related_entities == [path]
    assert TaskToolInput(agent="narrator", task="写故事", related_entities=[path]).related_entities == [path]
    assert TaskToolInput(agent="style_planner", task="选择文风", related_entities=[path]).related_entities == [path]
    assert TaskToolInput(agent="character_designer", task="维护角色", related_entities=[path]).related_entities == [path]
    assert TaskToolInput(agent="location_designer", task="维护地点", related_entities=[path]).related_entities == [path]
    assert TaskToolInput(agent="goal_keeper", task="维护目标", related_entities=[path]).related_entities == [path]
    assert CharacterChangeReport(report="无需长期变更").related_entities == []
    assert CharacterChangeReport(report="完成更名", related_entities=[path]).related_entities == [path]
    location_path = "entities/location/暮潮旅店/ENTITY.md"
    assert LocationChangeReport(report="无需长期变更").related_entities == []
    assert LocationChangeReport(report="完成更名", related_entities=[location_path]).related_entities == [location_path]
    style = "styles/冷峻叙事.md"
    assert StylePlanningReport(report="适合克制表达", style_paths=[style]).style_paths == [style]
    assert TaskToolInput(agent="narrator", task="写故事", style_paths=[style]).style_paths == [style]

    with pytest.raises(ValidationError, match="只能显式传递给允许接收实体的子代理"):
        TaskToolInput(agent="world_researcher", task="调查", related_entities=[path])
    with pytest.raises(ValidationError, match="不能重新传递"):
        TaskToolInput(agent="narrator", task="修改", task_id="existing", related_entities=[path])
    with pytest.raises(ValidationError, match="重复路径"):
        WorldResearchReport(report="事实", related_entities=[path, path])
    with pytest.raises(ValidationError, match="只能传递给 narrator"):
        TaskToolInput(agent="style_planner", task="选择", style_paths=[style])


def test_report_refs_are_unbounded_unique_pairs_available_to_every_child() -> None:
    refs = [ReportRef(name=f"报告 {index}", id=f"report-{index}") for index in range(25)]

    for agent in AGENTS.children():
        task = TaskToolInput(agent=agent.name, task="使用完整报告", task_id="existing", report_refs=refs)
        assert task.report_refs == refs

    with pytest.raises(ValidationError, match="重复报告 id"):
        TaskToolInput(
            agent="narrator",
            task="写故事",
            report_refs=[ReportRef(name="甲", id="same"), ReportRef(name="乙", id="same")],
        )
