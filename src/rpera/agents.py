from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from .models import CharacterSnapshotPath, EntitySnapshotPath, LocationSnapshotPath, StylePath


RECOMMENDED_SEQUENCE_PLACEHOLDER = "{recommended_agent_sequence}"


class AgentTask(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    task: str = Field(min_length=1, max_length=20_000)
    reason: str = Field(default="", max_length=4_000)


class ReportRef(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=100)
    id: str = Field(min_length=1, max_length=100)


class TaskToolInput(AgentTask):
    agent: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    task_id: str | None = Field(default=None, description="传入已有 task_id 继续同一子代理会话；省略则创建新会话")
    related_entities: list[EntitySnapshotPath] = Field(default_factory=list, max_length=20)
    style_paths: list[StylePath] = Field(default_factory=list, max_length=5)
    report_refs: list[ReportRef] = Field(
        default_factory=list,
        description="原样传入已完成 task 返回的报告 name 和 id；Runtime 会向目标 Agent 注入完整报告",
    )

    @field_validator("related_entities")
    @classmethod
    def unique_related_entities(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("related_entities 不能包含重复路径")
        return value

    @field_validator("style_paths")
    @classmethod
    def unique_style_paths(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("style_paths 不能包含重复路径")
        return value

    @field_validator("report_refs")
    @classmethod
    def unique_report_refs(cls, value: list[ReportRef]) -> list[ReportRef]:
        ids = [item.id for item in value]
        if len(ids) != len(set(ids)):
            raise ValueError("report_refs 不能包含重复报告 id")
        return value

    @model_validator(mode="after")
    def validate_entity_transfer(self) -> TaskToolInput:
        if self.related_entities and self.agent not in {"character_designer", "location_designer", "EroticOrNot", "role_player", "style_planner", "narrator"}:
            raise ValueError("related_entities 只能传递给 character_designer、location_designer、EroticOrNot、role_player、style_planner 或 narrator")
        if self.related_entities and self.task_id is not None:
            raise ValueError("继续子代理会话时不能重新传递 related_entities")
        if self.style_paths and self.agent != "narrator":
            raise ValueError("style_paths 只能传递给 narrator")
        if self.style_paths and self.task_id is not None:
            raise ValueError("继续 narrator 会话时不能重新传递 style_paths")
        return self


class WorldResearchReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    report: str = Field(min_length=1, max_length=20_000)
    related_entities: list[EntitySnapshotPath] = Field(default_factory=list, max_length=20)

    @field_validator("related_entities")
    @classmethod
    def unique_related_entities(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("related_entities 不能包含重复路径")
        return value


class StylePlanningReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    report: str = Field(min_length=1, max_length=20_000)
    style_paths: list[StylePath] = Field(default_factory=list, max_length=5)

    @field_validator("style_paths")
    @classmethod
    def unique_style_paths(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("style_paths 不能包含重复路径")
        return value


class CharacterChangeReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    report: str = Field(min_length=1, max_length=4_000)
    related_entities: list[CharacterSnapshotPath] = Field(default_factory=list, max_length=1)


class LocationChangeReport(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    report: str = Field(min_length=1, max_length=4_000)
    related_entities: list[LocationSnapshotPath] = Field(default_factory=list, max_length=1)


class ComplianceReviewResult(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    approved: StrictBool
    reason: str = Field(min_length=1, max_length=20_000)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    label: str
    prompt_file: str
    task_description_file: str | None
    input_model: type[BaseModel]
    allowed_capabilities: frozenset[str]
    template_order: int
    is_main: bool = False
    can_disable: bool = True
    fixed_response_file: str | None = None
    produces_reports: bool = False


class AgentRegistry:
    def __init__(self, specs: tuple[AgentSpec, ...]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs) or sum(spec.is_main for spec in specs) != 1:
            raise ValueError("Agent 注册表必须包含唯一名称和一个主 Agent")
        if any(spec.is_main and spec.can_disable for spec in specs):
            raise ValueError("主 Agent 不能被禁用")
        if any(spec.is_main != (spec.task_description_file is None) for spec in specs):
            raise ValueError("主 Agent 不能声明 task 描述文件，子 Agent 必须声明 task 描述文件")
        if any(spec.is_main and spec.produces_reports for spec in specs):
            raise ValueError("主 Agent 不能生成子任务报告")

    def get(self, name: str) -> AgentSpec:
        return self._specs[name]

    def all(self) -> tuple[AgentSpec, ...]:
        return tuple(self._specs.values())

    def main(self) -> AgentSpec:
        return next(spec for spec in self._specs.values() if spec.is_main)

    def children(self) -> tuple[AgentSpec, ...]:
        return tuple(spec for spec in self._specs.values() if not spec.is_main)

    def disableable_children(self) -> tuple[AgentSpec, ...]:
        return tuple(spec for spec in self.children() if spec.can_disable)

    def report_producing_children(self) -> tuple[AgentSpec, ...]:
        return tuple(spec for spec in self.children() if spec.produces_reports)


AGENTS = AgentRegistry((
    AgentSpec("coordinator", "主代理 / 总调度师", "coordinator.md", None, AgentTask, frozenset({"file_read", "narrative_publish"}), 0, True, False),
    AgentSpec("story_summarizer", "故事总结 Agent", "story_summarizer.md", "task_descriptions/story_summarizer.md", AgentTask, frozenset({"story_summary_read", "story_summary_edit", "story_history_read"}), template_order=1, produces_reports=True),
    AgentSpec("world_researcher", "世界检索 Agent", "world_researcher.md", "task_descriptions/world_researcher.md", AgentTask, frozenset({"entity_search", "entity_read", "research_report"}), template_order=10, produces_reports=True),
    AgentSpec("character_designer", "角色设计 Agent", "character_designer.md", "task_descriptions/character_designer.md", AgentTask, frozenset({"entity_search", "entity_read", "image_read", "character_portrait_generate", "character_create", "character_edit", "character_rename", "character_report"}), template_order=11, produces_reports=True),
    AgentSpec("location_designer", "地点设计 Agent", "location_designer.md", "task_descriptions/location_designer.md", AgentTask, frozenset({"entity_search", "entity_read", "location_create", "location_edit", "location_rename", "location_report"}), template_order=12, produces_reports=True),
    AgentSpec("compliance_reviewer", "合规性审核 Agent", "compliance_reviewer.md", "task_descriptions/compliance_reviewer.md", AgentTask, frozenset(), template_order=13, fixed_response_file="compliance_reviewer.json", produces_reports=True),
    AgentSpec("EroticOrNot", "够色了吗 Agent", "EroticOrNot.md", "task_descriptions/EroticOrNot.md", AgentTask, frozenset(), template_order=15, produces_reports=True),
    AgentSpec("role_player", "角色扮演 Agent", "role_player.md", "task_descriptions/role_player.md", AgentTask, frozenset(), template_order=20, produces_reports=True),
    AgentSpec("style_planner", "文风规划 Agent", "style_planner.md", "task_descriptions/style_planner.md", AgentTask, frozenset({"style_read", "style_report"}), template_order=30, produces_reports=True),
    AgentSpec("narrator", "叙事 Agent", "narrator.md", "task_descriptions/narrator.md", AgentTask, frozenset({"file_read", "file_write", "file_edit"}), can_disable=False, template_order=98, produces_reports=True),
    AgentSpec("consistency_checker", "一致性检查 Agent", "consistency_checker.md", "task_descriptions/consistency_checker.md", AgentTask, frozenset({"file_read"}), template_order=99, produces_reports=True),
))


@dataclass(frozen=True)
class AgentDescriptor:
    name: str
    label: str
    is_main: bool = False
    can_disable: bool = True
    uses_fixed_response: bool = False
    produces_reports: bool = False


AGENT_CATALOG = tuple(
    AgentDescriptor(spec.name, spec.label, spec.is_main, spec.can_disable, spec.fixed_response_file is not None, spec.produces_reports)
    for spec in (AGENTS.main(), *sorted(AGENTS.children(), key=lambda item: item.template_order))
)


def agent_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in AGENTS.all())


def coordinator_prompt(template: str, enabled_children: tuple[AgentSpec, ...]) -> str:
    ordered = sorted(enabled_children, key=lambda spec: spec.template_order)
    sequence = " -> ".join(spec.name for spec in ordered)
    recommendation = (
        f"本回合可参考以下专业 Agent 委派序列：{sequence}。"
        "这只是推荐顺序；应根据实际任务自主跳过不需要的 Agent，也可以调整调用顺序。"
    )
    return template.replace(RECOMMENDED_SEQUENCE_PLACEHOLDER, recommendation)


def tool_schema(model: type[BaseModel]) -> dict[str, Any]:
    return model.model_json_schema()
