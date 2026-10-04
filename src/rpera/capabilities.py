from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agents import CharacterChangeReport, ComplianceReviewResult, LocationChangeReport, ReportRef, StylePlanningReport, TextReport, WorldResearchReport
from .drawing_providers import NovelAIClient, StableDiffusionWebUIClient
from .draft_files import DraftFileStore
from .content_store import ContentStore
from .models import CharacterPortraitGenerationSettings, CharacterSnapshotPath, ContentName, DrawingPreset, EntityAliases, EntitySnapshotPath, LocationSnapshotPath, NarrativeTokenCountingSettings, NetworkSettings, StylePath
from .storage import SaveStore
from .token_counting import ENCODING_NAME, count_narrative_tokens


_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
P = ParamSpec("P")
R = TypeVar("R")


class CapabilityError(RuntimeError):
    pass


class CapabilityPermissionError(CapabilityError):
    pass


class EntitySearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=4_000)


class EntityReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[EntitySnapshotPath] = Field(min_length=1)


class ReportReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report_refs: list[ReportRef] = Field(min_length=1)


class StyleReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths: list[StylePath] = Field(min_length=1, max_length=5)


class ImageReadInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    path: str = Field(min_length=1, max_length=255)


class CharacterPortraitGenerateInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    description: str = Field(min_length=1, max_length=8_000)


class CharacterCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName
    description: str = Field(max_length=4_000)
    aliases: EntityAliases
    profile: dict[str, Any]


class CharacterEditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: CharacterSnapshotPath
    old_text: str = Field(min_length=1, max_length=170_000)
    new_text: str = Field(max_length=170_000)


class CharacterRenameInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: CharacterSnapshotPath
    new_name: ContentName


class LocationCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName
    description: str = Field(max_length=4_000)
    aliases: EntityAliases
    profile: dict[str, Any]


class LocationEditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: LocationSnapshotPath
    old_text: str = Field(min_length=1, max_length=170_000)
    new_text: str = Field(max_length=170_000)


class LocationRenameInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: LocationSnapshotPath
    new_name: ContentName


class GoalReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoalCreateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", max_length=4_000)
    profile: dict[str, Any]


class GoalEditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(min_length=1, max_length=170_000)
    new_text: str = Field(max_length=170_000)


class FileReadInput(BaseModel):
    path: str


class FileWriteInput(BaseModel):
    path: str
    content: str = Field(max_length=2_000_000)


class FileEditInput(BaseModel):
    path: str
    old_text: str = Field(max_length=2_000_000)
    new_text: str = Field(max_length=2_000_000)


class StorySummaryReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StorySummaryEditInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    old_text: str = Field(min_length=1, max_length=2_000_000)
    new_text: str = Field(max_length=2_000_000)


class StoryHistoryReadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    after_turn_number: int = Field(default=0, ge=0, strict=True)


class NarrativePublishInput(BaseModel):
    path: str


class NarrativeTokenCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class CapabilityContext:
    save_id: str
    turn_id: str
    agent: str
    emit: Callable[[str, dict[str, Any]], Awaitable[None]]
    session_id: str = ""
    tool_part_id: str = ""
    drawing_preset: DrawingPreset | None = None
    portrait_settings: CharacterPortraitGenerationSettings | None = None
    network_settings: NetworkSettings | None = None
    token_counting_settings: NarrativeTokenCountingSettings | None = None
    report_resolver: Callable[[list[ReportRef]], Awaitable[list[dict[str, Any]]]] | None = None


CapabilityHandler = Callable[[BaseModel, CapabilityContext], Awaitable[dict[str, Any]]]

RETRIEVAL_COUNTER_WARNING = (
    "计数器用于帮助你了解小型世界观中当前已阅读资料的规模，并在资料足够时避免重复阅读。"
    "可检索实体总数不是阅读目标，禁止将100%阅读作为参考指标；只应按当前任务需要获取资料。"
)


@dataclass(frozen=True)
class CapabilitySpec:
    name: str
    input_model: type[BaseModel]
    handler: CapabilityHandler


class CapabilityRegistry:
    def __init__(self, specs: list[CapabilitySpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Capability 名称重复")
        invalid = [spec.name for spec in specs if not _TOOL_NAME.fullmatch(spec.name)]
        if invalid:
            raise ValueError(f"模型 Tool 名称不兼容：{', '.join(invalid)}")

    def get(self, name: str) -> CapabilitySpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise CapabilityError(f"未知 Capability：{name}") from error

    def all(self) -> tuple[CapabilitySpec, ...]:
        return tuple(self._specs.values())


class CapabilityExecutor:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    async def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        context: CapabilityContext,
        allowed_capabilities: set[str],
    ) -> dict[str, Any]:
        spec = self.registry.get(name)
        if name not in allowed_capabilities:
            raise CapabilityPermissionError(f"Agent {context.agent} 无权调用 Capability：{name}")
        try:
            request = spec.input_model.model_validate(arguments)
        except ValidationError as error:
            raise CapabilityError(f"Capability {name} 参数无效：{error}") from error
        return await spec.handler(request, context)


def build_runtime_capability_registry(saves: SaveStore, styles: ContentStore) -> CapabilityRegistry:
    async def report_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, ReportReadInput)
        if context.report_resolver is None:
            raise CapabilityPermissionError("缺少报告读取上下文")
        reports = await context.report_resolver(value.report_refs)
        return {"report_documents": reports, "target_agent": context.agent}

    async def entity_search(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, EntitySearchInput)
        results = await _thread(saves.search_entities, context.save_id, value.query)
        counter = await _retrieval_counter(saves, context)
        payload = {
            "query": value.query,
            "results": results,
            "retrieval_counter": counter,
            "retrieval_counter_warning": RETRIEVAL_COUNTER_WARNING,
        }
        await context.emit("entity.search", payload)
        return payload

    async def entity_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, EntityReadInput)
        if context.agent == "character_designer":
            reader = saves.read_entities_for_character_edit
        elif context.agent == "location_designer":
            reader = saves.read_entities_for_location_edit
        else:
            reader = saves.read_entities
        documents = await _thread(reader, context.save_id, value.paths)
        counter = await _retrieval_counter(saves, context, read_documents=documents)
        payload = {
            "requested_paths": value.paths,
            "documents": documents,
            "retrieval_counter": counter,
            "retrieval_counter_warning": RETRIEVAL_COUNTER_WARNING,
            "target_agent": context.agent,
        }
        await context.emit("entity.read", payload)
        return payload

    async def research_report(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, WorldResearchReport)
        available = set(await _thread(saves.required_entity_paths, context.save_id))
        messages = await _thread(saves.load_session, context.save_id, context.session_id)
        for message in messages:
            for part in message["parts"]:
                if part["type"] != "tool" or part["tool_name"] != "entity_read" or part["state"] != "completed":
                    continue
                try:
                    output = json.loads(part["output"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(output, dict):
                    available.update(_entity_paths(output.get("documents")))
        unavailable = [path for path in value.related_entities if path not in available]
        if unavailable:
            raise CapabilityError(f"报告引用了本次研究未获得的实体：{', '.join(unavailable)}")
        return value.model_dump()

    async def style_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, StyleReadInput)
        messages = await _thread(saves.load_session, context.save_id, context.session_id)
        listed = _initial_style_paths(messages)
        unlisted = [path for path in value.paths if path not in listed]
        if unlisted:
            raise CapabilityError(f"请求了初始清单之外的文风：{', '.join(unlisted)}")
        documents = [document.model_dump() for document in await _thread(styles.read_enabled_styles, value.paths)]
        payload = {"requested_paths": value.paths, "documents": documents}
        await context.emit("style.read", payload)
        return payload

    async def style_report(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, StylePlanningReport)
        available: set[str] = set()
        messages = await _thread(saves.load_session, context.save_id, context.session_id)
        for message in messages:
            for part in message["parts"]:
                if part["type"] != "tool" or part["tool_name"] != "style_read" or part["state"] != "completed":
                    continue
                try:
                    output = json.loads(part["output"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(output, dict):
                    available.update(_style_paths(output.get("documents")))
        unavailable = [path for path in value.style_paths if path not in available]
        if unavailable:
            raise CapabilityError(f"报告引用了本次规划未读取的文风：{', '.join(unavailable)}")
        return value.model_dump()

    async def compliance_report(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        return _require(request, ComplianceReviewResult).model_dump()

    async def text_report(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        return _require(request, TextReport).model_dump()

    async def image_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, ImageReadInput)
        payload = await _thread(saves.read_image_artifact, context.save_id, value.path)
        await context.emit("image.read", {key: item for key, item in payload.items() if key != "data"})
        return payload

    async def character_portrait_generate(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, CharacterPortraitGenerateInput)
        if context.agent != "character_designer":
            raise CapabilityPermissionError("角色立绘生成仅对角色设计 Agent 开放")
        if context.portrait_settings is None or not context.portrait_settings.enabled:
            raise CapabilityError("角色立绘生成能力未启用")
        if context.drawing_preset is None:
            raise CapabilityError("角色立绘生成缺少默认绘画 Preset")
        client = (NovelAIClient if context.drawing_preset.provider == "novelai" else StableDiffusionWebUIClient)(
            context.drawing_preset,
            network_settings=context.network_settings,
        )
        generated = await client.generate(value.description, context.portrait_settings)
        payload = await _thread(saves.write_generated_image_artifact, context.save_id, generated.data)
        payload["seed"] = generated.seed
        await context.emit("character.portrait_generated", payload)
        return payload

    async def character_create(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, CharacterCreateInput)
        character = await _thread(
            saves.create_character,
            context.save_id,
            value.name,
            value.description,
            value.aliases,
            value.profile,
        )
        character = (await _thread(
            saves.read_entities_exact_for_character_edit,
            context.save_id,
            [character["path"]],
        ))[0]
        payload = {"operation": "create", "character": character}
        await context.emit("character.created", payload)
        return payload

    async def character_edit(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, CharacterEditInput)
        available = await _character_paths_for_session(saves, context)
        if value.path not in available:
            raise CapabilityError(f"编辑前必须先取得角色完整正文：{value.path}")
        character = await _thread(
            saves.edit_character,
            context.save_id,
            value.path,
            value.old_text,
            value.new_text,
        )
        character = (await _thread(
            saves.read_entities_exact_for_character_edit,
            context.save_id,
            [character["path"]],
        ))[0]
        payload = {"operation": "edit", "character": character}
        await context.emit("character.edited", payload)
        return payload

    async def character_rename(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, CharacterRenameInput)
        available = await _character_paths_for_session(saves, context)
        if value.path not in available:
            raise CapabilityError(f"更名前必须先取得角色完整正文：{value.path}")
        character = await _thread(saves.rename_character, context.save_id, value.path, value.new_name)
        character = (await _thread(
            saves.read_entities_exact_for_character_edit,
            context.save_id,
            [character["path"]],
        ))[0]
        payload = {"operation": "rename", "previous_path": value.path, "character": character}
        await context.emit("character.renamed", payload)
        return payload

    async def character_report(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, CharacterChangeReport)
        await _thread(saves.read_entities_exact, context.save_id, value.related_entities)
        return value.model_dump()

    async def location_create(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, LocationCreateInput)
        location = await _thread(
            saves.create_location,
            context.save_id,
            value.name,
            value.description,
            value.aliases,
            value.profile,
        )
        location = (await _thread(
            saves.read_entities_exact_for_location_edit,
            context.save_id,
            [location["path"]],
        ))[0]
        payload = {"operation": "create", "location": location}
        await context.emit("location.created", payload)
        return payload

    async def location_edit(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, LocationEditInput)
        available = await _location_paths_for_session(saves, context)
        if value.path not in available:
            raise CapabilityError(f"编辑前必须先取得地点完整正文：{value.path}")
        location = await _thread(
            saves.edit_location,
            context.save_id,
            value.path,
            value.old_text,
            value.new_text,
        )
        location = (await _thread(
            saves.read_entities_exact_for_location_edit,
            context.save_id,
            [location["path"]],
        ))[0]
        payload = {"operation": "edit", "location": location}
        await context.emit("location.edited", payload)
        return payload

    async def location_rename(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, LocationRenameInput)
        available = await _location_paths_for_session(saves, context)
        if value.path not in available:
            raise CapabilityError(f"更名前必须先取得地点完整正文：{value.path}")
        location = await _thread(saves.rename_location, context.save_id, value.path, value.new_name)
        location = (await _thread(
            saves.read_entities_exact_for_location_edit,
            context.save_id,
            [location["path"]],
        ))[0]
        payload = {"operation": "rename", "previous_path": value.path, "location": location}
        await context.emit("location.renamed", payload)
        return payload

    async def location_report(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, LocationChangeReport)
        await _thread(saves.read_entities_exact, context.save_id, value.related_entities)
        return value.model_dump()

    async def goal_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        _require(request, GoalReadInput)
        return {"goal": await _thread(saves.read_goal, context.save_id, for_edit=True)}

    async def goal_create(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, GoalCreateInput)
        goal = await _thread(saves.create_goal, context.save_id, value.description, value.profile)
        await context.emit("goal.created", {"goal": goal})
        return {"goal": goal}

    async def goal_edit(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, GoalEditInput)
        goal = await _thread(saves.edit_goal, context.save_id, value.old_text, value.new_text)
        await context.emit("goal.edited", {"goal": goal})
        return {"goal": goal}

    async def file_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, FileReadInput)
        _require_narrative_path(value.path, context.agent, {"coordinator", "narrator", "consistency_checker"})
        content = await _thread(DraftFileStore(saves.draft_dir(context.save_id, context.turn_id)).read, value.path)
        return {"path": value.path, "content": content}

    async def file_write(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, FileWriteInput)
        _require_narrative_path(value.path, context.agent, {"narrator"})
        files = DraftFileStore(saves.draft_dir(context.save_id, context.turn_id))
        await _thread(files.write, value.path, value.content)
        return {"path": value.path, "content": value.content}

    async def file_edit(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, FileEditInput)
        _require_narrative_path(value.path, context.agent, {"narrator"})
        files = DraftFileStore(saves.draft_dir(context.save_id, context.turn_id))
        await _thread(files.edit, value.path, value.old_text, value.new_text)
        return {"path": value.path, "content": await _thread(files.read, value.path)}

    async def narrative_token_count(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        _require(request, NarrativeTokenCountInput)
        if context.agent not in {"narrator", "consistency_checker"}:
            raise CapabilityPermissionError("仅叙事 Agent 和一致性检查 Agent 可以检查叙事 Token")
        settings = context.token_counting_settings
        if settings is None or not settings.enabled:
            raise CapabilityError("叙事 Token 计数能力未启用")
        path = "narrative.md"
        content = await _thread(DraftFileStore(saves.draft_dir(context.save_id, context.turn_id)).read, path)
        token_count = await _thread(count_narrative_tokens, content)
        excess_tokens = max(0, token_count - settings.max_tokens)
        return {
            "path": path,
            "encoding": ENCODING_NAME,
            "token_count": token_count,
            "max_tokens": settings.max_tokens,
            "over_limit": excess_tokens > 0,
            "excess_tokens": excess_tokens,
        }

    async def story_summary_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        _require(request, StorySummaryReadInput)
        return await _thread(saves.read_story_summary, context.save_id)

    async def story_summary_edit(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, StorySummaryEditInput)
        if context.agent != "story_summarizer":
            raise CapabilityPermissionError("仅故事总结 Agent 可以编辑故事摘要")
        return await _thread(saves.edit_story_summary, context.save_id, context.turn_id, value.old_text, value.new_text)

    async def story_history_read(request: BaseModel, context: CapabilityContext) -> dict[str, Any]:
        value = _require(request, StoryHistoryReadInput)
        if context.agent != "story_summarizer":
            raise CapabilityPermissionError("仅故事总结 Agent 可以阅读完整历史回合")
        return await _thread(saves.read_story_history, context.save_id, context.turn_id, value.after_turn_number)

    async def narrative_publish(_request: BaseModel, _context: CapabilityContext) -> dict[str, Any]:
        raise RuntimeError("narrative_publish 必须由 Runtime 的发布事务执行")

    return CapabilityRegistry(
        [
            CapabilitySpec("entity_search", EntitySearchInput, entity_search),
            CapabilitySpec("entity_read", EntityReadInput, entity_read),
            CapabilitySpec("report_read", ReportReadInput, report_read),
            CapabilitySpec("research_report", WorldResearchReport, research_report),
            CapabilitySpec("style_read", StyleReadInput, style_read),
            CapabilitySpec("style_report", StylePlanningReport, style_report),
            CapabilitySpec("compliance_report", ComplianceReviewResult, compliance_report),
            CapabilitySpec("erotic_report", TextReport, text_report),
            CapabilitySpec("role_report", TextReport, text_report),
            CapabilitySpec("image_read", ImageReadInput, image_read),
            CapabilitySpec("character_portrait_generate", CharacterPortraitGenerateInput, character_portrait_generate),
            CapabilitySpec("character_create", CharacterCreateInput, character_create),
            CapabilitySpec("character_edit", CharacterEditInput, character_edit),
            CapabilitySpec("character_rename", CharacterRenameInput, character_rename),
            CapabilitySpec("character_report", CharacterChangeReport, character_report),
            CapabilitySpec("location_create", LocationCreateInput, location_create),
            CapabilitySpec("location_edit", LocationEditInput, location_edit),
            CapabilitySpec("location_rename", LocationRenameInput, location_rename),
            CapabilitySpec("location_report", LocationChangeReport, location_report),
            CapabilitySpec("goal_read", GoalReadInput, goal_read),
            CapabilitySpec("goal_create", GoalCreateInput, goal_create),
            CapabilitySpec("goal_edit", GoalEditInput, goal_edit),
            CapabilitySpec("file_read", FileReadInput, file_read),
            CapabilitySpec("file_write", FileWriteInput, file_write),
            CapabilitySpec("file_edit", FileEditInput, file_edit),
            CapabilitySpec("narrative_token_count", NarrativeTokenCountInput, narrative_token_count),
            CapabilitySpec("story_summary_read", StorySummaryReadInput, story_summary_read),
            CapabilitySpec("story_summary_edit", StorySummaryEditInput, story_summary_edit),
            CapabilitySpec("story_history_read", StoryHistoryReadInput, story_history_read),
            CapabilitySpec("narrative_publish", NarrativePublishInput, narrative_publish),
        ]
    )


async def _retrieval_counter(
    saves: SaveStore,
    context: CapabilityContext,
    *,
    read_documents: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    messages = await _thread(saves.load_session, context.save_id, context.session_id)
    read_paths: set[str] = set()
    for message in messages:
        for part in message["parts"]:
            if part["type"] != "tool" or part["state"] != "completed":
                continue
            if part["tool_name"] != "entity_read":
                continue
            try:
                output = json.loads(part["output"] or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(output, dict):
                continue
            read_paths.update(_entity_paths(output.get("documents")))

    read_paths.update(_entity_paths(read_documents))
    read_paths.difference_update(await _thread(saves.required_entity_paths, context.save_id))
    return {
        "total_searchable_entities": await _thread(saves.searchable_entity_count, context.save_id),
        "read_entities": len(read_paths),
    }


def _entity_paths(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {
        path
        for item in items
        if isinstance(item, dict)
        and isinstance(path := item.get("path"), str)
        and path
    }


def _style_paths(items: Any) -> set[str]:
    if not isinstance(items, list):
        return set()
    return {
        str(item["path"])
        for item in items
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }


def _initial_style_paths(messages: list[dict[str, Any]]) -> set[str]:
    for message in messages:
        if message.get("role") != "user":
            continue
        text = "".join(
            str(part.get("content") or "")
            for part in message.get("parts", [])
            if part.get("type") == "text"
        )
        try:
            initial = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return set()
        if not isinstance(initial, dict):
            return set()
        return _style_paths(initial.get("available_styles"))
    return set()


async def _character_paths_for_session(saves: SaveStore, context: CapabilityContext) -> set[str]:
    messages = await _thread(saves.load_session, context.save_id, context.session_id)
    available: set[str] = set()
    for message in messages:
        if message.get("role") == "user":
            text = "".join(
                str(part.get("content") or "")
                for part in message.get("parts", [])
                if part.get("type") == "text"
            )
            try:
                initial = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                initial = None
            if isinstance(initial, dict):
                available.update(_entity_paths(initial.get("related_entity_documents")))
        for part in message.get("parts", []):
            if part.get("type") != "tool" or part.get("state") != "completed":
                continue
            try:
                output = json.loads(part.get("output") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(output, dict):
                continue
            if part.get("tool_name") == "entity_read":
                available.update(_entity_paths(output.get("documents")))
            elif part.get("tool_name") in {"character_create", "character_edit", "character_rename"}:
                character = output.get("character")
                if isinstance(character, dict) and isinstance(character.get("path"), str):
                    available.add(character["path"])
    return available


async def _location_paths_for_session(saves: SaveStore, context: CapabilityContext) -> set[str]:
    messages = await _thread(saves.load_session, context.save_id, context.session_id)
    available: set[str] = set()
    for message in messages:
        if message.get("role") == "user":
            text = "".join(
                str(part.get("content") or "")
                for part in message.get("parts", [])
                if part.get("type") == "text"
            )
            try:
                initial = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                initial = None
            if isinstance(initial, dict):
                available.update(_entity_paths(initial.get("related_entity_documents")))
        for part in message.get("parts", []):
            if part.get("type") != "tool" or part.get("state") != "completed":
                continue
            try:
                output = json.loads(part.get("output") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(output, dict):
                continue
            if part.get("tool_name") == "entity_read":
                available.update(_entity_paths(output.get("documents")))
            elif part.get("tool_name") in {"location_create", "location_edit", "location_rename"}:
                location = output.get("location")
                if isinstance(location, dict) and isinstance(location.get("path"), str):
                    available.add(location["path"])
    return available


def _require_narrative_path(path: str, agent: str, agents: set[str]) -> None:
    if agent not in agents or path != "narrative.md":
        raise CapabilityPermissionError(f"Agent {agent} 无权操作该草稿文件")


def _require(request: BaseModel, expected: type[BaseModel]) -> Any:
    if not isinstance(request, expected):
        raise TypeError(f"Capability 请求类型错误：期望 {expected.__name__}")
    return request


async def _thread(function: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    import asyncio

    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancelled:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        raise cancelled
