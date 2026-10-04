from __future__ import annotations

import asyncio
import json
import re
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, BinaryIO, ParamSpec, TypeVar

from pydantic import BaseModel, ValidationError

from .agents import AGENTS, AgentRegistry, AgentSpec, CharacterChangeReport, ComplianceReviewResult, LocationChangeReport, ReportRef, StylePlanningReport, TaskToolInput, TextReport, WorldResearchReport, agent_names, coordinator_prompt, tool_schema
from .capabilities import CapabilityContext, CapabilityExecutor, CapabilityRegistry, build_runtime_capability_registry
from .config import CapabilitySettingsStore, DrawingPresetStore, NetworkSettingsStore, RuntimeSettingsStore
from .content_store import ContentStore
from .draft_files import DraftFileStore
from .message_compat import image_block
from .model_errors import ModelFallbackError
from .models import CharacterPortraitGenerationSettings, DrawingPreset, NarrativeTokenCountingSettings, NetworkSettings, RuntimeEvent, RuntimeSettings, SaveSummary
from .prompt_store import NARRATIVE_TOKEN_LIMIT_PLACEHOLDER, PromptSnapshot, PromptStore, render_save_narrative_settings
from .providers import FallbackModelClient, ModelClient
from .storage import SaveBusyError, SaveStore


CONSECUTIVE_TOOL_FAILURE_LIMIT = 8
USER_ABORT_MESSAGE = "用户中止，回合已中止"
SHUTDOWN_MESSAGE = "服务关闭，中断未完成回合"
BLOCKED_TASK_TEXT = "请开始工作"
BLOCKED_TASK_DESCRIPTION = "该子代理不接收具体任务文本，只接受实体和报告，请只要求“开始任务”。"
AUTOMATIC_PUBLISH_TEXT = "已自动发布 narrative.md。"
CONFIGURABLE_CAPABILITIES = frozenset({"character_portrait_generate", "narrative_token_count"})
P = ParamSpec("P")
R = TypeVar("R")
IMAGE_DATA_URL = re.compile(r"data:image/[-+.a-zA-Z0-9]+;base64,[a-zA-Z0-9+/=]+")


async def _thread(function: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
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


def _redact_image_text(value: str) -> str:
    return IMAGE_DATA_URL.sub(lambda match: match.group(0).split(",", 1)[0] + ",[omitted]", value)


def _public_tool_output(name: str, output: dict[str, Any]) -> dict[str, Any]:
    if name != "image_read":
        return output
    return {key: value for key, value in output.items() if key != "data"}


def _redact_image_data(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_image_text(value)
    if isinstance(value, list):
        return [_redact_image_data(item) for item in value]
    if not isinstance(value, dict):
        return value
    redacted: dict[str, Any] = {}
    for key, item in value.items():
        if key == "url" and isinstance(item, str) and item.startswith("data:image/"):
            redacted[key] = item.split(",", 1)[0] + ",[omitted]"
        else:
            redacted[key] = _redact_image_data(item)
    return redacted


class EventHub:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[asyncio.Queue[bool]]] = defaultdict(set)
        self._deleting_saves: set[str] = set()

    def subscribe(self, save_id: str) -> asyncio.Queue[bool]:
        if save_id in self._deleting_saves:
            raise KeyError(f"存档正在删除：{save_id}")
        queue: asyncio.Queue[bool] = asyncio.Queue(maxsize=1)
        self._subscribers[save_id].add(queue)
        return queue

    def unsubscribe(self, save_id: str, queue: asyncio.Queue[bool]) -> None:
        self._subscribers[save_id].discard(queue)

    def begin_save_deletion(self, save_id: str) -> bool:
        if save_id in self._deleting_saves:
            return False
        self._deleting_saves.add(save_id)
        return True

    def close_save(self, save_id: str) -> None:
        for queue in self._subscribers.pop(save_id, set()):
            if not queue.empty():
                _ = queue.get_nowait()
            queue.put_nowait(True)

    def cancel_save_deletion(self, save_id: str) -> None:
        self._deleting_saves.discard(save_id)

    def publish(self, save_id: str, _event: RuntimeEvent) -> None:
        for queue in self._subscribers[save_id]:
            if queue.empty():
                queue.put_nowait(False)


@dataclass
class RunContext:
    save_id: str
    turn_id: str
    models: dict[str, ModelClient]
    max_delegations: int
    enabled_agents: frozenset[str]
    blocked_instruction_agents: frozenset[str]
    always_attach_report_agents: frozenset[str]
    force_publish_narrative: bool
    force_start_delegation: bool
    block_coordinator_narrative_read: bool
    prompts: dict[str, str]
    task_descriptions: dict[str, str]
    capability_descriptions: dict[str, str]
    fixed_responses: dict[str, ComplianceReviewResult]
    prefills: dict[str, str]
    enabled_configurable_capabilities: frozenset[str]
    drawing_preset: DrawingPreset | None
    portrait_settings: CharacterPortraitGenerationSettings | None
    network_settings: NetworkSettings | None
    token_counting_settings: NarrativeTokenCountingSettings | None
    started: float
    delegation_count: int = 0


class ChildTaskError(RuntimeError):
    def __init__(self, task_id: str, agent: str, message: str) -> None:
        self.task_id = task_id
        self.agent = agent
        super().__init__(message)


class ReportReferenceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code: str = code
        super().__init__(message)


class RequiredReportError(ValueError):
    def __init__(self, blocked_agent: str, required_agent: str) -> None:
        self.blocked_agent = blocked_agent
        self.required_agent = required_agent
        super().__init__(f"委派 {blocked_agent} 前必须先取得 {required_agent} 的报告")


class TurnAbortError(RuntimeError):
    pass


class AgentRunner:
    """One persistent, serial message/part loop shared by every Agent."""

    def __init__(self, runtime: AgentRuntime, registry: AgentRegistry, capabilities: CapabilityRegistry) -> None:
        self.runtime = runtime
        self.registry = registry
        self.capabilities = capabilities

    async def run(self, spec: AgentSpec, session_id: str, context: RunContext) -> str | CharacterChangeReport | ComplianceReviewResult | LocationChangeReport | WorldResearchReport | StylePlanningReport | TextReport:
        consecutive_tool_failures = 0
        while True:
            messages = await _thread(self.runtime.saves.load_session, context.save_id, session_id)
            story = await _thread(self.runtime.saves.load_story, context.save_id, context.turn_id)
            is_first_model_response = not any(message["role"] == "assistant" for message in messages)
            model = context.models[spec.name]
            primary_model = model.primary if isinstance(model, FallbackModelClient) else model
            fallback_model = model.fallback if isinstance(model, FallbackModelClient) else None
            model_metadata = primary_model.metadata()
            provider_messages = await _thread(
                self._provider_messages,
                messages,
                story,
                context.save_id,
                coordinator=spec.is_main,
                image_reader=self.runtime.saves.read_turn_image_data,
                generated_reader=self.runtime.saves.read_generated_image_data,
                artifact_reader=self.runtime.saves.read_image_artifact,
                target_model=model_metadata,
            )
            prefill = context.prefills.get(spec.name)
            if prefill is not None:
                provider_messages.append({"role": "assistant", "content": prefill, "_prefix": True})
            tools = self._tools_for(
                spec,
                context.capability_descriptions,
                context.enabled_agents,
                context.task_descriptions,
                context.block_coordinator_narrative_read,
                context.enabled_configurable_capabilities,
                context.token_counting_settings.max_tokens if context.token_counting_settings else None,
            )
            system_prompt = context.prompts[spec.name]
            fixed_response = context.fixed_responses.get(spec.name)
            if fixed_response is not None:
                model_metadata = {**model_metadata, "response_source": "fixed_file"}
            message_id = await _thread(self.runtime.saves.create_assistant_message, context.save_id, session_id, model_metadata)
            if fixed_response is not None:
                content = json.dumps(fixed_response.model_dump(), ensure_ascii=False)
                await _thread(
                    self.runtime.saves.record_model_result,
                    context.save_id,
                    message_id,
                    content,
                    None,
                    [],
                    None,
                    None,
                )
                await self.runtime._emit(context.save_id, context.turn_id, "agent.fixed_response", {
                    "session_id": session_id,
                    "message_id": message_id,
                    "agent": spec.name,
                    "configured_model": model.metadata(),
                    "result": fixed_response.model_dump(),
                })
                return fixed_response
            await self.runtime._emit(context.save_id, context.turn_id, "model.request", {"session_id": session_id, "message_id": message_id, "agent": spec.name, "model": model.metadata(), "system_prompt": system_prompt, "messages": _redact_image_data(provider_messages), "tools": tools})
            started = time.monotonic()
            try:
                try:
                    result = await primary_model.complete(system_prompt, provider_messages, tools)
                except ModelFallbackError as primary_error:
                    if fallback_model is None:
                        raise
                    await self.runtime._emit(context.save_id, context.turn_id, "model.fallback", {
                        "session_id": session_id, "message_id": message_id, "agent": spec.name,
                        "reason": primary_error.reason, "error": _redact_image_text(str(primary_error)),
                        "primary_model": model_metadata, "fallback_model": fallback_model.metadata(),
                    })
                    model_metadata = fallback_model.metadata()
                    fallback_messages = await _thread(
                        self._provider_messages, messages, story, context.save_id,
                        coordinator=spec.is_main,
                        image_reader=self.runtime.saves.read_turn_image_data,
                        generated_reader=self.runtime.saves.read_generated_image_data,
                        artifact_reader=self.runtime.saves.read_image_artifact,
                        target_model=model_metadata,
                    )
                    if prefill is not None:
                        fallback_messages.append({"role": "assistant", "content": prefill, "_prefix": True})
                    await self.runtime._emit(context.save_id, context.turn_id, "model.request", {
                        "session_id": session_id, "message_id": message_id, "agent": spec.name,
                        "model": model_metadata, "system_prompt": system_prompt,
                        "messages": _redact_image_data(fallback_messages), "tools": tools,
                        "attempt": "fallback",
                    })
                    try:
                        result = await fallback_model.complete(system_prompt, fallback_messages, tools)
                    except Exception as fallback_error:
                        raise RuntimeError(f"备用 AI 失败：{fallback_error}（首选 AI 失败：{primary_error}）") from fallback_error
            except BaseException as error:
                await asyncio.shield(_thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, _redact_image_text(str(error))))
                raise
            await self.runtime._emit(context.save_id, context.turn_id, "model.response", {"session_id": session_id, "message_id": message_id, "agent": spec.name, "model": model_metadata, "content": result.content, "tool_calls": [call.model_dump() for call in result.tool_calls], "reasoning": result.reasoning, "usage": result.usage, "raw_response": result.raw_response, "duration_ms": round((time.monotonic() - started) * 1000)})
            calls = [call.model_dump() for call in result.tool_calls]
            if len(calls) > 1:
                rejection = {"ok": False, "error": {"code": "multiple_tool_calls", "message": "一次响应只能调用一个工具；本组调用均未执行"}}
                parts = await _thread(self.runtime.saves.record_model_result, context.save_id, message_id, result.content, result.reasoning, calls, rejection, result.provider_content, model_metadata)
                for part in parts:
                    await self.runtime._emit(context.save_id, context.turn_id, "tool.failed", {"session_id": session_id, "part_id": part["id"], "agent": spec.name, "tool": part["tool_name"], "input": part["input"], "result": rejection})
                consecutive_tool_failures += 1
                if consecutive_tool_failures >= CONSECUTIVE_TOOL_FAILURE_LIMIT:
                    raise RuntimeError(f"Agent {spec.name} 连续 {CONSECUTIVE_TOOL_FAILURE_LIMIT} 次工具调用失败")
                continue
            force_start = (
                spec.is_main
                and context.force_start_delegation
                and is_first_model_response
                and not calls
                and bool(result.content.strip())
            )
            if force_start:
                first_child = next(
                    child
                    for child in sorted(self.registry.children(), key=lambda item: item.template_order)
                    if child.name in context.enabled_agents
                )
                calls = [{
                    "id": f"automatic-task-{message_id}",
                    "name": "task",
                    "arguments": {"agent": first_child.name, "task": BLOCKED_TASK_TEXT, "reason": ""},
                }]
            parts = await _thread(
                self.runtime.saves.record_model_result,
                context.save_id,
                message_id,
                "" if force_start else result.content,
                None if force_start else result.reasoning,
                calls,
                None,
                None if force_start else result.provider_content,
                model_metadata,
            )
            if not parts:
                text = result.content.strip()
                if spec.is_main:
                    draft_exists = context.force_publish_narrative and text and await _thread(
                        DraftFileStore(self.runtime.saves.draft_dir(context.save_id, context.turn_id)).exists,
                        "narrative.md",
                    )
                    if draft_exists:
                        part = await _thread(
                            self.runtime.saves.create_automatic_publish_part,
                            context.save_id,
                            message_id,
                        )
                        await self.runtime._emit(context.save_id, context.turn_id, "tool.started", {
                            "session_id": session_id,
                            "part_id": part["id"],
                            "provider_call_id": part["provider_call_id"],
                            "agent": spec.name,
                            "tool": "narrative_publish",
                            "input": part["input"],
                        })
                        output, events = await _thread(
                            self.runtime.saves.publish_narrative,
                            context.save_id,
                            context.turn_id,
                            session_id,
                            part["id"],
                            round((time.monotonic() - context.started) * 1000),
                            message_id,
                            AUTOMATIC_PUBLISH_TEXT,
                        )
                        for event in events:
                            self.runtime.events.publish(context.save_id, event)
                        return output["content"]
                    await _thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, "coordinator 必须发布 narrative.md")
                    raise RuntimeError("coordinator 的普通文本不能完成回合；必须发布 narrative.md")
                if spec.name == "world_researcher":
                    await _thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, "world_researcher 必须提交 research_report")
                    raise RuntimeError("world_researcher 的普通文本不能完成任务；必须提交 research_report")
                if spec.name == "style_planner":
                    await _thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, "style_planner 必须提交 style_report")
                    raise RuntimeError("style_planner 的普通文本不能完成任务；必须提交 style_report")
                if spec.name == "character_designer":
                    await _thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, "character_designer 必须提交 character_report")
                    raise RuntimeError("character_designer 的普通文本不能完成任务；必须提交 character_report")
                if spec.name == "location_designer":
                    await _thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, "location_designer 必须提交 location_report")
                    raise RuntimeError("location_designer 的普通文本不能完成任务；必须提交 location_report")
                if spec.name == "compliance_reviewer":
                    await _thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, "compliance_reviewer 必须提交 compliance_report")
                    raise RuntimeError("compliance_reviewer 的普通文本不能完成任务；必须提交 compliance_report")
                if spec.name in ("EroticOrNot", "role_player"):
                    tool_name = "erotic_report" if spec.name == "EroticOrNot" else "role_report"
                    await _thread(self.runtime.saves.fail_assistant_message, context.save_id, message_id, f"{spec.name} 必须提交 {tool_name}")
                    raise RuntimeError(f"{spec.name} 的普通文本不能完成任务；必须提交 {tool_name}")
                return text
            completion, tool_succeeded = await self._execute_tool(spec, session_id, parts[0], context)
            if tool_succeeded:
                consecutive_tool_failures = 0
            else:
                consecutive_tool_failures += 1
                if consecutive_tool_failures >= CONSECUTIVE_TOOL_FAILURE_LIMIT:
                    raise RuntimeError(f"Agent {spec.name} 连续 {CONSECUTIVE_TOOL_FAILURE_LIMIT} 次工具调用失败")
            if completion is not None:
                return completion

    def _tools_for(
        self,
        spec: AgentSpec,
        capability_descriptions: dict[str, str],
        enabled_agents: frozenset[str] | None = None,
        task_descriptions: dict[str, str] | None = None,
        block_coordinator_narrative_read: bool = False,
        enabled_configurable_capabilities: frozenset[str] = frozenset(),
        narrative_token_limit: int | None = None,
    ) -> list[dict[str, Any]]:
        tools = []
        for cap in self.capabilities.all():
            if not (cap.name in spec.allowed_capabilities or (
                cap.name == "story_summary_read" and enabled_agents is not None
                and "story_summarizer" in enabled_agents
            )):
                continue
            if cap.name in CONFIGURABLE_CAPABILITIES and cap.name not in enabled_configurable_capabilities:
                continue
            if spec.is_main and block_coordinator_narrative_read and cap.name == "file_read":
                continue
            description = capability_descriptions[cap.name]
            if cap.name == "narrative_token_count":
                if narrative_token_limit is None:
                    raise RuntimeError("叙事 Token 计数能力缺少冻结的 Token 上限")
                description = description.replace(NARRATIVE_TOKEN_LIMIT_PLACEHOLDER, str(narrative_token_limit))
            tools.append(self._tool(cap.name, description, cap.input_model))
        if spec.is_main:
            children = tuple(
                child for child in self.registry.children()
                if enabled_agents is None or child.name in enabled_agents
            )
            descriptions = task_descriptions or {}
            catalog = "\n".join(f"- {child.name}（{child.label}）：{descriptions[child.name]}" for child in children)
            task_tool = self._tool("task", f"同步启动或继续一个专业 Agent。传入 task_id 继续原会话，省略则创建新会话；report_refs 可把当前回合已完成 task 返回的完整报告原样传给目标 Agent。Runtime 会按用户设置自动附加指定来源的报告；若尚未取得该来源的报告，则会在委派列表中更靠后的 Agent 前要求先委派该来源。根据使用条件选择：\n{catalog}", TaskToolInput)
            task_tool["function"]["parameters"]["properties"]["agent"]["enum"] = [child.name for child in children]
            tools.insert(0, task_tool)
        return tools

    @staticmethod
    def _tool(name: str, description: str, model: type[BaseModel]) -> dict[str, Any]:
        return {"type": "function", "function": {"name": name, "description": description, "parameters": tool_schema(model)}}

    async def _execute_tool(
        self, caller: AgentSpec, session_id: str, part: dict[str, Any], context: RunContext
    ) -> tuple[str | CharacterChangeReport | ComplianceReviewResult | LocationChangeReport | WorldResearchReport | StylePlanningReport | TextReport | None, bool]:
        part_id = str(part["id"])
        name = str(part["tool_name"])
        raw_input = str(part["input"])
        await _thread(self.runtime.saves.set_tool_running, context.save_id, part_id)
        await self.runtime._emit(context.save_id, context.turn_id, "tool.started", {"session_id": session_id, "part_id": part_id, "provider_call_id": part["provider_call_id"], "agent": caller.name, "tool": name, "input": raw_input})
        try:
            arguments = self._arguments(name, raw_input)
            if name == "task" and caller.is_main:
                output = await self._task(caller, session_id, part_id, arguments, context)
            elif name == "task":
                raise ValueError(f"Agent {caller.name} 无权调用工具：task")
            elif name == "file_read" and caller.is_main and context.block_coordinator_narrative_read:
                raise ValueError("主代理已禁止读取 narrative.md")
            elif name not in self._allowed_capabilities(caller, context):
                raise ValueError(f"Agent {caller.name} 无权调用工具：{name}")
            elif name == "narrative_publish":
                path = str(arguments.get("path", ""))
                if path != "narrative.md":
                    raise ValueError("narrative_publish 只接受 narrative.md")
                output, events = await _thread(self.runtime.saves.publish_narrative, context.save_id, context.turn_id, session_id, part_id, round((time.monotonic() - context.started) * 1000))
                for event in events:
                    self.runtime.events.publish(context.save_id, event)
                return "", True
            else:
                output = await self.runtime._execute_capability(name, arguments, context, caller.name, session_id, part_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            output: dict[str, Any] = {"ok": False, "error": {"code": self._error_code(error), "message": _redact_image_text(str(error))}}
            if isinstance(error, ChildTaskError):
                output.update({"task_id": error.task_id, "agent": error.agent})
            if isinstance(error, RequiredReportError):
                output.update({
                    "blocked_agent": error.blocked_agent,
                    "required_delegation": {"agent": error.required_agent},
                })
            await _thread(self.runtime.saves.error_tool, context.save_id, part_id, output)
            await self.runtime._emit(context.save_id, context.turn_id, "tool.failed", {"session_id": session_id, "part_id": part_id, "agent": caller.name, "tool": name, "input": raw_input, "result": output})
            return None, False
        await _thread(self.runtime.saves.complete_tool, context.save_id, part_id, output)
        await self.runtime._emit(context.save_id, context.turn_id, "tool.completed", {"session_id": session_id, "part_id": part_id, "agent": caller.name, "tool": name, "input": raw_input, "result": _public_tool_output(name, output)})
        if name == "research_report":
            if caller.name != "world_researcher":
                raise RuntimeError("research_report 只能终结 world_researcher")
            return WorldResearchReport.model_validate(output), True
        if name == "style_report":
            if caller.name != "style_planner":
                raise RuntimeError("style_report 只能终结 style_planner")
            return StylePlanningReport.model_validate(output), True
        if name == "character_report":
            if caller.name != "character_designer":
                raise RuntimeError("character_report 只能终结 character_designer")
            return CharacterChangeReport.model_validate(output), True
        if name == "location_report":
            if caller.name != "location_designer":
                raise RuntimeError("location_report 只能终结 location_designer")
            return LocationChangeReport.model_validate(output), True
        if name == "compliance_report":
            if caller.name != "compliance_reviewer":
                raise RuntimeError("compliance_report 只能终结 compliance_reviewer")
            return ComplianceReviewResult.model_validate(output), True
        if name in ("erotic_report", "role_report"):
            expected_agent = "EroticOrNot" if name == "erotic_report" else "role_player"
            if caller.name != expected_agent:
                raise RuntimeError(f"{name} 只能终结 {expected_agent}")
            return TextReport.model_validate(output), True
        return None, True

    @staticmethod
    def _allowed_capabilities(caller: AgentSpec, context: RunContext) -> set[str]:
        allowed = set(caller.allowed_capabilities)
        if "story_summarizer" in context.enabled_agents:
            allowed.add("story_summary_read")
        allowed -= CONFIGURABLE_CAPABILITIES - context.enabled_configurable_capabilities
        return allowed

    async def _task(
        self,
        caller: AgentSpec,
        parent_session_id: str,
        part_id: str,
        arguments: dict[str, Any],
        context: RunContext,
    ) -> dict[str, Any]:
        request = TaskToolInput.model_validate(arguments)
        try:
            child = self.registry.get(request.agent)
        except KeyError as error:
            raise ValueError(f"未知专业 Agent：{request.agent}") from error
        if child.is_main:
            raise ValueError("task 不能调用 coordinator")
        if child.name not in context.enabled_agents:
            raise ValueError(f"Agent 已禁用：{child.name}")
        if context.delegation_count >= context.max_delegations:
            raise ValueError(f"总调度师超过每轮 {context.max_delegations} 次委派预算")
        related_entity_paths = list(request.related_entities)
        required_entity_paths: list[str] = []
        if request.task_id is None and child.accepts_entities:
            required_entity_paths = await _thread(self.runtime.saves.required_entity_paths, context.save_id)
            related_entity_paths.extend(path for path in required_entity_paths if path not in related_entity_paths)
        if request.related_entities:
            reported = await self._reported_entity_paths(context.save_id, parent_session_id)
            unreported = [
                path for path in request.related_entities
                if path not in reported and path not in required_entity_paths
            ]
            if unreported:
                raise ValueError(f"相关实体未由当前 researcher 报告：{', '.join(unreported)}")
        style_documents: list[dict[str, Any]] = []
        if request.style_paths:
            reported_styles = await self._reported_style_selections(context.save_id, parent_session_id)
            if request.style_paths not in reported_styles:
                raise ValueError("narrator 文风及顺序必须与当前 style_planner 的一份完整报告一致")
            style_documents = await self._hydrate_style_context(
                caller.name, child.name, "structured_task", request.style_paths
            )
        report_refs = await self._merge_report_refs(
            context.save_id,
            parent_session_id,
            request.report_refs,
            context.always_attach_report_agents & context.enabled_agents,
            request.task_id,
        )
        report_documents = await self._resolve_report_refs(
            context.save_id, parent_session_id, report_refs
        )
        report_agents = {str(document["source_agent"]) for document in report_documents}
        required_agent = next((
            spec.name
            for spec in sorted(self.registry.children(), key=lambda item: item.template_order)
            if spec.name in context.enabled_agents
            and spec.name in context.always_attach_report_agents
            and spec.template_order < child.template_order
            and spec.name not in report_agents
        ), None)
        if required_agent is not None:
            raise RequiredReportError(child.name, required_agent)
        first_story_turn = False
        if child.name == "story_summarizer":
            story = await _thread(self.runtime.saves.load_story, context.save_id, context.turn_id)
            first_story_turn = story["story"]["turns"][-1]["turn_number"] == 1
            await _thread(self.runtime.saves.ensure_story_summary, context.save_id)
        instruction_blocked = child.name in context.blocked_instruction_agents
        transfers: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
        if related_entity_paths or report_refs:
            async def resolve_transfer_reports(refs: list[ReportRef]) -> list[dict[str, Any]]:
                return await self._resolve_report_refs(context.save_id, parent_session_id, refs)

            async def ignore_transfer_event(_kind: str, _payload: dict[str, Any]) -> None:
                return None

            transfer_context = CapabilityContext(
                context.save_id, context.turn_id, child.name,
                ignore_transfer_event, report_resolver=resolve_transfer_reports,
            )
            if request.task_id is None and related_entity_paths:
                entity_input = {"paths": related_entity_paths}
                entity_output = await self.runtime.capabilities.execute(
                    "entity_read", entity_input, transfer_context, {"entity_read"},
                )
                transfers.append(("entity_read", entity_input, entity_output))
            if report_refs:
                report_input = {"report_refs": [ref.model_dump() for ref in report_refs]}
                report_output = await self.runtime.capabilities.execute(
                    "report_read", report_input, transfer_context, {"report_read"},
                )
                transfers.append(("report_read", report_input, report_output))
        trace_entity_documents = transfers[0][2]["documents"] if transfers and transfers[0][0] == "entity_read" else []
        available_styles = None
        if child.name == "style_planner" and request.task_id is None:
            available_styles = [
                style.model_dump(exclude={"enabled"})
                for style in await _thread(self.runtime.styles.list_enabled_styles)
            ]
        child_session_id = await _thread(
            self.runtime.saves.prepare_child_task,
            context.save_id,
            context.turn_id,
            parent_session_id,
            part_id,
            child.name,
            {
                "task": BLOCKED_TASK_TEXT if instruction_blocked else request.task,
                "reason": "" if instruction_blocked else request.reason,
                "related_entities": related_entity_paths,
                "style_paths": request.style_paths,
                "report_refs": [item.model_dump() for item in report_refs],
            },
            request.task_id,
            transfers,
            available_styles,
            style_documents,
        )
        context.delegation_count += 1
        await self.runtime._emit(context.save_id, context.turn_id, "task.created", {
            "session_id": parent_session_id,
            "part_id": part_id,
            "task_id": child_session_id,
            "agent": child.name,
            "initiator_agent": caller.name,
            "target_agent": child.name,
            "task": request.task,
            "reason": request.reason,
            "instruction_blocked": instruction_blocked,
            "related_entities": [
                {"path": document["path"], "name": document.get("name", "")}
                for document in trace_entity_documents
            ],
            "reports": [
                {
                    "name": document["name"],
                    "source_agent": document["source_agent"],
                    "source_task": document["source_task"],
                }
                for document in report_documents
            ],
            "styles": [
                {"path": document["path"], "name": document.get("name", "")}
                for document in style_documents
            ],
        })
        try:
            if first_story_turn:
                result = "现在是首个回合，无可总结内容。"
                message_id = await _thread(
                    self.runtime.saves.create_assistant_message,
                    context.save_id, child_session_id, {"provider": "runtime", "model": "story_summary_precheck"},
                )
                await _thread(
                    self.runtime.saves.record_model_result,
                    context.save_id, message_id, result, None, [], None, None,
                )
                await self.runtime._emit(context.save_id, context.turn_id, "agent.fixed_response", {
                    "session_id": child_session_id, "message_id": message_id,
                    "agent": child.name, "response_source": "runtime", "result": result,
                })
            else:
                result = await self.run(child, child_session_id, context)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise ChildTaskError(child_session_id, child.name, _redact_image_text(str(error))) from error
        if child.name == "world_researcher":
            if not isinstance(result, WorldResearchReport):
                raise ChildTaskError(child_session_id, child.name, "world_researcher 未返回结构化报告")
            documents = await self._hydrate_entity_context(
                child.name, caller.name, "structured_report", result.related_entities, context
            )
            return self._task_result(child, part_id, {
                "ok": True,
                "task_id": child_session_id,
                "agent": child.name,
                "result": result.model_dump(),
                "related_entity_documents": documents,
            })
        if child.name == "character_designer":
            if not isinstance(result, CharacterChangeReport):
                raise ChildTaskError(child_session_id, child.name, "character_designer 未返回结构化报告")
            documents = await self._hydrate_entity_context(
                child.name, caller.name, "structured_report", result.related_entities, context
            )
            return self._task_result(child, part_id, {
                "ok": True,
                "task_id": child_session_id,
                "agent": child.name,
                "result": result.model_dump(),
                "related_entity_documents": documents,
            })
        if child.name == "location_designer":
            if not isinstance(result, LocationChangeReport):
                raise ChildTaskError(child_session_id, child.name, "location_designer 未返回结构化报告")
            documents = await self._hydrate_entity_context(
                child.name, caller.name, "structured_report", result.related_entities, context
            )
            return self._task_result(child, part_id, {
                "ok": True,
                "task_id": child_session_id,
                "agent": child.name,
                "result": result.model_dump(),
                "related_entity_documents": documents,
            })
        if child.name == "compliance_reviewer":
            if not isinstance(result, ComplianceReviewResult):
                raise ChildTaskError(child_session_id, child.name, "compliance_reviewer 未返回结构化审核结果")
            return self._task_result(child, part_id, {
                "ok": True,
                "task_id": child_session_id,
                "agent": child.name,
                "result": result.model_dump(),
            })
        if child.name == "style_planner":
            if not isinstance(result, StylePlanningReport):
                raise ChildTaskError(child_session_id, child.name, "style_planner 未返回结构化报告")
            return self._task_result(child, part_id, {
                "ok": True,
                "task_id": child_session_id,
                "agent": child.name,
                "result": result.model_dump(),
            })
        if child.name in ("EroticOrNot", "role_player"):
            if not isinstance(result, TextReport):
                raise ChildTaskError(child_session_id, child.name, f"{child.name} 未返回结构化报告")
            return self._task_result(child, part_id, {
                "ok": True,
                "task_id": child_session_id,
                "agent": child.name,
                "result": result.model_dump(),
            })
        if child.name == "goal_keeper":
            if not isinstance(result, str):
                raise ChildTaskError(child_session_id, child.name, "目标维护 Agent 未正常结束任务")
            goal = await _thread(self.runtime.saves.read_goal, context.save_id)
            return self._task_result(child, part_id, {
                "ok": True,
                "task_id": child_session_id,
                "agent": child.name,
                "result": result,
                "goal": goal,
            })
        if not isinstance(result, str):
            raise ChildTaskError(child_session_id, child.name, "子 Agent 返回类型无效")
        return self._task_result(child, part_id, {
            "ok": True,
            "task_id": child_session_id,
            "agent": child.name,
            "result": result,
        })

    def _task_result(self, spec: AgentSpec, part_id: str, result: dict[str, Any]) -> dict[str, Any]:
        if spec.produces_reports:
            result["report_ref"] = self._report_ref(spec, part_id).model_dump()
        return result

    @staticmethod
    def _report_ref(spec: AgentSpec, part_id: str) -> ReportRef:
        return ReportRef(name=f"{spec.name} 报告", id=part_id)

    async def _merge_report_refs(
        self,
        save_id: str,
        parent_session_id: str,
        explicit_refs: list[ReportRef],
        always_attach_agents: frozenset[str],
        target_task_id: str | None,
    ) -> list[ReportRef]:
        merged = list(explicit_refs)
        seen = {ref.id for ref in explicit_refs}
        if not always_attach_agents:
            return merged
        messages = await _thread(self.runtime.saves.load_session, save_id, parent_session_id)
        for message in messages:
            for part in message["parts"]:
                part_id = str(part["id"])
                child_session_id = str(part["child_session_id"] or "")
                if (
                    part["type"] != "tool"
                    or part["tool_name"] != "task"
                    or part["state"] != "completed"
                    or not child_session_id
                    or part_id in seen
                    or child_session_id == target_task_id
                ):
                    continue
                try:
                    agent = await _thread(self.runtime.saves.session_agent, save_id, child_session_id)
                    spec = self.registry.get(agent)
                except KeyError:
                    continue
                if agent not in always_attach_agents or not spec.produces_reports:
                    continue
                merged.append(self._report_ref(spec, part_id))
                seen.add(part_id)
        return merged

    async def _resolve_report_refs(
        self,
        save_id: str,
        parent_session_id: str,
        refs: list[ReportRef],
    ) -> list[dict[str, Any]]:
        if not refs:
            return []
        messages = await _thread(self.runtime.saves.load_session, save_id, parent_session_id)
        parts = {
            str(part["id"]): part
            for message in messages
            for part in message["parts"]
            if part["type"] == "tool" and part["tool_name"] == "task" and part["state"] == "completed"
        }
        documents: list[dict[str, Any]] = []
        for index, ref in enumerate(refs):
            part = parts.get(ref.id)
            if part is None or not part["child_session_id"]:
                raise ReportReferenceError(
                    "report_not_found",
                    f'找不到报告：report_refs[{index}]，name="{ref.name}"，id="{ref.id}"',
                )
            child_session_id = str(part["child_session_id"])
            try:
                agent = await _thread(self.runtime.saves.session_agent, save_id, child_session_id)
                spec = self.registry.get(agent)
                expected_ref = self._report_ref(spec, ref.id)
                output = json.loads(part["output"] or "{}")
                if not isinstance(output, dict):
                    raise ValueError
                stored_ref = ReportRef.model_validate(output.get("report_ref"))
                if (
                    output.get("ok") is not True
                    or output.get("agent") != agent
                    or output.get("task_id") != child_session_id
                    or stored_ref != expected_ref
                ):
                    raise ValueError
            except (KeyError, json.JSONDecodeError, TypeError, ValidationError, ValueError):
                raise ReportReferenceError(
                    "report_not_found",
                    f'找不到报告：report_refs[{index}]，name="{ref.name}"，id="{ref.id}"',
                ) from None
            if ref.name != expected_ref.name:
                raise ReportReferenceError(
                    "report_name_mismatch",
                    f'报告名称不匹配：id="{ref.id}" 对应名称为“{expected_ref.name}”，收到“{ref.name}”',
                )
            result = output.get("result")
            try:
                if agent == "world_researcher":
                    content = WorldResearchReport.model_validate(result).report
                elif agent == "character_designer":
                    content = CharacterChangeReport.model_validate(result).report
                elif agent == "location_designer":
                    content = LocationChangeReport.model_validate(result).report
                elif agent == "compliance_reviewer":
                    content = ComplianceReviewResult.model_validate(result).model_dump_json()
                elif agent == "style_planner":
                    content = StylePlanningReport.model_validate(result).report
                elif agent in ("EroticOrNot", "role_player"):
                    content = TextReport.model_validate(result).report
                elif isinstance(result, str):
                    content = result
                else:
                    raise ValueError
                source = TaskToolInput.model_validate_json(str(part["input"]))
            except (ValidationError, ValueError):
                raise ReportReferenceError(
                    "report_not_found",
                    f'找不到报告：report_refs[{index}]，name="{ref.name}"，id="{ref.id}"',
                ) from None
            documents.append({
                "name": ref.name,
                "id": ref.id,
                "source_agent": agent,
                "source_task_id": child_session_id,
                "source_task": source.task,
                "content": content,
            })
        return documents

    async def _hydrate_entity_context(
        self,
        sender: str,
        receiver: str,
        transfer: str,
        paths: list[str],
        context: RunContext,
    ) -> list[dict[str, Any]]:
        allowed = {
            ("world_researcher", "coordinator", "structured_report"),
            ("character_designer", "coordinator", "structured_report"),
            ("location_designer", "coordinator", "structured_report"),
            ("coordinator", "character_designer", "structured_task"),
            ("coordinator", "location_designer", "structured_task"),
            ("coordinator", "EroticOrNot", "structured_task"),
            ("coordinator", "goal_keeper", "structured_task"),
            ("coordinator", "role_player", "structured_task"),
            ("coordinator", "style_planner", "structured_task"),
            ("coordinator", "narrator", "structured_task"),
        }
        if (sender, receiver, transfer) not in allowed:
            raise ValueError("实体上下文传递方向或方式无效")
        if receiver == "character_designer":
            reader = self.runtime.saves.read_entities_exact_for_character_edit
        elif receiver == "location_designer":
            reader = self.runtime.saves.read_entities_exact_for_location_edit
        else:
            reader = self.runtime.saves.read_entities_exact
        return await _thread(reader, context.save_id, paths)

    async def _hydrate_style_context(
        self,
        sender: str,
        receiver: str,
        transfer: str,
        paths: list[str],
    ) -> list[dict[str, Any]]:
        if (sender, receiver, transfer) != ("coordinator", "narrator", "structured_task"):
            raise ValueError("文风上下文传递方向或方式无效")
        documents = await _thread(self.runtime.styles.read_enabled_styles, paths)
        return [document.model_dump() for document in documents]

    async def _reported_entity_paths(self, save_id: str, parent_session_id: str) -> set[str]:
        messages = await _thread(self.runtime.saves.load_session, save_id, parent_session_id)
        reported: set[str] = set()
        for message in messages:
            for part in message["parts"]:
                if part["type"] != "tool" or part["tool_name"] != "task" or part["state"] != "completed":
                    continue
                child_session_id = part["child_session_id"]
                if not child_session_id:
                    continue
                source_agent = await _thread(self.runtime.saves.session_agent, save_id, child_session_id)
                if source_agent not in {"world_researcher", "character_designer", "location_designer"}:
                    continue
                try:
                    output = json.loads(part["output"] or "{}")
                    if (
                        not isinstance(output, dict)
                        or output.get("ok") is not True
                        or output.get("agent") != source_agent
                        or output.get("task_id") != child_session_id
                    ):
                        continue
                    if source_agent == "world_researcher":
                        report_paths = WorldResearchReport.model_validate(output.get("result")).related_entities
                    elif source_agent == "character_designer":
                        report_paths = CharacterChangeReport.model_validate(output.get("result")).related_entities
                    else:
                        report_paths = LocationChangeReport.model_validate(output.get("result")).related_entities
                except (json.JSONDecodeError, TypeError, ValidationError):
                    continue
                reported.update(report_paths)
        return reported

    async def _reported_style_selections(self, save_id: str, parent_session_id: str) -> list[list[str]]:
        messages = await _thread(self.runtime.saves.load_session, save_id, parent_session_id)
        reported: list[list[str]] = []
        for message in messages:
            for part in message["parts"]:
                if part["type"] != "tool" or part["tool_name"] != "task" or part["state"] != "completed":
                    continue
                child_session_id = part["child_session_id"]
                if not child_session_id:
                    continue
                if await _thread(self.runtime.saves.session_agent, save_id, child_session_id) != "style_planner":
                    continue
                try:
                    output = json.loads(part["output"] or "{}")
                    if (
                        not isinstance(output, dict)
                        or output.get("ok") is not True
                        or output.get("agent") != "style_planner"
                        or output.get("task_id") != child_session_id
                    ):
                        continue
                    report = StylePlanningReport.model_validate(output.get("result"))
                except (json.JSONDecodeError, TypeError, ValidationError):
                    continue
                reported.append(report.style_paths)
        return reported

    @staticmethod
    def _arguments(name: str, raw: str) -> dict[str, Any]:
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"工具 {name} 参数不是合法 JSON 对象：{error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"工具 {name} 参数必须是 JSON 对象")
        return value

    @staticmethod
    def _error_code(error: Exception) -> str:
        if isinstance(error, ChildTaskError):
            return "child_failed"
        if isinstance(error, ReportReferenceError):
            return error.code
        if isinstance(error, RequiredReportError):
            return "required_report_missing"
        if isinstance(error, (ValidationError, ValueError)):
            return "validation_error"
        return "tool_error"

    @staticmethod
    def _provider_messages(
        messages: list[dict[str, Any]],
        story: dict[str, Any],
        save_id: str = "",
        *,
        coordinator: bool = False,
        image_reader: Callable[[str, str], dict[str, str]] | None = None,
        generated_reader: Callable[[str, str], dict[str, str]] | None = None,
        artifact_reader: Callable[[str, str], dict[str, Any]] | None = None,
        target_model: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        story_messages = AgentRunner._story_messages(
            story,
            save_id,
            coordinator=coordinator,
            image_reader=image_reader,
            generated_reader=generated_reader,
        )
        converted = list(story_messages)
        for index, message in enumerate(messages):
            parts = message["parts"]
            text = "".join(str(part["content"] or "") for part in parts if part["type"] == "text")
            if message["role"] == "user":
                if coordinator and index == 0:
                    continue
                converted.append({"role": "user", "content": text})
                continue
            tool_parts = [part for part in parts if part["type"] == "tool"]
            reasoning_parts = [part for part in parts if part["type"] == "reasoning"]
            if not text and not tool_parts and not reasoning_parts:
                continue
            assistant: dict[str, Any] = {"role": "assistant", "content": text or None}
            assistant["_reasoning_content"] = (
                "".join(str(part["content"] or "") for part in reasoning_parts)
                if reasoning_parts
                else None
            )
            try:
                model = json.loads(message["model"] or "{}")
            except (json.JSONDecodeError, TypeError):
                model = {}
            if (isinstance(model, dict) and isinstance(model.get("provider_content"), dict)
                and (target_model is None or (
                    model.get("provider") == target_model.get("provider")
                    and model.get("xai_protocol") == target_model.get("xai_protocol")
                    and model.get("openai_protocol", "chat_completions") == target_model.get("openai_protocol", "chat_completions")
                    and model.get("preset_id") == target_model.get("preset_id")
                    and model.get("model") == target_model.get("model")
                    and model.get("base_url") == target_model.get("base_url")
                ))):
                assistant["_provider_content"] = model["provider_content"]
            if tool_parts:
                assistant["tool_calls"] = [
                    {"id": part["provider_call_id"], "type": "function", "function": {"name": part["tool_name"], "arguments": part["input"]}}
                    for part in tool_parts
                ]
            converted.append(assistant)
            for part in tool_parts:
                if part["state"] in {"completed", "error"}:
                    content: Any = part["output"] or "{}"
                    if part["tool_name"] == "image_read" and part["state"] == "completed":
                        try:
                            image = json.loads(content)
                        except (json.JSONDecodeError, TypeError) as error:
                            raise RuntimeError("已完成的 image_read 结果不是合法 JSON") from error
                        if not isinstance(image, dict) or not isinstance(image.get("data"), str) or not isinstance(image.get("mime_type"), str):
                            raise RuntimeError("已完成的 image_read 结果缺少图片数据")
                        metadata = {key: value for key, value in image.items() if key != "data"}
                        content = [
                            {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)},
                            image_block(image["mime_type"], image["data"]),
                        ]
                    elif part["tool_name"] == "character_portrait_generate" and part["state"] == "completed":
                        try:
                            metadata = json.loads(content)
                        except (json.JSONDecodeError, TypeError) as error:
                            raise RuntimeError("已完成的角色立绘生成结果不是合法 JSON") from error
                        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
                            raise RuntimeError("已完成的角色立绘生成结果缺少 artifact 路径")
                        if artifact_reader is None:
                            raise RuntimeError("缺少角色立绘 artifact 读取器")
                        image = artifact_reader(save_id, metadata["path"])
                        if image.get("sha256") != metadata.get("sha256"):
                            raise RuntimeError("角色立绘 artifact 的内容与已完成 Tool Result 不一致")
                        content = [
                            {"type": "text", "text": json.dumps(metadata, ensure_ascii=False)},
                            image_block(str(image["mime_type"]), str(image["data"])),
                        ]
                    elif part["tool_name"] == "story_history_read" and part["state"] == "completed":
                        try:
                            history = json.loads(content)
                            references = [
                                (turn, image["id"], source)
                                for turn in history["turns"]
                                for source, images in (
                                    ("player", turn["player"]["images"]),
                                    ("generated", turn.get("generated_images", [])),
                                )
                                for image in images
                            ]
                        except (json.JSONDecodeError, TypeError, KeyError) as error:
                            raise RuntimeError("已完成的 story_history_read 结果无效") from error
                        if references:
                            blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
                            for turn, image_id, source in references:
                                reader = image_reader if source == "player" else generated_reader
                                if reader is None:
                                    raise RuntimeError("缺少历史回合图片读取器")
                                image = reader(save_id, image_id)
                                reference = {"image_id": image_id} if source == "player" else {
                                    "turn_number": turn["turn_number"], "source": "generated", "image_id": image_id,
                                }
                                blocks.append({"type": "text", "text": json.dumps(reference, ensure_ascii=False)})
                                blocks.append(image_block(image["mime_type"], image["data"]))
                            content = blocks
                    converted.append({
                        "role": "tool",
                        "tool_call_id": part["provider_call_id"],
                        "content": content,
                        "_tool_error": part["state"] == "error",
                    })
        return converted

    @staticmethod
    def _story_messages(
        value: dict[str, Any],
        save_id: str,
        *,
        coordinator: bool,
        image_reader: Callable[[str, str], dict[str, str]] | None,
        generated_reader: Callable[[str, str], dict[str, str]] | None = None,
    ) -> list[dict[str, Any]]:
        story = value["story"]
        if coordinator:
            messages: list[dict[str, Any]] = [{"role": "system", "content": "<recent_story>"}]
            if story["opening"]:
                messages.append({"role": "assistant", "content": story["opening"]})
            for index, turn in enumerate(story["turns"]):
                player = turn["player"]
                player_message: dict[str, Any] = {
                    "role": "user",
                    "content": AgentRunner._content_with_images(
                        save_id,
                        player["content"],
                        player["images"],
                        image_reader,
                    ),
                }
                if not turn["has_ai_output"] and index + 1 < len(story["turns"]):
                    player_message["_separate_next_user"] = True
                messages.append(player_message)
                if turn["has_ai_output"]:
                    messages.append({"role": "assistant", "content": turn["narrative"]})
                generated = turn.get("generated_images", [])
                if generated:
                    # All supported providers accept image input in user messages.
                    # Keep the narrative as assistant text and explicitly identify
                    # these image blocks as AI output from that same turn.
                    messages.append({
                        "role": "user",
                        "content": AgentRunner._content_with_images(
                            save_id,
                            f"第 {turn['turn_number']} 回合 AI 生成的图片（非玩家上传）：",
                            generated,
                            generated_reader,
                        ),
                        "_separate_next_user": index + 1 < len(story["turns"]),
                    })
            messages.append({"role": "system", "content": "</recent_story>"})
            return messages

        references = [
            (turn, image, source)
            for turn in story["turns"]
            for source, images in (
                ("player", turn["player"]["images"]),
                ("generated", turn.get("generated_images", [])),
            )
            for image in images
        ]
        serialized = json.dumps(value, ensure_ascii=False)
        if not references:
            return [{"role": "user", "content": serialized, "_separate_next_user": True}]
        content: list[dict[str, Any]] = [{"type": "text", "text": serialized}]
        for turn, reference, source in references:
            reader = image_reader if source == "player" else generated_reader
            if reader is None:
                raise RuntimeError("缺少回合图片读取器")
            image = reader(save_id, reference["id"])
            if not isinstance(image, dict) or not isinstance(image.get("mime_type"), str) or not isinstance(image.get("data"), str):
                raise RuntimeError("回合图片读取结果缺少图片数据")
            label = {"image_id": reference["id"]} if source == "player" else {
                "turn_number": turn["turn_number"], "source": "generated", "image_id": reference["id"],
            }
            content.append({"type": "text", "text": json.dumps(label, ensure_ascii=False)})
            content.append(image_block(image["mime_type"], image["data"]))
        return [{"role": "user", "content": content, "_separate_next_user": True}]

    @staticmethod
    def _content_with_images(
        save_id: str,
        text: str,
        references: Any,
        image_reader: Callable[[str, str], dict[str, str]] | None,
    ) -> str | list[dict[str, Any]]:
        if references is None or references == []:
            return text
        if not isinstance(references, list):
            raise RuntimeError("回合图片引用必须是数组")
        if image_reader is None:
            raise RuntimeError("缺少回合图片读取器")
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for reference in references:
            if not isinstance(reference, dict) or not isinstance(reference.get("id"), str):
                raise RuntimeError("回合图片引用缺少合法 ID")
            image = image_reader(save_id, reference["id"])
            if not isinstance(image, dict) or not isinstance(image.get("mime_type"), str) or not isinstance(image.get("data"), str):
                raise RuntimeError("回合图片读取结果缺少图片数据")
            content.append(image_block(image["mime_type"], image["data"]))
        return content


class AgentRuntime:
    def __init__(self, saves: SaveStore, styles: ContentStore, model: ModelClient, events: EventHub, settings: RuntimeSettingsStore, prompts: PromptStore, capability_settings: CapabilitySettingsStore, drawing_presets: DrawingPresetStore, network_settings: NetworkSettingsStore) -> None:
        self.saves, self.styles, self.model, self.events, self.settings, self.prompts = saves, styles, model, events, settings, prompts
        self.capability_settings = capability_settings
        self.drawing_presets = drawing_presets
        self.network_settings = network_settings
        self.capability_registry = build_runtime_capability_registry(saves, styles)
        self.capabilities = CapabilityExecutor(self.capability_registry)
        self.runner = AgentRunner(self, AGENTS, self.capability_registry)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._tasks: set[asyncio.Task[None]] = set()
        self._turn_tasks: dict[tuple[str, str], asyncio.Task[None]] = {}

    async def start_turn(
        self,
        save_id: str,
        player_input: str,
        images: list[tuple[str, BinaryIO]] | None = None,
    ) -> str:
        async with self._locks[save_id]:
            settings = await _thread(self.settings.get)
            enabled_agents = self._enabled_agents(settings)
            prompts = await _thread(
                self._load_prompts,
                save_id,
                enabled_agents,
                frozenset(settings.prefill_agents),
                frozenset(settings.blocked_instruction_agents),
                settings.use_compliance_fixed_response,
            )
            capability_snapshot = await self._capability_snapshot(enabled_agents)
            turn = await _thread(self.saves.create_turn, save_id, player_input, settings.context_turns, images)
            self._schedule(save_id, turn.id, settings, enabled_agents, prompts, capability_snapshot)
            return turn.id

    async def retry_turn(self, save_id: str, turn_id: str) -> str:
        async with self._locks[save_id]:
            settings = await _thread(self.settings.get)
            enabled_agents = self._enabled_agents(settings)
            prompts = await _thread(
                self._load_prompts,
                save_id,
                enabled_agents,
                frozenset(settings.prefill_agents),
                frozenset(settings.blocked_instruction_agents),
                settings.use_compliance_fixed_response,
            )
            capability_snapshot = await self._capability_snapshot(enabled_agents)
            turn = await _thread(self.saves.retry_turn, save_id, turn_id, settings.context_turns)
            self._schedule(save_id, turn.id, settings, enabled_agents, prompts, capability_snapshot, True)
            return turn.id

    @staticmethod
    def _enabled_agents(settings: RuntimeSettings) -> frozenset[str]:
        return frozenset(agent_names()) - set(settings.disabled_agents)

    def _load_prompts(
        self,
        save_id: str,
        enabled_agents: frozenset[str],
        prefill_agents: frozenset[str],
        blocked_instruction_agents: frozenset[str],
        use_compliance_fixed_response: bool,
    ) -> PromptSnapshot:
        fixed_response_agents: frozenset[str] = (
            frozenset({"compliance_reviewer"}) if use_compliance_fixed_response else frozenset()
        )
        snapshot = self.prompts.load(enabled_agents, prefill_agents, fixed_response_agents)
        enabled_children = tuple(spec for spec in AGENTS.children() if spec.name in enabled_agents)
        prompts = dict(snapshot.prompts)
        prompts[AGENTS.main().name] = coordinator_prompt(prompts[AGENTS.main().name], enabled_children)
        save = self.saves.get_save(save_id)
        for name in ("narrator", "consistency_checker"):
            if name in prompts:
                prompts[name] = render_save_narrative_settings(prompts[name], save)
        task_descriptions = {
            name: f"{description}\n{BLOCKED_TASK_DESCRIPTION}"
            if name in blocked_instruction_agents else description
            for name, description in snapshot.task_descriptions.items()
        }
        return PromptSnapshot(
            prompts,
            task_descriptions,
            snapshot.capability_descriptions,
            snapshot.fixed_responses,
            snapshot.prefills,
        )

    async def _capability_snapshot(
        self,
        enabled_agents: frozenset[str],
    ) -> tuple[NarrativeTokenCountingSettings | None, CharacterPortraitGenerationSettings | None, DrawingPreset | None, NetworkSettings | None]:
        configured = await _thread(self.capability_settings.get)
        token_counting = configured.narrative_token_counting
        enabled_token_counting = token_counting if token_counting.enabled else None
        portrait = configured.character_portrait_generation
        if not portrait.enabled or "character_designer" not in enabled_agents:
            return enabled_token_counting, None, None, None
        return (
            enabled_token_counting,
            portrait,
            await _thread(self.drawing_presets.get_default),
            await _thread(self.network_settings.get),
        )

    def _schedule(self, save_id: str, turn_id: str, settings: RuntimeSettings, enabled_agents: frozenset[str], prompts: PromptSnapshot, capability_snapshot: tuple[NarrativeTokenCountingSettings | None, CharacterPortraitGenerationSettings | None, DrawingPreset | None, NetworkSettings | None], resumed: bool = False) -> None:
        task = asyncio.create_task(self._run_locked(save_id, turn_id, settings, enabled_agents, prompts, capability_snapshot, resumed))
        key = (save_id, turn_id)
        self._tasks.add(task)
        self._turn_tasks[key] = task

        def cleanup(done: asyncio.Task[None]) -> None:
            self._tasks.discard(done)
            if self._turn_tasks.get(key) is done:
                self._turn_tasks.pop(key, None)

        task.add_done_callback(cleanup)

    async def abort_turn(self, save_id: str, turn_id: str) -> str:
        turn = await _thread(self.saves.get_turn, save_id, turn_id)
        if turn.status == "interrupted":
            return turn.id
        if turn.status != "running":
            raise TurnAbortError(f"只有执行中的回合可以中止，当前状态：{turn.status}")
        task = self._turn_tasks.get((save_id, turn_id))
        if task is None or task.done():
            raise TurnAbortError("当前进程中找不到该回合的活动执行")
        task.cancel(USER_ABORT_MESSAGE)
        await asyncio.gather(task, return_exceptions=True)
        turn = await _thread(self.saves.get_turn, save_id, turn_id)
        if turn.status == "interrupted":
            return turn.id
        raise TurnAbortError(f"回合已进入其他状态：{turn.status}")

    async def delete_save(self, save_id: str) -> None:
        async with self._locks[save_id]:
            if not self.events.begin_save_deletion(save_id):
                raise SaveBusyError("当前存档正在删除")
            try:
                await _thread(self.saves.delete_save, save_id)
            except BaseException:
                self.events.cancel_save_deletion(save_id)
                raise
            self.events.close_save(save_id)

    async def branch_save(self, save_id: str, turn_id: str, name: str) -> SaveSummary:
        async with self._locks[save_id]:
            return await _thread(self.saves.branch_save, save_id, turn_id, name)

    async def _run_locked(self, save_id: str, turn_id: str, settings: RuntimeSettings, enabled_agents: frozenset[str], prompts: PromptSnapshot, capability_snapshot: tuple[NarrativeTokenCountingSettings | None, CharacterPortraitGenerationSettings | None, DrawingPreset | None, NetworkSettings | None], resumed: bool) -> None:
        async with self._locks[save_id]:
            try:
                names = tuple(spec.name for spec in AGENTS.all() if spec.name in enabled_agents)
                models = await _thread(self.model.bind_for_agents, names) if hasattr(self.model, "bind_for_agents") else {name: self.model.bind() for name in names}
                delegation_count = await _thread(self.saves.task_count, save_id, turn_id)
                token_counting_settings, portrait_settings, drawing_preset, network_settings = capability_snapshot
                context = RunContext(
                    save_id=save_id,
                    turn_id=turn_id,
                    models=models,
                    max_delegations=settings.max_delegations,
                    enabled_agents=enabled_agents,
                    blocked_instruction_agents=frozenset(settings.blocked_instruction_agents),
                    always_attach_report_agents=frozenset(settings.always_attach_report_agents) - {"narrator"},
                    force_publish_narrative=settings.force_publish_narrative,
                    force_start_delegation=settings.force_start_delegation,
                    block_coordinator_narrative_read=settings.block_coordinator_narrative_read,
                    prompts=prompts.prompts,
                    task_descriptions=prompts.task_descriptions,
                    capability_descriptions=prompts.capability_descriptions,
                    fixed_responses=prompts.fixed_responses,
                    prefills=prompts.prefills,
                    enabled_configurable_capabilities=frozenset(
                        name for name, enabled in (
                            ("character_portrait_generate", portrait_settings is not None),
                            ("narrative_token_count", token_counting_settings is not None),
                        ) if enabled
                    ),
                    drawing_preset=drawing_preset,
                    portrait_settings=portrait_settings,
                    network_settings=network_settings,
                    token_counting_settings=token_counting_settings,
                    started=time.monotonic(),
                    delegation_count=delegation_count,
                )
                turn = await _thread(self.saves.get_turn, save_id, turn_id)
                await self._emit(save_id, turn_id, "turn.resumed" if resumed else "turn.started", {"player_input": turn.player_input, "delegation_budget": {"max_delegations": settings.max_delegations}})
                root_session = await _thread(self.saves.root_session_id, save_id, turn_id)
                await self.runner.run(AGENTS.main(), root_session, context)
                if (await _thread(self.saves.get_turn, save_id, turn_id)).status != "completed":
                    raise RuntimeError("coordinator 未发布故事")
            except asyncio.CancelledError as error:
                message = str(error) or SHUTDOWN_MESSAGE
                await asyncio.shield(_thread(self.saves.interrupt_tools, save_id, turn_id))
                event = await asyncio.shield(_thread(self.saves.fail_turn, save_id, turn_id, "turn.interrupted", {"message": message}))
                if event:
                    self.events.publish(save_id, event)
                raise
            except Exception as error:
                await _thread(self.saves.interrupt_tools, save_id, turn_id)
                event = await _thread(self.saves.fail_turn, save_id, turn_id, "turn.failed", _redact_image_data({"error_type": type(error).__name__, "message": str(error)}))
                if event:
                    self.events.publish(save_id, event)

    async def shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel(SHUTDOWN_MESSAGE)
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def _execute_capability(self, name: str, arguments: dict[str, Any], context: RunContext, agent: str, session_id: str, part_id: str) -> dict[str, Any]:
        async def emit(kind: str, payload: dict[str, Any]) -> None:
            await self._emit(context.save_id, context.turn_id, kind, {"session_id": session_id, "part_id": part_id, "agent": agent, **payload})

        async def resolve_reports(refs: list[ReportRef]) -> list[dict[str, Any]]:
            root_session_id = await _thread(self.saves.root_session_id, context.save_id, context.turn_id)
            return await self.runner._resolve_report_refs(context.save_id, root_session_id, refs)

        return await self.capabilities.execute(
            name,
            arguments,
            CapabilityContext(
                context.save_id,
                context.turn_id,
                agent,
                emit,
                session_id,
                part_id,
                context.drawing_preset,
                context.portrait_settings,
                context.network_settings,
                context.token_counting_settings,
                resolve_reports if name == "report_read" else None,
            ),
            self.runner._allowed_capabilities(AGENTS.get(agent), context),
        )

    async def _emit(self, save_id: str, turn_id: str | None, event_type: str, payload: dict[str, Any]) -> RuntimeEvent:
        event = await _thread(self.saves.add_event, save_id, turn_id, event_type, _redact_image_data(payload))
        self.events.publish(save_id, event)
        return event
