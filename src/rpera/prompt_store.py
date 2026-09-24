from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from .agents import ComplianceReviewResult, RECOMMENDED_SEQUENCE_PLACEHOLDER, AgentRegistry
from .models import SaveNarrativeSettings


SAVE_NARRATIVE_SETTINGS_PLACEHOLDER = "{save_narrative_settings}"


def render_save_narrative_settings(template: str, settings: SaveNarrativeSettings) -> str:
    mode = {
        "roleplay": "角色扮演模式：玩家作为角色存在于故事中。",
        "director": "导演模式：玩家不作为故事角色存在；玩家输入用于指导故事发展，不要据此把玩家写成角色。",
    }[settings.play_mode]
    person = {"first": "第一人称", "second": "第二人称", "third": "第三人称"}[settings.narrative_person]
    language = {"zh": "中文", "ja": "日语", "en": "英语"}[settings.language]
    requirements = json.dumps(settings.other_requirements, ensure_ascii=False) if settings.other_requirements else "无"
    person_instruction = f"{person}。"
    if settings.play_mode == "director":
        person_instruction += "不预设人称所指的故事人物；如需指定，以其他要求为准。"
    instructions = (
        f"本存档的叙事设置：\n- 参与模式：{mode}\n- 叙事人称：{person_instruction}\n"
        f"- 叙事语言：{language}。\n- 其他要求：{requirements}"
    )
    return template.replace(SAVE_NARRATIVE_SETTINGS_PLACEHOLDER, instructions)


class PromptConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True)
class PromptSnapshot:
    prompts: dict[str, str]
    task_descriptions: dict[str, str]
    capability_descriptions: dict[str, str]
    fixed_responses: dict[str, ComplianceReviewResult]
    prefills: dict[str, str]


class PromptStore:
    def __init__(
        self,
        root: Path,
        agents: AgentRegistry,
        fixed_response_root: Path | None = None,
        prefill_root: Path | None = None,
    ) -> None:
        self.root = root
        self.agents = agents
        self.fixed_response_root = fixed_response_root or root.parent / "mock_responses"
        self.prefill_root = prefill_root or root.parent / "prefills"

    def load(
        self,
        enabled_agents: frozenset[str],
        prefill_agents: frozenset[str] = frozenset(),
    ) -> PromptSnapshot:
        prompts: dict[str, str] = {}
        task_descriptions: dict[str, str] = {}
        capability_descriptions: dict[str, str] = {}
        fixed_responses: dict[str, ComplianceReviewResult] = {}
        prefills: dict[str, str] = {}
        for spec in self.agents.all():
            if spec.name not in enabled_agents:
                continue
            path = self.root / spec.prompt_file
            prompts[spec.name] = self._read_text(path, "Agent Prompt")
            if spec.task_description_file is not None:
                description_path = self.root / spec.task_description_file
                description = self._read_text(description_path, "Agent task 描述")
                if len(description) > 4_000:
                    raise PromptConfigurationError(f"Agent task 描述不能超过 4000 字符：{description_path}")
                task_descriptions[spec.name] = description
            if spec.fixed_response_file is not None:
                response_path = self.fixed_response_root / spec.fixed_response_file
                response = self._read_text(response_path, "Agent 固定响应")
                try:
                    fixed_responses[spec.name] = ComplianceReviewResult.model_validate_json(response)
                except ValidationError as error:
                    raise PromptConfigurationError(f"Agent 固定响应格式无效：{response_path}：{error}") from error
            if spec.name in prefill_agents and spec.fixed_response_file is None:
                prefill_path = self.prefill_root / f"{spec.name}.md"
                prefills[spec.name] = self._read_text(prefill_path, "Agent 尾部续写")

        capability_names = {
            name
            for spec in self.agents.all()
            if spec.name in enabled_agents
            for name in spec.allowed_capabilities
        }
        if "story_summarizer" in enabled_agents:
            capability_names.add("story_summary_read")
        for name in sorted(capability_names):
            description_path = self.root / "capabilities" / f"{name}.md"
            description = self._read_text(description_path, "Capability 描述")
            if len(description) > 4_000:
                raise PromptConfigurationError(f"Capability 描述不能超过 4000 字符：{description_path}")
            capability_descriptions[name] = description

        coordinator = prompts[self.agents.main().name]
        count = coordinator.count(RECOMMENDED_SEQUENCE_PLACEHOLDER)
        if count != 1:
            raise PromptConfigurationError(
                f"主代理 Prompt 必须且只能包含一个 {RECOMMENDED_SEQUENCE_PLACEHOLDER} 占位符"
            )
        for name in ("narrator", "consistency_checker"):
            if name in prompts and prompts[name].count(SAVE_NARRATIVE_SETTINGS_PLACEHOLDER) != 1:
                raise PromptConfigurationError(
                    f"{name} Prompt 必须且只能包含一个 {SAVE_NARRATIVE_SETTINGS_PLACEHOLDER} 占位符"
                )
        return PromptSnapshot(prompts, task_descriptions, capability_descriptions, fixed_responses, prefills)

    @staticmethod
    def _read_text(path: Path, label: str) -> str:
        if not path.is_file():
            raise PromptConfigurationError(f"{label}文件不存在：{path}")
        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as error:
            raise PromptConfigurationError(f"{label}文件不是有效的 UTF-8：{path}") from error
        except OSError as error:
            raise PromptConfigurationError(f"无法读取{label}文件：{path}") from error
        if not content:
            raise PromptConfigurationError(f"{label}文件不能为空：{path}")
        return content
