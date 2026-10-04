from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, StrictBool, model_validator

from .content import character_snapshot_path, content_name, entity_aliases, entity_snapshot_path, entity_type, location_snapshot_path, style_path
from .gemini_thinking import GeminiThinkingLevel, validate_gemini_thinking_level


ProviderName = Literal["openai_compatible", "deepseek", "google_gemini", "anthropic", "xai"]
XaiProtocol = Literal["responses", "chat_completions"]
OpenAiProtocol = Literal["chat_completions", "responses"]
DrawingProviderName = Literal["stable_diffusion_webui", "novelai"]
NovelAIImageModel = Literal["nai-diffusion-5-full", "nai-diffusion-5-curated", "nai-diffusion-4-5-full", "nai-diffusion-4-5-curated"]
NetworkMode = Literal["direct", "custom"]
ContentName = Annotated[str, BeforeValidator(content_name)]
EntitySnapshotPath = Annotated[str, BeforeValidator(entity_snapshot_path)]
CharacterSnapshotPath = Annotated[str, BeforeValidator(character_snapshot_path)]
LocationSnapshotPath = Annotated[str, BeforeValidator(location_snapshot_path)]
EntityType = Annotated[str, BeforeValidator(entity_type)]
EntityAliases = Annotated[list[str], BeforeValidator(entity_aliases)]
StylePath = Annotated[str, BeforeValidator(style_path)]


class AiPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    provider: ProviderName = "deepseek"
    base_url: str = "https://api.deepseek.com"
    api_key: str = ""
    model: str = "deepseek-chat"
    xai_protocol: XaiProtocol = "responses"
    openai_protocol: OpenAiProtocol = "chat_completions"
    timeout_seconds: int = Field(default=120, ge=5, le=600)
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int = Field(default=65_536, ge=1, le=1_048_576)
    thinking_level: GeminiThinkingLevel | None = None

    @model_validator(mode="after")
    def validate_thinking_level(self) -> AiPreset:
        validate_gemini_thinking_level(self.provider, self.model, self.thinking_level)
        return self


class PublicAiPreset(BaseModel):
    id: str
    name: str
    provider: ProviderName
    base_url: str
    model: str
    xai_protocol: XaiProtocol
    openai_protocol: OpenAiProtocol
    timeout_seconds: int
    temperature: float
    top_p: float
    max_tokens: int
    thinking_level: GeminiThinkingLevel | None
    has_api_key: bool
    masked_api_key: str


class PresetWrite(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    provider: ProviderName
    base_url: str = ""
    api_key: str = ""
    model: str = Field(min_length=1, max_length=200)
    xai_protocol: XaiProtocol = "responses"
    openai_protocol: OpenAiProtocol = "chat_completions"
    timeout_seconds: int = Field(default=120, ge=5, le=600)
    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    max_tokens: int = Field(default=65_536, ge=1, le=1_048_576)
    thinking_level: GeminiThinkingLevel | None = None

    @model_validator(mode="after")
    def validate_thinking_level(self) -> PresetWrite:
        validate_gemini_thinking_level(self.provider, self.model, self.thinking_level)
        return self


class ModelListRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    preset_id: str | None = None
    provider: ProviderName
    base_url: str = Field(min_length=1)
    api_key: str = ""
    xai_protocol: XaiProtocol = "responses"
    openai_protocol: OpenAiProtocol = "chat_completions"
    timeout_seconds: int = Field(default=120, ge=5, le=600)


class PresetCollection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    main_preset_id: str
    agent_preset_overrides: dict[str, str] = Field(default_factory=dict)
    agent_streaming: dict[str, bool] = Field(default_factory=dict)
    fallback_enabled: bool = False
    fallback_preset_id: str | None = None
    presets: list[AiPreset]


class PublicPresetCollection(BaseModel):
    main_preset_id: str
    agent_preset_overrides: dict[str, str]
    agent_streaming: dict[str, bool]
    fallback_enabled: bool
    fallback_preset_id: str | None
    presets: list[PublicAiPreset]
    gemini_thinking_levels: dict[str, list[GeminiThinkingLevel]]


class PresetDuplicate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)


class AgentPresetSettingsWrite(BaseModel):
    main_preset_id: str
    agent_preset_overrides: dict[str, str] = Field(default_factory=dict)
    agent_streaming: dict[str, bool] = Field(default_factory=dict)


class AiFallbackSettingsWrite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    preset_id: str | None = None


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_delegations: int = Field(default=20, ge=4)
    context_turns: int = Field(ge=2, strict=True)
    disabled_agents: list[str]
    prefill_agents: list[str] = Field(default_factory=list)
    blocked_instruction_agents: list[str]
    always_attach_report_agents: list[str] = Field(default_factory=list)
    force_publish_narrative: bool = False
    force_start_delegation: bool = False
    block_coordinator_narrative_read: bool = False
    use_compliance_fixed_response: bool = False


class NetworkSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: NetworkMode = "direct"
    proxy_url: str = ""
    proxy_username: str = ""
    proxy_password: str = ""


class PublicNetworkSettings(BaseModel):
    mode: NetworkMode
    proxy_url: str
    proxy_username: str
    has_proxy_password: bool
    masked_proxy_password: str


class NetworkSettingsWrite(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    mode: NetworkMode = "direct"
    proxy_url: str = Field(default="", max_length=2_000)
    proxy_username: str = Field(default="", max_length=500)
    proxy_password: str = Field(default="", max_length=2_000)
    clear_proxy_password: bool = False


class ModelOption(BaseModel):
    id: str
    owned_by: str = ""


class ConnectionTestResult(BaseModel):
    ok: bool
    latency_ms: int
    model: str


class DrawingPreset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    provider: DrawingProviderName = "stable_diffusion_webui"
    base_url: str = "http://127.0.0.1:7860"
    auth_username: str = ""
    auth_password: str = ""
    api_key: str = ""
    model: NovelAIImageModel = "nai-diffusion-5-full"
    timeout_seconds: int = Field(default=300, ge=5, le=600)


class PublicDrawingPreset(BaseModel):
    id: str
    name: str
    provider: DrawingProviderName
    base_url: str
    auth_username: str
    has_auth_password: bool
    has_api_key: bool
    model: NovelAIImageModel
    timeout_seconds: int


class DrawingPresetWrite(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    provider: DrawingProviderName = "stable_diffusion_webui"
    base_url: str = ""
    auth_username: str = Field(default="", max_length=200)
    auth_password: str = Field(default="", max_length=2_000)
    clear_auth_password: bool = False
    api_key: str = Field(default="", max_length=2_000)
    clear_api_key: bool = False
    model: NovelAIImageModel = "nai-diffusion-5-full"
    timeout_seconds: int = Field(default=300, ge=5, le=600)


class DrawingPresetCollection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_preset_id: str | None = None
    presets: list[DrawingPreset] = Field(default_factory=list)


class PublicDrawingPresetCollection(BaseModel):
    default_preset_id: str | None
    presets: list[PublicDrawingPreset]


class CharacterPortraitGenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    checkpoint: str = Field(default="", max_length=500)
    sampler_name: str = Field(default="Euler", min_length=1, max_length=200)
    scheduler: str = Field(default="Automatic", min_length=1, max_length=200)
    novelai_sampler: Literal["k_euler_ancestral", "k_euler", "k_dpmpp_2m", "k_dpmpp_sde"] = "k_euler_ancestral"
    steps: int = Field(default=20, ge=1, le=150)
    cfg_scale: float = Field(default=7.0, ge=0, le=30)
    width: int = Field(default=512, ge=64, le=2048, multiple_of=8)
    height: int = Field(default=512, ge=64, le=2048, multiple_of=8)
    positive_prompt: str = Field(default="", max_length=8_000)
    negative_prompt: str = Field(default="", max_length=8_000)


class NarrativeTokenCountingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_tokens: int = Field(default=4_000, ge=1, le=1_048_576, strict=True)


class CapabilitySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_portrait_generation: CharacterPortraitGenerationSettings = Field(
        default_factory=CharacterPortraitGenerationSettings
    )
    narrative_token_counting: NarrativeTokenCountingSettings = Field(
        default_factory=NarrativeTokenCountingSettings
    )


class DrawingResourceOption(BaseModel):
    id: str
    label: str


class DrawingResources(BaseModel):
    active_checkpoint: str = ""
    checkpoints: list[DrawingResourceOption]
    samplers: list[DrawingResourceOption]
    schedulers: list[DrawingResourceOption]


class DrawingConnectionTestResult(BaseModel):
    ok: bool
    latency_ms: int
    active_checkpoint: str = ""


class DrawingTestRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content_prompt: str = Field(min_length=1, max_length=8_000)


class DrawingDefaultWrite(BaseModel):
    preset_id: str | None = None


class DrawingTestResult(BaseModel):
    image_data_url: str
    width: int
    height: int
    seed: int | None = None


class ScenarioSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName
    description: str = Field(default="", max_length=4_000)


class StyleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: StylePath
    name: ContentName
    description: str = Field(default="", max_length=4_000)
    enabled: bool


class StyleDocument(StyleSummary):
    content: str = Field(max_length=100_000)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(default="", max_length=4_000)
    opening: str = Field(default="", max_length=20_000)


class ContentSourceCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName


class ContentSourceUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(max_length=4_000)


class ContentStyleCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName


class ContentStyleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(max_length=4_000)
    content: str = Field(max_length=100_000)


class ContentStyleEnabledUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: StrictBool


class ContentRename(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName


class ContentScenarioCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName


class ContentScenarioUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(max_length=4_000)
    opening: str = Field(max_length=20_000)


class ContentEntityCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName
    type: EntityType


class ContentEntityUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(max_length=4_000)
    aliases: EntityAliases
    visibility: Literal["world_truth"]
    required: StrictBool
    content: str = Field(max_length=100_000)

class ContentEntityMove(ContentRename):
    type: EntityType


class SaveEntityUpdate(ContentEntityUpdate):
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")


class SaveEntityMove(ContentEntityMove):
    revision: str = Field(pattern=r"^[0-9a-f]{64}$")


class SaveEntityDelete(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: str = Field(pattern=r"^[0-9a-f]{64}$")


class ScenarioRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_kind: Literal["world", "mod"]
    source_name: ContentName
    name: ContentName


class ScenarioOption(ScenarioRef):
    description: str = Field(default="", max_length=4_000)

    def as_ref(self) -> ScenarioRef:
        return ScenarioRef(source_kind=self.source_kind, source_name=self.source_name, name=self.name)

    def matches(self, other: ScenarioRef) -> bool:
        return self.as_ref() == other


class WorldSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName
    description: str = ""
    scenarios: list[ScenarioSummary] = Field(default_factory=list)


class ModSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName
    description: str = Field(default="", max_length=4_000)
    scenarios: list[ScenarioSummary] = Field(default_factory=list)


class ModPage(BaseModel):
    mods: list[ModSummary]
    page: int
    page_size: int
    total: int
    total_pages: int


class LockedModSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: ContentName
    source_hash: str


class SaveNarrativeSettings(BaseModel):
    play_mode: Literal["roleplay", "director"] = "roleplay"
    narrative_person: Literal["first", "second", "third"] = "second"
    language: Literal["zh", "ja", "en"] = "zh"
    other_requirements: str = Field(default="", max_length=4_000)


class SaveCreate(SaveNarrativeSettings):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    world_name: ContentName
    name: str = Field(min_length=1, max_length=80)
    scenario: ScenarioRef | None = None
    mod_names: list[ContentName] = Field(default_factory=list, max_length=32)


class SaveRename(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)


class SaveBranch(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=80)


class SaveSummary(SaveNarrativeSettings):
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    source_world_name: ContentName
    source_world_hash: str
    scenario: ScenarioRef | None = None
    opening: str = ""
    mods: list[LockedModSummary] = Field(default_factory=list)
    created_at: str
    last_played_at: str
    branch_source_save_id: str | None = None
    branch_source_turn_id: str | None = None
    branch_source_save_name: str | None = None
    branch_source_turn_number: int | None = None


class SavePage(BaseModel):
    saves: list[SaveSummary]
    page: int
    page_size: int
    total: int
    total_pages: int


class TurnCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=20_000)


class TurnImageSummary(BaseModel):
    id: str
    original_name: str
    mime_type: str
    byte_size: int
    width: int
    height: int
    position: int
    content_url: str


class GeneratedImageSummary(BaseModel):
    id: str
    agent: str
    mime_type: str
    width: int
    height: int
    content_url: str


class TurnSummary(BaseModel):
    id: str
    player_input: str
    narrative: str | None
    status: str
    created_at: str
    completed_at: str | None
    images: list[TurnImageSummary] = Field(default_factory=list)
    generated_images: list[GeneratedImageSummary] = Field(default_factory=list)


class ModelResult(BaseModel):
    content: str
    raw_response: dict[str, Any]
    reasoning: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list["ToolCall"] = Field(default_factory=list)
    provider_content: dict[str, Any] | None = None


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: str | dict[str, Any] = "{}"


class RuntimeEvent(BaseModel):
    id: int
    turn_id: str | None
    type: str
    payload: dict[str, Any]
    created_at: str


class EventPage(BaseModel):
    events: list[RuntimeEvent]
    page: int
    page_size: int
    total: int
    total_pages: int


class TracePage(EventPage):
    cursor: int
