from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from threading import Lock
from urllib.parse import urlparse

from .agents import AGENTS, agent_names
from .gemini_thinking import supported_gemini_thinking_levels
from .models import (
    AgentPresetSettingsWrite,
    AiFallbackSettingsWrite,
    AiPreset,
    CapabilitySettings,
    PresetCollection,
    PresetWrite,
    PublicAiPreset,
    PublicPresetCollection,
    RuntimeSettings,
    NetworkSettings,
    NetworkSettingsWrite,
    PublicNetworkSettings,
    DrawingPreset,
    DrawingPresetCollection,
    DrawingPresetWrite,
    ModelListRequest,
    PublicDrawingPreset,
    PublicDrawingPresetCollection,
)


class PresetStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "config" / "ai_presets.json"
        self._write_lock = Lock()

    def public(self) -> PublicPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            return PublicPresetCollection(
                main_preset_id=collection.main_preset_id,
                agent_preset_overrides=collection.agent_preset_overrides,
                agent_streaming=collection.agent_streaming,
                fallback_enabled=collection.fallback_enabled,
                fallback_preset_id=collection.fallback_preset_id,
                presets=[self._public_preset(preset) for preset in collection.presets],
                gemini_thinking_levels=supported_gemini_thinking_levels(),
            )

    def resolve(self, agent_names: tuple[str, ...]) -> dict[str, AiPreset]:
        with self._write_lock:
            collection = self._load_unlocked()
            return {
                name: self._find(
                    collection, collection.agent_preset_overrides.get(name, collection.main_preset_id)
                ).model_copy(deep=True)
                for name in agent_names
            }

    def resolve_with_streaming(self, agent_names: tuple[str, ...]) -> dict[str, tuple[AiPreset, bool]]:
        with self._write_lock:
            collection = self._load_unlocked()
            return {
                name: (
                    self._find(collection, collection.agent_preset_overrides.get(name, collection.main_preset_id)).model_copy(deep=True),
                    collection.agent_streaming.get(name, False),
                )
                for name in agent_names
            }

    def resolve_with_fallback(self, names: tuple[str, ...]) -> dict[str, tuple[AiPreset, AiPreset | None, bool]]:
        with self._write_lock:
            collection = self._load_unlocked()
            fallback = self._find(collection, collection.fallback_preset_id) if collection.fallback_enabled and collection.fallback_preset_id else None
            return {
                name: (
                    self._find(collection, collection.agent_preset_overrides.get(name, collection.main_preset_id)).model_copy(deep=True),
                    fallback.model_copy(deep=True) if fallback else None,
                    collection.agent_streaming.get(name, False),
                )
                for name in names
            }

    def update_fallback_settings(self, request: AiFallbackSettingsWrite) -> PublicPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            if request.enabled and request.preset_id is None:
                raise ValueError("启用备用 AI 时必须选择 AI Preset")
            if request.preset_id is not None:
                self._find(collection, request.preset_id)
            collection.fallback_enabled = request.enabled
            collection.fallback_preset_id = request.preset_id
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def get(self, preset_id: str) -> AiPreset:
        with self._write_lock:
            collection = self._load_unlocked()
            return self._find(collection, preset_id).model_copy(deep=True)

    def model_list_preset(self, request: ModelListRequest) -> AiPreset:
        with self._write_lock:
            api_key = request.api_key
            name = "未保存的 AI Preset"
            if request.preset_id is not None:
                current = self._find(self._load_unlocked(), request.preset_id)
                name = current.name
                if not api_key and request.provider == current.provider:
                    api_key = current.api_key
            return AiPreset(
                id=request.preset_id or "unsaved",
                name=name,
                provider=request.provider,
                base_url=request.base_url.rstrip("/"),
                api_key=api_key,
                model="",
                xai_protocol=request.xai_protocol,
                openai_protocol=request.openai_protocol,
                timeout_seconds=request.timeout_seconds,
            )

    def create(self, request: PresetWrite) -> PublicPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            preset = self._from_write(str(uuid.uuid4()), request)
            collection.presets.append(preset)
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def update(self, preset_id: str, request: PresetWrite) -> PublicPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            current = self._find(collection, preset_id)
            api_key = request.api_key
            if not api_key and request.provider == current.provider:
                api_key = current.api_key
            replacement = self._from_write(preset_id, request, api_key=api_key)
            index = collection.presets.index(current)
            collection.presets[index] = replacement
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def duplicate(self, preset_id: str, name: str) -> PublicPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            source = self._find(collection, preset_id)
            duplicate = source.model_copy(update={"id": str(uuid.uuid4()), "name": name}, deep=True)
            collection.presets.append(duplicate)
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def delete(self, preset_id: str) -> PublicPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            preset = self._find(collection, preset_id)
            if len(collection.presets) == 1:
                raise ValueError("至少保留一个 AI Preset")
            collection.presets.remove(preset)
            replacement_id = collection.presets[0].id
            if collection.main_preset_id == preset_id:
                collection.main_preset_id = replacement_id
            collection.agent_preset_overrides = {
                name: replacement_id if assigned_id == preset_id else assigned_id
                for name, assigned_id in collection.agent_preset_overrides.items()
            }
            if collection.fallback_preset_id == preset_id:
                collection.fallback_preset_id = None
                collection.fallback_enabled = False
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def update_agent_settings(self, request: AgentPresetSettingsWrite) -> PublicPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            known_agents = set(agent_names()) - {"coordinator"}
            unknown_agents = set(request.agent_preset_overrides) - known_agents
            if unknown_agents:
                raise ValueError(f"未知 Agent：{', '.join(sorted(unknown_agents))}")
            unknown_streaming_agents = set(request.agent_streaming) - set(agent_names())
            if unknown_streaming_agents:
                raise ValueError(f"未知 Agent：{', '.join(sorted(unknown_streaming_agents))}")
            self._find(collection, request.main_preset_id)
            for preset_id in request.agent_preset_overrides.values():
                self._find(collection, preset_id)
            collection.main_preset_id = request.main_preset_id
            collection.agent_preset_overrides = request.agent_preset_overrides
            collection.agent_streaming = request.agent_streaming
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def _load_unlocked(self) -> PresetCollection:
        if self.path.exists():
            collection = PresetCollection.model_validate_json(self.path.read_text(encoding="utf-8"))
            self._validate_collection(collection)
            return collection
        preset = AiPreset(id=str(uuid.uuid4()), name="默认配置")
        collection = PresetCollection(main_preset_id=preset.id, presets=[preset])
        self._write_unlocked(collection)
        return collection

    def _write_unlocked(self, collection: PresetCollection) -> None:
        self._validate_collection(collection)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4()}.tmp")
        temporary.write_text(
            json.dumps(collection.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    @staticmethod
    def _find(collection: PresetCollection, preset_id: str) -> AiPreset:
        for preset in collection.presets:
            if preset.id == preset_id:
                return preset
        raise KeyError(f"未知 AI Preset：{preset_id}")

    @staticmethod
    def _validate_collection(collection: PresetCollection) -> None:
        if not collection.presets:
            raise ValueError("AI Preset 集合不能为空")
        ids = [preset.id for preset in collection.presets]
        if len(ids) != len(set(ids)):
            raise ValueError("AI Preset ID 重复")
        if collection.main_preset_id not in ids:
            raise ValueError("主代理的 AI Preset 不存在")
        if any(preset_id not in ids for preset_id in collection.agent_preset_overrides.values()):
            raise ValueError("子代理的 AI Preset 不存在")
        if collection.fallback_preset_id is not None and collection.fallback_preset_id not in ids:
            raise ValueError("备用 AI Preset 不存在")
        if collection.fallback_enabled and collection.fallback_preset_id is None:
            raise ValueError("启用备用 AI 时必须选择 AI Preset")

    @staticmethod
    def _base_url(request: PresetWrite) -> str:
        if not request.base_url:
            raise ValueError("AI Preset 的 Base URL 不能为空")
        return request.base_url.rstrip("/")

    @classmethod
    def _from_write(cls, preset_id: str, request: PresetWrite, api_key: str | None = None) -> AiPreset:
        return AiPreset(
            id=preset_id,
            name=request.name,
            provider=request.provider,
            base_url=cls._base_url(request),
            api_key=request.api_key if api_key is None else api_key,
            model=request.model,
            xai_protocol=request.xai_protocol,
            openai_protocol=request.openai_protocol,
            timeout_seconds=request.timeout_seconds,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            thinking_level=request.thinking_level,
        )

    @classmethod
    def _public_collection(cls, collection: PresetCollection) -> PublicPresetCollection:
        return PublicPresetCollection(
            main_preset_id=collection.main_preset_id,
            agent_preset_overrides=collection.agent_preset_overrides,
            agent_streaming=collection.agent_streaming,
            fallback_enabled=collection.fallback_enabled,
            fallback_preset_id=collection.fallback_preset_id,
            presets=[cls._public_preset(preset) for preset in collection.presets],
            gemini_thinking_levels=supported_gemini_thinking_levels(),
        )

    @staticmethod
    def _public_preset(preset: AiPreset) -> PublicAiPreset:
        key = preset.api_key
        masked = "" if not key else f"{key[:3]}***{key[-3:]}" if len(key) > 6 else "***"
        return PublicAiPreset(
            id=preset.id,
            name=preset.name,
            provider=preset.provider,
            base_url=preset.base_url,
            model=preset.model,
            xai_protocol=preset.xai_protocol,
            openai_protocol=preset.openai_protocol,
            timeout_seconds=preset.timeout_seconds,
            temperature=preset.temperature,
            top_p=preset.top_p,
            max_tokens=preset.max_tokens,
            thinking_level=preset.thinking_level,
            has_api_key=bool(key),
            masked_api_key=masked,
        )


class DrawingPresetStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "config" / "drawing_presets.json"
        self._write_lock = Lock()

    def public(self) -> PublicDrawingPresetCollection:
        with self._write_lock:
            return self._public_collection(self._load_unlocked())

    def get(self, preset_id: str) -> DrawingPreset:
        with self._write_lock:
            return self._find(self._load_unlocked(), preset_id).model_copy(deep=True)

    def get_default(self) -> DrawingPreset | None:
        with self._write_lock:
            collection = self._load_unlocked()
            if collection.default_preset_id is None:
                return None
            return self._find(collection, collection.default_preset_id).model_copy(deep=True)

    def create(self, request: DrawingPresetWrite) -> PublicDrawingPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            preset = self._from_write(str(uuid.uuid4()), request)
            collection.presets.append(preset)
            if collection.default_preset_id is None:
                collection.default_preset_id = preset.id
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def update(self, preset_id: str, request: DrawingPresetWrite) -> PublicDrawingPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            current = self._find(collection, preset_id)
            base_url = self._base_url(request)
            endpoint_changed = base_url != current.base_url or request.auth_username != current.auth_username or request.provider != current.provider
            password = request.auth_password
            if request.clear_auth_password:
                password = ""
            elif not password and not endpoint_changed:
                password = current.auth_password
            if not password and endpoint_changed:
                request = request.model_copy(update={"auth_username": ""})
            api_key = request.api_key
            if request.clear_api_key:
                api_key = ""
            elif not api_key and base_url == current.base_url and request.provider == current.provider:
                api_key = current.api_key
            replacement = self._from_write(preset_id, request, password=password, api_key=api_key)
            collection.presets[collection.presets.index(current)] = replacement
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def duplicate(self, preset_id: str, name: str) -> PublicDrawingPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            source = self._find(collection, preset_id)
            duplicate = source.model_copy(update={"id": str(uuid.uuid4()), "name": name}, deep=True)
            collection.presets.append(duplicate)
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def delete(self, preset_id: str) -> PublicDrawingPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            preset = self._find(collection, preset_id)
            collection.presets.remove(preset)
            if collection.default_preset_id == preset_id:
                collection.default_preset_id = collection.presets[0].id if collection.presets else None
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def set_default(self, preset_id: str | None) -> PublicDrawingPresetCollection:
        with self._write_lock:
            collection = self._load_unlocked()
            if preset_id is not None:
                self._find(collection, preset_id)
            collection.default_preset_id = preset_id
            self._write_unlocked(collection)
            return self._public_collection(collection)

    def _load_unlocked(self) -> DrawingPresetCollection:
        if not self.path.exists():
            return DrawingPresetCollection()
        collection = DrawingPresetCollection.model_validate_json(self.path.read_text(encoding="utf-8"))
        self._validate_collection(collection)
        return collection

    def _write_unlocked(self, collection: DrawingPresetCollection) -> None:
        self._validate_collection(collection)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4()}.tmp")
        temporary.write_text(json.dumps(collection.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    @staticmethod
    def _find(collection: DrawingPresetCollection, preset_id: str) -> DrawingPreset:
        for preset in collection.presets:
            if preset.id == preset_id:
                return preset
        raise KeyError(f"未知绘画 Preset：{preset_id}")

    @staticmethod
    def _validate_collection(collection: DrawingPresetCollection) -> None:
        ids = [preset.id for preset in collection.presets]
        if len(ids) != len(set(ids)):
            raise ValueError("绘画 Preset ID 重复")
        if collection.default_preset_id is not None and collection.default_preset_id not in ids:
            raise ValueError("默认绘画 Preset 不存在")

    @staticmethod
    def _base_url(request: DrawingPresetWrite) -> str:
        if not request.base_url:
            raise ValueError("绘画 Preset 的 Base URL 不能为空")
        base_url = request.base_url.rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("绘画 Preset 的 Base URL 必须是无认证信息的 HTTP URL")
        return base_url

    @classmethod
    def _from_write(cls, preset_id: str, request: DrawingPresetWrite, password: str | None = None, api_key: str | None = None) -> DrawingPreset:
        if bool(request.auth_username) != bool(password if password is not None else request.auth_password):
            raise ValueError("Basic Auth 的用户名和密码必须同时填写")
        if request.provider == "novelai" and request.auth_username:
            raise ValueError("NovelAI 使用 API Token，不支持 Basic Auth")
        if request.provider == "stable_diffusion_webui" and (api_key if api_key is not None else request.api_key):
            raise ValueError("WebUI 不使用 NovelAI API Token")
        return DrawingPreset(
            id=preset_id,
            name=request.name,
            provider=request.provider,
            base_url=cls._base_url(request),
            auth_username=request.auth_username,
            auth_password=("" if request.clear_auth_password else request.auth_password) if password is None else password,
            api_key=("" if request.clear_api_key else request.api_key) if api_key is None else api_key,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
        )

    @classmethod
    def _public_collection(cls, collection: DrawingPresetCollection) -> PublicDrawingPresetCollection:
        return PublicDrawingPresetCollection(
            default_preset_id=collection.default_preset_id,
            presets=[cls._public_preset(preset) for preset in collection.presets],
        )

    @staticmethod
    def _public_preset(preset: DrawingPreset) -> PublicDrawingPreset:
        return PublicDrawingPreset(
            id=preset.id,
            name=preset.name,
            provider=preset.provider,
            base_url=preset.base_url,
            auth_username=preset.auth_username,
            has_auth_password=bool(preset.auth_password),
            has_api_key=bool(preset.api_key),
            model=preset.model,
            timeout_seconds=preset.timeout_seconds,
        )


class CapabilitySettingsStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "config" / "capability_settings.json"
        self._write_lock = Lock()

    def get(self) -> CapabilitySettings:
        with self._write_lock:
            if self.path.exists():
                return CapabilitySettings.model_validate_json(self.path.read_text(encoding="utf-8"))
            settings = CapabilitySettings()
            self._write_unlocked(settings)
            return settings.model_copy(deep=True)

    def update(self, settings: CapabilitySettings) -> CapabilitySettings:
        with self._write_lock:
            self._write_unlocked(settings)
            return settings.model_copy(deep=True)

    def _write_unlocked(self, settings: CapabilitySettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4()}.tmp")
        temporary.write_text(
            json.dumps(settings.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)


class RuntimeSettingsStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "config" / "runtime_settings.json"
        self._write_lock = Lock()

    def get(self) -> RuntimeSettings:
        with self._write_lock:
            if self.path.exists():
                settings = RuntimeSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
                self._validate(settings)
                return settings
            settings = RuntimeSettings(context_turns=4, disabled_agents=[], blocked_instruction_agents=[])
            self._write_unlocked(settings)
            return settings

    def update(self, settings: RuntimeSettings) -> RuntimeSettings:
        with self._write_lock:
            self._validate(settings)
            self._write_unlocked(settings)
            return settings

    def _write_unlocked(self, settings: RuntimeSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4()}.tmp")
        temporary.write_text(
            json.dumps(settings.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    @staticmethod
    def _validate(settings: RuntimeSettings) -> None:
        if len(settings.disabled_agents) != len(set(settings.disabled_agents)):
            raise ValueError("禁用 Agent 不能重复")
        disableable = {spec.name for spec in AGENTS.disableable_children()}
        invalid = set(settings.disabled_agents) - disableable
        if invalid:
            raise ValueError(f"Agent 不存在或不能禁用：{', '.join(sorted(invalid))}")
        if len(settings.prefill_agents) != len(set(settings.prefill_agents)):
            raise ValueError("尾部续写 Agent 不能重复")
        invalid_prefill = set(settings.prefill_agents) - set(agent_names())
        if invalid_prefill:
            raise ValueError(f"尾部续写 Agent 不存在：{', '.join(sorted(invalid_prefill))}")
        if len(settings.blocked_instruction_agents) != len(set(settings.blocked_instruction_agents)):
            raise ValueError("屏蔽主代理指令的 Agent 不能重复")
        children = {spec.name for spec in AGENTS.children()}
        invalid_blocked = set(settings.blocked_instruction_agents) - children
        if invalid_blocked:
            raise ValueError(f"屏蔽主代理指令的 Agent 不存在：{', '.join(sorted(invalid_blocked))}")
        if len(settings.always_attach_report_agents) != len(set(settings.always_attach_report_agents)):
            raise ValueError("强制附加报告的 Agent 不能重复")
        report_producers = {spec.name for spec in AGENTS.report_producing_children()}
        invalid_report_producers = set(settings.always_attach_report_agents) - report_producers
        if invalid_report_producers:
            raise ValueError(f"强制附加报告的 Agent 不存在或不生成报告：{', '.join(sorted(invalid_report_producers))}")


class NetworkSettingsStore:
    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "config" / "network_settings.json"
        self._write_lock = Lock()

    def get(self) -> NetworkSettings:
        with self._write_lock:
            if self.path.exists():
                settings = NetworkSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
                self._validate(settings)
                return settings
            settings = NetworkSettings()
            self._write_unlocked(settings)
            return settings

    def public(self) -> PublicNetworkSettings:
        settings = self.get()
        return PublicNetworkSettings(
            mode=settings.mode,
            proxy_url=settings.proxy_url,
            proxy_username=settings.proxy_username,
            has_proxy_password=bool(settings.proxy_password),
            masked_proxy_password="***" if settings.proxy_password else "",
        )

    def update(self, payload: NetworkSettingsWrite) -> PublicNetworkSettings:
        with self._write_lock:
            current = (
                NetworkSettings.model_validate_json(self.path.read_text(encoding="utf-8"))
                if self.path.exists()
                else NetworkSettings()
            )
            password = "" if payload.clear_proxy_password else payload.proxy_password or current.proxy_password
            settings = NetworkSettings(
                mode=payload.mode,
                proxy_url=payload.proxy_url,
                proxy_username=payload.proxy_username,
                proxy_password=password,
            )
            self._validate(settings)
            self._write_unlocked(settings)
            return PublicNetworkSettings(
                mode=settings.mode,
                proxy_url=settings.proxy_url,
                proxy_username=settings.proxy_username,
                has_proxy_password=bool(settings.proxy_password),
                masked_proxy_password="***" if settings.proxy_password else "",
            )

    def _write_unlocked(self, settings: NetworkSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4()}.tmp")
        temporary.write_text(
            json.dumps(settings.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    @staticmethod
    def _validate(settings: NetworkSettings) -> None:
        if settings.mode != "custom":
            return
        if not settings.proxy_url:
            raise ValueError("自定义代理模式必须填写代理地址")
        parsed = urlparse(settings.proxy_url)
        if parsed.scheme not in {"http", "https", "socks5", "socks5h"}:
            raise ValueError("代理地址仅支持 http、https、socks5 或 socks5h")
        if not parsed.hostname:
            raise ValueError("代理地址缺少主机名")
        try:
            port = parsed.port
        except ValueError as error:
            raise ValueError("代理地址端口无效") from error
        if port is None:
            raise ValueError("代理地址必须包含端口")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("请在单独的代理用户名和密码字段中填写认证信息")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("代理地址不能包含路径、查询参数或片段")
