from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.parse import urlencode

import httpx

from .agents import agent_names
from .anthropic import parse_response as parse_anthropic_response
from .anthropic import parse_stream as parse_anthropic_stream
from .anthropic import request_payload as anthropic_request_payload
from .config import NetworkSettingsStore, PresetStore
from .gemini import parse_response as parse_gemini_response
from .gemini import parse_stream as parse_gemini_stream
from .gemini import request_payload as gemini_request_payload
from .message_compat import content_blocks
from .models import AiPreset, ModelOption, ModelResult, NetworkSettings, ToolCall
from .network import async_client_options
from .xai import parse_response as parse_xai_response
from .xai import parse_stream as parse_xai_stream
from .xai import request_payload as xai_request_payload


class ModelClient(Protocol):
    def bind(self) -> ModelClient: ...

    def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, ModelClient]: ...

    def metadata(self) -> dict[str, str | int | float]: ...

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult: ...


class ConfiguredModelClient:
    def __init__(self, presets: PresetStore, network_settings: NetworkSettingsStore | None = None) -> None:
        self.presets = presets
        self.network_settings = network_settings

    def bind_for_agents(self, names: tuple[str, ...] = agent_names()) -> dict[str, ModelClient]:
        network = self.network_settings.get() if self.network_settings else NetworkSettings()
        return {
            name: BoundModelClient(preset, network_settings=network, streaming=streaming)
            for name, (preset, streaming) in self.presets.resolve_with_streaming(names).items()
        }

    def bind(self) -> ModelClient:
        return self.bind_for_agents()["coordinator"]

    def metadata(self) -> dict[str, str | int | float]:
        return self.bind_for_agents()["coordinator"].metadata()

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        return await self.bind_for_agents()["coordinator"].complete(system_prompt, messages, tools)


class BoundModelClient:
    def __init__(
        self,
        preset: AiPreset,
        transport: httpx.AsyncBaseTransport | None = None,
        network_settings: NetworkSettings | None = None,
        streaming: bool = False,
    ) -> None:
        self.preset = preset
        self.transport = transport
        self.network_settings = network_settings or NetworkSettings()
        self.streaming = streaming

    def bind_for_agents(self, names: tuple[str, ...]) -> dict[str, ModelClient]:
        return {name: self for name in names}

    def bind(self) -> ModelClient:
        return self

    def metadata(self) -> dict[str, str | int | float]:
        return {
            "preset_id": self.preset.id,
            "preset_name": self.preset.name,
            "provider": self.preset.provider,
            "base_url": self.preset.base_url,
            "model": self.preset.model,
            "xai_protocol": self.preset.xai_protocol,
            "timeout_seconds": self.preset.timeout_seconds,
            "temperature": self.preset.temperature,
            "top_p": self.preset.top_p,
            "max_tokens": self.preset.max_tokens,
            "streaming": self.streaming,
        }

    async def complete(self, system_prompt: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ModelResult:
        preset = self.preset
        if not preset.api_key:
            raise RuntimeError(f"AI Preset“{preset.name}”尚未设置 API Key")
        if preset.provider == "anthropic":
            payload = anthropic_request_payload(
                system_prompt,
                messages,
                tools,
                model=preset.model,
                temperature=preset.temperature,
                top_p=preset.top_p,
                max_tokens=preset.max_tokens,
                stream=self.streaming,
            )
            if self.streaming:
                events = await self._request_native_stream("messages", payload, "Anthropic")
                return parse_anthropic_stream(events)
            return parse_anthropic_response(await self._request_json("POST", "messages", payload))
        if preset.provider == "xai" and preset.xai_protocol == "responses":
            payload = xai_request_payload(
                system_prompt,
                messages,
                tools,
                model=preset.model,
                temperature=preset.temperature,
                top_p=preset.top_p,
                max_tokens=preset.max_tokens,
                stream=self.streaming,
            )
            if self.streaming:
                events = await self._request_native_stream("responses", payload, "xAI")
                return parse_xai_stream(events)
            return parse_xai_response(await self._request_json("POST", "responses", payload))
        if preset.provider == "google_gemini":
            payload = gemini_request_payload(
                system_prompt,
                messages,
                tools,
                temperature=preset.temperature,
                top_p=preset.top_p,
                max_tokens=preset.max_tokens,
            )
            if self.streaming:
                return await self._request_gemini_stream(payload)
            raw = await self._request_json("POST", f"{self._gemini_model()}:generateContent", payload)
            return parse_gemini_response(raw)
        provider_messages: list[dict[str, Any]] = []
        for message in messages:
            provider_message = {
                key: value for key, value in message.items() if not key.startswith("_")
            }
            tool_images: list[dict[str, Any]] = []
            if isinstance(provider_message.get("content"), list):
                blocks = content_blocks(provider_message["content"])
                if provider_message.get("role") == "tool":
                    provider_message["content"] = "".join(
                        block["text"] for block in blocks if block["type"] == "text"
                    )
                    tool_images = [block for block in blocks if block["type"] == "image_url"]
                else:
                    provider_message["content"] = blocks
            if preset.provider == "deepseek" and message.get("role") == "assistant":
                if tools and "_reasoning_content" in message:
                    reasoning = message["_reasoning_content"]
                    provider_message["reasoning_content"] = reasoning if isinstance(reasoning, str) else ""
            if (
                provider_message.get("role") == "assistant"
                and provider_message.get("content") is None
                and not provider_message.get("tool_calls")
                and "reasoning_content" not in provider_message
            ):
                continue
            provider_messages.append(provider_message)
            if tool_images:
                provider_messages.append({"role": "user", "content": tool_images})
        payload: dict[str, Any] = {
            "model": preset.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *provider_messages,
            ],
            "stream": self.streaming,
            "temperature": preset.temperature,
            "top_p": preset.top_p,
            "max_tokens": preset.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            if preset.provider == "xai":
                payload["parallel_tool_calls"] = False
        if self.streaming:
            return await self._request_stream(payload)
        raw = await self._request_json("POST", "chat/completions", payload)
        try:
            message = raw["choices"][0]["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("模型接口响应缺少 choices[0].message") from error
        if not isinstance(content, str):
            raise RuntimeError("模型接口响应的 message.content 不是字符串")
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal:
            raise RuntimeError(f"模型拒绝生成：{refusal}")
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            reasoning = json.dumps(reasoning, ensure_ascii=False)
        usage = raw.get("usage")
        return ModelResult(
            content=content,
            reasoning=reasoning,
            usage=usage if isinstance(usage, dict) else {},
            raw_response=raw,
            tool_calls=self._tool_calls(message.get("tool_calls")),
        )

    async def _request_stream(self, payload: dict[str, Any]) -> ModelResult:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        usage: dict[str, Any] = {}
        chunks: list[dict[str, Any]] = []
        tool_parts: dict[int, dict[str, str]] = {}
        done = False
        try:
            async with httpx.AsyncClient(
                timeout=self.preset.timeout_seconds,
                **async_client_options(self.network_settings, self.preset.base_url, self.transport),
            ) as client:
                async with client.stream(
                    "POST",
                    self._endpoint("chat/completions"),
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/event-stream" not in content_type:
                        raise RuntimeError("模型接口不支持流式模式，请关闭该 Agent 的流式开关")
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            done = True
                            break
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise RuntimeError("模型接口返回了非法的流式分片") from error
                        if not isinstance(chunk, dict):
                            raise RuntimeError("模型接口返回了非对象的流式分片")
                        chunks.append(chunk)
                        chunk_usage = chunk.get("usage")
                        if isinstance(chunk_usage, dict):
                            usage = chunk_usage
                        choices = chunk.get("choices", [])
                        if not isinstance(choices, list):
                            raise RuntimeError("模型接口流式分片的 choices 不是数组")
                        for choice in choices:
                            if not isinstance(choice, dict):
                                continue
                            delta = choice.get("delta", {})
                            if not isinstance(delta, dict):
                                continue
                            content = delta.get("content")
                            if isinstance(content, str):
                                content_parts.append(content)
                            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                            if isinstance(reasoning, str):
                                reasoning_parts.append(reasoning)
                            for call in delta.get("tool_calls", []) if isinstance(delta.get("tool_calls"), list) else []:
                                if not isinstance(call, dict) or not isinstance(call.get("index"), int):
                                    continue
                                part = tool_parts.setdefault(call["index"], {"id": "", "name": "", "arguments": ""})
                                if isinstance(call.get("id"), str):
                                    part["id"] += call["id"]
                                function = call.get("function")
                                if isinstance(function, dict):
                                    if isinstance(function.get("name"), str):
                                        part["name"] += function["name"]
                                    if isinstance(function.get("arguments"), str):
                                        part["arguments"] += function["arguments"]
        except httpx.HTTPStatusError as error:
            detail = self._safe_error(error.response.text)
            raise RuntimeError(f"模型接口返回 HTTP {error.response.status_code}: {detail}") from error
        except httpx.ReadTimeout as error:
            raise RuntimeError(f"等待模型响应超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectTimeout as error:
            raise RuntimeError(f"连接模型接口超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.WriteTimeout as error:
            raise RuntimeError(f"发送模型请求超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.PoolTimeout as error:
            raise RuntimeError(f"等待模型连接池超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectError as error:
            raise RuntimeError(f"无法建立模型连接：{self._safe_error(str(error)) or type(error).__name__}") from error
        except httpx.HTTPError as error:
            raise RuntimeError(f"模型接口通信失败：{self._safe_error(str(error)) or type(error).__name__}") from error
        if not done:
            raise RuntimeError("模型接口流式响应在完成前中断")
        return ModelResult(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts) or None,
            usage=usage,
            raw_response={"chunks": chunks},
            tool_calls=[ToolCall(**part) for _, part in sorted(tool_parts.items()) if part["id"] and part["name"]],
        )

    async def _request_gemini_stream(self, payload: dict[str, Any]) -> ModelResult:
        chunks: list[dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(
                timeout=self.preset.timeout_seconds,
                **async_client_options(self.network_settings, self.preset.base_url, self.transport),
            ) as client:
                async with client.stream(
                    "POST",
                    self._endpoint(f"{self._gemini_model()}:streamGenerateContent?alt=sse"),
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/event-stream" not in content_type:
                        raise RuntimeError("Gemini 接口不支持 SSE 流式模式，请关闭该 Agent 的流式开关")
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise RuntimeError("Gemini 接口返回了非法的流式分片") from error
                        if not isinstance(chunk, dict):
                            raise RuntimeError("Gemini 接口返回了非对象的流式分片")
                        chunks.append(chunk)
        except httpx.HTTPStatusError as error:
            detail = self._safe_error(error.response.text)
            raise RuntimeError(f"模型接口返回 HTTP {error.response.status_code}: {detail}") from error
        except httpx.ReadTimeout as error:
            raise RuntimeError(f"等待模型响应超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectTimeout as error:
            raise RuntimeError(f"连接模型接口超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.WriteTimeout as error:
            raise RuntimeError(f"发送模型请求超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.PoolTimeout as error:
            raise RuntimeError(f"等待模型连接池超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectError as error:
            raise RuntimeError(f"无法建立模型连接：{self._safe_error(str(error)) or type(error).__name__}") from error
        except httpx.HTTPError as error:
            raise RuntimeError(f"模型接口通信失败：{self._safe_error(str(error)) or type(error).__name__}") from error
        return parse_gemini_stream(chunks)

    async def _request_native_stream(
        self,
        path: str,
        payload: dict[str, Any],
        provider_name: str,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(
                timeout=self.preset.timeout_seconds,
                **async_client_options(self.network_settings, self.preset.base_url, self.transport),
            ) as client:
                async with client.stream(
                    "POST",
                    self._endpoint(path),
                    headers=self._headers(),
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").lower()
                    if "text/event-stream" not in content_type:
                        raise RuntimeError(f"{provider_name} 接口不支持 SSE 流式模式，请关闭该 Agent 的流式开关")
                    async for line in response.aiter_lines():
                        if not line or line.startswith(":") or not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data or data == "[DONE]":
                            continue
                        try:
                            event = json.loads(data)
                        except json.JSONDecodeError as error:
                            raise RuntimeError(f"{provider_name} 接口返回了非法的流式事件") from error
                        if not isinstance(event, dict):
                            raise RuntimeError(f"{provider_name} 接口返回了非对象的流式事件")
                        events.append(event)
        except httpx.HTTPStatusError as error:
            detail = self._safe_error(error.response.text)
            raise RuntimeError(f"模型接口返回 HTTP {error.response.status_code}: {detail}") from error
        except httpx.ReadTimeout as error:
            raise RuntimeError(f"等待模型响应超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectTimeout as error:
            raise RuntimeError(f"连接模型接口超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.WriteTimeout as error:
            raise RuntimeError(f"发送模型请求超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.PoolTimeout as error:
            raise RuntimeError(f"等待模型连接池超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectError as error:
            raise RuntimeError(f"无法建立模型连接：{self._safe_error(str(error)) or type(error).__name__}") from error
        except httpx.HTTPError as error:
            raise RuntimeError(f"模型接口通信失败：{self._safe_error(str(error)) or type(error).__name__}") from error
        return events

    @staticmethod
    def _tool_calls(value: Any) -> list[ToolCall]:
        if not isinstance(value, list):
            return []
        calls: list[ToolCall] = []
        for call in value:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                continue
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                continue
            arguments = function.get("arguments", "{}")
            if isinstance(arguments, (str, dict)):
                calls.append(ToolCall(id=call["id"], name=function["name"], arguments=arguments))
        return calls

    async def list_models(self) -> list[ModelOption]:
        if not self.preset.api_key:
            raise RuntimeError(f"AI Preset“{self.preset.name}”尚未设置 API Key")
        if self.preset.provider == "google_gemini":
            return await self._list_gemini_models()
        if self.preset.provider == "anthropic":
            return await self._list_anthropic_models()
        raw = await self._request_json("GET", "models")
        data = raw.get("data")
        if not isinstance(data, list):
            raise RuntimeError("模型列表响应缺少 data 数组")
        models: list[ModelOption] = []
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                models.append(ModelOption(id=item["id"], owned_by=str(item.get("owned_by", ""))))
        return sorted(models, key=lambda item: item.id.lower())

    async def _list_gemini_models(self) -> list[ModelOption]:
        models: dict[str, ModelOption] = {}
        page_token = ""
        seen_page_tokens: set[str] = set()
        while True:
            query = urlencode({"pageSize": 1000, **({"pageToken": page_token} if page_token else {})})
            raw = await self._request_json("GET", f"models?{query}")
            data = raw.get("models")
            if not isinstance(data, list):
                raise RuntimeError("Gemini 模型列表响应缺少 models 数组")
            for item in data:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    continue
                methods = item.get("supportedGenerationMethods")
                if not isinstance(methods, list) or "generateContent" not in methods:
                    continue
                model_id = item["name"].removeprefix("models/")
                models.setdefault(model_id, ModelOption(id=model_id, owned_by=str(item.get("displayName", "Google"))))
            token = raw.get("nextPageToken")
            if not isinstance(token, str) or not token:
                break
            if token in seen_page_tokens:
                break
            seen_page_tokens.add(token)
            page_token = token
        return sorted(models.values(), key=lambda item: item.id.lower())

    async def _list_anthropic_models(self) -> list[ModelOption]:
        models: dict[str, ModelOption] = {}
        after_id = ""
        seen_ids: set[str] = set()
        while True:
            query = urlencode({"limit": 1000, **({"after_id": after_id} if after_id else {})})
            raw = await self._request_json("GET", f"models?{query}")
            data = raw.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Anthropic 模型列表响应缺少 data 数组")
            for item in data:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    models.setdefault(item["id"], ModelOption(id=item["id"], owned_by=str(item.get("display_name", "Anthropic"))))
            if raw.get("has_more") is not True or not isinstance(raw.get("last_id"), str) or not raw["last_id"]:
                break
            if raw["last_id"] in seen_ids:
                break
            seen_ids.add(raw["last_id"])
            after_id = raw["last_id"]
        return sorted(models.values(), key=lambda item: item.id.lower())

    async def test_connection(self) -> ModelResult:
        needs_reasoning_room = self.preset.provider in {"google_gemini", "anthropic"} or (
            self.preset.provider == "xai" and self.preset.xai_protocol == "responses"
        )
        max_tokens = min(256, self.preset.max_tokens) if needs_reasoning_room else min(8, self.preset.max_tokens)
        client = BoundModelClient(
            self.preset.model_copy(update={"max_tokens": max_tokens}),
            transport=self.transport,
            network_settings=self.network_settings,
        )
        return await client.complete(
            "你正在执行连接测试。",
            [{"role": "user", "content": "只回复 OK。"}],
        )

    def _endpoint(self, path: str) -> str:
        return f"{self.preset.base_url.rstrip('/')}/{path}"

    def _gemini_model(self) -> str:
        return f"models/{self.preset.model.removeprefix('models/')}"

    def _headers(self) -> dict[str, str]:
        if self.preset.provider == "google_gemini":
            return {
                "x-goog-api-key": self.preset.api_key,
                "Content-Type": "application/json",
            }
        if self.preset.provider == "anthropic":
            return {
                "x-api-key": self.preset.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            }
        return {
            "Authorization": f"Bearer {self.preset.api_key}",
            "Content-Type": "application/json",
        }

    def _safe_error(self, detail: str) -> str:
        safe = detail[:4000]
        if self.preset.api_key:
            safe = safe.replace(self.preset.api_key, "***")
        if self.network_settings.proxy_password:
            safe = safe.replace(self.network_settings.proxy_password, "***")
        return safe

    async def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        timeout = self.preset.timeout_seconds
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                **async_client_options(self.network_settings, self.preset.base_url, self.transport),
            ) as client:
                response = await client.request(
                    method,
                    self._endpoint(path),
                    headers=self._headers(),
                    json=payload,
                )
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            detail = self._safe_error(error.response.text)
            raise RuntimeError(
                f"模型接口返回 HTTP {error.response.status_code}: {detail}"
            ) from error
        except httpx.ReadTimeout as error:
            raise RuntimeError(f"等待模型响应超时（{timeout} 秒）") from error
        except httpx.ConnectTimeout as error:
            raise RuntimeError(f"连接模型接口超时（{timeout} 秒）") from error
        except httpx.WriteTimeout as error:
            raise RuntimeError(f"发送模型请求超时（{timeout} 秒）") from error
        except httpx.PoolTimeout as error:
            raise RuntimeError(f"等待模型连接池超时（{timeout} 秒）") from error
        except httpx.ConnectError as error:
            raise RuntimeError(
                f"无法建立模型连接：{self._safe_error(str(error)) or type(error).__name__}"
            ) from error
        except httpx.HTTPError as error:
            raise RuntimeError(
                f"模型接口通信失败：{self._safe_error(str(error)) or type(error).__name__}"
            ) from error
        try:
            raw = response.json()
        except ValueError as error:
            raise RuntimeError("模型接口没有返回合法 JSON") from error
        if not isinstance(raw, dict):
            raise RuntimeError("模型接口 JSON 响应不是对象")
        return raw
