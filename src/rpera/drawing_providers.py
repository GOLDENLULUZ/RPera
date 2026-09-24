from __future__ import annotations

import asyncio
import base64
import json
import secrets
from io import BytesIO
from zipfile import BadZipFile, ZipFile
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image, UnidentifiedImageError

from .models import CharacterPortraitGenerationSettings, DrawingPreset, DrawingResourceOption, DrawingResources, DrawingTestResult, NetworkSettings
from .network import async_client_options


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 4_194_304


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime_type: str
    width: int
    height: int
    seed: int | None


class StableDiffusionWebUIClient:
    def __init__(
        self,
        preset: DrawingPreset,
        transport: httpx.AsyncBaseTransport | None = None,
        network_settings: NetworkSettings | None = None,
    ) -> None:
        self.preset = preset
        self.transport = transport
        self.network_settings = network_settings or NetworkSettings()

    async def resources(self) -> DrawingResources:
        options, models, samplers, schedulers = await asyncio.gather(
            self._request_json("GET", "options"),
            self._request_json("GET", "sd-models"),
            self._request_json("GET", "samplers"),
            self._optional_schedulers(),
        )
        return DrawingResources(
            active_checkpoint=str(options.get("sd_model_checkpoint", "")),
            checkpoints=self._options(models, "title", "model_name"),
            samplers=self._options(samplers, "name"),
            schedulers=self._options(schedulers, "label", "name"),
        )

    async def _optional_schedulers(self) -> list[dict[str, Any]]:
        try:
            result = await self._request_json("GET", "schedulers")
        except RuntimeError:
            return []
        return result if isinstance(result, list) else []

    async def test_connection(self) -> str:
        options = await self._request_json("GET", "options")
        return str(options.get("sd_model_checkpoint", ""))

    async def generate(
        self,
        content_prompt: str,
        settings: CharacterPortraitGenerationSettings,
    ) -> GeneratedImage:
        positive = ", ".join(part for part in (settings.positive_prompt, content_prompt) if part)
        payload: dict[str, Any] = {
            "prompt": positive,
            "negative_prompt": settings.negative_prompt,
            "sampler_name": settings.sampler_name,
            "scheduler": settings.scheduler,
            "steps": settings.steps,
            "cfg_scale": settings.cfg_scale,
            "width": settings.width,
            "height": settings.height,
            "batch_size": 1,
            "n_iter": 1,
        }
        if settings.checkpoint:
            payload["override_settings"] = {"sd_model_checkpoint": settings.checkpoint}
        raw = await self._request_json("POST", "txt2img", payload)
        images = raw.get("images")
        if not isinstance(images, list) or not images or not isinstance(images[0], str):
            raise RuntimeError("WebUI 生成响应缺少图片")
        image, width, height, mime_type = self._validate_image(images[0])
        seed = self._seed(raw.get("info"))
        return GeneratedImage(image, mime_type, width, height, seed)

    async def generate_test(
        self,
        content_prompt: str,
        settings: CharacterPortraitGenerationSettings,
    ) -> DrawingTestResult:
        generated = await self.generate(content_prompt, settings)
        encoded = base64.b64encode(generated.data).decode("ascii")
        return DrawingTestResult(
            image_data_url=f"data:{generated.mime_type};base64,{encoded}",
            width=generated.width,
            height=generated.height,
            seed=generated.seed,
        )

    @staticmethod
    def _options(raw: object, *keys: str) -> list[DrawingResourceOption]:
        if not isinstance(raw, list):
            raise RuntimeError("WebUI 资源列表响应不是数组")
        result: list[DrawingResourceOption] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            value = next((candidate for key in keys if isinstance((candidate := item.get(key)), str) and candidate), None)
            if value:
                result.append(DrawingResourceOption(id=value, label=value))
        return sorted(result, key=lambda item: item.label.lower())

    @staticmethod
    def _seed(info: Any) -> int | None:
        if not isinstance(info, str):
            return None
        try:
            value = json.loads(info).get("seed")
        except (ValueError, AttributeError):
            return None
        return value if isinstance(value, int) else None

    def _endpoint(self, path: str) -> str:
        return f"{self.preset.base_url}/sdapi/v1/{path}"

    def _auth(self) -> httpx.BasicAuth | None:
        return httpx.BasicAuth(self.preset.auth_username, self.preset.auth_password) if self.preset.auth_username else None

    async def _request_json(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        try:
            async with httpx.AsyncClient(
                timeout=self.preset.timeout_seconds,
                **async_client_options(self.network_settings, self.preset.base_url, self.transport),
            ) as client:
                response = await client.request(method, self._endpoint(path), json=payload, auth=self._auth())
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"WebUI 返回 HTTP {error.response.status_code}: {self._safe_error(error.response.text)}") from error
        except httpx.ReadTimeout as error:
            raise RuntimeError(f"等待 WebUI 响应超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectTimeout as error:
            raise RuntimeError(f"连接 WebUI 超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.ConnectError as error:
            raise RuntimeError(f"无法建立 WebUI 连接：{self._safe_error(str(error)) or type(error).__name__}") from error
        except httpx.HTTPError as error:
            raise RuntimeError(f"WebUI 通信失败：{self._safe_error(str(error)) or type(error).__name__}") from error
        try:
            return response.json()
        except ValueError as error:
            raise RuntimeError("WebUI 没有返回合法 JSON") from error

    def _safe_error(self, detail: str) -> str:
        safe = detail[:4000]
        for password in (self.preset.auth_password, self.network_settings.proxy_password):
            if password:
                safe = safe.replace(password, "***")
        return safe

    @staticmethod
    def _validate_image(encoded: str) -> tuple[bytes, int, int, str]:
        if encoded.startswith("data:"):
            encoded = encoded.split(",", 1)[-1]
        if len(encoded) > MAX_IMAGE_BYTES * 2:
            raise RuntimeError("WebUI 返回的图片过大")
        try:
            image = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise RuntimeError("WebUI 返回了非法图片数据") from error
        if len(image) > MAX_IMAGE_BYTES:
            raise RuntimeError("WebUI 返回的图片过大")
        if image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) >= 24:
            width, height = int.from_bytes(image[16:20], "big"), int.from_bytes(image[20:24], "big")
            mime_type = "image/png"
        elif image.startswith(b"\xff\xd8"):
            width, height = StableDiffusionWebUIClient._jpeg_size(image)
            mime_type = "image/jpeg"
        elif image.startswith(b"RIFF") and image[8:12] == b"WEBP":
            width, height = StableDiffusionWebUIClient._webp_size(image)
            mime_type = "image/webp"
        else:
            raise RuntimeError("WebUI 返回的内容不是支持的图片")
        if not width or not height or width * height > MAX_IMAGE_PIXELS:
            raise RuntimeError("WebUI 返回的图片尺寸超出限制")
        return image, width, height, mime_type

    @staticmethod
    def _jpeg_size(image: bytes) -> tuple[int, int]:
        index = 2
        while index + 9 <= len(image):
            if image[index] != 0xFF:
                index += 1
                continue
            marker = image[index + 1]
            index += 2
            if marker in {0xD8, 0xD9}:
                continue
            if index + 2 > len(image):
                break
            length = int.from_bytes(image[index:index + 2], "big")
            if length < 2 or index + length > len(image):
                break
            if 0xC0 <= marker <= 0xC3 and index + 7 <= len(image):
                return int.from_bytes(image[index + 5:index + 7], "big"), int.from_bytes(image[index + 3:index + 5], "big")
            index += length
        raise RuntimeError("无法读取 JPEG 图片尺寸")

    @staticmethod
    def _webp_size(image: bytes) -> tuple[int, int]:
        kind = image[12:16]
        if kind == b"VP8X" and len(image) >= 30:
            return int.from_bytes(image[24:27], "little") + 1, int.from_bytes(image[27:30], "little") + 1
        if kind == b"VP8 " and len(image) >= 30 and image[23:26] == b"\x9d\x01\x2a":
            return int.from_bytes(image[26:28], "little") & 0x3FFF, int.from_bytes(image[28:30], "little") & 0x3FFF
        if kind == b"VP8L" and len(image) >= 25 and image[20] == 0x2F:
            bits = int.from_bytes(image[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        raise RuntimeError("无法读取 WebP 图片尺寸")


class NovelAIClient:
    """NovelAI's image endpoint returns a ZIP, not WebUI's base64 JSON."""

    def __init__(
        self,
        preset: DrawingPreset,
        transport: httpx.AsyncBaseTransport | None = None,
        network_settings: NetworkSettings | None = None,
    ) -> None:
        self.preset = preset
        self.transport = transport
        self.network_settings = network_settings or NetworkSettings()

    async def test_connection(self) -> str:
        await self._request("GET", "/user/information")
        return self.preset.model

    async def generate(self, content_prompt: str, settings: CharacterPortraitGenerationSettings) -> GeneratedImage:
        if not self.preset.api_key:
            raise RuntimeError("NovelAI API Token 未配置")
        if not 1 <= settings.steps <= 50 or not 0 <= settings.cfg_scale <= 10:
            raise RuntimeError("NovelAI 的 Steps 必须为 1～50，CFG 必须为 0～10")
        if not (64 <= settings.width <= 1600 and 64 <= settings.height <= 1600
                and settings.width % 64 == 0 and settings.height % 64 == 0):
            raise RuntimeError("NovelAI 的宽高必须为 64～1600 且为 64 的倍数")
        positive = ", ".join(part for part in (settings.positive_prompt, content_prompt) if part)
        seed = secrets.randbelow(2**32 - 1) + 1
        negative = settings.negative_prompt
        parameters: dict[str, Any] = {
            "width": settings.width,
            "height": settings.height,
            "n_samples": 1,
            "steps": settings.steps,
            "scale": settings.cfg_scale,
            "sampler": settings.novelai_sampler,
            "seed": seed,
            "negative_prompt": negative,
            "uc": negative,
            "v4_prompt": {
                "caption": {"base_caption": positive, "char_captions": []},
                "use_coords": False,
                "use_order": True,
            },
            "v4_negative_prompt": {"caption": {"base_caption": negative, "char_captions": []}},
            "params_version": 4 if self.preset.model.startswith("nai-diffusion-5-") else 3,
            "legacy_uc": False,
            "noise_schedule": "karras",
            "qualityToggle": False,
            "ucPreset": 4,
            "cfg_rescale": 0,
        }
        response = await self._request(
            "POST",
            "/ai/generate-image",
            {"input": positive, "model": self.preset.model, "action": "generate", "parameters": parameters},
        )
        if len(response.content) > MAX_IMAGE_BYTES * 2:
            raise RuntimeError("NovelAI 返回的图片压缩包过大")
        try:
            with ZipFile(BytesIO(response.content)) as archive:
                images = [item for item in archive.infolist() if not item.is_dir() and item.filename.lower().endswith((".png", ".jpg", ".jpeg"))]
                if len(images) != 1 or images[0].file_size > MAX_IMAGE_BYTES:
                    raise RuntimeError("NovelAI 生成响应缺少单张有效图片")
                with archive.open(images[0]) as entry:
                    data = entry.read(MAX_IMAGE_BYTES + 1)
        except (BadZipFile, EOFError, OSError, ValueError) as error:
            raise RuntimeError("NovelAI 没有返回合法图片压缩包") from error
        if len(data) > MAX_IMAGE_BYTES:
            raise RuntimeError("NovelAI 返回的图片过大")
        try:
            with Image.open(BytesIO(data)) as image:
                width, height = image.size
                mime_type = {"PNG": "image/png", "JPEG": "image/jpeg"}.get(image.format or "")
                if mime_type is None or width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                    raise RuntimeError("NovelAI 返回的图片格式或尺寸不受支持")
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise RuntimeError("NovelAI 返回了非法图片数据") from error
        return GeneratedImage(data, mime_type, width, height, seed)

    async def generate_test(self, content_prompt: str, settings: CharacterPortraitGenerationSettings) -> DrawingTestResult:
        generated = await self.generate(content_prompt, settings)
        return DrawingTestResult(
            image_data_url=f"data:{generated.mime_type};base64,{base64.b64encode(generated.data).decode('ascii')}",
            width=generated.width,
            height=generated.height,
            seed=generated.seed,
        )

    async def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> httpx.Response:
        if not self.preset.api_key:
            raise RuntimeError("NovelAI API Token 未配置")
        try:
            async with httpx.AsyncClient(
                timeout=self.preset.timeout_seconds,
                **async_client_options(self.network_settings, self.preset.base_url, self.transport),
            ) as client:
                response = await client.request(
                    method,
                    f"{self.preset.base_url}{path}",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.preset.api_key}", "Accept": "application/zip" if payload else "application/json"},
                )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as error:
            raise RuntimeError(f"NovelAI 返回 HTTP {error.response.status_code}: {self._safe_error(error.response.text[:4000])}") from error
        except httpx.TimeoutException as error:
            raise RuntimeError(f"等待 NovelAI 响应超时（{self.preset.timeout_seconds} 秒）") from error
        except httpx.HTTPError as error:
            raise RuntimeError(f"NovelAI 通信失败：{self._safe_error(str(error))}") from error

    def _safe_error(self, detail: str) -> str:
        for secret in (self.preset.api_key, self.network_settings.proxy_password):
            if secret:
                detail = detail.replace(secret, "***")
        return detail[:4000]
