from __future__ import annotations

import base64
import binascii
import re
from typing import Any


IMAGE_DATA_URL = re.compile(r"data:(image/(?:jpeg|png));base64,([a-zA-Z0-9+/]+={0,2})")


def content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        raise RuntimeError("模型消息内容必须是文本或内容块数组")
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise RuntimeError("模型消息包含非法内容块")
        block_type = block.get("type")
        if block_type == "text" and isinstance(block.get("text"), str):
            blocks.append({"type": "text", "text": block["text"]})
            continue
        if block_type == "image_url":
            image_url = block.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else None
            if isinstance(url, str):
                image_data_url(url)
                blocks.append({"type": "image_url", "image_url": {"url": url}})
                continue
        raise RuntimeError("模型消息包含非法图片内容块")
    return blocks


def image_data_url(url: str) -> tuple[str, str]:
    match = IMAGE_DATA_URL.fullmatch(url)
    if match is None:
        raise RuntimeError("图片内容必须是 JPEG 或 PNG base64 data URL")
    try:
        base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as error:
        raise RuntimeError("图片内容包含非法 base64 数据") from error
    return match.group(1), match.group(2)


def image_block(mime_type: str, data: str) -> dict[str, Any]:
    block = {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}}
    content_blocks([block])
    return block


def _merge_content(left: Any, right: Any) -> Any:
    if isinstance(left, str) and isinstance(right, str):
        return f"{left}\n{right}"
    blocks = content_blocks(left)
    incoming = content_blocks(right)
    if blocks and incoming and blocks[-1]["type"] == incoming[0]["type"] == "text":
        blocks[-1]["text"] = f"{blocks[-1]['text']}\n{incoming[0]['text']}"
        incoming = incoming[1:]
    return [*blocks, *incoming]


def system_messages_as_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Downgrade inline system messages and merge adjacent user content."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        current = dict(message)
        separate_after_current = bool(current.pop("_separate_next_user", False))
        if current.get("role") == "system":
            current["role"] = "user"
        if (
            current.get("role") == "user"
            and converted
            and converted[-1].get("role") == "user"
        ):
            if converted[-1].pop("_separate_next_user", False):
                converted[-1]["content"] = [
                    *content_blocks(converted[-1].get("content")),
                    {"type": "text", "text": "\n"},
                    *content_blocks(current.get("content")),
                ]
            else:
                converted[-1]["content"] = _merge_content(
                    converted[-1].get("content"), current.get("content")
                )
            if separate_after_current:
                converted[-1]["_separate_next_user"] = True
            continue
        if separate_after_current:
            current["_separate_next_user"] = True
        converted.append(current)
    for message in converted:
        message.pop("_separate_next_user", None)
    return converted
