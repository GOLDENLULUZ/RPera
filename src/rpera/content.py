from __future__ import annotations

import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import yaml


_WINDOWS_FORBIDDEN = frozenset('<>:"/\\|?*')
_WINDOWS_RESERVED = frozenset(
    {"CON", "PRN", "AUX", "NUL", "CLOCK$"}
    | {f"COM{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")}
    | {f"LPT{suffix}" for suffix in (*range(1, 10), "¹", "²", "³")}
)
ENTITY_TYPES = ("character", "location", "event", "item", "organization", "rule", "player_trait")
SCENARIO_DOCUMENT_LIMIT = 65_000
ENTITY_DOCUMENT_LIMIT = 170_000
STYLE_DOCUMENT_LIMIT = 170_000


def content_name(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("名称必须是字符串")
    normalized = unicodedata.normalize("NFC", value)
    if not normalized or not normalized.strip():
        raise ValueError("名称不能为空")
    if normalized != normalized.strip():
        raise ValueError("名称不能包含首尾空白")
    if normalized in {".", ".."} or normalized.endswith("."):
        raise ValueError("名称不能是相对路径或以点结尾")
    if len(normalized) > 80:
        raise ValueError("名称不能超过 80 个字符")
    if len(normalized.encode("utf-8")) > 240:
        raise ValueError("名称的 UTF-8 编码不能超过 240 字节")
    if any(character in _WINDOWS_FORBIDDEN for character in normalized):
        raise ValueError("名称包含跨平台文件名禁用字符")
    if any(
        character != " "
        and (unicodedata.category(character).startswith("C") or unicodedata.category(character).startswith("Z"))
        for character in normalized
    ):
        raise ValueError("名称包含控制字符或不可见字符")
    if normalized.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError("名称是平台保留文件名")
    return normalized


def content_name_key(value: str) -> str:
    return content_name(value).casefold()


def is_internal_temporary_name(
    value: str, *, target_name: str | None = None, target_suffix: str | None = None
) -> bool:
    if not value.startswith(".") or not value.endswith(".tmp"):
        return False
    target, separator, token = value[1:-4].rpartition(".")
    if not target or not separator:
        return False
    if target_name is not None and target != target_name:
        return False
    if target_suffix is not None and not target.endswith(target_suffix):
        return False
    try:
        return str(uuid.UUID(token)) == token
    except ValueError:
        return False


def markdown_frontmatter(text: str, subject: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 3)
    marker_length = 5
    if end == -1 and text.endswith("\n---"):
        end = len(text) - 4
        marker_length = 4
    if end == -1:
        raise ValueError(f"{subject} frontmatter 未闭合")
    raw = text[3:end].strip()
    body = text[end + marker_length :]
    if body.startswith("\n"):
        body = body[1:]
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ValueError(f"{subject} frontmatter 不是有效 YAML") from error
    if parsed is None:
        return {}, body
    if not isinstance(parsed, dict):
        raise ValueError(f"{subject} frontmatter 必须是对象")
    return parsed, body


def content_description(text: Any) -> str:
    if not isinstance(text, str):
        raise ValueError("简介必须是字符串")
    if len(text) > 4_000:
        raise ValueError("简介不能超过 4000 字符")
    return text


def entity_type(value: Any) -> str:
    if value not in ENTITY_TYPES:
        raise ValueError(f"不支持的实体类型：{value}")
    return value


def parse_scenario_document(text: str) -> tuple[str, str]:
    if len(text) > SCENARIO_DOCUMENT_LIMIT:
        raise ValueError("场景文件超过大小限制")
    has_frontmatter = text.startswith("---\n")
    front, opening = markdown_frontmatter(text, "场景")
    unknown = set(front) - {"description"}
    if unknown:
        raise ValueError(f"场景 frontmatter 不允许字段：{', '.join(sorted(map(str, unknown)))}")
    description = content_description(front.get("description", ""))
    if not has_frontmatter:
        opening = opening.strip()
    if len(opening) > 20_000:
        raise ValueError("场景 opening 不能超过 20000 字符")
    return description, opening


def serialize_scenario_document(description: str, opening: str) -> str:
    content_description(description)
    if len(opening) > 20_000:
        raise ValueError("场景 opening 不能超过 20000 字符")
    return _frontmatter({"description": description}, opening)


@dataclass(frozen=True)
class ParsedEntity:
    description: str
    aliases: list[str]
    visibility: str
    required: bool
    content: str


def parse_entity_document(text: str) -> ParsedEntity:
    if len(text) > ENTITY_DOCUMENT_LIMIT:
        raise ValueError("实体文件超过大小限制")
    front, body = markdown_frontmatter(text, "实体")
    unknown = set(front) - {"description", "aliases", "visibility", "required"}
    if unknown:
        raise ValueError(f"实体 frontmatter 不允许字段：{', '.join(sorted(map(str, unknown)))}")
    description = content_description(front.get("description", ""))
    aliases = entity_aliases(front.get("aliases", []))
    visibility = front.get("visibility", "world_truth")
    if visibility != "world_truth":
        raise ValueError("实体 visibility 当前只允许 world_truth")
    required = front.get("required", False)
    if not isinstance(required, bool):
        raise ValueError("实体 required 必须是布尔值")
    if len(body) > 100_000:
        raise ValueError("实体 content 不能超过 100000 字符")
    return ParsedEntity(description, aliases, visibility, required, body)


def serialize_entity_document(entity: ParsedEntity) -> str:
    description = content_description(entity.description)
    aliases = entity_aliases(entity.aliases)
    if entity.visibility != "world_truth":
        raise ValueError("实体 visibility 当前只允许 world_truth")
    if not isinstance(entity.required, bool):
        raise ValueError("实体 required 必须是布尔值")
    if len(entity.content) > 100_000:
        raise ValueError("实体 content 不能超过 100000 字符")
    return _frontmatter(
        {
            "description": description,
            "aliases": aliases,
            "visibility": entity.visibility,
            "required": entity.required,
        },
        entity.content,
    )


@dataclass(frozen=True)
class ParsedStyle:
    description: str
    enabled: bool
    content: str


def parse_style_document(text: str) -> ParsedStyle:
    if len(text) > STYLE_DOCUMENT_LIMIT:
        raise ValueError("文风文件超过大小限制")
    front, body = markdown_frontmatter(text, "文风")
    unknown = set(front) - {"description", "enabled"}
    if unknown:
        raise ValueError(f"文风 frontmatter 不允许字段：{', '.join(sorted(map(str, unknown)))}")
    description = content_description(front.get("description", ""))
    if "enabled" not in front:
        raise ValueError("文风 enabled 为必填字段")
    enabled = front["enabled"]
    if not isinstance(enabled, bool):
        raise ValueError("文风 enabled 必须是布尔值")
    if len(body) > 100_000:
        raise ValueError("文风 content 不能超过 100000 字符")
    return ParsedStyle(description, enabled, body)


def serialize_style_document(style: ParsedStyle) -> str:
    description = content_description(style.description)
    if not isinstance(style.enabled, bool):
        raise ValueError("文风 enabled 必须是布尔值")
    if len(style.content) > 100_000:
        raise ValueError("文风 content 不能超过 100000 字符")
    return _frontmatter({"description": description, "enabled": style.enabled}, style.content)


def entity_aliases(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("实体 aliases 必须是数组")
    aliases: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            item = item.strip()
            if not item:
                continue
        alias = content_name(item)
        if "," in alias:
            raise ValueError("实体 alias 不能包含英文半角逗号")
        key = content_name_key(alias)
        if key not in seen:
            seen.add(key)
            aliases.append(alias)
    if len(aliases) > 20:
        raise ValueError("实体 aliases 不能超过 20 项")
    return aliases


def _frontmatter(fields: dict[str, Any], body: str) -> str:
    raw = yaml.safe_dump(fields, allow_unicode=True, sort_keys=False, default_flow_style=False).rstrip("\n")
    return f"---\n{raw}\n---\n\n{body}"


def entity_snapshot_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("实体快照路径不能为空")
    if "\\" in value:
        raise ValueError("实体快照路径必须使用正斜杠")
    path = PurePosixPath(value)
    parts = path.parts
    is_world_entity = len(parts) == 4 and parts[0] == "entities" and parts[3] == "ENTITY.md"
    is_mod_entity = len(parts) == 6 and parts[0] == "mods" and parts[2] == "entities" and parts[5] == "ENTITY.md"
    if path.is_absolute() or not (is_world_entity or is_mod_entity):
        raise ValueError("实体快照路径格式无效")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("实体快照路径不能包含路径穿越")
    for part in parts[1:-1]:
        if part != "entities":
            content_name(part)
    if path.as_posix() != value:
        raise ValueError("实体快照路径必须使用规范形式")
    return value


def character_snapshot_path(value: Any) -> str:
    checked = entity_snapshot_path(value)
    parts = PurePosixPath(checked).parts
    type_index = 1 if parts[0] == "entities" else 3
    if parts[type_index] != "character":
        raise ValueError("角色实体路径必须指向 character 类型")
    return checked


def location_snapshot_path(value: Any) -> str:
    checked = entity_snapshot_path(value)
    parts = PurePosixPath(checked).parts
    type_index = 1 if parts[0] == "entities" else 3
    if parts[type_index] != "location":
        raise ValueError("地点实体路径必须指向 location 类型")
    return checked


def style_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("文风路径不能为空")
    if "\\" in value:
        raise ValueError("文风路径必须使用正斜杠")
    path = PurePosixPath(value)
    parts = path.parts
    if path.is_absolute() or len(parts) != 2 or parts[0] != "styles" or not parts[1].endswith(".md"):
        raise ValueError("文风路径格式无效")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("文风路径不能包含路径穿越")
    name = content_name(parts[1][:-3])
    if path.as_posix() != value or value != f"styles/{name}.md":
        raise ValueError("文风路径必须使用规范形式")
    return value
