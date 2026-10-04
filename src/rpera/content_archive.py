from __future__ import annotations

import stat
from pathlib import PurePosixPath
from zipfile import ZipInfo

from .content import ENTITY_TYPES, content_name


ARCHIVE_UPLOAD_LIMIT = 16 * 1024 * 1024
ARCHIVE_CONTENT_LIMIT = 64 * 1024 * 1024
ARCHIVE_ENTRY_LIMIT = 4096


def archive_path(value: str, *, directory: bool) -> tuple[str, ...]:
    raw = value[:-1] if directory else value
    path = PurePosixPath(raw)
    parts = path.parts
    if not raw or not parts or path.is_absolute() or path.as_posix() != raw or "\\" in raw:
        raise ValueError(f"压缩包路径不规范：{value}")
    if parts[0] not in {"worlds", "mods"}:
        raise ValueError("压缩包必须以 worlds/ 或 mods/ 开头")
    if directory:
        valid = len(parts) <= 2
        if len(parts) == 3:
            valid = parts[2] in {"scenarios", "entities"}
        elif len(parts) in {4, 5}:
            valid = parts[2] == "entities" and parts[3] in ENTITY_TYPES
    else:
        valid = (
            (len(parts) == 3 and parts[2] == "description.md")
            or (len(parts) == 4 and parts[2] == "scenarios" and parts[3].endswith(".md"))
            or (len(parts) == 6 and parts[2] == "entities" and parts[3] in ENTITY_TYPES and parts[5] == "ENTITY.md")
        )
    if not valid:
        raise ValueError(f"压缩包包含不支持的内容路径：{value}")
    for index, part in enumerate(parts):
        name = part[:-3] if not directory and len(parts) == 4 and index == 3 else part
        if content_name(name) != name:
            raise ValueError(f"压缩包名称必须使用规范 Unicode：{value}")
    return parts


def validate_member(info: ZipInfo) -> tuple[str, ...]:
    if info.orig_filename != info.filename:
        raise ValueError("压缩包路径包含空字符")
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type and file_type != (stat.S_IFDIR if info.is_dir() else stat.S_IFREG):
        raise ValueError(f"压缩包只允许普通文件和目录：{info.filename}")
    if info.flag_bits & 1:
        raise ValueError("不支持加密压缩包")
    if info.is_dir() and info.file_size:
        raise ValueError("压缩包目录不能包含文件数据")
    return archive_path(info.filename, directory=info.is_dir())
