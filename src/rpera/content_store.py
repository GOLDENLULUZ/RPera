from __future__ import annotations

import os
import shutil
import stat
import uuid
import zlib
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

from .content_archive import ARCHIVE_CONTENT_LIMIT, ARCHIVE_ENTRY_LIMIT, ARCHIVE_UPLOAD_LIMIT, archive_path, validate_member

from .content import (
    ENTITY_DOCUMENT_LIMIT,
    SCENARIO_DOCUMENT_LIMIT,
    STYLE_DOCUMENT_LIMIT,
    ParsedEntity,
    ParsedStyle,
    content_description,
    content_name,
    content_name_key,
    entity_type,
    is_internal_temporary_name,
    parse_entity_document,
    parse_scenario_document,
    parse_style_document,
    serialize_entity_document,
    serialize_scenario_document,
    serialize_style_document,
    style_path,
)
from .models import (
    ContentEntityCreate,
    ContentEntityMove,
    ContentEntityUpdate,
    ContentRename,
    ContentScenarioCreate,
    ContentScenarioUpdate,
    ContentSourceCreate,
    ContentSourceUpdate,
    ContentStyleCreate,
    ContentStyleUpdate,
    StyleDocument,
    StyleSummary,
)


SourceKind = Literal["world", "mod"]


class ContentStoreError(Exception):
    pass


class UnsafeContentError(ContentStoreError):
    pass


class ContentNotFoundError(ContentStoreError):
    pass


class ContentConflictError(ContentStoreError):
    pass


@dataclass(frozen=True)
class ScenarioRecord:
    path: Path
    description: str
    opening: str


@dataclass(frozen=True)
class EntityRecord:
    directory: Path
    document: Path
    type: str
    parsed: ParsedEntity


@dataclass(frozen=True)
class StyleRecord:
    path: Path
    parsed: ParsedStyle


@dataclass(frozen=True)
class SourceScan:
    path: Path
    description: str
    scenarios: list[ScenarioRecord]
    entities: list[EntityRecord]


class ContentStore:
    def __init__(self, data_dir: Path, lock: RLock | None = None) -> None:
        self.data_dir = data_dir
        self.lock = lock if lock is not None else RLock()
        self.staging_dir = data_dir / ".content-staging"
        self.deleted_dir = data_dir / ".deleted-content"

    def list_sources(self, kind: SourceKind) -> list[dict[str, Any]]:
        with self.lock:
            return [self._source_summary(kind, self._scan_source(path)) for path in self._source_directories(kind)]

    def get_source(self, kind: SourceKind, name: str) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, name))
            return self._source_summary(kind, scan)

    def create_source(self, kind: SourceKind, request: ContentSourceCreate) -> dict[str, Any]:
        with self.lock:
            self._assert_source_available(kind, request.name)
            self._ensure_directory(self.data_dir, "数据根目录")
            active = self._active_dir(kind)
            self._ensure_directory(active, "内容库")
            target = active / request.name
            target.mkdir()
            self._sync_directory(active)
            return self._source_summary(kind, self._scan_source(target))

    def update_source(self, kind: SourceKind, name: str, request: ContentSourceUpdate) -> dict[str, Any]:
        with self.lock:
            source = self._source_directory(kind, name)
            self._scan_source(source)
            self._atomic_write(source / "description.md", request.description)
            return self._source_summary(kind, self._scan_source(source))

    def rename_source(self, kind: SourceKind, name: str, request: ContentRename) -> dict[str, Any]:
        with self.lock:
            source = self._source_directory(kind, name)
            self._scan_source(source)
            target = source.parent / request.name
            if target != source:
                self._assert_source_available(kind, request.name, source)
                self._rename(source, target)
            return self._source_summary(kind, self._scan_source(target))

    def delete_source(self, kind: SourceKind, name: str) -> dict[str, Any]:
        with self.lock:
            source = self._source_directory(kind, name)
            self._scan_source(source)
            path = self._source_relative_path(kind, source.name)
            self._remove_active(source)
            return {"name": source.name, "path": path, "deleted": True}

    def export_source(self, kind: SourceKind, name: str) -> tuple[str, bytes]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, name))
            prefix = self._source_relative_path(kind, scan.path.name)
            output = BytesIO()
            total = 0
            entries = 1
            with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
                archive.writestr(f"{prefix}/", b"")
                for path in sorted(scan.path.rglob("*")):
                    relative = path.relative_to(scan.path).as_posix()
                    if (
                        (path.parent == scan.path and self._is_internal_temporary(path, target_name="description.md"))
                        or (path.parent == scan.path / "scenarios" and self._is_internal_temporary(path, target_suffix=".md"))
                        or (path.parent.parent.parent == scan.path / "entities" and self._is_internal_temporary(path, target_name="ENTITY.md"))
                    ):
                        continue
                    directory = stat.S_ISDIR(path.lstat().st_mode)
                    member = f"{prefix}/{relative}" + ("/" if directory else "")
                    try:
                        archive_path(member, directory=directory)
                    except ValueError as error:
                        raise UnsafeContentError(str(error)) from error
                    entries += 1
                    if entries > ARCHIVE_ENTRY_LIMIT:
                        raise UnsafeContentError("压缩包条目数量超过限制")
                    if directory:
                        self._require_directory(path, "内容目录")
                        archive.writestr(member, b"")
                    else:
                        total += self._require_regular(path, "内容文件").st_size
                        if total > ARCHIVE_CONTENT_LIMIT:
                            raise UnsafeContentError("导出内容超过大小限制")
                        archive.write(path, member)
            data = output.getvalue()
            if len(data) > ARCHIVE_UPLOAD_LIMIT:
                raise UnsafeContentError("导出压缩包超过大小限制")
            return scan.path.name, data

    def import_source(self, kind: SourceKind, data: bytes) -> dict[str, Any]:
        if len(data) > ARCHIVE_UPLOAD_LIMIT:
            raise UnsafeContentError("压缩包不能超过 16 MiB")
        with self.lock:
            self._ensure_directory(self.data_dir, "数据根目录")
            self._ensure_directory(self.staging_dir, "内容 staging 目录")
            staging = self.staging_dir / f"import-{uuid.uuid4()}"
            try:
                with ZipFile(BytesIO(data)) as archive:
                    members = archive.infolist()
                    if not members or len(members) > ARCHIVE_ENTRY_LIMIT:
                        raise ValueError("压缩包为空或条目数量超过限制")
                    roots: set[tuple[str, str]] = set()
                    seen: set[tuple[str, ...]] = set()
                    canonical: dict[tuple[str, ...], tuple[str, ...]] = {}
                    checked = []
                    total = 0
                    for info in members:
                        parts = validate_member(info)
                        if parts[0] != ("worlds" if kind == "world" else "mods"):
                            raise ValueError("压缩包类型与当前导入列表不一致")
                        key = tuple(part.casefold() for part in parts)
                        if key in seen:
                            raise ValueError(f"压缩包路径重复：{info.filename}")
                        seen.add(key)
                        for length in range(1, len(parts) + 1):
                            prefix = parts[:length]
                            prefix_key = key[:length]
                            if canonical.setdefault(prefix_key, prefix) != prefix:
                                raise ValueError(f"压缩包路径大小写冲突：{info.filename}")
                        if len(parts) >= 2:
                            roots.add((parts[0], parts[1]))
                        total += info.file_size
                        if total > ARCHIVE_CONTENT_LIMIT:
                            raise ValueError("解压内容不能超过 64 MiB")
                        checked.append((info, parts))
                    if len(roots) != 1:
                        raise ValueError("压缩包必须只包含一个世界或模组")
                    original_name = next(iter(roots))[1]
                    staging.mkdir()
                    total = 0
                    for info, parts in checked:
                        if len(parts) <= 2:
                            continue
                        target = staging.joinpath(*parts[2:])
                        if info.is_dir():
                            target.mkdir(parents=True, exist_ok=True)
                            continue
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as source, target.open("xb") as destination:
                            while chunk := source.read(64 * 1024):
                                total += len(chunk)
                                if total > ARCHIVE_CONTENT_LIMIT:
                                    raise ValueError("解压内容不能超过 64 MiB")
                                destination.write(chunk)
                            destination.flush()
                            os.fsync(destination.fileno())
                    self._scan_source(staging)
                    active = self._active_dir(kind)
                    self._ensure_directory(active, "内容库")
                    occupied = {content_name_key(path.name) for path in self._source_directories(kind)}
                    name = original_name
                    number = 1
                    while content_name_key(name) in occupied:
                        suffix = f"({number})"
                        base = original_name
                        while len(base + suffix) > 80 or len((base + suffix).encode("utf-8")) > 240:
                            base = base[:-1]
                        name = base.rstrip(" .") + suffix
                        number += 1
                    for directory in sorted((path for path in staging.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True):
                        self._sync_directory(directory)
                    self._sync_directory(staging)
                    target = active / name
                    self._rename(staging, target)
                    return self._source_summary(kind, self._scan_source(target))
            except (ValueError, BadZipFile, NotImplementedError, RuntimeError, EOFError, zlib.error) as error:
                raise UnsafeContentError(f"无法导入压缩包：{error}") from error
            finally:
                self._cleanup_hidden(staging)

    def list_styles(self) -> list[StyleSummary]:
        with self.lock:
            return [self._style_summary(record) for record in self._style_records()]

    def list_enabled_styles(self) -> list[StyleSummary]:
        with self.lock:
            return [self._style_summary(record) for record in self._style_records() if record.parsed.enabled]

    def read_styles(self, paths: list[str]) -> list[StyleDocument]:
        return self._read_styles(paths, enabled_only=False)

    def read_enabled_styles(self, paths: list[str]) -> list[StyleDocument]:
        return self._read_styles(paths, enabled_only=True)

    def _read_styles(self, paths: list[str], *, enabled_only: bool) -> list[StyleDocument]:
        with self.lock:
            indexed = {self._style_relative_path(record.path.stem): record for record in self._style_records()}
            documents: list[StyleDocument] = []
            for requested in paths:
                try:
                    canonical = style_path(requested)
                except ValueError as error:
                    raise UnsafeContentError(str(error)) from error
                record = indexed.get(canonical)
                if record is None:
                    raise ContentNotFoundError(f"文风引用不存在：{canonical}")
                if enabled_only and not record.parsed.enabled:
                    raise ContentNotFoundError(f"文风已禁用：{canonical}")
                documents.append(self._style_document(record))
            return documents

    def get_style(self, name: str) -> StyleDocument:
        with self.lock:
            return self._style_document(self._style_record(name))

    def create_style(self, request: ContentStyleCreate) -> StyleDocument:
        with self.lock:
            self._assert_style_available(request.name)
            self._ensure_directory(self.data_dir, "数据根目录")
            root = self.data_dir / "styles"
            self._ensure_directory(root, "文风库")
            path = root / f"{request.name}.md"
            self._atomic_write(path, serialize_style_document(ParsedStyle("", True, "")))
            return self._style_document(self._read_style(path))

    def update_style(self, name: str, request: ContentStyleUpdate) -> StyleDocument:
        with self.lock:
            record = self._style_record(name)
            self._atomic_write(
                record.path,
                serialize_style_document(ParsedStyle(request.description, record.parsed.enabled, request.content)),
            )
            return self._style_document(self._read_style(record.path))

    def set_style_enabled(self, name: str, enabled: bool) -> StyleDocument:
        with self.lock:
            record = self._style_record(name)
            parsed = ParsedStyle(record.parsed.description, enabled, record.parsed.content)
            self._atomic_write(record.path, serialize_style_document(parsed))
            return self._style_document(self._read_style(record.path))

    def rename_style(self, name: str, request: ContentRename) -> StyleDocument:
        with self.lock:
            record = self._style_record(name)
            target = record.path.parent / f"{request.name}.md"
            if target != record.path:
                self._assert_style_available(request.name, record.path)
                self._rename(record.path, target)
            return self._style_document(self._read_style(target))

    def delete_style(self, name: str) -> dict[str, Any]:
        with self.lock:
            record = self._style_record(name)
            result = self._style_summary(record)
            self._remove_active(record.path)
            return {"name": result.name, "path": result.path, "deleted": True}

    def list_scenarios(self, kind: SourceKind, source_name: str) -> list[dict[str, Any]]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            return [self._scenario_dict(kind, scan.path, record) for record in scan.scenarios]

    def create_scenario(self, kind: SourceKind, source_name: str, request: ContentScenarioCreate) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            self._assert_scenario_available(scan, request.name)
            root = scan.path / "scenarios"
            self._ensure_directory(root, "场景目录")
            path = root / f"{request.name}.md"
            self._atomic_write(path, serialize_scenario_document("", ""))
            return self._scenario_dict(kind, scan.path, self._read_scenario(path))

    def update_scenario(self, kind: SourceKind, source_name: str, name: str, request: ContentScenarioUpdate) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            record = self._scenario_record(scan, name)
            self._atomic_write(record.path, serialize_scenario_document(request.description, request.opening))
            return self._scenario_dict(kind, scan.path, self._read_scenario(record.path))

    def rename_scenario(self, kind: SourceKind, source_name: str, name: str, request: ContentRename) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            record = self._scenario_record(scan, name)
            target = record.path.parent / f"{request.name}.md"
            if target != record.path:
                self._assert_scenario_available(scan, request.name, record.path)
                self._rename(record.path, target)
            return self._scenario_dict(kind, scan.path, self._read_scenario(target))

    def delete_scenario(self, kind: SourceKind, source_name: str, name: str) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            record = self._scenario_record(scan, name)
            result = self._scenario_dict(kind, scan.path, record)
            self._remove_active(record.path)
            return {"name": result["name"], "path": result["path"], "deleted": True}

    def list_entities(self, kind: SourceKind, source_name: str) -> list[dict[str, Any]]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            return [self._entity_dict(kind, scan.path, record, include_content=False) for record in scan.entities]

    def get_entity(self, kind: SourceKind, source_name: str, name: str) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            return self._entity_dict(kind, scan.path, self._entity_record(scan, name))

    def create_entity(self, kind: SourceKind, source_name: str, request: ContentEntityCreate) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            self._assert_entity_available(scan, request.name)
            self._ensure_directory(self.staging_dir, "内容 staging 目录")
            entities_dir = scan.path / "entities"
            self._ensure_directory(entities_dir, "实体目录")
            type_dir = entities_dir / request.type
            self._ensure_directory(type_dir, "实体类型目录")
            staging = self.staging_dir / f"entity-{uuid.uuid4()}"
            try:
                staging.mkdir()
                empty = ParsedEntity("", [], "world_truth", False, "")
                self._write_new(staging / "ENTITY.md", serialize_entity_document(empty).encode("utf-8"))
                self._sync_directory(staging)
                target = type_dir / request.name
                os.replace(staging, target)
                self._sync_directory(type_dir)
                self._sync_directory(self.staging_dir)
            finally:
                self._cleanup_hidden(staging)
            return self._entity_dict(kind, scan.path, self._read_entity(target, request.type))

    def update_entity(self, kind: SourceKind, source_name: str, name: str, request: ContentEntityUpdate) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            record = self._entity_record(scan, name)
            parsed = ParsedEntity(request.description, request.aliases, request.visibility, request.required, request.content)
            self._atomic_write(record.document, serialize_entity_document(parsed))
            return self._entity_dict(kind, scan.path, self._read_entity(record.directory, record.type))

    def move_entity(self, kind: SourceKind, source_name: str, name: str, request: ContentEntityMove) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            record = self._entity_record(scan, name)
            target_parent = scan.path / "entities" / request.type
            target = target_parent / request.name
            if target != record.directory:
                self._assert_entity_available(scan, request.name, record.directory)
                self._ensure_directory(target_parent, "实体类型目录")
                self._rename(record.directory, target)
            return self._entity_dict(kind, scan.path, self._read_entity(target, request.type))

    def delete_entity(self, kind: SourceKind, source_name: str, name: str) -> dict[str, Any]:
        with self.lock:
            scan = self._scan_source(self._source_directory(kind, source_name))
            record = self._entity_record(scan, name)
            result = self._entity_dict(kind, scan.path, record)
            self._remove_active(record.directory)
            return {"name": result["name"], "path": result["path"], "deleted": True}

    def _source_directories(self, kind: SourceKind) -> list[Path]:
        root = self._active_dir(kind)
        if root.is_symlink():
            raise UnsafeContentError("内容库必须是非符号链接目录")
        if not root.exists():
            return []
        self._require_directory(root, "内容库")
        result: list[Path] = []
        names: set[str] = set()
        for path in root.iterdir():
            self._require_directory(path, "世界或模组")
            name = self._safe_name(path.name)
            key = content_name_key(name)
            if key in names:
                raise UnsafeContentError(f"内容名称重复：{name}")
            names.add(key)
            result.append(path)
        return sorted(result, key=lambda item: content_name_key(item.name))

    def _style_records(self) -> list[StyleRecord]:
        root = self.data_dir / "styles"
        if not root.exists() and not root.is_symlink():
            return []
        self._require_directory(root, "文风库")
        result: list[StyleRecord] = []
        names: set[str] = set()
        for path in root.iterdir():
            if self._is_internal_temporary(path, target_suffix=".md"):
                continue
            self._require_regular(path, "文风文件")
            if path.suffix != ".md":
                raise UnsafeContentError(f"文风库只允许 Markdown 文件：{path.name}")
            name = self._safe_name(path.stem)
            key = content_name_key(name)
            if key in names:
                raise UnsafeContentError(f"文风名称重复：{name}")
            names.add(key)
            result.append(self._read_style(path))
        return sorted(result, key=lambda item: content_name_key(item.path.stem))

    def _read_style(self, path: Path) -> StyleRecord:
        try:
            parsed = parse_style_document(self._read_text(path, STYLE_DOCUMENT_LIMIT, "文风"))
        except ValueError as error:
            raise UnsafeContentError(str(error)) from error
        return StyleRecord(path, parsed)

    def _style_record(self, name: str) -> StyleRecord:
        safe_name = self._safe_name(name)
        key = content_name_key(safe_name)
        for record in self._style_records():
            if content_name_key(record.path.stem) == key:
                return record
        raise ContentNotFoundError(f"文风不存在：{safe_name}")

    def _assert_style_available(self, name: str, current: Path | None = None) -> None:
        key = content_name_key(name)
        for record in self._style_records():
            if record.path != current and content_name_key(record.path.stem) == key:
                raise ContentConflictError(f"文风名称已存在：{name}")

    def _source_directory(self, kind: SourceKind, name: str) -> Path:
        safe_name = self._safe_name(name)
        key = content_name_key(safe_name)
        for path in self._source_directories(kind):
            if content_name_key(path.name) == key:
                return path
        label = "世界" if kind == "world" else "模组"
        raise ContentNotFoundError(f"{label}不存在：{safe_name}")

    def _scan_source(self, source: Path) -> SourceScan:
        self._require_directory(source, "世界或模组")
        description_path = source / "description.md"
        description = ""
        if description_path.exists() or description_path.is_symlink():
            try:
                description = content_description(self._read_text(description_path, 4_000, "简介"))
            except ValueError as error:
                raise UnsafeContentError(str(error)) from error
        return SourceScan(source, description, self._scan_scenarios(source), self._scan_entities(source))

    def _scan_scenarios(self, source: Path) -> list[ScenarioRecord]:
        root = source / "scenarios"
        if not root.exists() and not root.is_symlink():
            return []
        self._require_directory(root, "场景目录")
        result: list[ScenarioRecord] = []
        names: set[str] = set()
        for path in root.iterdir():
            if self._is_internal_temporary(path, target_suffix=".md"):
                continue
            self._require_regular(path, "场景文件")
            if path.suffix != ".md":
                raise UnsafeContentError(f"场景目录只允许 Markdown 文件：{path.name}")
            name = self._safe_name(path.stem)
            key = content_name_key(name)
            if key in names:
                raise UnsafeContentError(f"来源内场景名称重复：{name}")
            names.add(key)
            result.append(self._read_scenario(path))
        return sorted(result, key=lambda item: content_name_key(item.path.stem))

    def _scan_entities(self, source: Path) -> list[EntityRecord]:
        root = source / "entities"
        if not root.exists() and not root.is_symlink():
            return []
        self._require_directory(root, "实体目录")
        result: list[EntityRecord] = []
        names: set[str] = set()
        for type_dir in root.iterdir():
            self._require_directory(type_dir, "实体类型目录")
            try:
                checked_type = entity_type(type_dir.name)
            except ValueError as error:
                raise UnsafeContentError(str(error)) from error
            for directory in type_dir.iterdir():
                self._require_directory(directory, "实体目录")
                name = self._safe_name(directory.name)
                key = content_name_key(name)
                if key in names:
                    raise UnsafeContentError(f"来源内实体名称重复：{name}")
                names.add(key)
                children = [
                    path for path in directory.iterdir() if not self._is_internal_temporary(path, target_name="ENTITY.md")
                ]
                if len(children) != 1 or children[0].name != "ENTITY.md":
                    raise UnsafeContentError(f"实体目录必须只包含 ENTITY.md：{name}")
                result.append(self._read_entity(directory, checked_type))
        return sorted(result, key=lambda item: content_name_key(item.directory.name))

    def _read_scenario(self, path: Path) -> ScenarioRecord:
        try:
            description, opening = parse_scenario_document(self._read_text(path, SCENARIO_DOCUMENT_LIMIT, "场景"))
        except ValueError as error:
            raise UnsafeContentError(str(error)) from error
        return ScenarioRecord(path, description, opening)

    def _read_entity(self, directory: Path, checked_type: str) -> EntityRecord:
        document = directory / "ENTITY.md"
        try:
            parsed = parse_entity_document(self._read_text(document, ENTITY_DOCUMENT_LIMIT, "实体"))
        except ValueError as error:
            raise UnsafeContentError(str(error)) from error
        return EntityRecord(directory, document, checked_type, parsed)

    def _scenario_record(self, scan: SourceScan, name: str) -> ScenarioRecord:
        safe_name = self._safe_name(name)
        key = content_name_key(safe_name)
        for record in scan.scenarios:
            if content_name_key(record.path.stem) == key:
                return record
        raise ContentNotFoundError(f"场景不存在：{safe_name}")

    def _entity_record(self, scan: SourceScan, name: str) -> EntityRecord:
        safe_name = self._safe_name(name)
        key = content_name_key(safe_name)
        for record in scan.entities:
            if content_name_key(record.directory.name) == key:
                return record
        raise ContentNotFoundError(f"实体不存在：{safe_name}")

    def _assert_source_available(self, kind: SourceKind, name: str, current: Path | None = None) -> None:
        key = content_name_key(name)
        for path in self._source_directories(kind):
            if path != current and content_name_key(path.name) == key:
                raise ContentConflictError(f"内容名称已存在：{name}")

    @staticmethod
    def _assert_scenario_available(scan: SourceScan, name: str, current: Path | None = None) -> None:
        key = content_name_key(name)
        for record in scan.scenarios:
            if record.path != current and content_name_key(record.path.stem) == key:
                raise ContentConflictError(f"场景名称已存在：{name}")

    @staticmethod
    def _assert_entity_available(scan: SourceScan, name: str, current: Path | None = None) -> None:
        key = content_name_key(name)
        for record in scan.entities:
            if record.directory != current and content_name_key(record.directory.name) == key:
                raise ContentConflictError(f"实体名称已存在：{name}")

    def _source_summary(self, kind: SourceKind, scan: SourceScan) -> dict[str, Any]:
        return {
            "name": scan.path.name,
            "description": scan.description,
            "path": self._source_relative_path(kind, scan.path.name),
            "scenario_count": len(scan.scenarios),
            "entity_count": len(scan.entities),
        }

    def _scenario_dict(self, kind: SourceKind, source: Path, record: ScenarioRecord) -> dict[str, Any]:
        return {
            "name": record.path.stem,
            "description": record.description,
            "opening": record.opening,
            "path": f"{self._source_relative_path(kind, source.name)}/scenarios/{record.path.name}",
        }

    def _entity_dict(self, kind: SourceKind, source: Path, record: EntityRecord, include_content: bool = True) -> dict[str, Any]:
        result = {
            "name": record.directory.name,
            "type": record.type,
            "description": record.parsed.description,
            "aliases": record.parsed.aliases,
            "visibility": record.parsed.visibility,
            "required": record.parsed.required,
            "path": f"{self._source_relative_path(kind, source.name)}/entities/{record.type}/{record.directory.name}/ENTITY.md",
        }
        if include_content:
            result["content"] = record.parsed.content
        return result

    @staticmethod
    def _style_relative_path(name: str) -> str:
        return style_path(f"styles/{name}.md")

    def _style_summary(self, record: StyleRecord) -> StyleSummary:
        return StyleSummary(
            path=self._style_relative_path(record.path.stem),
            name=record.path.stem,
            description=record.parsed.description,
            enabled=record.parsed.enabled,
        )

    def _style_document(self, record: StyleRecord) -> StyleDocument:
        return StyleDocument(**self._style_summary(record).model_dump(), content=record.parsed.content)

    def _active_dir(self, kind: SourceKind) -> Path:
        return self.data_dir / ("worlds" if kind == "world" else "mods")

    def _prepare_private_dirs(self) -> None:
        self._ensure_directory(self.data_dir, "数据根目录")
        self._ensure_directory(self.staging_dir, "内容 staging 目录")
        self._ensure_directory(self.deleted_dir, "内容删除目录")

    def _ensure_directory(self, path: Path, subject: str) -> None:
        if path.exists() or path.is_symlink():
            self._require_directory(path, subject)
            return
        path.mkdir()
        self._sync_directory(path.parent)

    def _rename(self, source: Path, target: Path) -> None:
        if target.exists() or target.is_symlink():
            raise ContentConflictError(f"目标名称已存在：{target.name}")
        os.replace(source, target)
        self._sync_directory(source.parent)
        if target.parent != source.parent:
            self._sync_directory(target.parent)

    def _remove_active(self, path: Path) -> None:
        self._prepare_private_dirs()
        deleted = self.deleted_dir / f"{path.name}-{uuid.uuid4()}"
        os.replace(path, deleted)
        self._sync_directory(path.parent)
        self._sync_directory(self.deleted_dir)
        self._cleanup_hidden(deleted)

    def _atomic_write(self, path: Path, text: str) -> None:
        temporary = path.parent / f".{path.name}.{uuid.uuid4()}.tmp"
        try:
            self._write_new(temporary, text.encode("utf-8"))
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        finally:
            self._cleanup_hidden(temporary)

    @staticmethod
    def _is_internal_temporary(
        path: Path, *, target_name: str | None = None, target_suffix: str | None = None
    ) -> bool:
        try:
            if not stat.S_ISREG(path.lstat().st_mode):
                return False
        except FileNotFoundError:
            return False
        return is_internal_temporary_name(path.name, target_name=target_name, target_suffix=target_suffix)

    @staticmethod
    def _cleanup_hidden(path: Path) -> None:
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    @staticmethod
    def _write_new(path: Path, data: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _read_text(path: Path, character_limit: int, subject: str) -> str:
        file_stat = ContentStore._require_regular(path, subject)
        if file_stat.st_size > character_limit * 4 + 1024:
            raise UnsafeContentError(f"{subject}超过大小限制")
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError as error:
            raise UnsafeContentError(f"{subject}不是合法 UTF-8") from error
        if len(text) > character_limit:
            raise UnsafeContentError(f"{subject}超过大小限制")
        return text

    @staticmethod
    def _require_regular(path: Path, subject: str) -> os.stat_result:
        try:
            file_stat = path.lstat()
        except FileNotFoundError as error:
            raise UnsafeContentError(f"{subject}不存在：{path.name}") from error
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            raise UnsafeContentError(f"{subject}必须是非符号链接普通文件：{path.name}")
        return file_stat

    @staticmethod
    def _require_directory(path: Path, subject: str) -> None:
        try:
            path_stat = path.lstat()
        except FileNotFoundError as error:
            raise UnsafeContentError(f"{subject}不存在：{path.name}") from error
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
            raise UnsafeContentError(f"{subject}必须是非符号链接目录：{path.name}")

    @staticmethod
    def _safe_name(value: Any) -> str:
        try:
            return content_name(value)
        except ValueError as error:
            raise UnsafeContentError(str(error)) from error

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _source_relative_path(kind: SourceKind, name: str) -> str:
        return f"{'worlds' if kind == 'world' else 'mods'}/{name}"
