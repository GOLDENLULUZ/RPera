from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from collections import defaultdict
from collections.abc import Generator
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path, PurePosixPath
from threading import Lock, RLock
from typing import Any, BinaryIO

from PIL import Image, ImageSequence, UnidentifiedImageError

from .content import (
    ParsedEntity,
    character_snapshot_path,
    content_description,
    content_name,
    content_name_key,
    entity_aliases,
    entity_snapshot_path,
    entity_type,
    is_internal_temporary_name,
    location_snapshot_path,
    parse_entity_document,
    parse_scenario_document,
    serialize_entity_document,
)
from .entities import EntityEntry, EntityStore
from .models import (
    EventPage,
    GeneratedImageSummary,
    LockedModSummary,
    ModPage,
    ModSummary,
    RuntimeEvent,
    SaveCreate,
    SaveEntityUpdate,
    SavePage,
    SaveSummary,
    Scenario,
    ScenarioOption,
    ScenarioRef,
    ScenarioSummary,
    TracePage,
    TurnImageSummary,
    TurnSummary,
    WorldSummary,
)
from .story_summary import StorySummaryStore
from .trace import TRACE_EVENT_TYPES, project_trace_event


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


IMAGE_ARTIFACT_MAX_BYTES = 10 * 1024 * 1024
IMAGE_ARTIFACT_MAX_DIMENSION = 8192
IMAGE_MIME_TYPES = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
}
IMAGE_EXTENSIONS = {
    "PNG": ".png",
    "JPEG": ".jpg",
}


class SaveBusyError(Exception):
    pass


class SaveEntityConflictError(Exception):
    pass


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    player_input TEXT NOT NULL,
    narrative TEXT,
    status TEXT NOT NULL CHECK(status IN ('running', 'completed', 'failed', 'interrupted')),
    root_session_id TEXT,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    agent TEXT NOT NULL,
    parent_session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    kind TEXT NOT NULL DEFAULT 'model' CHECK(kind IN ('model', 'story')),
    model TEXT,
    error TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS parts (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    type TEXT NOT NULL CHECK(type IN ('text', 'reasoning', 'tool')),
    content TEXT,
    provider_call_id TEXT,
    tool_name TEXT,
    input TEXT,
    state TEXT CHECK(state IN ('pending', 'running', 'completed', 'error')),
    output TEXT,
    child_session_id TEXT REFERENCES sessions(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(message_id, sequence),
    CHECK((type = 'tool' AND state IS NOT NULL) OR (type IN ('text', 'reasoning') AND state IS NULL))
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id TEXT REFERENCES turns(id),
    type TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turn_images (
    id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    stored_name TEXT NOT NULL UNIQUE,
    original_name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    byte_size INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(turn_id, position)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_running_turn
ON turns(status) WHERE status = 'running';

CREATE UNIQUE INDEX IF NOT EXISTS one_root_session
ON sessions(turn_id) WHERE parent_session_id IS NULL;

CREATE UNIQUE INDEX IF NOT EXISTS message_sequence
ON messages(session_id, sequence);

"""


class TurnAlreadyRunningError(RuntimeError):
    pass


class TurnRetryError(RuntimeError):
    pass


def _read_description(root: Path) -> str:
    path = root / "description.md"
    if not path.exists() and not path.is_symlink():
        return ""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"简介文件不是普通文件：{path}")
    return content_description(path.read_text(encoding="utf-8"))


def _read_scenario(path: Path) -> Scenario:
    description, opening = parse_scenario_document(path.read_text(encoding="utf-8"))
    return Scenario(description=description, opening=opening)


def _scenario_summaries(root: Path) -> list[ScenarioSummary]:
    scenarios_dir = root / "scenarios"
    if not scenarios_dir.exists() and not scenarios_dir.is_symlink():
        return []
    if scenarios_dir.is_symlink() or not scenarios_dir.is_dir():
        raise ValueError(f"场景路径不是普通目录：{scenarios_dir}")
    discovered: list[tuple[str, Scenario]] = []
    names: set[str] = set()
    for path in sorted(scenarios_dir.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"场景目录只允许 Markdown 文件：{path.name}")
        if is_internal_temporary_name(path.name, target_suffix=".md"):
            continue
        if path.suffix != ".md":
            raise ValueError(f"场景目录只允许 Markdown 文件：{path.name}")
        name = content_name(path.stem)
        key = content_name_key(name)
        if key in names:
            raise ValueError(f"来源内场景名称重复：{name}")
        names.add(key)
        discovered.append((name, _read_scenario(path)))
    return [
        ScenarioSummary(name=name, description=scenario.description)
        for name, scenario in sorted(discovered, key=lambda item: content_name_key(item[0]))
    ]


def _world_summary(path: Path) -> WorldSummary:
    return WorldSummary(
        name=content_name(path.name),
        description=_read_description(path),
        scenarios=_scenario_summaries(path),
    )


def _mod_summary(path: Path) -> ModSummary:
    return ModSummary(
        name=content_name(path.name),
        description=_read_description(path),
        scenarios=_scenario_summaries(path),
    )


class WorldLibrary:
    def __init__(self, data_dir: Path, content_lock: Any | None = None) -> None:
        self.worlds_dir = data_dir / "worlds"
        self.content_lock = content_lock if content_lock is not None else RLock()

    def list_worlds(self) -> list[WorldSummary]:
        with self.content_lock:
            return [summary for _, summary in self._all_world_paths()]

    def get_world_path(self, world_name: str) -> Path:
        with self.content_lock:
            key = content_name_key(world_name)
            for path, summary in self._all_world_paths():
                if content_name_key(summary.name) == key:
                    return path
            raise KeyError(f"未知世界观：{world_name}")

    def _all_world_paths(self) -> list[tuple[Path, WorldSummary]]:
        if self.worlds_dir.is_symlink():
            raise ValueError("世界库根目录不允许使用符号链接")
        if not self.worlds_dir.exists():
            return []
        if not self.worlds_dir.is_dir():
            raise ValueError("世界库根目录必须是目录")
        discovered: list[tuple[Path, WorldSummary]] = []
        names: set[str] = set()
        for path in sorted(self.worlds_dir.iterdir()):
            if path.is_symlink():
                raise ValueError(f"世界根目录不允许使用符号链接：{path.name}")
            if not path.is_dir():
                raise ValueError(f"世界库只允许目录：{path.name}")
            summary = _world_summary(path)
            key = content_name_key(summary.name)
            if key in names:
                raise ValueError(f"世界名称重复：{summary.name}")
            names.add(key)
            discovered.append((path, summary))
        return discovered


class ModLibrary:
    def __init__(self, data_dir: Path, content_lock: Any | None = None) -> None:
        self.mods_dir = data_dir / "mods"
        self.content_lock = content_lock if content_lock is not None else RLock()

    def list_mods(self, page: int = 1, page_size: int = 10) -> ModPage:
        with self.content_lock:
            mods = self._all_mods()
            total = len(mods)
            total_pages = math.ceil(total / page_size)
            page = min(page, max(total_pages, 1))
            start = (page - 1) * page_size
            return ModPage(
                mods=mods[start : start + page_size],
                page=page,
                page_size=page_size,
                total=total,
                total_pages=total_pages,
            )

    def get_mod_path(self, mod_name: str) -> Path:
        with self.content_lock:
            key = content_name_key(mod_name)
            for path, summary in self._all_mod_paths():
                if content_name_key(summary.name) == key:
                    return path
            raise KeyError(f"未知模组：{mod_name}")

    def get_mod(self, mod_name: str) -> ModSummary:
        with self.content_lock:
            key = content_name_key(mod_name)
            for _, summary in self._all_mod_paths():
                if content_name_key(summary.name) == key:
                    return summary
            raise KeyError(f"未知模组：{mod_name}")

    def _all_mods(self) -> list[ModSummary]:
        return [summary for _, summary in self._all_mod_paths()]

    def _all_mod_paths(self) -> list[tuple[Path, ModSummary]]:
        if self.mods_dir.is_symlink():
            raise ValueError("模组库根目录不允许使用符号链接")
        if not self.mods_dir.exists():
            return []
        if not self.mods_dir.is_dir():
            raise ValueError("模组库根目录必须是目录")
        discovered: list[tuple[Path, ModSummary]] = []
        names: set[str] = set()
        for path in sorted(self.mods_dir.iterdir()):
            if path.is_symlink():
                raise ValueError(f"模组根目录不允许使用符号链接：{path.name}")
            if not path.is_dir():
                raise ValueError(f"模组库只允许目录：{path.name}")
            summary = _mod_summary(path)
            key = content_name_key(summary.name)
            if key in names:
                raise ValueError(f"模组名称重复：{summary.name}")
            names.add(key)
            discovered.append((path, summary))
        return discovered


class SaveStore:
    def __init__(self, data_dir: Path, worlds: WorldLibrary, mods: ModLibrary, content_lock: Any | None = None) -> None:
        self.saves_dir = data_dir / "saves"
        self.deleted_saves_dir = data_dir / ".deleted_saves"
        self.worlds = worlds
        self.mods = mods
        self.content_lock = content_lock if content_lock is not None else RLock()
        self.saves_dir.mkdir(parents=True, exist_ok=True)
        self.deleted_saves_dir.mkdir(parents=True, exist_ok=True)
        self._save_locks: dict[str, Lock] = defaultdict(Lock)

    def create_save(self, request: SaveCreate) -> SaveSummary:
        save_id = str(uuid.uuid4())
        save_dir = self.saves_dir / save_id
        staging_dir = self.saves_dir / f".creating-{save_id}"
        try:
            staging_dir.mkdir()
            state_dir = staging_dir / "current"
            state_dir.mkdir()
            with self.content_lock:
                world_path = self.worlds.get_world_path(request.world_name)
                world = _world_summary(world_path)
                self._validate_world_tree(world_path)
                if len({content_name_key(name) for name in request.mod_names}) != len(request.mod_names):
                    raise ValueError("不能重复启用同一个模组")
                selected_mods = [(self.mods.get_mod_path(name), self.mods.get_mod(name)) for name in request.mod_names]
                for path, _ in selected_mods:
                    self._validate_world_tree(path)
                scenarios = self._scenario_options(world.name, request.mod_names)
                scenario = request.scenario or (scenarios[0].as_ref() if scenarios else None)
                if scenario and not any(item.matches(scenario) for item in scenarios):
                    raise KeyError(f"世界观中不存在场景：{scenario.name}")
                snapshot = state_dir / "world_snapshot"
                shutil.copytree(world_path, snapshot, symlinks=True)
                mods_dir = snapshot / "mods"
                mods_dir.mkdir()
                locked_mods: list[LockedModSummary] = []
                for path, mod in sorted(selected_mods, key=lambda item: content_name_key(item[1].name)):
                    source_hash = self._hash_tree(path)
                    target = mods_dir / mod.name
                    shutil.copytree(path, target, symlinks=True)
                    if source_hash != self._hash_tree(path) or source_hash != self._hash_tree(target):
                        raise ValueError(f"模组内容在复制期间发生变化：{mod.name}")
                    locked_mods.append(LockedModSummary(name=mod.name, source_hash=source_hash))
            self._validate_world_tree(snapshot)
            EntityStore(snapshot).entries()
            opening = self._scenario_opening(snapshot, scenario)
            (state_dir / "artifacts").mkdir()
            (state_dir / "exports").mkdir()
            (state_dir / "turns").mkdir()
            (staging_dir / "revisions").mkdir()
            now = utc_now()
            save_data = {
                "id": save_id,
                "name": request.name,
                "source_world_name": world.name,
                "source_world_hash": self._hash_tree(snapshot),
                "scenario": scenario.model_dump() if scenario else None,
                "opening": opening,
                "play_mode": request.play_mode,
                "narrative_person": request.narrative_person,
                "language": request.language,
                "other_requirements": request.other_requirements,
                "mods": [mod.model_dump() for mod in locked_mods],
                "created_at": now,
                "last_played_at": now,
                "branch_source_save_id": None,
                "branch_source_turn_id": None,
                "branch_source_save_name": None,
                "branch_source_turn_number": None,
            }
            self._write_save(staging_dir, save_data)
            self._initialize_database(state_dir / "state.sqlite3")
            self._copy_state(state_dir, staging_dir / "revisions" / "opening")
            os.replace(staging_dir, save_dir)
            return SaveSummary.model_validate(save_data)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def scenario_options(self, world_name: str, mod_names: list[str]) -> list[ScenarioOption]:
        with self.content_lock:
            return self._scenario_options(world_name, mod_names)

    def _scenario_options(self, world_name: str, mod_names: list[str]) -> list[ScenarioOption]:
        world = _world_summary(self.worlds.get_world_path(world_name))
        options = [
            ScenarioOption(**scenario.model_dump(), source_kind="world", source_name=world.name)
            for scenario in world.scenarios
        ]
        if len({content_name_key(name) for name in mod_names}) != len(mod_names):
            raise ValueError("不能重复启用同一个模组")
        for mod_name in sorted(mod_names, key=content_name_key):
            mod = self.mods.get_mod(mod_name)
            options.extend(
                ScenarioOption(
                    source_kind="mod",
                    source_name=mod.name,
                    name=scenario.name,
                    description=scenario.description,
                )
                for scenario in mod.scenarios
            )
        return options

    def read_image_artifact(self, save_id: str, path: str) -> dict[str, Any]:
        relative = PurePosixPath(path)
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "artifacts":
            raise ValueError("图片路径必须是当前存档 artifacts/ 下的顶层文件")
        filename = relative.parts[1]
        if filename in {"", ".", ".."} or "\\" in path:
            raise ValueError("图片路径不是合法的存档 artifact 路径")
        artifacts = self._state_dir(save_id) / "artifacts"
        if artifacts.is_symlink() or not artifacts.is_dir():
            raise ValueError("存档 artifacts 根路径必须是非符号链接目录")
        target = artifacts / filename
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"图片 artifact 不存在或不是普通文件：{path}")
        size = target.stat().st_size
        if size > IMAGE_ARTIFACT_MAX_BYTES:
            raise ValueError(f"图片 artifact 超过 {IMAGE_ARTIFACT_MAX_BYTES // (1024 * 1024)} MiB 上限：{path}")
        data = target.read_bytes()
        if len(data) != size:
            raise ValueError(f"读取图片 artifact 时文件发生变化：{path}")
        image_info = self._validate_image_data(data, path)
        return {
            "path": relative.as_posix(),
            "mime_type": image_info[0],
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "data": base64.b64encode(data).decode("ascii"),
        }

    def write_generated_image_artifact(self, save_id: str, data: bytes) -> dict[str, Any]:
        with self._save_locks[save_id]:
            artifacts = self._state_dir(save_id) / "artifacts"
            if artifacts.is_symlink() or not artifacts.is_dir():
                raise ValueError("存档 artifacts 根路径必须是非符号链接目录")
            mime_type, width, height, extension = self._validate_image_data(data, "生成图片")
            image_id = str(uuid.uuid4())
            path = f"artifacts/character-portrait-{image_id}{extension}"
            target = artifacts / PurePosixPath(path).name
            temporary = artifacts / f".writing-character-portrait-{image_id}"
            published = False
            try:
                with temporary.open("xb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, target)
                published = True
                self._sync_directory(artifacts)
            except Exception:
                temporary.unlink(missing_ok=True)
                if published:
                    target.unlink(missing_ok=True)
                raise
            return {
                "path": path,
                "mime_type": mime_type,
                "size": len(data),
                "width": width,
                "height": height,
                "sha256": hashlib.sha256(data).hexdigest(),
            }

    @staticmethod
    def _validate_image_data(data: bytes, label: str) -> tuple[str, int, int, str]:
        if len(data) > IMAGE_ARTIFACT_MAX_BYTES:
            raise ValueError(f"图片超过 {IMAGE_ARTIFACT_MAX_BYTES // (1024 * 1024)} MiB 上限：{label}")
        return SaveStore._inspect_image(data, label)

    @staticmethod
    def _inspect_image(source: bytes | Path, label: str) -> tuple[str, int, int, str]:
        def open_image():
            return Image.open(BytesIO(source) if isinstance(source, bytes) else source)

        try:
            with open_image() as image:
                image_format = str(image.format).upper()
                mime_type = IMAGE_MIME_TYPES.get(image_format)
                extension = IMAGE_EXTENSIONS.get(image_format)
                if mime_type is None or extension is None:
                    raise ValueError(f"图片只支持 PNG 或 JPEG：{label}")
                width, height = image.size
                if width > IMAGE_ARTIFACT_MAX_DIMENSION or height > IMAGE_ARTIFACT_MAX_DIMENSION:
                    raise ValueError(f"图片任一边不能超过 {IMAGE_ARTIFACT_MAX_DIMENSION} 像素：{label}")
                image.verify()
            with open_image() as image:
                for frame in ImageSequence.Iterator(image):
                    _ = frame.load()
        except (Image.DecompressionBombError, EOFError, UnidentifiedImageError, OSError) as error:
            raise ValueError(f"图片不是完整有效的 PNG 或 JPEG：{label}") from error
        return mime_type, width, height, extension

    @staticmethod
    def _scenario_opening(snapshot: Path, scenario_ref: ScenarioRef | None) -> str:
        if scenario_ref is None:
            return ""
        if scenario_ref.source_kind == "mod":
            source_root = snapshot / "mods" / scenario_ref.source_name
        else:
            source_root = snapshot
            source_root = snapshot
        scenario_path = source_root / "scenarios" / f"{scenario_ref.name}.md"
        if not scenario_path.is_file():
            raise ValueError("场景文件不存在")
        return _read_scenario(scenario_path).opening

    def list_saves(self, page: int = 1, page_size: int = 10) -> SavePage:
        saves = self._all_saves()
        saves.sort(key=lambda save: (save.last_played_at, save.created_at, save.id), reverse=True)
        total = len(saves)
        total_pages = math.ceil(total / page_size)
        page = min(page, max(total_pages, 1))
        start = (page - 1) * page_size
        return SavePage(
            saves=saves[start : start + page_size],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )

    def get_save(self, save_id: str) -> SaveSummary:
        return SaveSummary.model_validate(self._load_save(self._save_dir(save_id)))

    def branch_save(self, save_id: str, turn_id: str, name: str) -> SaveSummary:
        with self._save_locks[save_id]:
            source = self._save_dir(save_id)
            with self.connect(save_id) as db:
                row = db.execute("SELECT status FROM turns WHERE id = ?", (turn_id,)).fetchone()
            if row is None:
                raise KeyError(f"未知回合：{turn_id}")
            if row["status"] != "completed":
                raise ValueError("只能从已完成的回合创建分支")
            revisions = source / "revisions"
            target_revision = revisions / turn_id
            if not target_revision.is_dir():
                with self.connect(save_id) as db:
                    latest = db.execute("SELECT id, status FROM turns ORDER BY rowid DESC LIMIT 1").fetchone()
                if latest and latest["id"] == turn_id:
                    self._checkpoint_turn_unlocked(save_id, turn_id)
                else:
                    raise ValueError("该回合缺少完整检查点，无法创建分支")
            new_id = str(uuid.uuid4())
            temporary = self.saves_dir / f".creating-{new_id}"
            target = self.saves_dir / new_id
            published = False
            try:
                temporary.mkdir()
                (temporary / "revisions").mkdir()
                # Keep earlier checkpoints so the new timeline can branch again.
                with sqlite3.connect(target_revision / "state.sqlite3") as db:
                    rows = db.execute("SELECT id, status FROM turns ORDER BY rowid").fetchall()
                self._copy_state(revisions / "opening", temporary / "revisions" / "opening")
                for row in rows:
                    previous_id = str(row[0])
                    if row[1] != "completed":
                        continue
                    if not (revisions / previous_id).is_dir():
                        raise ValueError(f"历史回合缺少完整检查点：{previous_id}")
                    self._copy_state(revisions / previous_id, temporary / "revisions" / previous_id)
                self._copy_state(target_revision, temporary / "current")
                data = self._load_save(source)
                now = utc_now()
                origin_name = data["name"]
                data.update({
                    "id": new_id, "name": name, "created_at": now, "last_played_at": now,
                    "branch_source_save_id": save_id, "branch_source_turn_id": turn_id,
                    "branch_source_save_name": origin_name, "branch_source_turn_number": len(rows),
                })
                self._write_save(temporary, data)
                os.replace(temporary, target)
                published = True
                self._sync_directory(self.saves_dir)
                return SaveSummary.model_validate(data)
            except Exception:
                if published:
                    shutil.rmtree(target)
                raise
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)

    def rename_save(self, save_id: str, name: str) -> SaveSummary:
        with self._save_locks[save_id]:
            save_dir = self._save_dir(save_id)
            with self.connect(save_id) as db:
                if db.execute("SELECT 1 FROM turns WHERE status = 'running' LIMIT 1").fetchone():
                    raise SaveBusyError("回合正在执行，暂不能重命名存档")
            save_data = self._load_save(save_dir)
            save_data["name"] = name
            self._write_save(save_dir, save_data)
            return SaveSummary.model_validate(save_data)

    def delete_save(self, save_id: str) -> None:
        save_dir = self._save_dir(save_id)
        with self.connect(save_id) as db:
            running_turn = db.execute(
                "SELECT 1 FROM turns WHERE status = 'running' LIMIT 1"
            ).fetchone()
        if running_turn is not None:
            raise SaveBusyError("当前存档有正在执行的回合，无法删除")

        deleted_dir = self.deleted_saves_dir / f"{save_dir.name}-{uuid.uuid4()}"
        os.replace(save_dir, deleted_dir)
        shutil.rmtree(deleted_dir)

    def searchable_entity_count(self, save_id: str) -> int:
        return sum(not entry.required for entry in self._entity_store(save_id).entries())

    def list_entities(self, save_id: str) -> list[dict[str, Any]]:
        with self._save_locks[save_id]:
            return [self._entry_dict(entry) for entry in self._entity_store(save_id).entries()]

    def get_save_entity(self, save_id: str, path: str) -> dict[str, Any]:
        checked = entity_snapshot_path(path)
        with self._save_locks[save_id]:
            return self._save_entity_unlocked(save_id, checked)

    def _save_entity_unlocked(self, save_id: str, path: str) -> dict[str, Any]:
        store = self._entity_store(save_id)
        if not any(entry.path == path for entry in store.entries()):
            raise KeyError(f"存档实体不存在：{path}")
        document = store.read_exact([path])[0]
        raw = (store.root / path).read_bytes()
        result = self._document_dict(document)
        result["type"] = path.split("/")[1 if path.startswith("entities/") else 3]
        result["revision"] = hashlib.sha256(raw).hexdigest()
        return result

    def _assert_save_entity_writable(self, save_id: str) -> None:
        with self.connect(save_id) as db:
            latest = db.execute("SELECT id, status FROM turns ORDER BY rowid DESC LIMIT 1").fetchone()
            if latest and latest["status"] == "running":
                raise SaveBusyError("回合正在执行，暂不能修改存档实体")
            if latest and latest["status"] == "completed" and not (
                self._save_dir(save_id) / "revisions" / str(latest["id"])
            ).is_dir():
                raise SaveBusyError("回合检查点尚未完成，暂不能修改存档实体")

    @staticmethod
    def _assert_save_entity_revision(current: dict[str, Any], revision: str) -> None:
        if current["revision"] != revision:
            raise SaveEntityConflictError("实体已被修改，请重新选择实体后再编辑")

    def create_save_entity(self, save_id: str, name: str, type: str) -> dict[str, Any]:
        checked_name = content_name(name)
        checked_type = entity_type(type)
        with self._save_locks[save_id]:
            save_dir = self._state_dir(save_id)
            self._assert_save_entity_writable(save_id)
            try:
                self._assert_entity_name_available(self._entity_store(save_id).entries(), "", checked_name)
            except ValueError as error:
                raise SaveEntityConflictError(str(error)) from error
            entities = save_dir / "world_snapshot" / "entities"
            self._ensure_real_directory(entities)
            parent = entities / checked_type
            self._ensure_real_directory(parent)
            staging_root = save_dir / ".entity-staging"
            self._ensure_real_directory(staging_root)
            staging = staging_root / f"entity-{uuid.uuid4()}"
            target = parent / checked_name
            try:
                staging.mkdir()
                document = serialize_entity_document(ParsedEntity("", [], "world_truth", False, ""))
                self._write_new_file(staging / "ENTITY.md", document.encode("utf-8"))
                self._sync_directory(staging)
                if target.exists() or target.is_symlink():
                    raise SaveEntityConflictError(f"实体名称已存在：{checked_name}")
                os.replace(staging, target)
                self._sync_directory(parent)
                self._sync_directory(staging_root)
            finally:
                if staging.exists() and not staging.is_symlink():
                    shutil.rmtree(staging, ignore_errors=True)
            return self._save_entity_unlocked(save_id, f"entities/{checked_type}/{checked_name}/ENTITY.md")

    def update_save_entity(self, save_id: str, path: str, request: SaveEntityUpdate) -> dict[str, Any]:
        checked = entity_snapshot_path(path)
        with self._save_locks[save_id]:
            self._assert_save_entity_writable(save_id)
            current = self._save_entity_unlocked(save_id, checked)
            self._assert_save_entity_revision(current, request.revision)
            if current["type"] in {"character", "location"}:
                try:
                    existing_profile = json.loads(current["content"])
                except json.JSONDecodeError:
                    existing_profile = None
                if isinstance(existing_profile, dict):
                    try:
                        updated_profile = json.loads(request.content)
                    except json.JSONDecodeError as error:
                        raise ValueError("JSON 实体正文编辑后必须保持为合法 JSON 对象") from error
                    if not isinstance(updated_profile, dict):
                        raise ValueError("JSON 实体正文编辑后必须保持为合法 JSON 对象")
            document = serialize_entity_document(ParsedEntity(
                request.description, request.aliases, request.visibility, request.required, request.content,
            ))
            target = self._state_dir(save_id) / "world_snapshot" / checked
            self._atomic_replace_file(target, document.encode("utf-8"))
            return self._save_entity_unlocked(save_id, checked)

    def move_save_entity(self, save_id: str, path: str, name: str, type: str, revision: str) -> dict[str, Any]:
        checked = entity_snapshot_path(path)
        checked_name = content_name(name)
        checked_type = entity_type(type)
        with self._save_locks[save_id]:
            self._assert_save_entity_writable(save_id)
            current = self._save_entity_unlocked(save_id, checked)
            self._assert_save_entity_revision(current, revision)
            prefix = self._entity_source_prefix(checked)
            try:
                self._assert_entity_name_available(
                    self._entity_store(save_id).entries(), prefix, checked_name, current_path=checked,
                )
            except ValueError as error:
                raise SaveEntityConflictError(str(error)) from error
            source = self._state_dir(save_id) / "world_snapshot" / checked
            target_parent = source.parent.parent.parent / checked_type
            self._ensure_real_directory(target_parent)
            target = target_parent / checked_name
            if source.parent == target:
                return current
            if target.exists() or target.is_symlink():
                raise SaveEntityConflictError(f"实体名称已存在：{checked_name}")
            os.replace(source.parent, target)
            self._sync_directory(source.parent.parent)
            if source.parent.parent != target_parent:
                self._sync_directory(target_parent)
            new_path = f"{prefix}entities/{checked_type}/{checked_name}/ENTITY.md"
            return self._save_entity_unlocked(save_id, new_path)

    def delete_save_entity(self, save_id: str, path: str, revision: str) -> dict[str, Any]:
        checked = entity_snapshot_path(path)
        with self._save_locks[save_id]:
            self._assert_save_entity_writable(save_id)
            current = self._save_entity_unlocked(save_id, checked)
            self._assert_save_entity_revision(current, revision)
            source = self._state_dir(save_id) / "world_snapshot" / checked
            staging = self._state_dir(save_id) / ".entity-staging"
            self._ensure_real_directory(staging)
            removed = staging / f"deleted-{uuid.uuid4()}"
            os.replace(source.parent, removed)
            self._sync_directory(source.parent.parent)
            self._sync_directory(staging)
            shutil.rmtree(removed)
            return {"path": checked, "deleted": True}

    def search_entities(self, save_id: str, query: str, limit: int = 12) -> list[dict[str, Any]]:
        return [self._entry_dict(entry) for entry in self._entity_store(save_id).search(query, limit)]

    def read_entities(self, save_id: str, paths: list[str]) -> list[dict[str, Any]]:
        return [self._document_dict(document) for document in self._entity_store(save_id).read(paths)]

    def read_entities_exact(self, save_id: str, paths: list[str]) -> list[dict[str, Any]]:
        return [self._document_dict(document) for document in self._entity_store(save_id).read_exact(paths)]

    def read_entities_for_character_edit(self, save_id: str, paths: list[str]) -> list[dict[str, Any]]:
        with self._save_locks[save_id]:
            documents = self._entity_store(save_id).read(paths)
            return [self._character_edit_document(save_id, document) for document in documents]

    def read_entities_exact_for_character_edit(self, save_id: str, paths: list[str]) -> list[dict[str, Any]]:
        with self._save_locks[save_id]:
            documents = self._entity_store(save_id).read_exact(paths)
            return [self._character_edit_document(save_id, document) for document in documents]

    def read_entities_for_location_edit(self, save_id: str, paths: list[str]) -> list[dict[str, Any]]:
        with self._save_locks[save_id]:
            documents = self._entity_store(save_id).read(paths)
            return [self._location_edit_document(save_id, document) for document in documents]

    def read_entities_exact_for_location_edit(self, save_id: str, paths: list[str]) -> list[dict[str, Any]]:
        with self._save_locks[save_id]:
            documents = self._entity_store(save_id).read_exact(paths)
            return [self._location_edit_document(save_id, document) for document in documents]

    def _entity_store(self, save_id: str) -> EntityStore:
        return EntityStore(self._state_dir(save_id) / "world_snapshot")

    def required_entity_paths(self, save_id: str) -> list[str]:
        return self._entity_store(save_id).required_paths()

    def required_entity_documents(self, save_id: str) -> list[dict[str, Any]]:
        return [self._document_dict(document) for document in self._entity_store(save_id).required_documents()]

    def create_character(
        self,
        save_id: str,
        name: str,
        description: str,
        aliases: list[str],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        checked_name = content_name(name)
        document = self._character_document(description, aliases, profile, required=False)
        save_dir = self._state_dir(save_id)
        with self._save_locks[save_id]:
            store = self._entity_store(save_id)
            self._assert_entity_name_available(store.entries(), "", checked_name)
            snapshot = save_dir / "world_snapshot"
            entities = snapshot / "entities"
            characters = entities / "character"
            self._ensure_real_directory(entities)
            self._ensure_real_directory(characters)
            staging_root = save_dir / ".entity-staging"
            self._ensure_real_directory(staging_root)
            staging = staging_root / f"character-{uuid.uuid4()}"
            target = characters / checked_name
            try:
                staging.mkdir()
                self._write_new_file(staging / "ENTITY.md", document.encode("utf-8"))
                self._sync_directory(staging)
                if target.exists() or target.is_symlink():
                    raise ValueError(f"实体名称已存在：{checked_name}")
                os.replace(staging, target)
                self._sync_directory(characters)
                self._sync_directory(staging_root)
            finally:
                if staging.exists() and not staging.is_symlink():
                    shutil.rmtree(staging, ignore_errors=True)
            path = f"entities/character/{checked_name}/ENTITY.md"
            return self.read_entities_exact(save_id, [path])[0]

    def edit_character(
        self,
        save_id: str,
        path: str,
        old_text: str,
        new_text: str,
    ) -> dict[str, Any]:
        checked_path = character_snapshot_path(path)
        if not old_text:
            raise ValueError("old_text 不能为空")
        with self._save_locks[save_id]:
            existing = self._entity_store(save_id).read_exact([checked_path])[0]
            target = self._state_dir(save_id) / "world_snapshot" / checked_path
            current = target.read_text(encoding="utf-8")
            if current.count(old_text) != 1:
                raise ValueError("old_text 必须在角色文件中唯一出现一次")
            updated = current.replace(old_text, new_text, 1)
            parsed = parse_entity_document(updated)
            if parsed.visibility != existing.visibility or parsed.required != existing.required:
                raise ValueError("character_edit 不允许修改 visibility 或 required")
            try:
                old_profile = json.loads(existing.content)
            except json.JSONDecodeError:
                old_profile = None
            if isinstance(old_profile, dict):
                try:
                    new_profile = json.loads(parsed.content)
                except json.JSONDecodeError as error:
                    raise ValueError("JSON 角色正文编辑后必须保持为合法 JSON 对象") from error
                if not isinstance(new_profile, dict):
                    raise ValueError("JSON 角色正文编辑后必须保持为合法 JSON 对象")
            self._atomic_replace_file(target, updated.encode("utf-8"))
            return self.read_entities_exact(save_id, [checked_path])[0]

    def rename_character(self, save_id: str, path: str, new_name: str) -> dict[str, Any]:
        checked_path = character_snapshot_path(path)
        checked_name = content_name(new_name)
        with self._save_locks[save_id]:
            store = self._entity_store(save_id)
            store.read_exact([checked_path])
            source_prefix = self._entity_source_prefix(checked_path)
            self._assert_entity_name_available(store.entries(), source_prefix, checked_name, current_path=checked_path)
            source_file = self._state_dir(save_id) / "world_snapshot" / checked_path
            source = source_file.parent
            target = source.parent / checked_name
            if target == source:
                return self.read_entities_exact(save_id, [checked_path])[0]
            if target.exists() or target.is_symlink():
                raise ValueError(f"实体名称已存在：{checked_name}")
            os.replace(source, target)
            self._sync_directory(source.parent)
            parts = list(Path(checked_path).parts)
            parts[-2] = checked_name
            renamed_path = "/".join(parts)
            return self.read_entities_exact(save_id, [renamed_path])[0]

    def create_location(
        self,
        save_id: str,
        name: str,
        description: str,
        aliases: list[str],
        profile: dict[str, Any],
    ) -> dict[str, Any]:
        checked_name = content_name(name)
        document = self._location_document(description, aliases, profile, required=False)
        save_dir = self._state_dir(save_id)
        with self._save_locks[save_id]:
            store = self._entity_store(save_id)
            self._assert_entity_name_available(store.entries(), "", checked_name)
            snapshot = save_dir / "world_snapshot"
            entities = snapshot / "entities"
            locations = entities / "location"
            self._ensure_real_directory(entities)
            self._ensure_real_directory(locations)
            staging_root = save_dir / ".entity-staging"
            self._ensure_real_directory(staging_root)
            staging = staging_root / f"location-{uuid.uuid4()}"
            target = locations / checked_name
            try:
                staging.mkdir()
                self._write_new_file(staging / "ENTITY.md", document.encode("utf-8"))
                self._sync_directory(staging)
                if target.exists() or target.is_symlink():
                    raise ValueError(f"实体名称已存在：{checked_name}")
                os.replace(staging, target)
                self._sync_directory(locations)
                self._sync_directory(staging_root)
            finally:
                if staging.exists() and not staging.is_symlink():
                    shutil.rmtree(staging, ignore_errors=True)
            path = f"entities/location/{checked_name}/ENTITY.md"
            return self.read_entities_exact(save_id, [path])[0]

    def edit_location(
        self,
        save_id: str,
        path: str,
        old_text: str,
        new_text: str,
    ) -> dict[str, Any]:
        checked_path = location_snapshot_path(path)
        if not old_text:
            raise ValueError("old_text 不能为空")
        with self._save_locks[save_id]:
            existing = self._entity_store(save_id).read_exact([checked_path])[0]
            target = self._state_dir(save_id) / "world_snapshot" / checked_path
            current = target.read_text(encoding="utf-8")
            if current.count(old_text) != 1:
                raise ValueError("old_text 必须在地点文件中唯一出现一次")
            updated = current.replace(old_text, new_text, 1)
            parsed = parse_entity_document(updated)
            if parsed.visibility != existing.visibility or parsed.required != existing.required:
                raise ValueError("location_edit 不允许修改 visibility 或 required")
            try:
                old_profile = json.loads(existing.content)
            except json.JSONDecodeError:
                old_profile = None
            if isinstance(old_profile, dict):
                try:
                    new_profile = json.loads(parsed.content)
                except json.JSONDecodeError as error:
                    raise ValueError("JSON 地点正文编辑后必须保持为合法 JSON 对象") from error
                if not isinstance(new_profile, dict):
                    raise ValueError("JSON 地点正文编辑后必须保持为合法 JSON 对象")
            self._atomic_replace_file(target, updated.encode("utf-8"))
            return self.read_entities_exact(save_id, [checked_path])[0]

    def rename_location(self, save_id: str, path: str, new_name: str) -> dict[str, Any]:
        checked_path = location_snapshot_path(path)
        checked_name = content_name(new_name)
        with self._save_locks[save_id]:
            store = self._entity_store(save_id)
            store.read_exact([checked_path])
            source_prefix = self._entity_source_prefix(checked_path)
            self._assert_entity_name_available(store.entries(), source_prefix, checked_name, current_path=checked_path)
            source_file = self._state_dir(save_id) / "world_snapshot" / checked_path
            source = source_file.parent
            target = source.parent / checked_name
            if target == source:
                return self.read_entities_exact(save_id, [checked_path])[0]
            if target.exists() or target.is_symlink():
                raise ValueError(f"实体名称已存在：{checked_name}")
            os.replace(source, target)
            self._sync_directory(source.parent)
            parts = list(Path(checked_path).parts)
            parts[-2] = checked_name
            renamed_path = "/".join(parts)
            return self.read_entities_exact(save_id, [renamed_path])[0]

    @staticmethod
    def _character_document(
        description: str,
        aliases: list[str],
        profile: dict[str, Any],
        *,
        required: bool,
    ) -> str:
        content = json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False)
        return serialize_entity_document(ParsedEntity(
            content_description(description),
            entity_aliases(aliases),
            "world_truth",
            required,
            content,
        ))

    @staticmethod
    def _location_document(
        description: str,
        aliases: list[str],
        profile: dict[str, Any],
        *,
        required: bool,
    ) -> str:
        content = json.dumps(profile, ensure_ascii=False, indent=2, allow_nan=False)
        return serialize_entity_document(ParsedEntity(
            content_description(description),
            entity_aliases(aliases),
            "world_truth",
            required,
            content,
        ))

    @staticmethod
    def _entity_source_prefix(path: str) -> str:
        parts = Path(path).parts
        return "" if parts[0] == "entities" else f"mods/{parts[1]}/"

    @staticmethod
    def _assert_entity_name_available(
        entries: list[EntityEntry],
        source_prefix: str,
        name: str,
        *,
        current_path: str | None = None,
    ) -> None:
        key = content_name_key(name)
        for entry in entries:
            if entry.path == current_path or SaveStore._entity_source_prefix(entry.path) != source_prefix:
                continue
            if content_name_key(entry.name) == key:
                raise ValueError(f"实体名称已存在：{name}")

    @staticmethod
    def _ensure_real_directory(path: Path) -> None:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"目录路径无效：{path.name}")
            return
        path.mkdir()
        SaveStore._sync_directory(path.parent)

    @staticmethod
    def _write_new_file(path: Path, data: bytes) -> None:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _atomic_replace_file(path: Path, data: bytes) -> None:
        temporary = path.parent / f".{path.name}.{uuid.uuid4()}.tmp"
        try:
            SaveStore._write_new_file(temporary, data)
            os.replace(temporary, path)
            SaveStore._sync_directory(path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _entry_dict(entry: Any) -> dict[str, Any]:
        return {
            "path": entry.path,
            "name": entry.name,
            "description": entry.description,
            "aliases": entry.aliases,
            "visibility": entry.visibility,
            "required": entry.required,
        }

    @staticmethod
    def _document_dict(document: Any) -> dict[str, Any]:
        return {
            "path": document.path,
            "name": document.name,
            "description": document.description,
            "aliases": document.aliases,
            "visibility": document.visibility,
            "content": document.content,
            "required": document.required,
        }

    def _character_edit_document(self, save_id: str, document: Any) -> dict[str, Any]:
        result = self._document_dict(document)
        try:
            checked_path = character_snapshot_path(document.path)
        except ValueError:
            return result
        target = self._state_dir(save_id) / "world_snapshot" / checked_path
        result.pop("content")
        result["document"] = target.read_text(encoding="utf-8")
        return result

    def _location_edit_document(self, save_id: str, document: Any) -> dict[str, Any]:
        result = self._document_dict(document)
        try:
            checked_path = location_snapshot_path(document.path)
        except ValueError:
            return result
        target = self._state_dir(save_id) / "world_snapshot" / checked_path
        result.pop("content")
        result["document"] = target.read_text(encoding="utf-8")
        return result

    def create_turn(
        self,
        save_id: str,
        player_input: str,
        context_turns: int,
        images: list[tuple[str, BinaryIO]] | None = None,
    ) -> TurnSummary:
        with self._save_locks[save_id]:
            return self._create_turn_unlocked(save_id, player_input, context_turns, images or [])

    def _create_turn_unlocked(
        self,
        save_id: str,
        player_input: str,
        context_turns: int,
        images: list[tuple[str, BinaryIO]],
    ) -> TurnSummary:
        turn_id = str(uuid.uuid4())
        root_session_id = str(uuid.uuid4())
        now = utc_now()
        save = self.get_save(save_id)
        with self.connect(save_id) as db:
            latest = db.execute("SELECT id, status FROM turns ORDER BY rowid DESC LIMIT 1").fetchone()
        if latest and latest["status"] == "completed":
            self._checkpoint_turn_unlocked(save_id, str(latest["id"]))
        save_dir = self._state_dir(save_id)
        turn_dir = save_dir / "turns" / turn_id
        artifacts_dir = save_dir / "artifacts"
        if artifacts_dir.is_symlink() or not artifacts_dir.is_dir():
            raise ValueError("存档 artifacts 根路径必须是非符号链接目录")
        prepared_images: list[dict[str, Any]] = []
        staged: list[tuple[Path, Path]] = []
        finalized: list[Path] = []
        try:
            for position, (original_name, source) in enumerate(images):
                safe_name = Path(original_name.replace("\\", "/")).name[:255] or "image"
                image_id = str(uuid.uuid4())
                temporary = artifacts_dir / f".uploading-{image_id}"
                staged.append((temporary, artifacts_dir / image_id))
                size = 0
                digest = hashlib.sha256()
                source.seek(0)
                with temporary.open("xb") as handle:
                    while chunk := source.read(1024 * 1024):
                        size += len(chunk)
                        if size > IMAGE_ARTIFACT_MAX_BYTES:
                            raise ValueError(
                                f"图片超过 {IMAGE_ARTIFACT_MAX_BYTES // (1024 * 1024)} MiB 上限：{safe_name}"
                            )
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                mime_type, width, height, extension = self._inspect_image(temporary, safe_name)
                target = artifacts_dir / f"{image_id}{extension}"
                staged[-1] = (temporary, target)
                prepared_images.append({
                    "id": image_id,
                    "position": position,
                    "stored_name": target.name,
                    "original_name": safe_name,
                    "mime_type": mime_type,
                    "byte_size": size,
                    "width": width,
                    "height": height,
                    "sha256": digest.hexdigest(),
                })
            self._touch_unlocked(save_id)
            turn_dir.mkdir()
            self._sync_directory(turn_dir.parent)
            (turn_dir / "drafts").mkdir()
            self._sync_directory(turn_dir)
            self._sync_directory(turn_dir / "drafts")
            with self.connect(save_id) as db:
                db.execute(
                    "INSERT INTO turns (id, player_input, status, root_session_id, created_at) VALUES (?, ?, 'running', ?, ?)",
                    (turn_id, player_input, root_session_id, now),
                )
                db.execute(
                    "INSERT INTO sessions (id, turn_id, agent, parent_session_id, created_at) VALUES (?, ?, 'coordinator', NULL, ?)",
                    (root_session_id, turn_id, now),
                )
                db.executemany(
                    """INSERT INTO turn_images
                       (id, turn_id, position, stored_name, original_name, mime_type, byte_size, width, height, sha256, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            image["id"], turn_id, image["position"], image["stored_name"],
                            image["original_name"], image["mime_type"], image["byte_size"],
                            image["width"], image["height"], image["sha256"], now,
                        )
                        for image in prepared_images
                    ],
                )
                initial = self._build_story(db, turn_id, context_turns, save.opening)
                self._insert_text_message(db, root_session_id, "user", json.dumps(initial, ensure_ascii=False), now)
                for temporary, target in staged:
                    os.replace(temporary, target)
                    finalized.append(target)
                self._sync_directory(artifacts_dir)
        except sqlite3.IntegrityError as error:
            shutil.rmtree(turn_dir, ignore_errors=True)
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
            for target in finalized:
                target.unlink(missing_ok=True)
            if "turns.status" in str(error):
                raise TurnAlreadyRunningError("当前存档已有正在执行的回合") from error
            raise
        except Exception:
            shutil.rmtree(turn_dir, ignore_errors=True)
            for temporary, _ in staged:
                temporary.unlink(missing_ok=True)
            for target in finalized:
                target.unlink(missing_ok=True)
            raise
        return TurnSummary(
            id=turn_id,
            player_input=player_input,
            narrative=None,
            status="running",
            created_at=now,
            completed_at=None,
            images=[self._turn_image_summary(save_id, turn_id, image) for image in prepared_images],
        )

    def fail_turn(
        self,
        save_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> RuntimeEvent | None:
        now = utc_now()
        with self.connect(save_id) as db:
            status = "interrupted" if event_type == "turn.interrupted" else "failed"
            cursor = db.execute(
                """UPDATE turns SET status = ?, completed_at = ?
                   WHERE id = ? AND status = 'running'""",
                (status, now, turn_id),
            )
            if cursor.rowcount != 1:
                return None
            event = self._insert_event(db, turn_id, event_type, payload, now)
        return event

    def recover_interrupted_turns(self) -> None:
        for save in self._all_saves():
            with self._save_locks[save.id]:
                # A crash after the publication transaction but before the directory
                # rename leaves the completed turn in current without a checkpoint.
                with self.connect(save.id) as db:
                    latest = db.execute("SELECT id, status FROM turns ORDER BY rowid DESC LIMIT 1").fetchone()
                if latest and latest["status"] == "completed":
                    self._checkpoint_turn_unlocked(save.id, str(latest["id"]))
            with self.connect(save.id) as db:
                rows = db.execute("SELECT id FROM turns WHERE status = 'running'").fetchall()
            for row in rows:
                self.interrupt_tools(save.id, str(row["id"]))
                self.fail_turn(
                    save.id,
                    str(row["id"]),
                    "turn.interrupted",
                    {"message": "服务重启，中断未完成回合"},
                )

    def _all_saves(self) -> list[SaveSummary]:
        saves: list[SaveSummary] = []
        for path in self.saves_dir.iterdir():
            save_path = path / "save.json"
            if path.is_dir() and save_path.is_file():
                saves.append(SaveSummary.model_validate_json(save_path.read_text(encoding="utf-8")))
        return saves

    def list_turns(self, save_id: str) -> list[TurnSummary]:
        with self.connect(save_id) as db:
            rows = db.execute("SELECT * FROM turns ORDER BY rowid").fetchall()
            return [
                TurnSummary.model_validate({
                    **dict(row),
                    "images": self._turn_images(db, save_id, str(row["id"])),
                    "generated_images": self._generated_images(db, save_id, str(row["id"])),
                })
                for row in rows
            ]

    def _turn_images(self, db: sqlite3.Connection, save_id: str, turn_id: str) -> list[TurnImageSummary]:
        rows = db.execute(
            "SELECT * FROM turn_images WHERE turn_id = ? ORDER BY position",
            (turn_id,),
        ).fetchall()
        return [self._turn_image_summary(save_id, turn_id, dict(row)) for row in rows]

    @staticmethod
    def _turn_image_summary(save_id: str, turn_id: str, image: dict[str, Any]) -> TurnImageSummary:
        return TurnImageSummary(
            id=str(image["id"]),
            original_name=str(image["original_name"]),
            mime_type=str(image["mime_type"]),
            byte_size=int(image["byte_size"]),
            width=int(image["width"]),
            height=int(image["height"]),
            position=int(image["position"]),
            content_url=f"/api/saves/{save_id}/turns/{turn_id}/images/{image['id']}/content",
        )

    def get_turn_image_content(self, save_id: str, turn_id: str, image_id: str) -> tuple[Path, str, str]:
        with self.connect(save_id) as db:
            row = db.execute(
                "SELECT * FROM turn_images WHERE id = ? AND turn_id = ?",
                (image_id, turn_id),
            ).fetchone()
        if row is None:
            raise KeyError(f"未知回合图片：{image_id}")
        path = self._turn_image_path(save_id, row)
        return path, str(row["mime_type"]), str(row["original_name"])

    def read_turn_image_data(self, save_id: str, image_id: str) -> dict[str, str]:
        with self.connect(save_id) as db:
            row = db.execute("SELECT * FROM turn_images WHERE id = ?", (image_id,)).fetchone()
        if row is None:
            raise KeyError(f"未知回合图片：{image_id}")
        path = self._turn_image_path(save_id, row)
        data = path.read_bytes()
        if len(data) != int(row["byte_size"]) or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"回合图片内容与上传记录不一致：{image_id}")
        return {"mime_type": str(row["mime_type"]), "data": base64.b64encode(data).decode("ascii")}

    def _turn_image_path(self, save_id: str, row: sqlite3.Row) -> Path:
        artifacts = self._state_dir(save_id) / "artifacts"
        if artifacts.is_symlink() or not artifacts.is_dir():
            raise ValueError("存档 artifacts 根路径必须是非符号链接目录")
        path = artifacts / str(row["stored_name"])
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"回合图片文件不存在：{row['id']}")
        return path

    def get_turn(self, save_id: str, turn_id: str) -> TurnSummary:
        with self.connect(save_id) as db:
            row = db.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
            if row is None:
                raise KeyError(f"未知回合：{turn_id}")
            return TurnSummary.model_validate({
                **dict(row),
                "images": self._turn_images(db, save_id, turn_id),
                "generated_images": self._generated_images(db, save_id, turn_id),
            })

    @staticmethod
    def _selected_generated_parts(db: sqlite3.Connection, turn_id: str) -> list[sqlite3.Row]:
        return db.execute(
            """WITH generated AS (
                   SELECT parts.id, parts.output, sessions.agent,
                          ROW_NUMBER() OVER (PARTITION BY sessions.id ORDER BY parts.rowid DESC) AS image_order,
                          sessions.rowid AS session_order
                   FROM parts JOIN sessions ON sessions.id = parts.session_id
                   WHERE sessions.turn_id = ? AND sessions.parent_session_id IS NOT NULL
                     AND parts.type = 'tool' AND parts.tool_name = 'character_portrait_generate'
                     AND parts.state = 'completed'
               )
               SELECT id, output, agent FROM generated WHERE image_order = 1 ORDER BY session_order""",
            (turn_id,),
        ).fetchall()

    def _generated_images(self, db: sqlite3.Connection, save_id: str, turn_id: str) -> list[GeneratedImageSummary]:
        rows = self._selected_generated_parts(db, turn_id)
        result = []
        for row in rows:
            metadata = json.loads(row["output"])
            result.append(GeneratedImageSummary(
                id=str(row["id"]), agent=str(row["agent"]),
                mime_type=metadata["mime_type"], width=metadata["width"], height=metadata["height"],
                content_url=f"/api/saves/{save_id}/turns/{turn_id}/generated-images/{row['id']}/content",
            ))
        return result

    def get_generated_image_content(self, save_id: str, turn_id: str, part_id: str) -> tuple[Path, str]:
        path, artifact = self._generated_image_artifact(save_id, part_id, turn_id)
        return path, str(artifact["mime_type"])

    def read_generated_image_data(self, save_id: str, part_id: str) -> dict[str, str]:
        _, artifact = self._generated_image_artifact(save_id, part_id)
        return {"mime_type": str(artifact["mime_type"]), "data": str(artifact["data"])}

    def _generated_image_artifact(
        self, save_id: str, part_id: str, turn_id: str | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        with self.connect(save_id) as db:
            row = db.execute(
                """SELECT parts.output, sessions.turn_id FROM parts JOIN sessions ON sessions.id = parts.session_id
                   WHERE parts.id = ? AND sessions.parent_session_id IS NOT NULL
                     AND parts.type = 'tool' AND parts.tool_name = 'character_portrait_generate'
                     AND parts.state = 'completed'""",
                (part_id,),
            ).fetchone()
        if row is None or (turn_id is not None and row["turn_id"] != turn_id):
            raise KeyError(f"未知生成图片：{part_id}")
        metadata = json.loads(row["output"])
        artifact = self.read_image_artifact(save_id, metadata["path"])
        if any(artifact[field] != metadata[field] for field in ("mime_type", "size", "sha256")):
            raise ValueError(f"生成图片内容与 Tool Result 不一致：{part_id}")
        return self._state_dir(save_id) / metadata["path"], artifact

    def retry_turn(self, save_id: str, turn_id: str, context_turns: int) -> TurnSummary:
        with self._save_locks[save_id]:
            save = self.get_save(save_id)
            try:
                with self.connect(save_id) as db:
                    latest = db.execute("SELECT id, status FROM turns ORDER BY rowid DESC LIMIT 1").fetchone()
                    if latest is None or latest["id"] != turn_id:
                        exists = db.execute("SELECT 1 FROM turns WHERE id = ?", (turn_id,)).fetchone()
                        if exists is None:
                            raise KeyError(f"未知回合：{turn_id}")
                        raise TurnRetryError("只能继续存档中最后一个失败或中断回合")
                    if latest["status"] not in {"failed", "interrupted"}:
                        raise TurnRetryError(f"只有失败或中断回合可以继续，当前状态：{latest['status']}")
                    story = self._build_story(db, turn_id, context_turns, save.opening)
                    root = db.execute(
                        "SELECT root_session_id FROM turns WHERE id = ?", (turn_id,)
                    ).fetchone()
                    initial_part = db.execute(
                        """SELECT parts.id
                           FROM messages
                           JOIN parts ON parts.message_id = messages.id
                           WHERE messages.session_id = ? AND messages.sequence = 0
                             AND messages.role = 'user' AND messages.kind = 'model'
                             AND parts.sequence = 0 AND parts.type = 'text'""",
                        (root["root_session_id"],),
                    ).fetchone()
                    if initial_part is None:
                        raise RuntimeError("主代理 session 缺少初始完整正文记录")
                    self._touch_unlocked(save_id)
                    now = utc_now()
                    db.execute(
                        "UPDATE parts SET content = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(story, ensure_ascii=False), now, initial_part["id"]),
                    )
                    db.execute(
                        """UPDATE turns
                           SET status = 'running', narrative = NULL, completed_at = NULL
                           WHERE id = ?""",
                        (turn_id,),
                    )
                    db.execute(
                        """DELETE FROM messages
                           WHERE id = (
                               SELECT messages.id
                               FROM messages
                               JOIN turns ON turns.root_session_id = messages.session_id
                               WHERE turns.id = ?
                               ORDER BY messages.sequence DESC
                               LIMIT 1
                           )
                             AND role = 'assistant'
                             AND kind = 'model'
                             AND NOT EXISTS (
                                 SELECT 1 FROM parts
                                 WHERE parts.message_id = messages.id AND parts.type = 'tool'
                             )""",
                        (turn_id,),
                    )
            except sqlite3.IntegrityError as error:
                if "turns.status" in str(error):
                    raise TurnAlreadyRunningError("当前存档已有正在执行的回合") from error
                raise
        return self.get_turn(save_id, turn_id)

    def load_story(self, save_id: str, turn_id: str) -> dict[str, Any]:
        with self.connect(save_id) as db:
            row = db.execute(
                """SELECT parts.content
                   FROM turns
                   JOIN messages ON messages.session_id = turns.root_session_id
                   JOIN parts ON parts.message_id = messages.id
                   WHERE turns.id = ? AND messages.sequence = 0
                     AND messages.role = 'user' AND messages.kind = 'model'
                     AND parts.sequence = 0 AND parts.type = 'text'""",
                (turn_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"回合缺少完整正文记录：{turn_id}")
        try:
            value = json.loads(row["content"])
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError("完整正文记录不是合法 JSON") from error
        return self._validate_story(value)

    def ensure_story_summary(self, save_id: str) -> None:
        StorySummaryStore(self._state_dir(save_id)).ensure()

    def read_story_summary(self, save_id: str) -> dict[str, Any]:
        return StorySummaryStore(self._state_dir(save_id)).read()

    def edit_story_summary(self, save_id: str, turn_id: str, old_text: str, new_text: str) -> dict[str, Any]:
        current_number = self.load_story(save_id, turn_id)["story"]["turns"][-1]["turn_number"]
        return StorySummaryStore(self._state_dir(save_id)).edit(old_text, new_text, current_number)

    def read_story_history(self, save_id: str, turn_id: str, after_turn_number: int, limit: int = 20) -> dict[str, Any]:
        with self.connect(save_id) as db:
            rows = db.execute(
                """SELECT turns.*,
                          (SELECT COUNT(*) FROM turns AS numbered WHERE numbered.rowid <= turns.rowid) AS turn_number
                   FROM turns
                   WHERE turns.rowid < (SELECT rowid FROM turns WHERE id = ?)
                   ORDER BY turns.rowid LIMIT ? OFFSET ?""",
                (turn_id, limit, after_turn_number),
            ).fetchall()
            turns = [self._story_turn(db, row, include_ai_output=True) for row in rows]
        return {"turns": turns, "next_after_turn_number": turns[-1]["turn_number"] if turns else None}

    def _build_story(
        self,
        db: sqlite3.Connection,
        current_turn_id: str,
        context_turns: int,
        opening: str,
    ) -> dict[str, Any]:
        current = db.execute(
            """SELECT turns.*,
                      (SELECT COUNT(*) FROM turns AS numbered WHERE numbered.rowid <= turns.rowid) AS turn_number
               FROM turns WHERE id = ?""",
            (current_turn_id,),
        ).fetchone()
        if current is None:
            raise KeyError(f"未知回合：{current_turn_id}")
        rows = db.execute(
            """SELECT turns.*,
                      (SELECT COUNT(*) FROM turns AS numbered WHERE numbered.rowid <= turns.rowid) AS turn_number
               FROM turns
               WHERE rowid < (SELECT rowid FROM turns WHERE id = ?)
               ORDER BY rowid DESC LIMIT ?""",
            (current_turn_id, min(context_turns, 2**63 - 1)),
        ).fetchall()
        turns = [self._story_turn(db, row, include_ai_output=True) for row in reversed(rows)]
        turns.append(self._story_turn(db, current, include_ai_output=False))
        return self._validate_story({"story": {"opening": opening, "turns": turns}})

    def _story_turn(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        include_ai_output: bool,
    ) -> dict[str, Any]:
        images = [
            {"id": str(image["id"])}
            for image in db.execute(
                "SELECT id FROM turn_images WHERE turn_id = ? ORDER BY position",
                (row["id"],),
            ).fetchall()
        ]
        has_ai_output = include_ai_output and row["status"] == "completed"
        turn: dict[str, Any] = {
            "turn_number": int(row["turn_number"]),
            "player": {"content": str(row["player_input"]), "images": images},
            "has_ai_output": has_ai_output,
        }
        generated = self._selected_generated_parts(db, str(row["id"]))
        if generated:
            turn["generated_images"] = [{"id": str(part["id"])} for part in generated]
        if has_ai_output:
            narrative = row["narrative"]
            if not isinstance(narrative, str) or not narrative.strip():
                raise ValueError(f"已完成回合缺少正式正文：{row['id']}")
            turn["narrative"] = narrative
        return turn

    @staticmethod
    def _validate_story(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or set(value) != {"story"}:
            raise ValueError("完整正文记录顶层必须且只能包含 story")
        story = value["story"]
        if not isinstance(story, dict) or set(story) != {"opening", "turns"}:
            raise ValueError("完整正文记录 story 结构无效")
        if not isinstance(story["opening"], str) or not isinstance(story["turns"], list):
            raise ValueError("完整正文记录 opening 或 turns 类型无效")
        previous_number = 0
        for index, turn in enumerate(story["turns"]):
            if not isinstance(turn, dict):
                raise ValueError(f"完整正文记录 turns[{index}] 不是对象")
            has_ai_output = turn.get("has_ai_output")
            expected = {"turn_number", "player", "has_ai_output"}
            if has_ai_output is True:
                expected.add("narrative")
            if "generated_images" in turn:
                expected.add("generated_images")
            if set(turn) != expected or type(has_ai_output) is not bool:
                raise ValueError(f"完整正文记录 turns[{index}] 字段无效")
            number = turn["turn_number"]
            if type(number) is not int or number <= previous_number:
                raise ValueError(f"完整正文记录 turns[{index}] 编号无效")
            previous_number = number
            player = turn["player"]
            if not isinstance(player, dict) or set(player) != {"content", "images"}:
                raise ValueError(f"完整正文记录 turns[{index}].player 结构无效")
            if not isinstance(player["content"], str) or not isinstance(player["images"], list):
                raise ValueError(f"完整正文记录 turns[{index}].player 类型无效")
            for image in player["images"]:
                if not isinstance(image, dict) or set(image) != {"id"} or not isinstance(image["id"], str):
                    raise ValueError(f"完整正文记录 turns[{index}] 图片引用无效")
            if "generated_images" in turn:
                if not isinstance(turn["generated_images"], list):
                    raise ValueError(f"完整正文记录 turns[{index}] 生成图片引用无效")
                for image in turn["generated_images"]:
                    if not isinstance(image, dict) or set(image) != {"id"} or not isinstance(image["id"], str):
                        raise ValueError(f"完整正文记录 turns[{index}] 生成图片引用无效")
            if not player["content"] and not player["images"]:
                raise ValueError(f"完整正文记录 turns[{index}] 玩家输入为空")
            if has_ai_output and (not isinstance(turn["narrative"], str) or not turn["narrative"].strip()):
                raise ValueError(f"完整正文记录 turns[{index}] 正文无效")
        return value

    def root_session_id(self, save_id: str, turn_id: str) -> str:
        with self.connect(save_id) as db:
            row = db.execute("SELECT root_session_id FROM turns WHERE id = ?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError(f"未知回合：{turn_id}")
        return str(row["root_session_id"])

    def task_count(self, save_id: str, turn_id: str) -> int:
        with self.connect(save_id) as db:
            return int(
                db.execute(
                    """SELECT COUNT(*) FROM parts
                       WHERE type = 'tool' AND tool_name = 'task'
                         AND child_session_id IS NOT NULL
                         AND session_id = (SELECT root_session_id FROM turns WHERE id = ?)""",
                    (turn_id,),
                ).fetchone()[0]
            )

    def prepare_child_task(
        self,
        save_id: str,
        turn_id: str,
        parent_session_id: str,
        task_part_id: str,
        agent: str,
        task: dict[str, Any],
        task_id: str | None,
        related_entity_documents: list[dict[str, Any]] | None = None,
        available_styles: list[dict[str, Any]] | None = None,
        style_documents: list[dict[str, Any]] | None = None,
        report_documents: list[dict[str, Any]] | None = None,
    ) -> str:
        session_id = task_id or str(uuid.uuid4())
        now = utc_now()
        initial: dict[str, Any] = {"task": task}
        if task_id is None and agent == "world_researcher":
            initial["required_world_entities_instruction"] = "这些是对故事长期必要的信息，因此直接提供给你。"
            initial["required_world_entities"] = self.required_entity_documents(save_id)
        if related_entity_documents:
            if agent not in {"character_designer", "location_designer", "EroticOrNot", "role_player", "style_planner", "narrator"} or task_id is not None:
                raise ValueError("相关实体正文只能注入新建 character_designer、location_designer、EroticOrNot、role_player、style_planner 或 narrator 会话")
            initial["related_entity_documents"] = related_entity_documents
        if available_styles is not None:
            if agent != "style_planner" or task_id is not None:
                raise ValueError("文风清单只能注入新建 style_planner 会话")
            initial["available_styles"] = available_styles
        if style_documents:
            if agent != "narrator" or task_id is not None:
                raise ValueError("文风正文只能注入新建 narrator 会话")
            initial["style_documents"] = style_documents
        if report_documents:
            initial["report_documents_instruction"] = "以下 report_documents 是 Runtime 验真的完整报告原文，请结合当前任务直接使用，不要要求主代理重新转述。"
            initial["report_documents"] = report_documents
        with self.connect(save_id) as db:
            parent = db.execute(
                "SELECT turn_id FROM sessions WHERE id = ?", (parent_session_id,)
            ).fetchone()
            if parent is None or parent["turn_id"] != turn_id:
                raise ValueError("父 session 不属于当前回合")
            part = db.execute(
                "SELECT session_id, tool_name, state, child_session_id FROM parts WHERE id = ?",
                (task_part_id,),
            ).fetchone()
            if part is None or part["session_id"] != parent_session_id or part["tool_name"] != "task" or part["state"] != "running" or part["child_session_id"] is not None:
                raise ValueError("父 task Tool part 无法关联 child session")
            if task_id is None:
                db.execute(
                    "INSERT INTO sessions (id, turn_id, agent, parent_session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                    (session_id, turn_id, agent, parent_session_id, now),
                )
                self._insert_text_message(db, session_id, "user", json.dumps(initial, ensure_ascii=False), now)
            else:
                child = db.execute(
                    "SELECT turn_id, agent, parent_session_id FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if child is None:
                    raise ValueError("task_id 不存在")
                if child["turn_id"] != turn_id or child["parent_session_id"] != parent_session_id or child["agent"] != agent:
                    raise ValueError("task_id 不属于当前回合、父 Agent 或目标 Agent")
                continuation: dict[str, Any] = {"task": task, "continuation": True}
                if report_documents:
                    continuation["report_documents_instruction"] = initial["report_documents_instruction"]
                    continuation["report_documents"] = report_documents
                self._insert_text_message(db, session_id, "user", json.dumps(continuation, ensure_ascii=False), now)
            db.execute(
                "UPDATE parts SET child_session_id = ?, updated_at = ? WHERE id = ?",
                (session_id, now, task_part_id),
            )
        return session_id

    def session_agent(self, save_id: str, session_id: str) -> str:
        with self.connect(save_id) as db:
            row = db.execute("SELECT agent FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"未知 session：{session_id}")
        return str(row["agent"])

    def create_assistant_message(self, save_id: str, session_id: str, model: dict[str, Any]) -> str:
        message_id = str(uuid.uuid4())
        now = utc_now()
        with self.connect(save_id) as db:
            sequence = int(db.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0])
            db.execute(
                "INSERT INTO messages (id, session_id, sequence, role, kind, model, created_at) VALUES (?, ?, ?, 'assistant', 'model', ?, ?)",
                (message_id, session_id, sequence, json.dumps(model, ensure_ascii=False), now),
            )
        return message_id

    def record_model_result(
        self,
        save_id: str,
        message_id: str,
        content: str,
        reasoning: str | None,
        tool_calls: list[dict[str, Any]],
        rejection: dict[str, Any] | None = None,
        provider_content: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        now = utc_now()
        created: list[dict[str, Any]] = []
        with self.connect(save_id) as db:
            message = db.execute("SELECT session_id, model FROM messages WHERE id = ?", (message_id,)).fetchone()
            if message is None:
                raise KeyError(f"未知 message：{message_id}")
            session_id = str(message["session_id"])
            if provider_content is not None:
                try:
                    model = json.loads(message["model"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    model = {}
                if not isinstance(model, dict):
                    model = {}
                model["provider_content"] = provider_content
                db.execute(
                    "UPDATE messages SET model = ? WHERE id = ?",
                    (json.dumps(model, ensure_ascii=False), message_id),
                )
            sequence = 0
            if content:
                part_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO parts (id, message_id, session_id, sequence, type, content, created_at, updated_at) VALUES (?, ?, ?, ?, 'text', ?, ?, ?)",
                    (part_id, message_id, session_id, sequence, content, now, now),
                )
                sequence += 1
            if reasoning:
                part_id = str(uuid.uuid4())
                db.execute(
                    "INSERT INTO parts (id, message_id, session_id, sequence, type, content, created_at, updated_at) VALUES (?, ?, ?, ?, 'reasoning', ?, ?, ?)",
                    (part_id, message_id, session_id, sequence, reasoning, now, now),
                )
                sequence += 1
            for call in tool_calls:
                part_id = str(uuid.uuid4())
                state = "error" if rejection is not None else "pending"
                output = json.dumps(rejection, ensure_ascii=False) if rejection is not None else None
                serialized_input = call["arguments"] if isinstance(call["arguments"], str) else json.dumps(call["arguments"], ensure_ascii=False)
                db.execute(
                    """INSERT INTO parts
                       (id, message_id, session_id, sequence, type, provider_call_id, tool_name, input, state, output, created_at, updated_at)
                       VALUES (?, ?, ?, ?, 'tool', ?, ?, ?, ?, ?, ?, ?)""",
                    (part_id, message_id, session_id, sequence, call["id"], call["name"], serialized_input, state, output, now, now),
                )
                created.append({"id": part_id, "provider_call_id": call["id"], "tool_name": call["name"], "input": serialized_input, "state": state, "output": rejection})
                sequence += 1
        return created

    def fail_assistant_message(self, save_id: str, message_id: str, error: str) -> None:
        with self.connect(save_id) as db:
            db.execute("UPDATE messages SET error = ? WHERE id = ?", (error, message_id))

    def set_tool_running(self, save_id: str, part_id: str) -> None:
        with self.connect(save_id) as db:
            cursor = db.execute(
                "UPDATE parts SET state = 'running', updated_at = ? WHERE id = ? AND type = 'tool' AND state = 'pending'",
                (utc_now(), part_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Tool part 不处于 pending 状态")

    def create_automatic_publish_part(self, save_id: str, message_id: str) -> dict[str, str]:
        now = utc_now()
        part_id = str(uuid.uuid4())
        provider_call_id = f"automatic-publish-{part_id}"
        raw_input = json.dumps({"path": "narrative.md"}, ensure_ascii=False)
        with self.connect(save_id) as db:
            message = db.execute("SELECT session_id FROM messages WHERE id = ?", (message_id,)).fetchone()
            if message is None:
                raise KeyError(f"未知 message：{message_id}")
            sequence = int(db.execute(
                "SELECT COALESCE(MAX(sequence), -1) + 1 FROM parts WHERE message_id = ?",
                (message_id,),
            ).fetchone()[0])
            db.execute(
                """INSERT INTO parts
                   (id, message_id, session_id, sequence, type, provider_call_id, tool_name, input, state, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'tool', ?, 'narrative_publish', ?, 'running', ?, ?)""",
                (part_id, message_id, message["session_id"], sequence, provider_call_id, raw_input, now, now),
            )
        return {"id": part_id, "provider_call_id": provider_call_id, "input": raw_input}

    def complete_tool(self, save_id: str, part_id: str, output: dict[str, Any]) -> None:
        self._finish_tool(save_id, part_id, "completed", output)

    def error_tool(self, save_id: str, part_id: str, output: dict[str, Any]) -> None:
        self._finish_tool(save_id, part_id, "error", output)

    def _finish_tool(self, save_id: str, part_id: str, state: str, output: dict[str, Any]) -> None:
        with self.connect(save_id) as db:
            cursor = db.execute(
                "UPDATE parts SET state = ?, output = ?, updated_at = ? WHERE id = ? AND state = 'running'",
                (state, json.dumps(output, ensure_ascii=False), utc_now(), part_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("Tool part 不处于 running 状态")

    def interrupt_tools(self, save_id: str, turn_id: str) -> None:
        now = utc_now()
        with self.connect(save_id) as db:
            rows = db.execute(
                """SELECT id, tool_name, child_session_id FROM parts
                   WHERE session_id IN (SELECT id FROM sessions WHERE turn_id = ?)
                     AND type = 'tool' AND state IN ('pending', 'running')""",
                (turn_id,),
            ).fetchall()
            for row in rows:
                result: dict[str, Any] = {"ok": False, "error": {"code": "execution_interrupted", "message": "Tool execution was interrupted"}}
                if row["tool_name"] == "task" and row["child_session_id"]:
                    result["task_id"] = row["child_session_id"]
                db.execute(
                    "UPDATE parts SET state = 'error', output = ?, updated_at = ? WHERE id = ?",
                    (json.dumps(result, ensure_ascii=False), now, row["id"]),
                )

    def load_session(self, save_id: str, session_id: str) -> list[dict[str, Any]]:
        with self.connect(save_id) as db:
            messages = db.execute(
                "SELECT * FROM messages WHERE session_id = ? AND kind = 'model' ORDER BY sequence", (session_id,)
            ).fetchall()
            loaded: list[dict[str, Any]] = []
            for message in messages:
                parts = db.execute("SELECT * FROM parts WHERE message_id = ? ORDER BY sequence", (message["id"],)).fetchall()
                loaded.append({**dict(message), "parts": [dict(part) for part in parts]})
        return loaded

    def publish_narrative(
        self,
        save_id: str,
        turn_id: str,
        session_id: str,
        part_id: str,
        duration_ms: int,
        replacement_message_id: str | None = None,
        replacement_text: str | None = None,
    ) -> tuple[dict[str, Any], list[RuntimeEvent]]:
        with self._save_locks[save_id]:
            result = self._publish_narrative_unlocked(
                save_id, turn_id, session_id, part_id, duration_ms, replacement_message_id, replacement_text,
            )
            try:
                self._checkpoint_turn_unlocked(save_id, turn_id)
            except Exception:
                # Publication has already committed. Keep its success visible; block
                # manual edits until the checkpoint can be rebuilt from current.
                logging.getLogger(__name__).exception("无法为已完成回合保存检查点：%s", turn_id)
            return result

    def _publish_narrative_unlocked(
        self,
        save_id: str,
        turn_id: str,
        session_id: str,
        part_id: str,
        duration_ms: int,
        replacement_message_id: str | None = None,
        replacement_text: str | None = None,
    ) -> tuple[dict[str, Any], list[RuntimeEvent]]:
        from .draft_files import DraftFileStore

        with self.connect(save_id) as db:
            turn = db.execute("SELECT root_session_id FROM turns WHERE id = ?", (turn_id,)).fetchone()
        if turn is None:
            raise KeyError(f"未知回合：{turn_id}")
        if turn["root_session_id"] != session_id:
            raise ValueError("只有当前 root session 可以发布故事")
        narrative = DraftFileStore(self.draft_dir(save_id, turn_id)).read("narrative.md")
        if not narrative.strip():
            raise ValueError("narrative.md 不能为空")
        now = utc_now()
        output = {"status": "published", "path": "narrative.md", "content": narrative}
        events: list[RuntimeEvent] = []
        with self.connect(save_id) as db:
            cursor = db.execute(
                "UPDATE turns SET narrative = ?, status = 'completed', completed_at = ? WHERE id = ? AND status = 'running'",
                (narrative, now, turn_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("故事发布时回合已不再运行")
            cursor = db.execute(
                "UPDATE parts SET state = 'completed', output = ?, updated_at = ? WHERE id = ? AND state = 'running' AND tool_name = 'narrative_publish'",
                (json.dumps(output, ensure_ascii=False), now, part_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("发布 Tool part 不处于 running 状态")
            if replacement_message_id is not None:
                cursor = db.execute(
                    "UPDATE parts SET content = ?, updated_at = ? WHERE message_id = ? AND session_id = ? AND type = 'text'",
                    (replacement_text, now, replacement_message_id, session_id),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("自动发布只能替换一条主代理纯文本")
            self._insert_text_message(db, session_id, "assistant", narrative, now, kind="story")
            events.append(self._insert_event(db, turn_id, "tool.completed", {"session_id": session_id, "part_id": part_id, "agent": "coordinator", "tool": "narrative_publish", "input": json.dumps({"path": "narrative.md"}, ensure_ascii=False), "result": output}, now))
            events.append(self._insert_event(db, turn_id, "narrative.delta", {"content": narrative}, now))
            events.append(self._insert_event(db, turn_id, "turn.completed", {"duration_ms": duration_ms}, now))
        return output, events

    def draft_dir(self, save_id: str, turn_id: str) -> Path:
        path = self._state_dir(save_id) / "turns" / turn_id / "drafts"
        if not path.is_dir():
            raise KeyError(f"回合草稿目录不存在：{turn_id}")
        return path

    def add_event(
        self,
        save_id: str,
        turn_id: str | None,
        event_type: str,
        payload: dict[str, Any],
    ) -> RuntimeEvent:
        created_at = utc_now()
        serialized = json.dumps(payload, ensure_ascii=False)
        with self.connect(save_id) as db:
            cursor = db.execute(
                "INSERT INTO events (turn_id, type, payload, created_at) VALUES (?, ?, ?, ?)",
                (turn_id, event_type, serialized, created_at),
            )
            if cursor.lastrowid is None:
                raise RuntimeError("无法取得事件 ID")
            event_id = cursor.lastrowid
        return RuntimeEvent(
            id=event_id,
            turn_id=turn_id,
            type=event_type,
            payload=payload,
            created_at=created_at,
        )

    def list_events(self, save_id: str, after: int = 0, limit: int = 500) -> list[RuntimeEvent]:
        with self.connect(save_id) as db:
            rows = db.execute(
                "SELECT * FROM events WHERE id > ? ORDER BY id LIMIT ?",
                (after, limit),
            ).fetchall()
        return [
            RuntimeEvent(
                id=row["id"],
                turn_id=row["turn_id"],
                type=row["type"],
                payload=json.loads(row["payload"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def list_event_page(self, save_id: str, page: int | None = None, page_size: int = 50) -> EventPage:
        with self.connect(save_id) as db:
            total = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            total_pages = math.ceil(total / page_size)
            current_page = min(page if page is not None else max(total_pages, 1), max(total_pages, 1))
            rows = db.execute(
                "SELECT * FROM events ORDER BY id LIMIT ? OFFSET ?",
                (page_size, (current_page - 1) * page_size),
            ).fetchall()
        return EventPage(
            events=[
                RuntimeEvent(
                    id=row["id"],
                    turn_id=row["turn_id"],
                    type=row["type"],
                    payload=json.loads(row["payload"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ],
            page=current_page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
        )

    def list_trace_page(self, save_id: str, page: int | None = None, page_size: int = 50) -> TracePage:
        placeholders = ", ".join("?" for _ in TRACE_EVENT_TYPES)
        with self.connect(save_id) as db:
            db.execute("BEGIN")
            cursor = int(db.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()[0])
            rows = db.execute(
                f"SELECT * FROM events WHERE type IN ({placeholders}) ORDER BY id",
                TRACE_EVENT_TYPES,
            ).fetchall()
            part_inputs = {
                str(row["id"]): str(row["input"])
                for row in db.execute("SELECT id, input FROM parts WHERE type = 'tool'").fetchall()
            }
        source_events: list[RuntimeEvent] = []
        for row in rows:
            payload = json.loads(row["payload"])
            part_input = part_inputs.get(str(payload.get("part_id")))
            if part_input is not None:
                payload.setdefault("input", part_input)
            if row["type"] == "task.created" and part_input is not None:
                try:
                    task_input = json.loads(part_input)
                except json.JSONDecodeError:
                    task_input = {}
                if isinstance(task_input, dict):
                    payload.setdefault("initiator_agent", "coordinator")
                    payload.setdefault("target_agent", payload.get("agent"))
                    payload.setdefault("reason", task_input.get("reason", ""))
                    payload.setdefault("related_entities", [
                        {"path": path, "name": ""}
                        for path in task_input.get("related_entities", [])
                        if isinstance(path, str)
                    ])
                    payload.setdefault("reports", [
                        {"name": report.get("name", "")}
                        for report in task_input.get("report_refs", [])
                        if isinstance(report, dict)
                    ])
                    payload.setdefault("styles", [
                        {"path": path, "name": ""}
                        for path in task_input.get("style_paths", [])
                        if isinstance(path, str)
                    ])
            source_events.append(RuntimeEvent(
                id=row["id"],
                turn_id=row["turn_id"],
                type=row["type"],
                payload=payload,
                created_at=row["created_at"],
            ))
        events = [
            projected
            for event in source_events
            if (projected := project_trace_event(event)) is not None
        ]
        total = len(events)
        total_pages = math.ceil(total / page_size)
        current_page = min(page if page is not None else max(total_pages, 1), max(total_pages, 1))
        offset = (current_page - 1) * page_size
        return TracePage(
            events=events[offset:offset + page_size],
            page=current_page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            cursor=cursor,
        )

    def list_turn_events(self, save_id: str, turn_id: str) -> list[RuntimeEvent]:
        with self.connect(save_id) as db:
            rows = db.execute(
                "SELECT * FROM events WHERE turn_id = ? ORDER BY id",
                (turn_id,),
            ).fetchall()
        return [
            RuntimeEvent(
                id=row["id"],
                turn_id=row["turn_id"],
                type=row["type"],
                payload=json.loads(row["payload"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    @contextmanager
    def connect(self, save_id: str) -> Generator[sqlite3.Connection, None, None]:
        db = sqlite3.connect(self._state_dir(save_id) / "state.sqlite3")
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def _save_dir(self, save_id: str) -> Path:
        try:
            parsed = str(uuid.UUID(save_id))
        except ValueError as error:
            raise KeyError("无效存档 ID") from error
        path = self.saves_dir / parsed
        if path.is_symlink() or not path.is_dir():
            raise KeyError(f"未知存档：{save_id}")
        return path

    def _state_dir(self, save_id: str) -> Path:
        path = self._save_dir(save_id) / "current"
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"存档工作目录缺失：{save_id}")
        return path

    @staticmethod
    def _copy_state(source: Path, target: Path) -> None:
        """Copy a self-contained version; SQLite WAL files are never copied directly."""
        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("存档状态目录不能包含符号链接")

        def ignore(directory: str, names: list[str]) -> set[str]:
            database_files = {"state.sqlite3", "state.sqlite3-wal", "state.sqlite3-shm", "state.sqlite3-journal"}
            return database_files.intersection(names) if Path(directory) == source else set()

        shutil.copytree(source, target, ignore=ignore)
        with sqlite3.connect(source / "state.sqlite3") as original:
            with sqlite3.connect(target / "state.sqlite3") as copy:
                original.backup(copy)
        for path in target.rglob("*"):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        for path in sorted((item for item in target.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
            SaveStore._sync_directory(path)
        SaveStore._sync_directory(target)

    def _checkpoint_turn_unlocked(self, save_id: str, turn_id: str) -> None:
        save_dir = self._save_dir(save_id)
        revisions = save_dir / "revisions"
        target = revisions / turn_id
        if target.is_dir():
            return
        temporary = revisions / f".creating-{uuid.uuid4()}"
        published = False
        try:
            self._copy_state(self._state_dir(save_id), temporary)
            os.replace(temporary, target)
            published = True
            self._sync_directory(revisions)
        except Exception:
            if published:
                shutil.rmtree(target)
            raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def _initialize_database(path: Path) -> None:
        with sqlite3.connect(path) as db:
            db.executescript(SCHEMA)

    @staticmethod
    def _load_save(save_dir: Path) -> dict[str, Any]:
        return json.loads((save_dir / "save.json").read_text(encoding="utf-8"))

    @staticmethod
    def _write_save(save_dir: Path, save_data: dict[str, Any]) -> None:
        temporary = save_dir / "save.tmp"
        temporary.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(save_dir / "save.json")

    @staticmethod
    def _sync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _touch(self, save_id: str) -> None:
        with self._save_locks[save_id]:
            self._touch_unlocked(save_id)

    def _touch_unlocked(self, save_id: str) -> None:
        save_dir = self._save_dir(save_id)
        save_data = self._load_save(save_dir)
        save_data["last_played_at"] = utc_now()
        self._write_save(save_dir, save_data)

    @staticmethod
    def _insert_event(
        db: sqlite3.Connection,
        turn_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        created_at: str,
    ) -> RuntimeEvent:
        cursor = db.execute(
            "INSERT INTO events (turn_id, type, payload, created_at) VALUES (?, ?, ?, ?)",
            (turn_id, event_type, json.dumps(payload, ensure_ascii=False), created_at),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("无法取得事件 ID")
        return RuntimeEvent(
            id=cursor.lastrowid,
            turn_id=turn_id,
            type=event_type,
            payload=payload,
            created_at=created_at,
        )

    @staticmethod
    def _insert_text_message(
        db: sqlite3.Connection,
        session_id: str,
        role: str,
        content: str,
        created_at: str,
        *,
        kind: str = "model",
    ) -> str:
        message_id = str(uuid.uuid4())
        sequence = int(db.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)).fetchone()[0])
        db.execute(
            "INSERT INTO messages (id, session_id, sequence, role, kind, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (message_id, session_id, sequence, role, kind, created_at),
        )
        db.execute(
            "INSERT INTO parts (id, message_id, session_id, sequence, type, content, created_at, updated_at) VALUES (?, ?, ?, 0, 'text', ?, ?, ?)",
            (str(uuid.uuid4()), message_id, session_id, content, created_at, created_at),
        )
        return message_id

    @staticmethod
    def _validate_world_tree(root: Path) -> None:
        if root.is_symlink():
            raise ValueError("世界观根目录不允许使用符号链接")
        file_count = 0
        total_size = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"世界观不允许符号链接：{path.relative_to(root)}")
            if path.is_file():
                file_count += 1
                total_size += path.stat().st_size
            elif not path.is_dir():
                raise ValueError(f"世界观包含不支持的文件类型：{path.relative_to(root)}")
        if file_count > 1000 or total_size > 100_000_000:
            raise ValueError("世界观超过首版文件数量或容量限制")

    @staticmethod
    def _hash_tree(root: Path) -> str:
        digest = hashlib.sha256()
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        return digest.hexdigest()
