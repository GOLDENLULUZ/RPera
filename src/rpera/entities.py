from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from .content import (
    content_name,
    content_name_key,
    entity_snapshot_path,
    entity_type,
    is_internal_temporary_name,
    parse_entity_document,
)


TERM_RE = re.compile(r"[\w\u4e00-\u9fff]+")


@dataclass
class EntityEntry:
    path: str
    name: str
    description: str = ""
    aliases: list[str] = field(default_factory=list)
    visibility: str = "world_truth"
    required: bool = False
    source_order: int = field(default=0, repr=False)


@dataclass
class EntityDocument:
    path: str
    name: str
    description: str
    aliases: list[str]
    visibility: str
    content: str
    required: bool = False


@dataclass(frozen=True)
class EntityRoot:
    root: Path
    source_order: int


def _score_entry(entry: EntityEntry, terms: list[str]) -> int:
    name = entry.name.casefold()
    score = 0
    for term in terms:
        if term == name:
            score += 8
        elif term in name:
            score += 5
        if any(term == alias.casefold() for alias in entry.aliases):
            score += 6
        elif any(term in alias.casefold() for alias in entry.aliases):
            score += 3
        if term in entry.description.casefold():
            score += 1
    return score


def search_entry_list(entries: list[EntityEntry], query: str, limit: int = 12) -> list[EntityEntry]:
    candidates = [entry for entry in entries if not entry.required]
    terms = [term.casefold() for term in TERM_RE.findall(query) if term.strip()]
    if not terms:
        return candidates[:limit]
    scored = [(_score_entry(entry, terms), entry) for entry in candidates]
    scored = [item for item in scored if item[0] > 0]
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].name.casefold(),
            item[1].source_order,
            item[1].path,
        )
    )
    return [entry for _, entry in scored[:limit]]


class EntityStore:
    def __init__(self, snapshot_root: Path) -> None:
        if snapshot_root.is_symlink() or not snapshot_root.is_dir():
            raise ValueError("实体快照根路径必须是非符号链接目录")
        self.root = snapshot_root
        self._entries: list[EntityEntry] | None = None

    def entries(self) -> list[EntityEntry]:
        if self._entries is None:
            self._entries = self._scan()
        return self._entries

    def required_entries(self) -> list[EntityEntry]:
        return [entry for entry in self.entries() if entry.required]

    def required_paths(self) -> list[str]:
        return [entry.path for entry in self.required_entries()]

    def search(self, query: str, limit: int = 12) -> list[EntityEntry]:
        return search_entry_list(self.entries(), query, limit)

    def required_documents(self) -> list[EntityDocument]:
        return [self._document(entry) for entry in self.required_entries()]

    def read(self, paths: list[str]) -> list[EntityDocument]:
        indexed = {entry.path: entry for entry in self.entries()}
        documents: list[EntityDocument] = []
        for path in paths:
            entry = indexed.get(path)
            if entry is not None and not entry.required:
                documents.append(self._document(entry))
        return documents

    def read_exact(self, paths: list[str]) -> list[EntityDocument]:
        indexed = {entry.path: entry for entry in self.entries()}
        documents: list[EntityDocument] = []
        for path in paths:
            entry = indexed.get(path)
            if entry is None:
                raise ValueError(f"实体引用不存在：{path}")
            documents.append(self._document(entry))
        return documents

    def _scan(self) -> list[EntityEntry]:
        parsed: list[EntityEntry] = []
        names_by_source: dict[Path, set[str]] = {}
        for root, name, path in self._entity_files():
            document = parse_entity_document(path.read_text(encoding="utf-8"))
            names = names_by_source.setdefault(root.root, set())
            name_key = content_name_key(name)
            if name_key in names:
                raise ValueError(f"来源内实体名称重复：{name}")
            names.add(name_key)
            snapshot_path = entity_snapshot_path(path.relative_to(self.root).as_posix())
            entry = EntityEntry(
                path=snapshot_path,
                name=content_name(name),
                description=document.description,
                aliases=document.aliases,
                visibility=document.visibility,
                required=document.required,
                source_order=root.source_order,
            )
            parsed.append(entry)
        return sorted(
            parsed,
            key=lambda entry: (entry.source_order, entry.path),
        )

    def _entity_files(self) -> Iterator[tuple[EntityRoot, str, Path]]:
        for root in self._roots():
            entities_dir = root.root / "entities"
            if not entities_dir.exists() and not entities_dir.is_symlink():
                continue
            if entities_dir.is_symlink() or not entities_dir.is_dir():
                raise ValueError("实体根路径必须是非符号链接目录")
            for type_dir in sorted(entities_dir.iterdir()):
                if type_dir.is_symlink() or not type_dir.is_dir():
                    raise ValueError(f"实体类型路径必须是非符号链接目录：{type_dir.name}")
                entity_type(type_dir.name)
                for entity_dir in sorted(type_dir.iterdir()):
                    if entity_dir.is_symlink() or not entity_dir.is_dir():
                        raise ValueError(f"实体路径必须是非符号链接目录：{entity_dir.name}")
                    name = content_name(entity_dir.name)
                    children = [
                        child
                        for child in entity_dir.iterdir()
                        if child.is_symlink()
                        or not child.is_file()
                        or not is_internal_temporary_name(child.name, target_name="ENTITY.md")
                    ]
                    if len(children) != 1 or children[0].name != "ENTITY.md":
                        raise ValueError(f"实体目录必须只包含 ENTITY.md：{name}")
                    path = entity_dir / "ENTITY.md"
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(f"实体文件必须是非符号链接普通文件：{name}")
                    yield root, name, path

    def _roots(self) -> list[EntityRoot]:
        roots = [EntityRoot(self.root, 0)]
        mods_dir = self.root / "mods"
        if not mods_dir.exists() and not mods_dir.is_symlink():
            return roots
        if mods_dir.is_symlink() or not mods_dir.is_dir():
            raise ValueError("模组快照根路径必须是非符号链接目录")
        mods: list[tuple[str, Path]] = []
        for path in sorted(item for item in mods_dir.iterdir() if item.is_dir()):
            if path.is_symlink():
                raise ValueError(f"模组根目录不允许使用符号链接：{path.name}")
            name = content_name(path.name)
            mods.append((name, path))
        for order, (_source_name, path) in enumerate(sorted(mods, key=lambda item: content_name_key(item[0])), start=1):
            roots.append(EntityRoot(path, order))
        return roots

    def _document(self, entry: EntityEntry) -> EntityDocument:
        document = parse_entity_document((self.root / entry.path).read_text(encoding="utf-8"))
        return EntityDocument(
            path=entry.path,
            name=entry.name,
            description=document.description,
            aliases=document.aliases,
            visibility=document.visibility,
            content=document.content,
            required=document.required,
        )
