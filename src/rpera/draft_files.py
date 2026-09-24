from __future__ import annotations

import os
import re
import stat
import uuid
from pathlib import Path


_DIRECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_MAX_FILE_BYTES = 2_000_000


class DraftFileError(RuntimeError):
    pass


class DraftFileStore:
    """Crash-safe operations on one direct child of a Runtime-owned draft directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def read(self, name: str) -> str:
        path = self._path(name)
        file_stat = self._regular_file(path)
        if file_stat.st_size > _MAX_FILE_BYTES:
            raise DraftFileError("草稿文件超过大小限制")
        try:
            data = path.read_bytes()
        except OSError as error:
            raise DraftFileError(f"无法读取草稿文件：{error}") from error
        if len(data) > _MAX_FILE_BYTES:
            raise DraftFileError("草稿文件超过大小限制")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DraftFileError("草稿文件不是合法 UTF-8") from error

    def exists(self, name: str) -> bool:
        path = self._path(name)
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            return False
        if stat.S_ISLNK(file_stat.st_mode):
            raise DraftFileError(f"草稿文件不能是符号链接：{path.name}")
        if not stat.S_ISREG(file_stat.st_mode):
            raise DraftFileError(f"草稿路径不是普通文件：{path.name}")
        return True

    def write(self, name: str, content: str) -> None:
        path = self._path(name)
        try:
            path.lstat()
        except FileNotFoundError:
            pass
        else:
            raise DraftFileError(f"草稿文件已经存在：{name}")
        self._replace(path, content)

    def edit(self, name: str, old_text: str, new_text: str) -> None:
        current = self.read(name)
        if current.count(old_text) != 1:
            raise DraftFileError("old_text 必须在草稿中唯一出现一次")
        self._replace(self._path(name), current.replace(old_text, new_text, 1))

    def _path(self, name: str) -> Path:
        if not _DIRECT_NAME.fullmatch(name) or name in {".", ".."}:
            raise DraftFileError("只允许一个安全的直属文件名")
        try:
            root_stat = self.root.lstat()
        except FileNotFoundError as error:
            raise DraftFileError("草稿目录不存在") from error
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            raise DraftFileError("草稿根路径必须是非符号链接目录")
        return self.root / name

    @staticmethod
    def _regular_file(path: Path) -> os.stat_result:
        try:
            file_stat = path.lstat()
        except FileNotFoundError as error:
            raise DraftFileError(f"草稿文件不存在：{path.name}") from error
        if stat.S_ISLNK(file_stat.st_mode):
            raise DraftFileError(f"草稿文件不能是符号链接：{path.name}")
        if not stat.S_ISREG(file_stat.st_mode):
            raise DraftFileError(f"草稿路径不是普通文件：{path.name}")
        return file_stat

    def _replace(self, path: Path, content: str) -> None:
        data = content.encode("utf-8")
        if len(data) > _MAX_FILE_BYTES:
            raise DraftFileError("草稿文件超过大小限制")
        temporary = self.root / f".{path.name}.{uuid.uuid4()}.tmp"
        try:
            with temporary.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            raise DraftFileError(f"无法原子写入草稿文件：{error}") from error
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
