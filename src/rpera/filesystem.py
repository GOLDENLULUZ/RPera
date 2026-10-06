from __future__ import annotations

import ctypes
import os
from pathlib import Path


if os.name == "nt":
    from ctypes import wintypes

    _move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
    _move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
    _move_file.restype = wintypes.BOOL


def atomic_replace(source: Path, target: Path) -> None:
    """Same-volume publication, with no copy/delete fallback or automatic retry."""
    if os.name != "nt":
        os.replace(source, target)
        return

    def windows_path(path: Path) -> str:
        absolute = os.path.abspath(path)
        if absolute.startswith("\\\\?\\"):
            return absolute
        if absolute.startswith("\\\\"):
            return "\\\\?\\UNC\\" + absolute[2:]
        return "\\\\?\\" + absolute

    # REPLACE_EXISTING | WRITE_THROUGH. COPY_ALLOWED is deliberately absent.
    if not _move_file(windows_path(source), windows_path(target), 0x1 | 0x8):
        raise ctypes.WinError(ctypes.get_last_error())


def sync_directory(path: Path) -> None:
    """Flush POSIX directory entries; Windows publishes via WRITE_THROUGH moves.

    Windows has no unprivileged, portable equivalent of directory fsync. File
    contents are still fsynced before publication; directory durability follows
    the native filesystem's guarantees, not a simulated POSIX directory flush.
    """
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
