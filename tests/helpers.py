from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rpera.app import PROJECT_ROOT


def make_symlink(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("Windows symlink setup requires Developer Mode or symlink privileges")
        raise


def make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "worlds").mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "data" / "worlds" / "雾港", data_dir / "worlds" / "雾港")
    (data_dir / "mods").mkdir()
    shutil.copytree(PROJECT_ROOT / "data" / "mods" / "读心大师", data_dir / "mods" / "读心大师")
    return data_dir


def wait_for_turn(client: TestClient, save_id: str, timeout: float = 30) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turns = client.get(f"/api/saves/{save_id}/turns").json()
        if turns and turns[-1]["status"] != "running":
            return turns[-1]
        time.sleep(0.01)
    raise AssertionError("turn did not finish")


def wait_for_revision(path: Path) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_dir():
            return
        time.sleep(0.01)
    raise AssertionError(f"回合检查点未完成：{path}")
