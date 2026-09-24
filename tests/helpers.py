from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from rpera.app import PROJECT_ROOT


def make_data_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    (data_dir / "worlds").mkdir(parents=True)
    shutil.copytree(PROJECT_ROOT / "data" / "worlds" / "雾港", data_dir / "worlds" / "雾港")
    (data_dir / "mods").mkdir()
    shutil.copytree(PROJECT_ROOT / "data" / "mods" / "读心大师", data_dir / "mods" / "读心大师")
    return data_dir


def wait_for_turn(client: TestClient, save_id: str, timeout: float = 10) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turns = client.get(f"/api/saves/{save_id}/turns").json()
        if turns and turns[-1]["status"] != "running":
            return turns[-1]
        time.sleep(0.01)
    raise AssertionError("turn did not finish")
