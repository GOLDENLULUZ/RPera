from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from rpera.app import create_app
from rpera.server_config import load_server_config
from tests.helpers import make_data_dir


def write_server_config(data_dir: Path, settings: Mapping[str, object]) -> None:
    config_dir = data_dir / "config"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "server.json").write_text(json.dumps(settings), encoding="utf-8")


def test_server_config_defaults_to_local_without_auth(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    config = load_server_config(data_dir)
    assert config.lan_access is False
    assert config.port == 8000
    assert config.basic_auth.enabled is False

    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/worlds").status_code == 200


def test_server_config_requires_credentials_and_rejects_invalid_json(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    for settings in (
        {"lan_access": "yes"},
        {"port": 0},
        {"port": 65536},
        {"port": "8080"},
        {"port": True},
        {"unknown": True},
        {"basic_auth": {"enabled": True, "username": "alice", "password": ""}},
        {"basic_auth": {"enabled": True, "username": "a:b", "password": "secret"}},
    ):
        write_server_config(data_dir, settings)
        with pytest.raises(ValidationError):
            load_server_config(data_dir)

    write_server_config(data_dir, {"basic_auth": {"enabled": True}})
    with pytest.raises(ValidationError):
        create_app(data_dir=data_dir)

    (data_dir / "config" / "server.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_server_config(data_dir)


def test_basic_auth_protects_page_assets_api_and_sse(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    write_server_config(data_dir, {
        "lan_access": True,
        "basic_auth": {"enabled": True, "username": "玩家", "password": "秘密:123"},
    })
    assert load_server_config(data_dir).lan_access is True

    with TestClient(create_app(data_dir=data_dir)) as client:
        for path in ("/", "/static/app.js", "/api/worlds", "/api/saves/test/events", "/docs"):
            response = client.get(path)
            assert response.status_code == 401
            assert response.headers["www-authenticate"] == 'Basic realm="RPera", charset="UTF-8"'

        assert client.put("/api/runtime-settings", json={}).status_code == 401
        assert client.get("/api/worlds", auth=("玩家", "wrong")).status_code == 401
        assert client.get("/api/worlds", headers={"Authorization": "Basic !!!"}).status_code == 401
        assert client.get("/api/worlds", auth=("玩家", "秘密:123")).status_code == 200
        assert client.get("/static/app.js", auth=("玩家", "秘密:123")).status_code == 200


def test_user_data_dir_selects_server_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = make_data_dir(tmp_path)
    write_server_config(data_dir, {"lan_access": True})
    monkeypatch.setenv("RPERA_USER_DATA_DIR", str(data_dir))
    monkeypatch.delenv("RPERA_DATA_DIR", raising=False)
    assert load_server_config().lan_access is True

    other_data = tmp_path / "shared"
    other_data.mkdir()
    write_server_config(other_data, {"lan_access": False})
    monkeypatch.setenv("RPERA_DATA_DIR", str(other_data))
    assert load_server_config().lan_access is False


def test_custom_port_comes_from_server_json_not_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = make_data_dir(tmp_path)
    write_server_config(data_dir, {"port": 8080})
    monkeypatch.setenv("RPERA_PORT", "9000")

    assert load_server_config(data_dir).port == 8080
