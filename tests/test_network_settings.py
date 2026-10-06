from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.models import NetworkSettings
from rpera.network import async_client_options, is_local_target
from tests.helpers import make_data_dir
from tests.test_prototype import ScriptedModelClient


def test_network_settings_api_persists_and_masks_password(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())

    with TestClient(app) as client:
        assert client.get("/api/network-settings").json() == {
            "mode": "direct",
            "proxy_url": "",
            "proxy_username": "",
            "has_proxy_password": False,
            "masked_proxy_password": "",
        }
        response = client.put(
            "/api/network-settings",
            json={
                "mode": "custom",
                "proxy_url": "socks5h://192.168.1.2:7891",
                "proxy_username": "proxy-user",
                "proxy_password": "proxy-secret",
            },
        )
        assert response.status_code == 200
        assert response.json()["masked_proxy_password"] == "***"
        assert "proxy-secret" not in response.text

        path = data_dir / "config" / "network_settings.json"
        assert "proxy-secret" in path.read_text(encoding="utf-8")
        if os.name != "nt":
            assert path.stat().st_mode & 0o777 == 0o600

        preserved = client.put(
            "/api/network-settings",
            json={
                "mode": "custom",
                "proxy_url": "http://192.168.1.2:7890",
                "proxy_username": "proxy-user",
                "proxy_password": "",
            },
        ).json()
        assert preserved["has_proxy_password"] is True

        cleared = client.put(
            "/api/network-settings",
            json={
                "mode": "custom",
                "proxy_url": "http://192.168.1.2:7890",
                "proxy_username": "proxy-user",
                "clear_proxy_password": True,
            },
        ).json()
        assert cleared["has_proxy_password"] is False
        assert "proxy-secret" not in path.read_text(encoding="utf-8")


def test_network_settings_reject_invalid_custom_proxy(tmp_path: Path) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ScriptedModelClient())
    with TestClient(app) as client:
        for proxy_url in ("", "ftp://proxy.example:21", "http://user:pass@proxy.example:8080", "http://proxy.example/path"):
            response = client.put("/api/network-settings", json={"mode": "custom", "proxy_url": proxy_url})
            assert response.status_code == 422


def test_local_targets_bypass_custom_proxy() -> None:
    for url in (
        "http://localhost:7860",
        "http://127.0.0.1:7860",
        "http://192.168.1.20:7860",
        "http://10.0.0.2:7860",
        "http://172.16.0.2:7860",
        "http://[::1]:7860",
        "http://renderer.local:7860",
        "http://stable-diffusion:7860",
    ):
        assert is_local_target(url)
    assert not is_local_target("https://api.example.com/v1")


def test_client_options_disable_environment_and_apply_authenticated_proxy() -> None:
    settings = NetworkSettings(
        mode="custom",
        proxy_url="http://192.168.1.2:7890",
        proxy_username="proxy-user",
        proxy_password="proxy-secret",
    )
    assert async_client_options(settings, "http://127.0.0.1:7860") == {"trust_env": False}

    with patch("rpera.network.httpx.Proxy") as proxy:
        marker = object()
        proxy.return_value = marker
        options = async_client_options(settings, "https://api.example.com/v1")

    proxy.assert_called_once_with("http://192.168.1.2:7890", auth=("proxy-user", "proxy-secret"))
    assert options == {"trust_env": False, "proxy": marker}
