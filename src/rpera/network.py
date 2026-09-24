from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

import httpx

from .models import NetworkSettings


def is_local_target(url: str) -> bool:
    host = urlparse(url).hostname
    if not host:
        return False
    normalized = host.rstrip(".").lower()
    if normalized == "localhost" or normalized.endswith(".localhost") or normalized.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return "." not in normalized
    return address.is_loopback or address.is_private or address.is_link_local


def async_client_options(
    settings: NetworkSettings,
    target_url: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {"trust_env": False}
    if transport is not None:
        options["transport"] = transport
        return options
    if settings.mode == "custom" and not is_local_target(target_url):
        auth = (
            (settings.proxy_username, settings.proxy_password)
            if settings.proxy_username or settings.proxy_password
            else None
        )
        options["proxy"] = httpx.Proxy(settings.proxy_url, auth=auth)
    return options
