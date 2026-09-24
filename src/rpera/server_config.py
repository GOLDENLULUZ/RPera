from __future__ import annotations

import base64
import binascii
import os
import secrets
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BasicAuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, strict=True)
    username: str = ""
    password: str = ""

    @model_validator(mode="after")
    def require_credentials(self) -> BasicAuthConfig:
        if self.enabled and (not self.username or not self.password or ":" in self.username):
            raise ValueError("启用基本认证时必须填写用户名和密码，且用户名不能包含冒号")
        return self


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lan_access: bool = Field(default=False, strict=True)
    port: int = Field(default=8000, strict=True, ge=1, le=65535)
    basic_auth: BasicAuthConfig = Field(default_factory=BasicAuthConfig)


def load_server_config(data_dir: Path | None = None) -> ServerConfig:
    root = data_dir or Path(os.environ.get("RPERA_DATA_DIR") or os.environ.get("RPERA_USER_DATA_DIR") or PROJECT_ROOT)
    path = root / "config" / "server.json"
    if not path.exists():
        return ServerConfig()
    return ServerConfig.model_validate_json(path.read_text(encoding="utf-8"))


class BasicAuthMiddleware:
    def __init__(self, app: ASGIApp, config: BasicAuthConfig) -> None:
        self.app = app
        self.config = config

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.config.enabled:
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        authorization = headers.get(b"authorization", b"")
        try:
            scheme, encoded = authorization.split(b" ", 1)
            if scheme.lower() != b"basic":
                raise ValueError("invalid authentication scheme")
            credentials = base64.b64decode(encoded, validate=True).decode("utf-8")
            username, password = credentials.split(":", 1)
        except (ValueError, UnicodeError, binascii.Error):
            username, password = "", ""

        username_matches = secrets.compare_digest(username.encode("utf-8"), self.config.username.encode("utf-8"))
        password_matches = secrets.compare_digest(password.encode("utf-8"), self.config.password.encode("utf-8"))
        if username_matches and password_matches:
            await self.app(scope, receive, send)
            return

        response = PlainTextResponse(
            "Unauthorized",
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="RPera", charset="UTF-8"', "Cache-Control": "no-store"},
        )
        await response(scope, receive, send)
