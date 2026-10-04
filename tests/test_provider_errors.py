from __future__ import annotations

import httpx
import pytest

from rpera.models import AiPreset
from rpera.providers import BoundModelClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "streaming"),
    [
        ("deepseek", False),
        ("deepseek", True),
        ("google_gemini", True),
        ("anthropic", True),
    ],
)
async def test_cloudflare_html_errors_are_concise(provider: str, streaming: bool) -> None:
    html = (
        "<!doctype html><html><title>Cloudflare Tunnel error | Cloudflare</title>"
        '<script>errorCode: 1033</script><span>Ray ID: a427240e7e57e2c6</span>'
        + "<div>irrelevant styles and scripts</div>" * 300
        + "</html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(530, text=html, headers={"content-type": "text/html"})

    preset = AiPreset(
        id="test", name="测试", provider=provider, base_url="https://provider.example/v1",
        api_key="secret", model="example-model",
    )
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler), streaming=streaming)

    with pytest.raises(RuntimeError) as exc:
        await client.complete("system", [{"role": "user", "content": "hello"}])

    message = str(exc.value)
    assert message == "模型接口返回 HTTP 530: Cloudflare Tunnel 错误（1033）；Ray ID：a427240e7e57e2c6"
    assert "<!doctype" not in message


@pytest.mark.asyncio
async def test_provider_error_details_preserve_actionable_message_but_limit_length() -> None:
    responses = [
        httpx.Response(400, json={"error": {"code": "invalid_model", "message": "模型不可用"}}),
        httpx.Response(502, text="<html><body>" + "internal failure" * 300 + "</body></html>"),
        httpx.Response(429, text="rate limited " * 300),
        httpx.Response(401, json={"error": {"message": "secret " + "invalid token " * 100}}),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return responses.pop(0)

    preset = AiPreset(
        id="test", name="测试", provider="deepseek", base_url="https://provider.example/v1",
        api_key="secret", model="example-model",
    )
    client = BoundModelClient(preset, transport=httpx.MockTransport(handler))
    messages = []
    for _ in range(4):
        with pytest.raises(RuntimeError) as exc:
            await client.complete("system", [{"role": "user", "content": "hello"}])
        messages.append(str(exc.value))

    assert messages[0] == "模型接口返回 HTTP 400: invalid_model: 模型不可用"
    assert messages[1] == "模型接口返回 HTTP 502: 接口返回了 HTML 错误页面"
    assert messages[2].startswith("模型接口返回 HTTP 429: rate limited")
    assert len(messages[2]) < 550
    assert messages[2].endswith("…")
    assert messages[3].startswith("模型接口返回 HTTP 401: *** invalid token")
    assert "secret" not in messages[3]
    assert len(messages[3]) < 550
