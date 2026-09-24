from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from rpera.capabilities import (
    CapabilityContext,
    CapabilityError,
    CapabilityExecutor,
    CapabilityPermissionError,
    CapabilityRegistry,
    CapabilitySpec,
)


class AddInput(BaseModel):
    left: int
    right: int


async def test_capability_executor_validates_permissions_arguments_and_events() -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    async def emit(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    async def add(request: BaseModel, _context: CapabilityContext) -> dict[str, Any]:
        assert isinstance(request, AddInput)
        return {"sum": request.left + request.right}

    executor = CapabilityExecutor(CapabilityRegistry([CapabilitySpec("math_add", AddInput, add)]))
    context = CapabilityContext("save", "turn", "tester", emit)

    assert await executor.execute("math_add", {"left": 2, "right": 3}, context, {"math_add"}) == {"sum": 5}
    assert events == []

    with pytest.raises(CapabilityPermissionError, match="无权"):
        await executor.execute("math_add", {"left": 2, "right": 3}, context, set())

    with pytest.raises(CapabilityError, match="参数无效"):
        await executor.execute("math_add", {"left": "bad", "right": 3}, context, {"math_add"})


def test_capability_registry_rejects_duplicate_names() -> None:
    async def noop(_request: BaseModel, _context: CapabilityContext) -> dict[str, Any]:
        return {}

    spec = CapabilitySpec("duplicate", AddInput, noop)
    with pytest.raises(ValueError, match="重复"):
        CapabilityRegistry([spec, spec])
