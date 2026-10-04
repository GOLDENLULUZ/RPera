from __future__ import annotations

import re
from typing import Any


CONTENT_BLOCK_REASONS = frozenset({"PROHIBITED_CONTENT", "SAFETY", "BLOCKLIST", "SPII", "RECITATION"})


def is_content_block_error(error: dict[str, Any]) -> bool:
    if any(isinstance(error.get(key), str) and error[key] in CONTENT_BLOCK_REASONS for key in ("status", "code")):
        return True
    details = error.get("details")
    if isinstance(details, list) and any(
        isinstance(detail, dict) and isinstance(detail.get("reason"), str)
        and detail["reason"] in CONTENT_BLOCK_REASONS for detail in details
    ):
        return True
    message = error.get("message")
    return isinstance(message, str) and bool(re.search(r"\bPROHIBITED_CONTENT\b", message))


class ModelFallbackError(RuntimeError):
    """A provider rejected content or is temporarily unavailable."""

    def __init__(self, reason: str, message: str) -> None:
        self.reason = reason
        super().__init__(message)
