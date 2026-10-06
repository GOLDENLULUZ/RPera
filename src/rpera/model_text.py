from __future__ import annotations

import json
import re
from typing import Any


def render_model_text(value: Any) -> str:
    """Present structured context with verbatim strings instead of JSON escapes.

    Field paths retain hierarchy and list order. Fences delimit strings without
    indenting, interpreting, or normalizing their contents; lengths identify the
    original text independently of the newline before the closing fence.
    """
    sections: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            sections.append(f"{path} — object")
            for key, child in item.items():
                suffix = f".{key}" if key.isidentifier() else f"[{json.dumps(key, ensure_ascii=False)}]"
                visit(child, path + suffix)
        elif isinstance(item, list):
            sections.append(f"{path} — array ({len(item)} items)")
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, str):
            fence = "`" * max(3, 1 + max((len(run) for run in re.findall(r"`+", item)), default=0))
            sections.append(f"{path} — text ({len(item)} characters)\n{fence}text\n{item}\n{fence}")
        else:
            sections.append(f"{path} — {json.dumps(item, ensure_ascii=False, allow_nan=False)}")

    visit(value, "$")
    return "\n\n".join(sections)
