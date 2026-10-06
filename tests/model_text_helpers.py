from __future__ import annotations

import json
import re
from typing import Any


def decode_model_text(text: str) -> Any:
    """Decode the readable projection for scripted models, not for production.

    Stored results and native Tool Call arguments remain ordinary JSON. Tests
    can therefore inspect either representation without modifying real requests.
    """
    if not text.startswith("$ — "):
        return json.loads(text)
    position = 0
    result: Any = None

    def line() -> str:
        nonlocal position
        end = text.find("\n", position)
        if end < 0:
            end = len(text)
        value = text[position:end]
        position = end + 1
        return value

    def keys(path: str) -> list[str | int]:
        parts: list[str | int] = []
        offset = 1
        while offset < len(path):
            if path[offset] == ".":
                match = re.match(r"\.([^.[\]]+)", path[offset:])
                assert match
                parts.append(match[1])
                offset += len(match[0])
            else:
                assert path[offset] == "["
                key, end = json.JSONDecoder().raw_decode(path[offset + 1:])
                parts.append(key)
                offset += end + 2
                assert path[offset - 1] == "]"
        return parts

    while position < len(text):
        header = line()
        if not header:
            continue
        path, separator, kind = header.rpartition(" — ")
        assert separator
        if kind == "object":
            value: Any = {}
        elif kind.startswith("array ("):
            value = [None] * int(kind.split("(", 1)[1].split(" ", 1)[0])
        elif kind.startswith("text ("):
            length = int(kind.split("(", 1)[1].split(" ", 1)[0])
            opening = line()
            assert opening.endswith("text")
            fence = opening[:-4]
            value = text[position:position + length]
            position += length
            assert text[position:position + 1] == "\n"
            position += 1
            assert line() == fence
        else:
            value = json.loads(kind)
        parts = keys(path)
        if not parts:
            result = value
        else:
            parent = result
            for key in parts[:-1]:
                parent = parent[key]
            parent[parts[-1]] = value
    return result
