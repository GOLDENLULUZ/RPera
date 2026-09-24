from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .draft_files import DraftFileError, DraftFileStore


STORY_SUMMARY_FILE = "story_summary.json"
EMPTY_STORY_SUMMARY = '{"turns": []}\n'


class StorySummaryStore:
    """One validated, atomically edited JSON summary in a save directory."""

    def __init__(self, save_dir: Path) -> None:
        self.files = DraftFileStore(save_dir)

    def ensure(self) -> None:
        if not self.files.exists(STORY_SUMMARY_FILE):
            self.files.write(STORY_SUMMARY_FILE, EMPTY_STORY_SUMMARY)

    def read(self) -> dict[str, Any]:
        content = self.files.read(STORY_SUMMARY_FILE)
        return {"path": STORY_SUMMARY_FILE, "content": content, "summary": self._validate(content)}

    def edit(self, old_text: str, new_text: str, current_turn_number: int) -> dict[str, Any]:
        content = self.files.read(STORY_SUMMARY_FILE)
        self._validate(content)
        if not old_text or content.count(old_text) != 1:
            raise DraftFileError("old_text 必须在故事摘要中唯一出现一次")
        replacement = content.replace(old_text, new_text, 1)
        summary = self._validate(replacement)
        if any(item["turn_number"] >= current_turn_number for item in summary["turns"]):
            raise ValueError("故事摘要不能包含当前或未来回合")
        self.files.edit(STORY_SUMMARY_FILE, old_text, new_text)
        return {"path": STORY_SUMMARY_FILE, "content": replacement, "summary": summary}

    @staticmethod
    def _validate(content: str) -> dict[str, Any]:
        try:
            value = json.loads(content)
        except json.JSONDecodeError as error:
            raise ValueError(f"故事摘要不是合法 JSON：{error}") from error
        if not isinstance(value, dict) or set(value) != {"turns"} or not isinstance(value["turns"], list):
            raise ValueError("故事摘要必须是包含 turns 数组的 JSON 对象")
        previous = 0
        for item in value["turns"]:
            if (not isinstance(item, dict) or set(item) != {"turn_number", "summary"}
                    or type(item["turn_number"]) is not int or item["turn_number"] <= previous
                    or not isinstance(item["summary"], str) or not item["summary"].strip()):
                raise ValueError("故事摘要必须按递增 turn_number 保存非空 summary，且不能重复编号")
            previous = item["turn_number"]
        return value
