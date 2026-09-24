from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from tests.helpers import make_data_dir, wait_for_turn
from tests.test_prototype import ScriptedModelClient
from tests.test_turn_images import png_bytes


def wait_for_revision(path: Path) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.is_dir():
            return
        time.sleep(0.01)
    raise AssertionError(f"回合检查点未完成：{path}")


def test_branch_restores_completed_turn_files_and_history(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        source = client.post("/api/saves", json={"world_name": "雾港", "name": "原时间线"}).json()
        save_id = source["id"]
        base = f"/api/saves/{save_id}"
        entity_path = "entities/character/伊蕾/ENTITY.md"

        def describe(text: str) -> None:
            original = client.get(f"{base}/entity", params={"path": entity_path}).json()
            response = client.put(f"{base}/entity", params={"path": entity_path}, json={
                "revision": original["revision"], "description": text,
                "aliases": original["aliases"], "visibility": original["visibility"],
                "required": original["required"], "content": original["content"],
            })
            assert response.status_code == 200

        describe("第一回合之前")
        assert client.post(f"{base}/turns", json={"content": "行动一"}).status_code == 202
        first = wait_for_turn(client, save_id)
        assert first["status"] == "completed"
        describe("第二回合之前")
        summary = data_dir / "saves" / save_id / "current" / "story_summary.json"
        summary.write_text(json.dumps({"turns": [{"turn_number": 1, "summary": "先前的摘要"}]}, ensure_ascii=False))
        image = png_bytes((30, 60, 90))
        assert client.post(f"{base}/turns", data={"content": "行动二"}, files=[
            ("images", ("回合二.png", image, "image/png")),
        ]).status_code == 202
        second = wait_for_turn(client, save_id)
        assert second["status"] == "completed"
        snapshot = data_dir / "saves" / save_id / "revisions" / second["id"]
        wait_for_revision(snapshot)
        assert (snapshot / "world_snapshot" / entity_path).is_file()

        describe("第三回合之前")
        summary.write_text(json.dumps({"turns": [{"turn_number": 1, "summary": "改过的摘要"}]}, ensure_ascii=False))
        assert client.post(f"{base}/turns", json={"content": "行动三"}).status_code == 202
        third = wait_for_turn(client, save_id)
        assert third["status"] == "completed"

        invalid = client.post(f"{base}/turns/{third['id']}/branch", json={"name": " "})
        assert invalid.status_code == 422
        missing = client.post(f"{base}/turns/not-a-turn/branch", json={"name": "不存在"})
        assert missing.status_code == 404
        branch_response = client.post(f"{base}/turns/{second['id']}/branch", json={"name": "第二回合分支"})
        assert branch_response.status_code == 201, branch_response.text
        branch = branch_response.json()
        branch_id = branch["id"]
        assert branch_id != save_id
        assert branch["branch_source_save_id"] == save_id
        assert branch["branch_source_turn_id"] == second["id"]
        assert branch["branch_source_turn_number"] == 2
        assert [turn["id"] for turn in client.get(f"/api/saves/{branch_id}/turns").json()] == [first["id"], second["id"]]
        branch_entity = client.get(f"/api/saves/{branch_id}/entity", params={"path": entity_path}).json()
        assert branch_entity["description"] == "第二回合之前"
        assert client.get(f"{base}/entity", params={"path": entity_path}).json()["description"] == "第三回合之前"
        branch_summary = data_dir / "saves" / branch_id / "current" / "story_summary.json"
        assert json.loads(branch_summary.read_text())["turns"][0]["summary"] == "先前的摘要"
        branched_second = client.get(f"/api/saves/{branch_id}/turns").json()[1]
        assert client.get(branched_second["images"][0]["content_url"]).content == image
        assert (data_dir / "saves" / branch_id / "revisions" / first["id"]).is_dir()

        earlier = client.post(f"/api/saves/{branch_id}/turns/{first['id']}/branch", json={"name": "第一回合分支"})
        assert earlier.status_code == 201
        assert client.get(f"/api/saves/{earlier.json()['id']}/entity", params={"path": entity_path}).json()[
            "description"
        ] == "第一回合之前"

        assert client.post(f"/api/saves/{branch_id}/turns", json={"content": "分支的第三回合"}).status_code == 202
        branch_third = wait_for_turn(client, branch_id)
        assert branch_third["status"] == "completed"
        assert app.state.saves.load_story(branch_id, branch_third["id"])["story"]["turns"][-1]["turn_number"] == 3
        assert len(client.get(f"{base}/turns").json()) == 3
        assert client.delete(f"/api/saves/{branch_id}").status_code == 200
        assert (data_dir / "saves" / save_id / "revisions" / second["id"]).is_dir()


def test_branch_rejects_unfinished_turn_and_completed_checkpoint_recovers(tmp_path: Path) -> None:
    data_dir = make_data_dir(tmp_path)
    app = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(app) as client:
        source = client.post("/api/saves", json={"world_name": "雾港", "name": "恢复"}).json()
        save_id = source["id"]
        turn = app.state.saves.create_turn(save_id, "失败的尝试", 4)
        app.state.saves.fail_turn(save_id, turn.id, "turn.failed", {"message": "测试"})
        response = client.post(f"/api/saves/{save_id}/turns/{turn.id}/branch", json={"name": "错误分支"})
        assert response.status_code == 409
        assert client.post(f"/api/saves/{save_id}/turns", json={"content": "后来的行动"}).status_code == 202
        completed = wait_for_turn(client, save_id)
        assert completed["status"] == "completed"
        revision = data_dir / "saves" / save_id / "revisions" / completed["id"]
        # Mimic an interruption after SQLite publication but before checkpoint rename.
        wait_for_revision(revision)
        shutil.rmtree(revision)
        app.state.saves.recover_interrupted_turns()
        assert revision.is_dir()
        branch = client.post(f"/api/saves/{save_id}/turns/{completed['id']}/branch", json={"name": "恢复分支"})
        assert branch.status_code == 201
        assert [item["status"] for item in client.get(f"/api/saves/{branch.json()['id']}/turns").json()] == [
            "failed", "completed",
        ]


def test_checkpoint_copy_failure_keeps_publication_and_blocks_edits_until_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(data_dir=make_data_dir(tmp_path), model_client=ScriptedModelClient())
    with TestClient(app) as client:
        source = client.post("/api/saves", json={"world_name": "雾港", "name": "未完成复制"}).json()
        save_id = source["id"]
        original_copy = app.state.saves._copy_state

        def unavailable(_source: Path, _target: Path) -> None:
            raise OSError("测试：检查点磁盘不可写")

        monkeypatch.setattr(app.state.saves, "_copy_state", unavailable)
        assert client.post(f"/api/saves/{save_id}/turns", json={"content": "回合"}).status_code == 202
        completed = wait_for_turn(client, save_id)
        assert completed["status"] == "completed"
        assert completed["narrative"]
        denied = client.post(f"/api/saves/{save_id}/entities", json={"name": "不能覆盖检查点", "type": "event"})
        assert denied.status_code == 409
        monkeypatch.setattr(app.state.saves, "_copy_state", original_copy)
        result = client.post(f"/api/saves/{save_id}/turns/{completed['id']}/branch", json={"name": "修复后"})
        assert result.status_code == 201
