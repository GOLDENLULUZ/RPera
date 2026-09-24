from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread
from typing import Any

from fastapi.testclient import TestClient

import rpera.content_store as content_store_module
import rpera.storage as storage_module
from rpera.app import create_app
from rpera.content import markdown_frontmatter, parse_scenario_document
from rpera.entities import EntityStore
from tests.helpers import wait_for_turn
from tests.test_prototype import ScriptedModelClient


def test_content_crud_uses_separate_create_save_and_rename_operations(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    app = create_app(data_dir=data_dir)

    with TestClient(app) as client:
        world = client.post("/api/content/worlds", json={"name": "空白世界"})
        assert world.status_code == 201
        assert world.json() == {
            "name": "空白世界",
            "description": "",
            "path": "worlds/空白世界",
            "scenario_count": 0,
            "entity_count": 0,
        }
        world_dir = data_dir / "worlds" / "空白世界"
        assert list(world_dir.iterdir()) == []

        saved = client.put("/api/content/worlds/空白世界", json={"description": "一个世界。"})
        assert saved.status_code == 200
        assert saved.json()["name"] == "空白世界"
        renamed = client.post("/api/content/worlds/空白世界/rename", json={"name": "中文世界"})
        assert renamed.status_code == 200
        assert renamed.json()["description"] == "一个世界。"
        world_dir = data_dir / "worlds" / "中文世界"

        scenario = client.post("/api/content/worlds/中文世界/scenarios", json={"name": "初见"})
        assert scenario.status_code == 201
        assert scenario.json()["opening"] == ""
        scenario_path = world_dir / "scenarios" / "初见.md"
        front, body = markdown_frontmatter(scenario_path.read_text(encoding="utf-8"), "场景")
        assert front == {"description": ""}
        assert body == ""

        saved_scenario = client.put(
            "/api/content/worlds/中文世界/scenarios/初见",
            json={"description": "雨夜", "opening": "雨落在青石路上。"},
        )
        assert saved_scenario.status_code == 200
        renamed_scenario = client.post(
            "/api/content/worlds/中文世界/scenarios/初见/rename",
            json={"name": "雨夜初见"},
        )
        assert renamed_scenario.status_code == 200
        assert renamed_scenario.json()["opening"] == "雨落在青石路上。"
        assert not scenario_path.exists()

        entity = client.post(
            "/api/content/worlds/中文世界/entities",
            json={"name": "旅人", "type": "character"},
        )
        assert entity.status_code == 201
        assert entity.json()["content"] == ""
        entity_path = world_dir / "entities" / "character" / "旅人" / "ENTITY.md"
        entity_front, entity_body = markdown_frontmatter(entity_path.read_text(encoding="utf-8"), "实体")
        assert entity_front == {
            "description": "",
            "aliases": [],
            "visibility": "world_truth",
            "required": False,
        }
        assert entity_body == ""

        payload = {
            "description": "一口古井",
            "aliases": ["水井", "水井", "井"],
            "visibility": "world_truth",
            "required": True,
            "content": "井底没有回声。",
        }
        saved_entity = client.put("/api/content/worlds/中文世界/entities/旅人", json=payload)
        assert saved_entity.status_code == 200
        assert saved_entity.json()["aliases"] == ["水井", "井"]
        moved_entity = client.post(
            "/api/content/worlds/中文世界/entities/旅人/rename",
            json={"name": "古井", "type": "location"},
        )
        assert moved_entity.status_code == 200
        assert moved_entity.json()["content"] == "井底没有回声。"
        assert not entity_path.exists()

        client.post("/api/content/mods", json={"name": "天气模组"})
        renamed_mod = client.post("/api/content/mods/天气模组/rename", json={"name": "四季模组"})
        assert renamed_mod.status_code == 200

    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.get("/api/content/worlds/中文世界").json()["scenario_count"] == 1
        assert client.get("/api/content/worlds/中文世界/entities/古井").json()["type"] == "location"
        assert client.delete("/api/content/worlds/中文世界/entities/古井").json()["deleted"] is True
        assert client.delete("/api/content/worlds/中文世界/scenarios/雨夜初见").json()["deleted"] is True
        assert client.delete("/api/content/mods/四季模组").json()["deleted"] is True
        assert client.delete("/api/content/worlds/中文世界").json()["deleted"] is True


def test_content_api_rejects_conflicts_invalid_fields_and_unsafe_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.post("/api/content/worlds", json={"name": "Alpha"}).status_code == 201
        assert client.post("/api/content/worlds", json={"name": "alpha"}).status_code == 409
        assert client.post("/api/content/worlds", json={"name": " bad"}).status_code == 422
        assert client.put("/api/content/worlds/Alpha", json={"name": "Beta", "description": "x"}).status_code == 422
        assert client.get("/api/content/worlds/missing").status_code == 404

        assert client.post("/api/content/worlds/Alpha/scenarios", json={"name": "One"}).status_code == 201
        assert client.post("/api/content/worlds/Alpha/scenarios", json={"name": "one"}).status_code == 409
        assert client.put(
            "/api/content/worlds/Alpha/scenarios/One",
            json={"description": "x" * 4_001, "opening": ""},
        ).status_code == 422

        assert client.post(
            "/api/content/worlds/Alpha/entities", json={"name": "Person", "type": "person"}
        ).status_code == 422
        assert client.post(
            "/api/content/worlds/Alpha/entities", json={"name": "Person", "type": "character"}
        ).status_code == 201
        base = {
            "description": "",
            "aliases": [],
            "visibility": "world_truth",
            "required": False,
            "content": "",
        }
        assert client.put(
            "/api/content/worlds/Alpha/entities/Person", json={**base, "required": "true"}
        ).status_code == 422
        assert client.put(
            "/api/content/worlds/Alpha/entities/Person", json={**base, "aliases": [f"alias {i}" for i in range(21)]}
        ).status_code == 422
        assert client.put(
            "/api/content/worlds/Alpha/entities/Person", json={**base, "aliases": ["first,second"]}
        ).status_code == 422
        trimmed_alias = client.put(
            "/api/content/worlds/Alpha/entities/Person",
            json={**base, "aliases": [" alias ", " ", "", "alias"]},
        )
        assert trimmed_alias.status_code == 200
        assert trimmed_alias.json()["aliases"] == ["alias"]
        maximum = {
            **base,
            "description": "d" * 4_000,
            "aliases": [f"alias-{i}-" + "a" * 60 for i in range(20)],
            "content": "c" * 100_000,
        }
        assert client.put("/api/content/worlds/Alpha/entities/Person", json=maximum).status_code == 200

        escaped_maximum = {**base, "description": "\ufeff" * 4_000, "content": "c" * 100_000}
        assert client.put("/api/content/worlds/Alpha/entities/Person", json=escaped_maximum).status_code == 200

        broken = data_dir / "worlds" / "Alpha" / "entities" / "location" / "Broken"
        broken.mkdir(parents=True)
        assert client.get("/api/content/worlds/Alpha").status_code == 400
        assert client.post("/api/saves", json={"world_name": "Alpha", "name": "损坏内容"}).status_code == 400
        broken.rmdir()

    description = data_dir / "worlds" / "Alpha" / "description.md"
    description.write_bytes(b"\xff")
    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.get("/api/content/worlds").status_code == 400


def test_content_round_trips_boundaries_and_ignores_only_internal_temporary_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.post("/api/content/worlds", json={"name": "边界世界"}).status_code == 201
        assert client.post("/api/content/worlds/边界世界/scenarios", json={"name": "开局"}).status_code == 201
        opening = "\n第一行\n\n"
        scenario = client.put(
            "/api/content/worlds/边界世界/scenarios/开局",
            json={"description": "\ufeff" * 4_000, "opening": opening},
        )
        assert scenario.status_code == 200
        assert scenario.json()["opening"] == opening

        assert client.post(
            "/api/content/worlds/边界世界/entities", json={"name": "旅人", "type": "character"}
        ).status_code == 201
        content = "\n正文\n"
        entity = client.put(
            "/api/content/worlds/边界世界/entities/旅人",
            json={
                "description": "",
                "aliases": [],
                "visibility": "world_truth",
                "required": False,
                "content": content,
            },
        )
        assert entity.status_code == 200
        assert entity.json()["content"] == content

        world = data_dir / "worlds" / "边界世界"
        temporary_id = "12345678-1234-4234-8234-123456789abc"
        entity_dir = world / "entities" / "character" / "旅人"
        scenario_temporary = world / "scenarios" / f".开局.md.{temporary_id}.tmp"
        entity_temporary = entity_dir / f".ENTITY.md.{temporary_id}.tmp"
        interrupted = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os\n"
                    "from pathlib import Path\n"
                    "from rpera.content_store import ContentStore\n"
                    f"ContentStore._write_new(Path({str(scenario_temporary)!r}), '未发布'.encode('utf-8'))\n"
                    f"ContentStore._write_new(Path({str(entity_temporary)!r}), '未发布'.encode('utf-8'))\n"
                    "os._exit(9)\n"
                ),
            ],
            check=False,
        )
        assert interrupted.returncode == 9
        assert client.get("/api/content/worlds/边界世界").status_code == 200
        assert client.get("/api/content/worlds/边界世界/scenarios").json()[0]["opening"] == opening
        assert client.get("/api/content/worlds/边界世界/entities/旅人").json()["content"] == content
        assert client.get("/api/worlds").status_code == 200
        assert client.post("/api/saves", json={"world_name": "边界世界", "name": "残留仍可创建"}).status_code == 201

        invalid_temporary = world / "scenarios" / ".开局.md.not-a-uuid.tmp"
        invalid_temporary.write_text("不是内部临时文件", encoding="utf-8")
        assert client.get("/api/content/worlds/边界世界").status_code == 400


def test_content_rejects_broken_library_symlink_and_malformed_frontmatter(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "worlds").symlink_to(tmp_path / "missing", target_is_directory=True)
    (data_dir / "mods").symlink_to(tmp_path / "missing", target_is_directory=True)
    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.get("/api/content/worlds").status_code == 400
        assert client.get("/api/worlds").status_code == 400
        assert client.get("/api/content/mods").status_code == 400
        assert client.get("/api/mods").status_code == 400

    malformed = "---\ndescription: 简介\n---oops\n正文"
    try:
        parse_scenario_document(malformed)
    except ValueError as error:
        assert "frontmatter 未闭合" in str(error)
    else:
        raise AssertionError("非独占结束标记不应被接受")
    assert parse_scenario_document("  开始  \n") == ("", "开始")


def test_content_rejects_broken_nested_symlinks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    world = data_dir / "worlds" / "断链世界"
    world.mkdir(parents=True)
    missing = tmp_path / "missing"
    app = create_app(data_dir=data_dir)

    description = world / "description.md"
    description.symlink_to(missing)
    with TestClient(app) as client:
        assert client.get("/api/worlds").status_code == 400
        assert client.get("/api/content/worlds").status_code == 400
    description.unlink()

    scenarios = world / "scenarios"
    scenarios.symlink_to(missing, target_is_directory=True)
    with TestClient(app) as client:
        assert client.get("/api/worlds").status_code == 400
        assert client.get("/api/content/worlds").status_code == 400
    scenarios.unlink()

    entities = world / "entities"
    entities.symlink_to(missing, target_is_directory=True)
    with TestClient(app) as client:
        assert client.get("/api/content/worlds").status_code == 400
    try:
        EntityStore(world).entries()
    except ValueError as error:
        assert "符号链接" in str(error)
    else:
        raise AssertionError("断裂实体符号链接不应被视为不存在")


def test_editor_and_save_use_the_same_document_rules_and_snapshot_isolation(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        client.post("/api/content/worlds", json={"name": "隔离世界"})
        client.post("/api/content/worlds/隔离世界/scenarios", json={"name": "开局"})
        client.put(
            "/api/content/worlds/隔离世界/scenarios/开局",
            json={"description": "旧简介", "opening": "旧序章"},
        )
        client.post("/api/content/worlds/隔离世界/entities", json={"name": "旧角色", "type": "character"})
        save = client.post(
            "/api/saves",
            json={
                "world_name": "隔离世界",
                "name": "冻结存档",
                "scenario": {"source_kind": "world", "source_name": "隔离世界", "name": "开局"},
            },
        ).json()
        assert save["opening"] == "旧序章"

        client.put(
            "/api/content/worlds/隔离世界/scenarios/开局",
            json={"description": "新简介", "opening": "新序章"},
        )
        assert client.delete("/api/content/worlds/隔离世界/entities/旧角色").status_code == 200
        assert client.delete("/api/content/worlds/隔离世界").status_code == 200
        assert not (data_dir / "worlds" / "隔离世界").exists()

    snapshot = data_dir / "saves" / save["id"] / "current" / "world_snapshot"
    assert (snapshot / "scenarios" / "开局.md").is_file()
    assert (snapshot / "entities" / "character" / "旧角色" / "ENTITY.md").is_file()
    restarted = create_app(data_dir=data_dir, model_client=ScriptedModelClient())
    with TestClient(restarted) as client:
        assert client.get(f"/api/saves/{save['id']}").json()["opening"] == "旧序章"
        assert [entity["name"] for entity in client.get(f"/api/saves/{save['id']}/entities").json()] == ["旧角色"]
        client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续已有冒险。"})
        assert wait_for_turn(client, save["id"])["status"] == "completed"

def test_atomic_save_and_shared_snapshot_lock_preserve_complete_versions(tmp_path: Path, monkeypatch: Any) -> None:
    data_dir = tmp_path / "data"
    app = create_app(data_dir=data_dir)
    with TestClient(app, raise_server_exceptions=False) as client:
        client.post("/api/content/worlds", json={"name": "锁世界"})
        client.put("/api/content/worlds/锁世界", json={"description": "旧简介"})
        client.post("/api/content/worlds/锁世界/scenarios", json={"name": "开局"})
        client.put("/api/content/worlds/锁世界/scenarios/开局", json={"description": "", "opening": "旧序章"})

        real_replace = content_store_module.os.replace

        def fail_description_replace(source: Any, target: Any) -> None:
            if Path(target).name == "description.md" and Path(source).name.endswith(".tmp"):
                raise OSError("injected write failure")
            real_replace(source, target)

        monkeypatch.setattr(content_store_module.os, "replace", fail_description_replace)
        response = client.put("/api/content/worlds/锁世界", json={"description": "不会写入"})
        assert response.status_code == 500
        monkeypatch.setattr(content_store_module.os, "replace", real_replace)
        assert client.get("/api/content/worlds/锁世界").json()["description"] == "旧简介"

    copied = Event()
    release_copy = Event()
    edit_finished = Event()
    real_copytree = shutil.copytree
    world_path = data_dir / "worlds" / "锁世界"

    def blocked_copytree(source: Any, target: Any, *args: Any, **kwargs: Any) -> Path:
        result = real_copytree(source, target, *args, **kwargs)
        if Path(source) == world_path:
            copied.set()
            release_copy.wait(2)
        return result

    monkeypatch.setattr(storage_module.shutil, "copytree", blocked_copytree)

    def create_save() -> None:
        with TestClient(app) as client:
            response = client.post("/api/saves", json={"world_name": "锁世界", "name": "锁存档"})
            assert response.status_code == 201

    def edit_source() -> None:
        with TestClient(app) as client:
            response = client.put("/api/content/worlds/锁世界", json={"description": "新简介"})
            assert response.status_code == 200
        edit_finished.set()

    save_thread = Thread(target=create_save)
    save_thread.start()
    assert copied.wait(1)
    edit_thread = Thread(target=edit_source)
    edit_thread.start()
    assert not edit_finished.wait(0.05)
    release_copy.set()
    save_thread.join(2)
    edit_thread.join(2)
    assert edit_finished.is_set()

    save_dir = next(path for path in (data_dir / "saves").iterdir() if not path.name.startswith("."))
    assert (save_dir / "current" / "world_snapshot" / "description.md").read_text(encoding="utf-8") == "旧简介"
    assert world_path.joinpath("description.md").read_text(encoding="utf-8") == "新简介"
