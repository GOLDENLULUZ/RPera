from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.content import parse_entity_document


def _payload(entity: dict, **changes: object) -> dict:
    return {
        "description": entity["description"],
        "aliases": entity["aliases"],
        "visibility": entity["visibility"],
        "required": entity["required"],
        "content": entity["content"],
        "revision": entity["revision"],
        **changes,
    }


def test_save_entity_editor_isolated_crud_and_source_identity(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        client.post("/api/content/worlds", json={"name": "世界"})
        client.post("/api/content/mods", json={"name": "模组"})
        for kind, name in (("worlds", "世界"), ("mods", "模组")):
            client.post(f"/api/content/{kind}/{name}/entities", json={"name": "同名", "type": "character"})
        client.post("/api/content/mods/模组/entities", json={"name": "占用", "type": "item"})
        first = client.post("/api/saves", json={"world_name": "世界", "mod_names": ["模组"], "name": "一"}).json()
        second = client.post("/api/saves", json={"world_name": "世界", "mod_names": ["模组"], "name": "二"}).json()
        base = f"/api/saves/{first['id']}"
        world_path = "entities/character/同名/ENTITY.md"
        mod_path = "mods/模组/entities/character/同名/ENTITY.md"
        assert {entry["path"] for entry in client.get(f"{base}/entities").json()} == {
            world_path, mod_path, "mods/模组/entities/item/占用/ENTITY.md",
        }
        world = client.get(f"{base}/entity", params={"path": world_path}).json()
        mod = client.get(f"{base}/entity", params={"path": mod_path}).json()
        assert world["type"] == mod["type"] == "character"
        assert world["revision"] == mod["revision"]

        saved = client.put(
            f"{base}/entity", params={"path": mod_path},
            json=_payload(mod, description="模组角色", aliases=["别名", "别名"], required=True, content="模组正文"),
        )
        assert saved.status_code == 200
        assert saved.json()["aliases"] == ["别名"]
        assert saved.json()["required"] is True
        assert client.get(f"{base}/entity", params={"path": world_path}).json()["description"] == ""
        assert client.get(f"/api/saves/{second['id']}/entity", params={"path": mod_path}).json()["description"] == ""
        assert client.get("/api/content/mods/模组/entities/同名").json()["description"] == ""

        moved = client.post(
            f"{base}/entity/rename", params={"path": mod_path},
            json={"name": "改名", "type": "location", "revision": saved.json()["revision"]},
        )
        assert moved.status_code == 200
        moved_path = "mods/模组/entities/location/改名/ENTITY.md"
        assert moved.json()["path"] == moved_path
        assert moved.json()["content"] == "模组正文"
        assert client.get(f"{base}/entity", params={"path": mod_path}).status_code == 404

        created = client.post(f"{base}/entities", json={"name": "新建", "type": "event"})
        assert created.status_code == 201
        assert created.json()["path"] == "entities/event/新建/ENTITY.md"
        new_path = created.json()["path"]
        assert parse_entity_document((tmp_path / "saves" / first["id"] / "current" / "world_snapshot" / new_path).read_text(encoding="utf-8")).content == ""
        assert client.post(f"{base}/entities", json={"name": "同名", "type": "item"}).status_code == 409
        assert client.post(f"{base}/entity/rename", params={"path": moved_path}, json={"name": "占用", "type": "event", "revision": moved.json()["revision"]}).status_code == 409
        assert client.post(f"{base}/entity/rename", params={"path": moved_path}, json={"name": "新建", "type": "event", "revision": moved.json()["revision"]}).status_code == 200
        # The identical name is allowed because the existing entity belongs to the world, not the mod.
        renamed_mod_path = "mods/模组/entities/event/新建/ENTITY.md"
        assert client.get(f"{base}/entity", params={"path": renamed_mod_path}).status_code == 200
        assert client.request("DELETE", f"{base}/entity", params={"path": renamed_mod_path}, json={"revision": moved.json()["revision"]}).status_code == 200
        assert client.get(f"{base}/entity", params={"path": renamed_mod_path}).status_code == 404
        assert client.request("DELETE", f"{base}/entity", params={"path": new_path}, json={"revision": created.json()["revision"]}).status_code == 200
        assert client.get(f"/api/saves/{second['id']}/entity", params={"path": mod_path}).status_code == 200


def test_save_entity_editor_rejects_stale_invalid_and_running_writes(tmp_path: Path) -> None:
    app = create_app(data_dir=tmp_path)
    with TestClient(app) as client:
        client.post("/api/content/worlds", json={"name": "世界"})
        client.post("/api/content/worlds/世界/entities", json={"name": "角色", "type": "character"})
        client.put(
            "/api/content/worlds/世界/entities/角色",
            json={"description": "旧", "aliases": [], "visibility": "world_truth", "required": False, "content": '{"name": "角色"}'},
        )
        save = client.post("/api/saves", json={"world_name": "世界", "name": "存档"}).json()
        base = f"/api/saves/{save['id']}"
        path = "entities/character/角色/ENTITY.md"
        current = client.get(f"{base}/entity", params={"path": path}).json()
        assert current["description"] == "旧"
        assert client.put(f"{base}/entity", params={"path": path}, json=_payload(current, content="broken json")).status_code == 400
        assert client.get(f"{base}/entity", params={"path": path}).json()["revision"] == current["revision"]
        updated = client.put(f"{base}/entity", params={"path": path}, json=_payload(current, description="新"))
        assert updated.status_code == 200
        assert client.put(f"{base}/entity", params={"path": path}, json=_payload(current, description="过期")).status_code == 409
        assert client.post(f"{base}/entity/rename", params={"path": path}, json={"name": "改名", "type": "item", "revision": current["revision"]}).status_code == 409
        assert client.request("DELETE", f"{base}/entity", params={"path": path}, json={"revision": current["revision"]}).status_code == 409
        assert client.get(f"{base}/entity", params={"path": "../config"}).status_code == 400
        assert client.get(f"{base}/entity", params={"path": "entities/item/不存在/ENTITY.md"}).status_code == 404

        app.state.saves.create_turn(save["id"], "进行中", 4)
        assert client.post(f"{base}/entities", json={"name": "新建", "type": "item"}).status_code == 409
        assert client.put(f"{base}/entity", params={"path": path}, json=_payload(updated.json(), description="运行中")).status_code == 409
        assert client.post(f"{base}/entity/rename", params={"path": path}, json={"name": "改名", "type": "item", "revision": updated.json()["revision"]}).status_code == 409
        assert client.request("DELETE", f"{base}/entity", params={"path": path}, json={"revision": updated.json()["revision"]}).status_code == 409
        assert client.get(f"{base}/entity", params={"path": path}).json()["description"] == "新"
        assert client.get("/api/content/worlds/世界/entities/角色").json()["description"] == "旧"
