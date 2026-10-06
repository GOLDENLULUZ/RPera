from __future__ import annotations

import stat
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Event
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest
from fastapi.testclient import TestClient

from rpera.app import create_app
from rpera.content_archive import ARCHIVE_CONTENT_LIMIT, ARCHIVE_ENTRY_LIMIT, ARCHIVE_UPLOAD_LIMIT


def package(entries: list[tuple[str | ZipInfo, bytes]]) -> bytes:
    output = BytesIO()
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in entries:
            if isinstance(name, str):
                member = ZipInfo(name)
                # ZipInfo normalizes Windows separators; preserve intentionally
                # malformed test paths so the importer receives the real input.
                member.filename = name
                member.compress_type = ZIP_DEFLATED
            else:
                member = name
            archive.writestr(member, data)
    return output.getvalue()


@pytest.mark.parametrize("plural", ["worlds", "mods"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_archive_round_trip_preserves_bytes_and_can_create_adventure(tmp_path: Path, plural: str, newline: str) -> None:
    data_dir = tmp_path / "data"
    prefix = f"{plural}/雾港"
    documents = {
        "description.md": "中文简介\r\n".encode(),
        "scenarios/初见.md": "---\ndescription: 雨夜\n---\n\n雨落下来。\n".replace("\n", newline).encode(),
        "entities/character/旅人/ENTITY.md": "---\naliases: [客人]\nrequired: true\n---\n\n详细设定\n".replace("\n", newline).encode(),
    }
    data = package([(f"{prefix}/{name}", body) for name, body in documents.items()])
    with TestClient(create_app(data_dir=data_dir)) as client:
        imported = client.post(f"/api/content/{plural}/import", content=data)
        assert imported.status_code == 201
        assert imported.json()["name"] == "雾港"
        assert imported.json()["scenario_count"] == 1
        assert imported.json()["entity_count"] == 1
        exported = client.get(f"/api/content/{plural}/雾港/export")
        assert exported.status_code == 200
        assert exported.headers["content-type"] == "application/zip"
        assert "filename*=UTF-8''" in exported.headers["content-disposition"]
        with ZipFile(BytesIO(exported.content)) as archive:
            for name, body in documents.items():
                assert archive.read(f"{prefix}/{name}") == body
        second = client.post(f"/api/content/{plural}/import", content=exported.content)
        assert second.status_code == 201
        assert second.json()["name"] == "雾港(1)"
        if plural == "mods":
            assert client.post("/api/content/worlds", json={"name": "基础世界"}).status_code == 201
        save = client.post("/api/saves", json={
            "name": "导入冒险",
            "world_name": "雾港(1)" if plural == "worlds" else "基础世界",
            "mod_names": [] if plural == "worlds" else ["雾港(1)"],
            "scenario": {"source_kind": "world" if plural == "worlds" else "mod", "source_name": "雾港(1)", "name": "初见"},
        })
        assert save.status_code == 201
        assert save.json()["opening"] == "雨落下来。\n"
        snapshot = data_dir / "saves" / save.json()["id"] / "current" / "world_snapshot"
        source_snapshot = snapshot if plural == "worlds" else snapshot / "mods" / "雾港(1)"
        for name, body in documents.items():
            assert (source_snapshot / name).read_bytes() == body
        assert client.delete(f"/api/content/{plural}/雾港(1)").status_code == 200
        assert (source_snapshot / "description.md").read_bytes() == documents["description.md"]
    with TestClient(create_app(data_dir=data_dir)) as client:
        assert client.get(f"/api/content/{plural}/雾港").status_code == 200
    assert list((data_dir / ".content-staging").iterdir()) == []


def test_empty_sources_numbering_and_concurrent_imports(tmp_path: Path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        client.post("/api/content/worlds", json={"name": "Alpha"})
        client.post("/api/content/worlds", json={"name": "alpha(2)"})
        client.post("/api/content/mods", json={"name": "Alpha"})
        exported = client.get("/api/content/worlds/Alpha/export")
        with ZipFile(BytesIO(exported.content)) as archive:
            assert archive.namelist() == ["worlds/Alpha/"]
        assert client.post("/api/content/worlds/import", content=exported.content).json()["name"] == "Alpha(1)"
        assert client.post("/api/content/worlds/import", content=exported.content).json()["name"] == "Alpha(3)"
        numbered = client.get("/api/content/worlds/Alpha(1)/export").content
        assert client.post("/api/content/worlds/import", content=numbered).json()["name"] == "Alpha(1)(1)"
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda _: client.post("/api/content/worlds/import", content=exported.content), range(3)))
        assert {response.json()["name"] for response in responses} == {"Alpha(4)", "Alpha(5)", "Alpha(6)"}
        assert all(response.status_code == 201 for response in responses)
        mod = client.get("/api/content/mods/Alpha/export").content
        assert client.post("/api/content/mods/import", content=mod).json()["name"] == "Alpha(1)"


@pytest.mark.parametrize("name", ["a" * 80, "界" * 80, "a" * 76 + " .界界"])
def test_numbering_keeps_names_within_cross_platform_limits(tmp_path: Path, name: str) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        assert client.post("/api/content/worlds", json={"name": name}).status_code == 201
        archive = client.get(f"/api/content/worlds/{name}/export").content
        response = client.post("/api/content/worlds/import", content=archive)
        assert response.status_code == 201
        imported = response.json()["name"]
        assert imported.endswith("(1)")
        assert len(imported) <= 80
        assert len(imported.encode()) <= 240
        assert (tmp_path / "worlds" / imported).is_dir()


@pytest.mark.parametrize("name", ["a" * 80, "界" * 80])
def test_long_legal_scenario_names_round_trip(tmp_path: Path, name: str) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        data = package([(f"worlds/长场景/scenarios/{name}.md", b"")])
        assert client.post("/api/content/worlds/import", content=data).status_code == 201
        archive = client.get("/api/content/worlds/长场景/export")
        assert archive.status_code == 200
        imported = client.post("/api/content/worlds/import", content=archive.content)
        assert imported.status_code == 201
        assert client.get("/api/content/worlds/长场景(1)/scenarios").json()[0]["name"] == name


def test_corrupt_deflate_member_is_an_input_error(tmp_path: Path) -> None:
    data = bytearray(package([("worlds/损坏/description.md", b"valid text")]))
    name_length, extra_length = struct.unpack_from("<HH", data, 26)
    offset = 30 + name_length + extra_length
    data[offset] = 0x07  # Invalid DEFLATE block type; central directory remains valid.
    with TestClient(create_app(data_dir=tmp_path), raise_server_exceptions=False) as client:
        assert client.post("/api/content/worlds/import", content=bytes(data)).status_code == 400
        assert client.get("/api/content/worlds").json() == []
    assert list((tmp_path / ".content-staging").iterdir()) == []


@pytest.mark.parametrize("entries", [
    [],
    [(".", b"")],
    [("./", b"")],
    [("worlds/安全/../description.md", b"bad")],
    [("/worlds/安全/description.md", b"bad")],
    [("worlds\\安全\\description.md", b"bad")],
    [("worlds/安全//description.md", b"bad")],
    [("worlds/安全/description.md", b"a"), ("worlds/其他/", b"")],
    [("mods/安全/", b"")],
    [("worlds/安全/description.md", b"a"), ("worlds/安全/description.md", b"b")],
    [("worlds/安全/description.md", b"a"), ("worlds/安全/Description.md", b"b")],
    [("worlds/Alpha/description.md", b"a"), ("worlds/alpha/scenarios/开局.md", b"b")],
    [("worlds/安全/run.py", b"print('bad')")],
    [("worlds/安全/entities/character/旅人/", b"")],
    [("worlds/安全/description.md", b"\xff")],
    [("worlds/安全/scenarios/开局.md", b"---\nunknown: true\n---\n")],
    [("worlds/安全/entities/character/旅人/ENTITY.md", b"---\nrequired: yesplease\n---\n")],
    [("worlds/安全/entities/character/旅人/ENTITY.md", b""), ("worlds/安全/entities/location/旅人/ENTITY.md", b"")],
    [("worlds/安全/scenarios/开局.md", b"x" * 20001)],
])
def test_invalid_packages_leave_library_unchanged(tmp_path: Path, entries: list[tuple[str, bytes]]) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        client.post("/api/content/worlds", json={"name": "原世界"})
        before = client.get("/api/content/worlds").json()
        assert client.post("/api/content/worlds/import", content=package(entries)).status_code == 400
        assert client.get("/api/content/worlds").json() == before
    assert list((tmp_path / ".content-staging").iterdir()) == []
    assert not (tmp_path / "description.md").exists()


def test_special_entries_corruption_and_size_limits(tmp_path: Path) -> None:
    symlink = ZipInfo("worlds/安全/description.md")
    symlink.create_system = 3
    symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
    with TestClient(create_app(data_dir=tmp_path)) as client:
        assert client.post("/api/content/worlds/import", content=package([(symlink, b"/etc/passwd")])).status_code == 400
        assert client.post("/api/content/worlds/import", content=b"not a ZIP").status_code == 400
        assert client.post("/api/content/worlds/import", content=b"x" * (ARCHIVE_UPLOAD_LIMIT + 1)).status_code == 413
        bomb = package([("worlds/安全/description.md", b"x" * (ARCHIVE_CONTENT_LIMIT + 1))])
        assert client.post("/api/content/worlds/import", content=bomb).status_code == 400
        many = package([(f"worlds/安全/scenarios/场景{index}.md", b"") for index in range(ARCHIVE_ENTRY_LIMIT + 1)])
        assert client.post("/api/content/worlds/import", content=many).status_code == 400
        assert client.get("/api/content/worlds").json() == []
    assert list((tmp_path / ".content-staging").iterdir()) == []


def test_export_preserves_empty_directories_and_skips_internal_temporary_files(tmp_path: Path) -> None:
    with TestClient(create_app(data_dir=tmp_path)) as client:
        client.post("/api/content/worlds", json={"name": "空目录"})
        source = tmp_path / "worlds" / "空目录"
        (source / "scenarios").mkdir()
        (source / "entities" / "character").mkdir(parents=True)
        (source / ".description.md.12345678-1234-4234-8234-123456789abc.tmp").write_text("未发布")
        archive = client.get("/api/content/worlds/空目录/export")
        assert archive.status_code == 200
        with ZipFile(BytesIO(archive.content)) as zipped:
            assert set(zipped.namelist()) == {
                "worlds/空目录/", "worlds/空目录/scenarios/", "worlds/空目录/entities/", "worlds/空目录/entities/character/",
            }
        assert client.post("/api/content/worlds/import", content=archive.content).status_code == 201
        assert (tmp_path / "worlds/空目录(1)/entities/character").is_dir()


def test_export_serializes_with_source_edits(tmp_path: Path, monkeypatch: Any) -> None:
    exporting, release, editing = Event(), Event(), Event()
    original_write = ZipFile.write

    def paused_write(archive: ZipFile, filename: Any, *args: Any, **kwargs: Any) -> None:
        exporting.set()
        assert release.wait(5)
        original_write(archive, filename, *args, **kwargs)

    with TestClient(create_app(data_dir=tmp_path)) as client:
        client.post("/api/content/worlds", json={"name": "一致世界"})
        client.put("/api/content/worlds/一致世界", json={"description": "旧简介"})
        monkeypatch.setattr(ZipFile, "write", paused_write)

        def edit():
            editing.set()
            return client.put("/api/content/worlds/一致世界", json={"description": "新简介"})

        with ThreadPoolExecutor(max_workers=2) as pool:
            download = pool.submit(client.get, "/api/content/worlds/一致世界/export")
            try:
                assert exporting.wait(5)
                updated = pool.submit(edit)
                assert editing.wait(5)
                assert not updated.done()
            finally:
                release.set()
            response = download.result(timeout=5)
            assert response.status_code == 200
            assert updated.result(timeout=5).status_code == 200
        with ZipFile(BytesIO(response.content)) as archive:
            assert archive.read("worlds/一致世界/description.md").decode() == "旧简介"
        assert (tmp_path / "worlds/一致世界/description.md").read_text() == "新简介"


@pytest.mark.parametrize("published", [False, True])
def test_process_exit_at_import_publication_leaves_only_complete_active_content(tmp_path: Path, published: bool) -> None:
    data_dir = tmp_path / "data"
    archive_file = tmp_path / "input.zip"
    archive_file.write_bytes(package([
        ("worlds/原世界/description.md", "完整简介".encode()),
        ("worlds/原世界/entities/character/旅人/ENTITY.md", "完整实体".encode()),
    ]))
    with TestClient(create_app(data_dir=data_dir)) as client:
        client.post("/api/content/worlds", json={"name": "原世界"})
    script = (
        "import os\nfrom pathlib import Path\nfrom rpera.content_store import ContentStore\n"
        f"store = ContentStore(Path({str(data_dir)!r}))\n"
        "rename = store._rename\n"
        "def stop(source, target):\n"
        + ("    rename(source, target)\n" if published else "")
        + "    os._exit(9)\n"
        "store._rename = stop\n"
        f"store.import_source('world', Path({str(archive_file)!r}).read_bytes())\n"
    )
    result = subprocess.run([sys.executable, "-c", script], check=False)
    assert result.returncode == 9
    with TestClient(create_app(data_dir=data_dir)) as client:
        names = {source["name"] for source in client.get("/api/content/worlds").json()}
        assert names == ({"原世界", "原世界(1)"} if published else {"原世界"})
        if published:
            assert (data_dir / "worlds/原世界(1)/description.md").read_text() == "完整简介"
            assert (data_dir / "worlds/原世界(1)/entities/character/旅人/ENTITY.md").read_text() == "完整实体"
        response = client.post("/api/content/worlds/import", content=archive_file.read_bytes())
        assert response.status_code == 201
        assert response.json()["name"] == ("原世界(2)" if published else "原世界(1)")
