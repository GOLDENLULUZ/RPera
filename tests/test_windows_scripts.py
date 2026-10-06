from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from rpera.app import PROJECT_ROOT


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Requires native Windows cmd and PowerShell")


def git(root: Path, *arguments: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "RPera tests", "GIT_AUTHOR_EMAIL": "tests@example.invalid",
        "GIT_COMMITTER_NAME": "RPera tests", "GIT_COMMITTER_EMAIL": "tests@example.invalid",
    }
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], env=env,
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return result.stdout.strip()


@pytest.fixture
def update_checkout(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    remote = tmp_path / "remote.git"
    remote.mkdir()
    git(remote, "init", "--bare", "--initial-branch=main")
    source = tmp_path / "source"
    git(tmp_path, "clone", str(remote), str(source))
    shutil.copy(PROJECT_ROOT / "update.bat", source / "update.bat")
    shutil.copy(PROJECT_ROOT / ".gitignore", source / ".gitignore")
    shutil.copy(PROJECT_ROOT / ".gitattributes", source / ".gitattributes")
    (source / "scripts").mkdir()
    shutil.copy(PROJECT_ROOT / "scripts" / "update.ps1", source / "scripts" / "update.ps1")
    git(source, "add", ".")
    git(source, "commit", "-m", "initial")
    git(source, "push", "origin", "main")
    checkout = tmp_path / "游戏 副本"
    git(tmp_path, "clone", str(remote), str(checkout))
    for directory in ("config", "saves"):
        (checkout / directory).mkdir()
        (checkout / directory / "personal.txt").write_text("keep", encoding="utf-8")
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "uv.cmd").write_text(
        '@echo off\necho %*>>"%RPERA_TEST_UV_LOG%"\nexit /b %RPERA_TEST_UV_CODE%\n',
        encoding="ascii",
    )
    env = {
        **os.environ, "PATH": str(tools) + os.pathsep + os.environ["PATH"],
        "RPERA_TEST_UV_LOG": str(tmp_path / "uv.log"), "RPERA_TEST_UV_CODE": "0",
    }
    return source, checkout, env


def update(checkout: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["cmd.exe", "/d", "/c", str(checkout / "update.bat")],
        cwd=checkout.parent, env=env, input="\n", capture_output=True,
        text=True, encoding="utf-8", errors="replace", timeout=30,
    )


def publish(source: Path) -> None:
    (source / "release.txt").write_text("new release", encoding="utf-8")
    git(source, "add", ".")
    git(source, "commit", "-m", "release")
    git(source, "push", "origin", "main")


def test_update_fast_forwards_preserves_personal_files_and_survives_self_update(update_checkout) -> None:
    source, checkout, env = update_checkout
    (source / "update.bat").write_text("@echo off\necho UNEXPECTED_REENTRY\nexit /b 42\n", encoding="ascii")
    (source / "scripts" / "update.ps1").write_text("throw 'UNEXPECTED_REENTRY'\n", encoding="ascii")
    publish(source)
    result = update(checkout, env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Update complete" in result.stdout
    assert "UNEXPECTED_REENTRY" not in result.stdout
    assert git(checkout, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")
    assert Path(env["RPERA_TEST_UV_LOG"]).read_text().strip() == "sync --locked --no-dev"
    for directory in ("config", "saves"):
        assert (checkout / directory / "personal.txt").read_text() == "keep"


@pytest.mark.parametrize("state", ["modified", "staged", "untracked", "diverged", "detached", "no_upstream"])
def test_update_rejects_unsafe_git_states_without_rewriting_files(update_checkout, state: str) -> None:
    source, checkout, env = update_checkout
    if state in {"modified", "staged", "diverged"}:
        (checkout / ".gitignore").write_text("# local edit\n", encoding="utf-8")
        if state in {"staged", "diverged"}:
            git(checkout, "add", ".gitignore")
        if state == "diverged":
            git(checkout, "commit", "-m", "local change")
    elif state == "untracked":
        (checkout / "local.txt").write_text("keep", encoding="utf-8")
    elif state == "no_upstream":
        git(checkout, "branch", "--unset-upstream")
    else:
        git(checkout, "checkout", "--detach")
    head = git(checkout, "rev-parse", "HEAD")
    local_ignore = (checkout / ".gitignore").read_bytes()
    publish(source)
    result = update(checkout, env)
    assert result.returncode != 0
    assert git(checkout, "rev-parse", "HEAD") == head
    assert (checkout / ".gitignore").read_bytes() == local_ignore
    assert not (checkout / "release.txt").exists()
    assert not Path(env["RPERA_TEST_UV_LOG"]).exists()
    if state == "untracked":
        assert (checkout / "local.txt").read_text() == "keep"
    if state == "no_upstream":
        assert "This branch has no upstream" in result.stdout


def test_update_dependency_failure_reports_that_source_has_already_updated(update_checkout) -> None:
    source, checkout, env = update_checkout
    publish(source)
    result = update(checkout, {**env, "RPERA_TEST_UV_CODE": "7"})
    assert result.returncode != 0
    assert "Source update finished, but dependency setup failed" in result.stdout
    assert git(checkout, "rev-parse", "HEAD") == git(source, "rev-parse", "HEAD")


def test_update_refuses_incoming_tracked_file_that_collides_with_ignored_user_data(update_checkout) -> None:
    source, checkout, env = update_checkout
    head = git(checkout, "rev-parse", "HEAD")
    (source / "config").mkdir()
    (source / "config" / "personal.txt").write_text("incoming tracked file", encoding="utf-8")
    git(source, "add", "--force", "config/personal.txt")
    publish(source)
    result = update(checkout, env)
    assert result.returncode != 0
    assert git(checkout, "rev-parse", "HEAD") == head
    assert (checkout / "config" / "personal.txt").read_text() == "keep"
    assert not (checkout / "release.txt").exists()
    assert not Path(env["RPERA_TEST_UV_LOG"]).exists()


def test_start_explains_missing_uv_and_returns_failure(tmp_path: Path) -> None:
    root = tmp_path / "中文 空格"
    root.mkdir()
    shutil.copy(PROJECT_ROOT / "start.bat", root / "start.bat")
    result = subprocess.run(
        ["cmd.exe", "/d", "/c", str(root / "start.bat")], cwd=tmp_path,
        env={**os.environ, "PATH": str(Path(os.environ["SystemRoot"]) / "System32")},
        input="\n", capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
    )
    assert result.returncode != 0
    assert "uv was not found" in result.stdout
    assert "https://astral.sh/uv/install.ps1" in result.stdout


def test_update_explains_zip_downloads_without_creating_git_history(update_checkout, tmp_path: Path) -> None:
    _, _, env = update_checkout
    root = tmp_path / "ZIP 下载"
    root.mkdir()
    shutil.copy(PROJECT_ROOT / "update.bat", root / "update.bat")
    (root / "scripts").mkdir()
    shutil.copy(PROJECT_ROOT / "scripts" / "update.ps1", root / "scripts" / "update.ps1")
    result = update(root, env)
    assert result.returncode != 0
    assert "ZIP downloads must be updated manually" in result.stdout
    assert not (root / ".git").exists()
    assert not Path(env["RPERA_TEST_UV_LOG"]).exists()
