from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rpera.agents import AGENTS
from rpera.app import create_app
from rpera.prompt_store import PromptConfigurationError, PromptStore
from tests.helpers import make_data_dir


PROJECT_PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
PROJECT_MOCK_RESPONSES = Path(__file__).resolve().parents[1] / "mock_responses"
PROJECT_PREFILLS = Path(__file__).resolve().parents[1] / "prefills"


def copy_prompts(tmp_path: Path) -> Path:
    root = tmp_path / "prompts"
    shutil.copytree(PROJECT_PROMPTS, root)
    shutil.copytree(PROJECT_MOCK_RESPONSES, tmp_path / "mock_responses")
    shutil.copytree(PROJECT_PREFILLS, tmp_path / "prefills")
    return root


def test_prompt_store_loads_every_registered_agent(tmp_path: Path) -> None:
    snapshot = PromptStore(copy_prompts(tmp_path), AGENTS).load(
        frozenset(spec.name for spec in AGENTS.all())
    )

    assert set(snapshot.prompts) == {spec.name for spec in AGENTS.all()}
    assert set(snapshot.task_descriptions) == {spec.name for spec in AGENTS.children()}
    expected_capabilities = {
        name for spec in AGENTS.all() for name in spec.allowed_capabilities
    }
    assert set(snapshot.capability_descriptions) == expected_capabilities
    assert {
        path.stem for path in (PROJECT_PROMPTS / "capabilities").glob("*.md")
    } == expected_capabilities
    assert snapshot.fixed_responses == {}
    assert snapshot.prefills == {}


@pytest.mark.parametrize(("content", "message"), [
    ("没有占位符", "必须且只能包含一个"),
    ("{recommended_agent_sequence}\n{recommended_agent_sequence}", "必须且只能包含一个"),
])
def test_prompt_store_rejects_invalid_coordinator_prompt(tmp_path: Path, content: str, message: str) -> None:
    root = copy_prompts(tmp_path)
    (root / "coordinator.md").write_text(content, encoding="utf-8")

    with pytest.raises(PromptConfigurationError, match=message):
        PromptStore(root, AGENTS).load(frozenset(spec.name for spec in AGENTS.all()))


@pytest.mark.parametrize("agent", ["narrator", "consistency_checker"])
@pytest.mark.parametrize("content", ["未包含设置", "{save_narrative_settings}\n{save_narrative_settings}"])
def test_prompt_store_requires_one_save_settings_placeholder(tmp_path: Path, agent: str, content: str) -> None:
    root = copy_prompts(tmp_path)
    (root / f"{agent}.md").write_text(content, encoding="utf-8")

    with pytest.raises(PromptConfigurationError, match=rf"{agent} Prompt 必须且只能包含一个"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator", "consistency_checker"}))


def test_prompt_store_ignores_disabled_checker_placeholder(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (root / "consistency_checker.md").write_text("未包含设置", encoding="utf-8")

    assert "consistency_checker" not in PromptStore(root, AGENTS).load(
        frozenset({"coordinator", "narrator"})
    ).prompts


@pytest.mark.parametrize("content", ["没有上限占位符", "{max_tokens}\n{max_tokens}"])
def test_prompt_store_requires_one_token_limit_placeholder(tmp_path: Path, content: str) -> None:
    root = copy_prompts(tmp_path)
    (root / "capabilities" / "narrative_token_count.md").write_text(content, encoding="utf-8")

    with pytest.raises(PromptConfigurationError, match="叙事 Token 计数 Capability 描述必须且只能包含一个"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator"}))


def test_invalid_save_settings_placeholder_does_not_create_turn(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (root / "narrator.md").write_text("叙事提示词没有存档设置", encoding="utf-8")
    app = create_app(data_dir=make_data_dir(tmp_path), prompt_dir=root)

    with TestClient(app) as client:
        save = client.post("/api/saves", json={"world_name": "雾港", "name": "配置错误"}).json()
        response = client.post(f"/api/saves/{save['id']}/turns", json={"content": "继续。"})
        assert response.status_code == 500
        assert "{save_narrative_settings}" in response.json()["detail"]
        assert client.get(f"/api/saves/{save['id']}/turns").json() == []


def test_prompt_store_rejects_missing_empty_and_non_utf8_child_prompts(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    target = root / "role_player.md"
    target.unlink()
    with pytest.raises(PromptConfigurationError, match="文件不存在"):
        PromptStore(root, AGENTS).load(frozenset(spec.name for spec in AGENTS.all()))

    target.write_text("", encoding="utf-8")
    with pytest.raises(PromptConfigurationError, match="不能为空"):
        PromptStore(root, AGENTS).load(frozenset(spec.name for spec in AGENTS.all()))

    target.write_bytes(b"\xff")
    with pytest.raises(PromptConfigurationError, match="不是有效的 UTF-8"):
        PromptStore(root, AGENTS).load(frozenset(spec.name for spec in AGENTS.all()))


def test_prompt_store_only_loads_enabled_agents(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (root / "role_player.md").unlink()
    (root / "task_descriptions" / "role_player.md").unlink()
    (root / "capabilities" / "entity_search.md").unlink()

    snapshot = PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator"}))

    assert set(snapshot.prompts) == {"coordinator", "narrator"}
    assert set(snapshot.task_descriptions) == {"narrator"}
    assert set(snapshot.capability_descriptions) == {
        "file_read",
        "file_write",
        "file_edit",
        "narrative_publish",
        "narrative_token_count",
        "entity_read",
        "report_read",
    }
    assert snapshot.fixed_responses == {}


def test_prompt_store_rejects_missing_enabled_task_description(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (root / "task_descriptions" / "role_player.md").unlink()

    with pytest.raises(PromptConfigurationError, match="文件不存在"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "role_player", "narrator"}))


def test_prompt_store_rejects_oversized_enabled_task_description(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    target = root / "task_descriptions" / "role_player.md"
    target.write_bytes(b"x" * 4_001)

    with pytest.raises(PromptConfigurationError, match="不能超过 4000 字符"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "role_player", "narrator"}))


def test_prompt_store_rejects_missing_empty_and_non_utf8_capability_descriptions(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    target = root / "capabilities" / "file_read.md"
    target.unlink()
    with pytest.raises(PromptConfigurationError, match="Capability 描述文件不存在"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator"}))

    target.write_text("", encoding="utf-8")
    with pytest.raises(PromptConfigurationError, match="Capability 描述文件不能为空"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator"}))

    target.write_bytes(b"\xff")
    with pytest.raises(PromptConfigurationError, match="Capability 描述文件不是有效的 UTF-8"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator"}))


def test_prompt_store_rejects_oversized_capability_description(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (root / "capabilities" / "file_read.md").write_bytes(b"x" * 4_001)

    with pytest.raises(PromptConfigurationError, match="Capability 描述不能超过 4000 字符"):
        PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator"}))


def test_capability_description_snapshot_does_not_change_with_file(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    target = root / "capabilities" / "file_read.md"
    store = PromptStore(root, AGENTS)
    snapshot = store.load(frozenset({"coordinator", "narrator"}))

    target.write_text("后来修改", encoding="utf-8")

    assert snapshot.capability_descriptions["file_read"] != "后来修改"
    assert store.load(
        frozenset({"coordinator", "narrator"})
    ).capability_descriptions["file_read"] == "后来修改"


@pytest.mark.parametrize(("content", "message"), [
    (b"not-json", "格式无效"),
    ('{"approved": "true", "reason": "通过"}'.encode(), "格式无效"),
    ('{"approved": true, "reason": ""}'.encode(), "格式无效"),
])
def test_prompt_store_rejects_invalid_enabled_fixed_response(
    tmp_path: Path, content: bytes, message: str
) -> None:
    root = copy_prompts(tmp_path)
    (tmp_path / "mock_responses" / "compliance_reviewer.json").write_bytes(content)

    with pytest.raises(PromptConfigurationError, match=message):
        PromptStore(root, AGENTS).load(
            frozenset({"coordinator", "compliance_reviewer", "narrator"}),
            fixed_response_agents=frozenset({"compliance_reviewer"}),
        )


def test_prompt_store_ignores_disabled_fixed_response(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (tmp_path / "mock_responses" / "compliance_reviewer.json").unlink()

    snapshot = PromptStore(root, AGENTS).load(frozenset({"coordinator", "narrator"}))

    assert snapshot.fixed_responses == {}


def test_fixed_response_snapshot_does_not_change_with_file(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    store = PromptStore(root, AGENTS)
    snapshot = store.load(
        frozenset({"coordinator", "compliance_reviewer", "narrator"}),
        fixed_response_agents=frozenset({"compliance_reviewer"}),
    )

    (tmp_path / "mock_responses" / "compliance_reviewer.json").write_text(
        '{"approved": false, "reason": "后来修改"}',
        encoding="utf-8",
    )

    assert snapshot.fixed_responses["compliance_reviewer"].approved is True
    assert store.load(
        frozenset({"coordinator", "compliance_reviewer", "narrator"}),
        fixed_response_agents=frozenset({"compliance_reviewer"}),
    ).fixed_responses["compliance_reviewer"].approved is False


def test_prompt_store_loads_only_active_model_prefills(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (tmp_path / "prefills" / "coordinator.md").write_text("我将继续处理：", encoding="utf-8")
    (tmp_path / "prefills" / "compliance_reviewer.md").write_text("审核结果：", encoding="utf-8")

    snapshot = PromptStore(root, AGENTS).load(
        frozenset({"coordinator", "compliance_reviewer", "narrator"}),
        frozenset({"coordinator", "compliance_reviewer"}),
    )

    assert snapshot.prefills == {
        "coordinator": "我将继续处理：",
        "compliance_reviewer": "审核结果：",
    }


def test_prompt_store_ignores_prefill_for_active_fixed_response(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)

    snapshot = PromptStore(root, AGENTS).load(
        frozenset({"coordinator", "compliance_reviewer", "narrator"}),
        frozenset({"compliance_reviewer"}),
        frozenset({"compliance_reviewer"}),
    )

    assert snapshot.prefills == {}
    assert snapshot.fixed_responses["compliance_reviewer"].reason


def test_prompt_store_rejects_empty_active_prefill(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    (tmp_path / "prefills" / "coordinator.md").write_bytes(b"")

    with pytest.raises(PromptConfigurationError, match="不能为空"):
        PromptStore(root, AGENTS).load(
            frozenset({"coordinator", "narrator"}),
            frozenset({"coordinator"}),
        )


def test_prefill_snapshot_does_not_change_with_file(tmp_path: Path) -> None:
    root = copy_prompts(tmp_path)
    target = tmp_path / "prefills" / "narrator.md"
    target.write_text("原始预填", encoding="utf-8")
    store = PromptStore(root, AGENTS)

    snapshot = store.load(
        frozenset({"coordinator", "narrator"}),
        frozenset({"narrator"}),
    )
    target.write_text("后来修改", encoding="utf-8")

    assert snapshot.prefills["narrator"] == "原始预填"
    assert store.load(
        frozenset({"coordinator", "narrator"}),
        frozenset({"narrator"}),
    ).prefills["narrator"] == "后来修改"
