from __future__ import annotations

import pytest

from rpera.content import markdown_frontmatter, parse_entity_document, parse_scenario_document, parse_style_document


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_frontmatter_understands_native_line_endings_without_rewriting_body(newline: str) -> None:
    body = '首行\r\n次行\n字面量 \\n 和 "引号"\n'
    document = newline.join(["---", "description: 简介", "---", "", ""]) + body
    assert markdown_frontmatter(document, "测试") == ({"description": "简介"}, body)
    assert parse_scenario_document(document) == ("简介", body)
    assert parse_entity_document(document).content == body
    style = document.replace("description: 简介", "description: 简介" + newline + "enabled: true")
    parsed_style = parse_style_document(style)
    assert parsed_style.enabled is True
    assert parsed_style.content == body


@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
def test_frontmatter_empty_body_and_unclosed_header(newline: str) -> None:
    assert markdown_frontmatter(newline.join(["---", "description: 简介", "---"]), "测试") == ({"description": "简介"}, "")
    with pytest.raises(ValueError, match="未闭合"):
        markdown_frontmatter(newline.join(["---", "description: 简介"]), "测试")
