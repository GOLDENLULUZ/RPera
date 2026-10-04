import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { JSDOM } from "jsdom";

function traceWindow() {
    const window = new JSDOM("<!doctype html><html><body></body></html>", { runScripts: "outside-only" }).window;
    window.eval(readFileSync(new URL("../static/trace-markdown.js", import.meta.url), "utf8"));
    const app = readFileSync(new URL("../static/app.js", import.meta.url), "utf8");
    const constants = app.slice(app.indexOf("const TRACE_MARKDOWN_FIELDS"), app.indexOf("const $ ="));
    const renderers = app.slice(app.indexOf("function traceSection("), app.indexOf("function handleEvent("));
    window.eval(`${constants}\n${renderers}\nwindow.traceTest = { traceSection, traceDataSection, renderToolResult };`);
    return window;
}

test("Markdown preserves real line breaks and escapes untrusted HTML", () => {
    const { traceSection, document } = traceWindow();
    const section = traceSection("内容", "# 标题\n第一行\n第二行\n\n- **重点**\n- `代码`\n\n<script>alert(1)</script>\n[危险](javascript:alert(1))");
    document.body.append(section);
    assert.equal(section.querySelector("h1")?.textContent, "标题");
    assert.equal(section.querySelector("p br")?.previousSibling?.textContent, "第一行");
    assert.equal(section.querySelector("strong")?.textContent, "重点");
    assert.equal(section.querySelector("code")?.textContent, "代码");
    assert.equal(section.querySelector("script"), null);
    assert.equal(section.querySelector('a[href^="javascript:"]'), null);
    assert.match(section.textContent, /<script>alert\(1\)<\/script>/);
    assert.equal(traceSection("字面量", "\\n").textContent.trimEnd(), "字面量\\n");
});

test("nested report tool replies expose Markdown content without JSON escaped newlines", () => {
    const { renderToolResult, document } = traceWindow();
    const section = renderToolResult("report_read", {
        report_documents: [{ name: "角色报告", id: "id-1", source_agent: "role_player", content: "第一段\n第二段\n\n- 建议" }],
        target_agent: "narrator",
    });
    document.body.append(section);
    assert.equal(section.querySelectorAll(".trace-markdown").length, 1);
    assert.equal(section.querySelector(".trace-markdown br")?.previousSibling?.textContent, "第一段");
    assert.equal(section.querySelector(".trace-markdown li")?.textContent, "建议");
    assert.match(section.textContent, /role_player/);
    assert.match(section.textContent, /narrator/);
    assert.doesNotMatch(section.querySelector(".trace-data")?.textContent ?? "", /第一段\\n第二段/);
});

test("tool arguments and entity documents render text while retaining metadata and full files", () => {
    const { traceDataSection, renderToolResult, document } = traceWindow();
    const input = traceDataSection("调用参数", { path: "narrative.md", content: "**章节**\n后续", report_refs: [{ id: "ref-1" }] });
    const output = renderToolResult("entity_read", {
        documents: [{ name: "旅店", path: "entities/location/旅店/ENTITY.md", content: "## 正文\n详情", document: "---\ndescription: 旅店\n---\n## 正文\n详情" }],
        target_agent: "narrator",
    });
    document.body.append(input, output);
    assert.equal(input.querySelector("strong")?.textContent, "章节");
    assert.match(input.querySelector(".trace-data")?.textContent ?? "", /narrative\.md/);
    assert.equal(output.querySelector("h2")?.textContent, "正文");
    assert.equal(output.querySelector("details")?.open, false);
    assert.match(output.querySelector("details")?.textContent ?? "", /description: 旅店/);
    assert.match(output.textContent, /entities\/location\/旅店\/ENTITY\.md/);
});
