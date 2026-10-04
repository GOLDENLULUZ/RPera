import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { JSDOM } from "jsdom";

function workshop() {
    const html = readFileSync(new URL("../templates/index.html", import.meta.url), "utf8");
    const window = new JSDOM(html, { runScripts: "outside-only" }).window;
    const app = readFileSync(new URL("../static/app.js", import.meta.url), "utf8");
    const sections = [
        ["async function api(", "function rememberWorkspaceScroll("],
        ["function fixedCreationBase(", "async function showCreation("],
        ["async function loadCreationSources(", "function adjustSelectedSourceCounts("],
        ["function renderCreationSourceList(", "function clearCreationEditor("],
        ["async function importSource(", "function openCreateSourceDialog("],
    ].map(([start, end]) => app.slice(app.indexOf(start), app.indexOf(end))).join("\n");
    window.requestAnimationFrame = (callback) => callback();
    window.eval(`
        const $ = (selector) => document.querySelector(selector);
        const state = {creationKind: "worlds", creationBusy: false, creationSources: [], selectedCreationSource: null};
        const elements = {creation: $("#creation-view"), creationStatus: $("#creation-status"), creationSourceList: $("#creation-source-list")};
        function empty(text) {const div = document.createElement("div"); div.textContent = text; return div;}
        function keepItemVisible() {}
        function clearCreationEditor() {state.selectedCreationSource = null;}
        async function loadSelectedCreationSource() {state.selectedCreationSource = state.creationSources.find(source => source.name === state.selectedCreationSourceName);}
        ${sections}
        window.archiveTest = {state, importSource, exportSource, renderCreationSourceList};
    `);
    return window;
}

test("ZIP upload sends binary and selects the server's automatically numbered source", async () => {
    const window = workshop();
    const source = {name: "雾港(2)", path: "worlds/雾港(2)", scenario_count: 1, entity_count: 1};
    const file = new window.File(["ZIP bytes"], "arbitrary-name.zip", {type: "application/zip"});
    Object.defineProperty(window.document.querySelector("#import-source-file"), "files", {value: [file]});
    const calls = [];
    window.fetch = async (path, options) => {
        calls.push({path, options});
        return {ok: true, json: async () => options.method === "POST" ? source : [source]};
    };
    await window.archiveTest.importSource();
    assert.equal(calls[0].path, "/api/content/worlds/import");
    assert.equal(calls[0].options.headers["Content-Type"], "application/zip");
    assert.equal(calls[0].options.body, file);
    assert.equal(window.archiveTest.state.selectedCreationSource.name, "雾港(2)");
    assert.match(window.document.querySelector("#creation-status").textContent, /已导入：雾港\(2\)/);
    assert.equal(window.archiveTest.state.creationBusy, false);
    window.close();
});

test("invalid ZIP displays the server error and retains the current selection", async () => {
    const window = workshop();
    window.archiveTest.state.selectedCreationSource = {name: "原世界"};
    const file = new window.File(["bad ZIP"], "bad.zip");
    Object.defineProperty(window.document.querySelector("#import-source-file"), "files", {value: [file]});
    window.fetch = async () => ({ok: false, json: async () => ({detail: "压缩包必须只包含一个世界或模组"})});
    await window.archiveTest.importSource();
    assert.equal(window.archiveTest.state.selectedCreationSource.name, "原世界");
    assert.match(window.document.querySelector("#creation-status").textContent, /必须只包含一个/);
    assert.equal(window.document.querySelector("#import-source-button").disabled, false);
    window.close();
});

test("export downloads a ZIP using the saved source name without saving pending edits", async () => {
    const window = workshop();
    window.archiveTest.state.creationKind = "mods";
    window.archiveTest.state.selectedCreationSource = {name: "读心大师"};
    window.document.querySelector("#source-description").value = "尚未保存";
    let requested;
    let download;
    const blob = new window.Blob(["ZIP bytes"], {type: "application/zip"});
    window.fetch = async (path, options) => {
        requested = {path, options};
        return {ok: true, blob: async () => blob};
    };
    window.URL.createObjectURL = (received) => {assert.equal(received, blob); return "blob:zip";};
    window.URL.revokeObjectURL = () => {};
    window.HTMLAnchorElement.prototype.click = function () {download = this.download;};
    await window.archiveTest.exportSource();
    assert.equal(requested.path, `/api/content/mods/${encodeURIComponent("读心大师")}/export`);
    assert.equal(requested.options.body, undefined);
    assert.equal(download, "读心大师-mod.zip");
    assert.equal(window.document.querySelector("#source-description").value, "尚未保存");
    assert.match(window.document.querySelector("#creation-status").textContent, /不含未保存/);
    window.close();
});

test("ZIP import is available only in world and mod categories", () => {
    const window = workshop();
    for (const kind of ["worlds", "mods", "styles"]) {
        window.archiveTest.state.creationKind = kind;
        window.archiveTest.renderCreationSourceList();
        assert.equal(window.document.querySelector("#import-source-button").classList.contains("hidden"), kind === "styles");
    }
    window.close();
});
