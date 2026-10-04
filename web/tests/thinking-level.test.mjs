import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { JSDOM } from "jsdom";

function settingsWindow() {
    const html = readFileSync(new URL("../templates/index.html", import.meta.url), "utf8");
    const window = new JSDOM(html, { runScripts: "outside-only" }).window;
    const app = readFileSync(new URL("../static/app.js", import.meta.url), "utf8");
    const sections = [
        ["function fillPresetForm(", "function startNewPreset("],
        ["function presetPayload(", "function modelListPayload("],
        ["function resetModelOptions(", "async function savePreset("],
        ["function updateProviderFields(", "function showSettingsSection("],
        ['elements.presetForm.addEventListener("input"', '$("#new-preset-button").addEventListener'],
    ].map(([start, end]) => app.slice(app.indexOf(start), app.indexOf(end))).join("\n");
    window.eval(`
        const $ = (selector) => document.querySelector(selector);
        const state = {presets: {
            deepseek_thinking_levels: {
                "deepseek-flash": ["none", "low", "high", "max"],
                "deepseek-v4-pro": ["none", "low", "high", "max"],
            },
            gemini_thinking_levels: {
                "gemini-3.8-flash": ["low", "medium", "high"],
                "gemini-3.6-flash": ["minimal", "low", "medium", "high"],
            },
        }};
        const elements = {presetForm: $("#preset-form")};
        ${sections}
        window.settingsTest = {fillPresetForm, presetPayload};
    `);
    return window;
}

function selectPreset(window, provider, model, thinking_level) {
    window.settingsTest.fillPresetForm({
        name: "测试", provider, model, thinking_level, base_url: "https://provider.example",
        timeout_seconds: 120, temperature: 1, top_p: 1, max_tokens: 65536, has_api_key: false,
    });
}

test("saved DeepSeek levels are visible and submitted, including explicit none versus automatic", () => {
    const window = settingsWindow();
    const select = window.document.querySelector("#thinking-level");
    for (const level of [null, "none", "low", "high", "max"]) {
        selectPreset(window, "deepseek", "deepseek-flash", level);
        assert.equal(window.document.querySelector("#thinking-field").classList.contains("hidden"), false);
        assert.deepEqual(Array.from(select.options, (option) => option.value), ["", "none", "low", "high", "max"]);
        assert.equal(select.value, level ?? "");
        assert.equal(window.settingsTest.presetPayload().thinking_level, level);
    }
    assert.match(window.document.querySelector("#thinking-hint").textContent, /默认高/);
    window.close();
});

test("changing model resets unsupported effort and retains supported DeepSeek effort", () => {
    const window = settingsWindow();
    const model = window.document.querySelector("#model");
    selectPreset(window, "deepseek", "deepseek-flash", "max");
    model.value = "deepseek-v4-pro";
    model.dispatchEvent(new window.Event("input", { bubbles: true }));
    assert.equal(window.settingsTest.presetPayload().thinking_level, "max");
    model.value = "deepseek-chat";
    model.dispatchEvent(new window.Event("input", { bubbles: true }));
    assert.equal(window.settingsTest.presetPayload().thinking_level, null);
    assert.equal(window.document.querySelector("#thinking-field").classList.contains("hidden"), true);
    assert.match(window.document.querySelector("#settings-status").textContent, /已改为自动/);
    window.close();
});

test("provider switching uses DeepSeek defaults and Gemini keeps its model-specific options", () => {
    const window = settingsWindow();
    const select = window.document.querySelector("#thinking-level");
    const provider = window.document.querySelector("#provider");
    for (const level of ["low", "medium", "high"]) {
        selectPreset(window, "google_gemini", "models/gemini-3.8-flash", level);
        assert.deepEqual(Array.from(select.options, (option) => option.value), ["", "low", "medium", "high"]);
        provider.value = "deepseek";
        provider.dispatchEvent(new window.Event("change"));
        assert.equal(window.document.querySelector("#model").value, "deepseek-flash");
        assert.equal(window.settingsTest.presetPayload().thinking_level, null);
        assert.equal(window.document.querySelector("#thinking-field").classList.contains("hidden"), false);
    }
    selectPreset(window, "google_gemini", "gemini-3.6-flash", "minimal");
    assert.deepEqual(Array.from(select.options, (option) => option.value), ["", "minimal", "low", "medium", "high"]);
    assert.equal(window.settingsTest.presetPayload().thinking_level, "minimal");
    provider.value = "openai_compatible";
    provider.dispatchEvent(new window.Event("change"));
    assert.equal(window.document.querySelector("#thinking-field").classList.contains("hidden"), true);
    assert.equal(window.settingsTest.presetPayload().thinking_level, null);
    window.close();
});
