const state = {
    save: null,
    source: null,
    events: new Set(),
    liveTraceEvents: new Map(),
    eventPage: null,
    eventPageRequest: 0,
    latestEventId: 0,
    turnNumbers: new Map(),
    followLatestTurn: true,
    activeStoryKey: null,
    foregroundScrollY: 0,
    backendScrollY: 0,
    running: false,
    activeTurnId: null,
    resumableTurnId: null,
    aborting: false,
    presets: null,
    selectedPresetId: null,
    creatingPreset: false,
    presetDirty: false,
    runtimeSettings: null,
    networkSettings: null,
    agentPresetSettings: null,
    savePage: 1,
    deleteAction: null,
    renameSave: null,
    branchTurnId: null,
    newAdventure: null,
    settingsSection: "ai",
    drawingPresets: null,
    selectedDrawingPresetId: null,
    creatingDrawingPreset: false,
    drawingPresetDirty: false,
    capabilitySettings: null,
    capabilitySettingsDirty: false,
    creationKind: "worlds",
    creationSources: [],
    selectedCreationSourceName: null,
    selectedCreationSource: null,
    creationSection: "basic",
    selectedEntityName: null,
    creationLoaded: false,
    creationBusy: false,
    backendSection: "trace",
    selectedSaveEntity: null,
    saveEntityBusy: false,
    pendingImages: [],
    generatedImages: new Map(),
    generatedImageRequests: new Map(),
};

const SCENARIOS_PER_PAGE = 5;
const MODS_PER_PAGE = 10;
const TOOL_LABELS = {
    entity_search: "搜索实体",
    entity_read: "读取实体",
    research_report: "提交研究报告",
    character_create: "创建角色",
    character_edit: "编辑角色",
    character_rename: "角色更名",
    character_portrait_generate: "生成角色立绘",
    character_report: "提交角色报告",
    role_report: "提交角色报告",
    erotic_report: "提交意见报告",
    location_create: "创建地点",
    location_edit: "编辑地点",
    location_rename: "地点更名",
    location_report: "提交地点报告",
    style_read: "读取文风",
    style_report: "提交文风报告",
    report_read: "读取报告",
    compliance_report: "提交审核报告",
    story_summary_read: "读取故事摘要",
    story_summary_edit: "编辑故事摘要",
    story_history_read: "读取故事历史",
    goal_read: "读取目标",
    goal_create: "创建目标",
    goal_edit: "编辑目标",
    file_read: "读取草稿",
    file_write: "写入草稿",
    file_edit: "编辑草稿",
    narrative_publish: "发布故事",
};
const TRACE_MARKDOWN_FIELDS = new Set([
    "content", "report", "document", "narrative", "task", "reason", "message", "summary",
    "old_text", "new_text", "opening", "player_input", "text",
]);
const TRACE_STRUCTURED_FIELDS = new Set(["documents", "report_documents", "results", "goal", "error", "summary", "turns", "character", "location"]);
const TRACE_FIELD_LABELS = {
    content: "内容", report: "报告", document: "文档", narrative: "故事", task: "任务",
    reason: "原因", message: "消息", summary: "摘要", old_text: "原文", new_text: "新文本",
    opening: "开局", player_input: "玩家输入", report_documents: "报告列表", documents: "实体列表",
    results: "结果列表", goal: "目标", error: "错误", turns: "回合", character: "角色", location: "地点",
};

const $ = (selector) => document.querySelector(selector);

function keepItemVisible(container, item) {
    if (!container || !item || !container.clientHeight) return;
    const bounds = item.getBoundingClientRect();
    const viewport = container.getBoundingClientRect();
    if (bounds.top < viewport.top) container.scrollTop += bounds.top - viewport.top;
    else if (bounds.bottom > viewport.bottom) container.scrollTop += bounds.bottom - viewport.bottom;
}

function keepTabVisible(container, button) {
    if (!container || !button) return;
    const bounds = button.getBoundingClientRect();
    const viewport = container.getBoundingClientRect();
    if (bounds.left < viewport.left) container.scrollLeft += bounds.left - viewport.left;
    else if (bounds.right > viewport.right) container.scrollLeft += bounds.right - viewport.right;
}

function revealMobileEditor(editor) {
    if (!window.matchMedia("(max-width: 720px)").matches || !editor || editor.classList.contains("hidden")) return;
    requestAnimationFrame(() => {
        const header = document.querySelector(".topbar").getBoundingClientRect().height;
        window.scrollTo(0, window.scrollY + editor.getBoundingClientRect().top - header - 12);
    });
}

const elements = {
    launcher: $("#launcher"),
    workspace: $("#workspace"),
    foreground: $("#foreground-view"),
    backend: $("#backend-view"),
    saveEntityList: $("#save-entity-list"),
    saveEntityEditor: $("#save-entity-editor"),
    saveEntityStatus: $("#save-entity-status"),
    saveEntityIdentityForm: $("#save-entity-identity-form"),
    saveEntityContentForm: $("#save-entity-content-form"),
    createSaveEntityDialog: $("#create-save-entity-dialog"),
    createSaveEntityForm: $("#create-save-entity-form"),
    settings: $("#settings-view"),
    creation: $("#creation-view"),
    adventureButton: $("#adventure-button"),
    foregroundButton: $("#foreground-button"),
    backendButton: $("#backend-button"),
    settingsButton: $("#settings-button"),
    creationButton: $("#creation-button"),
    connection: $("#connection-status"),
    worldList: $("#world-list"),
    saveList: $("#save-list"),
    savePagination: $("#save-pagination"),
    previousSavePage: $("#previous-save-page"),
    nextSavePage: $("#next-save-page"),
    savePageStatus: $("#save-page-status"),
    story: $("#story"),
    turnNavigation: $("#turn-navigation"),
    turnNavigationToggle: $("#turn-navigation-toggle"),
    turnNavigationList: $("#turn-navigation-list"),
    turnNavigationLatest: $("#turn-navigation-latest"),
    eventStream: $("#event-stream"),
    eventCount: $("#event-count"),
    eventPagination: $("#event-pagination"),
    previousEventPage: $("#previous-event-page"),
    nextEventPage: $("#next-event-page"),
    eventPageStatus: $("#event-page-status"),
    saveLabel: $("#save-label"),
    turnForm: $("#turn-form"),
    playerInput: $("#player-input"),
    pendingImages: $("#pending-images"),
    addImageButton: $("#add-image-button"),
    imageInput: $("#image-input"),
    sendButton: $("#send-button"),
    turnStatus: $("#turn-status"),
    abortTurnButton: $("#abort-turn-button"),
    retryTurnButton: $("#retry-turn-button"),
    imagePreviewDialog: $("#image-preview-dialog"),
    imagePreview: $("#image-preview"),
    closeImagePreview: $("#close-image-preview"),
    presetList: $("#preset-list"),
    presetForm: $("#preset-form"),
    drawingPresetList: $("#drawing-preset-list"),
    drawingPresetForm: $("#drawing-preset-form"),
    runtimeSettingsForm: $("#runtime-settings-form"),
    networkSettingsForm: $("#network-settings-form"),
    capabilitySettingsForm: $("#capability-settings-form"),
    launcherStatus: $("#launcher-status"),
    renameSaveDialog: $("#rename-save-dialog"),
    renameSaveForm: $("#rename-save-form"),
    renameSaveName: $("#rename-save-name"),
    renameSaveError: $("#rename-save-error"),
    renameSaveSubmit: $("#rename-save-submit"),
    branchSaveDialog: $("#branch-save-dialog"),
    branchSaveForm: $("#branch-save-form"),
    branchSaveName: $("#branch-save-name"),
    branchSaveOrigin: $("#branch-save-origin"),
    branchSaveError: $("#branch-save-error"),
    branchSaveSubmit: $("#branch-save-submit"),
    deleteConfirmation: $("#delete-confirmation"),
    deleteConfirmationForm: $("#delete-confirmation-form"),
    deleteConfirmationMessage: $("#delete-confirmation-message"),
    deleteConfirmationError: $("#delete-confirmation-error"),
    confirmDeleteButton: $("#confirm-delete-button"),
    createSaveDialog: $("#create-save-dialog"),
    createSaveForm: $("#create-save-form"),
    createSaveWorld: $("#create-save-world"),
    createSaveName: $("#create-save-name"),
    createSaveRequirements: $("#create-save-requirements"),
    createSaveButton: $("#create-save-button"),
    createSaveError: $("#create-save-error"),
    modSelectionStep: $("#mod-selection-step"),
    scenarioSelectionStep: $("#scenario-selection-step"),
    modSelectionActions: $("#mod-selection-actions"),
    scenarioSelectionActions: $("#scenario-selection-actions"),
    modList: $("#mod-list"),
    modPagination: $("#mod-pagination"),
    previousModPage: $("#previous-mod-page"),
    nextModPage: $("#next-mod-page"),
    modPageStatus: $("#mod-page-status"),
    selectedMods: $("#selected-mods"),
    nextAdventureStep: $("#next-adventure-step"),
    previousAdventureStep: $("#previous-adventure-step"),
    scenarioList: $("#scenario-list"),
    scenarioPagination: $("#scenario-pagination"),
    previousScenarioPage: $("#previous-scenario-page"),
    nextScenarioPage: $("#next-scenario-page"),
    scenarioPageStatus: $("#scenario-page-status"),
    selectedScenario: $("#selected-scenario"),
    creationSourceList: $("#creation-source-list"),
    creationStatus: $("#creation-status"),
    sourceRenameForm: $("#source-rename-form"),
    sourceContentForm: $("#source-content-form"),
    styleRenameForm: $("#style-rename-form"),
    styleContentForm: $("#style-content-form"),
    toggleStyleEnabledButton: $("#toggle-style-enabled-button"),
    scenarioEditorList: $("#creation-scenario-list"),
    entityList: $("#creation-entity-list"),
    entityEditor: $("#entity-editor"),
    entityIdentityForm: $("#entity-identity-form"),
    entityContentForm: $("#entity-content-form"),
    createSourceDialog: $("#create-source-dialog"),
    createSourceForm: $("#create-source-form"),
    createEntityDialog: $("#create-entity-dialog"),
    createEntityForm: $("#create-entity-form"),
};

async function api(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (!(options.body instanceof FormData)) headers["Content-Type"] = "application/json";
    const response = await fetch(path, {
        ...options,
        headers,
    });
    if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const detail = Array.isArray(body.detail)
            ? body.detail.map((item) => `${Array.isArray(item.loc) ? item.loc.slice(1).join(".") : "请求"}：${item.msg || JSON.stringify(item)}`).join("；")
            : body.detail;
        throw new Error(detail || `HTTP ${response.status}`);
    }
    return response.json();
}

function rememberWorkspaceScroll() {
    if (elements.workspace.classList.contains("hidden") || !state.save) return;
    if (elements.backend.classList.contains("hidden")) state.foregroundScrollY = window.scrollY;
    else state.backendScrollY = window.scrollY;
}

function setView(view, openingSave = false) {
    const backend = view === "backend";
    if (!openingSave && !elements.workspace.classList.contains("hidden")) {
        if (backend && !elements.foreground.classList.contains("hidden")) state.foregroundScrollY = window.scrollY;
        if (!backend && !elements.backend.classList.contains("hidden")) state.backendScrollY = window.scrollY;
    }
    elements.launcher.classList.add("hidden");
    elements.settings.classList.add("hidden");
    elements.creation.classList.add("hidden");
    elements.workspace.classList.remove("hidden");
    elements.foreground.classList.toggle("hidden", backend);
    elements.backend.classList.toggle("hidden", !backend);
    requestAnimationFrame(() => {
        if (elements.backend.classList.contains("hidden") === backend) return;
        if (backend) window.scrollTo(0, state.backendScrollY);
        else if (state.followLatestTurn && state.turnNumbers.size) scrollToStoryEnd();
        else {
            window.scrollTo(0, state.foregroundScrollY);
            syncTurnNavigation();
        }
    });
    if (backend) renderBackendSection();
    elements.adventureButton.classList.remove("active");
    elements.settingsButton.classList.remove("active");
    elements.creationButton.classList.remove("active");
    elements.foregroundButton.classList.toggle("active", !backend);
    elements.backendButton.classList.toggle("active", backend);
    keepTabVisible(document.querySelector(".view-switch"), backend ? elements.backendButton : elements.foregroundButton);
}

function showAdventure() {
    elements.settings.classList.add("hidden");
    elements.creation.classList.add("hidden");
    elements.settingsButton.classList.remove("active");
    elements.creationButton.classList.remove("active");
    if (state.save) {
        setView("foreground");
        return;
    }
    elements.workspace.classList.add("hidden");
    elements.launcher.classList.remove("hidden");
    elements.adventureButton.classList.add("active");
    elements.foregroundButton.classList.remove("active");
    elements.backendButton.classList.remove("active");
    keepTabVisible(document.querySelector(".view-switch"), elements.adventureButton);
}

async function showSettings() {
    rememberWorkspaceScroll();
    elements.launcher.classList.add("hidden");
    elements.workspace.classList.add("hidden");
    elements.creation.classList.add("hidden");
    elements.settings.classList.remove("hidden");
    elements.adventureButton.classList.remove("active");
    elements.foregroundButton.classList.remove("active");
    elements.backendButton.classList.remove("active");
    elements.creationButton.classList.remove("active");
    elements.settingsButton.classList.add("active");
    keepTabVisible(document.querySelector(".view-switch"), elements.settingsButton);
    await loadPresets();
    await Promise.all([loadRuntimeSettings(), loadAgentPresetSettings(), loadDrawingPresets(), loadCapabilitySettings(), loadNetworkSettings()]);
    showSettingsSection(state.settingsSection);
}

function createRow(name, detail, meta, actionLabel, onAction, className, destructiveAction = null, renameAction = null) {
    const row = document.createElement("article");
    row.className = className;
    const title = document.createElement("div");
    title.className = className === "world-row" ? "world-name" : "save-name";
    title.textContent = name;
    const description = document.createElement("div");
    description.textContent = detail;
    const metadata = document.createElement("div");
    metadata.className = className === "world-row" ? "world-meta" : "save-meta";
    metadata.textContent = meta;
    description.append(document.createElement("br"), metadata);
    const actions = document.createElement("div");
    actions.className = "row-actions";
    if (renameAction) actions.append(createRowAction("重命名", renameAction));
    actions.append(createRowAction(actionLabel, onAction));
    if (destructiveAction) actions.append(createRowAction("删除", destructiveAction, "danger"));
    row.append(title, description, actions);
    return row;
}

function createRowAction(label, onAction, className = "") {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `row-action ${className}`.trim();
    button.textContent = `[ ${label} ]`;
    button.addEventListener("click", onAction);
    return button;
}

async function loadLauncher(page = state.savePage) {
    const [worlds, savePage] = await Promise.all([api("/api/worlds"), api(`/api/saves?page=${page}`)]);
    state.savePage = savePage.page;
    elements.worldList.replaceChildren();
    elements.saveList.replaceChildren();
    if (!worlds.length) elements.worldList.append(empty("未发现世界观"));
    for (const world of worlds) {
        elements.worldList.append(createRow(
            world.name,
            world.description,
            world.name,
            "创建冒险",
            () => openNewAdventure(world),
            "world-row",
        ));
    }
    if (!savePage.saves.length) elements.saveList.append(empty("还没有存档"));
    for (const save of savePage.saves) {
        elements.saveList.append(createRow(
            save.name,
            save.source_world_name,
            `最后运行 ${formatTime(save.last_played_at)}`,
            "继续",
            () => openSave(save.id),
            "save-row",
            () => deleteSave(save),
            () => openRenameSave(save),
        ));
    }
    const hasPagination = savePage.total_pages > 1;
    elements.savePagination.classList.toggle("hidden", !hasPagination);
    elements.savePageStatus.textContent = `第 ${savePage.page} / ${savePage.total_pages} 页，共 ${savePage.total} 个存档`;
    elements.previousSavePage.disabled = savePage.page === 1;
    elements.nextSavePage.disabled = savePage.page === savePage.total_pages;
}

function empty(text) {
    const node = document.createElement("div");
    node.className = "empty";
    node.textContent = text;
    return node;
}

async function openNewAdventure(world) {
    state.newAdventure = {
        world,
        step: "mods",
        selectedModNames: new Set(),
        selectedMods: new Map(),
        modPage: 1,
        modPageData: null,
        scenarios: [],
        scenarioPage: 1,
        scenarioRef: null,
        loading: false,
    };
    elements.createSaveWorld.textContent = world.name;
    elements.createSaveForm.reset();
    elements.createSaveName.value = `${world.name}冒险`;
    elements.createSaveError.textContent = "";
    elements.createSaveError.classList.add("hidden");
    renderAdventureStep();
    elements.createSaveDialog.showModal();
    await loadModPage();
}

function renderAdventureStep() {
    const adventure = state.newAdventure;
    if (!adventure) return;
    const scenarios = adventure.step === "scenarios";
    elements.modSelectionStep.classList.toggle("hidden", scenarios);
    elements.scenarioSelectionStep.classList.toggle("hidden", !scenarios);
    elements.modSelectionActions.classList.toggle("hidden", scenarios);
    elements.scenarioSelectionActions.classList.toggle("hidden", !scenarios);
}

async function loadModPage(page = state.newAdventure?.modPage || 1) {
    const adventure = state.newAdventure;
    if (!adventure) return;
    adventure.loading = true;
    renderModPage();
    try {
        const modPage = await api(`/api/mods?page=${page}`);
        if (state.newAdventure !== adventure) return;
        adventure.modPage = modPage.page;
        adventure.modPageData = modPage;
    } catch (error) {
        if (state.newAdventure !== adventure) return;
        elements.createSaveError.textContent = error.message;
        elements.createSaveError.classList.remove("hidden");
    } finally {
        if (state.newAdventure === adventure) {
            adventure.loading = false;
            renderModPage();
        }
    }
}

function renderModPage() {
    const adventure = state.newAdventure;
    if (!adventure) return;
    const modPage = adventure.modPageData;
    elements.modList.replaceChildren();
    if (adventure.loading) {
        elements.modList.append(empty("正在加载模组……"));
    } else if (!modPage?.mods.length) {
        elements.modList.append(empty("未发现可选模组"));
    } else {
        for (const mod of modPage.mods) {
            const option = document.createElement("label");
            option.className = "mod-option";
            const input = document.createElement("input");
            input.type = "checkbox";
            input.value = mod.name;
            input.checked = adventure.selectedModNames.has(mod.name);
            input.addEventListener("change", () => {
                if (input.checked) {
                    adventure.selectedModNames.add(mod.name);
                    adventure.selectedMods.set(mod.name, mod);
                } else {
                    adventure.selectedModNames.delete(mod.name);
                    adventure.selectedMods.delete(mod.name);
                }
                renderSelectedMods();
            });
            const content = document.createElement("span");
            const name = document.createElement("strong");
            name.textContent = mod.name;
            const description = document.createElement("small");
            description.textContent = mod.description;
            content.append(name, description);
            option.append(input, content);
            elements.modList.append(option);
        }
    }
    const hasPagination = (modPage?.total_pages || 0) > 1;
    elements.modPagination.classList.toggle("hidden", !hasPagination);
    elements.modPageStatus.textContent = modPage
        ? `第 ${modPage.page} / ${modPage.total_pages} 页，共 ${modPage.total} 个模组`
        : "";
    elements.previousModPage.disabled = !modPage || modPage.page === 1 || adventure.loading;
    elements.nextModPage.disabled = !modPage || modPage.page === modPage.total_pages || adventure.loading;
    elements.nextAdventureStep.disabled = adventure.loading;
    renderSelectedMods();
}

function renderSelectedMods() {
    const adventure = state.newAdventure;
    if (!adventure) return;
    const names = [...adventure.selectedMods.values()].map((mod) => mod.name);
    elements.selectedMods.textContent = names.length ? `已选择 ${names.length} 个模组：${names.join("、")}` : "未选择模组";
}

async function showScenarioStep() {
    const adventure = state.newAdventure;
    if (!adventure || adventure.loading) return;
    adventure.step = "scenarios";
    adventure.loading = true;
    adventure.scenarios = [];
    adventure.scenarioPage = 1;
    adventure.scenarioRef = null;
    renderAdventureStep();
    renderScenarioPage();
    const query = new URLSearchParams();
    for (const modName of [...adventure.selectedModNames].sort()) query.append("mod_name", modName);
    try {
        const scenarios = await api(`/api/worlds/${encodeURIComponent(adventure.world.name)}/scenarios?${query}`);
        if (state.newAdventure !== adventure || adventure.step !== "scenarios") return;
        adventure.scenarios = scenarios;
        adventure.scenarioRef = scenarios[0] ? scenarioRef(scenarios[0]) : null;
    } catch (error) {
        if (state.newAdventure !== adventure || adventure.step !== "scenarios") return;
        elements.createSaveError.textContent = error.message;
        elements.createSaveError.classList.remove("hidden");
    } finally {
        if (state.newAdventure === adventure && adventure.step === "scenarios") {
            adventure.loading = false;
            renderScenarioPage();
        }
    }
}

function showModStep() {
    const adventure = state.newAdventure;
    if (!adventure) return;
    adventure.step = "mods";
    adventure.loading = false;
    renderAdventureStep();
    renderModPage();
}

function renderScenarioPage() {
    const adventure = state.newAdventure;
    if (!adventure) return;
    const scenarios = adventure.scenarios;
    const totalPages = Math.max(Math.ceil(scenarios.length / SCENARIOS_PER_PAGE), 1);
    adventure.scenarioPage = Math.min(adventure.scenarioPage, totalPages);
    const start = (adventure.scenarioPage - 1) * SCENARIOS_PER_PAGE;
    const pageScenarios = scenarios.slice(start, start + SCENARIOS_PER_PAGE);
    elements.scenarioList.replaceChildren();
    if (adventure.loading) {
        elements.scenarioList.append(empty("正在加载开局……"));
    } else if (!pageScenarios.length) {
        elements.scenarioList.append(empty("此组合尚未定义开局"));
    }
    for (const scenario of pageScenarios) {
        const option = document.createElement("label");
        option.className = "scenario-option";
        const input = document.createElement("input");
        input.type = "radio";
        input.name = "scenario";
        input.value = scenarioKey(scenario);
        input.checked = scenarioKey(scenario) === scenarioKey(adventure.scenarioRef);
        input.addEventListener("change", () => {
            adventure.scenarioRef = scenarioRef(scenario);
            renderScenarioSelection();
        });
        const content = document.createElement("span");
        const name = document.createElement("strong");
        name.textContent = scenario.name;
        const description = document.createElement("small");
        description.textContent = scenario.description;
        content.append(name, description);
        if (scenario.source_kind === "mod") {
            const source = document.createElement("small");
            source.className = "scenario-source";
            source.textContent = `来自模组：${adventure.selectedMods.get(scenario.source_name)?.name || scenario.source_name}`;
            content.append(source);
        }
        option.append(input, content);
        elements.scenarioList.append(option);
    }
    const hasPagination = totalPages > 1;
    elements.scenarioPagination.classList.toggle("hidden", !hasPagination);
    elements.scenarioPageStatus.textContent = `第 ${adventure.scenarioPage} / ${totalPages} 页，共 ${scenarios.length} 个开局`;
    elements.previousScenarioPage.disabled = adventure.scenarioPage === 1 || adventure.loading;
    elements.nextScenarioPage.disabled = adventure.scenarioPage === totalPages || adventure.loading;
    renderScenarioSelection();
}

function renderScenarioSelection() {
    const adventure = state.newAdventure;
    if (!adventure) return;
    const selected = adventure.scenarios.find((scenario) => scenarioKey(scenario) === scenarioKey(adventure.scenarioRef));
    elements.selectedScenario.textContent = selected ? `已选择：${selected.name}` : "不使用开局";
}

function scenarioRef(scenario) {
    return {
        source_kind: scenario.source_kind,
        source_name: scenario.source_name,
        name: scenario.name,
    };
}

function scenarioKey(scenario) {
    return scenario ? JSON.stringify([scenario.source_kind, scenario.source_name, scenario.name]) : "";
}

async function createSave(event) {
    if (event.submitter?.value !== "create") return;
    event.preventDefault();
    const adventure = state.newAdventure;
    if (!adventure) return;
    const name = elements.createSaveName.value.trim();
    if (!name) return;
    elements.createSaveButton.disabled = true;
    elements.createSaveError.textContent = "";
    elements.createSaveError.classList.add("hidden");
    try {
        const save = await api("/api/saves", {
            method: "POST",
            body: JSON.stringify({
                world_name: adventure.world.name,
                name,
                scenario: adventure.scenarioRef,
                mod_names: [...adventure.selectedModNames].sort(),
                play_mode: elements.createSaveForm.querySelector('input[name="play_mode"]:checked').value,
                narrative_person: elements.createSaveForm.querySelector('input[name="narrative_person"]:checked').value,
                language: elements.createSaveForm.querySelector('input[name="language"]:checked').value,
                other_requirements: elements.createSaveRequirements.value.trim(),
            }),
        });
        elements.createSaveDialog.close();
        await openSave(save.id);
    } catch (error) {
        elements.createSaveError.textContent = error.message;
        elements.createSaveError.classList.remove("hidden");
    } finally {
        elements.createSaveButton.disabled = false;
    }
}

function confirmDeletion(message, action) {
    state.deleteAction = action;
    elements.deleteConfirmationMessage.textContent = message;
    elements.deleteConfirmationError.textContent = "";
    elements.deleteConfirmationError.classList.add("hidden");
    elements.confirmDeleteButton.disabled = false;
    elements.deleteConfirmation.showModal();
}

function deleteSave(save) {
    confirmDeletion(`确认删除存档“${save.name}”？`, async () => {
        await api(`/api/saves/${save.id}`, { method: "DELETE" });
        setLauncherStatus("存档已删除");
        await loadLauncher(state.savePage);
    });
}

function openRenameSave(save) {
    state.renameSave = save;
    elements.renameSaveName.value = save.name;
    elements.renameSaveError.textContent = "";
    elements.renameSaveError.classList.add("hidden");
    elements.renameSaveDialog.showModal();
    elements.renameSaveName.select();
}

async function renameSave(event) {
    if (event.submitter?.value !== "rename") return;
    event.preventDefault();
    if (!state.renameSave) return;
    const name = elements.renameSaveName.value.trim();
    if (!name) return;
    elements.renameSaveSubmit.disabled = true;
    try {
        await api(`/api/saves/${state.renameSave.id}/rename`, {
            method: "POST",
            body: JSON.stringify({ name }),
        });
        await loadLauncher(state.savePage);
        elements.renameSaveDialog.close();
        setLauncherStatus("存档已重命名");
    } catch (error) {
        elements.renameSaveError.textContent = error.message;
        elements.renameSaveError.classList.remove("hidden");
    } finally {
        elements.renameSaveSubmit.disabled = false;
    }
}

function openBranchSave(turnId) {
    if (!state.save) return;
    state.branchTurnId = turnId;
    elements.branchSaveOrigin.textContent = `从“${state.save.name}”第 ${state.turnNumbers.get(turnId)} 回合结束处分支`;
    elements.branchSaveName.value = `${state.save.name} · 分支 ${state.turnNumbers.get(turnId)}`.slice(0, 80);
    elements.branchSaveError.textContent = "";
    elements.branchSaveError.classList.add("hidden");
    elements.branchSaveDialog.showModal();
    elements.branchSaveName.select();
}

async function createBranchSave(event) {
    if (event.submitter?.value !== "branch") return;
    event.preventDefault();
    if (!state.save || !state.branchTurnId) return;
    elements.branchSaveSubmit.disabled = true;
    try {
        const result = await api(`/api/saves/${state.save.id}/turns/${state.branchTurnId}/branch`, {
            method: "POST",
            body: JSON.stringify({ name: elements.branchSaveName.value.trim() }),
        });
        elements.branchSaveDialog.close();
        await openSave(result.id);
    } catch (error) {
        elements.branchSaveError.textContent = error.message;
        elements.branchSaveError.classList.remove("hidden");
    } finally {
        elements.branchSaveSubmit.disabled = false;
    }
}

function setLauncherStatus(message, error = false) {
    elements.launcherStatus.textContent = message;
    elements.launcherStatus.classList.toggle("error-text", error);
}

async function openSave(saveId) {
    if (state.source) state.source.close();
    state.save = await api(`/api/saves/${saveId}`);
    state.events.clear();
    state.liveTraceEvents.clear();
    state.eventPage = null;
    state.latestEventId = 0;
    state.turnNumbers = new Map();
    state.generatedImages = new Map();
    state.generatedImageRequests = new Map();
    state.followLatestTurn = true;
    state.activeStoryKey = null;
    state.foregroundScrollY = 0;
    state.backendScrollY = 0;
    elements.turnNavigationList.replaceChildren();
    elements.turnNavigation.classList.add("hidden");
    closeTurnNavigation();
    state.selectedSaveEntity = null;
    elements.saveEntityList.replaceChildren();
    clearSaveEntityEditor();
    setSaveEntityStatus("");
    elements.eventStream.replaceChildren();
    elements.story.replaceChildren();
    clearPendingImages();
    elements.eventCount.textContent = "0 条记录";
    elements.saveLabel.textContent = `${state.save.name} / ${state.save.source_world_name}`
        + (state.save.branch_source_save_name
            ? ` · 从“${state.save.branch_source_save_name}”第 ${state.save.branch_source_turn_number} 回合分支`
            : "");
    elements.launcher.classList.add("hidden");
    elements.foregroundButton.classList.remove("hidden");
    elements.backendButton.classList.remove("hidden");
    setView("foreground", true);
    state.followLatestTurn = false;
    if (state.save.opening) appendStory("序章", state.save.opening, "opening", "scenario");
    const turns = await api(`/api/saves/${saveId}/turns`);
    state.turnNumbers = new Map(turns.map((turn, index) => [turn.id, index + 1]));
    for (const turn of turns) renderTurn(turn);
    state.followLatestTurn = true;
    if (turns.length) scrollToStoryEnd();
    else window.scrollTo(0, 0);
    syncTurnNavigation();
    const runningTurn = turns.find((turn) => turn.status === "running");
    state.running = Boolean(runningTurn);
    state.activeTurnId = runningTurn?.id ?? null;
    state.aborting = false;
    const latestTurn = turns.at(-1);
    state.resumableTurnId = ["failed", "interrupted"].includes(latestTurn?.status) ? latestTurn.id : null;
    elements.turnStatus.textContent = "等待输入";
    elements.turnStatus.classList.remove("error-text");
    if (latestTurn?.status === "failed") {
        elements.turnStatus.textContent = "上一回合执行失败，可以从断点继续";
        elements.turnStatus.classList.add("error-text");
    } else if (latestTurn?.status === "interrupted") {
        elements.turnStatus.textContent = "上一回合已中止，可以从断点继续";
        elements.turnStatus.classList.remove("error-text");
    } else if (state.running) {
        elements.turnStatus.textContent = "Runtime 正在执行……";
        elements.turnStatus.classList.remove("error-text");
    }
    updateTurnControls();
    updateSaveEntityControls();
    await loadEventPage();
    connectEvents(saveId, state.latestEventId);
}

function renderTurn(turn) {
    appendStory("玩家", turn.player_input, "player", turn.id, turn.images || []);
    if (turn.narrative) {
        appendStory("故事", turn.narrative, "narrator", turn.id);
        addBranchAction(turn.id);
    }
    renderGeneratedImages(turn.id, turn.generated_images || []);
}

function renderGeneratedImages(turnId, images) {
    state.generatedImages.set(turnId, images);
    const key = `${turnId}-generated`;
    const existing = elements.story.querySelector(`[data-key="${CSS.escape(key)}"]`);
    const narrative = elements.story.querySelector(`[data-key="${CSS.escape(`${turnId}-narrator`)}"]`);
    const target = narrative?.querySelector(".story-content");
    existing?.remove();
    target?.querySelector(".turn-images")?.remove();
    if (!images.length) return;
    const gallery = createImageGallery(images);
    if (target) {
        target.append(gallery);
    } else {
        const entry = document.createElement("article");
        entry.className = "story-entry generated";
        entry.dataset.key = key;
        const label = document.createElement("div");
        label.className = "story-role";
        label.textContent = "生成图片";
        const content = document.createElement("div");
        content.className = "story-content";
        content.append(gallery);
        entry.append(label, content);
        const player = elements.story.querySelector(`[data-key="${CSS.escape(`${turnId}-player`)}"]`);
        player?.after(entry);
    }
}

async function refreshGeneratedImages(turnId) {
    if (!state.save || !turnId) return;
    const saveId = state.save.id;
    const request = (state.generatedImageRequests.get(turnId) || 0) + 1;
    state.generatedImageRequests.set(turnId, request);
    try {
        const turn = await api(`/api/saves/${saveId}/turns/${turnId}`);
        if (state.save?.id !== saveId || state.generatedImageRequests.get(turnId) !== request) return;
        renderGeneratedImages(turnId, turn.generated_images || []);
    } catch (error) {
        console.error("无法更新生成图片", error);
    }
}

function addBranchAction(turnId) {
    const entry = elements.story.querySelector(`[data-key="${CSS.escape(`${turnId}-narrator`)}"]`);
    const label = entry?.querySelector(".story-role");
    if (!label || label.querySelector("button")) return;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "text-button branch-action";
    button.textContent = "[ 从此处分支 ]";
    button.addEventListener("click", () => openBranchSave(turnId));
    label.append(button);
}

function appendStory(role, content, kind, turnId, images = []) {
    const key = `${turnId}-${kind}`;
    if (elements.story.querySelector(`[data-key="${CSS.escape(key)}"]`)) return;
    const entry = document.createElement("article");
    entry.className = `story-entry ${kind}`;
    entry.dataset.key = key;
    const label = document.createElement("div");
    label.className = "story-role";
    label.textContent = role;
    const text = document.createElement("div");
    text.className = "story-content";
    text.textContent = content;
    if (images.length) text.append(createImageGallery(images));
    entry.append(label, text);
    const narrative = kind === "player" ? elements.story.querySelector(`[data-key="${CSS.escape(`${turnId}-narrator`)}"]`) : null;
    if (narrative) narrative.before(entry);
    else elements.story.append(entry);
    if (kind === "opening" || kind === "player") addTurnNavigationEntry(entry, kind, turnId);
    if (kind === "narrator" && state.followLatestTurn && !elements.foreground.classList.contains("hidden")) scrollToStoryEnd();
}

function addTurnNavigationEntry(entry, kind, turnId) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.storyKey = entry.dataset.key;
    button.storyEntry = entry;
    button.textContent = kind === "opening" ? "序章" : String(state.turnNumbers.get(turnId)).padStart(2, "0");
    button.setAttribute("aria-label", kind === "opening" ? "跳转到序章" : `跳转到第 ${state.turnNumbers.get(turnId)} 回合`);
    button.addEventListener("click", () => {
        state.followLatestTurn = false;
        closeTurnNavigation();
        const offset = document.querySelector(".topbar").getBoundingClientRect().height + 14;
        window.scrollTo(0, window.scrollY + entry.getBoundingClientRect().top - offset);
        syncTurnNavigation();
    });
    elements.turnNavigationList.append(button);
    elements.turnNavigation.classList.remove("hidden");
    updateTurnNavigationToggle();
}

function scrollToStoryEnd() {
    window.scrollTo(0, document.documentElement.scrollHeight - window.innerHeight);
    syncTurnNavigation();
}

function updateTurnNavigationToggle() {
    const total = state.turnNumbers.size;
    const active = elements.turnNavigationList.querySelector("button.active");
    const current = active?.textContent || String(total).padStart(2, "0");
    elements.turnNavigationToggle.textContent = window.matchMedia("(max-width: 720px)").matches
        ? `${current}/${total}`
        : `回合 ${current} / ${total}`;
    elements.turnNavigationToggle.setAttribute("aria-label", `当前${current === "序章" ? "序章" : `第 ${Number(current)} 回合`}，共 ${total} 回合，打开回合导航`);
}

function syncTurnNavigation() {
    if (elements.foreground.classList.contains("hidden") || !state.save) return;
    const buttons = [...elements.turnNavigationList.children];
    if (!buttons.length) return;
    const top = document.querySelector(".topbar").getBoundingClientRect().height + 18;
    let active = buttons[0];
    for (const button of buttons) {
        if (button.storyEntry.getBoundingClientRect().top > top) break;
        active = button;
    }
    if (window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 4) active = buttons.at(-1);
    if (state.activeStoryKey !== active.dataset.storyKey) {
        for (const button of buttons) {
            button.classList.toggle("active", button === active);
            if (button === active) button.setAttribute("aria-current", "location");
            else button.removeAttribute("aria-current");
        }
        state.activeStoryKey = active.dataset.storyKey;
        const list = elements.turnNavigationList;
        const buttonBounds = active.getBoundingClientRect();
        const listBounds = list.getBoundingClientRect();
        if (buttonBounds.top < listBounds.top) list.scrollTop += buttonBounds.top - listBounds.top;
        else if (buttonBounds.bottom > listBounds.bottom) list.scrollTop += buttonBounds.bottom - listBounds.bottom;
        updateTurnNavigationToggle();
    }
}

function closeTurnNavigation() {
    elements.turnNavigation.classList.remove("open");
    elements.turnNavigationToggle.setAttribute("aria-expanded", "false");
}

function createImageGallery(images) {
    const gallery = document.createElement("div");
    gallery.className = "turn-images";
    for (const image of images) {
        const button = document.createElement("button");
        button.className = "turn-image-button";
        button.type = "button";
        button.title = image.original_name || (image.agent ? `${agentLabel(image.agent)}生成图片` : "查看图片");
        const preview = document.createElement("img");
        preview.src = image.content_url;
        preview.alt = image.original_name || (image.agent ? `${agentLabel(image.agent)}生成图片` : "玩家图片");
        button.append(preview);
        button.addEventListener("click", () => openImagePreview(image.content_url, preview.alt));
        gallery.append(button);
    }
    return gallery;
}

function openImagePreview(url, alt) {
    elements.imagePreview.src = url;
    elements.imagePreview.alt = alt;
    elements.imagePreviewDialog.showModal();
}

function addPendingImages(files) {
    for (const file of files) {
        state.pendingImages.push({ file, url: URL.createObjectURL(file) });
    }
    renderPendingImages();
}

function removePendingImage(index) {
    const [removed] = state.pendingImages.splice(index, 1);
    if (removed) URL.revokeObjectURL(removed.url);
    renderPendingImages();
}

function clearPendingImages() {
    for (const image of state.pendingImages) URL.revokeObjectURL(image.url);
    state.pendingImages = [];
    elements.imageInput.value = "";
    renderPendingImages();
}

function renderPendingImages() {
    elements.pendingImages.replaceChildren();
    state.pendingImages.forEach((image, index) => {
        const item = document.createElement("div");
        item.className = "pending-image";
        const preview = document.createElement("img");
        preview.src = image.url;
        preview.alt = image.file.name;
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "pending-image-remove";
        remove.setAttribute("aria-label", `移除 ${image.file.name}`);
        remove.textContent = "×";
        remove.addEventListener("click", () => removePendingImage(index));
        item.append(preview, remove);
        elements.pendingImages.append(item);
    });
    elements.pendingImages.classList.toggle("hidden", !state.pendingImages.length);
}

function connectEvents(saveId, after) {
    elements.connection.textContent = "连接中";
    elements.connection.className = "connection";
    const source = new EventSource(`/api/saves/${saveId}/events?after=${after}`);
    state.source = source;
    source.addEventListener("open", () => {
        elements.connection.textContent = "已连接";
        elements.connection.className = "connection live";
    });
    source.addEventListener("error", () => {
        elements.connection.textContent = "重连中";
        elements.connection.className = "connection error";
    });
    source.addEventListener("runtime", (message) => {
        const event = JSON.parse(message.data);
        if (state.events.has(event.id)) return;
        state.events.add(event.id);
        state.latestEventId = Math.max(state.latestEventId, event.id);
        const traceEvent = projectLiveTraceEvent(event);
        if (traceEvent) {
            state.liveTraceEvents.set(traceEvent.id, traceEvent);
            appendLiveTraceEvent(traceEvent);
        }
        handleEvent(event);
    });
}

async function loadEventPage(page) {
    if (!state.save) return;
    const saveId = state.save.id;
    const request = ++state.eventPageRequest;
    const query = page ? `?page=${page}` : "";
    const eventPage = await api(`/api/saves/${saveId}/trace/history${query}`);
    if (request !== state.eventPageRequest || state.save?.id !== saveId) return;
    state.eventPage = eventPage;
    for (const event of eventPage.events) state.events.add(event.id);
    state.latestEventId = Math.max(state.latestEventId, eventPage.cursor);
    elements.eventStream.replaceChildren(...eventPage.events.map(renderEvent));
    updateEventPagination();
    const newerEvents = [...state.liveTraceEvents.values()]
        .filter((event) => event.id > eventPage.cursor)
        .sort((left, right) => left.id - right.id);
    for (const id of state.liveTraceEvents.keys()) {
        if (id <= eventPage.cursor) state.liveTraceEvents.delete(id);
    }
    for (const event of newerEvents) appendLiveTraceEvent(event);
    return eventPage;
}

async function changeEventPage(page) {
    const loadedPage = await loadEventPage(page);
    if (!loadedPage || elements.backend.classList.contains("hidden") || state.backendSection !== "trace") return;
    const offset = document.querySelector(".topbar").getBoundingClientRect().height + 12;
    window.scrollTo(0, window.scrollY + elements.eventStream.getBoundingClientRect().top - offset);
}

function appendLiveTraceEvent(event) {
    if (!state.eventPage) return;
    const position = state.eventPage.total + 1;
    const targetPage = Math.ceil(position / state.eventPage.page_size);
    state.eventPage.total = position;
    state.eventPage.total_pages = Math.ceil(position / state.eventPage.page_size);
    if (targetPage === state.eventPage.page) {
        state.eventPage.events.push(event);
        elements.eventStream.append(renderEvent(event));
    }
    updateEventPagination();
}

function updateEventPagination() {
    const eventPage = state.eventPage;
    if (!eventPage) return;
    elements.eventCount.textContent = `${eventPage.total} 条记录`;
    elements.eventPagination.classList.toggle("hidden", eventPage.total_pages <= 1);
    elements.previousEventPage.disabled = eventPage.page <= 1;
    elements.nextEventPage.disabled = eventPage.page >= eventPage.total_pages;
    elements.eventPageStatus.textContent = eventPage.total
        ? `第 ${eventPage.page} / ${eventPage.total_pages} 页，共 ${eventPage.total} 条`
        : "暂无事件";
}

function projectLiveTraceEvent(event) {
    const payload = event.payload;
    let projected;
    if (["tool.started", "tool.completed", "tool.failed"].includes(event.type)) {
        if (payload.tool === "task" && event.type !== "tool.failed") return null;
        projected = tracePayload(payload, ["agent", "tool", "input", "result"]);
    } else if (event.type === "model.response") {
        if (payload.tool_calls?.length || typeof payload.content !== "string" || !payload.content.trim()) return null;
        projected = tracePayload(payload, ["agent", "content", "duration_ms"]);
    } else if (event.type === "model.fallback") {
        projected = tracePayload(payload, ["agent", "reason", "error", "primary_model", "fallback_model"]);
    } else if (event.type === "agent.fixed_response") {
        projected = tracePayload(payload, ["agent", "result"]);
    } else if (event.type === "task.created") {
        projected = tracePayload(payload, ["initiator_agent", "target_agent", "agent", "task", "reason", "instruction_blocked", "related_entities", "reports", "styles"]);
    } else if (["turn.started", "turn.resumed", "turn.completed", "turn.failed", "turn.interrupted"].includes(event.type)) {
        projected = tracePayload(payload, ["player_input", "duration_ms", "message", "error_type"]);
    } else {
        return null;
    }
    return { ...event, payload: projected };
}

function tracePayload(payload, keys) {
    return Object.fromEntries(keys.filter((key) => key in payload).map((key) => [key, payload[key]]));
}

function renderEvent(event) {
    const node = document.createElement("article");
    node.className = `event ${event.type.replaceAll(".", "-")}`;
    const head = document.createElement("div");
    head.className = "event-head";
    const actor = document.createElement("span");
    actor.className = "event-agent";
    const action = document.createElement("span");
    action.className = "event-action";
    const time = document.createElement("span");
    time.className = "event-time";
    time.textContent = formatTime(event.created_at);
    const payload = document.createElement("div");
    payload.className = "event-payload";
    const turn = traceTurnNumber(event.turn_id);

    if (event.type.startsWith("turn.")) {
        node.classList.add("turn-event", "open");
        actor.textContent = turn ? `第 ${turn} 回合` : "回合";
        action.textContent = turnAction(event.type);
        if (event.type === "turn.started" && event.payload.player_input) {
            payload.append(traceSection("玩家", event.payload.player_input));
        } else if (event.type === "turn.resumed") {
            payload.append(traceSection("状态", "从断点继续执行"));
        } else if (event.type === "turn.completed" && event.payload.duration_ms != null) {
            payload.append(traceSection("耗时", `${event.payload.duration_ms} ms`));
        } else if (event.payload.message) {
            payload.append(traceSection("原因", event.payload.message));
        }
    } else if (event.type === "task.created") {
        actor.textContent = agentLabel(event.payload.initiator_agent || "coordinator");
        action.textContent = `委派 → ${agentLabel(event.payload.target_agent || event.payload.agent)}`;
        payload.append(traceSection("任务", event.payload.task || ""));
        if (event.payload.reason) payload.append(traceSection("原因", event.payload.reason));
        payload.append(traceSection("指令已屏蔽", event.payload.instruction_blocked ? "是" : "否"));
        appendReferenceList(payload, "传递实体", event.payload.related_entities, (item) => item.name ? `${item.name} · ${item.path}` : item.path);
        appendReferenceList(payload, "传递报告", event.payload.reports, (item) => item.source_agent ? `${item.name} · 来自 ${agentLabel(item.source_agent)}` : item.name);
        appendReferenceList(payload, "传递文风", event.payload.styles, (item) => item.name ? `${item.name} · ${item.path}` : item.path);
    } else if (event.type === "model.fallback") {
        actor.textContent = agentLabel(event.payload.agent);
        action.textContent = "启用备用 AI";
        payload.append(traceSection("触发原因", event.payload.reason === "content_blocked" ? "内容拦截" : "接口暂时不可用"));
        payload.append(traceSection("首选 AI", event.payload.primary_model?.preset_name || ""));
        payload.append(traceSection("备用 AI", event.payload.fallback_model?.preset_name || ""));
        payload.append(traceSection("原始错误", event.payload.error || ""));
    } else if (event.type.startsWith("tool.")) {
        const tool = event.payload.tool || "tool";
        actor.textContent = agentLabel(event.payload.agent);
        if (tool === "task" && event.type === "tool.failed") {
            const input = parseTraceInput(event.payload.input);
            action.textContent = input?.agent ? `委派失败 → ${agentLabel(input.agent)}` : "委派失败";
        } else {
            action.textContent = event.type === "tool.started"
                ? `开始调用 · ${toolLabel(tool)}`
                : `${event.type === "tool.failed" ? "调用失败" : "工具回复"} · ${toolLabel(tool)}`;
        }
        payload.append(event.type === "tool.started"
            ? traceDataSection("调用参数", parseTraceInput(event.payload.input))
            : renderToolResult(tool, event.payload.result));
    } else {
        actor.textContent = agentLabel(event.payload.agent);
        action.textContent = event.type === "agent.fixed_response" ? "固定回复" : "最终回复";
        payload.append(traceDataSection("回复", event.payload.content ?? event.payload.result));
    }

    head.append(actor, action, time);
    if (!node.classList.contains("turn-event")) head.addEventListener("click", () => node.classList.toggle("open"));
    node.append(head, payload);
    return node;
}

function traceTurnNumber(turnId) {
    if (!turnId) return null;
    if (!state.turnNumbers.has(turnId)) state.turnNumbers.set(turnId, state.turnNumbers.size + 1);
    return state.turnNumbers.get(turnId);
}

function turnAction(type) {
    return {
        "turn.started": "新回合",
        "turn.resumed": "继续回合",
        "turn.completed": "回合完成",
        "turn.failed": "回合失败",
        "turn.interrupted": "回合中止",
    }[type] || type;
}

function agentLabel(name) {
    const agent = state.agentPresetSettings?.agents.find((item) => item.name === name);
    return agent?.label || name || "Runtime";
}

function toolLabel(name) {
    return TOOL_LABELS[name] || name;
}

function parseTraceInput(input) {
    if (typeof input !== "string") return input;
    try { return JSON.parse(input); } catch { return input; }
}

function traceSection(label, text, markdown = true) {
    const section = document.createElement("section");
    section.className = "trace-section";
    const title = document.createElement("div");
    title.className = "trace-section-title";
    title.textContent = label;
    const content = document.createElement("div");
    content.className = markdown ? "trace-text trace-markdown" : "trace-text";
    if (markdown) content.innerHTML = RPeraTraceMarkdown.render(String(text ?? ""));
    else content.textContent = text;
    section.append(title, content);
    return section;
}

function traceDataSection(label, value) {
    if (typeof value === "string") return traceSection(label, value);
    const section = traceSection(label, "", false);
    const fields = document.createElement("div");
    fields.className = "trace-fields";
    if (Array.isArray(value) && value.length && value.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
        value.forEach((item, index) => {
            const name = item.name || item.path || item.turn_number || `第 ${index + 1} 项`;
            fields.append(traceDataSection(String(name), item));
        });
    } else if (value && typeof value === "object" && !Array.isArray(value)) {
        const metadata = {};
        for (const [key, item] of Object.entries(value)) {
            if (typeof item === "string" && (TRACE_MARKDOWN_FIELDS.has(key) || item.includes("\n"))) {
                if (key === "document" && typeof value.content === "string") {
                    const details = document.createElement("details");
                    details.className = "trace-detail";
                    const summary = document.createElement("summary");
                    summary.textContent = "完整文件原文";
                    const original = document.createElement("pre");
                    original.className = "trace-data";
                    original.textContent = item;
                    details.append(summary, original);
                    fields.append(details);
                } else {
                    fields.append(traceSection(TRACE_FIELD_LABELS[key] || key, item));
                }
            } else if (item && typeof item === "object" && TRACE_STRUCTURED_FIELDS.has(key)) {
                fields.append(traceDataSection(TRACE_FIELD_LABELS[key] || key, item));
            } else {
                metadata[key] = item;
            }
        }
        if (Object.keys(metadata).length) fields.append(traceJson(metadata));
    } else {
        fields.append(traceJson(value));
    }
    section.lastElementChild.replaceWith(fields);
    return section;
}

function traceJson(value) {
    const content = document.createElement("pre");
    content.className = "trace-data";
    content.textContent = JSON.stringify(value ?? null, null, 2);
    return content;
}

function appendReferenceList(container, label, items, describe) {
    if (!Array.isArray(items) || !items.length) return;
    const section = document.createElement("section");
    section.className = "trace-section";
    const title = document.createElement("div");
    title.className = "trace-section-title";
    title.textContent = label;
    const list = document.createElement("ul");
    list.className = "trace-reference-list";
    for (const item of items) {
        const entry = document.createElement("li");
        entry.textContent = describe(item);
        list.append(entry);
    }
    section.append(title, list);
    container.append(section);
}

function renderToolResult(tool, result) {
    const container = document.createElement("div");
    container.className = "trace-result";
    container.append(traceDataSection("结果", result));
    return container;
}

function handleEvent(event) {
    if (event.type === "tool.completed" && event.payload.tool === "character_portrait_generate") {
        void refreshGeneratedImages(event.turn_id);
    }
    if (event.type === "turn.started") {
        state.running = true;
        state.activeTurnId = event.turn_id;
        state.resumableTurnId = null;
        state.aborting = false;
        elements.turnStatus.textContent = "Runtime 正在执行……";
    }
    if (event.type === "turn.resumed") {
        state.running = true;
        state.activeTurnId = event.turn_id;
        state.resumableTurnId = null;
        state.aborting = false;
        elements.turnStatus.textContent = "正在原会话中继续……";
    }
    if (event.type === "narrative.delta") {
        appendStory("故事", event.payload.content, "narrator", event.turn_id);
        renderGeneratedImages(event.turn_id, state.generatedImages.get(event.turn_id) || []);
    }
    if (event.type === "turn.completed") {
        void refreshGeneratedImages(event.turn_id);
        addBranchAction(event.turn_id);
        state.running = false;
        state.activeTurnId = null;
        state.resumableTurnId = null;
        state.aborting = false;
        elements.turnStatus.textContent = `完成 / ${event.payload.duration_ms} ms`;
    }
    if (event.type === "turn.failed" || event.type === "turn.interrupted") {
        void refreshGeneratedImages(event.turn_id);
        state.running = false;
        state.activeTurnId = null;
        state.resumableTurnId = event.turn_id;
        state.aborting = false;
        if (event.type === "turn.interrupted") {
            elements.turnStatus.textContent = "回合已中止，可以从断点继续";
            elements.turnStatus.classList.remove("error-text");
        } else {
            elements.turnStatus.textContent = `失败：${shortErrorMessage(event.payload.message)}`;
            elements.turnStatus.classList.add("error-text");
        }
    }
    updateTurnControls();
}

function shortErrorMessage(message) {
    const text = String(message || "未知错误").replace(/\s+/g, " ").trim();
    return text.length > 240 ? `${text.slice(0, 240)}…（详情见后台运行记录）` : text;
}

function updateTurnControls() {
    elements.playerInput.disabled = state.running;
    elements.addImageButton.disabled = state.running;
    elements.imageInput.disabled = state.running;
    elements.sendButton.disabled = state.running;
    elements.abortTurnButton.classList.toggle("hidden", !state.running || !state.activeTurnId);
    elements.abortTurnButton.disabled = state.aborting;
    elements.retryTurnButton.classList.toggle("hidden", !state.resumableTurnId || state.running);
    elements.retryTurnButton.disabled = state.running;
    if (!state.resumableTurnId) elements.turnStatus.classList.remove("error-text");
    updateSaveEntityControls();
}

async function retryResumableTurn() {
    if (!state.save || !state.resumableTurnId || state.running) return;
    const turnId = state.resumableTurnId;
    state.running = true;
    state.activeTurnId = turnId;
    updateTurnControls();
    elements.turnStatus.textContent = "正在从断点继续……";
    try {
        const result = await api(`/api/saves/${state.save.id}/turns/${turnId}/retry`, {
            method: "POST",
        });
        state.activeTurnId = result.turn_id;
        state.resumableTurnId = null;
        updateTurnControls();
    } catch (error) {
        state.running = false;
        state.activeTurnId = null;
        elements.turnStatus.textContent = shortErrorMessage(error.message);
        elements.turnStatus.classList.add("error-text");
        updateTurnControls();
    }
}

async function submitTurn(event) {
    event.preventDefault();
    const content = elements.playerInput.value.trim();
    if ((!content && !state.pendingImages.length) || !state.save || state.running) return;
    state.running = true;
    updateTurnControls();
    elements.turnStatus.textContent = "提交行动……";
    try {
        const body = new FormData();
        body.append("content", content);
        for (const image of state.pendingImages) body.append("images", image.file, image.file.name);
        const result = await api(`/api/saves/${state.save.id}/turns`, {
            method: "POST",
            body,
        });
        if (!state.turnNumbers.has(result.turn_id)) state.turnNumbers.set(result.turn_id, state.turnNumbers.size + 1);
        appendStory("玩家", content, "player", result.turn_id, result.images || []);
        if (state.followLatestTurn) scrollToStoryEnd();
        state.activeTurnId = result.turn_id;
        elements.playerInput.value = "";
        clearPendingImages();
        updateTurnControls();
    } catch (error) {
        state.running = false;
        updateTurnControls();
        elements.turnStatus.textContent = shortErrorMessage(error.message);
        elements.turnStatus.classList.add("error-text");
    }
}

async function abortRunningTurn() {
    if (!state.save || !state.activeTurnId || !state.running || state.aborting) return;
    const turnId = state.activeTurnId;
    state.aborting = true;
    elements.turnStatus.textContent = "正在中止……";
    updateTurnControls();
    try {
        await api(`/api/saves/${state.save.id}/turns/${turnId}/abort`, { method: "POST" });
        state.running = false;
        state.activeTurnId = null;
        state.resumableTurnId = turnId;
        state.aborting = false;
        elements.turnStatus.textContent = "回合已中止，可以从断点继续";
        elements.turnStatus.classList.remove("error-text");
        updateTurnControls();
    } catch (error) {
        if (!state.running || state.activeTurnId !== turnId) return;
        state.aborting = false;
        elements.turnStatus.textContent = shortErrorMessage(error.message);
        updateTurnControls();
        elements.turnStatus.classList.add("error-text");
    }
}

async function loadPresets(preferredId = state.selectedPresetId) {
    state.presets = await api("/api/ai-presets");
    renderAiFallbackSettings();
    const available = state.presets.presets.some((preset) => preset.id === preferredId);
    state.selectedPresetId = available ? preferredId : state.presets.main_preset_id;
    state.creatingPreset = false;
    renderPresetList();
    selectPreset(state.selectedPresetId);
    if (state.agentPresetSettings) renderAgentPresetFields();
}

function renderAiFallbackSettings() {
    const select = $("#ai-fallback-preset");
    select.replaceChildren(new Option("请选择 Preset", ""));
    for (const preset of state.presets.presets) {
        select.append(new Option(`${preset.name} / ${preset.model}`, preset.id));
    }
    select.value = state.presets.fallback_preset_id || "";
    $("#ai-fallback-enabled").checked = state.presets.fallback_enabled;
}

async function saveAiFallbackSettings(event) {
    event.preventDefault();
    const status = $("#ai-fallback-status");
    status.textContent = "保存中……";
    status.classList.remove("error-text");
    try {
        const enabled = $("#ai-fallback-enabled").checked;
        const presetId = $("#ai-fallback-preset").value || null;
        if (enabled && !presetId) throw new Error("启用备用 AI 时请选择 Preset");
        const updated = await api("/api/ai-fallback-settings", {
            method: "PUT",
            body: JSON.stringify({ enabled, preset_id: presetId }),
        });
        state.presets.fallback_enabled = updated.enabled;
        state.presets.fallback_preset_id = updated.preset_id;
        status.textContent = "备用 AI 已保存";
    } catch (error) {
        status.textContent = error.message;
        status.classList.add("error-text");
    }
}

async function loadRuntimeSettings() {
    state.runtimeSettings = await api("/api/runtime-settings");
    $("#max-delegations").value = state.runtimeSettings.max_delegations;
    $("#context-turns").value = state.runtimeSettings.context_turns;
    if (state.agentPresetSettings) renderAgentPresetFields();
    setRuntimeSettingsStatus("");
}

async function loadNetworkSettings() {
    state.networkSettings = await api("/api/network-settings");
    $("#network-mode").value = state.networkSettings.mode;
    $("#proxy-url").value = state.networkSettings.proxy_url;
    $("#proxy-username").value = state.networkSettings.proxy_username;
    $("#proxy-password").value = "";
    $("#clear-proxy-password").checked = false;
    $("#proxy-password-hint").textContent = state.networkSettings.has_proxy_password
        ? "已保存密码；留空将保持不变。"
        : "未保存密码。";
    updateNetworkFields();
    setNetworkSettingsStatus("");
}

function updateNetworkFields() {
    const custom = $("#network-mode").value === "custom";
    for (const field of ["#proxy-url", "#proxy-username", "#proxy-password", "#clear-proxy-password"]) {
        $(field).disabled = !custom;
    }
    $("#proxy-url").required = custom;
}

async function saveNetworkSettings(event) {
    event.preventDefault();
    setNetworkSettingsStatus("保存中……");
    try {
        state.networkSettings = await api("/api/network-settings", {
            method: "PUT",
            body: JSON.stringify({
                mode: $("#network-mode").value,
                proxy_url: $("#proxy-url").value,
                proxy_username: $("#proxy-username").value,
                proxy_password: $("#proxy-password").value,
                clear_proxy_password: $("#clear-proxy-password").checked,
            }),
        });
        await loadNetworkSettings();
        setNetworkSettingsStatus("网络设置已保存");
    } catch (error) {
        setNetworkSettingsStatus(error.message, true);
    }
}

function setNetworkSettingsStatus(message, error = false) {
    const status = $("#network-settings-status");
    status.textContent = message;
    status.classList.toggle("error-text", error);
}

async function loadAgentPresetSettings() {
    state.agentPresetSettings = await api("/api/agent-preset-settings");
    renderAgentPresetFields();
}

function renderAgentPresetFields() {
    const fields = $("#agent-preset-fields");
    fields.replaceChildren();
    if (!state.agentPresetSettings || !state.presets || !state.runtimeSettings) return;
    for (const agent of state.agentPresetSettings.agents) {
        const label = document.createElement("label");
        label.textContent = agent.label;
        const select = document.createElement("select");
        select.dataset.agent = agent.name;
        if (!agent.is_main) select.append(new Option("跟随主代理", ""));
        for (const preset of state.presets.presets) {
            select.append(new Option(`${preset.name} / ${preset.model}`, preset.id));
        }
        select.value = agent.is_main
            ? state.agentPresetSettings.main_preset_id
            : state.agentPresetSettings.agent_preset_overrides[agent.name] || "";
        const streaming = document.createElement("label");
        streaming.className = "checkbox-line";
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.dataset.agentStreaming = agent.name;
        checkbox.checked = state.agentPresetSettings.agent_streaming[agent.name] || false;
        const text = document.createElement("span");
        text.textContent = "流式模式";
        streaming.append(checkbox, text);
        const prefill = document.createElement("label");
        prefill.className = "checkbox-line";
        const prefillCheckbox = document.createElement("input");
        prefillCheckbox.type = "checkbox";
        prefillCheckbox.dataset.agentPrefill = agent.name;
        prefillCheckbox.checked = state.runtimeSettings.prefill_agents.includes(agent.name);
        const prefillText = document.createElement("span");
        prefillText.textContent = agent.supports_fixed_response
            ? "尾部续写（固定响应模式下不生效）"
            : "尾部续写";
        prefill.append(prefillCheckbox, prefillText);
        const fixedResponse = document.createElement("label");
        fixedResponse.className = "checkbox-line";
        if (agent.supports_fixed_response) {
            const fixedResponseCheckbox = document.createElement("input");
            fixedResponseCheckbox.type = "checkbox";
            fixedResponseCheckbox.dataset.complianceFixedResponse = "true";
            fixedResponseCheckbox.checked = state.runtimeSettings.use_compliance_fixed_response;
            const fixedResponseText = document.createElement("span");
            fixedResponseText.textContent = "使用固定响应";
            fixedResponse.append(fixedResponseCheckbox, fixedResponseText);
        }
        const instructionBlocking = document.createElement("label");
        instructionBlocking.className = "checkbox-line";
        if (!agent.is_main) {
            const instructionBlockingCheckbox = document.createElement("input");
            instructionBlockingCheckbox.type = "checkbox";
            instructionBlockingCheckbox.dataset.agentInstructionBlocked = agent.name;
            instructionBlockingCheckbox.checked = state.runtimeSettings.blocked_instruction_agents.includes(agent.name);
            const instructionBlockingText = document.createElement("span");
            instructionBlockingText.textContent = "屏蔽主代理指令";
            instructionBlocking.append(instructionBlockingCheckbox, instructionBlockingText);
        }
        const forceStartDelegation = document.createElement("label");
        forceStartDelegation.className = "checkbox-line";
        if (agent.is_main) {
            const forceStartDelegationCheckbox = document.createElement("input");
            forceStartDelegationCheckbox.type = "checkbox";
            forceStartDelegationCheckbox.dataset.forceStartDelegation = "true";
            forceStartDelegationCheckbox.checked = state.runtimeSettings.force_start_delegation;
            const forceStartDelegationText = document.createElement("span");
            forceStartDelegationText.textContent = "强制开始委派";
            forceStartDelegation.append(forceStartDelegationCheckbox, forceStartDelegationText);
        }
        const blockNarrativeRead = document.createElement("label");
        blockNarrativeRead.className = "checkbox-line";
        if (agent.is_main) {
            const blockNarrativeReadCheckbox = document.createElement("input");
            blockNarrativeReadCheckbox.type = "checkbox";
            blockNarrativeReadCheckbox.dataset.blockCoordinatorNarrativeRead = "true";
            blockNarrativeReadCheckbox.checked = state.runtimeSettings.block_coordinator_narrative_read;
            const blockNarrativeReadText = document.createElement("span");
            blockNarrativeReadText.textContent = "禁止读取 narrative.md";
            blockNarrativeRead.append(blockNarrativeReadCheckbox, blockNarrativeReadText);
        }
        const reportAttachment = document.createElement("label");
        reportAttachment.className = "checkbox-line";
        if (agent.name === "narrator") {
            const forcePublishCheckbox = document.createElement("input");
            forcePublishCheckbox.type = "checkbox";
            forcePublishCheckbox.dataset.forcePublishNarrative = "true";
            forcePublishCheckbox.checked = state.runtimeSettings.force_publish_narrative;
            const forcePublishText = document.createElement("span");
            forcePublishText.textContent = "强制发布";
            reportAttachment.append(forcePublishCheckbox, forcePublishText);
        } else if (agent.produces_reports) {
            const reportAttachmentCheckbox = document.createElement("input");
            reportAttachmentCheckbox.type = "checkbox";
            reportAttachmentCheckbox.dataset.agentReportAlwaysAttached = agent.name;
            reportAttachmentCheckbox.checked = state.runtimeSettings.always_attach_report_agents.includes(agent.name);
            const reportAttachmentText = document.createElement("span");
            reportAttachmentText.textContent = "强制附加报告";
            reportAttachment.append(reportAttachmentCheckbox, reportAttachmentText);
        }
        const availability = document.createElement("label");
        availability.className = "checkbox-line";
        if (agent.can_disable) {
            const enabled = document.createElement("input");
            enabled.type = "checkbox";
            enabled.dataset.agentEnabled = agent.name;
            enabled.checked = !state.runtimeSettings.disabled_agents.includes(agent.name);
            const availabilityText = document.createElement("span");
            availabilityText.textContent = "启用 Agent";
            availability.append(enabled, availabilityText);
        } else {
            const availabilityText = document.createElement("span");
            availabilityText.textContent = "始终启用";
            availability.append(availabilityText);
        }
        label.append(select, streaming, prefill);
        if (agent.supports_fixed_response) label.append(fixedResponse);
        if (agent.is_main) label.append(forceStartDelegation);
        if (agent.is_main) label.append(blockNarrativeRead);
        if (!agent.is_main) label.append(instructionBlocking);
        if (agent.produces_reports) label.append(reportAttachment);
        label.append(availability);
        fields.append(label);
    }
}

async function saveRuntimeSettings(event) {
    event.preventDefault();
    setRuntimeSettingsStatus("保存中……");
    try {
        const contextTurns = $("#context-turns").value;
        if (!/^[1-9]\d*$/.test(contextTurns) || BigInt(contextTurns) < 2n) {
            throw new Error("可见故事历史回合数必须是大于等于 2 的整数");
        }
        const payload = {
            max_delegations: Number($("#max-delegations").value),
            context_turns: "__CONTEXT_TURNS__",
            disabled_agents: Array.from($("#agent-preset-fields").querySelectorAll("input[data-agent-enabled]:not(:checked)"), (checkbox) => checkbox.dataset.agentEnabled),
            prefill_agents: Array.from($("#agent-preset-fields").querySelectorAll("input[data-agent-prefill]:checked"), (checkbox) => checkbox.dataset.agentPrefill),
            blocked_instruction_agents: Array.from($("#agent-preset-fields").querySelectorAll("input[data-agent-instruction-blocked]:checked"), (checkbox) => checkbox.dataset.agentInstructionBlocked),
            always_attach_report_agents: Array.from($("#agent-preset-fields").querySelectorAll("input[data-agent-report-always-attached]:checked"), (checkbox) => checkbox.dataset.agentReportAlwaysAttached),
            force_publish_narrative: Boolean($("#agent-preset-fields").querySelector("input[data-force-publish-narrative]:checked")),
            force_start_delegation: Boolean($("#agent-preset-fields").querySelector("input[data-force-start-delegation]:checked")),
            block_coordinator_narrative_read: Boolean($("#agent-preset-fields").querySelector("input[data-block-coordinator-narrative-read]:checked")),
            use_compliance_fixed_response: Boolean($("#agent-preset-fields").querySelector("input[data-compliance-fixed-response]:checked")),
        };
        state.runtimeSettings = await api("/api/runtime-settings", {
            method: "PUT",
            body: JSON.stringify(payload).replace('"__CONTEXT_TURNS__"', contextTurns),
        });
        const main = $("#agent-preset-fields select[data-agent='coordinator']");
        const overrides = {};
        const streaming = {};
        for (const select of $("#agent-preset-fields").querySelectorAll("select[data-agent]")) {
            if (select.dataset.agent !== "coordinator" && select.value) overrides[select.dataset.agent] = select.value;
        }
        for (const checkbox of $("#agent-preset-fields").querySelectorAll("input[data-agent-streaming]")) {
            if (checkbox.checked) streaming[checkbox.dataset.agentStreaming] = true;
        }
        state.presets = await api("/api/agent-preset-settings", {
            method: "PUT",
            body: JSON.stringify({ main_preset_id: main.value, agent_preset_overrides: overrides, agent_streaming: streaming }),
        });
        await loadAgentPresetSettings();
        $("#max-delegations").value = state.runtimeSettings.max_delegations;
        $("#context-turns").value = state.runtimeSettings.context_turns;
        setRuntimeSettingsStatus("通用设置已保存");
    } catch (error) {
        setRuntimeSettingsStatus(error.message, true);
    }
}

function setRuntimeSettingsStatus(message, error = false) {
    const status = $("#runtime-settings-status");
    status.textContent = message;
    status.classList.toggle("error-text", error);
}

function renderPresetList() {
    elements.presetList.replaceChildren();
    for (const preset of state.presets.presets) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "preset-item";
        button.classList.toggle("selected", preset.id === state.selectedPresetId);
        const name = document.createElement("span");
        name.className = "preset-item-name";
        name.textContent = preset.name;
        const meta = document.createElement("span");
        meta.className = "preset-item-meta";
        meta.textContent = `${preset.provider} / ${preset.model}`;
        button.append(name, meta);
        button.addEventListener("click", () => {
            selectPreset(preset.id);
            revealMobileEditor($("#ai-settings-panel .preset-editor"));
        });
        elements.presetList.append(button);
    }
    requestAnimationFrame(() => keepItemVisible(elements.presetList, elements.presetList.querySelector(".selected")));
}

function selectPreset(presetId) {
    const preset = state.presets.presets.find((item) => item.id === presetId);
    if (!preset) return;
    state.selectedPresetId = preset.id;
    state.creatingPreset = false;
    renderPresetList();
    fillPresetForm(preset);
    $("#preset-state").textContent = "编辑 Preset";
    $("#duplicate-preset-button").disabled = false;
    $("#delete-preset-button").disabled = state.presets.presets.length === 1;
}

function fillPresetForm(preset) {
    state.presetDirty = false;
    $("#preset-name").value = preset.name;
    $("#provider").value = preset.provider;
    $("#xai-protocol").value = preset.xai_protocol || "responses";
    $("#openai-protocol").value = preset.openai_protocol || "chat_completions";
    $("#base-url").value = preset.base_url;
    $("#api-key").value = "";
    $("#model").value = preset.model;
    $("#timeout").value = preset.timeout_seconds;
    $("#temperature").value = preset.temperature;
    $("#top-p").value = preset.top_p;
    $("#max-tokens").value = preset.max_tokens;
    $("#key-status").textContent = preset.has_api_key ? `已保存 ${preset.masked_api_key}` : "尚未设置";
    resetModelOptions();
    setPresetStatus("");
    updateProviderFields();
    updateGeminiThinkingFields(preset.thinking_level);
}

function startNewPreset() {
    state.creatingPreset = true;
    state.selectedPresetId = null;
    renderPresetList();
    fillPresetForm({
        name: "新 AI Preset",
        provider: "deepseek",
        base_url: "https://api.deepseek.com",
        model: "deepseek-chat",
        xai_protocol: "responses",
        openai_protocol: "chat_completions",
        timeout_seconds: 120,
        temperature: 1,
        top_p: 1,
        max_tokens: 65536,
        thinking_level: null,
        has_api_key: false,
        masked_api_key: "",
    });
    state.presetDirty = true;
    $("#preset-state").textContent = "尚未保存的新 Preset";
    $("#duplicate-preset-button").disabled = true;
    $("#delete-preset-button").disabled = true;
    $("#preset-name").focus();
}

function presetPayload() {
    return {
        name: $("#preset-name").value,
        provider: $("#provider").value,
        base_url: $("#base-url").value,
        api_key: $("#api-key").value,
        model: $("#model").value,
        xai_protocol: $("#xai-protocol").value,
        openai_protocol: $("#openai-protocol").value,
        timeout_seconds: Number($("#timeout").value),
        temperature: Number($("#temperature").value),
        top_p: Number($("#top-p").value),
        max_tokens: Number($("#max-tokens").value),
        thinking_level: $("#gemini-thinking-level").value || null,
    };
}

function modelListPayload() {
    return {
        preset_id: state.creatingPreset ? null : state.selectedPresetId,
        provider: $("#provider").value,
        base_url: $("#base-url").value,
        api_key: $("#api-key").value,
        xai_protocol: $("#xai-protocol").value,
        openai_protocol: $("#openai-protocol").value,
        timeout_seconds: Number($("#timeout").value),
    };
}

function resetModelOptions() {
    const modelSelect = $("#model-select");
    modelSelect.replaceChildren(new Option("尚未拉取模型", ""));
    modelSelect.disabled = true;
}

async function savePreset(event) {
    event.preventDefault();
    setPresetStatus("保存中……");
    try {
        const previousIds = new Set(state.presets.presets.map((preset) => preset.id));
        const path = state.creatingPreset ? "/api/ai-presets" : `/api/ai-presets/${state.selectedPresetId}`;
        const collection = await api(path, {
            method: state.creatingPreset ? "POST" : "PUT",
            body: JSON.stringify(presetPayload()),
        });
        let preferredId = state.selectedPresetId;
        if (state.creatingPreset) {
            preferredId = collection.presets.find((preset) => !previousIds.has(preset.id))?.id;
        }
        state.presets = collection;
        if (state.agentPresetSettings) await loadAgentPresetSettings();
        state.selectedPresetId = preferredId;
        state.creatingPreset = false;
        state.presetDirty = false;
        renderPresetList();
        selectPreset(preferredId);
        setPresetStatus("Preset 已保存");
    } catch (error) {
        setPresetStatus(error.message, true);
    }
}

async function duplicatePreset() {
    if (!state.selectedPresetId) return;
    const current = state.presets.presets.find((preset) => preset.id === state.selectedPresetId);
    const name = window.prompt("复制后的 Preset 名称", `${current.name} 副本`);
    if (!name?.trim()) return;
    const previousIds = new Set(state.presets.presets.map((preset) => preset.id));
    try {
        state.presets = await api(`/api/ai-presets/${state.selectedPresetId}/duplicate`, {
            method: "POST",
            body: JSON.stringify({ name: name.trim() }),
        });
        if (state.agentPresetSettings) await loadAgentPresetSettings();
        const duplicate = state.presets.presets.find((preset) => !previousIds.has(preset.id));
        state.selectedPresetId = duplicate?.id || state.selectedPresetId;
        renderPresetList();
        selectPreset(state.selectedPresetId);
        setPresetStatus("Preset 已复制");
    } catch (error) {
        setPresetStatus(error.message, true);
    }
}

async function deletePreset() {
    if (!state.selectedPresetId) return;
    const current = state.presets.presets.find((preset) => preset.id === state.selectedPresetId);
    confirmDeletion(`确认删除 AI Preset“${current.name}”？`, async () => {
        state.presets = await api(`/api/ai-presets/${state.selectedPresetId}`, { method: "DELETE" });
        state.selectedPresetId = state.presets.main_preset_id;
        if (state.agentPresetSettings) await loadAgentPresetSettings();
        renderPresetList();
        selectPreset(state.selectedPresetId);
        setPresetStatus("Preset 已删除");
    });
}

async function loadModels() {
    if (!$("#base-url").reportValidity() || !$("#timeout").reportValidity()) return;
    const payload = modelListPayload();
    const requestKey = JSON.stringify(payload);
    setPresetStatus("正在拉取模型列表……");
    try {
        const models = await api("/api/ai-presets/models", {
            method: "POST",
            body: JSON.stringify(payload),
        });
        if (JSON.stringify(modelListPayload()) !== requestKey) return;
        const options = [new Option("选择已拉取的模型", "")];
        for (const model of models) {
            const label = model.owned_by ? `${model.id} / ${model.owned_by}` : model.id;
            options.push(new Option(label, model.id));
        }
        const modelSelect = $("#model-select");
        modelSelect.replaceChildren(...options);
        modelSelect.disabled = models.length === 0;
        setPresetStatus(`已拉取 ${models.length} 个模型，可直接输入或选择`);
    } catch (error) {
        if (JSON.stringify(modelListPayload()) !== requestKey) return;
        setPresetStatus(error.message, true);
    }
}

async function testPreset() {
    if (!state.selectedPresetId || state.creatingPreset) {
        setPresetStatus("请先保存 Preset，再执行连接测试", true);
        return;
    }
    if (state.presetDirty) {
        setPresetStatus("请先保存当前修改，再执行连接测试", true);
        return;
    }
    const requestedPresetId = state.selectedPresetId;
    setPresetStatus("正在测试连接……");
    try {
        const result = await api(`/api/ai-presets/${requestedPresetId}/test`, { method: "POST" });
        if (state.selectedPresetId !== requestedPresetId) return;
        setPresetStatus(`连接成功 / ${result.model} / ${result.latency_ms} ms`);
    } catch (error) {
        if (state.selectedPresetId !== requestedPresetId) return;
        setPresetStatus(error.message, true);
    }
}

async function runPresetAction(action, message) {
    try {
        state.presets = await action();
        renderPresetList();
        selectPreset(state.selectedPresetId);
        setPresetStatus(message);
    } catch (error) {
        setPresetStatus(error.message, true);
    }
}

function updateProviderFields(applyDefaults = false) {
    $("#base-url").disabled = false;
    $("#xai-protocol-field").classList.toggle("hidden", $("#provider").value !== "xai");
    $("#openai-protocol-field").classList.toggle("hidden", $("#provider").value !== "openai_compatible");
    if (applyDefaults) {
        const defaults = {
            deepseek: ["https://api.deepseek.com", "deepseek-chat"],
            openai_compatible: ["", ""],
            google_gemini: ["https://generativelanguage.googleapis.com/v1beta", "gemini-flash-latest"],
            anthropic: ["https://api.anthropic.com/v1", "claude-sonnet-4-6"],
            xai: ["https://api.x.ai/v1", "grok-4.6"],
        }[$("#provider").value];
        if (defaults) {
            $("#base-url").value = defaults[0];
            $("#model").value = defaults[1];
        }
    }
    updateGeminiThinkingFields(undefined, applyDefaults);
}

function updateGeminiThinkingFields(preferred = $("#gemini-thinking-level").value, notify = false) {
    const model = $("#model").value.trim().replace(/^models\//, "");
    const levels = $("#provider").value === "google_gemini" ? state.presets?.gemini_thinking_levels?.[model] : null;
    const select = $("#gemini-thinking-level");
    select.replaceChildren(new Option("自动（模型默认）", ""));
    for (const level of levels || []) {
        const labels = {minimal: "最低", low: "低", medium: "中", high: "高"};
        select.add(new Option(labels[level], level));
    }
    select.value = levels?.includes(preferred) ? preferred : "";
    $("#gemini-thinking-field").classList.toggle("hidden", !levels);
    if (notify && preferred && !select.value) {
        setPresetStatus("当前模型不支持原思考强度，已改为自动；请保存 Preset");
        return true;
    }
    return false;
}

function setPresetStatus(message, error = false) {
    const status = $("#settings-status");
    status.textContent = error ? shortErrorMessage(message) : message;
    status.classList.toggle("error-text", error);
}

function showSettingsSection(section) {
    state.settingsSection = section;
    let activeButton;
    for (const button of document.querySelectorAll("[data-settings-section]")) {
        const active = button.dataset.settingsSection === section;
        button.classList.toggle("active", active);
        if (active) activeButton = button;
    }
    $("#ai-settings-panel").classList.toggle("hidden", section !== "ai");
    $("#drawing-settings-panel").classList.toggle("hidden", section !== "drawing");
    $("#capability-settings-panel").classList.toggle("hidden", section !== "capability");
    $("#general-settings-panel").classList.toggle("hidden", section !== "general");
    keepTabVisible($("#settings-view .settings-section-nav"), activeButton);
    requestAnimationFrame(() => {
        const list = section === "ai" ? elements.presetList : section === "drawing" ? elements.drawingPresetList : null;
        if (list) keepItemVisible(list, list.querySelector(".selected"));
    });
}

async function loadDrawingPresets(preferredId = state.selectedDrawingPresetId) {
    state.drawingPresets = await api("/api/drawing-presets");
    const available = state.drawingPresets.presets.some((preset) => preset.id === preferredId);
    state.selectedDrawingPresetId = available ? preferredId : state.drawingPresets.default_preset_id;
    state.creatingDrawingPreset = false;
    renderDrawingPresetList();
    if (state.selectedDrawingPresetId) selectDrawingPreset(state.selectedDrawingPresetId);
    else startNewDrawingPreset();
}

function renderDrawingPresetList() {
    elements.drawingPresetList.replaceChildren();
    for (const preset of state.drawingPresets.presets) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "preset-item";
        button.classList.toggle("selected", preset.id === state.selectedDrawingPresetId);
        button.textContent = preset.name;
        button.addEventListener("click", () => {
            selectDrawingPreset(preset.id);
            revealMobileEditor($("#drawing-settings-panel .preset-editor"));
        });
        elements.drawingPresetList.append(button);
    }
    requestAnimationFrame(() => keepItemVisible(elements.drawingPresetList, elements.drawingPresetList.querySelector(".selected")));
}

function selectDrawingPreset(presetId) {
    const preset = state.drawingPresets.presets.find((item) => item.id === presetId);
    if (!preset) return;
    state.selectedDrawingPresetId = preset.id;
    state.creatingDrawingPreset = false;
    state.drawingPresetDirty = false;
    renderDrawingPresetList();
    for (const [id, value] of Object.entries({
        "drawing-preset-name": preset.name, "drawing-provider": preset.provider, "drawing-base-url": preset.base_url,
        "drawing-auth-username": preset.auth_username, "drawing-timeout": preset.timeout_seconds, "drawing-model": preset.model,
    })) $("#" + id).value = value;
    $("#drawing-auth-password").value = "";
    $("#clear-drawing-auth-password").checked = false;
    $("#drawing-auth-status").textContent = preset.has_auth_password ? "已保存密码" : "未设置";
    $("#drawing-api-key").value = "";
    $("#clear-drawing-api-key").checked = false;
    $("#drawing-api-key-status").textContent = preset.has_api_key ? "已保存 Token" : "未设置";
    updateDrawingProviderFields();
    $("#drawing-preset-state").textContent = preset.id === state.drawingPresets.default_preset_id ? "编辑 Preset / 默认" : "编辑 Preset";
    $("#set-default-drawing-preset-button").disabled = preset.id === state.drawingPresets.default_preset_id;
    $("#duplicate-drawing-preset-button").disabled = false;
    $("#delete-drawing-preset-button").disabled = false;
    setDrawingStatus("");
}

function startNewDrawingPreset() {
    state.creatingDrawingPreset = true;
    state.selectedDrawingPresetId = null;
    state.drawingPresetDirty = true;
    renderDrawingPresetList();
    selectDrawingPresetValues({ name: "新绘画 Preset", provider: "stable_diffusion_webui", base_url: "http://127.0.0.1:7860", auth_username: "", model: "nai-diffusion-5-full", timeout_seconds: 300 });
    $("#drawing-preset-state").textContent = "尚未保存的新 Preset";
    $("#set-default-drawing-preset-button").disabled = true;
    $("#duplicate-drawing-preset-button").disabled = true;
    $("#delete-drawing-preset-button").disabled = true;
}

function selectDrawingPresetValues(preset) {
    for (const [id, value] of Object.entries({ "drawing-preset-name": preset.name, "drawing-provider": preset.provider, "drawing-base-url": preset.base_url, "drawing-auth-username": preset.auth_username, "drawing-model": preset.model, "drawing-timeout": preset.timeout_seconds })) $("#" + id).value = value;
    $("#drawing-auth-password").value = "";
    $("#clear-drawing-auth-password").checked = false;
    $("#drawing-auth-status").textContent = "未设置";
    $("#drawing-api-key").value = "";
    $("#clear-drawing-api-key").checked = false;
    $("#drawing-api-key-status").textContent = "未设置";
    updateDrawingProviderFields();
}

function defaultDrawingPreset() {
    return state.drawingPresets?.presets.find((preset) => preset.id === state.drawingPresets.default_preset_id);
}

function updateDrawingProviderFields() {
    const novelai = $("#drawing-provider").value === "novelai";
    const defaultNovelAI = defaultDrawingPreset()?.provider === "novelai";
    for (const label of document.querySelectorAll(".drawing-webui-field")) label.classList.toggle("hidden", novelai);
    for (const label of document.querySelectorAll(".drawing-novelai-field")) label.classList.toggle("hidden", !novelai);
    for (const label of document.querySelectorAll(".capability-webui-field")) label.classList.toggle("hidden", defaultNovelAI);
    for (const label of document.querySelectorAll(".capability-novelai-field")) label.classList.toggle("hidden", !defaultNovelAI);
    $("#capability-steps").max = defaultNovelAI ? "50" : "150";
    $("#capability-cfg-scale").max = defaultNovelAI ? "10" : "30";
    for (const id of ["capability-width", "capability-height"]) {
        $("#" + id).max = defaultNovelAI ? "1600" : "2048";
        $("#" + id).step = defaultNovelAI ? "64" : "8";
    }
}

function drawingPresetPayload() {
    const novelai = $("#drawing-provider").value === "novelai";
    return { name: $("#drawing-preset-name").value, provider: $("#drawing-provider").value, base_url: $("#drawing-base-url").value, auth_username: novelai ? "" : $("#drawing-auth-username").value, auth_password: novelai ? "" : $("#drawing-auth-password").value, clear_auth_password: novelai || $("#clear-drawing-auth-password").checked, api_key: novelai ? $("#drawing-api-key").value : "", clear_api_key: !novelai || $("#clear-drawing-api-key").checked, model: $("#drawing-model").value, timeout_seconds: Number($("#drawing-timeout").value) };
}

async function saveDrawingPreset(event) {
    event.preventDefault();
    setDrawingStatus("保存中……");
    try {
        const prior = new Set(state.drawingPresets.presets.map((preset) => preset.id));
        const path = state.creatingDrawingPreset ? "/api/drawing-presets" : `/api/drawing-presets/${state.selectedDrawingPresetId}`;
        const collection = await api(path, { method: state.creatingDrawingPreset ? "POST" : "PUT", body: JSON.stringify(drawingPresetPayload()) });
        state.selectedDrawingPresetId = state.creatingDrawingPreset ? collection.presets.find((preset) => !prior.has(preset.id))?.id : state.selectedDrawingPresetId;
        state.drawingPresets = collection;
        selectDrawingPreset(state.selectedDrawingPresetId);
        setDrawingStatus("绘画 Preset 已保存");
    } catch (error) { setDrawingStatus(error.message, true); }
}

function setCapabilityOptions(id, resources, selected) {
    const select = $("#" + id);
    const options = resources.map((item) => typeof item === "string"
        ? new Option(item, item)
        : new Option(item.label ?? item.id, item.id));
    if (selected && !options.some((option) => option.value === selected)) options.push(new Option(selected, selected));
    if (!options.length) options.push(new Option("没有可用资源", ""));
    select.replaceChildren(...options);
    select.value = selected || "";
}

async function loadCapabilitySettings() {
    state.capabilitySettings = await api("/api/capability-settings");
    const settings = state.capabilitySettings.character_portrait_generation;
    const tokenCounting = state.capabilitySettings.narrative_token_counting;
    $("#capability-enabled").checked = settings.enabled;
    setCapabilityOptions("capability-checkpoint", [], settings.checkpoint);
    setCapabilityOptions("capability-sampler", [], settings.sampler_name);
    setCapabilityOptions("capability-scheduler", [], settings.scheduler);
    $("#capability-novelai-sampler").value = settings.novelai_sampler;
    updateDrawingProviderFields();
    for (const [id, value] of Object.entries({
        "capability-steps": settings.steps,
        "capability-cfg-scale": settings.cfg_scale,
        "capability-width": settings.width,
        "capability-height": settings.height,
        "capability-positive-prompt": settings.positive_prompt,
        "capability-negative-prompt": settings.negative_prompt,
    })) $("#" + id).value = value;
    $("#token-counting-enabled").checked = tokenCounting.enabled;
    $("#token-counting-max-tokens").value = tokenCounting.max_tokens;
    state.capabilitySettingsDirty = false;
    setCapabilityStatus("");
}

function capabilitySettingsPayload() {
    return {
        character_portrait_generation: {
            enabled: $("#capability-enabled").checked,
            checkpoint: $("#capability-checkpoint").value,
            sampler_name: $("#capability-sampler").value,
            scheduler: $("#capability-scheduler").value,
            novelai_sampler: $("#capability-novelai-sampler").value,
            steps: Number($("#capability-steps").value),
            cfg_scale: Number($("#capability-cfg-scale").value),
            width: Number($("#capability-width").value),
            height: Number($("#capability-height").value),
            positive_prompt: $("#capability-positive-prompt").value,
            negative_prompt: $("#capability-negative-prompt").value,
        },
        narrative_token_counting: {
            enabled: $("#token-counting-enabled").checked,
            max_tokens: Number($("#token-counting-max-tokens").value),
        },
    };
}

function selectCapabilityPanel(button) {
    document.querySelectorAll("[data-capability-panel]").forEach((item) => {
        const selected = item === button;
        item.classList.toggle("selected", selected);
        item.setAttribute("aria-pressed", String(selected));
        $("#" + item.dataset.capabilityPanel).classList.toggle("hidden", !selected);
    });
}

async function saveCapabilitySettings(event) {
    event.preventDefault();
    setCapabilityStatus("保存中……");
    try {
        state.capabilitySettings = await api("/api/capability-settings", {
            method: "PUT",
            body: JSON.stringify(capabilitySettingsPayload()),
        });
        state.capabilitySettingsDirty = false;
        setCapabilityStatus("能力设置已保存");
    } catch (error) {
        setCapabilityStatus(error.message, true);
    }
}

async function loadCapabilityResources() {
    const defaultPresetId = state.drawingPresets?.default_preset_id;
    if (!defaultPresetId) {
        setCapabilityStatus("未设置默认绘画 Preset，无法拉取 WebUI 资源", true);
        return;
    }
    if (defaultDrawingPreset()?.provider !== "stable_diffusion_webui") return setCapabilityStatus("NovelAI 不需要拉取 WebUI 配置", true);
    setCapabilityStatus("正在拉取 WebUI 配置……");
    try {
        const resources = await api(`/api/drawing-presets/${defaultPresetId}/resources`);
        if (state.drawingPresets?.default_preset_id !== defaultPresetId) {
            setCapabilityStatus("默认绘画 Preset 已变化，请重新拉取 WebUI 配置", true);
            return;
        }
        setCapabilityOptions("capability-checkpoint", resources.checkpoints || [], $("#capability-checkpoint").value || resources.active_checkpoint);
        setCapabilityOptions("capability-sampler", resources.samplers || [], $("#capability-sampler").value);
        setCapabilityOptions("capability-scheduler", resources.schedulers || [], $("#capability-scheduler").value);
        state.capabilitySettingsDirty = true;
        setCapabilityStatus("已拉取 WebUI 配置，请保存能力设置");
    } catch (error) {
        setCapabilityStatus(error.message, true);
    }
}

async function testDrawingPreset() {
    if (!state.selectedDrawingPresetId || state.creatingDrawingPreset || state.drawingPresetDirty) return setDrawingStatus("请先保存当前 Preset，再执行连接测试", true);
    try { const result = await api(`/api/drawing-presets/${state.selectedDrawingPresetId}/test`, { method: "POST" }); setDrawingStatus(`连接成功 / ${result.active_checkpoint || "未报告模型"} / ${result.latency_ms} ms`); } catch (error) { setDrawingStatus(error.message, true); }
}

async function generateDrawingTest() {
    const contentPrompt = $("#drawing-test-content").value.trim();
    if (!state.selectedDrawingPresetId || state.creatingDrawingPreset || state.drawingPresetDirty) return setDrawingTestStatus("请先保存当前 Preset，再生成测试图", true);
    if (state.capabilitySettingsDirty) return setDrawingTestStatus("请先保存能力设置，再生成测试图", true);
    if (!contentPrompt) return setDrawingTestStatus("请输入正面内容词", true);
    setDrawingTestStatus("正在生成……");
    try { const result = await api(`/api/drawing-presets/${state.selectedDrawingPresetId}/test-generation`, { method: "POST", body: JSON.stringify({ content_prompt: contentPrompt }) }); const image = $("#drawing-test-image"); image.src = result.image_data_url; image.classList.remove("hidden"); setDrawingTestStatus(`生成完成 / ${result.width} x ${result.height}${result.seed === null ? "" : ` / seed ${result.seed}`}`); } catch (error) { setDrawingTestStatus(error.message, true); }
}

function setDrawingStatus(message, error = false) { const status = $("#drawing-settings-status"); status.textContent = error ? shortErrorMessage(message) : message; status.classList.toggle("error-text", error); }
function setDrawingTestStatus(message, error = false) { const status = $("#drawing-test-status"); status.textContent = error ? shortErrorMessage(message) : message; status.classList.toggle("error-text", error); }
function setCapabilityStatus(message, error = false) { const status = $("#capability-settings-status"); status.textContent = message; status.classList.toggle("error-text", error); }

function fixedCreationBase(kind, sourceName = null) {
    const base = `/api/content/${kind}`;
    return sourceName === null ? base : `${base}/${encodeURIComponent(sourceName)}`;
}

function creationKindLabel(kind = state.creationKind) {
    return { worlds: "世界", mods: "模组", styles: "文风" }[kind];
}

function isStyleCreation() {
    return state.creationKind === "styles";
}

function setControlsDisabled(container, disabled) {
    for (const control of container.querySelectorAll("button, input, select, textarea")) control.disabled = disabled;
}

function setCreationBusy(busy) {
    state.creationBusy = busy;
    setControlsDisabled(elements.creation, busy);
}

function setCreationStatus(message, error = false) {
    elements.creationStatus.textContent = message;
    elements.creationStatus.classList.toggle("error-text", error);
}

async function runCreation(action, loadingMessage = "") {
    if (state.creationBusy) return;
    setCreationBusy(true);
    if (loadingMessage) setCreationStatus(loadingMessage);
    try {
        await action();
    } catch (error) {
        setCreationStatus(error.message, true);
    } finally {
        setCreationBusy(false);
    }
}

async function showCreation() {
    if (elements.creation.classList.contains("hidden")) {
        rememberWorkspaceScroll();
        elements.launcher.classList.add("hidden");
        elements.workspace.classList.add("hidden");
        elements.settings.classList.add("hidden");
        elements.creation.classList.remove("hidden");
        elements.adventureButton.classList.remove("active");
        elements.foregroundButton.classList.remove("active");
        elements.backendButton.classList.remove("active");
        elements.settingsButton.classList.remove("active");
        elements.creationButton.classList.add("active");
        keepTabVisible(document.querySelector(".view-switch"), elements.creationButton);
    }
    if (!state.creationLoaded) await runCreation(loadCreationSources, "正在加载创作列表……");
}

async function loadCreationSources(preferredName = state.selectedCreationSourceName, kind = state.creationKind) {
    const sources = await api(`/api/content/${kind}`);
    state.creationLoaded = true;
    state.creationKind = kind;
    state.creationSources = sources;
    state.selectedCreationSourceName = sources.some((source) => source.name === preferredName) ? preferredName : null;
    renderCreationSourceList();
    if (state.selectedCreationSourceName) {
        await loadSelectedCreationSource();
    } else {
        clearCreationEditor();
        setCreationStatus("");
    }
}

function adjustSelectedSourceCounts(scenarioDelta, entityDelta) {
    const source = state.selectedCreationSource;
    if (!source) return;
    source.scenario_count += scenarioDelta;
    source.entity_count += entityDelta;
    const index = state.creationSources.findIndex((item) => item.name === source.name);
    if (index >= 0) state.creationSources[index] = source;
    renderCreationSourceList();
    renderCreationHeading();
}

function renderCreationSourceList() {
    elements.creationSourceList.replaceChildren();
    const label = creationKindLabel();
    $("#creation-source-label").textContent = `${label}列表`;
    $("#creation-description").textContent = isStyleCreation()
        ? "文风是全局实时资源；已有存档的后续新回合也会读取当前版本。"
        : "直接编辑世界与模组源文件；已有存档快照不会随之改变。";
    for (const tab of document.querySelectorAll("[data-creation-kind]")) {
        const selected = tab.dataset.creationKind === state.creationKind;
        tab.classList.toggle("active", selected);
        tab.setAttribute("aria-pressed", String(selected));
    }
    if (!state.creationSources.length) elements.creationSourceList.append(empty(`还没有${label}`));
    for (const source of state.creationSources) {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "creation-source-item";
        button.classList.toggle("selected", source.name === state.selectedCreationSourceName);
        button.classList.toggle("disabled-style", isStyleCreation() && !source.enabled);
        button.setAttribute("aria-pressed", String(source.name === state.selectedCreationSourceName));
        const name = document.createElement("strong");
        name.textContent = isStyleCreation() && !source.enabled ? `${source.name} [已禁用]` : source.name;
        const path = document.createElement("span");
        path.textContent = source.path;
        path.title = source.path;
        const detail = document.createElement("span");
        detail.textContent = isStyleCreation()
            ? (source.description || "尚未填写简介")
            : `${source.scenario_count} 场景 / ${source.entity_count} 实体`;
        detail.title = detail.textContent;
        button.append(name, path, detail);
        button.addEventListener("click", async () => {
            await selectCreationSource(source.name);
            if (state.selectedCreationSource?.name === source.name) revealMobileEditor($(".creation-detail"));
        });
        elements.creationSourceList.append(button);
    }
    requestAnimationFrame(() => keepItemVisible(elements.creationSourceList, elements.creationSourceList.querySelector(".selected")));
}

function clearCreationEditor() {
    state.selectedCreationSource = null;
    state.selectedEntityName = null;
    $("#creation-editor").classList.add("hidden");
    $("#style-editor").classList.add("hidden");
    $("#creation-empty").classList.remove("hidden");
    $("#creation-empty").textContent = `选择一个${creationKindLabel()}开始编辑`;
}

function renderCreationHeading() {
    if (!state.selectedCreationSource) return;
    $("#creation-selected-name").textContent = state.selectedCreationSource.name;
    $("#creation-selected-path").textContent = `${state.selectedCreationSource.path} / ${state.selectedCreationSource.scenario_count} 场景 / ${state.selectedCreationSource.entity_count} 实体`;
}

function renderStyleEnabledState() {
    const source = state.selectedCreationSource;
    if (!source || !isStyleCreation()) return;
    elements.toggleStyleEnabledButton.textContent = source.enabled ? "禁用文风" : "启用文风";
    elements.toggleStyleEnabledButton.setAttribute("aria-pressed", String(source.enabled));
}

async function selectCreationSource(name) {
    if ((name === state.selectedCreationSourceName && state.selectedCreationSource) || state.creationBusy) return;
    await runCreation(async () => {
        state.selectedCreationSourceName = name;
        state.selectedEntityName = null;
        state.creationSection = "basic";
        clearCreationEditor();
        renderCreationSourceList();
        renderCreationSection();
        await loadSelectedCreationSource();
    }, "正在加载来源……");
}

async function loadSelectedCreationSource() {
    if (!state.selectedCreationSourceName) {
        clearCreationEditor();
        return;
    }
    const source = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}`);
    state.selectedCreationSource = source;
    $("#creation-empty").classList.add("hidden");
    $("#creation-editor").classList.toggle("hidden", isStyleCreation());
    $("#style-editor").classList.toggle("hidden", !isStyleCreation());
    if (isStyleCreation()) {
        $("#style-selected-name").textContent = source.name;
        $("#style-selected-path").textContent = source.path;
        $("#style-name").value = source.name;
        $("#style-description").value = source.description;
        $("#style-content").value = source.content;
        renderStyleEnabledState();
        setCreationStatus("");
        return;
    }
    renderCreationHeading();
    $("#source-name").value = source.name;
    $("#source-description").value = source.description;
    renderCreationSection();
    setCreationStatus("");
}

async function selectCreationKind(kind) {
    if (kind === state.creationKind || state.creationBusy) return;
    await runCreation(async () => {
        await loadCreationSources(null, kind);
        state.creationSection = "basic";
        renderCreationSection();
    }, "正在加载创作列表……");
}

function renderCreationSection() {
    for (const button of document.querySelectorAll("[data-creation-section]")) {
        const selected = button.dataset.creationSection === state.creationSection;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", String(selected));
    }
    $("#creation-basic-panel").classList.toggle("hidden", state.creationSection !== "basic");
    $("#creation-scenarios-panel").classList.toggle("hidden", state.creationSection !== "scenarios");
    $("#creation-entities-panel").classList.toggle("hidden", state.creationSection !== "entities");
}

async function selectCreationSection(section) {
    if (section === state.creationSection || state.creationBusy) return;
    await runCreation(async () => {
        if (section === "scenarios") await loadScenarios();
        if (section === "entities") await loadEntities();
        if (section === "basic") await loadSelectedCreationSource();
        state.creationSection = section;
        state.selectedEntityName = null;
        renderCreationSection();
        setCreationStatus("");
    }, "正在加载内容……");
}

async function renameSource(event) {
    event.preventDefault();
    if (!state.selectedCreationSourceName) return;
    await runCreation(async () => {
        const previousName = state.selectedCreationSourceName;
        const source = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/rename`, {
            method: "POST",
            body: JSON.stringify({ name: isStyleCreation() ? $("#style-name").value : $("#source-name").value }),
        });
        state.selectedCreationSourceName = source.name;
        state.selectedCreationSource = source;
        state.creationSources = isStyleCreation()
            ? await api(fixedCreationBase(state.creationKind))
            : state.creationSources.filter((item) => item.name !== previousName).concat(source);
        if (isStyleCreation()) {
            $("#style-name").value = source.name;
            $("#style-selected-name").textContent = source.name;
            $("#style-selected-path").textContent = source.path;
        } else {
            $("#source-name").value = source.name;
        }
        renderCreationSourceList();
        if (!isStyleCreation()) renderCreationHeading();
        setCreationStatus("名称已更新");
    }, "正在改名……");
}

async function saveSourceContent(event) {
    event.preventDefault();
    if (!state.selectedCreationSourceName) return;
    await runCreation(async () => {
        const source = await api(fixedCreationBase(state.creationKind, state.selectedCreationSourceName), {
            method: "PUT",
            body: JSON.stringify(isStyleCreation()
                ? { description: $("#style-description").value, content: $("#style-content").value }
                : { description: $("#source-description").value }),
        });
        state.selectedCreationSource = source;
        const index = state.creationSources.findIndex((item) => item.name === source.name);
        if (index >= 0) state.creationSources[index] = source;
        renderCreationSourceList();
        if (isStyleCreation()) {
            $("#style-description").value = source.description;
            $("#style-content").value = source.content;
        } else {
            renderCreationHeading();
            $("#source-description").value = source.description;
        }
        setCreationStatus(isStyleCreation() ? "文风已保存" : "描述已保存");
    }, isStyleCreation() ? "正在保存文风……" : "正在保存描述……");
}

async function toggleCurrentStyleEnabled() {
    const source = state.selectedCreationSource;
    if (!source || !isStyleCreation()) return;
    await runCreation(async () => {
        const updated = await api(`${fixedCreationBase("styles", source.name)}/enabled`, {
            method: "PUT",
            body: JSON.stringify({ enabled: !source.enabled }),
        });
        state.selectedCreationSource = updated;
        const index = state.creationSources.findIndex((item) => item.name === updated.name);
        if (index >= 0) state.creationSources[index] = updated;
        renderCreationSourceList();
        renderStyleEnabledState();
        setCreationStatus(updated.enabled ? "文风已启用" : "文风已禁用");
    }, source.enabled ? "正在禁用文风……" : "正在启用文风……");
}

function openCreateSourceDialog() {
    const label = creationKindLabel();
    setControlsDisabled(elements.createSourceForm, false);
    elements.createSourceDialog.dataset.kind = state.creationKind;
    $("#create-source-title").textContent = `// 新建${label}`;
    $("#create-source-name").value = "";
    $("#create-source-error").textContent = "";
    $("#create-source-error").classList.add("hidden");
    elements.createSourceDialog.showModal();
    $("#create-source-name").focus();
}

async function createSource(event) {
    if (event.submitter?.value !== "create") return;
    event.preventDefault();
    const requestedKind = elements.createSourceDialog.dataset.kind;
    const payload = { name: $("#create-source-name").value };
    setCreationBusy(true);
    setControlsDisabled(elements.createSourceForm, true);
    try {
        const source = await api(fixedCreationBase(requestedKind), { method: "POST", body: JSON.stringify(payload) });
        elements.createSourceDialog.close();
        state.creationKind = requestedKind;
        state.selectedCreationSourceName = source.name;
        state.selectedCreationSource = source;
        state.creationSection = "basic";
        if (requestedKind === "styles") {
            await loadCreationSources(source.name, requestedKind);
            setCreationStatus("文风已创建");
            return;
        }
        state.creationSources.push(source);
        renderCreationSourceList();
        $("#creation-empty").classList.add("hidden");
        await loadSelectedCreationSource();
        setCreationStatus(`${creationKindLabel(requestedKind)}已创建`);
    } catch (error) {
        $("#create-source-error").textContent = error.message;
        $("#create-source-error").classList.remove("hidden");
    } finally {
        setControlsDisabled(elements.createSourceForm, false);
        setCreationBusy(false);
    }
}

function deleteCurrentSource() {
    const source = state.selectedCreationSource;
    if (!source) return;
    const detail = isStyleCreation() ? "" : `（${source.scenario_count} 个场景，${source.entity_count} 个实体）`;
    confirmDeletion(`确认删除${creationKindLabel()}“${source.name}”${detail}？`, async () => {
        await api(fixedCreationBase(state.creationKind, source.name), { method: "DELETE" });
        state.creationSources = state.creationSources.filter((item) => item.name !== source.name);
        state.selectedCreationSourceName = null;
        state.selectedCreationSource = null;
        renderCreationSourceList();
        clearCreationEditor();
        setCreationStatus(`${creationKindLabel()}已删除`);
    });
}

async function loadScenarios() {
    elements.scenarioEditorList.replaceChildren(empty("正在加载场景……"));
    const scenarios = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/scenarios`);
    elements.scenarioEditorList.replaceChildren();
    if (!scenarios.length) elements.scenarioEditorList.append(createScenarioEditor());
    for (const scenario of scenarios) elements.scenarioEditorList.append(createScenarioEditor(scenario));
}

function createScenarioEditor(scenario = null) {
    const draft = scenario === null;
    const article = document.createElement("article");
    article.className = "scenario-editor";
    article.dataset.draft = String(draft);
    article.dataset.name = scenario?.name || "";
    const head = document.createElement("div");
    head.className = "scenario-editor-head";
    const title = document.createElement("strong");
    title.textContent = draft ? "未保存草稿" : scenario.name;
    const path = document.createElement("span");
    path.className = "muted";
    path.textContent = draft ? "尚未创建文件" : scenario.path;
    head.append(title, path);
    const name = labeledControl("名称", "input", scenario?.name || "");
    name.control.maxLength = 80;
    name.control.required = true;
    const fields = document.createElement("div");
    fields.className = "scenario-editor-fields";
    fields.append(name.label);
    const actions = document.createElement("div");
    actions.className = "scenario-editor-actions";
    if (draft) {
        const create = document.createElement("button");
        create.type = "button";
        create.textContent = "创建场景";
        create.addEventListener("click", () => createScenario(article, name.control));
        const remove = document.createElement("button");
        remove.type = "button";
        remove.className = "danger";
        remove.textContent = "删除草稿";
        remove.addEventListener("click", () => article.remove());
        actions.append(create, remove);
        article.append(head, fields, actions);
        return article;
    }
    const description = labeledControl("描述", "textarea", scenario?.description || "");
    description.control.rows = 3;
    description.control.maxLength = 4000;
    const opening = labeledControl("开场正文", "textarea", scenario?.opening || "");
    opening.control.rows = 9;
    opening.control.maxLength = 20000;
    opening.control.className = "scenario-opening";
    fields.append(description.label, opening.label);
    const rename = document.createElement("button");
    rename.type = "button";
    rename.textContent = "应用改名";
    rename.addEventListener("click", () => renameScenario(article, name.control.value));
    const save = document.createElement("button");
    save.type = "button";
    save.textContent = "保存内容";
    save.addEventListener("click", () => saveScenario(article, description.control, opening.control));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "删除场景";
    remove.addEventListener("click", () => deleteScenario(article));
    actions.append(rename, save, remove);
    article.append(head, fields, actions);
    return article;
}

function labeledControl(text, tag, value) {
    const label = document.createElement("label");
    label.textContent = text;
    const control = document.createElement(tag);
    control.value = value;
    label.append(control);
    return { label, control };
}

function addScenarioDraft() {
    const editor = createScenarioEditor();
    elements.scenarioEditorList.append(editor);
    editor.scrollIntoView({ behavior: "smooth", block: "start" });
    editor.querySelector("input").focus();
}

async function createScenario(article, nameInput) {
    if (!nameInput.value.trim()) {
        setCreationStatus("请先填写场景名称", true);
        nameInput.focus();
        return;
    }
    await runCreation(async () => {
        const scenario = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/scenarios`, {
            method: "POST",
            body: JSON.stringify({ name: nameInput.value }),
        });
        article.replaceWith(createScenarioEditor(scenario));
        adjustSelectedSourceCounts(1, 0);
        setCreationStatus("场景已创建");
    }, "正在创建场景……");
}

async function renameScenario(article, name) {
    await runCreation(async () => {
        const scenario = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/scenarios/${encodeURIComponent(article.dataset.name)}/rename`, {
            method: "POST",
            body: JSON.stringify({ name }),
        });
        article.dataset.name = scenario.name;
        article.querySelector(".scenario-editor-head strong").textContent = scenario.name;
        article.querySelector(".scenario-editor-head .muted").textContent = scenario.path;
        article.querySelector("input").value = scenario.name;
        setCreationStatus("场景名称已更新");
    }, "正在改名……");
}

async function saveScenario(article, description, opening) {
    await runCreation(async () => {
        const scenario = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/scenarios/${encodeURIComponent(article.dataset.name)}`, {
            method: "PUT",
            body: JSON.stringify({ description: description.value, opening: opening.value }),
        });
        description.value = scenario.description;
        opening.value = scenario.opening;
        setCreationStatus("场景内容已保存");
    }, "正在保存场景……");
}

function deleteScenario(article) {
    const name = article.dataset.name;
    confirmDeletion(`确认删除场景“${name}”？`, async () => {
        await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/scenarios/${encodeURIComponent(name)}`, { method: "DELETE" });
        article.remove();
        if (!elements.scenarioEditorList.children.length) elements.scenarioEditorList.append(createScenarioEditor());
        adjustSelectedSourceCounts(-1, 0);
        setCreationStatus("场景已删除");
    });
}

async function loadEntities() {
    elements.entityList.replaceChildren(empty("正在加载实体……"));
    const entities = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/entities`);
    elements.entityList.replaceChildren();
    if (!entities.length) elements.entityList.append(empty("还没有实体"));
    for (const entity of entities) {
        elements.entityList.append(createEntityListButton(entity));
    }
    state.selectedEntityName = null;
    clearEntityEditor();
}

function createEntityListButton(entity, selected = false, onSelect = () => selectEntity(entity.name)) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "entity-item";
    button.dataset.name = entity.name;
    button.dataset.path = entity.path;
    button.classList.toggle("selected", selected);
    button.setAttribute("aria-pressed", String(selected));
    const name = document.createElement("strong");
    name.textContent = entity.name;
    const meta = document.createElement("span");
    const origin = entity.path.startsWith("mods/") ? `模组：${entity.path.split("/")[1]}` : "世界";
    meta.textContent = `${entityTypeLabel(entity.type)} · ${origin}`;
    button.title = entity.path;
    button.append(name, meta);
    button.addEventListener("click", async () => {
        await onSelect();
        if (button.classList.contains("selected")) revealMobileEditor(button.closest(".entity-layout")?.querySelector(".entity-editor"));
    });
    return button;
}

function entityTypeLabel(type) {
    return { character: "角色", location: "地点", event: "事件", item: "物品", organization: "组织", rule: "规则", player_trait: "玩家特质", goal: "目标" }[type] || type;
}

function clearEntityEditor() {
    elements.entityEditor.classList.add("hidden");
    $("#entity-editor-empty").classList.remove("hidden");
}

async function selectEntity(name) {
    if ((name === state.selectedEntityName && !elements.entityEditor.classList.contains("hidden")) || state.creationBusy) return;
    await runCreation(async () => {
        clearEntityEditor();
        await loadEntity(name);
        state.selectedEntityName = name;
        for (const button of elements.entityList.querySelectorAll(".entity-item")) {
            const selected = button.dataset.name === name;
            button.classList.toggle("selected", selected);
            button.setAttribute("aria-pressed", String(selected));
        }
        setCreationStatus("");
    }, "正在加载实体……");
}

async function loadEntity(name) {
    const entity = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/entities/${encodeURIComponent(name)}`);
    renderEntity(entity);
}

function renderEntity(entity) {
    $("#entity-name").value = entity.name;
    $("#entity-type").value = entity.type;
    $("#entity-description").value = entity.description;
    $("#entity-aliases").value = entity.aliases.join(", ");
    $("#entity-required").checked = entity.required;
    $("#entity-content").value = entity.content;
    $("#entity-editor-empty").classList.add("hidden");
    elements.entityEditor.classList.remove("hidden");
}

function entityPayload() {
    return {
        description: $("#entity-description").value,
        aliases: $("#entity-aliases").value.split(",").map((alias) => alias.trim()).filter(Boolean),
        visibility: "world_truth",
        required: $("#entity-required").checked,
        content: $("#entity-content").value,
    };
}

async function saveEntity(event) {
    event.preventDefault();
    if (!state.selectedEntityName) return;
    await runCreation(async () => {
        const entity = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/entities/${encodeURIComponent(state.selectedEntityName)}`, {
            method: "PUT",
            body: JSON.stringify(entityPayload()),
        });
        $("#entity-description").value = entity.description;
        $("#entity-aliases").value = entity.aliases.join(", ");
        $("#entity-required").checked = entity.required;
        $("#entity-content").value = entity.content;
        setCreationStatus("实体内容已保存");
    }, "正在保存实体……");
}

async function moveEntity(event) {
    event.preventDefault();
    if (!state.selectedEntityName) return;
    await runCreation(async () => {
        const entity = await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/entities/${encodeURIComponent(state.selectedEntityName)}/rename`, {
            method: "POST",
            body: JSON.stringify({ name: $("#entity-name").value, type: $("#entity-type").value }),
        });
        const selected = elements.entityList.querySelector(".entity-item.selected");
        state.selectedEntityName = entity.name;
        if (selected) {
            selected.replaceWith(createEntityListButton(entity, true));
        }
        $("#entity-name").value = entity.name;
        $("#entity-type").value = entity.type;
        setCreationStatus("实体名称与类型已更新");
    }, "正在更新名称与类型……");
}

function openCreateEntityDialog() {
    setControlsDisabled(elements.createEntityForm, false);
    elements.createEntityDialog.dataset.kind = state.creationKind;
    elements.createEntityDialog.dataset.source = state.selectedCreationSourceName || "";
    $("#create-entity-name").value = "";
    $("#create-entity-type").value = "character";
    $("#create-entity-error").textContent = "";
    $("#create-entity-error").classList.add("hidden");
    elements.createEntityDialog.showModal();
    $("#create-entity-name").focus();
}

async function createEntity(event) {
    if (event.submitter?.value !== "create") return;
    event.preventDefault();
    const requestedKind = elements.createEntityDialog.dataset.kind;
    const requestedSource = elements.createEntityDialog.dataset.source;
    const payload = { name: $("#create-entity-name").value, type: $("#create-entity-type").value };
    setCreationBusy(true);
    setControlsDisabled(elements.createEntityForm, true);
    try {
        const entity = await api(`${fixedCreationBase(requestedKind, requestedSource)}/entities`, { method: "POST", body: JSON.stringify(payload) });
        elements.createEntityDialog.close();
        state.selectedEntityName = entity.name;
        for (const button of elements.entityList.querySelectorAll(".entity-item")) {
            button.classList.remove("selected");
            button.setAttribute("aria-pressed", "false");
        }
        const emptyItem = elements.entityList.querySelector(".empty");
        if (emptyItem) emptyItem.remove();
        elements.entityList.append(createEntityListButton(entity, true));
        renderEntity(entity);
        adjustSelectedSourceCounts(0, 1);
        setCreationStatus("实体已创建");
    } catch (error) {
        $("#create-entity-error").textContent = error.message;
        $("#create-entity-error").classList.remove("hidden");
    } finally {
        setControlsDisabled(elements.createEntityForm, false);
        setCreationBusy(false);
    }
}

function deleteCurrentEntity() {
    if (!state.selectedEntityName) return;
    const name = state.selectedEntityName;
    confirmDeletion(`确认删除实体“${name}”？`, async () => {
        await api(`${fixedCreationBase(state.creationKind, state.selectedCreationSourceName)}/entities/${encodeURIComponent(name)}`, { method: "DELETE" });
        const selected = elements.entityList.querySelector(".entity-item.selected");
        if (selected) selected.remove();
        state.selectedEntityName = null;
        if (!elements.entityList.children.length) elements.entityList.append(empty("还没有实体"));
        clearEntityEditor();
        adjustSelectedSourceCounts(0, -1);
        setCreationStatus("实体已删除");
    });
}

function saveEntityType(path) {
    const parts = path.split("/");
    return parts[0] === "entities" ? parts[1] : parts[3];
}

function saveEntityUrl(path, suffix = "") {
    return `/api/saves/${state.save.id}/entity${suffix}?path=${encodeURIComponent(path)}`;
}

function renderBackendSection() {
    for (const button of document.querySelectorAll("[data-backend-section]")) {
        const selected = button.dataset.backendSection === state.backendSection;
        button.classList.toggle("active", selected);
        button.setAttribute("aria-pressed", String(selected));
    }
    $("#backend-trace-panel").classList.toggle("hidden", state.backendSection !== "trace");
    $("#backend-entities-panel").classList.toggle("hidden", state.backendSection !== "entities");
}

async function showBackendSection(section) {
    if (state.saveEntityBusy) return;
    state.backendSection = section;
    renderBackendSection();
    if (section === "entities") await runSaveEntity(loadSaveEntities, "正在加载存档实体……");
}

function setSaveEntityStatus(message, error = false) {
    elements.saveEntityStatus.textContent = message;
    elements.saveEntityStatus.classList.toggle("error-text", error);
}

function updateSaveEntityControls() {
    const locked = state.running || state.saveEntityBusy;
    setControlsDisabled(elements.saveEntityIdentityForm, locked);
    setControlsDisabled(elements.saveEntityContentForm, locked);
    $("#new-save-entity-button").disabled = locked;
    $("#create-save-entity-submit").disabled = locked;
    for (const button of elements.saveEntityList.querySelectorAll("button")) button.disabled = state.saveEntityBusy;
}

async function runSaveEntity(action, loadingMessage = "") {
    if (state.saveEntityBusy) return;
    state.saveEntityBusy = true;
    updateSaveEntityControls();
    if (loadingMessage) setSaveEntityStatus(loadingMessage);
    try {
        await action();
    } catch (error) {
        setSaveEntityStatus(error.message, true);
    } finally {
        state.saveEntityBusy = false;
        updateSaveEntityControls();
    }
}

function clearSaveEntityEditor() {
    state.selectedSaveEntity = null;
    elements.saveEntityEditor.classList.add("hidden");
    $("#save-entity-empty").classList.remove("hidden");
}

function renderSaveEntity(entity) {
    state.selectedSaveEntity = entity;
    $("#save-entity-path").textContent = entity.path;
    $("#save-entity-name").value = entity.name;
    $("#save-entity-type").value = entity.type;
    $("#save-entity-description").value = entity.description;
    $("#save-entity-aliases").value = entity.aliases.join(", ");
    $("#save-entity-required").checked = entity.required;
    $("#save-entity-content").value = entity.content;
    $("#save-entity-empty").classList.add("hidden");
    elements.saveEntityEditor.classList.remove("hidden");
    for (const button of elements.saveEntityList.querySelectorAll(".entity-item")) {
        const selected = button.dataset.path === entity.path;
        button.classList.toggle("selected", selected);
        button.setAttribute("aria-pressed", String(selected));
    }
    updateSaveEntityControls();
}

async function loadSaveEntity(path) {
    const entity = await api(saveEntityUrl(path));
    renderSaveEntity(entity);
}

async function loadSaveEntities() {
    const saveId = state.save.id;
    const entities = await api(`/api/saves/${saveId}/entities`);
    if (state.save?.id !== saveId) return;
    elements.saveEntityList.replaceChildren();
    clearSaveEntityEditor();
    if (!entities.length) elements.saveEntityList.append(empty("还没有实体"));
    for (const entity of entities) {
        elements.saveEntityList.append(createEntityListButton(
            { ...entity, type: saveEntityType(entity.path) }, false, () => selectSaveEntity(entity.path),
        ));
    }
    setSaveEntityStatus("");
    updateSaveEntityControls();
}

async function selectSaveEntity(path) {
    if (state.saveEntityBusy) return;
    await runSaveEntity(async () => {
        await loadSaveEntity(path);
        setSaveEntityStatus("");
    }, "正在加载实体……");
}

function openCreateSaveEntityDialog() {
    if (state.running || state.saveEntityBusy) return;
    $("#create-save-entity-name").value = "";
    $("#create-save-entity-type").value = "character";
    $("#create-save-entity-error").classList.add("hidden");
    elements.createSaveEntityDialog.showModal();
    $("#create-save-entity-name").focus();
}

async function createSaveEntity(event) {
    if (event.submitter?.value !== "create") return;
    event.preventDefault();
    if (state.saveEntityBusy || state.running) return;
    state.saveEntityBusy = true;
    updateSaveEntityControls();
    try {
        const entity = await api(`/api/saves/${state.save.id}/entities`, {
            method: "POST",
            body: JSON.stringify({ name: $("#create-save-entity-name").value, type: $("#create-save-entity-type").value }),
        });
        elements.createSaveEntityDialog.close();
        await loadSaveEntities();
        await loadSaveEntity(entity.path);
        setSaveEntityStatus("实体已创建");
    } catch (error) {
        $("#create-save-entity-error").textContent = error.message;
        $("#create-save-entity-error").classList.remove("hidden");
    } finally {
        state.saveEntityBusy = false;
        updateSaveEntityControls();
    }
}

async function saveSaveEntity(event) {
    event.preventDefault();
    if (!state.selectedSaveEntity || state.running) return;
    await runSaveEntity(async () => {
        const current = state.selectedSaveEntity;
        const entity = await api(saveEntityUrl(current.path), {
            method: "PUT",
            body: JSON.stringify({
                revision: current.revision,
                description: $("#save-entity-description").value,
                aliases: $("#save-entity-aliases").value.split(",").map((alias) => alias.trim()).filter(Boolean),
                visibility: "world_truth",
                required: $("#save-entity-required").checked,
                content: $("#save-entity-content").value,
            }),
        });
        renderSaveEntity(entity);
        setSaveEntityStatus("实体内容已保存");
    }, "正在保存实体……");
}

async function moveSaveEntity(event) {
    event.preventDefault();
    if (!state.selectedSaveEntity || state.running) return;
    await runSaveEntity(async () => {
        const current = state.selectedSaveEntity;
        const entity = await api(saveEntityUrl(current.path, "/rename"), {
            method: "POST",
            body: JSON.stringify({
                revision: current.revision,
                name: $("#save-entity-name").value,
                type: $("#save-entity-type").value,
            }),
        });
        await loadSaveEntities();
        renderSaveEntity(entity);
        setSaveEntityStatus("实体名称与类型已更新");
    }, "正在更新名称与类型……");
}

function deleteCurrentSaveEntity() {
    if (!state.selectedSaveEntity || state.running || state.saveEntityBusy) return;
    const { path, revision, name } = state.selectedSaveEntity;
    confirmDeletion(`确认删除当前存档中的实体“${name}”（${path}）？`, async () => {
        await api(saveEntityUrl(path), { method: "DELETE", body: JSON.stringify({ revision }) });
        await loadSaveEntities();
        setSaveEntityStatus("实体已删除");
    });
}

function formatTime(value) {
    return new Date(value).toLocaleString("zh-CN", { hour12: false });
}

elements.adventureButton.addEventListener("click", returnToMenu);
elements.foregroundButton.addEventListener("click", () => setView("foreground"));
elements.backendButton.addEventListener("click", async () => {
    setView("backend");
    if (state.backendSection === "entities") await showBackendSection("entities");
});
for (const button of document.querySelectorAll("[data-backend-section]")) {
    button.addEventListener("click", () => showBackendSection(button.dataset.backendSection));
}
$("#new-save-entity-button").addEventListener("click", openCreateSaveEntityDialog);
elements.createSaveEntityForm.addEventListener("submit", createSaveEntity);
elements.saveEntityIdentityForm.addEventListener("submit", moveSaveEntity);
elements.saveEntityContentForm.addEventListener("submit", saveSaveEntity);
$("#delete-save-entity-button").addEventListener("click", deleteCurrentSaveEntity);
elements.creationButton.addEventListener("click", showCreation);
elements.settingsButton.addEventListener("click", showSettings);
elements.turnForm.addEventListener("submit", submitTurn);
elements.turnNavigationToggle.addEventListener("click", () => {
    const open = elements.turnNavigation.classList.toggle("open");
    elements.turnNavigationToggle.setAttribute("aria-expanded", String(open));
});
elements.turnNavigationLatest.addEventListener("click", () => {
    state.followLatestTurn = true;
    closeTurnNavigation();
    scrollToStoryEnd();
});
document.addEventListener("click", (event) => {
    if (!elements.turnNavigation.contains(event.target)) closeTurnNavigation();
});
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeTurnNavigation();
});
let turnNavigationFrame = 0;
window.addEventListener("scroll", () => {
    if (turnNavigationFrame || !state.save || elements.foreground.classList.contains("hidden")) return;
    turnNavigationFrame = requestAnimationFrame(() => {
        turnNavigationFrame = 0;
        state.followLatestTurn = window.innerHeight + window.scrollY >= document.documentElement.scrollHeight - 120;
        syncTurnNavigation();
    });
}, { passive: true });
window.addEventListener("resize", () => requestAnimationFrame(() => {
    syncTurnNavigation();
    if (state.save) updateTurnNavigationToggle();
}));
elements.addImageButton.addEventListener("click", () => elements.imageInput.click());
elements.imageInput.addEventListener("change", () => {
    addPendingImages(elements.imageInput.files);
    elements.imageInput.value = "";
});
elements.closeImagePreview.addEventListener("click", () => elements.imagePreviewDialog.close());
elements.imagePreviewDialog.addEventListener("close", () => {
    elements.imagePreview.removeAttribute("src");
    elements.imagePreview.alt = "图片预览";
});
elements.abortTurnButton.addEventListener("click", abortRunningTurn);
elements.retryTurnButton.addEventListener("click", retryResumableTurn);
elements.previousSavePage.addEventListener("click", () => loadLauncher(state.savePage - 1));
elements.nextSavePage.addEventListener("click", () => loadLauncher(state.savePage + 1));
elements.previousEventPage.addEventListener("click", () => changeEventPage(state.eventPage.page - 1));
elements.nextEventPage.addEventListener("click", () => changeEventPage(state.eventPage.page + 1));
elements.presetForm.addEventListener("submit", savePreset);
elements.drawingPresetForm.addEventListener("submit", saveDrawingPreset);
elements.capabilitySettingsForm.addEventListener("submit", saveCapabilitySettings);
elements.capabilitySettingsForm.addEventListener("input", () => { state.capabilitySettingsDirty = true; });
document.querySelectorAll("[data-capability-panel]").forEach((button) => button.addEventListener("click", () => selectCapabilityPanel(button)));
elements.runtimeSettingsForm.addEventListener("submit", saveRuntimeSettings);
$("#ai-fallback-form").addEventListener("submit", saveAiFallbackSettings);
elements.networkSettingsForm.addEventListener("submit", saveNetworkSettings);
$("#network-mode").addEventListener("change", updateNetworkFields);
elements.deleteConfirmationForm.addEventListener("submit", async (event) => {
    if (event.submitter?.value !== "confirm") return;
    event.preventDefault();
    if (!state.deleteAction) return;
    elements.confirmDeleteButton.disabled = true;
    try {
        await state.deleteAction();
        elements.deleteConfirmation.close();
    } catch (error) {
        elements.deleteConfirmationError.textContent = error.message;
        elements.deleteConfirmationError.classList.remove("hidden");
    } finally {
        elements.confirmDeleteButton.disabled = false;
    }
});
elements.deleteConfirmation.addEventListener("close", () => { state.deleteAction = null; });
elements.renameSaveForm.addEventListener("submit", renameSave);
elements.branchSaveForm.addEventListener("submit", createBranchSave);
elements.renameSaveDialog.addEventListener("close", () => { state.renameSave = null; });
elements.createSaveForm.addEventListener("submit", createSave);
elements.createSaveDialog.addEventListener("close", () => { state.newAdventure = null; });
elements.nextAdventureStep.addEventListener("click", showScenarioStep);
elements.previousAdventureStep.addEventListener("click", showModStep);
elements.previousModPage.addEventListener("click", () => loadModPage(state.newAdventure.modPage - 1));
elements.nextModPage.addEventListener("click", () => loadModPage(state.newAdventure.modPage + 1));
elements.previousScenarioPage.addEventListener("click", () => {
    if (!state.newAdventure) return;
    state.newAdventure.scenarioPage -= 1;
    renderScenarioPage();
});
elements.nextScenarioPage.addEventListener("click", () => {
    if (!state.newAdventure) return;
    state.newAdventure.scenarioPage += 1;
    renderScenarioPage();
});
elements.presetForm.addEventListener("input", (event) => {
    state.presetDirty = true;
    if (["provider", "base-url", "api-key"].includes(event.target.id)) resetModelOptions();
    if (event.target.id === "model") updateGeminiThinkingFields(undefined, true);
});
$("#provider").addEventListener("change", () => {
    state.presetDirty = true;
    updateProviderFields(true);
});
$("#new-preset-button").addEventListener("click", startNewPreset);
$("#duplicate-preset-button").addEventListener("click", duplicatePreset);
$("#delete-preset-button").addEventListener("click", deletePreset);
$("#load-models-button").addEventListener("click", loadModels);
$("#model-select").addEventListener("change", (event) => {
    if (!event.target.value) return;
    $("#model").value = event.target.value;
    const levelReset = updateGeminiThinkingFields(undefined, true);
    state.presetDirty = true;
    if (!levelReset) setPresetStatus(`已选择模型 ${event.target.value}，请保存 Preset`);
});
$("#test-preset-button").addEventListener("click", testPreset);
for (const button of document.querySelectorAll("[data-settings-section]")) button.addEventListener("click", () => showSettingsSection(button.dataset.settingsSection));
elements.drawingPresetForm.addEventListener("input", () => { state.drawingPresetDirty = true; });
$("#drawing-provider").addEventListener("change", () => {
    state.drawingPresetDirty = true;
    $("#drawing-base-url").value = $("#drawing-provider").value === "novelai" ? "https://api.novelai.net" : "http://127.0.0.1:7860";
    $("#drawing-auth-username").value = "";
    $("#drawing-auth-password").value = "";
    $("#drawing-api-key").value = "";
    $("#drawing-auth-status").textContent = "未设置";
    $("#drawing-api-key-status").textContent = "未设置";
    updateDrawingProviderFields();
});
$("#new-drawing-preset-button").addEventListener("click", startNewDrawingPreset);
$("#test-drawing-preset-button").addEventListener("click", testDrawingPreset);
$("#generate-drawing-test-button").addEventListener("click", generateDrawingTest);
$("#load-capability-resources-button").addEventListener("click", loadCapabilityResources);
$("#set-default-drawing-preset-button").addEventListener("click", async () => {
    if (!state.selectedDrawingPresetId) return;
    try { state.drawingPresets = await api("/api/drawing-preset-settings", { method: "PUT", body: JSON.stringify({ preset_id: state.selectedDrawingPresetId }) }); selectDrawingPreset(state.selectedDrawingPresetId); setDrawingStatus("已设为默认绘画 Preset"); } catch (error) { setDrawingStatus(error.message, true); }
});
$("#duplicate-drawing-preset-button").addEventListener("click", async () => {
    const current = state.drawingPresets?.presets.find((preset) => preset.id === state.selectedDrawingPresetId);
    const name = current && window.prompt("复制后的绘画 Preset 名称", `${current.name} 副本`);
    if (!name?.trim()) return;
    try { state.drawingPresets = await api(`/api/drawing-presets/${current.id}/duplicate`, { method: "POST", body: JSON.stringify({ name: name.trim() }) }); state.selectedDrawingPresetId = state.drawingPresets.presets.at(-1).id; selectDrawingPreset(state.selectedDrawingPresetId); } catch (error) { setDrawingStatus(error.message, true); }
});
$("#delete-drawing-preset-button").addEventListener("click", async () => {
    if (!state.selectedDrawingPresetId) return;
    const id = state.selectedDrawingPresetId;
    if (!window.confirm("确认删除当前绘画 Preset？")) return;
    try { state.drawingPresets = await api(`/api/drawing-presets/${id}`, { method: "DELETE" }); state.selectedDrawingPresetId = state.drawingPresets.default_preset_id; if (state.selectedDrawingPresetId) selectDrawingPreset(state.selectedDrawingPresetId); else startNewDrawingPreset(); setDrawingStatus("绘画 Preset 已删除"); } catch (error) { setDrawingStatus(error.message, true); }
});
for (const button of document.querySelectorAll("[data-creation-kind]")) button.addEventListener("click", () => selectCreationKind(button.dataset.creationKind));
for (const button of document.querySelectorAll("[data-creation-section]")) button.addEventListener("click", () => selectCreationSection(button.dataset.creationSection));
elements.sourceRenameForm.addEventListener("submit", renameSource);
elements.sourceContentForm.addEventListener("submit", saveSourceContent);
elements.styleRenameForm.addEventListener("submit", renameSource);
elements.styleContentForm.addEventListener("submit", saveSourceContent);
$("#new-source-button").addEventListener("click", openCreateSourceDialog);
$("#delete-source-button").addEventListener("click", deleteCurrentSource);
$("#delete-style-button").addEventListener("click", deleteCurrentSource);
elements.toggleStyleEnabledButton.addEventListener("click", toggleCurrentStyleEnabled);
elements.createSourceForm.addEventListener("submit", createSource);
$("#new-scenario-button").addEventListener("click", addScenarioDraft);
$("#new-entity-button").addEventListener("click", openCreateEntityDialog);
elements.createEntityForm.addEventListener("submit", createEntity);
elements.entityIdentityForm.addEventListener("submit", moveEntity);
elements.entityContentForm.addEventListener("submit", saveEntity);
$("#delete-entity-button").addEventListener("click", deleteCurrentEntity);
async function returnToMenu() {
    if (state.source) state.source.close();
    clearPendingImages();
    state.source = null;
    state.save = null;
    state.eventPage = null;
    elements.connection.textContent = "未连接";
    elements.connection.className = "connection";
    elements.workspace.classList.add("hidden");
    elements.foregroundButton.classList.add("hidden");
    elements.backendButton.classList.add("hidden");
    showAdventure();
    state.savePage = 1;
    await loadLauncher();
}
Promise.all([loadLauncher(), loadPresets(), loadRuntimeSettings(), loadAgentPresetSettings(), loadDrawingPresets(), loadCapabilitySettings(), loadNetworkSettings()]).catch((error) => {
    elements.worldList.replaceChildren(empty(`启动失败：${error.message}`));
});
