import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function inputPlan(node, seen = new Set()) {
    if (!node || seen.has(node.id)) throw new Error("Connect a saved RefMod Load node; source preview cannot follow a cycle.");
    const next = new Set(seen).add(node.id);
    const widget = (name) => node.widgets?.find((item) => item.name === name)?.value;
    const upstream = (name) => {
        const input = node.inputs?.find((item) => item.name === name);
        const link = app.graph.links[input?.link];
        return app.graph.getNodeById(link?.origin_id);
    };
    if (node.type === "H3RefModLabLoad") return { type: "load", filename: widget("filename") };
    if (node.type === "H3RefModLabCombineV2") {
        const weights = JSON.parse(widget("strengths") || "{}");
        const inputs = node.inputs.filter((input) => input.link != null && /^refmods\.refmod_\d+$/.test(input.name));
        inputs.sort((a, b) => Number(a.name.split("_").at(-1)) - Number(b.name.split("_").at(-1)));
        return { type: "combine", inputs: inputs.map((input) => ({
            input: inputPlan(upstream(input.name), next),
            strength: weights[input.name.split(".").at(-1)] ?? 1,
        })) };
    }
    if (node.type === "H3RefModLabSetInstructions") return {
        type: "instructions", input: inputPlan(upstream("refmod"), next),
        ...Object.fromEntries(["description", "subject_name", "retention_strategy", "retention_details", "audio_retention_strategy", "audio_retention_details"].map((name) => [name, widget(name)])),
    };
    if (node.type === "H3RefModLabSelectSources") return {
        type: "select", input: inputPlan(upstream("refmod"), next), selection: widget("selection"),
    };
    throw new Error("Refresh sources needs a saved pack through Load, Combine or Select Sources. For an in-memory pack, queue this selector alone first.");
}

export function makePicker(node, selection, refresh) {
    const root = document.createElement("div");
    root.style.cssText = "padding:8px;box-sizing:border-box;overflow:auto;height:100%;color:var(--input-text,#eee);background:var(--comfy-input-bg,#222);font:12px sans-serif";
    const controls = document.createElement("div");
    controls.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;position:sticky;top:0;background:inherit;padding-bottom:8px;z-index:1";
    const status = document.createElement("div");
    status.style.cssText = "padding:4px 0 8px;white-space:pre-wrap";
    const grid = document.createElement("div");
    grid.style.cssText = "display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px";
    let sources = [];
    let catalogReady = false;
    const changed = () => { node.graph?.change(); node.setDirtyCanvas?.(true, true); };
    const chosen = () => selection.value === "all" ? new Set(sources.map((s) => s.source_id)) : new Set(JSON.parse(selection.value));
    const updateStatus = () => {
        try {
            const selected = chosen();
            const included = sources.filter((source) => selected.has(source.source_id));
            const active = included.filter((source) => source.strength > 0);
            const tokens = active.reduce((total, source) => total + source.tokens, 0);
            const missing = [...selected].filter((id) => !sources.some((source) => source.source_id === id));
            status.textContent = `${included.length}/${sources.length} selected · ${tokens.toLocaleString()} active DiT tokens\nPicture / Video / Audio numbers change after selection; check Inspect after Combine.${missing.length ? "\nUpstream sources changed. Downstream outputs are paused until you choose sources again, Keep available selection, or Select all, then run again." : ""}`;
        } catch {
            status.textContent = "Invalid saved selection. Choose Select all or Clear all to reset.";
        }
    };
    const render = () => {
        keepAvailable.disabled = !catalogReady;
        grid.replaceChildren();
        let selected;
        try { selected = chosen(); } catch { selected = new Set(); }
        for (const source of sources) {
            const card = document.createElement("label");
            card.style.cssText = "display:flex;flex-direction:column;gap:4px;border:1px solid #666;border-radius:5px;padding:5px;cursor:pointer;overflow:hidden";
            const image = document.createElement("img");
            image.src = source.thumbnail;
            image.alt = source.label;
            image.style.cssText = "width:100%;aspect-ratio:1;object-fit:contain";
            const caption = document.createElement("span");
            caption.textContent = source.label;
            const check = document.createElement("input");
            check.type = "checkbox";
            check.checked = selected.has(source.source_id);
            check.dataset.sourceId = source.source_id;
            check.setAttribute("aria-label", source.label);
            check.addEventListener("change", () => {
                selection.value = JSON.stringify([...grid.querySelectorAll("input:checked")].map((input) => input.dataset.sourceId));
                updateStatus(); changed();
            });
            const cost = document.createElement("span");
            cost.textContent = `${source.tokens.toLocaleString()} tokens${source.strength === 0 ? " · strength 0 (inactive)" : ""}`;
            card.append(image, caption, check, cost);
            grid.append(card);
        }
        updateStatus();
    };
    const button = (label, action) => {
        const element = document.createElement("button");
        element.textContent = label;
        element.type = "button";
        element.addEventListener("click", action);
        controls.append(element);
        return element;
    };
    const refreshButton = button("Refresh sources", async () => {
        refreshButton.disabled = true;
        status.textContent = "Loading source thumbnails…";
        try { sources = await refresh(); catalogReady = true; render(); }
        catch (error) { status.textContent = error.message; grid.replaceChildren(); }
        finally { refreshButton.disabled = false; }
    });
    button("Select all", () => { selection.value = "all"; render(); changed(); });
    button("Clear all", () => { selection.value = "[]"; render(); changed(); });
    const keepAvailable = button("Keep available selection", () => {
        try { const selected = chosen(); selection.value = JSON.stringify(sources.filter((source) => selected.has(source.source_id)).map((source) => source.source_id)); render(); changed(); }
        catch { status.textContent = "Invalid saved selection. Choose Select all or Clear all to reset."; }
    });
    keepAvailable.disabled = true;
    root.append(controls, status, grid);
    status.textContent = "Connect Load RefMod and click Refresh sources. All sources start enabled.";
    return { root, setSources(value) { sources = value; catalogReady = true; render(); }, reset() { sources = []; catalogReady = false; keepAvailable.disabled = true; grid.replaceChildren(); status.textContent = "Input changed. Click Refresh sources before generating."; } };
}

app.registerExtension({
    name: "H3RefModsLab.SourcePicker",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3RefModLabSelectSources") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const selection = this.widgets.find((widget) => widget.name === "selection");
            selection.type = "converted-widget";
            selection.draw = () => {};
            selection.computeSize = () => [0, -4];
            selection.serializeValue = () => selection.value;
            this.sourcePicker = makePicker(this, selection, async () => {
                const input = this.inputs.find((item) => item.name === "refmod");
                const link = app.graph.links[input?.link];
                const plan = inputPlan(app.graph.getNodeById(link?.origin_id));
                const response = await api.fetchApi("/h3-refmods-lab/sources", {
                    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(plan),
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || "Could not preview sources.");
                return data.sources;
            });
            this.addDOMWidget("source_picker", "custom", this.sourcePicker.root, { serialize: false, hideOnZoom: false,
                getMinHeight: () => 300, getMaxHeight: () => 600 });
            this.setSize([420, 500]);
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            if (message.sources) this.sourcePicker?.setSources(message.sources);
        };
        const connected = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            connected?.apply(this, arguments);
            this.sourcePicker?.reset();
        };
    },
});
