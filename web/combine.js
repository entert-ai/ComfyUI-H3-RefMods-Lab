import { app } from "../../../scripts/app.js";

// Native Autogrow owns socket creation. Store paired strength controls in one
// schema-backed widget so API graphs and saved workflows agree.
app.registerExtension({
    name: "H3RefModsLab.Combine",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3RefModLabCombineV2") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            const weights = this.widgets.find((widget) => widget.name === "strengths");
            weights.type = "converted-widget";
            weights.draw = () => {};
            weights.computeSize = () => [0, -4];
            weights.serializeValue = () => weights.value;
            const root = document.createElement("div");
            root.style.cssText = "padding:8px;color:var(--input-text,#eee);font:12px sans-serif;max-height:300px;overflow:auto";
            this.combineStrengths = {
                refresh: () => {
                    root.replaceChildren();
                    let values;
                    try {
                        values = JSON.parse(weights.value || "{}");
                        if (!values || Array.isArray(values) || typeof values !== "object") throw new Error("Invalid strengths");
                    } catch {
                        root.textContent = "Invalid strengths. ";
                        const reset = document.createElement("button");
                        reset.textContent = "Reset strengths";
                        reset.onclick = () => { weights.value = "{}"; this.combineStrengths.refresh(); this.graph?.change(); };
                        root.append(reset);
                        return;
                    }
                    const connected = (this.inputs || []).filter((input) => input.link != null && /^refmods\.refmod_\d+$/.test(input.name))
                        .sort((a, b) => Number(a.name.split("_").at(-1)) - Number(b.name.split("_").at(-1)));
                    for (const input of connected) {
                        const name = input.name.split(".").at(-1);
                        const row = document.createElement("label");
                        row.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:12px;padding:4px 0";
                        const caption = document.createElement("span");
                        caption.textContent = `RefMod ${Number(name.slice(7))} strength`;
                        const slider = document.createElement("input");
                        slider.type = "range"; slider.min = "0"; slider.max = "2"; slider.step = "0.05";
                        slider.value = values[name] ?? 1;
                        slider.setAttribute("aria-label", `RefMod ${Number(name.slice(7))} strength slider`);
                        slider.style.cssText = "flex:1;min-width:60px";
                        const field = document.createElement("input");
                        field.type = "number"; field.min = "0"; field.max = "2"; field.step = "0.05";
                        field.value = values[name] ?? 1;
                        field.setAttribute("aria-label", `RefMod ${Number(name.slice(7))} strength value`);
                        field.style.cssText = "width:65px;color:inherit;background:var(--comfy-input-bg,#222)";
                        field.addEventListener("input", () => {
                            if (field.value === "") return;
                            const value = Number(field.value);
                            if (!Number.isFinite(value) || value < 0 || value > 2) { field.value = values[name] ?? 1; return; }
                            slider.value = value;
                            values[name] = value;
                            weights.value = JSON.stringify(values);
                            this.graph?.change(); this.setDirtyCanvas?.(true, true);
                        });
                        slider.addEventListener("input", () => { field.value = slider.value; field.dispatchEvent(new Event("input")); });
                        row.append(caption, slider, field); root.append(row);
                    }
                    if (!connected.length) root.textContent = "Connect RefMod 1 to begin.";
                    const reset = document.createElement("button");
                    reset.textContent = "Reset strengths";
                    reset.addEventListener("click", () => { weights.value = "{}"; this.combineStrengths.refresh(); this.graph?.change(); });
                    root.append(reset);
                },
            };
            this.addDOMWidget("combine_strengths", "custom", root, { serialize: false, hideOnZoom: false,
                getMinHeight: () => 80, getMaxHeight: () => 300 });
            this.combineStrengths.refresh();
        };
        const connected = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            connected?.apply(this, arguments);
            // Run after native Autogrow has updated its slots.
            queueMicrotask(() => this.combineStrengths?.refresh());
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            configured?.apply(this, arguments);
            queueMicrotask(() => this.combineStrengths?.refresh());
        };
    },
});
