import { app } from "../../../scripts/app.js";
import { ComfyWidgets } from "../../../scripts/widgets.js";

app.registerExtension({
    name: "H3RefModsLab.Details",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["H3RefModLabInspect", "H3RefModLabSave"].includes(nodeData.name)) return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            created?.apply(this, arguments);
            this.refmodDetails = ComfyWidgets.STRING(this, "details", ["STRING", { multiline: true }], app).widget;
            this.refmodDetails.inputEl.readOnly = true;
            this.refmodDetails.serialize = false;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            executed?.apply(this, arguments);
            if (this.refmodDetails) this.refmodDetails.value = (message.text ?? []).join("\n");
        };
    },
});
