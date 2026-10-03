import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

class Element {
    children = []; style = {}; dataset = {}; listeners = {}; textContent = "";
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    setAttribute() {}
    addEventListener(name, callback) { this.listeners[name] = callback; }
    click() { if (!this.disabled) this.listeners.click?.(); }
    querySelectorAll() {
        const visit = (element) => [element, ...element.children.flatMap(visit)];
        return this.children.flatMap(visit).filter((element) => element.type === "checkbox" && element.checked);
    }
}
const source = readFileSync(new URL("../web/source-picker.js", import.meta.url), "utf8")
    .replace(/^import .*;\s*/gm, "").replace("export function", "function");
const context = { app: { registerExtension() {} }, document: { createElement() { return new Element(); } } };
vm.createContext(context);
vm.runInContext(source, context);
let changes = 0;
const node = { graph: { change() { changes++; } }, setDirtyCanvas() {} };
const selection = { value: '["keep","missing"]' };
const picker = context.makePicker(node, selection, async () => []);
const [controls, status, grid] = picker.root.children;
const button = (text) => controls.children.find((element) => element.textContent === text);
assert.equal(button("Keep available selection").disabled, true);
picker.setSources([
    { source_id: "keep", label: "Alice1", strength: 1, tokens: 6, thumbnail: "synthetic" },
    { source_id: "new", label: "Alice2", strength: 1, tokens: 6, thumbnail: "synthetic" },
]);
assert.match(status.textContent, /Upstream sources changed/);
assert.equal(selection.value, '["keep","missing"]'); // Rendering never guesses a replacement.
button("Keep available selection").click();
assert.deepEqual(JSON.parse(selection.value), ["keep"]);
assert.doesNotMatch(status.textContent, /Upstream sources changed/);
assert.equal(grid.children[1].children[2].checked, false);
button("Select all").click();
assert.equal(selection.value, "all");
assert.equal(grid.children[1].children[2].checked, true);
picker.reset();
assert.equal(button("Keep available selection").disabled, true);
assert.equal(selection.value, "all");
selection.value = '["missing"]';
picker.setSources([{ source_id: "new", label: "Alice1", strength: 1, tokens: 6, thumbnail: "synthetic" }]);
button("Keep available selection").click();
assert.deepEqual(JSON.parse(selection.value), []); // Explicitly retaining no survivors is valid.
assert.ok(changes >= 3);
console.log("Source-picker stale selection recovery tests passed.");
