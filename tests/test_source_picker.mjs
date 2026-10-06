import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

class Element {
    children = []; style = {}; dataset = {}; listeners = {}; textContent = "";
    append(...items) { this.children.push(...items); }
    replaceChildren(...items) { this.children = items; }
    setAttribute() {}
    addEventListener(name, callback) { this.listeners[name] = callback; }
    click() { if (!this.disabled) return this.listeners.click?.(); }
    querySelectorAll() {
        const visit = (element) => [element, ...element.children.flatMap(visit)];
        return this.children.flatMap(visit).filter((element) => element.type === "checkbox" && element.checked);
    }
}
const source = readFileSync(new URL("../web/source-picker.js", import.meta.url), "utf8")
    .replace(/^import .*;\s*/gm, "").replace("export function", "function");
const context = { app: { registerExtension(extension) { this.extension = extension; } }, document: { createElement() { return new Element(); } } };
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

// Refresh results must belong to the current input/catalog.
const card = (id) => ({ source_id: id, label: id, strength: 1, tokens: 6, thumbnail: "synthetic" });
const pending = [];
const asyncPicker = context.makePicker(node, { value: "all" }, () => new Promise((resolve, reject) => pending.push({ resolve, reject })));
const [asyncControls, asyncStatus, asyncGrid] = asyncPicker.root.children;
const asyncButton = (text) => asyncControls.children.find((element) => element.textContent === text);
const startRefresh = () => asyncButton("Refresh sources").click();
let oldRefresh = startRefresh();
asyncPicker.reset();
pending[0].resolve([card("obsolete")]);
await oldRefresh;
assert.equal(asyncGrid.children.length, 0);
assert.equal(asyncButton("Keep available selection").disabled, true);
assert.match(asyncStatus.textContent, /Input changed/);
oldRefresh = startRefresh();
asyncPicker.setSources([card("executed")]);
pending[1].resolve([card("obsolete")]);
await oldRefresh;
assert.equal(asyncGrid.children[0].children[1].textContent, "executed");
let failedRefresh = startRefresh();
pending[2].reject(new Error("Preview failed"));
await failedRefresh;
assert.equal(asyncGrid.children.length, 0);
assert.equal(asyncButton("Keep available selection").disabled, true);
asyncButton("Select all").click();
assert.equal(asyncGrid.children.length, 0);
oldRefresh = startRefresh();
asyncPicker.reset();
let newRefresh = startRefresh();
pending[3].reject(new Error("Obsolete failure"));
await oldRefresh;
assert.equal(asyncButton("Refresh sources").disabled, true);
assert.match(asyncStatus.textContent, /Loading/);
pending[4].resolve([card("current")]);
await newRefresh;
assert.equal(asyncGrid.children[0].children[1].textContent, "current");
assert.equal(asyncButton("Refresh sources").disabled, false);

// Graph stubs use the installed frontend's owning graphs and IO slots.
const graph = () => ({ nodes: [], links: {}, subgraphs: new Map(), getNodeById(id) { return this.nodes.find((node) => node.id === id); } });
const rootGraph = graph();
rootGraph.rootGraph = rootGraph;
context.app.graph = context.app.rootGraph = rootGraph;
const add = (graph, id, type, widgets = {}, inputs = []) => {
    const node = { graph, id, type, inputs, widgets: Object.entries(widgets).map(([name, value]) => ({ name, value })) };
    graph.nodes.push(node);
    return node;
};
const connect = (graph, id, origin, target, name, slot = 0) => {
    graph.links[id] = { origin_id: origin.id, origin_slot: slot };
    target.inputs.push({ name, link: id });
};
const load = add(rootGraph, 1, "H3RefModLabLoad", { filename: "root.safetensors" });
const reroute = add(rootGraph, 2, "Reroute");
connect(rootGraph, 1, load, reroute, "");
assert.equal(context.inputPlan(reroute).filename, "root.safetensors");
const inner = graph();
inner.rootGraph = rootGraph;
inner.inputNode = { id: -10 };
rootGraph.subgraphs.set("inner", inner);
const innerLoad = add(inner, 1, "H3RefModLabLoad", { filename: "inner.safetensors" });
const combine = add(inner, 2, "H3RefModLabCombineV2", { strengths: '{"refmod_1":0.5}' });
connect(inner, 1, innerLoad, combine, "refmods.refmod_1");
assert.equal(context.inputPlan(combine).inputs[0].input.filename, "inner.safetensors");
assert.equal(context.inputPlan(combine).inputs[0].strength, 0.5);
const instance = (parent, id, definition, outputLink) => {
    const node = add(parent, id, "subgraph");
    node.subgraph = definition;
    node.resolveSubgraphOutputLink = (slot) => ({ link: definition.links[outputLink[slot]] });
    return node;
};
const outer = instance(rootGraph, 3, inner, [1]);
assert.equal(context.inputPlan(outer).filename, "inner.safetensors");
const insideSelect = add(inner, 3, "H3RefModLabSelectSources", { selection: "[]" });
connect(inner, 2, inner.inputNode, insideSelect, "refmod");
connect(rootGraph, 2, load, outer, "refmod");
assert.equal(context.inputPlan(insideSelect).input.filename, "root.safetensors");
const nested = graph();
nested.rootGraph = rootGraph;
nested.inputNode = { id: -10 };
rootGraph.subgraphs.set("nested", nested);
const nestedSelect = add(nested, 1, "H3RefModLabSelectSources", { selection: "all" });
connect(nested, 1, nested.inputNode, nestedSelect, "refmod");
const nestedInstance = instance(inner, 4, nested, [1]);
connect(inner, 3, inner.inputNode, nestedInstance, "refmod");
assert.equal(context.inputPlan(nestedSelect).input.filename, "root.safetensors");
// Subgraph output passthrough follows the specific instance, even when reused.
inner.links[4] = { origin_id: -10, origin_slot: 0 };
outer.resolveSubgraphOutputLink = () => ({ link: inner.links[4] });
const otherLoad = add(rootGraph, 4, "H3RefModLabLoad", { filename: "other.safetensors" });
const otherInstance = instance(rootGraph, 5, inner, [4]);
connect(rootGraph, 3, otherLoad, otherInstance, "refmod");
const rootCombine = add(rootGraph, 6, "H3RefModLabCombineV2");
connect(rootGraph, 4, outer, rootCombine, "refmods.refmod_1");
connect(rootGraph, 5, otherInstance, rootCombine, "refmods.refmod_2");
assert.deepEqual(Array.from(context.inputPlan(rootCombine).inputs, (input) => input.input.filename), ["root.safetensors", "other.safetensors"]);
assert.throws(() => context.inputPlan(insideSelect), /multiple instances/i);
// Cycles fail promptly; an identical ID in another graph is not a cycle.
const cycleA = add(rootGraph, 7, "Reroute");
const cycleB = add(rootGraph, 8, "Reroute");
connect(rootGraph, 6, cycleA, cycleB, "");
connect(rootGraph, 7, cycleB, cycleA, "");
assert.throws(() => context.inputPlan(cycleA), /cycle/);
assert.equal(context.inputPlan(combine).inputs[0].input.filename, "inner.safetensors");
console.log("Source-picker graph traversal and refresh lifecycle tests passed.");

// Exercise the registered selector's Refresh button inside a subgraph.
let requestedPlan;
context.api = { async fetchApi(url, options) {
    assert.equal(url, "/h3-refmods-lab/sources");
    requestedPlan = JSON.parse(options.body);
    return { ok: true, async json() { return { sources: [card("preview")] }; } };
} };
class Selector {
    graph = inner;
    widgets = [{ name: "selection", value: "all" }];
    inputs = [];
    addDOMWidget() {}
    setSize() {}
}
await context.app.extension.beforeRegisterNodeDef(Selector, { name: "H3RefModLabSelectSources" });
const selector = new Selector();
connect(inner, 5, combine, selector, "refmod");
selector.onNodeCreated();
await selector.sourcePicker.root.children[0].children.find((button) => button.textContent === "Refresh sources").click();
assert.equal(requestedPlan.type, "combine");
assert.equal(requestedPlan.inputs[0].input.filename, "inner.safetensors");
assert.equal(selector.sourcePicker.root.children[2].children.length, 1);
console.log("Registered selector refresh uses its owning graph.");
