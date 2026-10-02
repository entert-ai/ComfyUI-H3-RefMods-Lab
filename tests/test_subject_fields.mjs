import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../web/subject-fields.js", import.meta.url), "utf8")
    .replace(/^import .*;\s*/m, "").replace("export function", "function");
const context = { app: { registerExtension() {} } };
vm.createContext(context);
vm.runInContext(source, context);
const graph = { nodes: [
    { type: "H3RefModLabCreate", widgets_values: ["old label", "portrait", "vae", 768, 8192, "Alice", "key", "fully_preserved", "identity"] },
    { type: "H3RefModLabCreateVideo", widgets_values: ["clip", "motion", "vae", 0, 3, 512, 8192, "audio vae", "Alice", "key", "weak_reference", "motion", "reference", "voice"] },
    { type: "H3RefModLabCreateAudio", widgets_values: ["voice", "timbre", "vae", 0, 3, 8192, "Alice", "key", "reference", "timbre"] },
    { type: "H3RefModLabSetInstructions", widgets_values: ["body", "Alice", "key", "partially_preserved", "shape", "reference", "voice"] },
] };
context.migrateSubjectFields(graph);
assert.deepEqual(graph.nodes[0].widgets_values, ["Alice", "portrait", 768, 8192, "fully_preserved", "identity"]);
assert.deepEqual(graph.nodes[1].widgets_values, ["Alice", "motion", 0, 3, 512, 8192, "weak_reference", "motion", "reference", "voice"]);
assert.deepEqual(graph.nodes[2].widgets_values, ["Alice", "timbre", 0, 3, 8192, "reference", "timbre"]);
assert.deepEqual(graph.nodes[3].widgets_values, ["body", "Alice", "partially_preserved", "shape", "reference", "voice"]);
const saved = JSON.stringify(graph);
context.migrateSubjectFields(graph);
assert.equal(JSON.stringify(graph), saved);
const nested = { definitions: { subgraphs: [{ nodes: [{ type: "H3RefModLabCreate", widgets_values: ["Alice", "", "vae", 768, 8192, "", "", "unspecified", ""] }] }] } };
context.migrateSubjectFields(nested);
assert.equal(nested.definitions.subgraphs[0].nodes[0].widgets_values.length, 6);
const previous = { nodes: [
    { type: "H3RefModLabCreate", widgets_values: ["Alice", "portrait", "vae", 768, 8192, "weak_reference", "face"] },
    { type: "H3RefModLabCreateVideo", widgets_values: ["Alice", "motion", "vae", 2, 4, 512, 8192, "audio vae", "weak_reference", "motion", "reference", "voice"] },
    { type: "H3RefModLabCreateAudio", widgets_values: ["Alice", "voice", "vae", 2, 4, 8192, "reference", "voice"] },
] };
context.migrateSubjectFields(previous);
assert.deepEqual(previous.nodes[0].widgets_values, ["Alice", "portrait", 768, 8192, "weak_reference", "face"]);
assert.deepEqual(previous.nodes[1].widgets_values, ["Alice", "motion", 2, 4, 512, 8192, "weak_reference", "motion", "reference", "voice"]);
assert.deepEqual(previous.nodes[2].widgets_values, ["Alice", "voice", 2, 4, 8192, "reference", "voice"]);
const migrated = JSON.stringify(previous);
context.migrateSubjectFields(previous);
assert.equal(JSON.stringify(previous), migrated);
console.log("Subject-field migration tests passed.");
