import { app } from "../../../scripts/app.js";

// Migrate positional widget values before LiteGraph applies the new schemas.
export function migrateSubjectFields(graph) {
    const bases = { H3RefModLabCreate: 5, H3RefModLabCreateVideo: 8, H3RefModLabCreateAudio: 6 };
    for (const node of graph.nodes || []) {
        const values = node.widgets_values;
        if (!Array.isArray(values)) continue;
        const base = bases[node.type];
        const oldCreate = base != null && values.length === base + (node.type === "H3RefModLabCreateVideo" ? 6 : 4);
        const oldEditor = node.type === "H3RefModLabSetInstructions" && values.length === 7;
        if (oldCreate || oldEditor) {
            if ((node.inputs || []).some((input) => input.link != null && ["name", "subject_name", "subject_key"].includes(input.name))) {
                throw new Error("Replace this older H3 Create/Instructions node: its former naming fields are linked inputs.");
            }
            if (oldCreate) {
                values[0] = values[base]?.trim() || values[0] || "Subject";
                values.splice(base, 2);
            } else {
                values[1] = values[1]?.trim() || "Subject";
                values.splice(2, 1);
            }
        }
    }
    for (const subgraph of graph.definitions?.subgraphs || []) migrateSubjectFields(subgraph);
}

app.registerExtension({
    name: "H3RefModsLab.SubjectFields",
    beforeConfigureGraph: migrateSubjectFields,
});
