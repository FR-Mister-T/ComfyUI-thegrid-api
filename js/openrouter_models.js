import { app } from "../../scripts/app.js";

const DEBOUNCE_MS = 500;

app.registerExtension({
    name: "ComfyUI.TheGridAPI.OpenRouterModelList",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "OpenRouterModelList") return;

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            let debounceTimer = null;

            const fetchAndUpdate = async () => {
                const freeOnly  = this.widgets.find(w => w.name === "free_only")?.value  ?? false;
                const modality  = this.widgets.find(w => w.name === "modality_filter")?.value ?? "all";
                const filterTxt = this.widgets.find(w => w.name === "filter_text")?.value ?? "";

                try {
                    const params = new URLSearchParams({
                        free_only:   freeOnly,
                        modality:    modality,
                        filter_text: filterTxt,
                    });
                    const resp = await fetch(`/thegrid-api/openrouter-models?${params}`);
                    if (!resp.ok) return;
                    const models = await resp.json();
                    if (!Array.isArray(models) || !models.length) return;

                    const w = this.widgets.find(w => w.name === "model_select");
                    if (!w) return;

                    w.options.values = models;
                    if (!models.includes(w.value)) w.value = models[0];
                    this.setDirtyCanvas(true, true);
                } catch (e) {
                    console.warn("[ComfyUI-thegrid-api] Failed to fetch filtered models:", e);
                }
            };

            const schedule = () => {
                clearTimeout(debounceTimer);
                debounceTimer = setTimeout(fetchAndUpdate, DEBOUNCE_MS);
            };

            for (const name of ["free_only", "modality_filter", "filter_text"]) {
                const w = this.widgets.find(w => w.name === name);
                if (!w) continue;
                const orig = w.callback;
                w.callback = (...args) => {
                    if (orig) orig.apply(w, args);
                    schedule();
                };
            }
        };
    },
});
