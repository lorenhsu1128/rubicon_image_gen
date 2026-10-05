// Make ComfyUI's default workflow a blank canvas.
// The stock default (shown at startup when no workflow tab is restored, and after closing the last
// tab) is a Z-Image example whose models this project does not have, so it popped a missing-model
// warning every time. The frontend loads the default graph by reference from one object, so it is
// emptied in place; the startup load can run before this module, so setup() also swaps an untouched
// stock default that is already on the canvas.
import { app } from "/scripts/app.js";

const graphs = window.comfyAPI?.defaultGraph;
const STOCK_UNET = "z_image_turbo_bf16.safetensors";

function blank() {
  return structuredClone(graphs.blankGraph);
}

if (graphs?.defaultGraph && graphs?.blankGraph) {
  const target = graphs.defaultGraph;
  for (const key of Object.keys(target)) delete target[key];
  Object.assign(target, blank());
  graphs.defaultGraphJSON = JSON.stringify(target);
}

function isStockDefault(graph) {
  const nodes = graph?._nodes ?? [];
  return nodes.length > 0 && nodes.length <= 12 &&
    nodes.some((n) => n.widgets?.some((w) => w.value === STOCK_UNET));
}

app.registerExtension({
  name: "mechpipe.blankDefault",
  async setup() {
    if (!graphs?.blankGraph || !isStockDefault(app.graph)) return;
    const workflow = app.extensionManager?.workflow?.activeWorkflow;
    if (workflow && !workflow.isTemporary) return;
    await app.loadGraphData(blank(), true, true, workflow ?? null);
  },
});
