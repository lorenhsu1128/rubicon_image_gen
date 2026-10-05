// Shows the prompt a MechPrompt / MechMarkMask node will use in its `full_prompt` box.
// The box is filled from /mech/prompt when the node is loaded empty and whenever one of the fields it
// depends on changes; text typed into the box is what the node outputs.
import { app } from "/scripts/app.js";

const FIELDS = { MechPrompt: ["template", "part", "view", "text"], MechMarkMask: ["mode", "text"] };

const widget = (node, name) => node.widgets?.find((w) => w.name === name);

async function refresh(node) {
  const query = new URLSearchParams({ type: node.comfyClass });
  for (const name of FIELDS[node.comfyClass]) query.set(name, widget(node, name)?.value ?? "");
  const res = await fetch(`/mech/prompt?${query}`);
  const out = widget(node, "full_prompt");
  if (!res.ok || !out) return;
  out.value = await res.text();
  node.setDirtyCanvas?.(true, true);
}

function debounce(fn, ms) {
  let t;
  return () => {
    clearTimeout(t);
    t = setTimeout(fn, ms);
  };
}

app.registerExtension({
  name: "mechpipe.promptPreview",
  nodeCreated(node) {
    if (!FIELDS[node.comfyClass]) return;
    const update = debounce(() => refresh(node), 300);
    for (const name of FIELDS[node.comfyClass]) {
      const w = widget(node, name);
      if (!w) continue;
      const cb = w.callback;
      w.callback = function (...args) {
        const r = cb?.apply(this, args);
        update();
        return r;
      };
      // multiline text boxes do not always fire the widget callback while typing
      w.element?.addEventListener?.("input", update);
    }
    // after a loaded workflow has set the saved values: fill the box only if it is empty
    setTimeout(() => {
      if (!widget(node, "full_prompt")?.value) refresh(node);
    }, 0);
  },
});
