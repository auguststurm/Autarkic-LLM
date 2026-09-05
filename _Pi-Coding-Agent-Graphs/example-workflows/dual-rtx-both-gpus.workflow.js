export const meta = {
  name: "dual_rtx_both_gpus",
  description:
    "Dual RTX: split parallel agents across GPU 0 (small) and GPU 1 (medium) so both cards generate at once",
  phases: [{ title: "Both GPUs" }, { title: "Merge" }],
};

const items = (args && args.items) || [
  "Summarize what 1+1 equals in one sentence.",
  "Summarize what 2+2 equals in one sentence.",
];
const promptFor =
  (args && args.promptFor) ||
  "Do the task. Reply in ≤5 lines.";

if (!Array.isArray(items) || items.length < 2) {
  throw new Error("args.items must be an array of at least 2 strings (one job per GPU)");
}

phase("Both GPUs");
const results = await parallel(
  items.map((item, i) => () =>
    agent(`${promptFor}\n\nITEM:\n${item}`, {
      label: i % 2 === 0 ? `gpu0-${i}` : `gpu1-${i}`,
      tier: i % 2 === 0 ? "small" : "medium",
    }),
  ),
);

phase("Merge");
return await agent(
  `Merge these results into one short report.\n\n${results
    .map((r, i) => `## ${i % 2 === 0 ? "GPU0/small" : "GPU1/medium"} item ${i}\n${r}`)
    .join("\n\n")}`,
  { label: "merge", tier: "big" },
);
