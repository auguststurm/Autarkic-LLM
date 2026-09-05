export const meta = {
  name: "search_topic_research_dual_rtx",
  description:
    "Pack → report. Findings and Skeptic run small+medium together (both Dual RTX cards).",
  phases: [
    { title: "Findings" },
    { title: "Skeptic" },
    { title: "Report" },
  ],
};

const topic = args && args.topic;
const packPath = args && args.packPath;
const reportPath = args && args.reportPath;
const stamp = (args && args.stamp) || "unknown";
const slug = (args && args.slug) || "topic";

function oneLine(s, max) {
  const t = String(s || "")
    .replace(/\s+/g, " ")
    .trim();
  if (!t) return "";
  return t.length > max ? t.slice(0, max - 1) + "…" : t;
}

const title =
  oneLine(args && args.title, 120) ||
  oneLine(topic, 120) ||
  slug;

if (!topic || !packPath || !reportPath) {
  throw new Error("args.topic, args.packPath, and args.reportPath are required");
}

const packRules = `No web. File tools only. Read PACK. Prefer Fetch extracts. No invention.
TOPIC: ${topic}
PACK: ${packPath}`;

phase("Findings");
const [findingsA, findingsB] = await parallel([
  () =>
    agent(
      `${packRules}

Must-answer questions in the pack: take the FIRST HALF (Q1 through midpoint; extra Q goes here if odd count).
≤40 lines. Each assigned Q: claim + URL, or Unknown.
Then short:
## Contested / single-source
## Freshness
`,
      { label: "findings-small", tier: "small" },
    ),
  () =>
    agent(
      `${packRules}

Must-answer questions in the pack: take the SECOND HALF (after midpoint).
≤40 lines. Each assigned Q: claim + URL, or Unknown.
Then short:
## Contested / single-source
## Freshness
`,
      { label: "findings-medium", tier: "medium" },
    ),
]);

const findings = `## First half (small)
${findingsA}

## Second half (medium)
${findingsB}`;

phase("Skeptic");
const [critiqueA, critiqueB] = await parallel([
  () =>
    agent(
      `Hostile review. No web. No new facts.

TOPIC: ${topic}
FINDINGS:
${findings}

≤30 lines: unsupported, SEO-as-fact, overgeneralization, stale, false causation, missing counters.
`,
      { label: "skeptic-small", tier: "small" },
    ),
  () =>
    agent(
      `Hostile review. No web. No new facts.

TOPIC: ${topic}
FINDINGS:
${findings}

≤30 lines: unsupported, SEO-as-fact, overgeneralization, stale, false causation, missing counters.
`,
      { label: "skeptic-medium", tier: "medium" },
    ),
]);

const critique = `## Skeptic small
${critiqueA}

## Skeptic medium
${critiqueB}`;

phase("Report");
const out = await agent(
  `Write the FINAL report file. No web. Obey BOTH skeptics. Re-read PACK for URLs only.

PACK: ${packPath}
REPORT (create/overwrite this exact path): ${reportPath}

TITLE: ${title}
STAMP: ${stamp}
SLUG: ${slug}
TOPIC (under **Topic:** fence only; never in H1):
${topic}

FINDINGS:
${findings}

SKEPTIC:
${critique}

Rules:
- Use EVERY section below in this order (fixed template)
- Sources = URLs from pack only; Unknown OK; no invention
- H1 exactly: # ${title} — ${stamp}
- Dense bullets; complete file; no ramble
- After write, reply only: SAVED ${reportPath}

# ${title} — ${stamp}

**As-of:** ${stamp}
**Slug:** ${slug}
**Topic:**
\`\`\`
(verbatim topic here)
\`\`\`
**Goal:** (one sentence from pack Goal if present)

## Data quality
## Executive summary
## Findings by question
### Q1. (must-answer text)
### Q2. …
## Key claims and confidence
| Claim | Confidence | Source URL |
|-------|------------|------------|
## Conflicts and open questions
## What to distrust / data gaps
## Sources
`,
  { label: "report-writer", tier: "big" },
);

log(typeof out === "string" ? out : JSON.stringify(out));
return { ok: true, topic, title, slug, stamp, packPath, reportPath, agentReply: out };
