export const meta = {
  name: "search_topic_research",
  description: "Synthesize research pack into dated topic report (no web)",
  phases: [
    { title: "Ingest" },
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

// No model id — medium tier / session default. Keep tiers on an available local model.

phase("Ingest");
const inventory = await agent(
  `Librarian. No web. Read pack with file tools only.

PACK: ${packPath}

≤40 lines, headings only:
## Counts (from pack; do not invent)
## Strongest evidence (URL each)
## Weak / SEO / stale
## Must-answer: answered vs Unknown
`,
  { label: "ingest" },
);

phase("Findings");
const findings = await agent(
  `Findings by must-answer. No web. Read PACK. Prefer Fetch extracts. No invention.

TOPIC: ${topic}
PACK: ${packPath}
INGEST:
${inventory}

≤50 lines. Each Q: claim + URL, or Unknown.
Then short:
## Contested / single-source
## Freshness
`,
  { label: "findings" },
);

phase("Skeptic");
const critique = await agent(
  `Hostile review. No web. No new facts.

TOPIC: ${topic}
FINDINGS:
${findings}

≤30 lines: unsupported, SEO-as-fact, overgeneralization, stale, false causation, missing counters.
`,
  { label: "skeptic" },
);

phase("Report");
const out = await agent(
  `Write the FINAL report file. No web. Obey skeptic. Re-read PACK for URLs only.

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
  { label: "report-writer" },
);

log(typeof out === "string" ? out : JSON.stringify(out));
return { ok: true, topic, title, slug, stamp, packPath, reportPath, agentReply: out };
