# Final report template

**Where it is written**

```text
$ROOT/reports/<slug>/<YYYY-MM-DD>-research-report.md
```

`$ROOT` = `$HOME/.pi/agent/skills/search-topic-research`  
`slug` / `YYYY-MM-DD` = see **Naming** below.

**Rules:** Keep every `##` heading below (order fixed). Use **Unknown** when evidence is missing. Every factual claim needs a URL from the pack. No invention. Dense bullets over prose.

---

## Report body (fixed headings)

```
# <TITLE> — <YYYY-MM-DD>

**As-of:** <YYYY-MM-DD>
**Slug:** <slug>
**Topic:**
  (verbatim TOPIC; use a fenced code block if multi-line)
**Goal:** <one-sentence respectful summary>

## Data quality
- Search/fetch counts (from pack)
- Snippet-only vs full-page fetch coverage
- Strongest source types; weakest / SEO / stale
- Overall confidence (high / medium / low) in one line

## Executive summary
- 5–9 bullets: the answers a decision-maker needs
- Prefer: claim — (source or Unknown)

## Findings by question
### Q1. <must-answer text>
- claim — URL   OR   Unknown
### Q2. <must-answer text>
- …
(one ### subsection per must-answer question)

## Key claims and confidence
| Claim | Confidence | Source URL |
|-------|------------|------------|
| … | high / medium / low / unknown | https://… |

## Conflicts and open questions
- Contested points (both sides + URLs)
- Single-source claims that need corroboration
- Open questions still worth answering

## What to distrust / data gaps
- Biased samples, SEO, stale, wrong geography, overgeneralization
- Explicit Unknowns list

## Sources
1. https://… — short note (role / limit)
2. …
(only URLs that appear in the pack)
```

---

## Naming

| Piece | Form | Rules |
|-------|------|--------|
| **slug** | `eu-ai-act-enforcement-timeline` | From TOPIC; 3–8 words; lowercase ASCII, digits, hyphens; ≤48 chars; no spaces, slashes, dots, or punctuation. Keep meaning; do not rebrand. |
| **STAMP / YYYY-MM-DD** | `2026-08-14` | **Only** from bash `date +%Y-%m-%d`. Includes **day**. Never invent. |
| **TITLE** | one line ≤120 chars | Report H1 only; not used in the filesystem path. |
| **Pack file** | `<YYYY-MM-DD>-research-pack.md` | Working corpus under `notes/`. |
| **Report file** | `<YYYY-MM-DD>-research-report.md` | Final deliverable under `reports/`. |

**Directory:** one folder per `slug` (not per day). Date (with day) lives in the **filename**.

```text
notes/<slug>/<YYYY-MM-DD>-research-pack.md      # intermediate evidence (host)
reports/<slug>/<YYYY-MM-DD>-research-report.md  # final report (workflow)
```

**Same day:** same paths → **overwrite**. Other days keep separate files.

**Example**

```text
Topic:  EU AI Act enforcement timeline 2025-2026
Slug:   eu-ai-act-enforcement-timeline
Stamp:  2026-08-14

notes/eu-ai-act-enforcement-timeline/2026-08-14-research-pack.md
reports/eu-ai-act-enforcement-timeline/2026-08-14-research-report.md
```
