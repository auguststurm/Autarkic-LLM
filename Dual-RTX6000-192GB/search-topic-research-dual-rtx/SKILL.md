---
name: search-topic-research-dual-rtx
description: >
  Dual RTX 6000: same Tavily research as search-topic-research, but Findings
  and Skeptic run on both GPUs at once. Host still does search. No invention.
  Invoke: /skill:search-topic-research-dual-rtx <topic or research brief>
---

# Search Topic Research (Dual RTX)

Adaptive Tavily research: plan → search waves → fetch → pack → workflow → dated report.
Depth follows **unanswered questions**, not a fixed quota. **No invented facts.**

## Where output goes

| Artifact | Path | Who writes |
|----------|------|------------|
| **Pack** (evidence) | `$ROOT/notes/<slug>/<YYYY-MM-DD>-research-pack.md` | Host |
| **Report** (deliverable) | `$ROOT/reports/<slug>/<YYYY-MM-DD>-research-report.md` | Workflow Report agent |
| **Script** | `$ROOT/scripts/research.workflow.js` | (read only) |

`$ROOT` = `$HOME/.pi/agent/skills/search-topic-research-dual-rtx` (from bash).

**Templates (keep headings):**  
`references/pack-template.md` · `references/report-template.md` (naming rules live there too).

## Invoke

```text
/skill:search-topic-research-dual-rtx <topic or brief>
```

| Token | Rule |
|-------|------|
| **TOPIC** | Everything after the command, **verbatim** (may be multi-line). Never rewrite intent. |
| **SLUG** | 3–8 words: lowercase ASCII, digits, hyphens; ≤48 chars; no spaces/slashes/dots. Keep meaning. |
| **TITLE** | One-line H1 summary ≤120 chars. Keep meaning; do not marketize. |
| **STAMP** | **Only** `date +%Y-%m-%d` in bash. Never invent dates / `new Date()`. |

Empty TOPIC → **stop** and ask. User-listed questions stay verbatim in the pack.

```text
/skill:search-topic-research-dual-rtx EU AI Act enforcement timeline 2025-2026
```

## STOP

1. Paths only from bash echoes below — never invent `$HOME` / years.
2. No `web_search` → stop (`TAVILY_API_KEY` + `@tavily/pi-extension`).
3. No `workflow` → stop (`@quintinshaw/pi-dynamic-workflows`).
4. Thin evidence → thin report. Unknown after real effort; never invent.
5. No fixed search count. Cover must-answers, then stop.
6. Workflow: this skill’s **script** only — never `name: "deep-research"`.
7. Host owns all Tavily. Workflow agents: no web (coding tools only).

## Paths + naming + FIRST tools (before any research text)

```
ROOT   = $HOME/.pi/agent/skills/search-topic-research-dual-rtx
PACK   = ROOT/notes/SLUG/STAMP-research-pack.md
REPORT = ROOT/reports/SLUG/STAMP-research-report.md
SCRIPT = ROOT/scripts/research.workflow.js
```

| Name piece | Condition |
|------------|-----------|
| **SLUG** | 3–8 words from TOPIC; `a-z`, `0-9`, `-` only; ≤48 chars; no spaces/slashes/dots; keep meaning |
| **STAMP** | bash `date +%Y-%m-%d` only (`2026-08-14`) |
| **Pack file** | `{STAMP}-research-pack.md` under `notes/{SLUG}/` |
| **Report file** | `{STAMP}-research-report.md` under `reports/{SLUG}/` |
| **TITLE** | H1 only (≤120 chars); **not** part of the path |

One folder per slug; **day** in **filename** (not `slug/2026-08-14/…`). Same day **overwrites**.

```bash
STAMP=$(date +%Y-%m-%d)
ROOT="${HOME}/.pi/agent/skills/search-topic-research-dual-rtx"
SLUG="eu-ai-act-enforcement-timeline"   # replace: sanitized from TOPIC only
mkdir -p "$ROOT/notes/$SLUG" "$ROOT/reports/$SLUG"
test -f "$ROOT/scripts/research.workflow.js" || { echo "MISSING_SCRIPT"; exit 1; }
echo "STAMP=$STAMP"
echo "ROOT=$ROOT"
echo "SLUG=$SLUG"
echo "PACK=$ROOT/notes/$SLUG/${STAMP}-research-pack.md"
echo "REPORT=$ROOT/reports/$SLUG/${STAMP}-research-report.md"
echo "SCRIPT=$ROOT/scripts/research.workflow.js"
```

- Sanitized **SLUG** only in shell — never paste multi-line TOPIC into bash.
- Stamp/date loop → re-run `date +%Y-%m-%d` only, then continue.
- `MISSING_SCRIPT` → stop. Write pack **only** to printed `PACK=`.

## Tavily (host)

Defaults (no forced recency):

```json
{
  "query": "...",
  "search_depth": "basic",
  "topic": "general",
  "include_answer": true,
  "max_results": 6
}
```

| Need | Knob |
|------|------|
| Dense / high-stakes | `search_depth: "advanced"` (sparingly) |
| News now | `topic: "news"` + `time_range` `day`/`week`/`month` |
| Finance | `topic: "finance"` |
| Recency window | `time_range` only if brief needs it — **omit** for historical/evergreen |
| Geo | `country` enum lowercase (`"japan"`, `"united states"`) **and** `topic: "general"` |
| Trust / noise | `include_domains` / `exclude_domains` |
| Exact phrase | `exact_match: true` + quotes in query |
| Full page | `web_fetch` (not more search) |

`web_fetch` — `urls` string or string array (never a stringified JSON array):

```json
{ "urls": "https://example.com/article", "extract_depth": "advanced", "format": "markdown" }
```

Param detail: read `references/tavily-guide.md` **once** if unsure. Prefer few `basic` searches + 1–3 fetches over large SERP dumps (local context).

## Method

### Must-answer

Derive **4–8** questions from TOPIC (user’s list first). Each ends **fact+URL** or **Unknown**. Last item: remaining unknowns.

### A — Open pack

Write `PACK` (overwrite) using **`references/pack-template.md`** headings (must-answer, plan, findings per Q, fetch extracts, URL inventory, freshness, conflicts, counts, unknowns).

### B — Search waves (append to pack after **each** call)

1. Wave 1: coverage for all questions.  
2. Wave 2: gaps / conflicts only.  
3. Wave 3: optional one high-value lead.

**After every search:** append findings+URLs under the right `### Qn` — do not hold evidence only in chat. No empty Q sections (use Unknown).

### C — Fetch

When snippets thin or source is primary: fetch, append under `## Fetch extracts`. Skip SEO spam.

### D — Close pack

All skeleton sections present; counts = actual tool calls.

### E — Workflow

1. Read full `SCRIPT` from disk.  
2. Call tool **`workflow`**:

```json
{
  "script": "<entire research.workflow.js>",
  "args": {
    "topic": "<TOPIC verbatim>",
    "title": "<TITLE one line>",
    "slug": "<SLUG>",
    "stamp": "<STAMP>",
    "packPath": "<absolute PACK from bash>",
    "reportPath": "<absolute REPORT from bash>"
  },
  "background": false,
  "concurrency": 2,
  "agentRetries": 2
}
```

No model id in args. This skill’s script already splits Findings and Skeptic across `small` (`:8080`) and `medium` (`:8081`). Report uses `big`. `/workflows-models` must map those three tiers.

### F — Verify

`test -s` or read REPORT at the printed path. Structure must match **`references/report-template.md`** (all `##` headings). Must-answers or Unknown; Sources = pack URLs only; no invention. Missing report → do not claim success; re-run workflow or write report on host from pack + returned agent text.

### G — Done

Absolute PACK + REPORT, 3–7 takeaways, unknowns, search/fetch counts.

## Local-model guardrails

1. **Bash owns STAMP/ROOT/paths** — never reason about calendar date or home.  
2. **Pack is the memory** — append after each search/fetch; chat is disposable.  
3. **Small tool outputs** — `max_results: 6`; advanced/fetch only when it changes answers.  
4. **Both cards during Findings and Skeptic** — already set (`concurrency: 2`). Report is one `big` stream. Host Tavily is your `/model` (one card).
5. **No rumination loops** — if stuck on stamp/path/slug, re-run the bash block once and continue.

## Multi-topic

Only if asked: full run per topic, sequential.
