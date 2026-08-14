# search-topic-research

Any-topic adaptive Tavily research → pack → multi-agent report.

This folder is an **example Pi skill** for [Pi Coding Agent graphs](../../pi-coding-agent-graphs.md). Install it into Pi’s global skills directory, then invoke `/skill:search-topic-research <topic>`. Full walkthrough (packages, Tavily key, model tiers, first run): that parent doc.

## Install from this repo

Pi loads skills from `~/.pi/agent/skills/` ([Pi skills](https://pi.dev/docs/latest/skills)). This skill writes packs and reports under that same path, so install it **globally**:

```bash
# from the Autarkic-LLM clone
mkdir -p ~/.pi/agent/skills
cp -a _Pi-Coding-Agent-Graphs/example-skills/search-topic-research \
      ~/.pi/agent/skills/search-topic-research
```

To track repo updates instead of copying:

```bash
ln -sfn "$(pwd)/_Pi-Coding-Agent-Graphs/example-skills/search-topic-research" \
        ~/.pi/agent/skills/search-topic-research
```

Then `/reload` in Pi (or start a new session). Confirm `/skill:search-topic-research` is listed.

## Where does research output go?

Everything is under the skill root:

```text
$HOME/.pi/agent/skills/search-topic-research/
├── notes/<slug>/<YYYY-MM-DD>-research-pack.md    # working evidence (host)
├── reports/<slug>/<YYYY-MM-DD>-research-report.md # final report (workflow)
├── scripts/research.workflow.js
└── references/
    ├── pack-template.md      # pack skeleton
    ├── report-template.md    # final report skeleton + naming rules
    ├── tavily-guide.md
    └── architecture.md
```

| File | Role |
|------|------|
| `*-research-pack.md` | Intermediate: queries, findings, fetches, URL inventory |
| `*-research-report.md` | Deliverable: templated research report |

Example:

```text
notes/eu-ai-act-enforcement-timeline/2026-08-14-research-pack.md
reports/eu-ai-act-enforcement-timeline/2026-08-14-research-report.md
```

Same **day** overwrites those two files. Other days keep separate files.

## Naming

| Piece | Rule |
|-------|------|
| **slug** | From topic; 3–8 words; `a-z0-9-` only; ≤48 chars; meaning-preserving |
| **YYYY-MM-DD** | `date +%Y-%m-%d` only (includes day; never invent) |
| **Pack** | `<YYYY-MM-DD>-research-pack.md` in `notes/<slug>/` |
| **Report** | `<YYYY-MM-DD>-research-report.md` in `reports/<slug>/` |
| **TITLE** | One-line H1 only — not in the path |

Full rules: `references/report-template.md`.

## Prerequisites

| Need | Why |
|------|-----|
| `TAVILY_API_KEY` (Pi’s env) | `web_search` / `web_fetch` |
| `@tavily/pi-extension` | Host Tavily tools |
| `@quintinshaw/pi-dynamic-workflows` | `workflow` + script runtime |
| `pi-subagents` | Subagent stack used with Pi multi-agent / workflow runs |
| `~/.pi/workflows/model-tiers.json` → models that exist | Medium tier for workflow agents |
| Writable skill `notes/` + `reports/` | Outputs |

```bash
pi install npm:@tavily/pi-extension
pi install npm:@quintinshaw/pi-dynamic-workflows
pi install npm:pi-subagents
# /workflows-models  → point small/medium/big at your local llama-cpp ids
# /reload after skill or package edits
```

## Invoke

```text
/skill:search-topic-research EU AI Act enforcement timeline 2025-2026
```

## Report template (final deliverable)

Headings are fixed — see `references/report-template.md`:

1. Metadata (As-of, Slug, Topic, Goal)  
2. Data quality  
3. Executive summary  
4. Findings by question  
5. Key claims and confidence (table)  
6. Conflicts and open questions  
7. What to distrust / data gaps  
8. Sources  

## Local models

- Short host skill; pack is durable memory (append after each search).
- `max_results: 6`, `concurrency: 1`, `agentRetries: 2`.
- Four workflow agents (Ingest → Findings → Skeptic → Report).
- STAMP/paths only from bash (`$HOME`, `date +%Y-%m-%d`).

If the model **loops on dates**, invents homes, or repeats garbage: fix server/sampler/KV first, then re-run. Keep `models.json` `contextWindow` / `maxTokens` aligned with the server.

## Fork a domain skill

1. Copy folder; rename frontmatter `name`.
2. Customize must-answer list + `references/report-template.md` (and workflow Report phase).
3. Keep host-Tavily + pack/workflow split; point tiers at your models.
4. Install under `~/.pi/agent/skills/<new-name>/` and update the hardcoded `$ROOT` paths in `SKILL.md` (and any references that assume `search-topic-research`).

## Fetch shape

```json
{ "urls": "https://example.com/article", "extract_depth": "advanced", "format": "markdown" }
```

Detail: `references/tavily-guide.md`.
