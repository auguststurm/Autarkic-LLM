# Pi Coding Agent: Graphs, Workflows & Tavily Research

How to drive this repo’s local `llama-server` with **Pi Coding Agent** multi-agent **workflows** (dynamic graphs) and optional **Tavily** web search for structured research reports.

Sits on top of [Agentic Harnesses](../agentic-harnesses.md) (connect Pi to the server). This doc covers packages, launch, the example research **skill**, and the lower-level `/workflows` commands.

> **Offline note:** weights and the LLM stay on your box (`127.0.0.1`). **Tavily is a cloud API** — queries and fetched pages leave the machine. For pure autarky, skip Tavily and use only local tools. Hardware guides still default to loopback-only `llama-server`.

## Purpose

**Input a topic → dated evidence pack + structured markdown report.** The recommended path is the generic skill [search-topic-research](example-skills/search-topic-research/) (any topic, not just one domain). It uses Tavily on the host, then a four-phase dynamic-workflow graph to synthesize the report.

Typical flow:

```text
you invoke the skill with a topic
  → host builds a research pack (Tavily search + fetch)
  → workflow: Ingest → Findings → Skeptic → Report
  → dated markdown under the skill’s notes/ and reports/
```

Use a **fresh session** per report (`/new`) so runs stay independent.

## Prerequisites

1. **Engine + model:** [local-setup.md](../local-setup.md) and your [hardware guide](../README.md#hardware-configurations-included) (e.g. [Windows RTX 4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) for Qwen3.6-27B Q4; [Dual RTX Qwen3.8](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) for roomy CUDA — ✅ tested; other 3.8 ports in [README](../README.md#qwen38-2026-08-14)).
2. **`llama-server` running** with that guide’s command.
3. **Pi `models.json`** from the same guide → `~/.pi/agent/models.json`. Keep `contextWindow` = server `--ctx-size` and `maxTokens` ≤ `--n-predict` ([agentic harnesses](../agentic-harnesses.md)).
4. **Pi Coding Agent** installed ([pi.dev](https://pi.dev)).

Long **pi-workflows** need a large **input** pin (field overflows at **32k/64k**, single requests **~70k+**) and enough **output** (`maxTokens` / `--n-predict`) for synthesis — often **8192–16384**, not 4k. Exact numbers are in your **hardware guide**. Cross-hardware dense Qwen 27B + Pi rules (3.6 / 3.8; no DRY, thinking off, K/V, two limits): [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Field-validated baselines: [Windows RTX 4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) (**96k q8/q8**, `maxTokens` **16384**); [Dual RTX 6000 Qwen3.8](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) (**262k q8/q8**, `maxTokens` **16384**). Match server and Pi; **new session** after garbage.

## Core stack

These are **Pi packages** (extensions/skills), not plain `npm install -g` apps. You install them with the **`pi install`** CLI after Pi itself is on your `PATH`. Packages land under `~/.pi/agent/npm/` (user/global) or `.pi/npm/` (project-local with `-l`), and Pi wires them into settings automatically.

### Packages

| Layer | Package | Install source | Role |
| --- | --- | --- | --- |
| Search | **[@tavily/pi-extension](https://www.npmjs.com/package/@tavily/pi-extension)** (official) | `npm:@tavily/pi-extension` · [Pi catalog](https://pi.dev/packages) | Tavily **Search** + **Extract** as tools `web_search` / `web_fetch` (depth, topic, time range, domains, answer, images, …). Needs `TAVILY_API_KEY` in the **process** env (see [below](#tavily-api-key)). |
| Orchestration / graphs | **[@quintinshaw/pi-dynamic-workflows](https://www.npmjs.com/package/@quintinshaw/pi-dynamic-workflows)** | `npm:@quintinshaw/pi-dynamic-workflows` · [docs](https://quintinshaw.github.io/pi-dynamic-workflows/) · [Pi package page](https://pi.dev/packages/@quintinshaw/pi-dynamic-workflows) | Dynamic JS workflows: `agent()`, `parallel()`, `pipeline()`, `phase()`, fan-out, model routing, resume/journal, `/workflows` TUI, saved workflows. |
| Subagents | **[pi-subagents](https://www.npmjs.com/package/pi-subagents)** | `npm:pi-subagents` | Isolated child Pi processes for specialist roles (used heavily by dynamic workflows). |
| Memory | **[pi-hermes-memory](https://www.npmjs.com/package/pi-hermes-memory)** | `npm:pi-hermes-memory` | Persistent memory + session search across runs (optional for pure one-shot research, useful for long-lived projects). |

Browse more packages: [pi.dev/packages](https://pi.dev/packages). Package system docs: [Pi packages](https://pi.dev/docs/latest/packages). Skill discovery: [Pi skills](https://pi.dev/docs/latest/skills).

### Install Pi CLI (if needed)

If `pi` is not installed yet:

```bash
# Official installer (Linux/macOS)
curl -fsSL https://pi.dev/install.sh | sh

# Or npm global
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
```

Quickstart: [pi.dev/docs/latest/quickstart](https://pi.dev/docs/latest/quickstart). Confirm:

```bash
pi --version
pi --help    # should list: install, remove, update, list, config, …
```

Point Pi at your local server first (hardware guide `models.json` → `~/.pi/agent/models.json`) — see [agentic harnesses](../agentic-harnesses.md).

### Install the core stack with `pi install`

**User-global** (recommended default — available in every project):

```bash
pi install npm:@tavily/pi-extension
pi install npm:@quintinshaw/pi-dynamic-workflows
pi install npm:pi-subagents
pi install npm:pi-hermes-memory   # optional
```

**Project-local** (writes to `.pi/settings.json`, shareable with a team; auto-installs on startup if missing):

```bash
cd /path/to/your-project
pi install -l npm:@tavily/pi-extension
pi install -l npm:@quintinshaw/pi-dynamic-workflows
pi install -l npm:pi-subagents
# pi install -l npm:pi-hermes-memory
```

| Command | What it does |
| --- | --- |
| `pi install npm:<pkg>` | Install from npm into **user** settings (`~/.pi/agent/`) |
| `pi install -l npm:<pkg>` | Install into **project** settings (`.pi/`) |
| `pi list` | Show installed package sources |
| `pi remove npm:<pkg>` | Uninstall / drop from settings (`pi uninstall` is an alias) |
| `pi update` | Update Pi and/or installed extensions |
| `pi config` | TUI to enable/disable package resources (Tab switches user vs project scope) |
| `pi -e npm:<pkg>` | Load a package **for this run only** (no permanent install) |

Other sources (same CLI): `pi install git:github.com/user/repo`, `pi install https://github.com/user/repo`, `pi install ./local/path`.

### After install

1. **`pi list`** — confirm all four (or three) sources appear.
2. Start Pi (`pi` or `TAVILY_API_KEY=… pi` if using Tavily).
3. In the Pi session, run **`/reload`** so extensions register tools and `/workflows`.
4. If you use project-local `.pi/` files, **trust** the project when prompted (`--approve` / `-a` on the CLI, or accept in the TUI). Untrusted project resources are ignored.
5. Sanity checks:
   - Tavily: tools **`web_search`** / **`web_fetch`** available when the API key is in the process env.
   - Workflows: `/workflows` responds (list / run / …).

> **Security:** extensions run with full agent tool access (shell, files, network). Prefer official/maintained packages; review source before installing unknown ones.

## Model configuration (dense Qwen 27B + Pi)

| Piece | Recommendation |
| --- | --- |
| Server | This repo’s **llama-cpp-turboquant** `llama-server` (fresh build for **Qwen3.8** / `qwen35`) |
| Model / quant | From **your** hardware guide — dense **Qwen3.6-27B** or **Qwen3.8-27B** UD quants on mid/large boxes (Dual RTX 3.8 ✅ tested) |
| Input window | Hardware guide `--ctx-size` = Pi `contextWindow`. Multi-agent research often needs **≫32k** |
| Output ceiling | Pi `maxTokens` ≤ `--n-predict` (report runs: **8192–16384** so synthesis is not truncated) |
| Thinking | **`--reasoning off`** (+ budget 0). Prefer over deprecated `enable_thinking` kwargs alone |
| Tools | **No DRY**; presence **0** for path-heavy agents; see [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware) |
| KV | Prefer **q8/q8** while it fits; turbo **V** only to buy context (K stays high precision) |

Example field-validated pins (do not copy numbers across hardware without reading that guide): [Windows RTX 4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) (24 GB) · [Dual RTX 6000 Qwen3.8](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) (192 GB, ✅ tested). Other 3.8 ports: [DGX Spark](../DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) · [M5 Pro](../M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md).

## Configuration & launch

### Tavily API key

Project `.env` (gitignored — never commit):

```bash
# .env
TAVILY_API_KEY=tvly-...
```

The extension does **not** auto-load `.env`. Put the key in the **process environment**:

```bash
source .env
TAVILY_API_KEY=$TAVILY_API_KEY pi
```

Correct setup → tools **`web_search`** and **`web_fetch`**.

### Directory layout

```text
~/.pi/agent/                    # Global Pi agent home
├── settings.json
├── models.json                 # From your hardware guide
├── skills/                     # Global skills (recommended home for the example)
│   └── search-topic-research/  # Copy or symlink from this repo
├── extensions/
├── agents/
├── sessions/
└── ...

~/.pi/workflows/                # dynamic-workflows: runs, journals, model-tiers
├── model-tiers.json            # small / medium / big → your local model ids
└── .../runs/*.json

<your-project>/
├── .env                        # TAVILY_API_KEY=...  (gitignored)
└── .pi/                        # Project-local overrides (optional)
    ├── settings.json
    └── skills/                 # Alternative skill discovery (after project trust)
```

The example skill **writes output under its own root** (`notes/` + `reports/`), not under the project you launched Pi from. After a global install that root is `~/.pi/agent/skills/search-topic-research/`.

## Example skill: search-topic-research

This is the generic, any-topic research skill shipped in this repo. It is the supported way to use Pi dynamic workflow graphs for research: the host does all Tavily work; the workflow script runs four specialist agents that only read the pack and write the report.

| Piece | Path in this repo |
| --- | --- |
| Skill (copy this folder) | [`example-skills/search-topic-research/`](example-skills/search-topic-research/) |
| User-facing notes | [`example-skills/search-topic-research/README.md`](example-skills/search-topic-research/README.md) |
| Agent instructions | [`example-skills/search-topic-research/SKILL.md`](example-skills/search-topic-research/SKILL.md) |
| Graph script | [`example-skills/search-topic-research/scripts/research.workflow.js`](example-skills/search-topic-research/scripts/research.workflow.js) |
| Pack / report templates | [`references/`](example-skills/search-topic-research/references/) |

### What it does

1. You pass a **topic or research brief** (verbatim).
2. The host derives a **slug** (3–8 hyphenated words) and a **stamp** from `date +%Y-%m-%d` (never invents the date).
3. The host searches and fetches with Tavily, appending evidence to a **pack** after every call.
4. The host loads `scripts/research.workflow.js` and calls the `workflow` tool with the pack + report paths.
5. Workflow agents (no web): **Ingest → Findings → Skeptic → Report**.
6. The Report agent writes dated markdown. You verify the file on disk.

```text
Host Tavily
  → notes/<slug>/<YYYY-MM-DD>-research-pack.md
  → workflow: Ingest → Findings → Skeptic → Report
  → reports/<slug>/<YYYY-MM-DD>-research-report.md
```

Same **day** overwrites those two files. Other days keep separate files. Thin evidence produces a thin report — the skill forbids invented facts.

### Install the skill

Pi discovers directories that contain `SKILL.md` under `~/.pi/agent/skills/` (global) or `.pi/skills/` (trusted project). This skill **hardcodes** its output root to `$HOME/.pi/agent/skills/search-topic-research`, so install it **globally** (copy or symlink). Project-local discovery would still write there.

From a clone of this repo:

```bash
mkdir -p ~/.pi/agent/skills

# Copy (stable; you re-copy after pulling skill updates)
cp -a _Pi-Coding-Agent-Graphs/example-skills/search-topic-research \
      ~/.pi/agent/skills/search-topic-research

# Or symlink (tracks the clone; break it if you customize in place)
# ln -sfn "$(pwd)/_Pi-Coding-Agent-Graphs/example-skills/search-topic-research" \
#         ~/.pi/agent/skills/search-topic-research
```

Confirm the script is present:

```bash
test -f ~/.pi/agent/skills/search-topic-research/scripts/research.workflow.js \
  && echo "skill ready"
```

In an existing Pi session run **`/reload`** so `/skill:search-topic-research` appears. New sessions pick it up automatically.

### Point workflow model tiers at your local model

Workflow agents use the **medium** tier. They must name a model that exists in your Pi `models.json` (the same local llama-cpp id from your hardware guide).

```text
/workflows-models
```

That TUI writes `~/.pi/workflows/model-tiers.json`. Point **small**, **medium**, and **big** at your local model (or leave small/big unused but keep medium valid). Example shape — replace the id with yours:

```json
{
  "tiers": {
    "small": "llama-cpp/your-model",
    "medium": "llama-cpp/your-model",
    "big": "llama-cpp/your-model"
  }
}
```

Without a valid medium mapping, the workflow can fail even when the host search succeeded.

### First run

1. Start **`llama-server`** (hardware guide).
2. Launch Pi with the Tavily key in the process env (see [Tavily API key](#tavily-api-key)).
3. Fresh session (`/new` or a new terminal).
4. Invoke the skill. Everything after the command is the topic, **verbatim**:

```text
/skill:search-topic-research EU AI Act enforcement timeline 2025-2026
```

Other valid briefs:

```text
/skill:search-topic-research Japan mobile game market: popular genres, top titles, and why players enjoy them

/skill:search-topic-research Compare RISC-V SBC options for a 2026 homelab NAS, focusing on ECC RAM and SATA
```

Empty topic → the skill stops and asks. User-listed questions stay in the pack as written.

5. Watch the host search, then the workflow phases. `/workflows` lists / watches / stops runs if you need to intervene.
6. When it finishes, confirm the two files exist (stamp is today’s date from the machine clock):

```bash
ls -l ~/.pi/agent/skills/search-topic-research/notes/*/*-research-pack.md
ls -l ~/.pi/agent/skills/search-topic-research/reports/*/*-research-report.md
```

Example for the first brief:

```text
~/.pi/agent/skills/search-topic-research/notes/eu-ai-act-enforcement-timeline/2026-08-14-research-pack.md
~/.pi/agent/skills/search-topic-research/reports/eu-ai-act-enforcement-timeline/2026-08-14-research-report.md
```

The host should also print absolute **PACK** and **REPORT** paths plus a short takeaway list. If the report file is missing, do not treat the run as successful — re-run the workflow or write the report on the host from the pack (the skill says this explicitly).

### What you should see in the report

Fixed headings (see [`references/report-template.md`](example-skills/search-topic-research/references/report-template.md)):

1. Metadata (As-of, Slug, Topic, Goal)
2. Data quality
3. Executive summary
4. Findings by question
5. Key claims and confidence (table)
6. Conflicts and open questions
7. What to distrust / data gaps
8. Sources (pack URLs only)

### How the graph is invoked

You do **not** type `/workflows run …` for this skill. The host reads `research.workflow.js` and calls the `workflow` tool with the whole script plus args (`topic`, `title`, `slug`, `stamp`, `packPath`, `reportPath`). Constraints baked in: `concurrency: 1`, `agentRetries: 2`, no model id in args (medium tier only).

Do not substitute a different workflow name (the skill forbids `name: "deep-research"`). Use this skill’s script.

Full host rules, Tavily knobs, and stop conditions: [`SKILL.md`](example-skills/search-topic-research/SKILL.md). Architecture one-pager: [`references/architecture.md`](example-skills/search-topic-research/references/architecture.md). Tavily parameter cheat sheet: [`references/tavily-guide.md`](example-skills/search-topic-research/references/tavily-guide.md).

### Fork a domain skill

1. Copy the folder; change frontmatter `name` in `SKILL.md`.
2. Customize the must-answer list and `references/report-template.md` (and the Report phase prompt in the workflow script if headings change).
3. Keep the host-Tavily + pack / workflow split.
4. Install the copy under `~/.pi/agent/skills/<new-name>/` and update any hardcoded `$ROOT` paths inside `SKILL.md` to match.
5. `/reload` and invoke `/skill:<new-name> …`.

## Ad-hoc `/workflows` (no skill)

You can still run a one-off graph from chat if you are exploring or do not want the skill:

```text
/workflows run Research <topic>: produce clean structured markdown and save it under reports/.
```

That path has no pack template, no auto-dated filenames, and no four-phase script unless you write one. After a **completed** successful run you can save a reusable command:

```text
/workflows save research-topic
```

Prefer the skill for repeatable research. Keep `/workflows` for monitoring, resume, and experiments.

### `/workflows` quick reference

```text
/workflows [list]
/workflows run <prompt>
/workflows status <id>
/workflows watch <id>
/workflows stop <id>
/workflows pause <id>
/workflows resume <id>
/workflows rm <id>
/workflows save <name> [runId]
/workflows-models
```

Run journals: `~/.pi/workflows/.../runs/*.json` — useful if chat hits max tokens but the workflow already finished.

## Observed behavior

- Tavily tools appear when the key is in the process env.
- Dynamic workflows support **parallel** specialists and a **synthesis** step; this skill runs its four agents **sequentially** (`concurrency: 1`) to stay stable on local dense Qwen.
- Report quality tracks **stable local settings** (context pin, agent sampling, no DRY for tool paths) and **pack quality** (the workflow cannot search).
- Large runs can still overflow a small `contextWindow` or hit **`maxTokens` / `--n-predict`** on the final “show the report” turn even when the workflow completed. Prefer reading the file on disk.

## Known gotchas

| Gotcha | What to do |
| --- | --- |
| Tavily key not loaded from `.env` alone | `source .env` then `TAVILY_API_KEY=$TAVILY_API_KEY pi` |
| `/skill:search-topic-research` missing | Copy/symlink into `~/.pi/agent/skills/`; `/reload`; trust the project if using `.pi/skills/` |
| `MISSING_SCRIPT` | The workflow JS is not at `$ROOT/scripts/research.workflow.js` — re-copy the whole skill folder |
| Workflow agents pick a cloud / missing model | `/workflows-models` — point **medium** at your local llama-cpp id |
| `/workflows save` before any success | Complete a run first, then save (ad-hoc path only) |
| `request exceeds the available context size` | Raise server `--ctx-size` **and** Pi `contextWindow` to the same value; restart both |
| Pi bar still shows **33k** after a guide bump | Rewrite `models.json` and **restart Pi** |
| `maximum output token limit` | Raise **both** `--n-predict` and Pi `maxTokens` (e.g. 8192–16384); or read `reports/*.md` and summarize in chat; open run JSON if synthesis finished |
| Path soup / tool loops on Qwen3.6 / 3.8 | No DRY; tool sampling; q8 V before turbo — [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware) |
| Turn-1 garbage after turbo V | New session; fall back to **q8/q8** at your pin; A/B only V type |
| Date / `$HOME` loops | The skill requires bash for stamp and paths; re-run that block once and continue. Fix sampler/KV if it persists |
| Context bleed between topics | Fresh session (`/new`) per report |
| Looking for `reports/` in the project cwd | Skill output is under `~/.pi/agent/skills/search-topic-research/{notes,reports}/` |

## Roadmap

- Optional **model tier** split (lighter Ingest/Skeptic on **small**, Report on **medium**) once local routing is field-tested
- Domain forks (keep the generic skill unchanged)
- Optional project-cwd output root (today the skill’s `$ROOT` is the global skill directory)

## Relationship to this repo

| Doc | Role |
| --- | --- |
| [README.md](../README.md) | Hardware table + entry points |
| [local-setup.md](../local-setup.md) | Build engine, download GGUFs |
| Hardware guides | Exact `llama-server` + `models.json` |
| [agentic-harnesses.md](../agentic-harnesses.md) | Pi / OpenClaw / Hermes; basic connection |
| **This file** | Workflows/graphs, Tavily, example research skill |
| [search-topic-research](example-skills/search-topic-research/) | Generic skill: pack → four-phase graph → dated report |
| [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) | Flag reference |

**Stack summary:** official Tavily tools + dynamic workflows + this skill give a practical multi-agent research graph without a custom orchestrator; the LLM remains the local model from this repo.

**Last Updated:** August 14, 2026 (example skill `search-topic-research` + install/invoke walkthrough)
