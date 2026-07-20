# Pi Coding Agent: Graphs, Workflows & Tavily Research

How to drive this repo’s local `llama-server` with **Pi Coding Agent** multi-agent **workflows** (dynamic graphs) and optional **Tavily** web search for structured research reports.

Sits on top of [Agentic Harnesses](agentic-harnesses.md) (connect Pi to the server). This doc covers packages, launch, layout, and a proven research pattern.

> **Offline note:** weights and the LLM stay on your box (`127.0.0.1`). **Tavily is a cloud API** — queries and fetched pages leave the machine. For pure autarky, skip Tavily and use only local tools. Hardware guides still default to loopback-only `llama-server`.

## Purpose

Multi-agent research: **input a country (or idea) → clean markdown in `reports/`**, with optional fresh web data via Tavily.

Typical report sections:

- Mobile marketplace overview  
- Popular game genres / styles  
- Top titles  
- Cultural / player-preference reasons those games succeed  

Use a **fresh session** per report (`/new`) so runs stay independent.

## Prerequisites

1. **Engine + model:** [local-setup.md](local-setup.md) and your [hardware guide](README.md#hardware-configurations-included) (e.g. [Windows RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) for Qwen3.6-27B Q4 agent settings).
2. **`llama-server` running** with that guide’s command.
3. **Pi `models.json`** from the same guide → `~/.pi/agent/models.json`. Keep `contextWindow` = server `--ctx-size` and `maxTokens` ≤ `--n-predict` ([agentic harnesses](agentic-harnesses.md)).
4. **Pi Coding Agent** installed ([pi.dev](https://pi.dev)).

Long **pi-workflows** (parallel specialists + synthesis) need a **large** context pin. Workflows have overflowed **32k**; prefer guides that document **~64k** agent profiles (or raise both server and Pi to match). Small windows cause `request exceeds the available context size` and incomplete turns.

## Core stack

| Layer | Package | Role |
| --- | --- | --- |
| Search | [`@tavily/pi-extension`](https://www.npmjs.com/package/@tavily/pi-extension) (official) | Tavily Search + Extract as `web_search` / `web_fetch` (depth, topic, time range, domains, answer, images, …) |
| Orchestration / graphs | [`@quintinshaw/pi-dynamic-workflows`](https://www.npmjs.com/package/@quintinshaw/pi-dynamic-workflows) | Dynamic JS workflows: `agent()`, `parallel()`, `pipeline()`, `phase()`, fan-out, model routing, resume/journal, `/workflows` TUI, saved workflows |
| Subagents | `pi-subagents` | Isolated child Pi processes for specialist roles |
| Memory | `pi-hermes-memory` | Persistent memory + session search across runs |

Install per each package’s current docs. After installs, run **`/reload`** in Pi. Project **trust** is required for local `.pi/` resources.

## Model configuration

| Piece | Recommendation |
| --- | --- |
| Server | This repo’s **llama-cpp-turboquant** `llama-server` |
| Model / quant | From **your** hardware guide (example: `Qwen3.6-27B-UD-Q4_K_XL` on 24 GB-class cards) |
| Context | Large enough for multi-agent tool dumps (**64k**-class for workflows; 32k is often too small) |
| Orchestrator | Local Qwen as primary; optional model tiers for lighter subagents if you add endpoints |
| Qwen thinking | **Off** — same as hardware guides (`--reasoning off` + `enable_thinking:false`) so Pi gets `message.content` and tools |

Agent-oriented flags validated on Pi tool loops (no DRY, `presence_penalty 0`, `--cache-ram 0` on hybrid Qwen): see [Windows RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md). Apply the same ideas on other CUDA boxes when using long workflows.

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
├── extensions/
├── agents/
├── sessions/
├── workflows/                  # dynamic-workflows: runs, journals, model-tiers
└── ...

<your-project>/
├── .env                        # TAVILY_API_KEY=...  (gitignored)
├── .pi/                        # Project-local overrides
│   ├── settings.json
│   └── ...
└── reports/                    # Markdown research reports (only output dir)
```

```bash
mkdir -p reports
```

## Usage pattern

1. Start **`llama-server`** (hardware guide).
2. Launch Pi with `TAVILY_API_KEY` in the environment (if using Tavily).
3. Fresh session per report (`/new` or new terminal).
4. Run a workflow, for example:

```text
/workflows run Research <Country> mobile game market: popular genres, top titles, and why players enjoy them. Produce clean structured markdown and save it under reports/ (e.g. reports/<country>-YYYY-MM-DD.md).
```

5. Monitor: `/workflows` (list, status, watch, stop, pause, resume, …).
6. After a **completed** successful run, save a reusable command:

```text
/workflows save research-market
```

7. Confirm the markdown landed in **`reports/`**. Until a saved workflow or skill auto-saves, keep the save path in the prompt.

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
```

Run journals: `~/.pi/workflows/.../runs/*.json` — useful if chat hits max tokens but the workflow already finished.

## Observed behavior

- Tavily tools appear when the key is in the process env.
- Dynamic workflows support **parallel** specialists and a **synthesis** step.
- Report quality tracks **stable local settings** (context pin, agent sampling, no DRY for tool paths).
- Large runs can still overflow a small `contextWindow` or hit **`maxTokens` / `--n-predict`** on the final “show the report” turn even when the workflow completed.

## Known gotchas

| Gotcha | What to do |
| --- | --- |
| Tavily key not loaded from `.env` alone | `source .env` then `TAVILY_API_KEY=$TAVILY_API_KEY pi` |
| `/workflows save` before any success | Complete a run first, then save |
| `request exceeds the available context size` | Raise server `--ctx-size` **and** Pi `contextWindow` to the same value; restart both |
| Pi bar still shows **33k** after a guide bump | Rewrite `models.json` and **restart Pi** |
| `maximum output token limit` | Raise `--n-predict` and Pi `maxTokens` (e.g. 4096); open the run JSON if synthesis finished |
| Path soup / tool loops on Qwen3.6 | No DRY; tool-friendly sampling — [RTX 4090 guide](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) |
| Context bleed between countries | Fresh session (`/new`) per report |
| Manual “save under reports/” every time | Temporary; bake into saved workflow or skill (roadmap) |

## Roadmap

- **Saved workflow or skill** that:
  - Accepts **country** (or idea) as argument  
  - Fixed multi-phase graph: market overview → genres → cultural analysis → synthesis  
  - Auto-saves `reports/<country>-YYYY-MM-DD.md`
- Optional **model tier** config for lighter subagents  
- Short **system prompt / skill** for consistent report structure  

## Relationship to this repo

| Doc | Role |
| --- | --- |
| [README.md](README.md) | Hardware table + entry points |
| [local-setup.md](local-setup.md) | Build engine, download GGUFs |
| Hardware guides | Exact `llama-server` + `models.json` |
| [agentic-harnesses.md](agentic-harnesses.md) | Pi / OpenClaw / Hermes; basic connection |
| **This file** | Workflows/graphs, Tavily, `reports/` harness |
| [llama-cpp-turboquant.md](llama-cpp-turboquant.md) | Flag reference |

**Stack summary:** official Tavily tools + dynamic workflows give a practical multi-agent research “graph” without a custom orchestrator; the LLM remains the local model from this repo.

**Last Updated:** July 2026
