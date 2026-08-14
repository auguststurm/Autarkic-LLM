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

1. **Engine + model:** [local-setup.md](local-setup.md) and your [hardware guide](README.md#hardware-configurations-included) (e.g. [Windows RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) for Qwen3.6-27B Q4; [Dual RTX Qwen3.8](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) for roomy CUDA — ✅ tested; other 3.8 ports in [README](README.md#qwen38-2026-08-14)).
2. **`llama-server` running** with that guide’s command.
3. **Pi `models.json`** from the same guide → `~/.pi/agent/models.json`. Keep `contextWindow` = server `--ctx-size` and `maxTokens` ≤ `--n-predict` ([agentic harnesses](agentic-harnesses.md)).
4. **Pi Coding Agent** installed ([pi.dev](https://pi.dev)).

Long **pi-workflows** need a large **input** pin (field overflows at **32k/64k**, single requests **~70k+**) and enough **output** (`maxTokens` / `--n-predict`) for synthesis — often **8192–16384**, not 4k. Exact numbers are in your **hardware guide**. Cross-hardware dense Qwen 27B + Pi rules (3.6 / 3.8; no DRY, thinking off, K/V, two limits): [agentic harnesses](agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Field-validated baselines: [Windows RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) (**96k q8/q8**, `maxTokens` **16384**); [Dual RTX 6000 Qwen3.8](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) (**262k q8/q8**, `maxTokens` **16384**). Match server and Pi; **new session** after garbage.

## Core stack

| Layer | Package | Role |
| --- | --- | --- |
| Search | [`@tavily/pi-extension`](https://www.npmjs.com/package/@tavily/pi-extension) (official) | Tavily Search + Extract as `web_search` / `web_fetch` (depth, topic, time range, domains, answer, images, …) |
| Orchestration / graphs | [`@quintinshaw/pi-dynamic-workflows`](https://www.npmjs.com/package/@quintinshaw/pi-dynamic-workflows) | Dynamic JS workflows: `agent()`, `parallel()`, `pipeline()`, `phase()`, fan-out, model routing, resume/journal, `/workflows` TUI, saved workflows |
| Subagents | `pi-subagents` | Isolated child Pi processes for specialist roles |
| Memory | `pi-hermes-memory` | Persistent memory + session search across runs |

Install per each package’s current docs. After installs, run **`/reload`** in Pi. Project **trust** is required for local `.pi/` resources.

## Model configuration (dense Qwen 27B + Pi)

| Piece | Recommendation |
| --- | --- |
| Server | This repo’s **llama-cpp-turboquant** `llama-server` (fresh build for **Qwen3.8** / `qwen35`) |
| Model / quant | From **your** hardware guide — dense **Qwen3.6-27B** or **Qwen3.8-27B** UD quants on mid/large boxes (Dual RTX 3.8 ✅ tested) |
| Input window | Hardware guide `--ctx-size` = Pi `contextWindow`. Multi-agent research often needs **≫32k** |
| Output ceiling | Pi `maxTokens` ≤ `--n-predict` (report runs: **8192–16384** so synthesis is not truncated) |
| Thinking | **`--reasoning off`** (+ budget 0). Prefer over deprecated `enable_thinking` kwargs alone |
| Tools | **No DRY**; presence **0** for path-heavy agents; see [agentic harnesses](agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware) |
| KV | Prefer **q8/q8** while it fits; turbo **V** only to buy context (K stays high precision) |

Example field-validated pins (do not copy numbers across hardware without reading that guide): [Windows RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) (24 GB) · [Dual RTX 6000 Qwen3.8](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) (192 GB, ✅ tested). Other 3.8 ports: [DGX Spark](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) · [M5 Pro](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md).

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
| `maximum output token limit` | Raise **both** `--n-predict` and Pi `maxTokens` (e.g. 8192–16384); or write `reports/*.md` and summarize in chat; open run JSON if synthesis finished |
| Path soup / tool loops on Qwen3.6 / 3.8 | No DRY; tool sampling; q8 V before turbo — [agentic harnesses](agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware) |
| Turn-1 garbage after turbo V | New session; fall back to **q8/q8** at your pin; A/B only V type |
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

**Last Updated:** August 14, 2026 (Dual RTX Qwen3.8 marked ✅ Tested)
