# Agentic Harnesses

How to actually *use* the models in this repo for agentic work (coding agents, Godot development, multi-agent systems), running fully offline against your local `llama-server`.

An **agentic harness** is the software wrapper around an LLM that turns it from a chatbot into an autonomous worker. It runs the loop: the model calls tools (read/write files, run shell commands, search, browse), the harness executes them, feeds the results back, and iterates until the task is done, while managing context/memory, tool definitions, and permissions (and, in multi-agent systems, coordinating several specialized agents). The harness holds no intelligence of its own; it points at a model endpoint. That's why any of these can target a local `llama-server` exposing an OpenAI-compatible API (the complete `models.json` in each hardware guide) and run **fully offline**.

## Ranked by how well they work with local models only

1. **[Pi Coding Agent](https://pi.dev)** (Hugging Face): the most local-first of the three and the harness these guides' full `models.json` files target. Installed as an npm package, configured with a small JSON file pointing at your local server, with a llama.cpp extension for browsing/loading/switching served models. A Claude Code-style experience with no API costs and nothing leaving the machine. See also Hugging Face's [Local Agents with llama.cpp](https://huggingface.co/docs/hub/en/agents-local) guide.
2. **[OpenClaw](https://docs.openclaw.ai)**: a persistent local agent with first-class support for any OpenAI-compatible endpoint (llama.cpp, Ollama, LM Studio, vLLM) and abundant "run free with a local LLM" guides. More capable/heavyweight than Pi. ⚠️ Treat its exec permissions carefully: an agent with shell access is effectively running code on your machine; scope it to a project folder, never your home directory or sudo. ([local-models docs](https://docs.openclaw.ai/gateway/local-models))
3. **[Hermes](https://github.com/nousresearch/hermes-agent)** (Nous Research): a self-hosted, self-improving multi-agent system (Leader/Specialist orchestration over MCP/ACP, persistent memory, auto-generated skills). Model-agnostic and runnable fully local via Ollama or any OpenAI-compatible server, but oriented toward hybrid/cloud routing (Nous Portal, OpenRouter), so local-only is supported rather than the default. ([project site](https://hermes-agent.nousresearch.com/))

These are independent third-party projects; local-model support evolves, so check each project's current docs. All of them connect the same way: point them at the `baseUrl` of your running `llama-server`.

## Connecting to your server

Each hardware guide includes a **complete** Pi `models.json` — the full `providers` → `llama-cpp` object with `baseUrl`, `api`, `apiKey`, and your model. Copy the entire JSON block into **`~/.pi/agent/models.json`** as-is (`mkdir -p ~/.pi/agent` first if needed). Do not assemble a wrapper by hand.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "your-model",
          "name": "Your Model Name",
          "contextWindow": 61440,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

(Shape only — use the numbers from your hardware guide.)

**Keep client and server aligned:**

- `contextWindow` = the server’s real `n_ctx_seq` (your pinned `--ctx-size` when using `--fit off`).
- `maxTokens` ≤ `--n-predict`.
- Restart **both** `llama-server` and Pi after changing either side. Pi’s status bar must match the pin (stale `models.json` is a common failure mode).

Hardware-specific numbers always come from **your** guide (pin table + `models.json`). Guide *shape* (download → cmake → PRIMARY → JSON): [Dual RTX Qwen3.8](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md). Tight Metal / `--fit` crush / turbo2 V: [M4 Air](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md). Field-validated **Pi + dense Qwen 27B** lessons below cover **Qwen3.6-27B** and **Qwen3.8-27B** (Dual RTX 3.8 is field-tested). **Muse Glimmer is a different model** — do not apply the Qwen `--reasoning off` row; see [Muse Glimmer 30B + Pi](#muse-glimmer-30b--pi-coding-agent).

## Qwen3.6-27B + Pi Coding Agent (cross-hardware)

> **Also applies to Qwen3.8-27B.** Same hybrid thinking, tool-loop risks, and two token limits. **✅ Field-tested on Dual RTX 6000** ([guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md), 2026-08-14). Other 3.8 ports still ⚠️ untested: [DGX Spark](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) · [M5 Pro](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md). Overview: [README — Qwen3.8](README.md#qwen38-2026-08-14).

These apply whenever Pi drives **dense Qwen 27B** (**Qwen3.6-27B** or **Qwen3.8-27B**) via this repo’s server. Exact `--ctx-size`, quant, and KV types stay in the **hardware guide**.

### Two different limits

| Knob | Server | Pi | Failure mode |
| --- | --- | --- | --- |
| **Input window** | `--ctx-size` | `contextWindow` | `request (N) exceeds the available context size` |
| **Output per turn** | `--n-predict` | `maxTokens` | `Model stopped because it reached the maximum output token limit` |

Raising one does not fix the other. Multi-agent workflows (Tavily, parallel specialists, long skills) often need **larger input** than a short chat (field overflows past **32k/64k**, single requests **~70k+** on heavy runs). Long market reports often need **larger output** than 4k–8k.

**On Qwen3.8, thinking tokens also count against `--n-predict` / `maxTokens`.** `--reasoning off` is not the same as a tiny output cap. If you turn thinking **on** with these Unsloth GGUFs: unset `reasoning_effort` typically renders as **XHIGH**; start at **`low`**; `medium` may have no dedicated template branch; leave `--reasoning-preserve` **off** for Pi. Details: [Dual RTX Qwen3.8 optionals](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md#qwen38-optionals).

### Server flags that matter for Pi tools

| Prefer for Pi + dense Qwen 27B (3.6 / 3.8) | Avoid for tool/agent sessions |
| --- | --- |
| **`--reasoning off`** (+ `--reasoning-budget 0`) so Pi gets `message.content` / tools | Relying only on deprecated `--chat-template-kwargs '{"enable_thinking":false}'` when the server warns to use `--reasoning`. Ignore verbosity-3 `consider … --reasoning-preserve` for Pi — that flag re-injects **server** think traces into history; Pi is the harness and thinking is already off |
| **`--fit off`** + pinned `--ctx-size` | Bare `--fit on` (can crush context) |
| **No DRY** (`--dry-multiplier`, …) | DRY — causes path/name corruption on Qwen3.x tool loops ([llama.cpp #20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| Tool-oriented sampling: e.g. **`temp 0.6`**, **`top_p 0.95`**, **`top_k 20`**, **`presence_penalty 0`**, **`repeat_penalty 1.0`** | High **presence** (e.g. 1.5 chat non-thinking / official Qwen3.8 instruct cards) during path-heavy tool use — can warp reused path tokens |
| Hybrid Qwen: omit checkpoint flags; consider **`--cache-ram 0`** if multi-turn state looks corrupt | Assuming checkpoints speed hybrid Gated-DeltaNet (often they do not — see [turboquant deep dive](llama-cpp-turboquant.md#prompt-cache--checkpointing)) |

### KV cache (K vs V) — stability vs context

Weights are fixed; **KV grows with context**. **K** (keys) routes attention and should stay precise (`q8_0` / f16). **V** (values) is more compressible (`q8_0` → `turbo4` → `turbo3` → `turbo2`). Policy:

1. Prefer **`q8_0` / `q8_0`** while it fits and tools stay coherent.  
2. Raise **`--ctx-size`** (and Pi `contextWindow`) when prompts overflow.  
3. Only then compress **V** with turbo* if VRAM/RAM is tight — smoke-test real `ls`/`read` paths on a **new** session first.  
4. Never crush **K** before **V**.

TurboQuant **fork** ≠ must use turbo **types**. Roomier machines keep high-precision KV — e.g. [M5 48 GB](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.6.md) (~196k q8/q8; [3.8 port](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md)) and [Dual RTX 6000 192 GB](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) (**full 262k q8/q8**, **field-tested with Pi on Qwen3.8** and [3.6](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.6.md)). Tighter boxes (Air, 24 GB CUDA under load) use turbo V to buy window. Field-validated 24 GB CUDA agent baseline: [Windows RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) (**96k q8/q8**, large `maxTokens`).

### Operational habits

- **New Pi session** (`/new`) after path soup, phrase loops, or wrong-tool thrash — compaction does not un-poison history.  
- After changing server pins, **restart Pi** so the status bar matches.  
- Prefer agents **writing long reports to disk** (`reports/`) and summarizing in chat when output length is the bottleneck.  
- Multi-agent graphs, Tavily, and the example research skill: **[Pi Coding Agent graphs](_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)**.

## Muse Glimmer 30B + Pi Coding Agent

> ⚠️ **Not field-tested in this repo yet.** Hardware pin: [Dual RTX 6000 Muse Glimmer](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md). Two-limit / no-DRY / `--fit off` rules still apply. **Qwen thinking-off does not.**

Muse Glimmer is Meta’s ~30B dense VLM (arch `muse-glimmer`, llama.cpp **`b10353+`**). It is trained as a reasoning + tool model. Official docs: [Meta llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/) · [prompting](https://dev.meta.ai/docs/muse-glimmer/prompting) · [Unsloth](https://unsloth.ai/docs/models/muse-glimmer).

| Prefer for Pi + Muse Glimmer | Avoid |
| --- | --- |
| **`--jinja`** (embedded Muse template; required for ATEM tools + reasoning split) | Skipping `--jinja`, or passing `--chat-template-file` |
| **`--chat-template-kwargs '{"reasoning_strength":"high"}'`** (`low` / `medium` / `high` / `xhigh`) | **`--reasoning off`** — **no-op** on this template. Same for `"reasoning_effort": "none"` |
| Default server split: thinking in `reasoning_content`, answer/tools in `content` | `--reasoning-format none` (dumps thinking into `content` and confuses Pi) |
| **`--n-predict` large enough** (Dual RTX primary **32768**) — thinking counts against the output cap | 4k–8k `maxTokens` on `high`/`xhigh` (empty `content`, `finish_reason: length`) |
| Official sampling: **`temp 1.0`**, **`top_p 0.95`**, **`top_k 64`**, **`presence 0`**, **`repeat 1.0`** | Blindly copying the Qwen Pi row (`temp 0.6` / `top_k 20`) or Qwen instruct **presence 1.5** |
| **`--parallel 1`** so one session gets the full `--ctx-size` | Raising `-np` without scaling `-c` (silent empty answers when a slot fills) |
| Leave the eom token out of custom stop lists (it is end-of-message, not end-of-turn) | Stopping on that token (collapses tool calling) |
| **No DRY** | DRY on path-heavy tool loops ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |

Need a **fresh** turboquant / llama.cpp (`LLM_ARCH_MUSE_GLIMMER`). Optional speed: **DFlash** (`--spec-type draft-dflash` + `dflash-kquant.gguf`), not Qwen MTP.

## Multi-agent workflows, Tavily & research graphs

Once Pi points at your local server, you can layer **dynamic workflows** (parallel specialists, synthesis) and optional **Tavily** web search. The recommended research path is the generic skill [`search-topic-research`](_Pi-Coding-Agent-Graphs/example-skills/search-topic-research/) (host Tavily → pack → four-phase graph → dated report). Packages, launch (`TAVILY_API_KEY`), skill install, and workflow gotchas are in **[Pi Coding Agent graphs](_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)**.

> Tavily is cloud search: the model stays local; search traffic does not. Skip it for pure offline use.
