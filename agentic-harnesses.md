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

Hardware-specific numbers always come from **your** guide. The [M4 MacBook Air Qwen guide](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) is the reference for agent-facing *layout* (host, fit, thinking, `models.json` shape). Field-validated **Pi + Qwen3.6-27B** agent lessons (any hardware) are below.

## Qwen3.6-27B + Pi Coding Agent (cross-hardware)

These apply whenever Pi drives **Qwen3.6-27B** (dense) via this repo’s server. Exact `--ctx-size`, quant, and KV types stay in the **hardware guide**.

### Two different limits

| Knob | Server | Pi | Failure mode |
| --- | --- | --- | --- |
| **Input window** | `--ctx-size` | `contextWindow` | `request (N) exceeds the available context size` |
| **Output per turn** | `--n-predict` | `maxTokens` | `Model stopped because it reached the maximum output token limit` |

Raising one does not fix the other. Multi-agent workflows (Tavily, parallel specialists, long skills) often need **larger input** than a short chat (field overflows past **32k/64k**, single requests **~70k+** on heavy runs). Long market reports often need **larger output** than 4k–8k.

### Server flags that matter for Pi tools

| Prefer for Pi + Qwen3.6 | Avoid for tool/agent sessions |
| --- | --- |
| **`--reasoning off`** (+ `--reasoning-budget 0`) so Pi gets `message.content` / tools | Relying only on deprecated `--chat-template-kwargs '{"enable_thinking":false}'` when the server warns to use `--reasoning` |
| **`--fit off`** + pinned `--ctx-size` | Bare `--fit on` (can crush context) |
| **No DRY** (`--dry-multiplier`, …) | DRY — causes path/name corruption on Qwen3.6 tool loops ([llama.cpp #20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| Tool-oriented sampling: e.g. **`temp 0.6`**, **`top_p 0.95`**, **`top_k 20`**, **`presence_penalty 0`**, **`repeat_penalty 1.0`** | High **presence** (e.g. 1.5 chat non-thinking) during path-heavy tool use — can warp reused path tokens |
| Hybrid Qwen: omit checkpoint flags; consider **`--cache-ram 0`** if multi-turn state looks corrupt | Assuming checkpoints speed hybrid Gated-DeltaNet (often they do not — see [turboquant deep dive](llama-cpp-turboquant.md#prompt-cache--checkpointing)) |

### KV cache (K vs V) — stability vs context

Weights are fixed; **KV grows with context**. **K** (keys) routes attention and should stay precise (`q8_0` / f16). **V** (values) is more compressible (`q8_0` → `turbo4` → `turbo3` → `turbo2`). Policy:

1. Prefer **`q8_0` / `q8_0`** while it fits and tools stay coherent.  
2. Raise **`--ctx-size`** (and Pi `contextWindow`) when prompts overflow.  
3. Only then compress **V** with turbo* if VRAM/RAM is tight — smoke-test real `ls`/`read` paths on a **new** session first.  
4. Never crush **K** before **V**.

TurboQuant **fork** ≠ must use turbo **types**. Roomier machines (e.g. [M5 48 GB](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.6.md) ~196k q8/q8) keep high-precision KV; tighter boxes (Air, 24 GB CUDA under load) use turbo V to buy window. Field-validated 24 GB CUDA agent baseline: [Windows RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) (**96k q8/q8**, large `maxTokens`).

### Operational habits

- **New Pi session** (`/new`) after path soup, phrase loops, or wrong-tool thrash — compaction does not un-poison history.  
- After changing server pins, **restart Pi** so the status bar matches.  
- Prefer agents **writing long reports to disk** (`reports/`) and summarizing in chat when output length is the bottleneck.  
- Multi-agent graphs, Tavily, and `reports/` layout: **[Pi Coding Agent graphs](pi-coding-agent-graphs.md)**.

## Multi-agent workflows, Tavily & research graphs

Once Pi points at your local server, you can layer **dynamic workflows** (parallel specialists, synthesis) and optional **Tavily** web search for structured reports under project **`reports/`**. Packages, launch (`TAVILY_API_KEY`), layout, and workflow gotchas are in **[Pi Coding Agent graphs](pi-coding-agent-graphs.md)**.

> Tavily is cloud search: the model stays local; search traffic does not. Skip it for pure offline use.
