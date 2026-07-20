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
- For **Qwen3.6**, guides disable thinking (`--reasoning off` + `enable_thinking:false`) so agents get normal `message.content` instead of empty replies.

The [M4 MacBook Air Qwen guide](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) is the reference for this agent-facing layout.

## Multi-agent workflows, Tavily & research graphs

Once Pi points at your local server, you can layer **dynamic workflows** (parallel specialists, synthesis) and optional **Tavily** web search for structured reports written under project **`reports/`**. Packages, launch (`TAVILY_API_KEY`), layout, and gotchas (context overflow, max tokens, Qwen tool paths) are in **[Pi Coding Agent graphs](pi-coding-agent-graphs.md)**.

> Tavily is cloud search: the model stays local; search traffic does not. Skip it for pure offline use.
