# Autarkic-LLM

> 🤖 **Setting up a machine? Don't read this whole repo — let an AI do it.** Prefer **Grok**. Copy the prompt in **[`ai-assisted-setup.md`](ai-assisted-setup.md)**, paste your hardware at the bottom, and it will use this repo to generate your build, `llama-server` command, model download, and Pi Coding Agent `models.json`.

**Per-machine llama.cpp configs for running Muse Glimmer, Qwen3.8, Qwen3.6, and Gemma 4 fully offline.**

Each hardware guide has the exact build flags and `llama-server` command for one machine, with a model and quant picked to fit its memory: no cloud, no API keys, nothing leaving the box. Configs marked **Tested** were run on that hardware; the rest are starting points (see the table below).

**Why Grok for this repo.** This project is about running open weights on *your* GPU/RAM. **Grok** is the assistant that actually helps with that — local llama.cpp, quants, KV, and a box that does not phone home. **OpenAI and Anthropic** are API companies: their products and docs are built around calling *their* models in the cloud, not standing up an offline `llama-server` on hardware you own. They are not a substitute for this repo. The setup prompt still works in other chats if that is what you have; Grok is the one this project favors.

## Approach

*Autarky* is self-sufficiency: a machine that runs its own models with nothing leaving it. In practice that means the largest model/quant that fits, KV-cache and attention tuned per backend, and sampling that holds up for agentic work (coding agents, multi-agent setups). To point an agent at a running server, see [Agentic Harnesses](agentic-harnesses.md).

## Current Focus

- Primary engine: **llama-cpp-turboquant** (the TurboQuant fork of llama.cpp); build it via [`local-setup.md`](local-setup.md)
- Preferred models: **Qwen3.8-27B** (dense VLM, Unsloth UD quants) on roomier boxes — **✅ tested** on Dual RTX 6000 day-of-release; **Muse Glimmer 30B** (Meta, Apache 2.0, Unsloth UD) as a Dual RTX starting point (⚠️ untested); **Qwen3.6** dense + MoE where still the tested path; Gemma 4 E2B for edge devices
- Emphasis on KV-cache optimization (TurboQuant), flash attention, agent-friendly Qwen settings (thinking off, pinned context), Muse Glimmer settings (thinking **cannot** be switched off — `reasoning_strength` + clean `reasoning_content`), and stable sampling (details in the [deep dive](llama-cpp-turboquant.md))
- **Pi Coding Agent + dense Qwen 27B (3.6 / 3.8):** cross-hardware lessons (two token limits, no DRY, K/V policy, hybrid flags) in [agentic harnesses](agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). **Muse Glimmer + Pi** is a different row: [Muse Glimmer 30B + Pi](agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent). Multi-agent research in [Pi graphs](_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)

### Qwen3.8 (2026-08-14)

[Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) shipped **2026-08-14**. Hardware guides reuse each box’s **tested Qwen3.6** knobs (same pin, KV, Pi shape). **Dual RTX 6000 is already ✅ Tested on Qwen3.8 + Pi** the same day:

| Machine | Backend | Status | Qwen3.8 guide |
| --- | --- | --- | --- |
| Dual RTX 6000 Pro Max-Q (192 GB) | CUDA | ✅ **Tested** (2026-08-14, Pi agent) | [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) — Q8 @ 262k q8/q8 |
| DGX Spark Founders Edition (128 GB) | CUDA (GB10) | ⚠️ Untested (ported from 3.6) | [DGX-Spark-Qwen3.8.md](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) — Q6 @ 262k q8/turbo4 |
| MacBook Pro M5 (48 GB) | Metal | ⚠️ Untested (ported from 3.6) | [M5-MacBook-Pro-Qwen3.8.md](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md) — Q5 @ 196k q8/q8 |

Use a **fresh** turboquant build (arch tag `qwen35`). For untested ports: smoke-test load → first decode → Pi tools, then report results. GGUF names and the Q8→Q4 ladder: [`local-setup.md`](local-setup.md#understanding-gguf-quants-why-so-many-files). MTP / thinking / vision optionals: [Dual RTX Qwen3.8](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md#qwen38-optionals). Catalog: [`local-setup.md`](local-setup.md#model-catalog-hugging-face).

### Muse Glimmer 30B (2026-08)

[Muse Glimmer 30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) is Meta Superintelligence Labs’ open **~30B** dense VLM (Apache 2.0, arch `muse-glimmer`). Native context **131072** (Unsloth: up to **262144**). Needs llama.cpp **`b10353+`**. **Do not** copy Qwen `--reasoning off` onto it.

| Machine | Backend | Status | Muse Glimmer guide |
| --- | --- | --- | --- |
| Dual RTX 6000 Pro Max-Q (192 GB) | CUDA | ⚠️ Untested (researched 2026-08-14) | [Dual-RTX6000-Muse-Glimmer.md](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) — Q8 @ 131k q8/q8 |

## Hardware Configurations Included

**Tested** = run on the physical hardware by the maintainer. **Untested** = best-effort config from model size + llama.cpp options (including remaining Qwen3.8 ports); figures are estimates pending community reports.

| Hardware | Memory | Backend | Model | Tested | Guide |
| --- | --- | --- | --- | --- | --- |
| Jetson Orin Nano Super | 8 GB | CUDA (sm_87) | [Gemma 4 E2B Q4_K_S](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) | ⚠️ Untested | [guide](Jetson-Orin-Nano-Super/Jetson-Orin-Gemma4-E2B.md) |
| M4 Mac Mini | 16 GB | Metal | [Gemma 4 E2B Q4_K_S](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) (recommended) | ⚠️ Untested | [guide](M4-Mac-Mini-16GB/M4-Mac-Mini-Gemma-4-E2B.md) |
| M4 Mac Mini (experimental) | 16 GB | Metal | [Qwen3.6-35B-A3B UD-IQ2_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) (tight, turbo2 V, ~8k start) | ⚠️ Untested | [guide](M4-Mac-Mini-16GB/M4-Mac-Mini-Qwen3.6.md) |
| M2 Mac Mini | 16 GB | Metal | [Gemma 4 E2B Q4_K_S](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) (recommended) | ⚠️ Untested | [guide](M2-Mac-Mini-16GB/M2-Mac-Mini-Gemma-4-E2B.md) |
| M2 Mac Mini (experimental) | 16 GB | Metal | [Qwen3.6-35B-A3B UD-IQ2_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) (tight, turbo2 V, ~8k start) | ⚠️ Untested | [guide](M2-Mac-Mini-16GB/M2-Mac-Mini-Qwen3.6.md) |
| AMD 7900 XTX | 24 GB | Vulkan | [Qwen3.6-27B IQ4_NL](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) (tight) | ✅ Tested | [guide](AMD-7900-XTX/7900-XTX-Qwen3.6-27b.md) |
| AMD 7900 XTX | 24 GB | Vulkan | [Qwen3.6-35B-A3B IQ4_XS](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) | ✅ Tested | [guide](AMD-7900-XTX/7900-XTX-Qwen3.6-35b-a3b.md) |
| MacBook Air M4 | 24 GB | Metal | [Qwen3.6-35B-A3B UD-IQ4_NL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) (MoE, turbo2 V, 61k ctx) | ✅ Tested | [guide](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) |
| Windows RTX 4090 (WSL2) | 24 GB | CUDA | [Qwen3.6-27B UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) (96k q8/q8 Pi agent) | ✅ Tested | [guide](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) |
| MacBook Pro M5 | 48 GB | Metal | [Qwen3.6-27B UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) (196k ctx) | ✅ Tested | [guide](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.6.md) |
| MacBook Pro M5 | 48 GB | Metal | [Qwen3.8-27B UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) (196k q8/q8, ported from 3.6) | ⚠️ Untested | [guide](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md) |
| DGX Spark Founders Edition | 128 GB | CUDA (GB10) | [Qwen3.6-27B UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) | ✅ Tested | [guide](DGX-Spark-128GB/DGX-Spark-Qwen3.6.md) |
| DGX Spark Founders Edition | 128 GB | CUDA (GB10) | [Qwen3.8-27B UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) (262k q8/turbo4, ported from 3.6) | ⚠️ Untested | [guide](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Qwen3.6-27B UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) (262k q8/q8 Pi agent) | ✅ Tested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.6.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Qwen3.8-27B UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) (262k q8/q8 Pi agent) | ✅ Tested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Muse Glimmer 30B UD-Q8_K_XL](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/tree/main) (131k q8/q8, DFlash optional) | ⚠️ Untested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) |

## Quick Start

1. Find your row in the table. Open that guide — it is the daily recipe (download, cmake, `llama-server`, Pi `models.json`).
2. If you have not built the engine yet, do prerequisites in [`local-setup.md`](local-setup.md), then use **that guide’s** cmake (backend/arch live there).
3. Copy the guide’s `models.json` to **`~/.pi/agent/models.json`**. Match `contextWindow` to `--ctx-size` and `maxTokens` to `--n-predict`. [Agentic harnesses](agentic-harnesses.md).

**How to read a hardware guide:** pin table at the top → Download → Build → PRIMARY command → Confirm → Pi JSON → this-box fallbacks. Essays (GGUF names, flag encyclopedia, Pi theory) live in `local-setup.md`, `llama-cpp-turboquant.md`, and `agentic-harnesses.md`. Qwen3.6 and Qwen3.8 are **siblings** on the same machine, not replacements.

Your hardware is not in the table? Use [`ai-assisted-setup.md`](ai-assisted-setup.md). New to the words? [Glossary](glossary.md). Multi-agent / Tavily: [Pi Coding Agent graphs](_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md).

> **Note on "offline":** the model never phones home. Hardware guides default to **`--host 127.0.0.1`** (loopback only). Use `0.0.0.0` only on a trusted LAN when you deliberately expose the server (no auth). See [Common Best Practices](local-setup.md#5-common-best-practices).

## Documentation

- **[`ai-assisted-setup.md`](ai-assisted-setup.md)**: copy-paste prompt for **Grok** (preferred) to generate your setup from this repo
- **[`local-setup.md`](local-setup.md)**: prerequisites, clone & build, model catalog, [GGUF quant naming + Q4–Q8 ladder](local-setup.md#understanding-gguf-quants-why-so-many-files), download, `models.json` integration
- **[`llama-cpp-turboquant.md`](llama-cpp-turboquant.md)**: deep dive into fork internals, TurboQuant tiers, and a flag-by-flag `llama-server` reference (with a key-learnings TL;DR)
- **[`agentic-harnesses.md`](agentic-harnesses.md)**: Pi / OpenClaw / Hermes, ranked for local use, and how to connect them
- **[`_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md`](_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)**: Pi workflows/graphs, Tavily, and the example [`search-topic-research`](_Pi-Coding-Agent-Graphs/example-skills/search-topic-research/) skill
- **[`glossary.md`](glossary.md)**: plain-language terms + curated further reading
- **Hardware guides**: exact per-machine build flags and `llama-server` command (linked in the table above)

```text
Autarkic-LLM/
├── README.md
├── ai-assisted-setup.md            # Copy-paste prompt: let an LLM generate your setup
├── local-setup.md                  # Prerequisites, GGUF names, catalog, download
├── llama-cpp-turboquant.md         # Fork deep dive + full flag reference
├── agentic-harnesses.md            # Pi / OpenClaw / Hermes
├── _Pi-Coding-Agent-Graphs/
│   ├── pi-coding-agent-graphs.md
│   └── example-skills/search-topic-research/
├── glossary.md
├── AMD-7900-XTX/                   # Vulkan · Qwen3.6 MTP (tested)
├── DGX-Spark-128GB/                # 3.6 tested · 3.8 port untested
├── Dual-RTX6000-192GB/             # 3.6 + 3.8 tested · Muse untested
├── M5-MacBook-Pro-48GB/            # 3.6 tested · 3.8 port untested
├── M4-MacBook-Air-24GB/
├── M4-Mac-Mini-16GB/
├── M2-Mac-Mini-16GB/
├── Win-RTX4090-24GB/               # WSL2 paths: ~/AIML, ~/GitHub
└── Jetson-Orin-Nano-Super/         # Jetson paths: ~/models/...
```

**Paths:** Linux/macOS guides use `~/Documents/AIML/models` and `~/Documents/GitHub/llama-cpp-turboquant`. **Windows is WSL2** (`~/AIML`, `~/GitHub`). Jetson uses `~/models/…`. Any path works if `--model` matches.

This repository is intentionally pragmatic. Settings for **Tested** hardware have been validated on the physical machine; **Untested** configs are careful starting points and may need tuning. Corrections and results are welcome via issues/PRs.

**Last Updated:** 2026-08-20 (recipes + GGUF skip-box; Dual RTX 3.8 optionals are the 3.8 appendix)  
**Maintained by:** August Sturm  
**License:** see [LICENSE](LICENSE)
