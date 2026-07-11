# Autarkic-LLM

> 🤖 **Setting up a machine? Don't read this whole repo — let an AI do it.** Copy the prompt in **[`ai-assisted-setup.md`](ai-assisted-setup.md)** into Claude Code, Copilot, ChatGPT, Grok, or Gemini, paste your hardware at the bottom, and it'll use this repo as reference to generate your exact build, `llama-server` command, model download, and Pi Coding Agent `models.json`. Works whether or not you've done this before.

**Per-machine llama.cpp configs for running Qwen3.6 and Gemma 4 fully offline.**

Each hardware guide has the exact build flags and `llama-server` command for one machine, with a model and quant picked to fit its memory: no cloud, no API keys, nothing leaving the box. Configs marked **Tested** were run on that hardware; the rest are starting points (see the table below).

## Approach

*Autarky* is self-sufficiency: a machine that runs its own models with nothing leaving it. In practice that means the largest model/quant that fits, KV-cache and attention tuned per backend, and sampling that holds up for agentic work (coding agents, multi-agent setups). To point an agent at a running server, see [Agentic Harnesses](agentic-harnesses.md).

## Current Focus

- Primary engine: **llama-cpp-turboquant** (the TurboQuant fork of llama.cpp); build it via [`local-setup.md`](local-setup.md)
- Preferred models: Qwen3.6 series (dense + MoE) with Unsloth UD quants; Gemma 4 E2B for edge devices
- Emphasis on KV-cache optimization (TurboQuant), flash attention, agent-friendly Qwen settings (thinking off, pinned context), and stable sampling (details in the [deep dive](llama-cpp-turboquant.md))

## Hardware Configurations Included

**Tested** = run on the physical hardware by the maintainer. **Untested** = best-effort config from model size + llama.cpp options; figures are estimates pending community reports.

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
| Windows RTX 4090 (WSL2) | 24 GB | CUDA | [Qwen3.6-27B UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) | ⚠️ Untested | [guide](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) |
| MacBook Pro M5 | 48 GB | Metal | [Qwen3.6-27B UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) (196k ctx) | ✅ Tested | [guide](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.6.md) |
| DGX Spark Founders Edition | 128 GB | CUDA (GB10) | [Qwen3.6-27B UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) | ✅ Tested | [guide](DGX-Spark-128GB/DGX-Spark-Qwen3.6.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Qwen3.6-27B UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) | ⚠️ Untested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.6.md) |

## Quick Start

1. **Build the engine and get a model**: follow [`local-setup.md`](local-setup.md) for prerequisites, the clone & build, and the model catalog + download commands.
2. **Run your hardware's command**: open your machine's guide from the table above and run its exact `llama-server` command.
3. **(Optional) Drive it with an agent**: copy the complete `models.json` from your hardware guide to **`~/.pi/agent/models.json`** (see [agentic harnesses](agentic-harnesses.md)).

New to local inference? Start with the [Glossary](glossary.md).

> **Note on "offline":** the model never phones home. Hardware guides default to **`--host 127.0.0.1`** (loopback only). Use `0.0.0.0` only on a trusted LAN when you deliberately expose the server (no auth). See [Common Best Practices](local-setup.md#5-common-best-practices).

## Documentation

- **[`ai-assisted-setup.md`](ai-assisted-setup.md)**: a copy-paste prompt that has any cloud LLM generate your setup from this repo (start here)
- **[`local-setup.md`](local-setup.md)**: prerequisites, clone & build, model catalog + download, `models.json` integration
- **[`llama-cpp-turboquant.md`](llama-cpp-turboquant.md)**: deep dive into fork internals, TurboQuant tiers, and a flag-by-flag `llama-server` reference (with a key-learnings TL;DR)
- **[`agentic-harnesses.md`](agentic-harnesses.md)**: Pi / OpenClaw / Hermes, ranked for local use, and how to connect them
- **[`glossary.md`](glossary.md)**: plain-language terms + curated further reading
- **Hardware guides**: exact per-machine build flags and `llama-server` command (linked in the table above)

```text
Autarkic-LLM/
├── README.md
├── ai-assisted-setup.md            # Copy-paste prompt: let an LLM generate your setup
├── local-setup.md                  # Prerequisites, build, model catalog & download
├── llama-cpp-turboquant.md         # Fork deep dive + full flag reference
├── agentic-harnesses.md            # Pi / OpenClaw / Hermes
├── glossary.md                     # Terms + further reading
├── DGX-Spark-128GB/
│   └── DGX-Spark-Qwen3.6.md
├── Dual-RTX6000-192GB/
│   └── Dual-RTX6000-Qwen3.6.md
├── M5-MacBook-Pro-48GB/
│   └── M5-MacBook-Pro-Qwen3.6.md
├── M4-MacBook-Air-24GB/
│   └── M4-MacBook-Air-Qwen3.6.md
├── M4-Mac-Mini-16GB/
│   ├── M4-Mac-Mini-Qwen3.6.md
│   └── M4-Mac-Mini-Gemma-4-E2B.md
├── M2-Mac-Mini-16GB/
│   ├── M2-Mac-Mini-Qwen3.6.md
│   └── M2-Mac-Mini-Gemma-4-E2B.md
├── Win-RTX4090-24GB/
│   └── Windows-RTX4090-Qwen3.6.md
└── Jetson-Orin-Nano-Super/
    └── Jetson-Orin-Gemma4-E2B.md
```

This repository is intentionally pragmatic. Settings for **Tested** hardware have been validated on the physical machine; **Untested** configs are careful starting points and may need tuning. Corrections and results are welcome via issues/PRs.

**Last Updated:** July 2026  
**Maintained by:** August Sturm  
**License:** see [LICENSE](LICENSE)
