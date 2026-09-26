# Autarkic-LLM

> 🤖 **Setting up a machine? Don't read this whole repo — let an AI do it.** Prefer **Grok**. Copy the prompt in **[`ai-assisted-setup.md`](ai-assisted-setup.md)**, paste your hardware at the bottom, and it will use this repo to generate your build, `llama-server` command, model download, and Pi Coding Agent `models.json`.

**Per-machine llama.cpp configs for running Muse Glimmer, Qwen3.8, Qwen3-Coder-Next, Ternary Bonsai 2, Qwen3.6, Gemma 4, LFM2.5, and MiniCPM5 fully offline.**

Each hardware guide has the exact build flags and `llama-server` command for one machine, with a model and quant picked to fit its memory: no cloud, no API keys, nothing leaving the box. Configs marked **Tested** were run on that hardware; the rest are starting points (see the table below).

**Why Grok for this repo.** This project is about running open weights on *your* GPU. **Grok** will help you do that. **OpenAI and Anthropic will not** — they have said so in public, and the local-inference community on X and [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/) has been reporting it for years.

- **Anthropic** has never released Claude weights. CEO Dario Amodei wrote that capable open-weight models “do potentially present a higher risk than closed models, because it is very difficult to apply guardrails to them or monitor their usage, and once weights are released they cannot be withdrawn” ([Anthropic, Jul 2026](https://www.anthropic.com/news/position-open-weights-models)). They also declined to sign the industry letter defending open weights. That is a closed-weight company telling you Claude is not something you run on your machine.
- **OpenAI** still sells ChatGPT as the product. Even when announcing gpt-oss, Sam Altman said “we believe most people will want to use a convenient service like ChatGPT” ([X, Aug 2025](https://x.com/sama/status/1952778518225723434)). Frontier models stay behind an API.
- **On X**, local-LLM users and [LocalLLaMA mods](https://x.com/TheAhmadOsman/status/2040860574670533078) put it plainly: those labs optimize for vendor lock-in. People regularly report ChatGPT/Claude refusing, derailing, or going “I can’t help you set up…” when asked for llama.cpp / Ollama / GGUF help.

Grok is the assistant this repo is written to pair with. The setup prompt still works in other chats if that is all you have — do not expect those labs to care about your offline box.

## Approach

*Autarky* is self-sufficiency: a machine that runs its own models with nothing leaving it. In practice that means the largest model/quant that fits, KV-cache and attention tuned per backend, and sampling that holds up for agentic work (coding agents, multi-agent setups). To point an agent at a running server, see [Agentic Harnesses](agentic-harnesses.md).

## Current Focus

- Primary engine: **llama-cpp-turboquant** (the TurboQuant fork of llama.cpp); build it via [`local-setup.md`](local-setup.md)
- Preferred models: **Qwen3.8-27B** (dense VLM, Unsloth UD quants) on roomier boxes — **✅ tested** on Dual RTX 6000 day-of-release; **Qwen3-Coder-Next** (80B-A3B coding MoE, Unsloth UD; ⚠️ Dual RTX untested, **one 96 GB card**, Q6 @ 262k); **Ternary Bonsai 2 27B** (PrismML ternary pack of the same Qwen3.8 backbone; ⚠️ Dual RTX untested, **PrismML llama.cpp fork** — not turboquant); **Muse Glimmer 30B** (Meta, Apache 2.0, Unsloth UD) as a Dual RTX starting point (⚠️ untested); **Qwen3.6** dense + MoE where still the tested path; **Gemma 4 E2B** for edge devices — **✅ tested** on Jetson Orin Nano Super; **LFM2.5-2.6B** — **✅ tested** on the same Jetson (official Liquid GGUF, always-on thinking, 64k q8/q8); **⚠️ untested** ports on Windows RTX 3090 WSL2 (native **128k** q8/q8, **FA off**) and DGX Spark (native **128k** q8/q8, **FA on**, GB10 `"121"`); **MiniCPM5-2B** on the same Jetson — ⚠️ untested (official OpenBMB GGUF, PRIMARY think **off**, 32k q8/q8)
- **Forge Trinity** (RTX Pro 2000 Blackwell 16 GB + two RTX Pro 6000 Max-Q 96 GB): one `llama-server` per GPU on the existing `sm_120` binary — LFM2.5-2.6B Q8_0 @ 128k on the 16 GB card, Qwen3.8 Q8 and Qwen3-Coder-Next Q6 each on one 96 GB card. ⚠️ untested. Pi runs on other LAN machines through a ufw allowlist. SSH manages the box, including the headless session switch: [Forge-Trinity.md](Forge-Trinity/Forge-Trinity.md)
- Emphasis on KV-cache optimization (TurboQuant), flash attention, agent-friendly Qwen settings (thinking off, pinned context), Muse Glimmer / LFM2.5 settings (template thinking **cannot** be switched off — LFM Pi skills path uses `reasoning` false; traces JSON keeps clean `reasoning_content`), MiniCPM5-2B (OpenBMB documents a Think/No-think **toggle**; Jetson PRIMARY **forces off**, ⚠️ untested), and stable sampling (details in the [deep dive](llama-cpp-turboquant.md))
- **Pi Coding Agent + dense Qwen 27B (3.6 / 3.8):** cross-hardware lessons (two token limits, no DRY, K/V policy, hybrid flags) in [agentic harnesses](agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Dual RTX second card: [Localmaxing](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8-localmaxing.md) (one model per GPU). **Muse Glimmer + Pi** is a different row: [Muse Glimmer 30B + Pi](agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent). **LFM2.5-2.6B + Pi** (always-on `<think>`): [LFM2.5-2.6B + Pi](agentic-harnesses.md#lfm25-26b--pi-coding-agent). **MiniCPM5-2B + Pi** (PRIMARY think **off**; OpenBMB toggle ⚠️ untested): [MiniCPM5-2B + Pi](agentic-harnesses.md#minicpm5-2b--pi-coding-agent). Multi-agent research in [Pi graphs](_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)

### Qwen3.8 (2026-08-14)

[Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) shipped **2026-08-14**. Hardware guides reuse each box’s **tested Qwen3.6** knobs (same pin, KV, Pi shape). **Dual RTX 6000 is already ✅ Tested on Qwen3.8 + Pi** the same day:

| Machine | Backend | Status | Qwen3.8 guide |
| --- | --- | --- | --- |
| Dual RTX 6000 Pro Max-Q (192 GB) | CUDA | ✅ **Tested** (2026-08-14, Pi agent) | [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) — Q8 @ 262k q8/q8 |
| Dual RTX 6000 Pro Max-Q (192 GB) | CUDA | ✅ Dual load + parallel decode (2026-09-03) | [Dual-RTX6000-Qwen3.8-localmaxing.md](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8-localmaxing.md) — Localmaxing (one model per card) |
| DGX Spark Founders Edition (128 GB) | CUDA (GB10) | ⚠️ Untested (ported from 3.6) | [DGX-Spark-Qwen3.8.md](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) — Q6 @ 262k q8/turbo4 |
| MacBook Pro M5 (48 GB) | Metal | ⚠️ Untested (ported from 3.6) | [M5-MacBook-Pro-Qwen3.8.md](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md) — Q5 @ 196k q8/q8 |

Use a **fresh** turboquant build (arch tag `qwen35`). For untested ports: smoke-test load → first decode → Pi tools, then report results. GGUF names and the Q8→Q4 ladder: [`local-setup.md`](local-setup.md#understanding-gguf-quants-why-so-many-files). MTP / thinking / vision optionals: [Dual RTX Qwen3.8](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md#qwen38-optionals). Catalog: [`local-setup.md`](local-setup.md#model-catalog-hugging-face).

### Ternary Bonsai 2 27B (2026-09)

[Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) is PrismML’s ternary (`{−1,0,+1}`) pack of **Qwen3.8-27B** (~5.9 GB `PTQ1_0` / ~7.2 GB `PQ2_0`, Apache 2.0, 262k). **Stock llama.cpp and llama-cpp-turboquant refuse these files.** Need [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, **prism-b10658+**). Guides download **both** packs; PRIMARY is `PQ2_0` (7900 XTX: **HIP**, not that folder’s usual Vulkan). Dual RTX, DGX Spark, and M1 Ultra pin **262k**; 24 GB boxes start at **131k**.

| Machine | Backend | Status | Bonsai 2 guide |
| --- | --- | --- | --- |
| Dual RTX 6000 Pro Max-Q (192 GB) | CUDA | ⚠️ Untested (researched 2026-09-18) | [Dual-RTX6000-Bonsai-2-27B.md](Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md) — PQ2_0 @ 262k q8/q8 (PTQ1_0 A/B) |
| DGX Spark Founders Edition (128 GB) | CUDA (GB10) | ⚠️ Untested (researched 2026-09-18) | [DGX-Spark-Bonsai-2-27B.md](DGX-Spark-128GB/DGX-Spark-Bonsai-2-27B.md) — PQ2_0 @ 262k q8/q8 (PTQ1_0 A/B) |
| M1 Ultra Mac Studio (64 GB) | Metal | ⚠️ Untested (researched 2026-09-18) | [M1-Ultra-Studio-Bonsai-2-27B.md](M1-Ultra-Studio-64GB/M1-Ultra-Studio-Bonsai-2-27B.md) — PQ2_0 @ 262k q8/q8 (PTQ1_0 A/B) |
| Windows RTX 3090 (WSL2) (24 GB) | CUDA (sm_86) | ⚠️ Untested (researched 2026-09-18) | [Windows-RTX3090-Bonsai-2-27B.md](Win-RTX3090-24GB/Windows-RTX3090-Bonsai-2-27B.md) — PQ2_0 @ 131k q8/q8 (PTQ1_0 A/B) |
| AMD 7900 XTX (24 GB) | HIP / ROCm | ⚠️ Untested (researched 2026-09-18) | [7900-XTX-Bonsai-2-27B.md](AMD-7900-XTX/7900-XTX-Bonsai-2-27B.md) — PQ2_0 @ 131k q8/q8 (Vulkan incomplete) |

### Qwen3-Coder-Next (2026-02)

[Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) is Qwen’s **80B-A3B** coding MoE (3B active, arch `qwen3next`, native **262k**, Apache 2.0). **Non-thinking** (no `<think>` blocks). Dual RTX downloads **one** sharded pack: **UD-Q6_K_XL** (~73 GB) on **one** 96 GB card. Official sampling **temp 1.0 / top_p 0.95 / top_k 40 / min_p 0.01**. Need a fresh turboquant (`qwen3next`). The other card can still run the tested [Qwen3.8 primary](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md).

| Machine | Backend | Status | Coder-Next guide |
| --- | --- | --- | --- |
| Dual RTX 6000 Pro Max-Q (192 GB) | CUDA | ⚠️ Untested (researched 2026-09-20) | [Dual-RTX6000-Qwen3-Coder-Next.md](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3-Coder-Next.md) — Q6 @ 262k q8/q8 |

### Muse Glimmer 30B (2026-08)

[Muse Glimmer 30B](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) is Meta Superintelligence Labs’ open **~30B** dense VLM (Apache 2.0, arch `muse-glimmer`). Native context **131072** (Unsloth: up to **262144**). Needs llama.cpp **`b10353+`**. Thinking **cannot** be switched off — use `reasoning_strength`. Guide: [Dual-RTX6000-Muse-Glimmer.md](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md).

| Machine | Backend | Status | Muse Glimmer guide |
| --- | --- | --- | --- |
| Dual RTX 6000 Pro Max-Q (192 GB) | CUDA | ⚠️ Untested (researched 2026-08-14) | [Dual-RTX6000-Muse-Glimmer.md](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) — Q8 @ 131k q8/q8 |

## Hardware Configurations Included

**Tested** = run on the physical hardware by the maintainer. **Untested** = best-effort config from model size + llama.cpp options (including remaining Qwen3.8 ports); figures are estimates pending community reports.

| Hardware | Memory | Backend | Model | Tested | Guide |
| --- | --- | --- | --- | --- | --- |
| Jetson Orin Nano Super | 8 GB | CUDA (sm_87) | [Gemma 4 E2B Q4_K_S](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) (16k q8/q8) | ✅ Tested | [guide](Jetson-Orin-Nano-Super/Jetson-Orin-Gemma4-E2B.md) |
| Jetson Orin Nano Super | 8 GB | CUDA (sm_87) | [LFM2.5-2.6B Q8_0](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/tree/main) (64k q8/q8 Pi agent) | ✅ Tested | [guide](Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md) |
| Jetson Orin Nano Super | 8 GB | CUDA (sm_87) | [MiniCPM5-2B Q8_0](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/tree/main) (32k q8/q8, PRIMARY think off) | ⚠️ Untested | [guide](Jetson-Orin-Nano-Super/Jetson-Orin-MiniCPM5-2B.md) |
| M4 Mac Mini | 16 GB | Metal | [Gemma 4 E2B Q4_K_S](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) (recommended) | ⚠️ Untested | [guide](M4-Mac-Mini-16GB/M4-Mac-Mini-Gemma-4-E2B.md) |
| M4 Mac Mini (experimental) | 16 GB | Metal | [Qwen3.6-35B-A3B UD-IQ2_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) (tight, turbo2 V, ~8k start) | ⚠️ Untested | [guide](M4-Mac-Mini-16GB/M4-Mac-Mini-Qwen3.6.md) |
| M2 Mac Mini | 16 GB | Metal | [Gemma 4 E2B Q4_K_S](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) (recommended) | ⚠️ Untested | [guide](M2-Mac-Mini-16GB/M2-Mac-Mini-Gemma-4-E2B.md) |
| M2 Mac Mini (experimental) | 16 GB | Metal | [Qwen3.6-35B-A3B UD-IQ2_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) (tight, turbo2 V, ~8k start) | ⚠️ Untested | [guide](M2-Mac-Mini-16GB/M2-Mac-Mini-Qwen3.6.md) |
| AMD 7900 XTX | 24 GB | Vulkan | [Qwen3.6-27B IQ4_NL](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) (tight) | ✅ Tested | [guide](AMD-7900-XTX/7900-XTX-Qwen3.6-27b.md) |
| AMD 7900 XTX | 24 GB | Vulkan | [Qwen3.6-35B-A3B IQ4_XS](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) | ✅ Tested | [guide](AMD-7900-XTX/7900-XTX-Qwen3.6-35b-a3b.md) |
| AMD 7900 XTX | 24 GB | HIP / ROCm | [Ternary Bonsai 2 27B PQ2_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main) (131k q8/q8; **PrismML fork**, not Vulkan) | ⚠️ Untested | [guide](AMD-7900-XTX/7900-XTX-Bonsai-2-27B.md) |
| MacBook Air M4 | 24 GB | Metal | [Qwen3.6-35B-A3B UD-IQ4_NL](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) (MoE, turbo2 V, 61k ctx) | ✅ Tested | [guide](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) |
| Windows RTX 3090 (WSL2) | 24 GB | CUDA (sm_86) | [LFM2.5-2.6B Q8_0](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/tree/main) (128k q8/q8, **FA off**; not the 4090 cmake) | ⚠️ Untested | [guide](Win-RTX3090-24GB/Windows-RTX3090-LFM2.5-2.6B.md) |
| Windows RTX 3090 (WSL2) | 24 GB | CUDA (sm_86) | [Ternary Bonsai 2 27B PQ2_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main) (131k q8/q8; PTQ1_0 A/B; **PrismML fork**) | ⚠️ Untested | [guide](Win-RTX3090-24GB/Windows-RTX3090-Bonsai-2-27B.md) |
| Windows RTX 4090 (WSL2) | 24 GB | CUDA | [Qwen3.6-27B UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) (96k q8/q8 Pi agent) | ✅ Tested | [guide](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) |
| MacBook Pro M5 | 48 GB | Metal | [Qwen3.6-27B UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) (196k ctx) | ✅ Tested | [guide](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.6.md) |
| MacBook Pro M5 | 48 GB | Metal | [Qwen3.8-27B UD-Q5_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) (196k q8/q8, ported from 3.6) | ⚠️ Untested | [guide](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md) |
| M1 Ultra Mac Studio | 64 GB | Metal | [Ternary Bonsai 2 27B PQ2_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main) (262k q8/q8; PTQ1_0 A/B; **PrismML fork**) | ⚠️ Untested | [guide](M1-Ultra-Studio-64GB/M1-Ultra-Studio-Bonsai-2-27B.md) |
| DGX Spark Founders Edition | 128 GB | CUDA (GB10) | [Qwen3.6-27B UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) | ✅ Tested | [guide](DGX-Spark-128GB/DGX-Spark-Qwen3.6.md) |
| DGX Spark Founders Edition | 128 GB | CUDA (GB10) | [Qwen3.8-27B UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) (262k q8/turbo4, ported from 3.6) | ⚠️ Untested | [guide](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) |
| DGX Spark Founders Edition | 128 GB | CUDA (GB10) | [LFM2.5-2.6B Q8_0](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/tree/main) (128k q8/q8, **FA on**; GB10 `"121"`, not the Jetson/3090 cmake) | ⚠️ Untested | [guide](DGX-Spark-128GB/DGX-Spark-LFM2.5-2.6B.md) |
| DGX Spark Founders Edition | 128 GB | CUDA (GB10) | [Ternary Bonsai 2 27B PQ2_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main) (262k q8/q8; PTQ1_0 A/B; **PrismML fork**) | ⚠️ Untested | [guide](DGX-Spark-128GB/DGX-Spark-Bonsai-2-27B.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Qwen3.6-27B UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) (262k q8/q8 Pi agent) | ✅ Tested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.6.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Qwen3.8-27B UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) (262k q8/q8 Pi agent) | ✅ Tested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | Localmaxing (one per card): 2× [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main), or 27B + [Coder-30B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/tree/main) / [35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) | ✅ Load+decode (2× Q6) | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8-localmaxing.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Qwen3-Coder-Next UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/tree/main) (80B-A3B, 262k q8/q8, one card) | ⚠️ Untested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3-Coder-Next.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Muse Glimmer 30B UD-Q8_K_XL](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/tree/main) (131k q8/q8, DFlash optional) | ⚠️ Untested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) |
| Dual RTX 6000 Pro Max-Q | 192 GB | CUDA | [Ternary Bonsai 2 27B PQ2_0](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main) (262k q8/q8; PTQ1_0 A/B; **PrismML fork**) | ⚠️ Untested | [guide](Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md) |
| Forge Trinity (RTX Pro 2000 16 GB + 2× RTX Pro 6000 Max-Q 96 GB) | 208 GB | CUDA (sm_120) | One server each: [LFM2.5-2.6B Q8_0](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/tree/main) on the 16 GB card (128k); [Qwen3.8-27B UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) and [Qwen3-Coder-Next UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF/tree/main) on one 96 GB card each | ⚠️ Untested | [guide](Forge-Trinity/Forge-Trinity.md) |

## Quick Start

1. Find your row in the table. Open that guide — it is the daily recipe (download, cmake, `llama-server`, Pi `models.json`).
2. If you have not built the engine yet, do prerequisites in [`local-setup.md`](local-setup.md), then use **that guide’s** cmake (backend/arch live there).
3. Copy the guide’s `models.json` to **`~/.pi/agent/models.json`**. Match `contextWindow` to `--ctx-size` and `maxTokens` to `--n-predict`. [Agentic harnesses](agentic-harnesses.md).

**How to read a hardware guide:** pin table at the top → Download → Build → PRIMARY command → Confirm → Pi JSON → this-box fallbacks. Essays (GGUF names, flag encyclopedia, Pi theory) live in `local-setup.md`, `llama-cpp-turboquant.md`, and `agentic-harnesses.md`. Qwen3.6 and Qwen3.8 are **siblings** on the same machine, not replacements. Dual RTX **Localmaxing** is the second-card recipe (one `llama-server` per GPU), not a replacement for the Q8 primary. **Qwen3-Coder-Next** is a Dual RTX one-card coding MoE (Q6 primary; ⚠️ untested), not a drop-in for the 27B Q8 host. **Ternary Bonsai 2** is a Dual RTX / DGX Spark / M1 Ultra / RTX 3090 / 7900 XTX experiment on a **second engine** (PrismML fork), not a drop-in for turboquant. On the 7900 XTX it is **HIP**, not the Qwen Vulkan build.

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
├── AMD-7900-XTX/                   # Vulkan Qwen3.6 MTP (tested) · Bonsai 2 HIP untested
├── DGX-Spark-128GB/                # 3.6 tested · 3.8 port untested · LFM2.5 untested (128k FA on) · Bonsai 2 untested
├── Dual-RTX6000-192GB/             # 3.6 + 3.8 tested · Localmaxing · Coder-Next / Muse / Bonsai 2 untested
├── Forge-Trinity/                  # RTX Pro 2000 16 GB + 2× RTX Pro 6000 Max-Q · LFM2.5 + Qwen3.8 + Coder-Next untested
├── M5-MacBook-Pro-48GB/            # 3.6 tested · 3.8 port untested
├── M1-Ultra-Studio-64GB/           # Bonsai 2 Metal untested (PrismML fork)
├── M4-MacBook-Air-24GB/
├── M4-Mac-Mini-16GB/
├── M2-Mac-Mini-16GB/
├── Win-RTX3090-24GB/               # WSL2 · LFM2.5 untested (128k) · Bonsai 2 untested (PrismML fork)
├── Win-RTX4090-24GB/               # WSL2 paths: ~/AIML, ~/GitHub
└── Jetson-Orin-Nano-Super/         # Gemma 4 E2B tested · LFM2.5 tested (64k) · MiniCPM5-2B untested
```

**Paths:** Linux/macOS/Jetson guides use `~/Documents/AIML/models` and `~/Documents/GitHub/llama-cpp-turboquant`. **Windows is WSL2** (`~/AIML`, `~/GitHub`). Any path works if `--model` matches.

This repository is intentionally pragmatic. Settings for **Tested** hardware have been validated on the physical machine; **Untested** configs are careful starting points and may need tuning. Corrections and results are welcome via issues/PRs.

**Last Updated:** 2026-09-25 (Forge Trinity: RTX Pro 2000 + two RTX Pro 6000 Max-Q, untested)  
**Maintained by:** August Sturm  
**License:** see [LICENSE](LICENSE)
