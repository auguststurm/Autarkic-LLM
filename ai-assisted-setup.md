# Set Up Your Machine with an AI Assistant

**New here, or just setting up a new box? Don't read the whole repo — let an LLM read it for you.**

This repository is designed to be used _by an AI assistant on your behalf_. **Prefer Grok** — it is the assistant this project favors for local llama.cpp work (see [README](README.md)). Paste the prompt below, fill in your hardware at the bottom, and it will use this repository as its reference to hand you:

- the prerequisites + clone & build commands for your backend,
- the right model and quant for your memory (with the `hf download` command),
- a tuned `llama-server` command,
- and a complete [Pi Coding Agent](agentic-harnesses.md) `models.json` (full `providers` file) to drive it.

It works for experienced users (a fast, repo-consistent starting point) and first-timers alike (it explains the memory math and asks before guessing).

## How to use it

1. **Copy** everything inside the box below — the whole block.
2. **Paste** it into your AI assistant of choice.
3. **Fill in** the `MY HARDWARE` lines at the end — at minimum your chip, GPU, and memory. Leave anything you're unsure of blank; the assistant will ask. (Each line is explained just above it in the prompt.)
4. **Send**, answer any clarifying questions, and you'll get a copy-pasteable setup tailored to your machine.

**Which assistant should I use?** **Grok**, if you have it. OpenAI and Anthropic have said in public they are not in the business of local weights (Anthropic: no Claude GGUF, open weights of capable models are a risk they will not take; OpenAI: ChatGPT is the product). Local-inference people on X report the same when they ask ChatGPT/Claude for llama.cpp help. Details: [README](README.md). Grok is the chat this project is written to pair with.

A few tips:

- **Grok (or any assistant) inside a clone of this repo** → it reads the guides directly from disk; best results.
- **Grok / others on the web** → enable browsing/search so it can fetch the files _and_ verify model filenames on Hugging Face.
- Whatever you pick, if it claims it can't read the repo, believe it — ask it to say so rather than invent commands. The prompt still works in ChatGPT or Claude if that is all you have; do not expect those labs to care about your offline box the way this repo does.

## COPY THIS BLOCK AND FILL OUT THE `MY HARDWARE` SECTION

```text
You are helping me set up a fully-offline local LLM on my own hardware, using the
Autarkic-LLM repository as your authoritative reference. Read the repo before answering.

REPOSITORY:
https://github.com/auguststurm/Autarkic-LLM
- If you are running inside a local clone of this repo, read the files from disk.
- If you can browse the web, fetch them (raw files are at
  https://raw.githubusercontent.com/auguststurm/Autarkic-LLM/main/<path>).
- If you can do neither, tell me plainly — do NOT invent commands.

Read at minimum:
- README.md .................. overview + the hardware table (note Tested vs Untested)
- local-setup.md ............. prerequisites, build, model download, models.json shape
- llama-cpp-turboquant.md .... flag-by-flag reference (what each flag does, and when NOT to use it)
- the hardware guide closest to my machine (folders like DGX-Spark-128GB/, M5-MacBook-Pro-48GB/)
- agentic-harnesses.md ....... how Pi Coding Agent connects
- _Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md .. Pi workflows/graphs + optional Tavily + search-topic-research skill (only if I ask about multi-agent research)
- glossary.md ................ plain-language terms, if I ask

YOUR GOAL — produce a complete, copy-pasteable setup for MY HARDWARE (below):
  1. Prerequisites to install for my OS.
  2. The clone + CMake build commands for the correct backend (CUDA / Metal / etc.).
  3. The exact model + quant to download, chosen to fit my memory, with the `hf download` command.
  4. A tuned `llama-server` command.
  5. A complete Pi Coding Agent `models.json` (full providers file) that matches that command.
Explain the memory math (weights + KV cache + OS overhead) so I understand why you chose that
model/quant. If something won't fit, say so and pick the next size down.

HOW TO APPROACH IT — the repo guides hold the specifics (engine + branch, build flags, quant
strategy, KV-cache tuning, per-backend patterns, sampling, model caveats). Read them instead of
working from memory. Hardware guides are recipes: pin table, hf download, cmake, PRIMARY
llama-server, confirm, Pi models.json. Match that shape in your reply. Dual RTX Qwen3.8 is the
Qwen3.8 optional appendix (MTP, thinking, vision) and the current recipe template.
Do not paste GGUF naming essays — link local-setup.md. M4 MacBook Air is the tight-Metal
example (turbo2 V, fit-off, 61k), not the layout template. Windows is WSL2 (~/AIML, ~/GitHub).
AMD RDNA3 is Vulkan (AMD-7900-XTX/). Use the closest guide's conventions as your baseline.
The rest is judgment:
- Optimize for the largest, highest-quality model that fits my memory with headroom to spare, tuned
  for fully-offline agentic use — and match the flags to MY hardware, not to whichever guide you read.
- Stay current: if a newer or better-suited model or quant exists than the repo lists — a fresh
  release, a stronger quant, something that fits my use better — prefer it. Just confirm it's real,
  say why it beats the repo's pick, and flag that it isn't covered here yet.
- (always) Confirm any model/quant filename actually exists on its Hugging Face repo before giving me
  a download command — never guess a filename (the #1 failure mode).
- (always) Default the server to loopback (`--host 127.0.0.1`); expose it to my network only if I ask,
  and warn me that's unauthenticated.
- (always) Pin `--ctx-size` and set `--fit off` for agent use. Do not rely on bare `--fit on` — it can
  crush context and break Pi. Prefer lowering quant/context over silent fit.
- (always for Qwen3.6 / Qwen3.8 agents) `--reasoning off` and `--reasoning-budget 0` so Pi gets
  message.content (prefer this over deprecated enable_thinking chat-template-kwargs alone on
  current llama-server).
- (Muse Glimmer 30B only) Do **not** pass `--reasoning off` — it does nothing. Use `--jinja` and
  `--chat-template-kwargs '{"reasoning_strength":"high"}'` (or low/medium/xhigh). Official
  sampling is temp 1.0 / top_p 0.95 / top_k 64. Need llama.cpp/turboquant b10353+ (arch
  muse-glimmer). Optional DFlash: --spec-type draft-dflash + dflash-kquant.gguf. Dual RTX
  starting point: Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md (⚠️ untested).
- (always for hybrid Qwen 3.5/3.6/3.8) omit context-checkpoint flags unless I ask; they often don't
  help on hybrid attention (see llama-cpp-turboquant.md). For Pi + dense Qwen 27B tool stability
  (no DRY, sampling, contextWindow vs maxTokens, K/V), follow agentic-harnesses.md and the hardware
  guide.
- (Qwen3.8, released 2026-08-14) Prefer a Qwen3.8 hardware guide when one exists for my class of
  machine (Dual RTX 6000, DGX Spark, M5 Pro — see README "Qwen3.8"). Dual RTX 6000 Qwen3.8 is
  field-tested; DGX Spark and M5 Pro ports reuse tested 3.6 knobs but are still untested — flag
  that. Use a fresh turboquant build (arch qwen35). Confirm exact UD- quant filenames on
  unsloth/Qwen3.8-27B-GGUF.

PI CODING AGENT — the only harness we are configuring:
Once the server runs, give me a complete Pi `models.json` I can save as-is to
`~/.pi/agent/models.json` — the full object with `providers.llama-cpp`
(`baseUrl` http://127.0.0.1:8080/v1, `api` openai-completions, `apiKey` 1337)
and my single model entry. Do NOT give a bare model object that needs a wrapper.
Tell me to write it to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent` if needed).
`contextWindow` must match my pinned `--ctx-size`; `maxTokens` must not exceed my
`--n-predict`. Do not cover anything else about Pi.

PROCESS:
- If any hardware detail below is missing or "unknown", ASK me before guessing — or infer and
  clearly flag the assumption. Never fabricate specs, filenames, or flags.
- If my hardware closely matches a guide in the repo, start from that guide and adapt it.
- If it matches no guide, reason from the closest one plus the flag reference, tell me the config is
  untested, and suggest I report results back to the repo via an issue or PR.

I'll give you my hardware in the "MY HARDWARE" block below. If a line there is blank, ASK me for it
before proceeding rather than guessing what I want to run — at minimum you need my chip, GPU, and
memory, since those decide the backend, the model, and the quant. What each line means:
- Machine / chip — the computer and its main processor, e.g. MacBook Pro M4 Max, a PC with a Ryzen 9,
  a Jetson Orin Nano Super.
- GPU / accelerator — e.g. NVIDIA RTX 4080 with 16 GB VRAM, Apple Silicon (integrated), none / CPU-only.
- Memory — Apple Silicon: unified RAM (e.g. 36 GB). PC: system RAM (e.g. 64 GB); on a discrete GPU the
  VRAM above is usually what limits model size.
- OS + version — e.g. macOS 15, Ubuntu 24.04, Windows 11 + WSL2, JetPack 6.
- Main use — e.g. coding agent, chat, long-context work (affects context size and whether to reason).
- Extras (optional) — models I already have, free disk space, a context length I need.

MY HARDWARE:
- Machine / chip:
- GPU / accelerator:
- Memory:
- OS + version:
- Main use:
- Extras (optional):
```

---

## After it runs

- Start the server with the command it gave you, then watch the startup log — confirm **`n_ctx` / `n_ctx_seq`** match what you pinned (and that **decode** works, not only load). Keep Pi’s `contextWindow` in sync.
- Write the `models.json` it produced to **`~/.pi/agent/models.json`**, start Pi, and you're running fully offline against `http://127.0.0.1:8080/v1`.
- **Ran this on hardware that isn't in the [table](README.md#hardware-configurations-included) yet?** Please open an issue or PR with what worked — that's how the untested configs become tested ones.
- **Trying Qwen3.8?** Prefer the matching `*Qwen3.8.md` guide when one exists ([overview](README.md#qwen38-2026-08-14)). Dual RTX 6000 is already ✅ Tested; for other ports, smoke-test load → first decode → Pi tools, then report results.
- **Trying Muse Glimmer?** Use [Dual-RTX6000-Muse-Glimmer.md](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) on that box. Do not copy Qwen `--reasoning off`. Smoke-test load → first decode → Pi tools, then report results.
