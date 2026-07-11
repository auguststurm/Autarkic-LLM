# Set Up Your Machine with an AI Assistant

**New here, or just setting up a new box? Don't read the whole repo — let an LLM read it for you.**

This repository is designed to be used _by an AI assistant on your behalf_. Whether or not you've ever run a local model, you can paste the prompt below into the assistant you already use — **Claude Code, GitHub Copilot Chat, ChatGPT, Grok, Gemini**, and so on — fill in your hardware at the bottom, and it will use this repository as its reference to hand you:

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

**Which assistant should I use?** Any of them. A few tips:

- **Claude Code or Copilot inside a clone of this repo** → it reads the guides directly from disk; best results.
- **ChatGPT / Claude / Grok / Gemini on the web** → enable browsing/search so it can fetch the files _and_ verify model filenames on Hugging Face.
- Whatever you pick, if it claims it can't read the repo, believe it — ask it to say so rather than invent commands.

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
working from memory, and use the closest guide's conventions as your baseline. The M4 MacBook Air
Qwen guide is the reference pattern for modern flags (host, fit, thinking, TurboQuant, models.json).
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
- (always for Qwen3.6 agents) `--reasoning off` and
  `--chat-template-kwargs '{"enable_thinking":false}'`.
- (always for Qwen3.6) omit context-checkpoint flags unless I ask; they often don't help on hybrid
  attention (see llama-cpp-turboquant.md).

PI CODING AGENT — the only harness we are configuring:
Once the server runs, give me a complete Pi `models.json` I can save as-is — the full object with
`providers.llama-cpp` (`baseUrl` http://127.0.0.1:8080/v1, `api` openai-completions, `apiKey` 1337)
and my single model entry. Do NOT give a bare model object that needs a wrapper. `contextWindow`
must match my pinned `--ctx-size`; `maxTokens` must not exceed my `--n-predict`. Do not cover
anything else about Pi.

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
- Drop the `models.json` it produced where Pi expects it, point Pi at `http://127.0.0.1:8080/v1`, and you're running fully offline.
- **Ran this on hardware that isn't in the [table](README.md#hardware-configurations-included) yet?** Please open an issue or PR with what worked — that's how the untested configs become tested ones.
