# Dual RTX 6000 Pro Max-Q (192 GB) - Qwen3.8-27B

Optimized setup for dual NVIDIA RTX 6000 Pro Max-Q GPUs (192 GB total VRAM) on Ubuntu, using **llama-cpp-turboquant**. Primary knobs match the field-validated [Qwen3.6 Dual RTX 6000 guide](Dual-RTX6000-Qwen3.6.md) agent profile and were **re-validated on Qwen3.8** the day of release. Cross-hardware Pi lessons: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware) (same two-limit / no-DRY / tool-sampling rules apply).

> ✅ **Tested** on this hardware (**Thursday, August 14, 2026** — Qwen3.8 release day). Primary config below is **field-validated working** with **Pi Coding Agent** — full **262k** pin, **Q8_K_XL** weights, **q8/q8** KV on a single 96 GB card, stable tools + strong agent quality. Same recipe as the tested Qwen3.6 Dual RTX baseline; only the model file changed. Expect variance with thermals and background load; re-check `n_ctx_seq` after rebuilds.

### What carries over from Qwen3.6 (this box)

| Piece | Qwen3.6 (tested 2026-08-08) | Qwen3.8 (tested 2026-08-14) |
| --- | --- | --- |
| Hardware / engine | Dual RTX 6000, turboquant CUDA CC 12.0 | Same |
| Size class | Dense ~27B, UD-Q8_K_XL ~35 GB | Dense ~27B, UD-Q8_K_XL **~31.5 GB** |
| Native context | 262144 | **262144** (extensible to ~1M via YaRN — not used here) |
| Hybrid layout | Gated DeltaNet + Gated Attention | Same family (Qwen3.5 backbone; GGUF arch tag `qwen35`) |
| Primary pin | single GPU, q8/q8 KV, 16k out, Pi tools | **Same flags — confirmed on 3.8** |
| Vision | Text primary | Native VLM + optional `mmproj` (text/agent primary still) |

GGUF catalog: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Run notes: [Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8).

## Why so many GGUFs? (quick naming decoder)

The Hugging Face folder is **not** twenty different models. It is **one** Qwen3.8-27B (same weights family), packaged many ways so people with 12 GB laptops and people with 192 GB workstations can both load something useful. Each `.gguf` is a different **compression recipe** of those weights. llama.cpp loads one file at a time; you pick the tradeoff that fits your VRAM and quality bar.

**Analogy:** same album, different bitrates. Higher bits → closer to the master, bigger file. Lower bits → smaller/faster to move through memory, more quality loss.

### Anatomy of a name

Example primary file:

```text
Qwen3.8-27B-UD-Q8_K_XL.gguf
│         │  │  │  │  └─ size tier within that recipe (XL = roomier / higher quality mix)
│         │  │  │  └──── K-quant family (block scales; better quality-per-bit than old Q4_0-style)
│         │  │  └─────── bit width (~8-bit weights on average)
│         │  └────────── Unsloth Dynamic mixed-precision method
│         └───────────── parameter class (dense 27B)
└─────────────────────── model family + version
```

| Piece | What it means | How to read it |
| --- | --- | --- |
| **`Qwen3.8-27B`** | Which model | Same “brain”; only the file after this changes quality/size |
| **`UD-`** | **U**nsloth **D**ynamic | Per-layer mixed bits: sensitive tensors (attention, hybrid/DeltaNet-ish parts) stay higher precision; less sensitive ones compress harder. Usually **better quality at a given size** than a plain uniform quant |
| **`Q8` / `Q6` / `Q5` / `Q4` / `Q3` / `Q2`** | Rough **bits per weight** | Higher number → larger file, higher fidelity, often slightly slower decode (more bytes to stream). **Q8 ≈ near full precision; Q4 ≈ common “fits most GPUs” sweet spot** |
| **`IQ2` / `IQ3` / `IQ4_…`** | **I**mportance-aware low-bit schemes | Even more aggressive size cuts; great when VRAM is tight; can be a bit slower or pickier than plain `Q*_K` at the same size |
| **`_K`** | **K-quant** layout | Modern llama.cpp default family (super-blocks + scales). Prefer these over ancient `Q4_0` / `Q5_0` when both exist |
| **`_S` / `_M` / `_L` / `_XL`** | **S**mall / **M**edium / **L**arge / e**X**tra-**L**arge mix | Same bit *label*, different internal mix of which tensors stay fat. **`_M` = balanced default; `_XL` / Unsloth `UD-…_XL` = quality-leaning** (often a little bigger) |
| **No `UD-`** (e.g. `Qwen3.8-27B-Q8_0.gguf`) | “Classic” uniform (or non-Dynamic) quant | Still valid; often slightly worse quality-per-GB than the matching `UD-` file from Unsloth |
| **`mmproj-*.gguf`** | **Vision projector** | Not the LLM. Optional sidecar for image/video. Text/agent work does **not** need it |
| **`imatrix_*.gguf`** | Calibration matrix used *while building* quants | You generally **do not** download this to run the model |

### Why the repo lists so many near-duplicates

| You see… | Why it exists |
| --- | --- |
| `UD-Q4_K_XL` **and** `Q4_K_M` **and** `Q4_0` | Same ~4-bit *idea*, different recipes. Dynamic XL vs plain K_M vs old-style |
| Q8, Q6, Q5, Q4, Q3, Q2 / IQ* | Ladder of VRAM budgets (laptop → desktop → workstation) |
| BF16 / F16 (if present) | Barely compressed reference; huge; rarely needed when Q8 already “feels full” |
| MTP-labeled repos (other Qwen lines) | Multi-token prediction for speed — separate packaging; not required for this primary command |

You only need **one** weights file for `llama-server`. Extra files on the page are menu options, not dependencies (except optional `mmproj` for vision).

### Practical pick order (this 192 GB box)

1. **`UD-Q8_K_XL`** — this guide’s primary (max quality; you have the VRAM).  
2. **`UD-Q6_K_XL`** — if you want near-Q8 quality and a few GB free.  
3. **`UD-Q5_K_XL` / `UD-Q4_K_XL`** — only if you care about multi-instance, thermals, or speed A/Bs.  
4. Skip plain non-`UD` and very low IQ* unless you are size-constrained or comparing recipes.

More background (repo-wide): [local-setup — Understanding GGUF quants](../local-setup.md#understanding-gguf-quants-why-so-many-files) · [Glossary](../glossary.md) · [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs). Bit-ladder sizes/speed for *this* model: [Weight quant table](#weight-quant-q8-vs-q6-vs-q5-vs-q4-this-model) below.

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq`.

**Thinking / empty replies** (Qwen3.8 hybrid thinking): prefer **`--reasoning off`** (+ `--reasoning-budget 0`) for Pi `message.content` / tools. Do **not** pass deprecated `--chat-template-kwargs '{"enable_thinking":false}'` — current llama-server builds warn and prefer `--reasoning on|off`.

Shared Pi + dense-Qwen lessons (two token limits, no DRY, tool sampling, K/V, hybrid cache): [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Sampling for Pi tools (this repo):** keep the **Dual RTX / 4090 agent profile** (`temp 0.6`, `top_p 0.95`, `top_k 20`, **`presence_penalty 0`**, `repeat_penalty 1.0`) — **field-validated on Qwen3.8** on this box. Official Qwen3.8 cards suggest higher presence (1.5) and different temp/top_p for non-thinking chat — that can help pure chat but **hurts path reuse** in shell/write agents.

## Recommended model (primary — single GPU)

- **Model:** `Qwen3.8-27B-UD-Q8_K_XL.gguf` (~31.5 GB — best quality with massive KV headroom on 192 GB)
- **Tested path:** `~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf`

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

> Day-zero model: use a **fresh** turboquant / llama.cpp build. If load fails with an unknown architecture (`qwen35` / `qwen3_5`), `git pull` the fork and rebuild before changing flags.

## Build instructions (TurboQuant fork)

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

# TheTom TurboQuant fork — not ggml-org/llama.cpp
# https://github.com/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

Confirm turbo types (needed only if you try the optional turbo-V path):

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> Omit `-DCMAKE_CUDA_ARCHITECTURES="120"` to autodetect; set explicitly when cross-compiling (Blackwell = CC 12.0).

## Optimized llama-server command (PRIMARY — Q8_K_XL @ 262k, single GPU)

**Tested baseline (2026-08-14)** with Pi Coding Agent on Qwen3.8 — same compute/memory/agent flags as the Dual RTX Qwen3.6 primary; model path is the only intentional change. Use this command first.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 32 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why (field-tested on this box with Qwen3.8) |
| --- | --- |
| `--ctx-size 262144` | Full **native** train window; one 96 GB card has room for Q8 + KV — **confirmed** with Pi |
| `--cache-type-k/v q8_0` | High-precision KV while VRAM allows (agent routing/tools); turbo V only if you need more capacity later |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn cache restore issues ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)); correct multi-turn over cache speed |
| `--n-gpu-layers 99` + `--main-gpu 0` | Full offload on the primary GPU on a dual-card box — primary stays single-GPU for simplicity |
| `--reasoning off` (+ budget 0) | Thinking off for Pi tools / `message.content` — **not** `enable_thinking` kwargs |
| Tool sampling | `temp 0.6` / `top_p 0.95` / `top_k 20` / `presence 0` / `repeat 1.0` — strong Pi tool/coding results on this hardware with 3.8; **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 16384` | Report-length agent output (match Pi `maxTokens`) |
| `--fit off` + `127.0.0.1` | Pinned agent-visible context; loopback default |
| No checkpoint flags | Hybrid Qwen caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

### Weight quant: Q8 vs Q6 vs Q5 vs Q4 (this model)

No day-zero **Qwen3.8** KLD/PPL tables yet. Relative **quality** below follows Unsloth Dynamic **Qwen3.5/3.6 hybrid** dense trends (same backbone family; lower KLD ≈ closer to full precision) plus general llama.cpp K-quant behavior. **Speed** is bandwidth-bound decode on CUDA: lower bits move fewer weight bytes → modestly higher tok/s; prefill is more compute-bound so gaps shrink. On a **96 GB** card all four fit with full **262k q8/q8** KV — pick quality first unless you want free VRAM for multi-slot / vision experiments.

Unsloth `UD-*_K_XL` sizes from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF):

| Quant (Unsloth UD) | File size | Quality (vs full / Q8) | Decode speed (relative) | On this Dual RTX 6000 box |
| --- | --- | --- | --- | --- |
| **UD-Q8_K_XL** (primary) | **~31.5 GB** | Near-lossless; tiny residual vs BF16/F16. Best agent/coding fidelity. | Baseline (slowest of the four; still fast here) | **Use this** — **tested** primary; VRAM is not the bottleneck |
| **UD-Q6_K_XL** | **~25.9 GB** | Excellent / “practically perfect.” Mean KLD only a small step up from Q8 on prior Qwen hybrids; hard to notice in chat | ~5–15% faster decode than Q8 (typical CUDA) | Fine if you want ~5 GB free without a real quality cliff |
| **UD-Q5_K_XL** | **~20.2 GB** | Very high; slight edge-case loss vs Q6/Q8 on multi-step reasoning / hard coding | ~10–20% faster than Q8 | Good “high quality, leaner” option; still overkill for fit |
| **UD-Q4_K_XL** | **~17.9 GB** | Strong for size; Dynamic Q4 often best Pareto on prior Qwen GGUFs. More visible loss vs Q8 on long agent loops / precise tools | Fastest of the four (~15–25%+ vs Q8 when bandwidth-bound) | Prefer only for multi-instance, lower thermals, or A/B speed tests — not for max quality |

**How to read this**

- **Quality steps:** Q8 → Q6 is usually *tiny*; Q6 → Q5 still small; Q5 → Q4 is the first step many people *feel* on hard coding/agent work. Unsloth Dynamic + imatrix narrows the Q4 gap vs plain `Q4_K_M`.
- **Speed steps:** Smaller than quality folklore suggests on modern GPUs with good CUDA kernels; you buy more with Q4 than you lose from Q6→Q8. Exact tok/s need `llama-bench` on this box.
- **This hardware:** Primary stays **Q8** — tested with Pi on release day. Dropping to Q6/Q5 frees VRAM but is optional, not required for 262k.

Swap only the model path (and Pi `id`/`name` if you track it). Example Q6 download:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

**Confirm**

```text
log: n_ctx_seq (262144)
# model loads as qwen35 / Qwen3.8-27B (exact string varies by build)
nvidia-smi   # note MiB after load and after a short decode
```

Confirm after rebuilds: (1) load, (2) short decode, (3) Pi tools on a fresh session.

### Optional: TurboQuant V (capacity / decode tradeoff)

Primary already has enormous headroom (~31.5 GB weights on a 96 GB card). If you later raise load (multi-slot, vision, or decode experiments) and need to free KV:

```bash
# Same command as above, only:
#   --cache-type-k q8_0 --cache-type-v turbo4
# Keep K at q8_0. Smoke-test real ls/read on a new Pi session first.
```

### Optional: vision (`mmproj`)

Qwen3.8-27B is a native VLM. For image/video sessions (not required for Pi text/agent work):

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models
```

Add something like `--mmproj ~/Documents/AIML/models/mmproj-F16.gguf` only if your turboquant `llama-server` build exposes multimodal flags for this arch. Keep text/agent primary without mmproj until you need vision.

## Performance notes

- **Validated 2026-08-14 (release day):** primary single-GPU **Q8_K_XL @ 262k · q8/q8 · 16k out** worked very well with Pi Coding Agent on Qwen3.8 (tools, long agent sessions, full train window) — same knobs as the tested Qwen3.6 Dual RTX run (2026-08-08).
- Q8 weights are **slightly smaller** than 3.6 Q8 (~31.5 vs ~35 GB) → comfortable KV headroom at 262k q8/q8 on one 96 GB card.
- Primary uses **one GPU** by default. The other 96 GB card stays free (or idle).
- TurboQuant **fork** ≠ must use turbo **types**. Keep `q8_0`/`q8_0`; turbo V is a capacity lever, not the quality default.
- Official Qwen3.8 sampling (thinking vs instruct) differs from this Pi tool profile — see Unsloth/Qwen cards if you leave agent mode for pure chat/thinking sessions.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Cross-hardware Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Sibling 3.6 twin: [Dual-RTX6000-Qwen3.6.md](Dual-RTX6000-Qwen3.6.md).

## Pi Coding Agent `models.json`

Save this **entire** file to `~/.pi/agent/models.json` (copy-paste as-is — do not assemble a wrapper). Create parent dirs if needed: `mkdir -p ~/.pi/agent`. **Restart Pi** after writing so the status bar matches the pin.

`maxTokens` ≤ `--n-predict` (16384). `contextWindow` = `--ctx-size` (262144).

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B Q8_K_XL (262k q8/q8) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

## Alternate: multi-GPU layer split

> ⚠️ **Not part of the August 14 primary test.** Primary single-GPU Q8_K_XL is the validated path. Use multi-GPU only if a future larger quant/model does not fit on one card or you deliberately want to spread load.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --split-mode layer \
  --tensor-split 96,96 \
  --main-gpu 0 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 32 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

- Same agent flags as primary. Multi-GPU adds PCIe sync overhead; benchmark against single-GPU before committing.
- There is no mid-size Qwen3.8 MoE twin to the old 35B-A3B alternate yet — keep this as a split of the **same** 27B dense weights if you experiment.

**Last Updated:** August 14, 2026 (primary marked ✅ Tested — Qwen3.8 release-day Dual RTX + Pi run)
