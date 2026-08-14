# DGX Spark Founders Edition (128 GB) - Qwen3.8-27B

Optimized setup for the NVIDIA DGX Spark Founders Edition (GB10 GPU, 128 GB system RAM), using **llama-cpp-turboquant**. **Server / Pi knobs copy the field-validated [Qwen3.6 DGX Spark guide](DGX-Spark-Qwen3.6.md)** (same box, same agent-scale pin); only model paths, names, and release notes change for **Qwen3.8**. Cross-hardware Pi lessons: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware) (same two-limit / no-DRY / tool-sampling rules apply). Sibling high-VRAM CUDA guide (✅ tested on 3.8): [Dual RTX 6000 Qwen3.8](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md).

> ⚠️ **Not yet tested on this hardware with Qwen3.8.** Model released **2026-08-14**. Primary command is a best-effort port of the **tested Qwen3.6-27B Q6_K_XL @ 262k · q8/turbo4** baseline on this box. Confirm load + first decode + a short Pi tool loop before relying on it. Report results via an issue/PR.

### What carries over from Qwen3.6 (this box)

| Piece | Qwen3.6 (tested) | Qwen3.8 (this guide) |
| --- | --- | --- |
| Hardware / engine | DGX Spark GB10, turboquant CUDA **CC 12.1** | Same |
| Size class | Dense ~27B, **UD-Q6_K_XL ~22 GB** primary | Dense ~27B, **UD-Q6_K_XL ~25.9 GB** primary |
| Native context | 262144 | **262144** (extensible to ~1M via YaRN — not used here) |
| Hybrid layout | Gated DeltaNet + Gated Attention | Same family (Qwen3.5 backbone; GGUF arch tag `qwen35`) |
| Primary pin | full offload, **q8_0 K / turbo4 V**, 262k | **Same flags** |
| Higher-quality alternate | UD-Q8_K_XL (~35 GB) untested on 3.6 | UD-Q8_K_XL (**~31.5 GB**) — still optional |
| Vision | Text primary | Native VLM + optional `mmproj` (text/agent primary still) |

GGUF catalog: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Run notes: [Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8).

## Why so many GGUFs? (quick naming decoder)

The Hugging Face folder is **not** twenty different models. It is **one** Qwen3.8-27B (same weights family), packaged many ways so people with 12 GB laptops and people with 128 GB Sparks can both load something useful. Each `.gguf` is a different **compression recipe** of those weights. llama.cpp loads one file at a time; you pick the tradeoff that fits your memory and quality bar.

**Analogy:** same album, different bitrates. Higher bits → closer to the master, bigger file. Lower bits → smaller/faster to move through memory, more quality loss.

### Anatomy of a name

Example primary file:

```text
Qwen3.8-27B-UD-Q6_K_XL.gguf
│         │  │  │  │  └─ size tier within that recipe (XL = roomier / higher quality mix)
│         │  │  │  └──── K-quant family (block scales; better quality-per-bit than old Q4_0-style)
│         │  │  └─────── bit width (~6-bit weights on average)
│         │  └────────── Unsloth Dynamic mixed-precision method
│         └───────────── parameter class (dense 27B)
└─────────────────────── model family + version
```

| Piece | What it means | How to read it |
| --- | --- | --- |
| **`Qwen3.8-27B`** | Which model | Same “brain”; only the file after this changes quality/size |
| **`UD-`** | **U**nsloth **D**ynamic | Per-layer mixed bits: sensitive tensors (attention, hybrid/DeltaNet-ish parts) stay higher precision; less sensitive ones compress harder. Usually **better quality at a given size** than a plain uniform quant |
| **`Q8` / `Q6` / `Q5` / `Q4` / `Q3` / `Q2`** | Rough **bits per weight** | Higher number → larger file, higher fidelity, often slightly slower decode (more bytes to stream). **Q8 ≈ near full precision; Q4 ≈ common “fits most GPUs” sweet spot** |
| **`IQ2` / `IQ3` / `IQ4_…`** | **I**mportance-aware low-bit schemes | Even more aggressive size cuts; great when memory is tight; can be a bit slower or pickier than plain `Q*_K` at the same size |
| **`_K`** | **K-quant** layout | Modern llama.cpp default family (super-blocks + scales). Prefer these over ancient `Q4_0` / `Q5_0` when both exist |
| **`_S` / `_M` / `_L` / `_XL`** | **S**mall / **M**edium / **L**arge / e**X**tra-**L**arge mix | Same bit *label*, different internal mix of which tensors stay fat. **`_M` = balanced default; `_XL` / Unsloth `UD-…_XL` = quality-leaning** (often a little bigger) |
| **No `UD-`** (e.g. `Qwen3.8-27B-Q6_K.gguf`) | “Classic” uniform (or non-Dynamic) quant | Still valid; often slightly worse quality-per-GB than the matching `UD-` file from Unsloth |
| **`mmproj-*.gguf`** | **Vision projector** | Not the LLM. Optional sidecar for image/video. Text/agent work does **not** need it |
| **`imatrix_*.gguf`** | Calibration matrix used *while building* quants | You generally **do not** download this to run the model |

### Why the repo lists so many near-duplicates

| You see… | Why it exists |
| --- | --- |
| `UD-Q4_K_XL` **and** `Q4_K_M` **and** `Q4_0` | Same ~4-bit *idea*, different recipes. Dynamic XL vs plain K_M vs old-style |
| Q8, Q6, Q5, Q4, Q3, Q2 / IQ* | Ladder of memory budgets (laptop → desktop → Spark / workstation) |
| BF16 / F16 (if present) | Barely compressed reference; huge; rarely needed when Q6/Q8 already “feel full” |
| MTP-labeled repos (other Qwen lines) | Multi-token prediction for speed — separate packaging; not required for this primary command |

You only need **one** weights file for `llama-server`. Extra files on the page are menu options, not dependencies (except optional `mmproj` for vision).

### Practical pick order (this 128 GB Spark)

1. **`UD-Q6_K_XL`** — this guide’s primary (tested 3.6 recipe on this box: quality + full 262k with turbo V).  
2. **`UD-Q8_K_XL`** — optional max fidelity if you have headroom (see [alternate](#alternate-higher-quality-untested)).  
3. **`UD-Q5_K_XL` / `UD-Q4_K_XL`** — only for multi-instance, thermals, or speed A/Bs.  
4. Skip plain non-`UD` and very low IQ* unless you are size-constrained or comparing recipes.

More background (repo-wide): [local-setup — Understanding GGUF quants](../local-setup.md#understanding-gguf-quants-why-so-many-files) · [Glossary](../glossary.md) · [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs). Bit-ladder sizes/speed for *this* model: [Weight quant table](#weight-quant-q8-vs-q6-vs-q5-vs-q4-this-model) below.

## Pi Coding Agent: read this first

Pi needs a large `contextWindow`. Match it to the server’s real `n_ctx_seq`.

**Always pin `--ctx-size` and set `--fit off`.** Default `--fit on` can crush context and break Pi.

**Thinking / empty replies** (Qwen3.8 hybrid thinking): prefer **`--reasoning off`** (+ `--reasoning-budget 0`) for Pi `message.content` / tools. Do **not** pass deprecated `--chat-template-kwargs '{"enable_thinking":false}'` — current llama-server builds warn and prefer `--reasoning on|off`.

Shared Pi + dense-Qwen lessons (two token limits, no DRY, tool sampling, K/V, hybrid cache): [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Sampling for Pi tools (this repo):** prefer the cross-hardware agent profile (`temp 0.6`, `top_p 0.95`, `top_k 20`, **`presence_penalty 0`**, `repeat_penalty 1.0`). The original DGX Qwen3.6 guide used slightly different chat-era defaults (`temp 0.65` / `top_p 0.90` / `repeat 1.10`); primary below keeps those **hardware-tested** numbers. If path-heavy Pi tools misbehave, switch to the agent profile and a **new** Pi session.

## Recommended model (primary)

- **Model:** `Qwen3.8-27B-UD-Q6_K_XL.gguf` (~25.9 GB — tested-class quant on this box with TurboQuant V @ 262k)
- **Path:** `~/Documents/AIML/models/Qwen3.8-27B-UD-Q6_K_XL.gguf`

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf \
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

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121"
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

Confirm the binary accepts turbo types:

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> Omit `-DCMAKE_CUDA_ARCHITECTURES="121"` to autodetect on the build machine; set it explicitly when cross-compiling or sharing binaries (GB10 = CC 12.1).

## Optimized llama-server command (PRIMARY — Q6_K_XL @ 262k)

**Port of the DGX Spark Qwen3.6 tested baseline** — same compute/memory knobs; model path only changes. Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 28 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why (ported from tested Qwen3.6 DGX Spark) |
| --- | --- |
| `--ctx-size 262144` | Full **native** train window; stable on this box with TurboQuant V on 3.6 |
| `--cache-type-k q8_0 --cache-type-v turbo4` | CUDA quality-leaning turbo V (not Metal’s more aggressive turbo2); keeps K precise for routing/tools |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn cache restore issues ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)); correct multi-turn over cache speed |
| `--n-gpu-layers 99` | Full GPU offload on GB10 / CUDA-class devices |
| `--reasoning off` (+ budget 0) | Thinking off for Pi tools / `message.content` — **not** deprecated `enable_thinking` kwargs |
| Sampling | `temp 0.65` / `top_p 0.90` / `repeat 1.10` — matches this box’s tested 3.6 command; for path-heavy Pi see agent profile note above (**no DRY**) |
| `--n-predict 8192` | Matches tested DGX 3.6 + Pi `maxTokens`; raise toward **16384** (and Pi) if reports truncate |
| `--fit off` + `127.0.0.1` | Pinned agent-visible context; loopback default |
| `--threads 28` | Spark host CPU pairing from the 3.6 guide |
| No checkpoint flags | Hybrid Qwen caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

### Weight quant: Q8 vs Q6 vs Q5 vs Q4 (this model)

No day-zero **Qwen3.8** KLD/PPL tables yet. Relative **quality** below follows Unsloth Dynamic **Qwen3.5/3.6 hybrid** dense trends (same backbone family; lower KLD ≈ closer to full precision) plus general llama.cpp K-quant behavior. **Speed** is bandwidth-bound decode on CUDA: lower bits move fewer weight bytes → modestly higher tok/s; prefill is more compute-bound so gaps shrink. On **128 GB** Spark all four fit with a large 262k window under **q8/turbo4** — primary stays **Q6** to match the tested 3.6 recipe (room for KV + system without jumping to Q8).

Unsloth `UD-*_K_XL` sizes from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF):

| Quant (Unsloth UD) | File size | Quality (vs full / Q8) | Decode speed (relative) | On this DGX Spark box |
| --- | --- | --- | --- | --- |
| **UD-Q8_K_XL** | **~31.5 GB** | Near-lossless; tiny residual vs BF16/F16. Best agent/coding fidelity. | Baseline (slowest of the four) | Optional upgrade — see [alternate](#alternate-higher-quality-untested) |
| **UD-Q6_K_XL** (primary) | **~25.9 GB** | Excellent / “practically perfect.” Mean KLD only a small step up from Q8 on prior Qwen hybrids | ~5–15% faster decode than Q8 (typical CUDA) | **Use this** — tested-class 3.6 primary on this hardware |
| **UD-Q5_K_XL** | **~20.2 GB** | Very high; slight edge-case loss vs Q6/Q8 on multi-step reasoning / hard coding | ~10–20% faster than Q8 | Leaner if you want free RAM for multi-slot / vision |
| **UD-Q4_K_XL** | **~17.9 GB** | Strong for size; Dynamic Q4 often best Pareto on prior Qwen GGUFs. More visible loss vs Q8 on long agent loops | Fastest of the four (~15–25%+ vs Q8 when bandwidth-bound) | Prefer only for multi-instance or speed A/Bs — not max quality |

**How to read this**

- **Quality steps:** Q8 → Q6 is usually *tiny*; Q6 → Q5 still small; Q5 → Q4 is the first step many people *feel* on hard coding/agent work. Unsloth Dynamic + imatrix narrows the Q4 gap vs plain `Q4_K_M`.
- **Speed steps:** Smaller than quality folklore suggests on modern GPUs with good CUDA kernels. Exact tok/s need `llama-bench` on this box (3.6 ballpark on Spark: ~45–65 t/s prefill, 90–120+ t/s decode — re-measure for 3.8).
- **This hardware:** Primary stays **Q6 + turbo4 V** to match the tested Qwen3.6 DGX profile. Q8 is the quality alternate when you want max fidelity and still have ~90+ GB-class headroom for KV.

Swap only the model path (and Pi `id`/`name` if you track it). Example Q8 download:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

**Confirm**

```text
log: n_ctx_seq (262144)
# model loads as qwen35 / Qwen3.8-27B (exact string varies by build)
# or GET /v1/models
```

Smoke-test order: (1) load, (2) short chat completion, (3) new Pi session with real `ls` / `read` tools.

### Optional: milder / stronger KV

Primary uses **turbo4 V** (capacity at 262k). If tools feel soft and memory allows, try **q8/q8** first (same as Dual RTX primary) before changing weight quant:

```bash
# Same command as above, only:
#   --cache-type-k q8_0 --cache-type-v q8_0
# Smoke-test real ls/read on a new Pi session; watch memory at long context.
```

If you need *more* capacity later (multi-slot, vision, extreme load): keep K at `q8_0`, step V `turbo4` → `turbo3` → `turbo2` only as needed.

### Optional: vision (`mmproj`)

Qwen3.8-27B is a native VLM. For image/video sessions (not required for Pi text/agent work):

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models
```

Add something like `--mmproj ~/Documents/AIML/models/mmproj-F16.gguf` only if your turboquant `llama-server` build exposes multimodal flags for this arch. Keep text/agent primary without mmproj until you need vision.

## Performance notes

- **Starting point:** same Q6 + **q8/turbo4** + **262k** recipe that was **tested** for Qwen3.6-27B on this hardware.
- Qwen3.8 Q6 weights are **slightly larger** than the 3.6 Q6 path in this guide’s table (~25.9 GB vs ~22 GB listed for 3.6) → still comfortable on 128 GB; confirm after first decode.
- Expected ballpark from 3.6 on this box: ~45–65 t/s (prefill), 90–120+ t/s (decode) — **re-bench** for 3.8; treat as order-of-magnitude only.
- Full 262k context was stable with TurboQuant V on 3.6; re-check `n_ctx_seq` after rebuilds.
- Excellent for long agentic tasks with Hermes / Pi Coding Agent once smoke-tested.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Cross-hardware Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Sibling 3.6 twin: [DGX-Spark-Qwen3.6.md](DGX-Spark-Qwen3.6.md).

## Pi Coding Agent `models.json`

Save this **entire** file to `~/.pi/agent/models.json` (copy-paste as-is — do not assemble a wrapper). Create parent dirs if needed: `mkdir -p ~/.pi/agent`. **Restart Pi** after writing so the status bar matches the pin.

`maxTokens` ≤ `--n-predict` (8192). `contextWindow` = `--ctx-size` (262144).

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
          "name": "Qwen3.8-27B Q6_K_XL (262k q8/turbo4) - DGX Spark",
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

## Alternate: higher quality (untested)

Primary config above is the **ported 3.6 baseline**. If you want maximum weight fidelity and have headroom to spare:

- **Model:** `Qwen3.8-27B-UD-Q8_K_XL.gguf` (~31.5 GB)
- **Path:** `~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf`
- Use the same `llama-server` command as primary, swapping only `--model`. Large unified memory remains for KV at 262k with turbo V (or try q8/q8 if tools need it).

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

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
          "name": "Qwen3.8-27B Q8_K_XL (262k) - DGX Spark",
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

**Last Updated:** August 14, 2026 (day-zero Qwen3.8 guide; settings ported from tested DGX Spark Qwen3.6)
