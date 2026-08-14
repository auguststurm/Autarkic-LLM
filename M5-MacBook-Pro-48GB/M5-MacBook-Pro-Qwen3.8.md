# M5 MacBook Pro (48 GB) - Qwen3.8-27B

Optimized setup for MacBook Pro M5 Pro/Max with **48 GB unified memory**, using **llama-cpp-turboquant** on **Metal** (not CUDA). **Server / Pi knobs copy the field-validated [Qwen3.6 M5 MacBook Pro guide](M5-MacBook-Pro-Qwen3.6.md)** (same box, same agent-scale pin); only model paths, names, and release notes change for **Qwen3.8**. Cross-hardware Pi lessons: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Pattern reference for tighter Metal: [M4 Air](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md). CUDA Qwen3.8 twins (different knobs): [Dual RTX 6000](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) (✅ tested) · [DGX Spark](../DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) (⚠️ untested port).

> ⚠️ **Not yet tested on this hardware with Qwen3.8.** Model released **2026-08-14**. Primary command is a best-effort port of the **tested Qwen3.6-27B Q5_K_XL @ 196k · q8/q8 (Metal)** baseline on this box. Confirm load + **first decode** (not only load) + a short Pi tool loop before relying on it. Report results via an issue/PR.

### What carries over from Qwen3.6 (this box)

| Piece | Qwen3.6 (tested) | Qwen3.8 (this guide) |
| --- | --- | --- |
| Hardware / engine | M5 Pro/Max 48 GB, turboquant **Metal** | Same |
| Size class | Dense ~27B, **UD-Q5_K_XL ~20 GB** primary | Dense ~27B, **UD-Q5_K_XL ~20.2 GB** primary |
| Pinned context | **196608** (~196k) agent window | **Same pin** (not full 262k train length) |
| Hybrid layout | Gated DeltaNet + Gated Attention | Same family (Qwen3.5 backbone; GGUF arch tag `qwen35`) |
| Primary KV | **`q8_0` / `q8_0`** (no turbo required) | **Same** |
| Batch | 512 / 512 | Same |
| Vision | Text primary | Native VLM + optional `mmproj` (text/agent primary still) |

GGUF catalog: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF). Run notes: [Unsloth Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8).

## Memory reality (read this)

- **Weights:** `UD-Q5_K_XL` is ~20.2 GB. On 48 GB that leaves comfortable room for macOS + KV + compute (same story as the tested 3.6 Q5 primary).
- **Model train context:** Qwen3.8-27B natively supports **262144** (262k). This guide pins **196608** (~196k) as the **ported agent-scale window** from the tested M5 Qwen3.6 setup — not full train length, but far above the Air’s verified ~61k.
- **Why not turbo on the primary command?** The Air *needs* `turbo2` V so long context fits on 24 GB. Here, plain **`q8_0` / `q8_0`** was the tested quality baseline at 196k on 3.6. Turbo is optional headroom (see below), not required for survival.
- **Why not Q8 primary?** Q8 (~31.5 GB) leaves little unified memory for OS + large KV on a laptop; Q5 is the proven 48 GB balance. Q6 is a possible step-up if decode stays happy.
- Close heavy apps before launch. Prefer **one** long-lived `llama-server` process.

## Why so many GGUFs? (quick naming decoder)

The Hugging Face folder is **not** twenty different models. It is **one** Qwen3.8-27B (same weights family), packaged many ways so people with 16 GB Macs and people with 48–192 GB machines can both load something useful. Each `.gguf` is a different **compression recipe** of those weights. llama.cpp loads one file at a time; you pick the tradeoff that fits your unified memory and quality bar.

**Analogy:** same album, different bitrates. Higher bits → closer to the master, bigger file. Lower bits → smaller/faster to move through memory, more quality loss.

### Anatomy of a name

Example primary file:

```text
Qwen3.8-27B-UD-Q5_K_XL.gguf
│         │  │  │  │  └─ size tier within that recipe (XL = roomier / higher quality mix)
│         │  │  │  └──── K-quant family (block scales; better quality-per-bit than old Q4_0-style)
│         │  │  └─────── bit width (~5-bit weights on average)
│         │  └────────── Unsloth Dynamic mixed-precision method
│         └───────────── parameter class (dense 27B)
└─────────────────────── model family + version
```

| Piece | What it means | How to read it |
| --- | --- | --- |
| **`Qwen3.8-27B`** | Which model | Same “brain”; only the file after this changes quality/size |
| **`UD-`** | **U**nsloth **D**ynamic | Per-layer mixed bits: sensitive tensors (attention, hybrid/DeltaNet-ish parts) stay higher precision; less sensitive ones compress harder. Usually **better quality at a given size** than a plain uniform quant |
| **`Q8` / `Q6` / `Q5` / `Q4` / `Q3` / `Q2`** | Rough **bits per weight** | Higher number → larger file, higher fidelity, often slightly slower decode (more bytes to stream). **Q8 ≈ near full precision; Q4 ≈ common “fits most machines” sweet spot** |
| **`IQ2` / `IQ3` / `IQ4_…`** | **I**mportance-aware low-bit schemes | Even more aggressive size cuts; great when unified memory is tight; can be a bit slower or pickier than plain `Q*_K` at the same size |
| **`_K`** | **K-quant** layout | Modern llama.cpp default family (super-blocks + scales). Prefer these over ancient `Q4_0` / `Q5_0` when both exist |
| **`_S` / `_M` / `_L` / `_XL`** | **S**mall / **M**edium / **L**arge / e**X**tra-**L**arge mix | Same bit *label*, different internal mix of which tensors stay fat. **`_M` = balanced default; `_XL` / Unsloth `UD-…_XL` = quality-leaning** (often a little bigger) |
| **No `UD-`** (e.g. `Qwen3.8-27B-Q5_K_M.gguf`) | “Classic” uniform (or non-Dynamic) quant | Still valid; often slightly worse quality-per-GB than the matching `UD-` file from Unsloth |
| **`mmproj-*.gguf`** | **Vision projector** | Not the LLM. Optional sidecar for image/video. Text/agent work does **not** need it |
| **`imatrix_*.gguf`** | Calibration matrix used *while building* quants | You generally **do not** download this to run the model |

### Why the repo lists so many near-duplicates

| You see… | Why it exists |
| --- | --- |
| `UD-Q4_K_XL` **and** `Q4_K_M` **and** `Q4_0` | Same ~4-bit *idea*, different recipes. Dynamic XL vs plain K_M vs old-style |
| Q8, Q6, Q5, Q4, Q3, Q2 / IQ* | Ladder of memory budgets (Air / Mini → M5 48 GB → workstation) |
| BF16 / F16 (if present) | Barely compressed reference; huge; rarely needed when Q5 already feels strong |
| MTP-labeled repos (other Qwen lines) | Multi-token prediction for speed — separate packaging; not required for this primary command |

You only need **one** weights file for `llama-server`. Extra files on the page are menu options, not dependencies (except optional `mmproj` for vision).

### Practical pick order (this 48 GB Metal box)

1. **`UD-Q5_K_XL`** — this guide’s primary (tested 3.6 recipe: quality + ~196k with q8/q8).  
2. **`UD-Q6_K_XL`** — optional step-up if you close apps and confirm decode at 196k.  
3. **`UD-Q4_K_XL`** — leaner / multi-instance / speed A/Bs.  
4. **`UD-Q8_K_XL`** — only if you deliberately lower context or use turbo V; tight with OS + 196k.  
5. Skip plain non-`UD` and very low IQ* unless you are size-constrained or comparing recipes.

More background: [Glossary](../glossary.md) · [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs). Bit-ladder sizes/speed for *this* model: [Weight quant table](#weight-quant-q8-vs-q6-vs-q5-vs-q4-this-model) below.

## Pi Coding Agent: read this first

Pi needs a large `contextWindow`. Match it to the server’s real `n_ctx_seq`.

**Always pin `--ctx-size` and set `--fit off`.** Default `--fit on` can crush context (on tighter boxes it falls toward ~4096) and break Pi.

**Thinking / empty replies** (Qwen3.8 hybrid thinking): prefer **`--reasoning off`** (+ `--reasoning-budget 0`) so Pi gets `message.content` / tools. Do **not** pass deprecated `--chat-template-kwargs '{"enable_thinking":false}'` — current llama-server builds warn and prefer `--reasoning on|off`.

Shared Pi + dense-Qwen lessons (two token limits, no DRY, tool sampling, K/V, hybrid cache): [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Sampling for Pi tools (this repo):** prefer the cross-hardware agent profile (`temp 0.6`, `top_p 0.95`, `top_k 20`, **`presence_penalty 0`**, `repeat_penalty 1.0`). The original M5 Qwen3.6 guide used slightly different chat-era defaults (`temp 0.65` / `top_p 0.90` / `repeat 1.10`); primary below keeps those **hardware-tested** numbers. If path-heavy Pi tools misbehave, switch to the agent profile and a **new** Pi session.

## Recommended model (primary)

- **Model:** `Qwen3.8-27B-UD-Q5_K_XL.gguf` (~20.2 GB — tested-class quant on this box with q8/q8 @ 196k)
- **Path:** `~/Documents/AIML/models/Qwen3.8-27B-UD-Q5_K_XL.gguf`

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

> Day-zero model: use a **fresh** turboquant / llama.cpp **Metal** build. If load fails with an unknown architecture (`qwen35` / `qwen3_5`), `git pull` the fork and rebuild before changing flags.

## Build instructions (TurboQuant fork — Metal)

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

# TheTom TurboQuant fork — not ggml-org/llama.cpp
# https://github.com/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)

cd bin
mkdir -p ./kv-cache
```

Confirm the binary accepts turbo types (needed if you try the optional turbo path):

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> **Fork version:** tip that includes Metal turbo4 `rnorm` fix (**`b01afefed` / PR #200 content or later**). No manual Metal shader edit on current TheTom tip. Rebuild after `git pull`.  
> **Not CUDA:** do not copy `-DGGML_CUDA=ON` or `CMAKE_CUDA_ARCHITECTURES` from the Dual RTX / DGX guides onto this machine.

## Optimized llama-server command (PRIMARY — Q5_K_XL @ 196k)

**Port of the M5 Qwen3.6 tested baseline** — same Metal compute/memory knobs; model path only changes. Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 196608 \
  --fit off \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 512 \
  --batch-size 512 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 0 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why (ported from tested Qwen3.6 M5 Metal) |
| --- | --- |
| `--ctx-size 196608` | Large agent window verified on this hardware with Q5 on 3.6 |
| `--cache-type-k/v q8_0` | Quality baseline at 196k — room enough without turbo V on 48 GB |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn cache restore issues ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)); correct multi-turn over cache speed |
| `--flash-attn on` | Fast path; required if you later switch V to turbo* |
| `--ubatch-size` / `--batch-size` **512** | Comfortable on 48 GB; drop if you Metal-OOM |
| `--reasoning off` (+ budget 0) | Thinking off for Pi tools / `message.content` — **not** deprecated `enable_thinking` kwargs |
| Sampling | `temp 0.65` / `top_p 0.90` / `repeat 1.10` — matches this box’s tested 3.6 command; for path-heavy Pi see agent profile note above (**no DRY**) |
| `--n-predict 8192` | Matches tested M5 3.6 + Pi `maxTokens`; raise toward **16384** (and Pi) if reports truncate |
| `--fit off` + `127.0.0.1` | Pinned agent-visible context; loopback default |
| `--threads 0` | Let the runtime pick host threads on Apple Silicon |
| No `--n-gpu-layers` | Metal unified memory path (do not paste CUDA multi-GPU flags here) |
| No checkpoint flags | Hybrid Qwen caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

### Weight quant: Q8 vs Q6 vs Q5 vs Q4 (this model)

No day-zero **Qwen3.8** KLD/PPL tables yet. Relative **quality** below follows Unsloth Dynamic **Qwen3.5/3.6 hybrid** dense trends (same backbone family; lower KLD ≈ closer to full precision) plus general llama.cpp K-quant behavior. **Speed** on Metal is often memory-bandwidth bound: lower bits move fewer weight bytes → modestly higher tok/s; prefill can be closer across quants. On **48 GB** unified, prefer **Q5** for the tested 196k q8/q8 balance — Q8 fights the laptop budget.

Unsloth `UD-*_K_XL` sizes from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF):

| Quant (Unsloth UD) | File size | Quality (vs full / Q8) | Decode speed (relative) | On this M5 48 GB Metal box |
| --- | --- | --- | --- | --- |
| **UD-Q8_K_XL** | **~31.5 GB** | Near-lossless | Slowest of the four | Tight with OS + 196k; only with lower ctx and/or turbo V |
| **UD-Q6_K_XL** | **~25.9 GB** | Excellent / near-Q8 | ~5–15% faster than Q8 (typical) | Optional step-up; confirm **decode** at 196k with apps closed |
| **UD-Q5_K_XL** (primary) | **~20.2 GB** | Very high; tested-class primary on this box for 3.6 | Solid middle | **Use this** — quality + long agent window without turbo |
| **UD-Q4_K_XL** | **~17.9 GB** | Strong for size; more loss vs Q5/Q8 on hard agent loops | Fastest of the four | Multi-instance, thermals, or speed A/Bs |

**How to read this**

- **Quality steps:** Q8 → Q6 is usually *tiny*; Q6 → Q5 still small; Q5 → Q4 is the first step many people *feel* on hard coding/agent work.
- **Speed steps:** Modest on modern Apple Silicon; exact tok/s need `llama-bench` on this laptop (thermals matter on sustained load).
- **This hardware:** Primary stays **Q5 + q8/q8 @ 196k** to match the tested Qwen3.6 M5 profile.

Swap only the model path (and Pi `id`/`name` if you track it). Example Q4 download:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

**Confirm**

```text
log: n_ctx_seq (196608)
# model loads as qwen35 / Qwen3.8-27B (exact string varies by build)
# or GET /v1/models
# n_ctx_seq (196608) < n_ctx_train (262144) is expected
```

Smoke-test order: (1) load, (2) **first decode** (Metal can load then OOM on first token), (3) new Pi session with real `ls` / `read` tools.

### Fallbacks if you Metal-OOM

1. Close other apps (browsers, IDEs, other local servers).
2. Drop batch to `256` or `128`.
3. Drop context (e.g. `131072` or `65536`) rather than enabling bare `--fit on`.
4. Optional: keep context high and compress V with turbo (see below).

### Optional: more context via TurboQuant V

If you want headroom beyond 196k (or a heavier quant) and accept some decode cost:

```bash
# Same flags as primary, but e.g.:
#   --ctx-size 229376   # try upward carefully; confirm decode, not just load
#   --cache-type-k q8_0 --cache-type-v turbo4   # safer first turbo step on Metal
#   # or turbo3 / turbo2 if still tight
```

Verify output quality after enabling turbo on Metal (see [TurboQuant notes](../llama-cpp-turboquant.md#2-turboquant-kv-cache)). Smoke-test real Pi tools on a **new** session.

### Optional: vision (`mmproj`)

Qwen3.8-27B is a native VLM. For image/video sessions (not required for Pi text/agent work):

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models
```

Add something like `--mmproj ~/Documents/AIML/models/mmproj-F16.gguf` only if your turboquant `llama-server` build exposes multimodal flags for this arch. Keep text/agent primary without mmproj until you need vision.

## Performance notes

- **Starting point:** same **Q5 + q8/q8 + 196k** Metal recipe that was **tested** for Qwen3.6-27B on this hardware.
- Excellent quality/speed balance on 48 GB unified memory for dense Q5; strong agentic use with Pi / Hermes once smoke-tested.
- Metal backend is efficient on Apple Silicon; thermals on sustained load still matter on a laptop.
- After rebuilds, re-check actual `n_ctx` / `n_ctx_seq` and keep Pi’s `contextWindow` in sync.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Twin 3.6 guide: [M5-MacBook-Pro-Qwen3.6.md](M5-MacBook-Pro-Qwen3.6.md). Tighter Metal pattern: [M4 Air](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).

## Pi Coding Agent `models.json`

Save this **entire** file to `~/.pi/agent/models.json` (copy-paste as-is — do not assemble a wrapper). Create parent dirs if needed: `mkdir -p ~/.pi/agent`. **Restart Pi** after writing so the status bar matches the pin.

`maxTokens` ≤ `--n-predict` (8192). `contextWindow` = `--ctx-size` (196608).

**Q5_K_XL @ 196k (recommended):**

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
          "name": "Qwen3.8-27B Q5_K_XL (196k q8/q8) - M5 Pro",
          "contextWindow": 196608,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

## Alternate: higher weight quant (untested)

Primary **Q5 @ 196k** is the ported 3.6 baseline. For a step toward Q8-class fidelity without leaving Metal:

- **Model:** `Qwen3.8-27B-UD-Q6_K_XL.gguf` (~25.9 GB)
- Same `llama-server` command as primary, swap only `--model`.
- Close heavy apps; if Metal-OOM, drop batch or ctx, or use turbo V before abandoning the pin.

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf \
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
          "name": "Qwen3.8-27B Q6_K_XL (196k) - M5 Pro",
          "contextWindow": 196608,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

> There is no mid-size Qwen3.8 MoE twin to the old 35B-A3B Air/M5 alternate yet. Stay on dense 27B quants for this guide.

**Last Updated:** August 14, 2026 (day-zero Qwen3.8 guide; settings ported from tested M5 Metal Qwen3.6)
