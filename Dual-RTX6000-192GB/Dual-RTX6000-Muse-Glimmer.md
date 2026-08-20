# Dual RTX 6000 Pro Max-Q (192 GB) - Muse Glimmer 30B

Optimized setup for dual NVIDIA RTX 6000 Pro Max-Q GPUs (192 GB total VRAM) on Ubuntu, using **llama-cpp-turboquant**. Compute and memory knobs match the field-validated [Qwen3.8 Dual RTX 6000 guide](Dual-RTX6000-Qwen3.8.md) (single-GPU Q8, pinned context, `q8/q8` KV, Pi-shaped server). **Agent / sampling / reasoning knobs do not carry over** — Muse Glimmer is a different architecture with different official defaults. Cross-hardware Pi layout (two token limits, no DRY, pinned `contextWindow`) still applies: [agentic harnesses](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent).

> ⚠️ **Not yet tested on this hardware with Muse Glimmer.** Model released **August 2026** (Meta Superintelligence Labs, Apache 2.0). Primary command below is a researched starting point from [Meta’s llama.cpp deploy guide](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/), [Unsloth’s run notes](https://unsloth.ai/docs/models/muse-glimmer), and this box’s existing Dual RTX CUDA / Pi pin. Confirm **load → first decode → Pi tools** on a fresh session before relying on it. This box’s current turboquant (`b10465` / tip of `feature/turboquant-kv-cache`) already registers `muse-glimmer` and `draft-dflash`; older checkouts will not.

### How this differs from Dual RTX Qwen3.8

| Piece | Qwen3.8 (tested 2026-08-14) | Muse Glimmer (this guide) |
| --- | --- | --- |
| Hardware / engine | Dual RTX 6000, turboquant CUDA CC 12.0 | Same |
| Size class | Dense ~27B, UD-Q8_K_XL **~31.5 GB** | Dense **~29.6B** (incl. vision encoder), UD-Q8_K_XL **~32.3 GB** |
| Native context | 262144 | **131072** (Unsloth / Meta: longer contexts supported — optional **262144**) |
| Layout | Hybrid Gated DeltaNet + Gated Attention (`qwen35`) | Dense transformer, **gated attention**, **SWA 2048** on 3 of every 4 layers, GQA **32 Q / 2 KV** (`muse-glimmer`) |
| Primary pin | single GPU, q8/q8 KV, 16k out, **thinking off** | single GPU, q8/q8 KV, **32k out**, **thinking cannot be switched off** |
| Official sampling | Qwen thinking 1.0 / instruct 0.7 / Pi tools 0.6 · `top_k` 20 | Meta / Unsloth: **temp 1.0 · top_p 0.95 · top_k 64** |
| Speed companion | Optional **MTP** (`draft-mtp`) | Optional **DFlash** (`draft-dflash`, ~1.6 GB draft) |
| Vision | Optional `mmproj` | Optional `mmproj` (ViT-G/14 perception encoder; text/agent primary still) |
| KV at long ctx | Roomy on one 96 GB card | **Very cheap** — only ~¼ of layers keep a full-length cache |

GGUF catalog: [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) (Unsloth Dynamic). Official weights / GGUF: [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) · [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF). Run notes: [Unsloth Muse Glimmer](https://unsloth.ai/docs/models/muse-glimmer) · [Meta llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/) · [DFlash](https://dev.meta.ai/docs/muse-glimmer/spec-decode) · [prompting](https://dev.meta.ai/docs/muse-glimmer/prompting).

Need llama.cpp **`b10353` or newer** (PR [#26841](https://github.com/ggml-org/llama.cpp/pull/26841), commit `62bf73d`). Older builds refuse the file: `unknown model architecture: 'muse-glimmer'`.

## Why so many GGUFs? (quick naming decoder)

The Hugging Face folder is **not** twenty different models. It is **one** Muse Glimmer 30B (same weights family), packaged many ways so people with 12 GB laptops and people with 192 GB workstations can both load something useful. Each `.gguf` is a different **compression recipe** of those weights. llama.cpp loads one file at a time; you pick the tradeoff that fits your VRAM and quality bar.

**Analogy:** same album, different bitrates. Higher bits → closer to the master, bigger file. Lower bits → smaller/faster to move through memory, more quality loss.

### Anatomy of a name

Example primary file:

```text
Muse-Glimmer-30B-UD-Q8_K_XL.gguf
│               │  │  │  │  └─ size tier within that recipe (XL = roomier / higher quality mix)
│               │  │  │  └──── K-quant family (block scales; better quality-per-bit than old Q4_0-style)
│               │  │  └─────── bit width (~8-bit weights on average)
│               │  └────────── Unsloth Dynamic mixed-precision method
│               └───────────── parameter class (dense ~30B)
└───────────────────────────── model family
```

| Piece | What it means | How to read it |
| --- | --- | --- |
| **`Muse-Glimmer-30B`** | Which model | Same “brain”; only the file after this changes quality/size |
| **`UD-`** | **U**nsloth **D**ynamic | Per-layer mixed bits: sensitive tensors stay higher precision; less sensitive ones compress harder. Usually **better quality at a given size** than a plain uniform quant |
| **`Q8` / `Q6` / `Q5` / `Q4` / `Q3` / `Q2`** | Rough **bits per weight** | Higher number → larger file, higher fidelity, often slightly slower decode. **Q8 ≈ near full precision; Q4 ≈ common “fits most GPUs” sweet spot** |
| **`IQ2` / `IQ3` / `IQ4_…`** | **I**mportance-aware low-bit schemes | Even more aggressive size cuts; great when VRAM is tight; can be a bit slower or pickier than plain `Q*_K` at the same size |
| **`_K`** | **K-quant** layout | Modern llama.cpp default family (super-blocks + scales). Prefer these over ancient `Q4_0` / `Q5_0` when both exist |
| **`_S` / `_M` / `_L` / `_XL`** | **S**mall / **M**edium / **L**arge / e**X**tra-**L**arge mix | Same bit *label*, different internal mix of which tensors stay fat. **`_M` = balanced default; `_XL` / Unsloth `UD-…_XL` = quality-leaning** (often a little bigger) |
| **No `UD-`** (e.g. `Muse-Glimmer-30B-Q8_0.gguf`) | “Classic” uniform (or non-Dynamic) quant | Still valid; often slightly worse quality-per-GB than the matching `UD-` file from Unsloth |
| **`mmproj-*.gguf`** | **Vision projector** (perception encoder) | Not the LLM. Optional sidecar for images. Text/agent work does **not** need it |
| **`dflash-*.gguf` / `dflash-kquant.gguf`** | **DFlash drafter** | Not the LLM. Optional companion for speculative decode. See [DFlash](#optional-dflash-decode-speedup) |
| **`imatrix_*.gguf`** | Calibration matrix used *while building* quants | You generally **do not** download this to run the model |

### Why the repo lists so many near-duplicates

| You see… | Why it exists |
| --- | --- |
| `UD-Q4_K_XL` **and** Meta’s `KQuant-17GB-Q4_K_M` / `KQuant-Dynamic-Q4_K_XL` | Same ~4-bit *idea*, different recipes (Unsloth Dynamic vs Meta’s official 24/32 GB targets) |
| Q8, Q6, Q5, Q4, Q3, Q2 / IQ* | Ladder of VRAM budgets (laptop → desktop → workstation) |
| BF16 (folder on Unsloth; no official Meta BF16 GGUF) | Barely compressed reference; huge; rarely needed when Q8 already “feels full” |
| `dflash-*` + `mmproj-*` | Speed and vision sidecars — **not** extra copies of the LLM |

You only need **one** weights file for `llama-server`. Extra files on the page are menu options, not dependencies (except optional `mmproj` for vision and optional `dflash` for speculation).

### Practical pick order (this 192 GB box)

1. **`UD-Q8_K_XL`** — this guide’s primary (max quality; you have the VRAM).
2. **`UD-Q6_K_XL`** — if you want near-Q8 quality and a few GB free.
3. **`UD-Q5_K_XL` / `UD-Q4_K_XL`** — only if you care about multi-instance, thermals, or speed A/Bs. Meta measured **~0.2%** average degradation on their Dynamic 4-bit vs full precision (15-benchmark mean) — unusually small, but this box does not need to take that trade.
4. Skip plain non-`UD` and very low IQ* unless you are size-constrained or comparing recipes.

More background (repo-wide): [local-setup — Understanding GGUF quants](../local-setup.md#understanding-gguf-quants-why-so-many-files) · [Glossary](../glossary.md) · [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs). Bit-ladder sizes/speed for *this* model: [Weight quant table](#weight-quant-q8-vs-q6-vs-q5-vs-q4-this-model) below.

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq`. `--parallel` **splits** `-c` across slots (`n_ctx_slot = ctx / parallel`) — keep **`--parallel 1`** so one Pi session gets the full pin. Muse reasons at length; a 32k slot can fill on thinking alone, and a generation that runs out of room can return **empty `content` with no error**.

**Thinking cannot be switched off.** The chat template always opens a `to=self` reasoning channel. **`--reasoning off`**, **`--reasoning on`**, and `"reasoning_effort": "none"` are **no-ops** on Muse Glimmer. Control *how much* it thinks with `reasoning_strength`: `low` / `medium` / `high` / `xhigh` (template default **`high`**). Meta: use **high or xhigh** for coding and agentic tasks.

On `llama-server`, thinking lands in `message.reasoning_content` and **`message.content` stays clean** (default — no extra flag). That is what Pi wants for tools. Pass `--reasoning-format none` only if you *want* thinking inline in `content` (do not do that for Pi).

**`--jinja` is required.** The Muse template (role tags, ATEM tools, reasoning-strength line) is embedded in the GGUF. Without `--jinja`, tool calling and reasoning split break. Do **not** pass `--chat-template-file`.

**Never add `<|eom|>` as a stop string.** Stops are `<|end_of_text|>` and `<|eot|>`. `<|eom|>` is end-of-*message* (turn continues). Stopping on it collapses tool calling.

Shared Pi layout (two token limits, no DRY, K/V policy): [agentic harnesses](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent). **Do not** copy the Qwen “`--reasoning off` + temp 0.6 / top_k 20” row onto this model.

**Sampling for Pi tools (this model):** keep **Meta’s published defaults** (`temp 1.0`, `top_p 0.95`, `top_k 64`, **`presence_penalty 0`**, `repeat_penalty 1.0`). The Dual RTX Qwen agent profile is for Qwen hybrids, not Muse.

## Recommended model (primary — single GPU)

- **Model:** `Muse-Glimmer-30B-UD-Q8_K_XL.gguf` (~32.3 GB — best quality with massive KV headroom on 192 GB)
- **Intended path:** `~/Documents/AIML/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf`

```bash
hf download unsloth/Muse-Glimmer-30B-GGUF \
  Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

> Day-zero-ish model: use a **fresh** turboquant / llama.cpp build (**`b10353+`**). If load fails with `unknown model architecture: 'muse-glimmer'`, `git pull` the fork and rebuild before changing flags.

Confirm the checkout before you debug flags:

```bash
./llama-server --version
# expect version: 10353 or higher (this box’s tip has been 10465+)

grep -c LLM_ARCH_MUSE_GLIMMER ../../src/llama-arch.cpp
# expect >= 1  (run from build/bin; path is the turboquant clone)
```

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

Confirm turbo types (needed only if you try the optional turbo-V path) and DFlash (needed for the optional speed path):

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4

./llama-server --help | grep -i spec-type
# must list draft-dflash
```

> Omit `-DCMAKE_CUDA_ARCHITECTURES="120"` to autodetect; set explicitly when cross-compiling (Blackwell = CC 12.0).

If the fork checkout is older than Muse support, either update it or build [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `b10353+` with the same CUDA flags. Stock llama.cpp will reject `turbo*` cache types; the primary command below uses `q8_0`/`q8_0` and works on either.

## Optimized llama-server command (PRIMARY — Q8_K_XL @ 131k, single GPU)

**Research baseline (2026-08-14)** — Dual RTX compute/memory house style + **Meta / Unsloth Muse defaults**. Not yet smoked on this box. Use this command first (no DFlash, no mmproj).

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  --alias muse-glimmer-30b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --chat-template-kwargs '{"reasoning_strength":"high"}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 32 \
  --n-predict 32768 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why (researched for this box + this model) |
| --- | --- |
| `--ctx-size 131072` | Official **native** window (Meta server example; Unsloth default). One 96 GB card has room for Q8 + KV with a lot left over. Optional [262k](#optional-262k-context) below |
| `--cache-type-k/v q8_0` | High-precision KV while VRAM allows. Muse KV is **cheap** (GQA 2 KV heads + SWA on 3/4 layers — Meta measured ~2.1 GiB KV+scratch for Q4 + 131k + vision). turbo V is almost never required here |
| **No `--cache-ram 0`** | That flag is a **hybrid Qwen / DeltaNet** multi-turn workaround ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)). Muse is not that backbone — do not copy it blindly |
| `--n-gpu-layers 99` + `--main-gpu 0` | Full offload on the primary GPU on a dual-card box — primary stays single-GPU for simplicity |
| `--jinja` + `--alias muse-glimmer-30b` | Embedded Muse template (required). Alias matches Pi `id` / curl `"model"` |
| `--chat-template-kwargs '{"reasoning_strength":"high"}'` | Meta default for coding / agents. **Not** `--reasoning off` (no-op). Per-request override: `"chat_template_kwargs": {"reasoning_strength":"low"}` |
| **No `--reasoning off`** | Template always opens thinking. Server still splits it into `reasoning_content` so Pi `message.content` / tools stay clean |
| Tool sampling | `temp 1.0` / `top_p 0.95` / `top_k 64` / `presence 0` / `repeat 1.0` — Meta + Unsloth published defaults; **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 32768` | Thinking tokens count against the output cap. Mid-thought ceiling → **empty `content`**, `finish_reason: length`. 16k is tight for `high`/`xhigh`; 32k leaves room. Match Pi `maxTokens` |
| `--parallel 1` + `--fit off` + `127.0.0.1` | Full pin to one slot; loopback default. Do **not** raise `-np` without also scaling `-c` |
| `--flash-attn on` | Required for quantized KV; also what DFlash docs assume |
| No `--swa-full` | Would expand every local layer to full-length KV and throw away the architecture’s cheap-cache design |

### Weight quant: Q8 vs Q6 vs Q5 vs Q4 (this model)

Meta published a 15-benchmark mean **% degradation vs full precision** for their official 4-bit GGUFs: Dynamic K-quant **0.2%**, 17 GB K-quant **1.0%**. There are no public Unsloth KLD/PPL tables for the UD ladder yet. Relative **quality** below follows those Meta 4-bit numbers plus general llama.cpp K-quant / Unsloth Dynamic behavior. **Speed** is bandwidth-bound decode on CUDA: lower bits move fewer weight bytes → modestly higher tok/s; prefill is more compute-bound so gaps shrink. On a **96 GB** card all four fit with full **131k q8/q8** KV (and 262k) — pick quality first unless you want free VRAM for multi-slot / vision / DFlash experiments.

Unsloth `UD-*` sizes from [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF):

| Quant (Unsloth UD) | File size | Quality (vs full / Q8) | Decode speed (relative) | On this Dual RTX 6000 box |
| --- | --- | --- | --- | --- |
| **UD-Q8_K_XL** (primary) | **~32.3 GB** | Near-lossless; tiny residual vs BF16. Best agent/coding fidelity. | Baseline (slowest of the four; still fast here) | **Use this** — VRAM is not the bottleneck |
| **UD-Q6_K_XL** | **~26.3 GB** | Excellent / “practically perfect.” Hard to notice in chat | ~5–15% faster decode than Q8 (typical CUDA) | Fine if you want ~6 GB free without a real quality cliff |
| **UD-Q5_K_XL** | **~21.8 GB** | Very high; slight edge-case loss vs Q6/Q8 on hard multi-step work | ~10–20% faster than Q8 | Good “high quality, leaner” option; still overkill for fit |
| **UD-Q4_K_XL** | **~15.9 GB** | Strong for size. Meta’s own Dynamic 4-bit is **~0.2%** mean degradation — unusually good. More visible loss vs Q8 is still possible on long agent loops | Fastest of the four (~15–25%+ vs Q8 when bandwidth-bound) | Prefer only for multi-instance, lower thermals, or A/B speed tests — not for max quality |

Also on the Unsloth page (not the primary ladder): `Q8_0` **~29.6 GB** (classic uniform 8-bit; prefer `UD-Q8_K_XL`), `UD-Q5_K_M` **~19.2 GB**, plus IQ2/IQ3 and `UD-Q2/Q3_K_XL` for tiny boxes. BF16 weights are **~55.7–58 GB** — they fit one 96 GB card, but Q8 is the quality/size sweet spot here.

**How to read this**

- **Quality steps:** Q8 → Q6 is usually *tiny*; Q6 → Q5 still small; Q5 → Q4 is the first step many people *feel* — and on *this* model Meta’s 4-bit numbers suggest the cliff is smaller than typical. Unsloth Dynamic + imatrix narrows it further vs plain `Q4_K_M`.
- **Speed steps:** Smaller than quality folklore suggests on modern GPUs with good CUDA kernels. Exact tok/s need `llama-bench` on this box.
- **This hardware:** Primary stays **Q8**. Dropping to Q6/Q5 frees VRAM but is optional, not required for 131k or 262k.

Swap only the model path (and Pi `id`/`name` if you track it). Example Q6 download:

```bash
hf download unsloth/Muse-Glimmer-30B-GGUF \
  Muse-Glimmer-30B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

**Confirm**

```text
log: n_ctx_seq (131072)     # or n_ctx_slot = 131072 with n_slots = 1
# model loads as muse-glimmer / Muse-Glimmer-30B (exact string varies by build)
nvidia-smi   # note MiB after load and after a short decode
```

Smoke test (thinking should be in `reasoning_content`, answer in `content`):

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"muse-glimmer-30b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

If `content` begins with `to=self<|message|>`, the chat parser is too old — rebuild (`b10353+`).

Confirm after rebuilds: (1) load, (2) short decode, (3) Pi tools on a fresh session.

### Optional: DFlash decode speedup (Meta)

Muse Glimmer ships with a **DFlash** block-diffusion drafter (5 layers, block size **16**). It proposes a block of tokens; the 30B target verifies them in parallel. Accepted tokens match the no-spec path. Official GGUF draft is ~**1.6 GB**. Meta measured **~3.1×** decode on an RTX 5090 with their 17 GB K-quant + DFlash (greedy, llama.cpp). Expect less than 3× under temp **1.0** sampling (community vLLM on a Blackwell Pro 6000 saw ~2.3×).

Primary above is the simpler first-boot path **without** DFlash. For speed, download the Unsloth-mirrored draft and add the flags (after a build that lists `draft-dflash` in `--help`):

```bash
hf download unsloth/Muse-Glimmer-30B-GGUF \
  dflash-kquant.gguf \
  --local-dir ~/Documents/AIML/models
```

```bash
# Same PRIMARY command as above, plus:
  --spec-draft-model ~/Documents/AIML/models/dflash-kquant.gguf \
  --spec-type draft-dflash \
  --spec-draft-ngl 99 \
  --spec-draft-n-max 4
```

| Tip | Detail |
| --- | --- |
| Start with **`n-max 4`** | [Meta llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/) starting point. Draft block is 16; the flag is **clamped** to the trained block. Try **4–15** and keep the fastest on *this* box |
| Confirm support | `./llama-server --help \| grep -i spec` — need `draft-dflash` / `--spec-type` |
| Harmless warning | `[spec] failed to measure draft model memory` at startup — draft still loads |
| 262k + DFlash | If you raise context (below), also override the draft: `dflash.context_length=int:262144` |
| Pi tools | DFlash should not change quality of accepted tokens; still re-smoke `ls`/`read` once |
| If flags missing | `git pull` turboquant / rebuild — need the Muse-era speculative path |

Example (primary + DFlash only):

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  --spec-draft-model ~/Documents/AIML/models/dflash-kquant.gguf \
  --alias muse-glimmer-30b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --chat-template-kwargs '{"reasoning_strength":"high"}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 32 \
  --n-predict 32768 \
  --kv-unified \
  --spec-type draft-dflash \
  --spec-draft-ngl 99 \
  --spec-draft-n-max 4 \
  --log-verbosity 1
```

Official Meta draft filename is `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` in [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF); Unsloth’s `dflash-kquant.gguf` is the same companion mirrored into the UD repo.

### Optional: 262k context

Unsloth documents **131,072 default, up to 262,144**. GGUF metadata is often `muse-glimmer.context_length = 131072`, and llama-server will **cap** a larger `-c` to that unless you override it. Community-confirmed llama.cpp approach:

```bash
# Same PRIMARY command, only:
#   --ctx-size 262144 \
#   --override-kv "muse-glimmer.context_length=int:262144"
# If DFlash is loaded, include the draft too:
#   --override-kv "muse-glimmer.context_length=int:262144,dflash.context_length=int:262144"
```

Then set Pi `contextWindow` to **262144**. If the log still says `exceeds the training context ... capping`, write the metadata (from the llama.cpp / turboquant clone):

```bash
python gguf-py/gguf/scripts/gguf_set_metadata.py \
  ~/Documents/AIML/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  muse-glimmer.context_length 262144
```

KV stays small even at 262k (SWA + 2 KV heads). Quality of the *extended* half of the window is less documented than native 131k — smoke-test long prompts before you treat 262k as the daily pin.

### Optional: reasoning strength & budget

| Level | When |
| --- | --- |
| **`high`** (primary) | Meta default; coding / agentic tasks |
| **`xhigh`** | Hardest problems; burns more thinking tokens (raise `--n-predict` if you still hit `length`) |
| **`medium` / `low`** | Snappier tool loops, validated answers, cheap retries |

Server-wide (already in primary):

```bash
  --chat-template-kwargs '{"reasoning_strength":"high"}'
```

Per request:

```json
{"chat_template_kwargs": {"reasoning_strength": "low"}}
```

The OpenAI spelling `reasoning_effort` is **not** what llama.cpp’s Muse template reads — use `reasoning_strength`. You can also put `Reasoning strength: high` in the system prompt; a current GGUF template will not append a second line if one is already there.

Hard-cap thinking tokens (optional, not in primary):

```bash
#   --reasoning-budget 8192
```

A budget that is too small clips the model mid-thought the same way a tight `--n-predict` does. Prefer a large output cap first.

### Optional: Dual RTX Qwen-style sampling (A/B only)

Keep Meta’s row for this model. If you want a side-by-side with the [Qwen3.8 Dual RTX Pi profile](Dual-RTX6000-Qwen3.8.md) (path-heavy tools, cooler outputs):

```bash
# Not the Muse primary — A/B only:
#   --temp 0.6 --top-p 0.95 --top-k 20 --presence-penalty 0.0
```

Do **not** bring over Qwen instruct **presence 1.5** — it can warp reused path tokens in Pi shell/write agents.

### Optional: TurboQuant V (capacity / decode tradeoff)

Primary already has enormous headroom (~32 GB weights + a few GB of SWA KV on a 96 GB card). If you later raise load (multi-slot, vision + DFlash + 262k + large batches) and need to free KV:

```bash
# Same command as above, only:
#   --cache-type-k q8_0 --cache-type-v turbo4
# Keep K at q8_0. Smoke-test real ls/read on a new Pi session first.
```

You almost certainly will not need this for a single 30B + 131k/262k session.

### Optional: vision (`mmproj`)

Muse Glimmer is a native VLM (text + image in, text out; video is frames). Perception encoder is a frozen **ViT-G/14 ~1.8B**. For multimodal sessions (not required for Pi text/agent work). Actual Unsloth filenames (the Unsloth tutorial’s `mmproj-BF16.gguf` is a shorthand):

```bash
hf download unsloth/Muse-Glimmer-30B-GGUF \
  mmproj-Muse-Glimmer-30B-BF16.gguf \
  --local-dir ~/Documents/AIML/models
```

Leaner sidecars on the same repo: `mmproj-Muse-Glimmer-30B-Q8_0.gguf` (~2.05 GB), `mmproj-kquant.gguf` (~1.4 GB, Meta Q4 mirror).

Add `--mmproj ~/Documents/AIML/models/mmproj-Muse-Glimmer-30B-BF16.gguf` to the server command. Images are billed as prompt tokens (up to **4096** visual tokens per image). Keep text/agent primary **without** mmproj until you need vision.

CLI images go through `llama-mtmd-cli`, not `llama-cli` (and still need `--jinja`).

### Optional: other stacks (vLLM / SGLang)

This repo’s primary remains **GGUF + llama-cpp-turboquant + Pi**. Muse also has first-class **vLLM** and **SGLang** recipes (BF16 / FP8 / NVFP4 + native DFlash assistant weights). Dual RTX 6000 Pro Max-Q is **Blackwell (CC 12.0)** — those stacks are viable **if you leave the GGUF path**. See [Meta deploy](https://dev.meta.ai/docs/muse-glimmer/) and [SGLang Muse Glimmer](https://lmsysorg.mintlify.app/cookbook/autoregressive/Meta/MuseGlimmer). Unsloth’s current Muse collection is GGUF + bitsandbytes 4-bit + BF16, not a separate NVFP4 GGUF ladder like Qwen3.8.

## Performance notes

- **Not field-validated on this box yet.** Smoke-test before calling the pin “tested.” Expected: Q8 + 131k q8/q8 + 32k out fits one 96 GB card with tens of GB free (Meta’s Q4+vision+131k was **~19 GiB** total; Q8 adds ~15 GB of weights, not KV).
- KV is the reason 131k/262k is easy: **only the global layers** (1 of every 4) keep a full-length cache; local layers window at **2048**. Do not enable `--swa-full`.
- Primary uses **one GPU** by default. The other 96 GB card stays free (or idle).
- **Next speed lever on this box:** optional **DFlash** (`--spec-type draft-dflash`) before dropping weight quant. This is Muse’s analog of Qwen MTP — different draft, different flag.
- TurboQuant **fork** ≠ must use turbo **types**. Keep `q8_0`/`q8_0`; turbo V is a capacity lever, not the quality default.
- Official Muse sampling is **one** row (not Qwen’s thinking-vs-instruct split). Do not apply the Dual RTX Qwen Pi `temp 0.6` / `top_k 20` profile unless you are deliberately A/B testing.
- Tool calls are **ATEM** on the wire; `llama-server` parses them into OpenAI `tool_calls` when `--jinja` is on. Muse does **one tool call per turn** (no parallel tools).
- Knowledge cutoff: **2026-01-04**. License: **Apache 2.0**.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Cross-hardware Pi: [agentic harnesses — Muse](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent). Twin Qwen3.8 guide: [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-Qwen3.8.md). Unsloth: [Muse Glimmer](https://unsloth.ai/docs/models/muse-glimmer). Meta: [llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/) · [DFlash](https://dev.meta.ai/docs/muse-glimmer/spec-decode) · [prompting](https://dev.meta.ai/docs/muse-glimmer/prompting).

## Pi Coding Agent `models.json`

Save this **entire** file to `~/.pi/agent/models.json` (copy-paste as-is — do not assemble a wrapper). Create parent dirs if needed: `mkdir -p ~/.pi/agent`. **Restart Pi** after writing so the status bar matches the pin.

`maxTokens` ≤ `--n-predict` (32768). `contextWindow` = `--ctx-size` (131072). If you take the [262k](#optional-262k-context) option, change **both** the server pin and `contextWindow`.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "muse-glimmer-30b",
          "name": "Muse Glimmer 30B Q8_K_XL (131k q8/q8) - Dual RTX 6000",
          "contextWindow": 131072,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

## Alternate: multi-GPU layer split

> ⚠️ **Not needed for this 30B Q8 pin.** One 96 GB card holds weights + 131k/262k KV + DFlash + mmproj. Use multi-GPU only if a future larger quant/model does not fit on one card or you deliberately want to spread load.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  --alias muse-glimmer-30b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --split-mode layer \
  --tensor-split 96,96 \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --chat-template-kwargs '{"reasoning_strength":"high"}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 32 \
  --n-predict 32768 \
  --kv-unified \
  --log-verbosity 1
```

- Same agent flags as primary. Multi-GPU adds PCIe sync overhead; benchmark against single-GPU before committing.

**Last Updated:** August 14, 2026 (⚠️ untested on this box; knobs from [Meta llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/) + [Unsloth Muse Glimmer](https://unsloth.ai/docs/models/muse-glimmer) + Dual RTX Qwen3.8 house style)
