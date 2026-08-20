# local-setup.md

Common setup steps for all platforms in **Autarkic-LLM**.

## 1. Prerequisites

### Linux (Ubuntu 24.04+ recommended)

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 python3-pip python3-venv ninja-build

# For CUDA (DGX Spark, RTX cards)
# Install CUDA Toolkit from NVIDIA (JetPack for Jetson)
```

### macOS (Apple Silicon)

```bash
xcode-select --install
brew install cmake git python@3.11
```

### Windows (WSL2 recommended for best results)

Use **WSL2** with Ubuntu. Native Windows builds are possible but less stable for CUDA.

## 2. Clone & Build llama.cpp-turboquant

```bash
cd ~/Documents/GitHub

git clone --depth 1 --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp-turboquant

cd llama-cpp-turboquant

# Clean previous build (recommended when updating)
rm -rf build

mkdir build && cd build
```

**Platform-specific CMake commands** (see hardware-specific guides for exact flags):

**Linux / CUDA:**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)
```

> **CUDA architectures (optional but recommended).** Omit `-DCMAKE_CUDA_ARCHITECTURES` to autodetect on the build machine. Set it explicitly when cross-compiling or sharing binaries:
>
> | GPU | `-DCMAKE_CUDA_ARCHITECTURES` |
> | --- | --- |
> | Jetson Orin Nano | `"87"` |
> | RTX 4090 (Ada) | `"89"` |
> | RTX 6000 Pro Max-Q (Blackwell) | `"120"` |
> | DGX Spark GB10 | `"121"` |

**Linux / Vulkan (AMD RDNA3):**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build . --config Release -j$(nproc)
```

> Vulkan is the recommended backend on AMD Radeon GPUs (e.g. RX 7900 XTX). Do not build with `-DGGML_CUDA=ON` unless you also have an NVIDIA GPU.

**macOS (Metal):**

> **Metal turbo4 `rnorm`:** on current TheTom tip (**`b01afefed` / PR #200 content or later**), no manual Metal shader edit is required — Apple hardware guides assume this. If you are on an older fork commit and `llama-server` crashes at startup with `no member named 'rnorm'`, either `git pull` to tip or delete the two zero-writes to `rnorm` in `ggml/src/ggml-metal/ggml-metal.metal` (`dst.rnorm = half(0.0f);` and `blk.rnorm = half(0.0f);…`). Details: [PR #200](https://github.com/TheTom/llama-cpp-turboquant/pull/200).

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)
```

## 3. Model Download (Unsloth GGUF)

### Model catalog (Hugging Face)

All configs use [Unsloth](https://huggingface.co/unsloth) GGUF builds (Dynamic / "UD" quants) so they run on [llama.cpp](https://github.com/ggml-org/llama.cpp). Quant suffixes (`Q4`, `Q6`, `IQ2`, `_K_XL`, …) trade file size/memory for quality. **How to read those names and pick a level** is in [Understanding GGUF quants](#understanding-gguf-quants-why-so-many-files) below. Short defs: [Glossary](glossary.md) · [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).

| Model | Type | GGUF files & downloads | Original weights |
| --- | --- | --- | --- |
| **Muse Glimmer 30B** | Dense VLM, 131K ctx (⚠️ Dual RTX untested) | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/tree/main) | [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) |
| **Qwen3.8-27B** | Dense VLM, 262K ctx (✅ Dual RTX tested 2026-08-14) | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main) | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Qwen3.6-27B | Dense, 262K ctx (field-tested paths) | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Qwen3.6-35B-A3B | MoE (3B active) | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) | [QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6) |
| Gemma 4 E2B | Dense edge (PLE) | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) | [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) |

Collections: [Muse Glimmer (Unsloth)](https://huggingface.co/collections/unsloth/muse-glimmer) · [Qwen3.8 (Unsloth)](https://huggingface.co/collections/unsloth/qwen38) · [Qwen3.6 (Unsloth)](https://huggingface.co/collections/unsloth/qwen36) · [Gemma 4 (Unsloth)](https://huggingface.co/collections/unsloth/gemma-4). MTP variants (e.g. `*-MTP-GGUF`) offer ~1.5–2× faster decode via multi-token prediction. Muse Glimmer’s analog is **DFlash** (`dflash-kquant.gguf`, `--spec-type draft-dflash`). **Default rule:** pick the largest / highest-quality quant that still leaves headroom for OS + KV at your pinned context — each [hardware guide](README.md#hardware-configurations-included) names the exact file.

**Qwen3.8 guides** (knobs from each box’s tested 3.6 path): [Dual RTX 6000](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) **✅ Tested** (2026-08-14, Pi) · [DGX Spark](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) ⚠️ untested · [M5 MacBook Pro](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md) ⚠️ untested. Need a fresh turboquant build (`qwen35` arch).

**Muse Glimmer guide:** [Dual RTX 6000](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) ⚠️ untested (2026-08-14). Need llama.cpp / turboquant **`b10353+`** (`muse-glimmer` arch). Official sampling is **temp 1.0 / top_p 0.95 / top_k 64**. Thinking **cannot** be switched off (`--reasoning off` is a no-op); use `reasoning_strength` (`low`/`medium`/`high`/`xhigh`). Optional **DFlash**: `--spec-type draft-dflash` + `dflash-kquant.gguf`. Docs: [Unsloth](https://unsloth.ai/docs/models/muse-glimmer) · [Meta llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/).

**Unsloth extras worth knowing** ([Qwen3.8 guide](https://unsloth.ai/docs/models/qwen3.8) · [MTP](https://unsloth.ai/docs/models/mtp) · [Muse Glimmer](https://unsloth.ai/docs/models/muse-glimmer)): UD GGUFs are **Dynamic V3.0** (developer-role + better nested tool calls); Qwen optional **`--spec-type draft-mtp --spec-draft-n-max 2`** for ~1.4–2.2× decode on CUDA; official Qwen sampling differs for thinking (temp 1.0) vs instruct (temp 0.7 / presence 1.5) — Pi tool sessions on **Qwen** keep presence **0** (see Dual RTX Qwen guide). Blackwell boxes may also try **NVFP4** via vLLM/SGLang (different stack).

### Understanding GGUF quants (why so many files)

The Hugging Face “Files” tab is **not** twenty different models. It is **one** base model (e.g. Qwen3.8-27B), packaged many ways so a 16 GB laptop and a 192 GB workstation can both load something useful. Each `.gguf` is a different **compression recipe** of those weights. `llama-server` loads **one** weights file at a time; you pick the tradeoff that fits your memory and quality bar.

**Analogy:** same album, different bitrates. Higher bits → closer to the master, bigger file. Lower bits → smaller / faster to stream through memory, more quality loss.

This section is the repo-wide reference (same ideas as the Qwen3.8 hardware guides). Hardware guides still win for the **exact** filename and pin on your box.

#### Anatomy of a name

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
| **`Qwen3.8-27B`** (or `Qwen3.6-…`, `gemma-4-…`) | Which model | Same “brain”; only the suffix after this changes quality/size |
| **`UD-`** | **U**nsloth **D**ynamic | Per-layer mixed bits: sensitive tensors (attention, hybrid/DeltaNet-ish parts) stay higher precision; less sensitive ones compress harder. Usually **better quality at a given size** than a plain uniform quant |
| **`Q8` / `Q6` / `Q5` / `Q4` / `Q3` / `Q2`** | Rough **bits per weight** | Higher number → larger file, higher fidelity, often slightly slower decode. **Q8 ≈ near full precision; Q4 ≈ common “fits most GPUs” sweet spot** |
| **`IQ2` / `IQ3` / `IQ4_…`** | **I**mportance-aware low-bit schemes | Even more aggressive size cuts; great when memory is tight; can be a bit slower or pickier than plain `Q*_K` at the same size |
| **`_K`** | **K-quant** layout | Modern llama.cpp default family (super-blocks + scales). Prefer these over ancient `Q4_0` / `Q5_0` when both exist |
| **`_S` / `_M` / `_L` / `_XL`** | **S**mall / **M**edium / **L**arge / e**X**tra-**L**arge mix | Same bit *label*, different internal mix of which tensors stay fat. **`_M` = balanced default; `_XL` / Unsloth `UD-…_XL` = quality-leaning** (often a little bigger) |
| **No `UD-`** (e.g. `…-Q8_0.gguf`) | “Classic” uniform (or non-Dynamic) quant | Still valid; often slightly worse quality-per-GB than the matching `UD-` file |
| **`mmproj-*.gguf`** | **Vision projector** | Not the LLM. Optional sidecar for image/video. Text/agent work does **not** need it |
| **`imatrix_*.gguf`** | Calibration matrix used *while building* quants | You generally **do not** download this to run the model |

#### Why HF lists so many near-duplicates

| You see… | Why it exists |
| --- | --- |
| `UD-Q4_K_XL` **and** `Q4_K_M` **and** `Q4_0` | Same ~4-bit *idea*, different recipes (Dynamic XL vs plain K_M vs old-style) |
| Q8, Q6, Q5, Q4, Q3, Q2 / IQ* | Ladder of memory budgets (edge → laptop → desktop → workstation) |
| BF16 / F16 (if present) | Barely compressed reference; huge; rarely needed when Q5–Q8 already feels strong |
| MTP-labeled repos | Multi-token prediction packaging for speed — separate from the main UD ladder |

Extra files on the page are **menu options**, not dependencies (except optional `mmproj` for vision).

#### Q8 vs Q6 vs Q5 vs Q4 (quality vs speed)

Relative quality below follows Unsloth Dynamic **Qwen hybrid** dense trends (3.5/3.6 family; lower KLD ≈ closer to full precision) plus general llama.cpp K-quant behavior — there are no public day-zero **Qwen3.8** KLD tables yet. **Speed** on CUDA/Metal is often **memory-bandwidth bound**: lower bits move fewer weight bytes → modestly higher tok/s; prefill is more compute-bound so gaps shrink.

**Example sizes — Qwen3.8-27B Unsloth `UD-*_K_XL`** ([HF repo](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)):

| Quant (Unsloth UD) | File size | Quality (vs full / Q8) | Decode speed (relative) | When to use |
| --- | --- | --- | --- | --- |
| **UD-Q8_K_XL** | **~31.5 GB** | Near-lossless; tiny residual vs BF16/F16. Best agent/coding fidelity. | Baseline (slowest of the four) | Max quality when VRAM/unified memory is plentiful ([Dual RTX 6000](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) ✅ tested primary) |
| **UD-Q6_K_XL** | **~25.9 GB** | Excellent / “practically perfect.” Small step from Q8 on prior Qwen hybrids | ~5–15% faster than Q8 (typical) | Near-Q8 quality with a few GB free ([DGX Spark](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) primary class) |
| **UD-Q5_K_XL** | **~20.2 GB** | Very high; slight edge-case loss vs Q6/Q8 on multi-step reasoning / hard coding | ~10–20% faster than Q8 | Strong laptop / 48 GB balance ([M5 Pro](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md) primary class) |
| **UD-Q4_K_XL** | **~17.9 GB** | Strong for size; Dynamic Q4 often best Pareto on prior Qwen GGUFs. More visible loss vs Q8 on long agent loops | Fastest of the four (~15–25%+ vs Q8 when bandwidth-bound) | Fit tighter cards, multi-instance, thermals, or speed A/Bs ([Win RTX 4090](Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md)-class) |

**How to read the steps**

- **Quality:** Q8 → Q6 is usually *tiny*; Q6 → Q5 still small; **Q5 → Q4** is the first step many people *feel* on hard coding / agent tool loops. Unsloth Dynamic + imatrix narrows the Q4 gap vs plain `Q4_K_M`.
- **Speed:** Often smaller than folklore on modern GPUs; exact tok/s need `llama-bench` on *your* box.
- **KV is separate:** weight quant is the `.gguf` file. Context memory is `--cache-type-k/v` (and TurboQuant V) in the hardware guide — see [llama-cpp-turboquant.md](llama-cpp-turboquant.md#2-turboquant-kv-cache). Don’t confuse “Q8 weights” with “q8_0 KV”.

#### How to choose (and fine-tune) a quant for *your* machine

This is **deployment tuning** (which compressed weights to load), not training a new model (LoRA / full fine-tune). Order of operations:

1. **Start from your hardware guide** — it already picked a quant that fits that box’s tested or ported pin.  
2. **Budget memory as three buckets:** weights (fixed by quant) + **KV** (grows with `--ctx-size` × K/V precision) + OS / apps / compute scratch.  
3. **Prefer quality while it fits.** On roomy boxes, stay high (Q8/Q6). On tight boxes, drop weight quant *or* compress KV V *or* lower context — don’t silently rely on `--fit on` (it can crush agent context).  
4. **Tune one axis at a time** if something hurts:

| Goal | Prefer | Avoid first |
| --- | --- | --- |
| Max agent / coding fidelity | Higher weight quant (Q8 → Q6 → Q5); keep **K** at `q8_0` | Crushing K, or DRY on Qwen tools |
| Longer context / multi-agent prompts | Raise `--ctx-size` + Pi `contextWindow` together; then turbo **V** only if needed | Dropping to Q2/IQ2 while still using huge unused VRAM |
| Faster decode / cooler laptop | One step down weight quant (e.g. Q5 → Q4) or smaller batch | Jumping many steps at once |
| Fit a small GPU | Lower weight quant **and** realistic ctx; see [M4 Mini Qwen experiment](M4-Mac-Mini-16GB/M4-Mac-Mini-Qwen3.6.md) for tight-budget reasoning | Assuming MoE “3B active” means only 3B of weights must fit |

5. **Smoke-test after every change:** load → **first decode** (Metal can load then OOM) → short Pi tool loop (`ls` / `read`) on a **new** session.  
6. **Match Pi:** if you change only the quant file, keep `contextWindow` / `maxTokens` aligned with server `--ctx-size` / `--n-predict`. Update `id`/`name` in `models.json` if you track them.

**Practical pick ladder (dense ~27B, Unsloth UD):**  
Q8 (max quality) → Q6 (near-Q8, leaner) → Q5 (strong mid) → Q4 (common fit) → IQ3/IQ2 (last resort / edge). Prefer `UD-…_K_XL` over plain `Q*_K_M` / `Q4_0` when both exist at a similar size.

Further reading: [Unsloth Dynamic GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) · [Qwen3.8 run guide](https://unsloth.ai/docs/models/qwen3.8) · hardware examples [Dual RTX](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) · [DGX Spark](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) · [M5 Pro](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md).

### Download

> **Disk space:** GGUFs are large. Muse Glimmer 30B `UD-Q8_K_XL` is ~32.3 GB (`UD-Q6_K_XL` ~26.3 GB, `UD-Q4_K_XL` ~15.9 GB; optional `dflash-kquant.gguf` ~1.6 GB). Qwen3.8-27B `UD-Q8_K_XL` is ~31.5 GB, `UD-Q6_K_XL` ~25.9 GB, `UD-Q5_K_XL` ~20.2 GB, `UD-Q4_K_XL` ~17.9 GB. Qwen3.6-27B `Q6_K_XL` is ~22 GB; 35B-A3B `Q4_K_XL` ~22 GB (down to ~11.5 GB for the `IQ2_M` used on 16 GB Macs); Gemma 4 E2B `Q4_K_S` ~3 GB. Make sure you have the room — and note `hf_transfer` downloads can momentarily use extra space.

```bash
pip install -U huggingface_hub hf_transfer

# Muse Glimmer example (hf is the current CLI)
# Hardware guides use ~/Documents/AIML/models as a flat local-dir; any path works if --model matches.
hf download unsloth/Muse-Glimmer-30B-GGUF \
  Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models

# Qwen3.8 example (day-zero)
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models

# Qwen3.6 example (field-tested guides)
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models

# Gemma example (Mac Mini / Jetson guides)
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_S.gguf \
  --local-dir ~/Documents/AIML/models
```

> **Gated models:** the Unsloth GGUF repos above are generally open, but the *original* Google/Qwen weights (and some mirrors) may be gated. If a download 401s, run `hf auth login` and accept the model's license on its Hugging Face page first. Confirm the exact quant filename on the repo's "Files" tab: available quants vary per model.

## 4. Create KV Cache Directory

```bash
# From the build directory; the llama-server binary is in build/bin
cd bin
mkdir -p ./kv-cache
```

## 5. Common Best Practices

- **Network exposure (read this).** Guides default to `--host 127.0.0.1` (loopback only). Binding `--host 0.0.0.0` serves the model to *every device on your network with no authentication* — and typically with a shell-capable [agentic harness](agentic-harnesses.md) behind it. That is at odds with the "nothing leaves the box" goal. Keep `0.0.0.0` **only** on a trusted LAN where you deliberately reach the server from other machines, and put it behind a firewall.
- Always run `pkill -9 llama-server` before starting a new instance. The `-9` (SIGKILL) is deliberate, not lazy: llama-server's graceful shutdown can hang on SIGTERM/SIGINT (upstream issues [#11742](https://github.com/ggml-org/llama.cpp/issues/11742), [#20921](https://github.com/ggml-org/llama.cpp/issues/20921)), and it does **not** auto-save useful long-term KV on exit — so a hard kill loses nothing you were relying on for persistence.
- Use `--load-mode none` where the CUDA/Jetson guides previously had `--no-mmap` (same behavior: no file mapping). Do **not** also pass `--no-mmap` / `--mmap` / `--mlock`. Metal guides omit it and keep the default (`mmap`). AMD Vulkan guides never used `--no-mmap` — they stay on the default too.
- **Prefer pinned `--ctx-size` + `--fit off` for agent use (Pi / Hermes).** Default `--fit on` can crush context (sometimes toward ~4096) and break long sessions — documented on the [M4 MacBook Air guide](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md). **All hardware guides in this repo pin context and set `--fit off`.** If you experiment with `--fit on --fit-target <MiB>`, **leave `--n-gpu-layers` and `--ctx-size` unset** — on the current fork it aborts if `--n-gpu-layers` is set and won't shrink a pinned `--ctx-size`. Check the startup log for the context it allocated.
- **Qwen3.6 / Qwen3.8 + agents (Pi):** prefer **`--reasoning off`** (+ `--reasoning-budget 0`) so clients get normal `message.content` / tools. Current llama-server builds often **deprecate** `enable_thinking` via `--chat-template-kwargs` in favor of `--reasoning on|off`. Do not rely on context-checkpoint flags for hybrid Qwen (3.5/3.6/3.8 share the hybrid backbone) — see [checkpointing caveat](llama-cpp-turboquant.md#prompt-cache--checkpointing). For Pi tool stability (no DRY, sampling, KV K/V, `contextWindow` vs `maxTokens`), see [agentic harnesses — dense Qwen 27B + Pi](agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). **Qwen3.8** status: [README — Qwen3.8](README.md#qwen38-2026-08-14).
- **Muse Glimmer + agents (Pi):** **`--reasoning off` does nothing.** Use `--jinja` and `--chat-template-kwargs '{"reasoning_strength":"high"}'` (or `low`/`medium`/`xhigh`). Thinking goes to `reasoning_content` by default. Official sampling is **temp 1.0 / top_p 0.95 / top_k 64**. Need **`b10353+`**. See [Dual RTX Muse Glimmer](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) and [agentic harnesses — Muse](agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent).
- Monitor resources:
  - Linux: `htop`, `nvidia-smi -l 1`
  - macOS: `htop` + Activity Monitor
  - Jetson: `jtop`
- Update procedure: `git pull origin feature/turboquant-kv-cache` → delete `build/` → rebuild.

## 6. Pi Coding Agent / Hermes Integration

Each hardware guide includes a **complete** Pi Coding Agent `models.json`. Copy the entire JSON block from your guide into **`~/.pi/agent/models.json`** as-is (`mkdir -p ~/.pi/agent` first if needed). The shape is always:

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

> **Important:** match these settings to the model you actually loaded with `llama-server`. Mismatched values cause truncation, errors, or wasted memory.
>
> - **`id`** / **`name`**: identify the real model you launched.
> - **`contextWindow`**: must equal (or not exceed) your real server `n_ctx_seq` — with pinned `--ctx-size` + `--fit off`, this is your `--ctx-size`. **Input** limit (history + tools).
> - **`maxTokens`**: must not exceed your `--n-predict`. **Output** limit (one reply). Long reports need both sides raised together (e.g. 8192–16384).
> - Prefer the full file in your hardware guide over inventing numbers. Example values above match the [M4 Air](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) shape only.
> - Restart Pi after every change so the status bar matches the server.

## Next Steps

- Go to your hardware-specific folder (e.g. `DGX-Spark-128GB/`, `Dual-RTX6000-192GB/`, `M5-MacBook-Pro-48GB/`, `Win-RTX4090-24GB/`) for exact build flags, model recommendations, and optimized `llama-server` commands.
- **Qwen3.8:** [Dual RTX](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) **✅ Tested** · [DGX Spark](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md) / [M5 Pro](M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md) ports ⚠️ — overview in [README](README.md#qwen38-2026-08-14).
- **Muse Glimmer:** [Dual RTX](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md) ⚠️ untested — overview in [README](README.md#muse-glimmer-30b-2026-08).
- Pi multi-agent research / Tavily: [`_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md`](_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md).
- For what the TurboQuant fork adds and what every `llama-server` flag does (and when *not* to use it), see [`llama-cpp-turboquant.md`](llama-cpp-turboquant.md).
