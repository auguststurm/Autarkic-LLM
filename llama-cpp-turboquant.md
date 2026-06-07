# llama-cpp-turboquant: Deep Dive & Flag Reference

A deeper companion to [`local-setup.md`](local-setup.md): what the **llama-cpp-turboquant** fork adds over upstream llama.cpp, plus a flag-by-flag reference for the `llama-server` commands in the hardware guides (what each flag does, why we set the value we do, and when *not* to use it).

> Flag names and defaults below were checked against the [llama.cpp server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) and the fork. Behavior evolves: always confirm with `./llama-server --help` on your own build. Where a flag is fork-specific or non-standard, it's marked.

---

## Key learnings (TL;DR)

- **TurboQuant KV cache** (`--cache-type-k q8_0 --cache-type-v turbo4`) gives major speed/memory wins. `turbo*` is supported on Metal, CUDA, and Vulkan; the Mac guides use `q8_0`/`q8_0` as a conservative, tested baseline, but `turbo4`-V works there too (§2).
- **Stable sampling baseline:** `--repeat-penalty 1.10 --presence-penalty 0.0 --frequency-penalty 0.0 --min-p 0.0 --temp 0.65 --top-p 0.90` (Gemma: `--temp 0.75 --top-p 0.92`).
- **Checkpointing + cache warming** (`--ctx-checkpoints`, `--cache-ram`, `--slot-save-path`) can improve prefill/long-context stability, but see the Qwen3.x caveat in §5: it may not restore on hybrid models.
- **Conservative batch sizes** and **`--no-context-shift`** prevent crashes / silent context loss on memory-constrained systems.
- **Small dense models** (Gemma 4 E2B, ~2.3B effective params with Per-Layer Embeddings) and **MoE models with low active parameters** (Qwen3.6-35B-A3B) both excel on lower-memory devices, but the MoE's *full* weights must still fit in RAM. (Gemma 4 E2B is dense, "E" = edge/effective; the family's MoE member is the 26B-A4B.)

---

## 1. What the fork is

- **Repo / branch:** <https://github.com/TheTom/llama-cpp-turboquant>, branch **`feature/turboquant-kv-cache`** (the default branch is `master`, which does *not* carry these features; always clone/checkout the feature branch).
- It's a **long-lived feature branch that continuously syncs from `ggml-org/llama.cpp` master**, so it tracks upstream while adding the TurboQuant work. That means almost every standard llama.cpp flag still applies; the additions are layered on top.
- **What it adds over upstream:**
  - **TurboQuant KV cache:** extreme V-cache quantization tiers (`turbo2/turbo3/turbo4`) via a Randomized Walsh–Hadamard Transform + Lloyd-Max scalar quantization, based on the [TurboQuant design discussion](https://github.com/ggml-org/llama.cpp/discussions/20969).
  - **TQ weight-quant formats** (`TQ3_1S`, `TQ4_1S`): optional; we use Unsloth UD GGUFs instead, so these are advanced/optional (see §3).
  - **Cross-backend kernels** for CUDA, Metal, Vulkan, and HIP/ROCm.

---

## 2. TurboQuant KV cache

At long context the KV cache, not the weights, dominates memory. TurboQuant compresses it aggressively while protecting quality.

### Asymmetric K/V: the core idea

K and V are *not* equally sensitive. **K does not tolerate aggressive compression; V does** (V is dequantized per-token during generation). So the policy is always asymmetric: keep K at `f16` or `q8_0`, compress V hard. **Never start by compressing K symmetrically**: it causes perplexity blow-ups and failure modes on some model families.

### V-cache tiers (fork-only types)

```text
Type      ~Bits/val   Compression   Quality        Use when
--------  ---------   -----------   ------------   --------------------------------
turbo4    ~4.25-4.5   ~3.8x         <1% PPL        Safest; first contact w/ a model
turbo3    ~3.25-3.5   ~4.9x         <1.5% PPL      Fork's recommended default
turbo2    ~2.0        highest       <2% PPL        Memory-bound, very long context
```

Higher turbo number = **more** bits = **lighter** compression = better quality. (Exact bit counts vary slightly by version/source; treat as approximate.)

### Recommended progression (from the fork)

```text
Step 1 (safest):   --cache-type-k f16  --cache-type-v turbo4    # K untouched
Step 2:            --cache-type-k q8_0 --cache-type-v turbo4
Step 3 (default):  --cache-type-k q8_0 --cache-type-v turbo3    # "asymmetric turbo" sweet spot
Step 4 (aggro):    --cache-type-k q8_0 --cache-type-v turbo2    # long context; Boundary-V auto-protects
```

### What this repo actually uses

- **CUDA guides (DGX, Dual RTX 6000, Win 4090):** `--cache-type-k q8_0 --cache-type-v turbo4`, deliberately **more conservative** than the fork's `turbo3` default, prioritizing quality/stability.
- **Metal guides (all Macs) and Jetson:** `--cache-type-k q8_0 --cache-type-v q8_0`, no turbo. `turbo*` *is* supported on Metal, but the maintainer uses plain `q8_0` as a tested baseline; try `turbo4`-V if you want more headroom.

### When NOT to use aggressive V compression

- **Decode speed is critical:** turbo V adds ~30–40% decode slowdown on long contexts (per-token V dequant). For max generation speed use `q8_0`.
- **Reasoning / long-range-attention tasks:** attention precision across long sequences degrades first here.
- **Very small models (≲1B)** and some **MoE / quant-sensitive instruction-tuned** variants: more delicate; the paper's near-lossless claims hold for "reasonably sized" models.
- **`turbo*` requires `--flash-attn on`** (quantized KV needs the flash-attention kernels).

### Caveats

- These `turbo*` types are **fork-only**: upstream `--cache-type-v` only accepts `f16/bf16/q8_0/q4_0/q4_1/iq4_nl/q5_0/q5_1`. A vanilla llama.cpp build will reject `turbo4`.
- Apple Silicon: "TurboFlash" is off by default on some chips due to a corruption regression; verify output quality if you enable turbo on Metal.
- Boundary-V (layer-aware protection, auto-enabled for `turbo2`) is experimental.

---

## 3. TQ weight quantization (optional / advanced)

Separate from the KV cache, the fork can quantize *model weights* to `TQ4_1S` (~4.5-bit, fast `dp4a` path on CUDA) or `TQ3_1S` (~3.5-bit) via `llama-quantize model.f16.gguf out.gguf TQ4_1S`. **These guides do not use TQ weights**: we use Unsloth UD-quant GGUFs (see the [model catalog in `local-setup.md`](local-setup.md#model-catalog-hugging-face)). Listed here only so the term isn't a mystery; ignore unless you're rolling your own weight quants.

---

## 4. Build (CMake) flags

From [`local-setup.md`](local-setup.md) and the per-hardware guides. See the [llama.cpp build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

- **`-DGGML_CUDA=ON`**: build the CUDA backend (NVIDIA: DGX, RTX cards, Jetson).
- **`-DGGML_METAL=ON`**: build the Metal backend (Apple Silicon).
- **`-DGGML_METAL_EMBED_LIBRARY=ON`**: embed the Metal shader library into the binary so there's no external `.metallib` to ship/locate. Standard for Apple builds.
- **`-DCMAKE_CUDA_ARCHITECTURES="87"`**: compile for a specific GPU compute capability (Jetson Orin = `87`/sm_87). Omit to let CMake autodetect (the DGX guide does).
- **`-DGGML_CUDA_F16=ON`**: enable FP16 CUDA compute paths (faster on capable GPUs).
- **`-DGGML_CUDA_FA_ALL_QUANTS=ON`**: compile flash-attention CUDA kernels for **all** KV-quant type combinations. Needed when you want arbitrary quantized KV (e.g. turbo/odd combos) with flash attention; **significantly longer compile time**. Used in the Jetson guide.
- **`-DLLAMA_CURL=ON`**: link libcurl so the binary can fetch models remotely (`-hf` etc.). Needs `libcurl4-openssl-dev` on Debian/Ubuntu.

---

## 5. Runtime flag reference

Grouped by purpose. "Repo values" = what the hardware guides typically set. Upstream defaults are noted where useful.

### Model & memory

- **`--model PATH`**: the GGUF to load.
- **`--ctx-size N`** (`-c`): context window in tokens. Repo: 32k–262k by hardware. Bigger context = much larger KV cache; size it to your memory. With `--fit on` it's a *ceiling*, not a guarantee (see below).
- **`--n-gpu-layers 99`** (`-ngl`): offload all layers to GPU. `99` just means "all." On unified-memory Macs everything shares one pool, so this mainly forces GPU/Metal use; on discrete GPUs it controls how much lives in VRAM. Lower it to spill layers to CPU if VRAM is short.
- **`--no-mmap`**: load the whole model into RAM instead of memory-mapping it (upstream mmap is *on* by default). Repo uses it on machines with enough RAM for steadier latency (no page-in stalls). **Drop it (allow mmap) if the model is near your memory limit**: mmap lets the OS page weights instead of failing to allocate.
- **`--mlock`**: pin the model in RAM (no swap/compression). Not used by default here; consider it on Linux if you see swap thrash.
- **`--fit on` + `--fit-target <MiB>`** *(fork / recent)*: auto-fits the model to free device memory by adjusting GPU-layer offload and **reducing the *effective* context below `--ctx-size`** to make it load, leaving `<MiB>` headroom (we use `256`). Treat `--ctx-size` as a request and **check the startup log for the context actually allocated**. It cannot shrink the *weights*, so it won't rescue a quant larger than RAM (why the 16 GB Mac Mini needs an IQ2 quant, not Q4). Used in all Mac guides.

### KV cache & attention

- **`--cache-type-k TYPE` / `--cache-type-v TYPE`**: KV cache precision. Upstream default `f16`; allowed `f32,f16,bf16,q8_0,q4_0,q4_1,iq4_nl,q5_0,q5_1`, **plus `turbo2/3/4` for V in this fork**. See §2.
- **`--flash-attn on`** (`-fa`; values `on|off|auto`, default `auto`): the flash-attention kernel, faster and lower-memory. **Required for quantized KV** (q8_0/turbo). We set `on` explicitly.
- **`--kv-unified`**: one shared KV buffer across all server slots instead of per-slot buffers; **saves VRAM** and lets larger contexts fit. Most useful with `--parallel >1`; with `--parallel 1` the effect is small. CUDA guides keep it; the maintainer's Mac commands omit it (Metal pattern). Note: had a regression around upstream build b5913; verify if you hit odd KV behavior.
- **`--no-context-shift`**: when the context fills, **stop** instead of llama.cpp's default "rotating" behavior (silently discarding the oldest half and continuing). Important for **agentic correctness**: you want a clean stop, not silent context loss mid-task. Trade-off: long runs end at the limit rather than rolling.

### Prompt cache & checkpointing

These reuse computed state across requests so repeated prompts/turns don't reprocess from scratch.

- **`--slot-save-path ./kv-cache`**: directory to save/restore per-slot KV state (upstream default: disabled). Enables prompt-cache persistence.
- **`--cache-ram N`** (`-cram`): max RAM (MiB) for cached state/checkpoints. Upstream default `8192`; `-1` = no limit, `0` = disable. Repo sets 1024–8192 by RAM budget. Checkpoints currently live in **system RAM**.
- **`--ctx-checkpoints N`** (`-ctxcp`): max context checkpoints per slot (upstream default `32`). Repo uses `4–8` to bound RAM.
- **`--checkpoint-every-n-tokens N`** *(used by the maintainer's tested commands)*: checkpoint interval in tokens. Note: current **upstream** spells the spacing control `--checkpoint-min-step`/`-cms` (default `256`); the fork accepts `--checkpoint-every-n-tokens` (the tested DGX/M5/Air commands use it successfully). If your build rejects it, check `--help` for the current name.

> ⚠️ **Checkpointing may not help on Qwen3.x.** Qwen3.5/3.6 use a **hybrid Gated-DeltaNet (recurrent) attention**, and llama.cpp has documented bugs where **context checkpoints are never restored on hybrid/recurrent models, forcing full prompt re-processing on every turn** (see issues [#20225](https://github.com/ggml-org/llama.cpp/issues/20225), [#19794](https://github.com/ggml-org/llama.cpp/issues/19794), [#22384](https://github.com/ggml-org/llama.cpp/issues/22384)). So on Qwen3.6 these flags may add overhead without the prefill savings. Watch your logs for "invalidated context checkpoint" / full reprocessing each turn; the savings are real on non-hybrid models. Whether the turboquant fork carries a fix is build-dependent: verify.

### Batching & throughput

- **`--parallel N`** (`-np`): number of server slots (concurrent sequences). Repo uses `1` (single-user, max per-request context). Raise for serving multiple clients (pairs well with `--kv-unified`).
- **`--cont-batching`**: continuous batching (upstream default: enabled). Not set explicitly; relevant when `--parallel >1`.
- **`--ubatch-size N`** (`-ub`, physical batch; upstream default `512`) and **`--batch-size N`** (`-b`, logical batch; default `2048`): control prefill throughput vs. memory. Repo scales these to memory: 1024 on big-VRAM CUDA, 512 on mid, 128–256 on tight unified memory. **Lower them if you OOM during prompt processing** (the compute buffer scales with ubatch).

### Generation & reasoning

- **`--n-predict N`**: max tokens to generate per request (upstream default `-1` = unlimited). Repo caps at 4096–8192 as a safety bound.
- **`--reasoning-budget N`**: thinking-token budget. `-1` unrestricted, **`0` = end thinking immediately** (no chain-of-thought), `N>0` = budget. Repo uses **`0`** to run Qwen3.6's thinking model in **non-thinking mode** for lower latency in agentic loops. Set `-1` (or a positive N) if you want the model to reason.
- **`--reasoning on`** *(fork form; upstream uses `--reasoning-format none|deepseek|…`)*: enables reasoning-tag handling/parsing. Paired with `--reasoning-budget 0`, the net effect is: reasoning is parsed but no thinking tokens are emitted. Confirm the exact spelling on your build.
- **`--jinja`**: use the model's Jinja chat template (upstream default: enabled). Set explicitly here; needed for correct chat/tool formatting on modern templates.

### Sampling (quality/determinism)

Repo defaults: `--temp 0.65 --top-p 0.90 --min-p 0.0 --repeat-penalty 1.10 --presence-penalty 0.0 --frequency-penalty 0.0 --repeat-last-n 1024` (Gemma guides use `--temp 0.75 --top-p 0.92`).

- **`--temp`**: randomness. Lower (0.6–0.7) = more deterministic/code-friendly; higher = more creative.
- **`--top-p`**: nucleus sampling cutoff.
- **`--min-p`**: min probability floor; `0.0` disables it (relying on top-p).
- **`--repeat-penalty` / `--repeat-last-n`**: penalize repeats over the last N tokens (`1.10` is mild; `1.0` disables).
- **`--presence-penalty` / `--frequency-penalty`**: OpenAI-style penalties; `0.0` = off here.

> These are starting points, not gospel. Match the **model's** recommended sampling when the model card specifies one (Qwen/Gemma publish theirs).

### Server & logging

- **`--host 0.0.0.0 --port 8080`**: bind address/port. `0.0.0.0` exposes the server on your network; use `127.0.0.1` if you only want local access.
- **`--log-verbosity 2`**: verbose logs (useful for confirming the effective context after `--fit`, KV cache sizes, and checkpoint behavior).

---

## See also

- [`local-setup.md`](local-setup.md): clone, build, download, and `models.json` integration.
- Hardware guides: exact, per-machine commands ([README table](README.md#hardware-configurations-included)).
- [TurboQuant design discussion](https://github.com/ggml-org/llama.cpp/discussions/20969) · [llama.cpp server docs](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) · [build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
