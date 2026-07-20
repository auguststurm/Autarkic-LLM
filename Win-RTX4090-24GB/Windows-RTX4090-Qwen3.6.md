# Windows RTX 4090 (24 GB) - Qwen3.6-27B

Optimized setup for a Windows machine with an **NVIDIA GeForce RTX 4090 24 GB** (Ada Lovelace, **sm_89**), running under **WSL2**, using **llama-cpp-turboquant**.

Agent conventions match the [M4 MacBook Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) (loopback host, thinking off, pinned context, no Qwen checkpoint flags). **Compute and memory knobs are CUDA/Ada-specific** — full layer offload, FP16 CUDA paths, flash-attention for quantized KV, and prefill batches sized for ~1 TB/s GDDR6X — not the tight Metal batch/turbo2 pressure settings from the Air.

> ⚠️ **Not yet tested on this hardware.** Best-effort config from Ada 24 GB VRAM + llama.cpp/TurboQuant options and multi-turn Pi agent feedback. If you run it, please report results via an issue.

## Why this box is different from the Air / DGX templates

| | M4 Air (24 GB unified) | **RTX 4090 (24 GB discrete)** | DGX Spark (128 GB) |
| --- | --- | --- | --- |
| Memory | Unified; OS + weights + KV share one pool | **Dedicated GDDR6X**; host RAM separate | Huge unified-class pool |
| Bandwidth | Memory-bound; small batches | **~1008 GB/s** — large prefill batches pay off | High; large batches |
| Backend | Metal | **CUDA sm_89** | CUDA sm_121 |
| Offload | Implicit | **`--n-gpu-layers 99`** all layers in VRAM | Same pattern |
| Turbo V | Often `turbo2` to *survive* | Optional headroom for **64k+**; primary uses **q8/q8** for agent fidelity | `turbo4` |

On the 4090, weights live in **VRAM**, not unified RAM. Host RAM only needs enough to load/copy (`--no-mmap`). That is the opposite pressure model from the Air.

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq` (startup log or `GET /v1/models`).

**Thinking / empty replies** (Qwen3.6): `--reasoning off` and `--chat-template-kwargs '{"enable_thinking":false}'`.

**Pi + tools + pi-workflows fill context fast.** Primary pin is **65536**. Field run: multi-agent Korea research workflow hit **`request (32830) exceeds … (32768)`** while Pi status showed **`99.9%/33k`** — both server and client were still on **32k**. Parallel agents reported **41k–86k tok** each over the run; synthesis alone **~34k tok**. Match server + Pi or the overflow returns. For workflow packages, Tavily, and saving reports under `reports/`, see [Pi Coding Agent graphs](../pi-coding-agent-graphs.md).

**Apply checklist (both required):**

1. Restart `llama-server` with the command in this guide (confirm log: `n_ctx_seq` = **65536**).
2. Copy `models.json` below to `~/.pi/agent/models.json` and **restart Pi** (status bar should show **64k**, not 33k).

## Memory reality (RTX 4090 24 GB)

- **Chip:** Ada Lovelace, compute capability **8.9** → build with `-DCMAKE_CUDA_ARCHITECTURES="89"`.
- **Weights (primary):** `UD-Q4_K_XL` ~**17.6 GB**. Field note: Q4 @ 32k sat around **~19–20+ GB** VRAM mid-session — several GB free; 64k is the intended use of that headroom.
- **Why not Q5 as primary:** `UD-Q5_K_XL` ~20 GB leaves little for 64k + agent peaks.
- **KV:** primary **`q8_0` / `q8_0`** for agent fidelity. If 64k OOMs, drop batch first, then try **turbo4 V** (see fallbacks) — not as the first quality default.
- Confirm **decode**, not only load, after any context, KV type, or batch raise.

| Quant | ~Weights | Role on 4090 24 GB |
| --- | --- | --- |
| **`UD-Q4_K_XL`** | ~17.6 GB | **Primary** — q8 KV + **64k** for Pi workflows |
| `UD-Q5_K_XL` | ~20 GB | Optional quality bump; lower ctx/batch if OOM |
| `UD-IQ4_NL` | ~16.1 GB | Extreme context experiments with turbo V |

## Recommended model

- **Model:** `Qwen3.6-27B-UD-Q4_K_XL.gguf` (~17.6 GB)
- **Path:** `~/AIML/models/Qwen3.6-27B-UD-Q4_K_XL.gguf`

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --local-dir ~/AIML/models
```

## Build instructions (WSL2 Ubuntu, Ada-tuned TurboQuant)

Native **sm_89** + **FP16 CUDA** + **flash-attn kernels for all KV quant types**.

```bash
cd ~/GitHub/llama-cpp-turboquant

# TheTom TurboQuant fork — not ggml-org/llama.cpp
# https://github.com/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON

cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

| CMake flag | Why on RTX 4090 |
| --- | --- |
| `-DGGML_CUDA=ON` | Discrete NVIDIA path (not Metal/Vulkan) |
| `-DCMAKE_CUDA_ARCHITECTURES="89"` | Ada Lovelace only — matches 4090 SMs |
| `-DGGML_CUDA_F16=ON` | FP16 tensor-core friendly compute on Ada |
| `-DGGML_CUDA_FA_ALL_QUANTS=ON` | FA kernels for turbo/q8 KV combos; longer compile |

Confirm turbo types:

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> Rebuild after `git pull`. If you move the binary to a different NVIDIA arch, rebuild for that GPU (or omit `CMAKE_CUDA_ARCHITECTURES` to autodetect).

## Optimized llama-server command (Q4_K_XL @ 64k — Pi / pi-workflows)

Run from `~/GitHub/llama-cpp-turboquant/build/bin`.

**Goal:** multi-turn **tool/agent** + **pi-workflows** (parallel research agents, Tavily, graphs, synthesis). Research (llama.cpp [#20837](https://github.com/ggml-org/llama.cpp/issues/20837), [#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) covers tool loops / path degeneration / DeltaNet cache drift. **64k** is primary after a real multi-agent run overflowed **32k** and spent only ~**20 GB** VRAM.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/AIML/models/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 65536 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 256 \
  --batch-size 256 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 0 \
  --n-predict 4096 \
  --kv-unified \
  --log-verbosity 1
```

> **Do not enable DRY** for Pi/tool sessions. DRY ON → path/name corruptions; DRY OFF → clean paths.

### Why these values (agent / workflow-focused)

| Flag / value | Why |
| --- | --- |
| `UD-Q4_K_XL` | ~17.6 GB weights; field VRAM ~20 GB @ 32k leaves room for **64k** KV |
| **`--ctx-size 65536`** | Pi workflows: single requests already **>32k** (`32830 > 32768`); parallel agents reported **41k–86k tok** over a run |
| `--ubatch-size` / `--batch-size` **256** | Safer peak VRAM at 64k than 512 (raise to 512 if free VRAM after load is comfortable) |
| `--cache-type-k/v q8_0` | Multi-turn / tool fidelity on hybrid Qwen |
| **`--cache-ram 0`** | Avoid DeltaNet multi-turn cache restore corruption ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)); slower re-prefill |
| **No DRY** | Path/tool-arg degeneration ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| **Agent sampling** | `temp 0.6`, `top_p 0.95`, `top_k 20`, **`presence_penalty 0.0`**, `repeat_penalty 1.0` |
| **`--n-predict 4096`** | Workflow synthesis / report paste needs more than 2048; still caps runaway tool-retry soup |
| Thinking off | Pi tools / `message.content` |
| Ada offload / FA | 4090 defaults |

Confirm **`n_ctx` / `n_ctx_seq (65536)`** — if the log still says **32768**, you did not restart with this command:

```bash
nvidia-smi
# expect ~20–23 GB used with Q4 + q8 @ 64k; if OOM, use fallbacks below
```

### Field notes: pi-workflows multi-agent run

Example outcome (Korea mobile game research workflow):

| Observation | Meaning |
| --- | --- |
| VRAM high **19s → just over 20 GB** | Headroom for 64k; not VRAM-starved at 32k |
| Parallel agents **41k–86k tok** each; synthesis **~34k tok**; run **~364k tok** total | Context is the bottleneck, not decode speed |
| Market data agent **✗** | One worker failed; others + synthesis still completed |
| Pi bar **`99.9%/33k`** | Client still on **32k** `contextWindow` — update `models.json` + restart Pi |
| `request (32830) exceeds … (32768)` | Server still on **32k** — restart `llama-server` with **65536** |
| `maximum output token limit` after “read the report” | `maxTokens` / `--n-predict` too low for the answer (use **4096**) |
| Workflow JSON under `~/.pi/workflows/.../runs/*.json` | Can open/summarize offline if the chat turn fails |

### Context overflow (`request exceeds the available context size`)

```text
Error: 400: … request (32830 tokens) exceeds the available context size (32768 tokens) …
n_prompt_tokens: 32830, n_ctx: 32768
```

| Meaning | Fix |
| --- | --- |
| Prompt larger than server pin | `--ctx-size` ≥ request (primary **65536**) |
| Pi still shows **33k** | `models.json` `contextWindow` + **restart Pi** |
| Mismatch | `contextWindow` ≤ real `n_ctx_seq`; never higher than the running server |
| Still overflows 64k | New session; trim history; or turbo4 V + keep 64k if OOM forced a lower pin |

Hard reject — not sampling garbage.

### Known failure mode: Pi shell path soup + max tokens

```text
ls /hoomeeeee/augsusstttsssuu./i/worrfffswowowowoswww/
…
Error: Model stopped because it reached the maximum output token limit
```

1. Tool loop + **DRY** / high **presence_penalty** → path soup.  
2. Retries burn `n-predict` / Pi `maxTokens`.  
3. DeltaNet + prompt cache can also drift ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)).

**Recovery:** command above (no DRY, `presence 0`, `--cache-ram 0`), **new Pi session**, `maxTokens` **4096**.

### If paths still corrupt

1. Pi must not override sampling.  
2. Diagnostic: `--cache-type-k f16 --cache-type-v f16` and lower ctx if OOM.  
3. Chat-only optional: `--temp 0.7 --top-p 0.80 --top-k 20 --presence-penalty 1.5` (still no DRY).

### Fallbacks if you OOM at 64k

1. Keep 64k, use **`--cache-type-v turbo4`** (and keep batch 256).  
2. Drop batch to `128`.  
3. Drop to **`--ctx-size 49152`** (set Pi `contextWindow` to **49152**).  
4. Do **not** enable bare `--fit on` for Pi.

### Optional: Q5 quality bump

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --local-dir ~/AIML/models
```

Same agent flags; start **`--ctx-size 32768`** or **49152** and batch **256** until VRAM is healthy, then raise.

## WSL2 / host tips

1. **GPU visible in WSL:** `nvidia-smi` lists the 4090 inside the distro.
2. **Power / clocks (Windows):** high-performance plan; avoid GPU power-limit overlays while benchmarking.
3. **Don’t share the card heavily:** other CUDA apps steal VRAM.
4. **Host RAM:** `--no-mmap` wants ~18 GB+ free system RAM for Q4 load.
5. **Monitor with `nvidia-smi`** (some GUI tools crash under WSL process accounting).
6. Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## Performance notes

- High **decode** tok/s on a 4090 for 27B Q4; **prefill** at batch 256 is conservative for 64k peaks (raise batch if VRAM allows).
- **`--cache-ram 0`** makes multi-turn prefills slower; intentional for hybrid fidelity.
- TurboQuant V is a **VRAM fallback** at 64k, not required when ~20 GB used at 32k.
- Pattern references: [DGX Spark](../DGX-Spark-128GB/DGX-Spark-Qwen3.6.md) (CUDA shape), [M4 Air](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) (agent conventions only).

## Pi Coding Agent `models.json`

Save this **entire** file to `~/.pi/agent/models.json` (copy-paste as-is — do not assemble a wrapper). Create parent dirs if needed: `mkdir -p ~/.pi/agent`. **Restart Pi** after writing so the status bar shows **64k**, not **33k**.

`maxTokens` ≤ `--n-predict` (**4096**). `contextWindow` = **`--ctx-size`** (primary: **65536**).

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.6-27b",
          "name": "Qwen3.6-27B Q4_K_XL agent (64k) - RTX 4090",
          "contextWindow": 65536,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

**Last Updated:** July 2026
