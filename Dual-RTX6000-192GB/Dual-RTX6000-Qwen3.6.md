# Dual RTX 6000 Pro Max-Q (192 GB) - Qwen3.6-27B

Optimized setup for dual NVIDIA RTX 6000 Pro Max-Q GPUs (192 GB total VRAM) on Ubuntu, using **llama-cpp-turboquant**. Pi / tool conventions follow the field-validated [Windows RTX 4090 Qwen guide](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) and [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware); compute and memory knobs stay CUDA multi-GPU / high-VRAM.

> ✅ **Tested** on this hardware (**Saturday, August 8, 2026**). Primary config below is **field-validated working** with **Pi Coding Agent** — full **262k** pin, **Q8_K_XL** weights, **q8/q8** KV on a single 96 GB card, stable tools + strong agent quality. Expect variance with thermals and background load; re-check `n_ctx_seq` after rebuilds. Cross-hardware Pi notes: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq`.

**Thinking / empty replies** (Qwen3.6): prefer **`--reasoning off`** (+ `--reasoning-budget 0`) for Pi `message.content`. Do **not** pass deprecated `--chat-template-kwargs '{"enable_thinking":false}'` — current llama-server builds warn and prefer `--reasoning on|off`.

Shared Pi + Qwen3.6-27B lessons (two token limits, no DRY, tool sampling, K/V, hybrid cache): [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

## Recommended model (primary — single GPU)

- **Model:** `Qwen3.6-27B-UD-Q8_K_XL.gguf` (~35.3 GB — best quality with massive KV headroom on 192 GB)
- **Tested path:** `~/Documents/AIML/models/Qwen3.6-27B-UD-Q8_K_XL.gguf`

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
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

Confirm turbo types (needed only if you try the optional turbo-V path):

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> Omit `-DCMAKE_CUDA_ARCHITECTURES="120"` to autodetect; set explicitly when cross-compiling (Blackwell = CC 12.0).

## Optimized llama-server command (PRIMARY — Q8_K_XL @ 262k, single GPU)

**Tested baseline (2026-08-08)** with Pi Coding Agent. Use this command first.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-27B-UD-Q8_K_XL.gguf \
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

| Flag / value | Why (field-tested on this box) |
| --- | --- |
| `--ctx-size 262144` | Full train window; one 96 GB card has room for Q8 + KV — **confirmed** with Pi |
| `--cache-type-k/v q8_0` | High-precision KV while VRAM allows (agent routing/tools); turbo V only if you need more capacity later |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn cache restore issues ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)); correct multi-turn over cache speed |
| `--n-gpu-layers 99` + `--main-gpu 0` | Full offload on the primary GPU on a dual-card box — primary stays single-GPU for simplicity |
| `--reasoning off` (+ budget 0) | Thinking off for Pi tools / `message.content` — **not** `enable_thinking` kwargs |
| Tool sampling | `temp 0.6` / `top_p 0.95` / `top_k 20` / `presence 0` / `repeat 1.0` — strong Pi tool/coding results on this hardware; **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 16384` | Report-length agent output (match Pi `maxTokens`) |
| `--fit off` + `127.0.0.1` | Pinned agent-visible context; loopback default |
| No checkpoint flags | Qwen3.6 hybrid caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

**Confirm**

```text
log: n_ctx_seq (262144)
nvidia-smi   # note MiB after load and after a short decode
```

### Optional: TurboQuant V (capacity / decode tradeoff)

Primary already has enormous headroom. If you later raise load (heavier quant, multi-slot, or decode cost experiments) and need to free KV:

```bash
# Same command as above, only:
#   --cache-type-k q8_0 --cache-type-v turbo4
# Keep K at q8_0. Smoke-test real ls/read on a new Pi session first.
```

## Performance notes

- **Validated 2026-08-08:** primary single-GPU **Q8_K_XL @ 262k · q8/q8 · 16k out** worked very well with Pi Coding Agent (tools, long agent sessions, full train window).
- Primary uses **one GPU** by default — simple and fast for a ~35 GB model. The other 96 GB card stays free (or idle); large VRAM on GPU 0 covers weights + full-window KV at **q8/q8** without TurboQuant V.
- TurboQuant **fork** ≠ must use turbo **types**. This box is the roomiest CUDA profile in the repo: keep `q8_0`/`q8_0`; turbo V is a capacity lever for later experiments, not the quality default.
- For heavier models or deliberate multi-card spread, see the multi-GPU section below (still untested on this hardware).
- Ideal for heavy agentic workloads and long-context development.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Cross-hardware Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

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
          "id": "qwen3.6-27b",
          "name": "Qwen3.6-27B Q8_K_XL (262k q8/q8) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```


## Alternate: multi-GPU layer split

> ⚠️ **Not part of the August 8 primary test.** Primary single-GPU Q8_K_XL is the validated path. Use multi-GPU when the model does not fit on one card or you deliberately want to spread load across both 96 GB GPUs.

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
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

- **Model:** `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (~38.5 GB MoE)
- Multi-GPU adds PCIe sync overhead; benchmark against the single-GPU primary before committing.
- Same agent flags as primary (no `chat-template-kwargs`, tool sampling, `--cache-ram 0`).

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.6-35B-A3B",
          "name": "Qwen3.6-35B-A3B Q8_K_XL (262k q8/q8) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

**Last Updated:** August 14, 2026 (primary marked tested from 2026-08-08 Pi Coding Agent run)
