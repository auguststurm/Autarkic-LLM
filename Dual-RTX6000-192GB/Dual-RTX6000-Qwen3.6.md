# Dual RTX 6000 Pro Max-Q (192 GB) - Qwen3.6-27B

Optimized setup for dual NVIDIA RTX 6000 Pro Max-Q GPUs (192 GB total VRAM) on Ubuntu, using **llama-cpp-turboquant**. Command shape follows the [M4 MacBook Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) conventions (host, thinking off, pinned context, no Qwen checkpoint flags), with CUDA multi-GPU options.

> ⚠️ **Not yet tested on this hardware.** Best-effort starting config from model size + llama.cpp options. If you run it, please report results via an issue.

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq`.

**Thinking / empty replies** (Qwen3.6): `--reasoning off` and `--chat-template-kwargs '{"enable_thinking":false}'`.

## Recommended model (primary — single GPU)

- **Model:** `Qwen3.6-27B-UD-Q8_K_XL.gguf` (~35.3 GB — best quality with massive KV headroom on 192 GB)
- **Path:** `~/Documents/AIML/models/Qwen3.6-27B-UD-Q8_K_XL.gguf`

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

Confirm turbo types:

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> Omit `-DCMAKE_CUDA_ARCHITECTURES="120"` to autodetect; set explicitly when cross-compiling (Blackwell = CC 12.0).

## Optimized llama-server command (Q8_K_XL @ 262k, single GPU)

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-27B-UD-Q8_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
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
  --threads 32 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 262144` | Full train window; one 96 GB card has room for Q8 + KV |
| `--cache-type-v turbo4` | CUDA quality-leaning TurboQuant V |
| `--n-gpu-layers 99` | Full offload on the primary GPU |
| `--fit off` + thinking off + `127.0.0.1` | Same agent/autarky defaults as the Air reference |
| No checkpoint flags | Qwen3.6 hybrid caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

## Performance notes

- Primary config uses **one GPU** by default — simple and fast for a ~35 GB model. Large VRAM remains for KV at 262k.
- For heavier models or extreme multi-card use, see the multi-GPU section below.
- Ideal for heavy agentic workloads and long-context development.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## Pi Coding Agent `models.json`

Save this **entire** file as Pi’s `models.json` (copy-paste as-is — do not assemble a wrapper).

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
          "name": "Qwen3.6-27B Q8_K_XL (262k) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```


## Alternate: multi-GPU layer split

> ⚠️ **Untested on this hardware.** Use when the model does not fit on one card or you want to spread load across both 96 GB GPUs.

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
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
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
  --threads 32 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

- **Model:** `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (~38.5 GB MoE)
- Multi-GPU adds PCIe sync overhead; benchmark against the single-GPU primary before committing.

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
          "name": "Qwen3.6-35B-A3B Q8_K_XL (262k) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

**Last Updated:** July 2026
