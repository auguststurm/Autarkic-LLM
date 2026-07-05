# Dual RTX 6000 Pro Max-Q (192 GB) - Qwen3.6-27B

Optimized setup for dual NVIDIA RTX 6000 Pro Max-Q GPUs (192 GB total VRAM) on Ubuntu.

> ⚠️ **Not yet tested on this hardware.** This is a best-effort starting config based on the model size and llama.cpp options: performance figures are estimates. If you run it, please report your results via an issue.

## Recommended Model (primary — single GPU)

- **Model**: `Qwen3.6-27B-UD-Q8_K_XL.gguf` (~35.3 GB — best quality with massive KV headroom on 192 GB)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q8_K_XL.gguf`

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

> Omit `-DCMAKE_CUDA_ARCHITECTURES="120"` to autodetect on the build machine; set it explicitly when cross-compiling or sharing binaries (Blackwell = CC 12.0).

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q8_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 262144 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 32 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --cache-ram 8192 \
  --slot-save-path ./kv-cache \
  --checkpoint-min-step 16384 \
  --ctx-checkpoints 8 \
  --kv-unified \
  --log-verbosity 2
```

> **Checkpoint flags on Qwen3.6:** `--slot-save-path`, `--checkpoint-min-step`, and `--ctx-checkpoints` are kept here (harmless in the tested runs), but may be **no-ops on Qwen3.6** — its hybrid Gated-DeltaNet attention hits a known llama.cpp bug where context checkpoints aren't restored, forcing full prompt reprocessing each turn. Watch the log; if you see repeated full reprocessing, drop these three flags. Details: [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing).

## Performance Notes

- Primary config uses **one GPU** by default — simple and fast for a 35 GB model. ~157 GB VRAM remains for KV cache at 262k context.
- For heavier models or extreme context, see the multi-GPU section below.
- Ideal for heavy agentic workloads and long-context Godot development.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-27b",
  "name": "Qwen3.6-27B Q8_K_XL (262k) - Dual RTX 6000",
  "contextWindow": 262144,
  "maxTokens": 8192
}
```

## Alternate: multi-GPU layer split

> ⚠️ **Untested on this hardware.** Use when the model does not fit on one card or you want to spread KV across both 96 GB GPUs.

Split layers evenly across both GPUs:

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 262144 \
  --n-gpu-layers 99 \
  --split-mode layer \
  --tensor-split 96,96 \
  --main-gpu 0 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 32 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --cache-ram 8192 \
  --kv-unified \
  --log-verbosity 2
```

- **Model**: `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` (~38.5 GB MoE)
- Multi-GPU adds PCIe sync overhead; benchmark against the single-GPU primary before committing.

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6-35B-A3B Q8_K_XL (262k) - Dual RTX 6000",
  "contextWindow": 262144,
  "maxTokens": 8192
}
```

**Last Updated:** July 2026