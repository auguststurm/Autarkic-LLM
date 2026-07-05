# Dual RTX 6000 Pro Max-Q (192 GB) - Qwen3.6-27B

Optimized setup for dual NVIDIA RTX 6000 Pro Max-Q GPUs (192 GB total VRAM) on Ubuntu.

> ⚠️ **Not yet tested on this hardware.** This is a best-effort starting config based on the model size and llama.cpp options: performance figures are estimates. If you run it, please report your results via an issue.

## Recommended Model

- **Model**: `Qwen3.6-27B-UD-Q6_K_XL.gguf`
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q6_K_XL.gguf`

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q6_K_XL.gguf \
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

- Extremely high throughput due to dual 96 GB GPUs.
- Full 262k context with excellent stability.
- Ideal for heavy agentic workloads and long-context Godot development.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-27b",
  "name": "Qwen3.6-27B Q6_K_XL (262k) - Dual RTX 6000",
  "contextWindow": 262144,
  "maxTokens": 32768
}
```

**Last Updated:** July 2026
