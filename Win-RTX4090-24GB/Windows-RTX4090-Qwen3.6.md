# Windows RTX 4090 (24 GB) - Qwen3.6-27B

Optimized setup for Windows machine with NVIDIA RTX 4090 24 GB VRAM (WSL2 recommended).

> ⚠️ **Not yet tested on this hardware.** This is a best-effort starting config based on the model size and llama.cpp options: performance figures are estimates. If you run it, please report your results via an issue.

## Recommended Model

- **Model**: `Qwen3.6-27B-UD-Q5_K_XL.gguf`
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q5_K_XL.gguf`

## Build Instructions (WSL2 Ubuntu)

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
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 131072 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 512 \
  --batch-size 512 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 16 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --cache-ram 4096 \
  --slot-save-path ./kv-cache \
  --checkpoint-min-step 16384 \
  --ctx-checkpoints 6 \
  --kv-unified \
  --log-verbosity 2
```

> **Checkpoint flags on Qwen3.6:** `--slot-save-path`, `--checkpoint-min-step`, and `--ctx-checkpoints` are included here, but may be **no-ops on Qwen3.6** — its hybrid Gated-DeltaNet attention hits a known llama.cpp bug where context checkpoints aren't restored, forcing full prompt reprocessing each turn. Watch the log; if you see repeated full reprocessing, drop these three flags. Details: [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing).

## Performance Notes

- ⚠️ Memory is tight: `Qwen3.6-27B-UD-Q5_K_XL` is ~20 GB and the RTX 4090 has 24 GB VRAM, so weights + KV cache at 131072 context may not all fit. Add `--fit on --fit-target 256` (auto-offload/auto-reduce context) or lower `--ctx-size` if you hit OOM. `--cache-type-v turbo4` already compresses the V cache to help.
- Reduced context size compared to higher-VRAM systems to maintain stability.
- WSL2 recommended for best CUDA compatibility and performance.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-27b",
  "name": "Qwen3.6-27B Q5_K_XL (128k) - RTX 4090",
  "contextWindow": 131072,
  "maxTokens": 32768
}
```

**Last Updated:** June 2026
