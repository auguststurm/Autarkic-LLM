# M5 MacBook Pro (48 GB) - Qwen3.6-27B

Optimized setup for Apple MacBook Pro with M5 Pro/Max chip and 48 GB unified memory.

> ✅ **Tested by the maintainer on this hardware.** The settings below come from real runs; expect some variance with thermals and background load.

## Recommended Model

- **Model**: `Qwen3.6-27B-UD-Q5_K_XL.gguf`
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q5_K_XL.gguf`

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

# Apple Silicon: first apply the Metal rnorm patch from ../local-setup.md (step 2), or the shader fails to compile when llama-server starts
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)

cd bin
mkdir -p ./kv-cache
```

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 196608 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
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
  --threads 0 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --cache-ram 4096 \
  --slot-save-path ./kv-cache \
  --checkpoint-min-step 16384 \
  --ctx-checkpoints 8 \
  --log-verbosity 2
```

> **Checkpoint flags on Qwen3.6:** `--slot-save-path`, `--checkpoint-min-step`, and `--ctx-checkpoints` are kept here (harmless in the tested runs), but may be **no-ops on Qwen3.6** — its hybrid Gated-DeltaNet attention hits a known llama.cpp bug where context checkpoints aren't restored, forcing full prompt reprocessing each turn. Watch the log; if you see repeated full reprocessing, drop these three flags. Details: [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing).

## Performance Notes

- Excellent balance of quality and speed on 48 GB unified memory.
- Strong agentic performance with Hermes and Godot MCP workflows.
- Metal backend provides good efficiency on Apple Silicon.
- **No `--fit` here (intentional):** 48 GB has ample room for the ~20 GB model at 196608 context, so the context is pinned explicitly. `--fit` is omitted because the current fork aborts it when `--n-gpu-layers`/`--ctx-size` are set — this guide sets both. If you ever hit an OOM (e.g. running a larger quant), lower `--ctx-size` rather than adding `--fit`.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-27b",
  "name": "Qwen3.6-27B Q5_K_XL (196k) - M5 Pro",
  "contextWindow": 196608,
  "maxTokens": 32768
}
```

**Last Updated:** July 2026

---

Ready for review. Let me know if adjustments are needed before we continue to the next hardware.
