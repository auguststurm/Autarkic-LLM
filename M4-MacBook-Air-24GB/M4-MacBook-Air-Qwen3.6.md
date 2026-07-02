# M4 MacBook Air (24 GB) - Qwen3.6-35B-A3B

Optimized setup for MacBook Air M4 with 24 GB unified memory.

> ✅ **Tested by the maintainer on this hardware.** The settings below come from real runs; expect some variance with thermals and background load.

## Recommended Model

- **Model**: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (MoE - low active parameters)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

> **How a ~22 GB model runs on 24 GB:** the weights (~22.4 GB) plus KV cache would normally exceed 24 GB once macOS overhead is counted. The `--fit on --fit-target 256` flags (below) let llama-server auto-fit the model to free memory, leaving ~256 MiB headroom and **reducing the *effective* context below the requested `--ctx-size 65536` as needed**. So 65536 is a ceiling, not a guarantee; check the startup log for the context actually allocated. Close other apps for the best result.

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

# Apple Silicon: first apply the Metal rnorm patch from ../local-setup.md (step 2), or the shader fails to compile when llama-server starts
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)

cd ../bin
mkdir -p ./kv-cache
```

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 65536 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --fit on \
  --fit-target 256 \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 128 \
  --batch-size 128 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 0 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --cache-ram 1024 \
  --slot-save-path ./kv-cache \
  --checkpoint-min-step 16384 \
  --ctx-checkpoints 4 \
  --log-verbosity 2
```

> **Checkpoint flags on Qwen3.6:** `--slot-save-path`, `--checkpoint-min-step`, and `--ctx-checkpoints` are kept here (harmless in the tested runs), but may be **no-ops on Qwen3.6** — its hybrid Gated-DeltaNet attention hits a known llama.cpp bug where context checkpoints aren't restored, forcing full prompt reprocessing each turn. Watch the log; if you see repeated full reprocessing, drop these three flags. Details: [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing).

## Performance Notes

- MoE architecture (low active parameters) keeps decode fast despite the 35B total size.
- `--fit on --fit-target 256` is what makes this config viable on 24 GB: it trades requested context for a guaranteed load. Watch the startup log to see the effective context.
- `--ubatch-size`/`--batch-size` are kept at 128 to limit the compute-buffer footprint on tight memory.
- Good balance for lighter agentic tasks and daily development use; close memory-hungry apps before launching.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6-35B-A3B Q4_K_XL (64k) - M4 Air",
  "contextWindow": 65536,
  "maxTokens": 16384
}
```

**Last Updated:** June 2026
