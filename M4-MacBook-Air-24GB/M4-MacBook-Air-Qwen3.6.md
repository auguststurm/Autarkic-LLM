# M4 MacBook Air (24 GB) - Qwen3.6-35B-A3B

Optimized setup for MacBook Air M4 with 24 GB unified memory.

> ✅ **Tested by the maintainer on this hardware.** The settings below come from real runs; expect some variance with thermals and background load.

## Recommended Model

- **Model**: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (MoE - low active parameters)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`

> **How a ~22 GB model runs on 24 GB (and why there's no `--ctx-size`/`--n-gpu-layers`):** the weights (~22.4 GB) plus KV cache exceed 24 GB once macOS overhead is counted, so this config leans on `--fit on --fit-target 256` to auto-fit — it adjusts layer offload and **chooses the effective context itself**, leaving ~256 MiB headroom. We deliberately **omit `--ctx-size` and `--n-gpu-layers`**: on the current fork, `--fit` _aborts_ if `--n-gpu-layers` is set and _won't shrink_ a context you pin, so pinning either re-breaks the load. On this machine fit lands around **4096 tokens** (measured) — fine for lighter agentic use but small. For a larger usable context, drop to a smaller quant (e.g. a ~16–18 GB Q3/IQ4 variant) to free room for KV cache. Check the startup log for the context fit actually allocated. Close other apps for the best result.

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
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
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
- `--fit on --fit-target 256` is what makes this config viable on 24 GB: with `--ctx-size` and `--n-gpu-layers` omitted, it auto-sizes context and layer offload to guarantee a load (≈4096 tokens here, measured). Watch the startup log for the `fit` line reporting the effective context.
- `--ubatch-size`/`--batch-size` are kept at 128 to limit the compute-buffer footprint on tight memory.
- Good balance for lighter agentic tasks and daily development use; close memory-hungry apps before launching.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6-35B-A3B Q4_K_XL (4k) - M4 Air",
  "contextWindow": 4096,
  "maxTokens": 2048
}
```

**Last Updated:** July 2026
