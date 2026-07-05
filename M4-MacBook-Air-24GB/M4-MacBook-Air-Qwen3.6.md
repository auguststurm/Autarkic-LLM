# M4 MacBook Air (24 GB) - Qwen3.6-35B-A3B

Optimized setup for MacBook Air M4 with 24 GB unified memory.

> ✅ **Tested by the maintainer on this hardware.** The settings below come from real runs; expect some variance with thermals and background load.

## Pi Coding Agent: read this first

Pi needs a **large `contextWindow`** (system prompt + tools consume several thousand tokens before your task). If the server only allocates **~4096 tokens** (`n_ctx_seq (4096)` in the log), Pi will stop almost immediately with *"maximum output token limit"*.

**Do not use bare `--fit on` without a pinned `--ctx-size` on Q4_K_XL** — fit shrinks context to ~4096 on 24 GB, which is too small for Pi.

**Recommended for Pi:** the **IQ4_NL quant** (~18 GB) below — it frees enough memory for **32k+ context** on 24 GB. The Q4_K_XL path is kept as a higher-quality option but is tight on 24 GB with current fork builds.

## Recommended Model (Pi / agentic use)

- **Model**: `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` (~18 GB — best balance for Pi on 24 GB)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --local-dir ~/Documents/AIML/Models/unsloth/Qwen/LLM
```

## Alternative model (higher quality, tighter on 24 GB)

- **Model**: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (~22 GB)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- Use the **Q4 command block** in the server section below. If you Metal-OOM at startup or on first token, switch to IQ4_NL.

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

> **Fork version:** build from `feature/turboquant-kv-cache` at commit **`91b4f8cc8` or later** (Metal `rnorm` fix, [PR #200](https://github.com/TheTom/llama-cpp-turboquant/pull/200)). Rebuild after `git pull` — the Jun 2026 turbo4 merge (`0c8fcfe`) breaks Apple Silicon startup without this patch.

## Optimized llama-server Command (IQ4_NL — recommended for Pi)

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --flash-attn on \
  --fit off \
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
  --log-verbosity 2
```

> **`--fit off` + pinned `--ctx-size`:** fit is disabled so context stays at 32k for Pi. `turbo4` V-cache saves KV RAM on Metal (requires `--flash-attn on`). Confirm the startup log shows `n_ctx_seq (32768)`.

## Optimized llama-server Command (Q4_K_XL — higher quality, tighter)

Same flags, swap `--model` to the Q4 path. Try `--ctx-size 32768` first; only raise to `65536` if the log shows it loaded and inference does not Metal-OOM.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --flash-attn on \
  --fit off \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 64 \
  --batch-size 64 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 0 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --log-verbosity 2
```

> **Checkpoint flags omitted** — they do not help on Qwen3.6 hybrid attention and only add log noise. See [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing).

## Performance Notes

- MoE architecture (low active parameters) keeps decode fast despite the 35B total size.
- `n_ctx_seq (32768) < n_ctx_train (262144)` is normal — you are not using the model's full 262k train length.
- Close memory-hungry apps before launching. If you see `kIOGPUCommandBufferCallbackErrorOutOfMemory`, lower `--ctx-size` to `16384` or switch to IQ4_NL.
- After a fork rebuild, always check the log for the actual `n_ctx_seq` and match Pi's `contextWindow` to that value.

## Pi Coding Agent models.json Snippet

Use the block that matches the model and `--ctx-size` you launched. `maxTokens` must not exceed `--n-predict` (8192).

**IQ4_NL @ 32k (recommended):**

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6-35B-A3B IQ4_NL (32k) - M4 Air",
  "contextWindow": 32768,
  "maxTokens": 8192
}
```

**Q4_K_XL @ 32k:**

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6-35B-A3B Q4_K_XL (32k) - M4 Air",
  "contextWindow": 32768,
  "maxTokens": 8192
}
```

Nest either entry in the full `providers` wrapper from [`local-setup.md`](../local-setup.md#6-pi-coding-agent--hermes-integration). Point Pi at `http://127.0.0.1:8080/v1`.

**Last Updated:** July 2026