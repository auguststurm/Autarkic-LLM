# DGX Spark Founders Edition (128 GB) - Qwen3.6-27B

Optimized setup for the NVIDIA DGX Spark Founders Edition (GB10 GPU, 128 GB system RAM), using **llama-cpp-turboquant**. Command shape and Pi integration follow the same conventions as the [M4 MacBook Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) (host, thinking off, pinned context, no Qwen checkpoint flags), with CUDA-specific layers and TurboQuant V.

> ✅ **Tested by the maintainer on this hardware.** Settings below come from real runs; expect variance with thermals and background load. Re-check `n_ctx` after rebuilds.

## Pi Coding Agent: read this first

Pi needs a large `contextWindow`. Match it to the server’s real `n_ctx_seq`.

**Always pin `--ctx-size` and set `--fit off`.** Default `--fit on` can crush context and break Pi.

**Thinking / empty replies** (Qwen3.6):

- `--reasoning off`
- `--chat-template-kwargs '{"enable_thinking":false}'`

## Recommended model

- **Model:** `Qwen3.6-27B-UD-Q6_K_XL.gguf` (~22 GB)
- **Path:** `~/Documents/AIML/models/Qwen3.6-27B-UD-Q6_K_XL.gguf`

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q6_K_XL.gguf \
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

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121"
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

Confirm the binary accepts turbo types:

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> Omit `-DCMAKE_CUDA_ARCHITECTURES="121"` to autodetect on the build machine; set it explicitly when cross-compiling or sharing binaries (GB10 = CC 12.1).

## Optimized llama-server command (Q6_K_XL @ 262k)

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-27B-UD-Q6_K_XL.gguf \
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
  --threads 28 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 262144` | Full train window tested stable on this box with TurboQuant V |
| `--cache-type-k q8_0 --cache-type-v turbo4` | CUDA quality-leaning turbo (not the more aggressive turbo2 used on memory-bound Metal) |
| `--n-gpu-layers 99` | Full GPU offload on discrete/CUDA-class devices |
| `--fit off` | Keep the pinned context agent-visible |
| Thinking off | Agent-friendly non-thinking Qwen3.6 |
| `--host 127.0.0.1` | Local-only default (use `0.0.0.0` only if you intend LAN exposure — no auth) |
| No checkpoint flags | Qwen3.6 hybrid attention often won’t restore checkpoints usefully — see [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

Confirm **`n_ctx` / `n_ctx_seq (262144)`** in the log or `GET /v1/models`.

## Performance notes

- Expected tokens/sec: ~45–65 t/s (prefill), 90–120+ t/s (decode) on this hardware.
- Full 262k context is stable with TurboQuant KV cache.
- Excellent for long agentic tasks with Hermes / Pi Coding Agent.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## Pi Coding Agent `models.json`

Save this **entire** file as Pi’s `models.json` (copy-paste as-is — do not assemble a wrapper).

`maxTokens` ≤ `--n-predict` (8192). `contextWindow` = `--ctx-size`.

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
          "name": "Qwen3.6-27B Q6_K_XL (262k) - DGX Spark",
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```


## Alternate: higher quality (untested)

Primary config above is the **tested baseline**. If you want maximum quality and have headroom to spare:

- **Model:** `Qwen3.6-27B-UD-Q8_K_XL.gguf` (~35.3 GB)
- **Path:** `~/Documents/AIML/models/Qwen3.6-27B-UD-Q8_K_XL.gguf`
- Use the same `llama-server` command as above, swapping only `--model`. ~90 GB remains for KV at 262k context.

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

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
          "name": "Qwen3.6-27B Q8_K_XL (262k) - DGX Spark",
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

**Last Updated:** July 2026
