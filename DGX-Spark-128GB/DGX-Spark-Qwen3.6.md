# DGX Spark Founders Edition (128 GB) - Qwen3.6-27B

> ✅ **Tested** on this hardware. Qwen3.8 port (same knobs, ⚠️ untested): [DGX-Spark-Qwen3.8.md](DGX-Spark-Qwen3.8.md).

CUDA CC **12.1** (GB10) · llama-cpp-turboquant. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested |
| **Weights** | `Qwen3.6-27B-UD-Q6_K_XL.gguf` (~22 GB) |
| **Catalog** | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` K / **turbo4** V |
| **Output** | `--n-predict 8192` |
| **Sampling** | temp **0.65** · top_p **0.90** · repeat **1.10** |
| **Thinking** | `--reasoning off` |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need the engine built first? [local-setup.md](../local-setup.md).

## Download

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

## Build

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121"
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). GB10 = CC 12.1. Confirm turbo: `./llama-server --help | grep -A2 cache-type-v`.

## PRIMARY command

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-27B-UD-Q6_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --load-mode none \
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

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. `contextWindow` = 262144, `maxTokens` = 8192.

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
