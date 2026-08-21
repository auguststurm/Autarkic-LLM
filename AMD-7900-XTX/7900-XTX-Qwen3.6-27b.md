# AMD 7900 XTX (24 GB) - Qwen3.6-27B MTP

> ✅ **Tested by a user** on this hardware. Re-check `n_ctx` and MTP after rebuilds. **Vulkan, not CUDA.**

24 GB · RDNA3 · llama-cpp-turboquant. MTP is **on the primary command** here (~1.2–2× decode). Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested (community) |
| **Weights** | `Qwen3.6-27B-MTP-IQ4_NL.gguf` |
| **Catalog** | [unsloth/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF) |
| **Context** | `--ctx-size 128000` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` |
| **Spec** | `--spec-type draft-mtp --spec-draft-n-max 2` |
| **Thinking** | `--reasoning off` |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need `draft-mtp` in `./llama-server --help` (llama.cpp b9235+). Need the engine? [local-setup.md](../local-setup.md).

## Download

```bash
hf download unsloth/Qwen3.6-27B-MTP-GGUF \
  Qwen3.6-27B-MTP-IQ4_NL.gguf \
  --local-dir ~/Documents/AIML/models
```

## Build (Vulkan)

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). Do not use `-DGGML_CUDA=ON` on this GPU.

## PRIMARY command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-27B-MTP-IQ4_NL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 128000 \
  --fit off \
  --n-gpu-layers 99 \
  --no-mmproj \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.0 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 128 \
  --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.0 \
  --n-predict 65536 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 128000` | User-tested window on 24 GB with IQ4_NL + `q8_0` KV |
| `--cache-type-k/v q8_0` | Tested quality baseline; try `turbo4` V if you need more headroom |
| MTP flags | Speculative decode speedup for this MTP GGUF |
| Sampling | User-tested MTP sampling (`temp 0.6`, `top-k 20`, …) |
| `--fit off` + thinking off + `127.0.0.1` | Repo agent/autarky defaults |
| No checkpoint flags | Qwen3.6 hybrid caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

## Performance notes

- IQ4_NL for 27B is ~18–19 GB, leaving ~5 GB for KV and compute on 24 GB — stable but tight at 128k with `q8_0` V-cache.
- Expected generation: ~60–65 t/s on this hardware (varies with MTP acceptance).
- Vulkan on RDNA3 is well-supported; MTP speculative decoding often ~1.2–2× vs non-MTP (more for dense models).
- Predicting **2–4** tokens ahead is a good quality/throughput tradeoff; start with `--spec-draft-n-max 2`.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## Pi Coding Agent `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.6-27b-mtp",
          "name": "Qwen3.6-27B-MTP IQ4_NL (128k) - 7900 XTX",
          "contextWindow": 128000,
          "maxTokens": 65536
        }
      ]
    }
  }
}
```

> Note: `maxTokens` is high to match this guide’s `--n-predict`; for typical agent turns you may prefer a lower `maxTokens` (e.g. 8192) while leaving server `--n-predict` high.


**Last Updated:** July 2026
