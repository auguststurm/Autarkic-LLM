# AMD 7900 XTX (24 GB) - Qwen3.6-35B-A3B MTP

> ✅ **Tested by a user** on this hardware. Re-check `n_ctx` and MTP after rebuilds. **Vulkan, not CUDA.**

24 GB · RDNA3 · llama-cpp-turboquant. **Not** Unsloth UD — ByteShape **ShapeLearn** MTP (~18.6 GB). MTP is on the primary command. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested (community) |
| **Weights** | `Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf` (~18.6 GB) |
| **Catalog** | [byteshape/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` |
| **Spec** | `--spec-type draft-mtp --spec-draft-n-max 2` |
| **Thinking** | `--reasoning off` |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need `draft-mtp` in `--help` (b9235+). Need the engine? [local-setup.md](../local-setup.md).

## Download

```bash
hf download byteshape/Qwen3.6-35B-A3B-MTP-GGUF \
  Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf \
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
  --model ~/Documents/AIML/models/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
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
| `--ctx-size 262144` | User-tested full train window with this ShapeLearn quant |
| `--cache-type-k/v q8_0` | Tested baseline; turbo V optional if you need more headroom |
| MTP flags | Speculative decode for this MTP GGUF |
| Sampling | User-tested MTP sampling |
| `--fit off` + thinking off + `127.0.0.1` | Repo agent/autarky defaults |
| No checkpoint flags | Qwen3.6 hybrid caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

## MTP performance notes

- **MTP throughput is workload dependent.** Code completion, structured output, and repetitive content benefit most; highly creative generation benefits less.
- IQ4_XS-4.19bpw at ~18.6 GB leaves ~5.4 GB for KV and compute on 24 GB — stable at 262k with `q8_0` V-cache per tester report.
- Expected generation: ~120–140 t/s on this hardware (varies with MTP acceptance).
- Vulkan on AMD is well-supported; MTP often ~1.2–2× vs non-MTP (less for MoE).
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
          "id": "qwen3.6-35b-a3b-mtp",
          "name": "Qwen3.6-35B-A3B-MTP IQ4_XS (262k) - 7900 XTX",
          "contextWindow": 262144,
          "maxTokens": 65536
        }
      ]
    }
  }
}
```

> Note: `maxTokens` matches this guide’s high `--n-predict`; for typical agent turns you may prefer a lower `maxTokens` (e.g. 8192).


**Last Updated:** July 2026
