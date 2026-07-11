# AMD 7900 XTX (24 GB) - Qwen3.6-35B-A3B MTP

Optimized setup for AMD Radeon RX 7900 XTX with 24 GB VRAM running the **Qwen3.6-35B-A3B MTP** MoE model via Vulkan, using ByteShape’s ShapeLearn quantization and **llama-cpp-turboquant**.

> ✅ **Tested by a user on this hardware.** Settings below come from real runs; expect variance with thermals and background load. Flag modernization (host, thinking off, `--fit off`) matches current repo conventions — re-check `n_ctx` and MTP behavior after rebuilds.

## About this model

This is **not** the standard Unsloth UD quant — it’s ByteShape’s **ShapeLearn-quantized MTP (Multi-Token Prediction)** variant with vision support:

- **MTP**: predicts multiple tokens per forward pass, accelerating decode by accepting speculative tokens. Requires `--spec-type draft-mtp --spec-draft-n-max 2`.
- **ShapeLearn quants**: ByteShape learns the optimal datatype per tensor, maintaining high quality at low bitlengths. Hybrid KQ + IQ approach optimized for GPU throughput.

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq`.

**Thinking / empty replies** (Qwen3.6): `--reasoning off` and `--chat-template-kwargs '{"enable_thinking":false}'`.

## Recommended model

- **Model:** `Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf` (ShapeLearn; ~18.6 GB — best quality/size tradeoff for 24 GB)
- **Hugging Face:** `byteshape/Qwen3.6-35B-A3B-MTP-GGUF`
- **Path:** `~/Documents/AIML/models/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf`

```bash
hf download byteshape/Qwen3.6-35B-A3B-MTP-GGUF \
  Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf \
  --local-dir ~/Documents/AIML/models
```

> **Build requirement:** MTP needs a recent build (`draft-mtp` under `--spec-type`, added in llama.cpp b9235+). If `./llama-server --help` does not list `draft-mtp`, update the TurboQuant fork / rebuild.

## Build instructions (TurboQuant fork, Vulkan)

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

# TheTom TurboQuant fork — not ggml-org/llama.cpp
# https://github.com/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull

rm -rf build
mkdir build && cd build

# AMD RDNA3: use Vulkan, not CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

## Optimized llama-server command (MTP IQ4_XS @ 262k)

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

## Pi Coding Agent `models.json` snippet

```json
{
  "id": "qwen3.6-35b-a3b-mtp",
  "name": "Qwen3.6-35B-A3B-MTP IQ4_XS (262k) - 7900 XTX",
  "contextWindow": 262144,
  "maxTokens": 65536
}
```

> Note: `maxTokens` matches this guide’s high `--n-predict`; for typical agent turns you may prefer a lower `maxTokens` (e.g. 8192).

Nest in the full `providers` wrapper from [`local-setup.md`](../local-setup.md#6-pi-coding-agent--hermes-integration). Point Pi at `http://127.0.0.1:8080/v1`.

**Last Updated:** July 2026
