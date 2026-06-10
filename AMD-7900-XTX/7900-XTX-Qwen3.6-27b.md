# AMD 7900 XTX (24 GB) - Qwen3.6-27B

Optimized setup for AMD Radeon RX 7900 XTX with 24 GB VRAM running the **Qwen3.6-27B MTP** model via Vulkan, using Unsloths's quantization.

> ✅ **Tested by user on this hardware.** The settings below come from real runs; expect some variance with thermals and background load.

## About This Model

- **MTP**: predicts multiple tokens per forward pass, accelerating decode by accepting speculative tokens that match what the base model would have produced. Requires `--spec-type draft-mtp --spec-draft-n-max 2`.

## Recommended Model

- **Model**: `Qwen3.6-27B-MTP-IQ4_NL.gguf` (MTP variant — multi-token prediction for ~1.2–2x faster decode)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/Qwen3.6-27B-MTP-IQ4_NL.gguf`

> **Build requirement:** MTP support requires a recent build of llama.cpp (the `draft-mtp` value for `--spec-type` was added in b9235 ). If your `./llama-server --help` output does not list `draft-mtp` under `--spec-type`, please update to a newer build from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).


## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/Qwen3.6-27B-MTP-IQ4_NL.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 128000 \
  --n-gpu-layers 99 \
  --no-mmproj \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --batch-size 2048 \
  --ubatch-size 1024 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.0 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 128 \
  --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.0 \
  --n-predict 65536 \
  --cache-ram 4096 \
  --kv-unified \
  --log-verbosity 2
```

## Performance Notes

- IQ4_NL quant for 27B is ~18–19 GB, leaving ~5 GB for KV cache and compute buffers on the 24 GB VRAM — stable but tight at 128k context with `q8_0` V-cache compression.
- Expected token generation/sec: ~60-65 t/s on this hardware.
- Vulkan backend on RDNA3 AMD GPUs is well-supported in llama.cpp; MTP speculative decoding should deliver up to 1.2-2x speedups over non-MTP models (more for dense models).
- Predicting **2 to 4 tokens ahead** offers the best quality/throughput tradeoff for most workloads. The `--spec-draft-n-max 2` setting is a good starting point — feel free to change this value for your specific use case.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-27b-mtp",
  "name": "Qwen3.6-27B-MTP IQ4_NL (128k) - 7900 XTX",
  "contextWindow": 128000,
  "maxTokens": 65536
}
```

**Last Updated:** June 2026
