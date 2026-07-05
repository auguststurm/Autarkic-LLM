# AMD 7900 XTX (24 GB) - Qwen3.6-35B-A3B MTP

Optimized setup for AMD Radeon RX 7900 XTX with 24 GB VRAM running the **Qwen3.6-35B-A3B MTP** MoE model via Vulkan, using ByteShape's ShapeLearn quantization.

> ✅ **Tested by user on this hardware.** The settings below come from real runs; expect some variance with thermals and background load.

## About This Model

This is **not** the standard Unsloth UD quant — it's ByteShape's **ShapeLearn-quantized MTP (Multi-Token Prediction)** variant with vision support:

- **MTP**: predicts multiple tokens per forward pass, accelerating decode by accepting speculative tokens that match what the base model would have produced. Requires `--spec-type draft-mtp --spec-draft-n-max 2`.
- **ShapeLearn quants**: ByteShape learns the optimal datatype per tensor, maintaining high quality even at very low bitlengths. Uses a hybrid KQ + IQ approach optimized for GPU throughput.

## Recommended Model

- **Model**: `Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf` (ShapeLearn; ~18.6 GB — best quality/size tradeoff for 24 GB)
- **HuggingFace tag**: `byteshape/Qwen3.6-35B-A3B-MTP-GGUF:Qwen3.6-35B-A3B-IQ4_XS-4.19bpw`

> **Build requirement:** MTP support requires a recent build of llama.cpp (the `draft-mtp` value for `--spec-type` was added in b9235 ). If your `./llama-server --help` output does not list `draft-mtp` under `--spec-type`, please update to a newer build from the [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases).

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

# AMD RDNA3: use Vulkan, not CUDA
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/byteshape/Qwen/Qwen3.6-35B-A3B-IQ4_XS-4.19bpw.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 262144 \
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

## MTP Performance Notes

- **MTP throughput is workload dependent.** Code completion, structured output, and repetitive content benefit the most; highly creative or out-of-distribution generation benefits less.
- The IQ4_XS-4.19bpw at ~18.6 GB leaves ~5.4 GB for KV cache and compute buffers on the 24 GB VRAM — stable at 262k context with `q8_0` V-cache compression.
- Expected token generation/sec: ~120-140 t/s on this hardware.
- Vulkan on AMD GPUs is well-supported; MTP speculative decoding should deliver up to 1.2-2x speedups over non-MTP models (less for MoE).


## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-35b-a3b-mtp",
  "name": "Qwen3.6-35B-A3B-MTP IQ4_XS (262k) - 7900 XTX",
  "contextWindow": 262144,
  "maxTokens": 65536
}
```

**Last Updated:** July 2026
