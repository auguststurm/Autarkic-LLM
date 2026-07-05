# Nvidia Jetson Orin Nano Super - Gemma 4 E2B

Optimized setup for the Nvidia Jetson Orin Nano Super (8 GB LPDDR5, Ampere GPU, compute capability 8.7).

> ⚠️ **Not yet tested on this hardware.** This is a best-effort starting config based on the model size and llama.cpp options: performance figures are estimates. If you run it, please report your results via an issue.

## Recommended Model

- **Model**: `gemma-4-E2B-it-Q4_K_S.gguf`
- **Path**: `~/models/gemma-4-E2B/gemma-4-E2B-it-Q4_K_S.gguf`

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_CUDA_F16=ON \
  -DLLAMA_CURL=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON

cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/models/gemma-4-E2B/gemma-4-E2B-it-Q4_K_S.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 256 \
  --batch-size 256 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 0 --temp 0.75 --top-p 0.92 \
  --n-predict 4096 \
  --cache-ram 1024 \
  --slot-save-path ./kv-cache \
  --checkpoint-min-step 8192 \
  --ctx-checkpoints 4 \
  --kv-unified \
  --log-verbosity 2
```

## Performance Notes

- Q4_K_S is the sweet spot for the 8 GB memory limit.
- Run in **MAXN SUPER** power mode (`sudo nvpmodel -m 2 && sudo jetson_clocks`).
- Monitor with `jtop`.
- Excellent for edge agentic use and lightweight Godot assistance.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "gemma-4-e2b",
  "name": "Gemma 4 E2B Q4_K_S - Jetson Orin Nano Super",
  "contextWindow": 32768,
  "maxTokens": 8192
}
```

**Last Updated:** June 2026
