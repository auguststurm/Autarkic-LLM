# Nvidia Jetson Orin Nano Super - Gemma 4 E2B

Optimized setup for the Nvidia Jetson Orin Nano Super (8 GB LPDDR5, Ampere GPU, compute capability 8.7), using **llama-cpp-turboquant**. Command shape follows the [M4 MacBook Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) conventions (host, pinned context, `--fit off`) adapted for edge CUDA + Gemma.

> ⚠️ **Not yet tested on this hardware.** Best-effort starting config. If you run it, please report results via an issue.

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq`. On 8 GB, do not expect Air-class context — 32k is already ambitious; drop if you OOM.

## Recommended model

- **Model:** `gemma-4-E2B-it-Q4_K_S.gguf` (~3 GB)
- **Path:** `~/models/gemma-4-E2B/gemma-4-E2B-it-Q4_K_S.gguf` (common Jetson layout; any path works)

```bash
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_S.gguf \
  --local-dir ~/models/gemma-4-E2B
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

> `-DGGML_CUDA_FA_ALL_QUANTS=ON` lengthens compile time but enables flash-attention kernels for quantized KV combinations.

## Optimized llama-server command (Q4_K_S @ 32k)

```bash
pkill -9 llama-server

./llama-server \
  --model ~/models/gemma-4-E2B/gemma-4-E2B-it-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 \
  --fit off \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 256 \
  --batch-size 256 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 0 --temp 0.75 --top-p 0.92 \
  --n-predict 4096 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 32768` | Starting agent window; drop if 8 GB OOMs under load |
| `--cache-type-k/v q8_0` | Quality default; model is small — try turbo V only if raising context |
| `--n-gpu-layers 99` | Full GPU offload |
| `--fit off` | Keep pinned context agent-visible |
| `--host 127.0.0.1` | Local-only default on an edge device |
| Gemma sampling | `temp 0.75` / `top-p 0.92` — repo Gemma baseline |

### Fallbacks if you OOM

1. Run **MAXN SUPER** and free other processes (`sudo nvpmodel -m 2 && sudo jetson_clocks`).
2. Drop batch to `128` or `64`.
3. Drop `--ctx-size` to `16384` or `8192`.
4. Optional: `--cache-type-v turbo4` if you need more context than quality at the KV.

## Performance notes

- Q4_K_S is the sweet spot for the 8 GB memory limit.
- Run in **MAXN SUPER** power mode; monitor with `jtop`.
- Excellent for edge agentic use and lightweight assistance.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## Pi Coding Agent `models.json`

Save this **entire** file as Pi’s `models.json` (copy-paste as-is — do not assemble a wrapper).

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "gemma-4-e2b",
          "name": "Gemma 4 E2B Q4_K_S (32k) - Jetson Orin Nano Super",
          "contextWindow": 32768,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```


**Last Updated:** July 2026
