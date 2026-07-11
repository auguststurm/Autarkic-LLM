# Windows RTX 4090 (24 GB) - Qwen3.6-27B

Optimized setup for a Windows machine with NVIDIA RTX 4090 24 GB VRAM (**WSL2 recommended**), using **llama-cpp-turboquant**. Command shape follows the [M4 MacBook Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md) conventions (host, thinking off, pinned context, no Qwen checkpoint flags), with CUDA layers and turbo4 V.

> ⚠️ **Not yet tested on this hardware.** Best-effort starting config from model size + llama.cpp options. If you run it, please report results via an issue.

## Pi Coding Agent: read this first

**Always pin `--ctx-size` and set `--fit off`.** Match Pi’s `contextWindow` to real `n_ctx_seq`.

**Thinking / empty replies** (Qwen3.6): `--reasoning off` and `--chat-template-kwargs '{"enable_thinking":false}'`.

## Memory reality (24 GB VRAM)

- **Weights:** `UD-Q5_K_XL` is ~20 GB — quality-first with modest headroom for KV at 32k.
- **Longer context:** drop to `UD-Q4_K_XL` (~17.6 GB) or `UD-IQ4_NL` (~16.1 GB) and/or compress V harder (`turbo3`/`turbo2`). Prefer lowering `--ctx-size` over bare `--fit on` for agent use.
- Confirm **decode**, not only load, after any context raise.

## Recommended model

- **Model:** `Qwen3.6-27B-UD-Q5_K_XL.gguf` (~20 GB)
- **Path:** `~/Documents/AIML/models/Qwen3.6-27B-UD-Q5_K_XL.gguf`

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

## Build instructions (WSL2 Ubuntu, TurboQuant fork)

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

# TheTom TurboQuant fork — not ggml-org/llama.cpp
# https://github.com/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build . --config Release -j$(nproc)

cd bin
mkdir -p ./kv-cache
```

Confirm turbo types:

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

## Optimized llama-server command (Q5_K_XL @ 32k)

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 \
  --fit off \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 512 \
  --batch-size 512 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 16 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 32768` | Conservative start on 24 GB with ~20 GB weights |
| `--cache-type-v turbo4` | Frees V-cache vs plain `q8_0` without going full turbo2 |
| `--n-gpu-layers 99` | Full GPU offload |
| `--fit off` | Keep agent-visible context; drop ctx size if OOM |
| Thinking off + `127.0.0.1` | Agent/autarky defaults |
| No checkpoint flags | Qwen3.6 hybrid caveat — see [checkpointing](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

### Fallbacks if you OOM

1. Drop batch to `256` or `128`.
2. Drop `--ctx-size` (e.g. `16384`).
3. Switch to `UD-Q4_K_XL` or `UD-IQ4_NL` for 64k–128k experiments; confirm decode each step.
4. Optionally use `--cache-type-v turbo2` for more context at some decode cost.

## Performance notes

- WSL2 recommended for best CUDA compatibility and performance on Windows.
- 24 GB is the hard limit — quant + context are a trade, same idea as the Air’s unified-memory matrix.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## Pi Coding Agent `models.json` snippet

```json
{
  "id": "qwen3.6-27b",
  "name": "Qwen3.6-27B Q5_K_XL (32k) - RTX 4090",
  "contextWindow": 32768,
  "maxTokens": 8192
}
```

Nest in the full `providers` wrapper from [`local-setup.md`](../local-setup.md#6-pi-coding-agent--hermes-integration). Point Pi at `http://127.0.0.1:8080/v1`.

**Last Updated:** July 2026
