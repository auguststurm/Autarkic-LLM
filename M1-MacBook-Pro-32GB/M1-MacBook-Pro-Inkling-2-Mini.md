# M1 Pro MacBook Pro (32 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this machine (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn. Metal can load and then run out of memory on the first prompt.

**Metal.** 32 GB unified memory, shared with macOS. `uname -m` prints `arm64`. Build on this Mac. Port **8080**. Before starting, `lsof -nP -iTCP:8080 -sTCP:LISTEN` should print nothing.

Paths: `~/Documents/AIML/models` and `~/Documents/GitHub/llama.cpp-inkling`.

| Pin | Value |
| --- | --- |
| **Weights** | `~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (17.63 GB decimal, 16.42 GiB) |
| **Projector** | `mmproj-F16.gguf` (639 MB decimal, 0.60 GiB) |
| **Context** | `--ctx-size 131072`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 16384` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | ggml-org **master**, `-DGGML_METAL=ON` |

## Weights

Skip this when that `.gguf_file` is already under `~/Documents/AIML/models/Inkling-2-Mini-GGUF`.

Download the `Mainline_Support` name and leave it unchanged. Pass both paths as filenames. A token after `--include` is read as a filename, and `hf` then ignores `--include`.

```bash
mkdir -p ~/Documents/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models/Inkling-2-Mini-GGUF
```

## Build

```bash
cd ~/Documents/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)" --target llama-server
```

This source turns Metal on by default for macOS. The flag names that backend. The Metal library is embedded by default. If this directory is already on an older tag:

```bash
cd ~/Documents/GitHub/llama.cpp-inkling
git switch master
git pull --ff-only
```

`-DCMAKE_BUILD_TYPE=Release` is required because Unix Makefiles ignore `--config`. `unknown model architecture: 'inkling2_loop'` means the tree is too old.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin
./llama-server --list-devices
```

The list should show the Apple GPU.

## Server

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --mmproj ~/Documents/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

Checked against llama.cpp master source and this Q4 file’s header. Not a measured load on this Mac.

| Flag | Why |
| --- | --- |
| `--ctx-size 131072` | 131072 divides by 256, so the context pad does not change it. Weights are 16.42 GiB, the projector is 0.60 GiB, and eight full-context layers at `q8_0` are 2.125 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. Known payload is about 19.3 GiB of the 32 GB pool, leaving about 12.7 GiB before macOS and the Metal scratch buffer. That scratch is not measured |
| `--parallel 1` | One request, one slot. `n_seq_max` 1 keeps `n_ctx_seq` at 131072 |
| `--n-gpu-layers all` | Every layer on the GPU. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | This Mac has one GPU |
| `q8_0` KV | 2.125 GiB for the eight full-context layers. The default `f16` cache at this context is 4.00 GiB |
| `--n-predict 16384` | Matches Pi `maxTokens`. The binary default does not stop |
| Sampling | This Q4 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0. The CLI defaults are 0.80, 40, 0.95, 0.05 |

Text-only: drop `--mmproj` and skip the projector download. That removes 0.60 GiB from the payload.

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 131072
n_ctx_seq             = 131072
n_slots = 1, n_ctx_slot = 131072, kv_unified = 'false'
creating non-SWA KV cache, size = 131072 cells
creating     SWA KV cache, size = 1536 cells
n_ctx_seq (131072) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
flash_attn            = auto
```

The capacity line is this pin. The offload log names the Metal device. Activity Monitor → Memory, after load and after a short decode.

## Pi `models.json`

`id` must match `--alias`. `~/.pi/agent/models.json`. Restart Pi.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "inkling-2-mini",
          "name": "Inkling 2 Mini UD-Q4 (131072 q8/q8) - M1 Pro",
          "contextWindow": 131072,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

## Directory instance

`PI_CODING_AGENT_DIR` replaces `~/.pi/agent` for that process. From the project directory:

```bash
mkdir -p .pi/agent
```

Write the JSON above to `.pi/agent/models.json`. Write `.pi/agent/settings.json`:

```json
{
  "defaultProvider": "llama-cpp",
  "defaultModel": "inkling-2-mini"
}
```

```bash
PI_CODING_AGENT_DIR="$PWD/.pi/agent" pi
```

Models, settings, and sessions for that launch stay in `.pi/agent`. `/model` reloads `models.json`.

## Out of memory

Drop `--ctx-size` to `65536`, then `32768`, and set Pi `contextWindow` to that number. If the failure is the first prompt, set `--batch-size 512` and `--ubatch-size 512`, then both to `256`.

A log line `n_gpu_layers already set by user to -2` is the same step: lower `--ctx-size` and keep `--n-gpu-layers all`.

After 131072 decodes and Activity Monitor still shows several GiB free, set `--ctx-size 262144`. Eight full-context layers are then 4.25 GiB, and the known payload is about 21.4 GiB. Match Pi `contextWindow`.

**Last Updated:** 2026-11-21 (llama.cpp master, Metal, UD-Q4_K_XL @ 131072 q8/q8, one slot, untested)
