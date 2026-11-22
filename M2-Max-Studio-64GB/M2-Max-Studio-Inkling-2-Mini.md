# M2 Max Mac Studio (64 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this machine (written 2026-11-22). Confirm load, then first decode, before a Pi tool turn.

**Metal.** Apple M2 Max in the 2023 Mac Studio. 12-core CPU, 30-core or 38-core GPU, 400 GB/s unified memory. This guide is the **64 GB** configuration. `uname -m` prints `arm64`. Build on this Mac. Port **8080**. Before starting, `lsof -nP -iTCP:8080 -sTCP:LISTEN` should print nothing.

Paths: `~/Documents/AIML/models` and `~/Documents/GitHub/llama.cpp-inkling`.

| Pin | Value |
| --- | --- |
| **Weights** | `~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` (22.87 GB decimal, 21.30 GiB) |
| **Projector** | `mmproj-F16.gguf` (639 MB decimal, 0.60 GiB) |
| **Context** | `--ctx-size 1048576`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 16384` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | ggml-org **master**, `-DGGML_METAL=ON` |

## Weights

Three ways to run this model. The three server commands use them in this order.

1. **Censored.** `Inkling-2-Mini-UD-Q8_K_XL.gguf_file` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The Unsloth file is Q8, 21.30 GiB. That is the censored weights, and it is the quality file on this machine.
2. **Censored plus the LoRA.** The same Unsloth file, plus `Inkling-2-Mini-Uncensored-LoRA.gguf`. The AtomicChat uncensored release is a Q4 file, 15.39 GiB. To stay on Q8 and still remove the refusals, put their LoRA on this Unsloth file. The LoRA is the edit by itself. The baked Q4 file is the same edit already merged, at the lower quant.
3. **Uncensored file.** `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` from [AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF). Leave `--lora` off. The edit is already in the file.

```bash
mkdir -p ~/Documents/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models/Inkling-2-Mini-GGUF
```

```bash
mkdir -p ~/Documents/AIML/models
hf download AtomicChat/Inkling-2-Mini-Uncensored-GGUF \
  Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  Inkling-2-Mini-Uncensored-LoRA.gguf \
  mmproj-Inkling-2-Mini-Uncensored-F16.gguf \
  --local-dir ~/Documents/AIML/models/Inkling-2-Mini-Uncensored-GGUF
```

`mmproj-Inkling-2-Mini-Uncensored-F16.gguf` has the same vision weights as Unsloth `mmproj-F16.gguf`. Either projector works. A command that already has `--mmproj` can keep the Unsloth path.

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
  --model ~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  --mmproj ~/Documents/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 1048576 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

### Original with the uncensored LoRA

Same command. `--lora` applies [AtomicChat's adapter](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) to the weights above. Scale is 1.0.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  --lora ~/Documents/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-LoRA.gguf \
  --mmproj ~/Documents/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 1048576 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

### Uncensored model

Same command. `--model` is the baked file. Leave `--lora` off.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/Documents/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --mmproj ~/Documents/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 1048576 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

Checked against llama.cpp master source and the Q8 file. Not a measured load on this Mac.

| Flag | Why |
| --- | --- |
| `--ctx-size 1048576` | The GGUF `context_length`. 1048576 divides by 256, so the context pad does not change it. Weights are 21.30 GiB, the projector is 0.60 GiB, and eight full-context layers at `q8_0` are 17.00 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. Known payload is about 39.0 GiB of the 64 GB pool, leaving about 25 GiB before macOS and the Metal scratch buffer. That scratch is not measured |
| `--parallel 1` | One request, one slot. `n_seq_max` 1 keeps `n_ctx_seq` at 1048576 |
| `--n-gpu-layers all` | Every layer on the GPU. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | This Mac has one GPU |
| `q8_0` KV | 17.00 GiB for the eight full-context layers. The default `f16` cache at this context is 32 GiB |
| `--n-predict 16384` | Matches Pi `maxTokens`. The binary default does not stop |
| Sampling | This Q8 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0. The CLI defaults are 0.80, 40, 0.95, 0.05 |

Text-only: drop `--mmproj` and skip the projector download.

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 1048576
n_ctx_seq             = 1048576
n_slots = 1, n_ctx_slot = 1048576, kv_unified = 'false'
creating non-SWA KV cache, size = 1048576 cells
creating     SWA KV cache, size = 1536 cells
flash_attn            = auto
```

`n_ctx_seq` matches the GGUF `context_length`. The offload log names the Metal device. Activity Monitor → Memory, after load and after a short decode.

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
          "name": "Inkling 2 Mini UD-Q8 (1048576 q8/q8) - M2 Max",
          "contextWindow": 1048576,
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

**Last Updated:** 2026-11-22 (llama.cpp master, Metal, UD-Q8_K_XL @ 1048576 q8/q8, one slot, untested)
