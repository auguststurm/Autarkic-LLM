# M1 Max Mac Studio (64 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this machine (written 2026-11-27). Confirm load, then first decode, before a Pi tool turn.

**Metal.** Apple M1 Max in the 2022 Mac Studio. 10-core CPU, 24-core or 32-core GPU, 400 GB/s unified memory. This guide is the **64 GB** configuration. `uname -m` prints `arm64`. Build on this Mac. Port **8080**. Before starting, `lsof -nP -iTCP:8080 -sTCP:LISTEN` should print nothing.

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

1. **Censored.** `Inkling-2-Mini-UD-Q8_K_XL.gguf_file` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The Unsloth file is Q8, 21.30 GiB. Its header `context_length` is 1048576, and that is the context in the server commands.
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

On Apple this source defaults `GGML_METAL` to ON, and `GGML_METAL_EMBED_LIBRARY` follows it. The flag names that backend. If this directory already exists, update it and rebuild:

```bash
cd ~/Documents/GitHub/llama.cpp-inkling
git switch master
git pull --ff-only
```

Then run the cmake and build lines again. `-DCMAKE_BUILD_TYPE=Release` is required because Unix Makefiles ignore `--config`. `unknown model architecture: 'inkling2_loop'` means the tree is too old. The Q8 header sets `tokenizer.ggml.pre` to `inkling2`. Master includes `ignore_merges` for that pre-type from commit `d2bceda3d` (2026-11-23, #32799). A tree without that line still loads the file.

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

Same command. `--lora` applies [AtomicChat's adapter](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) to the weights above. The flag's default scale is 1.0, and this file's `adapter.lora_alpha` is 1.0.

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

Checked against llama.cpp master `8e23278c7` (2026-11-27) and the Q8 and AD-Q4_K headers. Not a measured load on this Mac.

| Flag | Why |
| --- | --- |
| `--ctx-size 1048576` | The Q8 header `inkling2_loop.context_length`. 1048576 divides by 256, so the context pad does not change it. Weights are 21.30 GiB, the projector is 0.60 GiB, and eight full-attention slots at `q8_0` are 17.00 GiB. Those slots are physical layers 5 and 23 once, and layers 11 and 17 on each of 3 passes. Each token is 8 KV heads × 128. Forty sliding-window slots are 1536 cells (window 1024 plus the default ubatch 512), 127.5 MiB. The Q8 tensor payload is 39.02 GiB of the 64 GiB pool, leaving about 25 GiB before macOS, the Metal scratch buffer, and any prompt-cache save. Scratch is not measured. The conv state is f32 across the 48 visits, about 3.7 MiB, and is outside that tensor total. The baked file is 15.39 GiB, so that command has a tensor payload of 33.11 GiB at this same context |
| `--parallel 1` | One slot. `llama-server` defaults to `-1`, and this source turns that into 4 slots with `kv_unified` true. `n_seq_max` 1 keeps `n_ctx_seq` at 1048576 and leaves `kv_unified` false |
| `--port 8080` | The server default is 9931. Pi `baseUrl` uses 8080 |
| `--n-gpu-layers all` | Every visit on the GPU. `block_count` is 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48. `all` is stored as `-2`. Fit does not change a context the user set. If the free-memory margin (default 1 GiB) is short, the log warns `n_gpu_layers already set by user to -2` and this pin stays |
| `--split-mode none` | This Mac has one GPU. The default is `layer` |
| `q8_0` KV | 17.00 GiB for the eight full-attention slots. The default `f16` cache for those slots is 32.00 GiB, and the Q8 tensor payload with that cache is 54.02 GiB before macOS |
| no `--load-mode none` | Default mmap. Metal maps the file with `newBufferWithBytesNoCopy`. `none` copies the 21.30 GiB weights into a second buffer in this same pool |
| `--n-predict 16384` | Matches Pi `maxTokens`. The default `-1` does not stop |
| Sampling | The Q8 header stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0. The baked AD-Q4_K header does not store those keys, so the same flags keep that command on this sampler. CLI defaults are 0.80, 40, 0.95, 0.05 |
| Prompt cache | Default `--cache-ram` is 8192 MiB. That cap is not reserved at load. Saved slots come out of this unified pool |

Text-only: drop `--mmproj` and skip the projector download. That removes 0.60 GiB from the tensor payload.

## Confirm

```text
n_seq_max             = 1
n_ctx                 = 1048576
n_ctx_seq             = 1048576
n_ubatch              = 512
flash_attn            = auto
kv_unified            = false
initializing, n_slots = 1, n_ctx_slot = 1048576, kv_unified = 'false'
creating non-SWA KV cache, size = 1048576 cells
creating     SWA KV cache, size = 1536 cells
```

`n_ctx_seq` matches the header `context_length`, so the log does not print the shorter-than-training line. `n_ubatch` 512 is what makes the sliding-window cache 1536 cells. The offload log names the Metal device. `n_gpu_layers already set by user to -2` is the fit warning from the row above, and the context in this block stays 1048576. Activity Monitor → Memory, after load and after a short decode. The tensor total above is not that reading.

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
          "name": "Inkling 2 Mini UD-Q8 (1048576 q8/q8) - M1 Max",
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

**Last Updated:** 2026-11-27 (llama.cpp master `8e23278c7`, Metal, UD-Q8_K_XL @ 1048576 q8/q8, one slot, untested)
