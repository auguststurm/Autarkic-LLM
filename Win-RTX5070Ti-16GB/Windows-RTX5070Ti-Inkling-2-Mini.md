# ASUS GeForce RTX 5070 Ti (16 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this card (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

WSL2. ASUS GeForce RTX 5070 Ti, GB203, **16 GB GDDR7**, compute capability **12.0**. `nvidia-smi` inside WSL is the live memory number. One GPU. If this card is not index 0, set `CUDA_VISIBLE_DEVICES` to its index. Port **8080**. Before starting, `ss -ltn | grep ':8080'` should print nothing.

Paths: `~/AIML/models` and `~/GitHub/llama.cpp-inkling`.

| Pin | Value |
| --- | --- |
| **Weights** | `~/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file` (13.63 GB decimal, 12.70 GiB) |
| **Context** | `--ctx-size 32768`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 8192` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | `~/GitHub/llama.cpp-inkling`, ggml-org **master**, `CMAKE_CUDA_ARCHITECTURES=120` |

## Weights

Skip this when that `.gguf_file` is already under `~/AIML/models/Inkling-2-Mini-GGUF`.

Download that `Mainline_Support` name and leave it unchanged. Pass it as a filename. A token after `--include` is read as a filename, and `hf` then ignores `--include`.

```bash
mkdir -p ~/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file \
  --local-dir ~/AIML/models/Inkling-2-Mini-GGUF
```

## Build

```bash
cd ~/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build --config Release -j"$(nproc)" --target llama-server
```

If this directory is already on an older tag:

```bash
cd ~/GitHub/llama.cpp-inkling
git switch master
git pull --ff-only
```

`"120"` is this GPU. CMake rewrites it to `120a`. The configure log should contain `Replacing 120 in CMAKE_CUDA_ARCHITECTURES with 120a`. `-DCMAKE_BUILD_TYPE=Release` is required because Unix Makefiles ignore `--config`. Blackwell needs CUDA 12.8 or newer. `no kernel image` means the arch rewrite did not happen. `unknown model architecture: 'inkling2_loop'` means the tree is too old.

## Server

```bash
cd ~/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 256 \
  --ubatch-size 256 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
```

Checked against llama.cpp master source and this Q3 file’s header. Not a measured load on this 5070 Ti.

| Flag | Why |
| --- | --- |
| `--model` | 12.70 GiB on 16 GB GDDR7. With the KV below, the known payload is about 13.4 GiB, leaving about 2.6 GiB before the compute buffer and whatever the desktop has already allocated |
| `--ctx-size 32768` | 32768 divides by 256, so the context pad does not change it. Eight full-context layers at `q8_0` are 0.53 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB |
| `--parallel 1` | One request, one slot. `n_seq_max` 1 keeps `n_ctx_seq` at 32768 |
| `--n-gpu-layers all` | Every layer on the GPU. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | This card is the one GPU |
| `--batch-size 256` `--ubatch-size 256` | Prefill is the memory peak on 16 GB. The binary defaults are 2048 and 512 |
| `q8_0` KV | 0.53 GiB for the eight full-context layers. The default `f16` cache at this context is 1.00 GiB |
| `--n-predict 8192` | Matches Pi `maxTokens`. The binary default does not stop |
| Sampling | This Q3 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0. The CLI defaults are 0.80, 40, 0.95, 0.05 |

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 32768
n_ctx_seq             = 32768
n_slots = 1, n_ctx_slot = 32768, kv_unified = 'false'
creating non-SWA KV cache, size = 32768 cells
creating     SWA KV cache, size = 1536 cells
n_ctx_seq (32768) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
flash_attn            = auto
nvidia-smi   # inside WSL, after load and after a short decode
```

The capacity line is this pin. The offload log lists one CUDA device. Loopback `--host 127.0.0.1` is reachable from Windows programs on this machine.

## Pi `models.json`

`id` must match `--alias`. Inside WSL the file is `~/.pi/agent/models.json`. Pi on Windows uses `%USERPROFILE%\.pi\agent\models.json` and the same `baseUrl`. Restart Pi.

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
          "name": "Inkling 2 Mini UD-Q3 (32768 q8/q8) - RTX 5070 Ti",
          "contextWindow": 32768,
          "maxTokens": 8192
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

Inside WSL:

```bash
PI_CODING_AGENT_DIR="$PWD/.pi/agent" pi
```

Pi on Windows, from that directory in PowerShell:

```powershell
$env:PI_CODING_AGENT_DIR = "$PWD\.pi\agent"
pi
```

Models, settings, and sessions for that launch stay in `.pi/agent`. `/model` reloads `models.json`.

## Out of memory

Drop `--batch-size` and `--ubatch-size` to `128`. Then drop `--ctx-size` to `16384`. Set Pi `contextWindow` to that number. If `16384` does not load, use `Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS.gguf_file` (11.66 GB decimal, 10.86 GiB) at `--ctx-size 32768`. Download that name as a filename.

**Last Updated:** 2026-11-21 (llama.cpp master, UD-Q3_K_XL @ 32768 q8/q8, one slot, sm_120, untested)
