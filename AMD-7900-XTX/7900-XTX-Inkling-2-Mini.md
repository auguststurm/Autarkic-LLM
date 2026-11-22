# AMD 7900 XTX (24 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this card (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

Radeon RX 7900 XTX, **24 GB**, RDNA3 (Navi 31). **Vulkan, not CUDA, and not HIP.** CUDA will not see this GPU. HIP is the ternary path in this folder. This file is a normal GGUF quant, and the servers already tested here load those through Vulkan.

Port **8080**. The other guides in this folder also use 8080. Before starting, `ss -ltn | grep ':8080'` should print nothing.

Paths: `~/Documents/AIML/models` and `~/Documents/GitHub/llama.cpp-inkling`.

| Pin | Value |
| --- | --- |
| **Weights** | `~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (17.63 GB decimal, 16.42 GiB) |
| **Projector** | not in the command |
| **Context** | `--ctx-size 128000`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 8192` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | ggml-org **master**, `-DGGML_VULKAN=ON` |

## Weights

Three ways to run this model. The three server commands use them in this order.

1. **Censored.** `Inkling-2-Mini-UD-Q4_K_XL.gguf_file` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The Unsloth file is Q4, 16.42 GiB. That is the censored weights for this machine.
2. **Censored plus the LoRA.** The same Unsloth file, plus `Inkling-2-Mini-Uncensored-LoRA.gguf`. The AtomicChat uncensored release is also a Q4 file, 15.39 GiB. The LoRA puts that edit on the Unsloth Q4 file. The baked file is the edit already merged, in AtomicChat's own Q4.
3. **Uncensored file.** `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` from [AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF). Leave `--lora` off. The edit is already in the file.

```bash
mkdir -p ~/Documents/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
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

`glslc` must be on `PATH`. CMake stops in `find_package(Vulkan)` without it. Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` into its own directory. Do not point this command at `llama-cpp-turboquant` or `llama.cpp-prism`.

```bash
cd ~/Documents/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON
cmake --build build --config Release -j"$(nproc)" --target llama-server
```

Do not pass `-DGGML_CUDA=ON`, `-DGGML_HIP=ON`, or `-DAMDGPU_TARGETS`. If this directory is already on an older tag:

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

The 7900 XTX line is the device to use. If it is not the first device, pass `--device` with that name on the server command.

## Server

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 128000 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
```

### Original with the uncensored LoRA

Same command. `--lora` applies [AtomicChat's adapter](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) to the weights above. Scale is 1.0.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --lora ~/Documents/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-LoRA.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 128000 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
```

### Uncensored model

Same command. `--model` is the baked file. Leave `--lora` off.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/Documents/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 128000 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
```


Checked against llama.cpp master source and the Q4 file header. Not a measured load on this 7900 XTX.

| Flag | Why |
| --- | --- |
| Vulkan build | This GPU has no CUDA. Standard GGUF quants are what the tested servers in this folder already run. HIP is for the ternary pack, which has no Vulkan kernels |
| `--model` Q4 | 16.42 GiB. The Q8 file does not leave a KV budget on 24 GB |
| `--ctx-size 128000` | 128000 divides by 256, so the context pad does not change it. The tested 27B server on this GPU uses this window. Eight full-context layers at `q8_0` are 2.08 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. With the weights that is about 18.6 GiB of known payload. The compute buffer is not measured. The GGUF `context_length` is 1048576 |
| `--parallel 1` | One request, one slot. Continuous batching is on by default, and every live slot is one step on this GPU. `n_seq_max` 1 keeps `n_ctx_seq` at 128000. Do not pass `--kv-unified` |
| `--n-gpu-layers all` | Every layer on the GPU. Default is `auto`. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48. Do not pass `--cpu-moe`. [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is a CUDA report about experts of layers 6–17 left on the host |
| `--split-mode none` | Default is `layer`. One visible GPU is already one device. `none` is the guard if a second device is visible |
| `q8_0` KV | Default `f16` for those eight layers at this context is 3.91 GiB. `q8_0` is 2.08 GiB |
| `--n-predict 8192` | Default is `-1`, which does not stop. One reply should not fill the 128000 window |
| Sampling | This Q4 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--cpu-moe` | That puts looped experts on the host. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

Images, only after `--list-devices` shows room: add `--mmproj ~/Documents/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf` and download that filename too. It is 639 MB decimal, 0.60 GiB.

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 128000
n_ctx_seq             = 128000
n_slots = 1, n_ctx_slot = 128000, kv_unified = 'false'
creating non-SWA KV cache, size = 128000 cells
creating     SWA KV cache, size = 1536 cells
n_ctx_seq (128000) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
flash_attn            = auto
```

The capacity line is this pin. The offload log lists one Vulkan device, and that device is the 7900 XTX.

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
          "name": "Inkling 2 Mini UD-Q4 (128000 q8/q8) - 7900 XTX",
          "contextWindow": 128000,
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

```bash
PI_CODING_AGENT_DIR="$PWD/.pi/agent" pi
```

Models, settings, and sessions for that launch stay in `.pi/agent`. `/model` reloads `models.json`.

## See also

- This GPU, tested Vulkan 27B: [7900-XTX-Qwen3.6-27b.md](7900-XTX-Qwen3.6-27b.md)
- This GPU, HIP ternary: [7900-XTX-Bonsai-2-27B.md](7900-XTX-Bonsai-2-27B.md)

**Last Updated:** 2026-11-21 (llama.cpp master, Vulkan, UD-Q4_K_XL @ 128000 q8/q8, one slot, untested)
