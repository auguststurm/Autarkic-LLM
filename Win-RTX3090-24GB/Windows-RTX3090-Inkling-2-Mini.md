# Windows RTX 3090 (24 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this card (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

WSL2. NVIDIA GeForce RTX 3090, product memory **24576 MiB**, compute capability **8.6**. One GPU. `nvidia-smi` inside WSL is the check. If this card is not index 0, set `CUDA_VISIBLE_DEVICES` to its index. Port **8080**. The other guides in this folder also use 8080. This card holds one server. Before starting, `ss -ltn | grep ':8080'` should print nothing.

Paths: `~/AIML/models` and `~/GitHub/llama.cpp-inkling`. A GGUF opened from `/mnt/c/...` segfaults. Use the `nvcc` already installed in this WSL box. Do not install a Linux NVIDIA driver over the WSL `libcuda` stub.

| Pin | Value |
| --- | --- |
| **Weights** | `~/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (17.63 GB decimal, 16.42 GiB) |
| **Projector** | not in the command |
| **Context** | `--ctx-size 65536`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 16384` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | `~/GitHub/llama.cpp-inkling`, ggml-org **master**, `CMAKE_CUDA_ARCHITECTURES=86` |

## Weights

Skip this when that `.gguf_file` is already under `~/AIML/models/Inkling-2-Mini-GGUF`.

The Q8 file is 21.30 GiB. On a 24576 MiB card that leaves 2.70 GiB before KV, a projector, and the compute buffer. This pin is the Q4 file. The repo-root `Inkling-2-Mini-UD-Q4_K_XL.gguf` is a different file. Leave the Hub name. Pass the path as a filename: a token after `--include` is a filename, and `hf` then ignores `--include`.

```bash
mkdir -p ~/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --local-dir ~/AIML/models/Inkling-2-Mini-GGUF
```

## Build

Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` into its own directory. Do not point this command at `llama-cpp-turboquant` or `llama.cpp-prism`.

```bash
cd ~/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build --config Release -j"$(nproc)" --target llama-server
```

If this directory is already on an older tag:

```bash
cd ~/GitHub/llama.cpp-inkling
git switch master
git pull --ff-only
```

`"86"` is this GPU. `-DCMAKE_BUILD_TYPE=Release` is required because Unix Makefiles ignore `--config`. `no kernel image` means the binary has no `sm_86` image. `unknown model architecture: 'inkling2_loop'` means the tree is too old.

## Server

```bash
cd ~/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 65536 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

Checked against llama.cpp master source and the Q4 file size on the Hub. Not a measured load on this 3090.

| Flag | Why |
| --- | --- |
| `--model` Q4 | 16.42 GiB. The Q8 file does not leave a KV budget on 24576 MiB |
| `--ctx-size 65536` | 65536 divides by 256, so this binary's context pad does not change it. Eight full-context layers at `q8_0` are 1.0625 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. With the weights that is 17.61 GiB of known payload, leaving about 6.4 GiB before the compute buffer and whatever Windows has already allocated. Neither number is measured on this card. The GGUF `context_length` is 1048576 |
| `--parallel 1` | One request, one slot. Continuous batching is on by default, and every live slot is one step on this GPU. `n_seq_max` 1 keeps `n_ctx_seq` at 65536. Do not pass `--kv-unified` |
| `--n-gpu-layers all` | Every layer on the GPU. Default is `auto`. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | Default is `layer`. One visible GPU is already one device. `none` is the guard if a second device is visible |
| `q8_0` KV | Default `f16` for those eight layers at this context is 2.00 GiB. `q8_0` is 1.0625 GiB |
| `--n-predict 16384` | Default is `-1`, which does not stop. Pi `maxTokens` is the same number |
| Sampling | This Q4 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

Images, only after `nvidia-smi` shows room: add `--mmproj ~/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf` and download that filename too. It is 639 MB decimal, 0.60 GiB.

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 65536
n_ctx_seq             = 65536
n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
creating non-SWA KV cache, size = 65536 cells
creating     SWA KV cache, size = 1536 cells
n_ctx_seq (65536) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
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
          "name": "Inkling 2 Mini UD-Q4 (65536 q8/q8) - RTX 3090",
          "contextWindow": 65536,
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

Keep `--parallel 1` and `--n-gpu-layers all`. Drop `--ctx-size` to `32768`, then `16384`, and set Pi `contextWindow` to that number.

`--fit on` does not shrink a context you set. If the pinned context does not fit, this binary aborts with `n_gpu_layers already set by user to -2`. A CUDA out-of-memory after that estimate is the same ladder. Do not pass `--cpu-moe` or `--n-cpu-moe`.

If 16384 still does not load, switch `--model` to `Mainline_Support/Inkling-2-Mini-UD-Q3_K_XL.gguf_file` (13.63 GB decimal, 12.70 GiB). Download that name as a filename. The repo-root `.gguf` of that stem is a different file.

After 65536 loads and `nvidia-smi` still shows several GiB free, the stretch is `--ctx-size 131072`. Eight full-context layers are then 2.125 GiB, and the known payload is about 18.67 GiB. Match Pi `contextWindow`. That load is not measured.

## See also

- This machine, LFM2.5: [Windows-RTX3090-LFM2.5-2.6B.md](Windows-RTX3090-LFM2.5-2.6B.md)
- This machine, Bonsai 2: [Windows-RTX3090-Bonsai-2-27B.md](Windows-RTX3090-Bonsai-2-27B.md)

**Last Updated:** 2026-11-21 (llama.cpp master, UD-Q4_K_XL @ 65536 q8/q8, one slot, sm_86, untested)
