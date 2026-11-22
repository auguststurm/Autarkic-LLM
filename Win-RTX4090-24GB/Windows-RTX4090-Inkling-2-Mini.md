# Windows RTX 4090 (24 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this card (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

WSL2. NVIDIA GeForce RTX 4090, product memory **24576 MiB**, compute capability **8.9** (Ada). One GPU. `nvidia-smi` inside WSL is the check. If this card is not index 0, set `CUDA_VISIBLE_DEVICES` to its index. Port **8080**. The other guides in this folder also use 8080. This card holds one server. Before starting, `ss -ltn | grep ':8080'` should print nothing.

Paths: `~/AIML/models` and `~/GitHub/llama.cpp-inkling`. Keep the GGUF under `~/AIML`. Do not open it from `/mnt/c`.

| Pin | Value |
| --- | --- |
| **Weights** | `~/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (17.63 GB decimal, 16.42 GiB) |
| **Projector** | not in the command |
| **Context** | `--ctx-size 98304`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Load** | `--load-mode none` |
| **Output** | `--n-predict 16384` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | `~/GitHub/llama.cpp-inkling`, ggml-org **master**, `CMAKE_CUDA_ARCHITECTURES=89` |

## Weights

Three ways to run this model. The three server commands use them in this order.

1. **Censored.** `Inkling-2-Mini-UD-Q4_K_XL.gguf_file` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The Unsloth file is Q4, 16.42 GiB. That is the censored weights for this machine.
2. **Censored plus the LoRA.** The same Unsloth file, plus `Inkling-2-Mini-Uncensored-LoRA.gguf`. The AtomicChat uncensored release is also a Q4 file, 15.39 GiB. The LoRA puts that edit on the Unsloth Q4 file. The baked file is the edit already merged, in AtomicChat's own Q4.
3. **Uncensored file.** `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` from [AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF). Leave `--lora` off. The edit is already in the file.

```bash
mkdir -p ~/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --local-dir ~/AIML/models/Inkling-2-Mini-GGUF
```

```bash
mkdir -p ~/AIML/models
hf download AtomicChat/Inkling-2-Mini-Uncensored-GGUF \
  Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  Inkling-2-Mini-Uncensored-LoRA.gguf \
  mmproj-Inkling-2-Mini-Uncensored-F16.gguf \
  --local-dir ~/AIML/models/Inkling-2-Mini-Uncensored-GGUF
```

`mmproj-Inkling-2-Mini-Uncensored-F16.gguf` has the same vision weights as Unsloth `mmproj-F16.gguf`. Either projector works. A command that already has `--mmproj` can keep the Unsloth path.

## Build

Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` into its own directory. Do not point this command at `llama-cpp-turboquant`.

```bash
cd ~/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89"
cmake --build build --config Release -j"$(nproc)" --target llama-server
```

If this directory is already on an older tag:

```bash
cd ~/GitHub/llama.cpp-inkling
git switch master
git pull --ff-only
```

`"89"` is this GPU. `-DCMAKE_BUILD_TYPE=Release` is required because Unix Makefiles ignore `--config`. Do not pass `GGML_CUDA_FA_ALL_QUANTS`. This source's default FlashAttention quant list already includes `q8_0-q8_0`. `no kernel image` means the binary has no `sm_89` image. `unknown model architecture: 'inkling2_loop'` means the tree is too old.

## Server

```bash
cd ~/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 98304 \
  --n-gpu-layers all \
  --split-mode none \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 256 \
  --ubatch-size 256 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

### Original with the uncensored LoRA

Same command. `--lora` applies [AtomicChat's adapter](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) to the weights above. Scale is 1.0.

```bash
cd ~/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file \
  --lora ~/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-LoRA.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 98304 \
  --n-gpu-layers all \
  --split-mode none \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 256 \
  --ubatch-size 256 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

### Uncensored model

Same command. `--model` is the baked file. Leave `--lora` off.

```bash
cd ~/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 98304 \
  --n-gpu-layers all \
  --split-mode none \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 256 \
  --ubatch-size 256 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```


Checked against llama.cpp master source, the Q4 file header, and this folder's WSL load note. Not a measured load of this model.

| Flag | Why |
| --- | --- |
| `--model` Q4 | 16.42 GiB. The Q8 file does not leave a KV budget on 24576 MiB |
| `--ctx-size 98304` | 98304 divides by 256, so the context pad does not change it. This box's agent prompts have passed 70k tokens. Eight full-context layers at `q8_0` are 1.59 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. With the weights that is about 18.1 GiB of known payload, leaving about 5.9 GiB before the compute buffer and whatever Windows has already allocated. Neither number is measured for this model. The GGUF `context_length` is 1048576 |
| `--parallel 1` | One request, one slot. Continuous batching is on by default, and every live slot is one step on this GPU. `n_seq_max` 1 keeps `n_ctx_seq` at 98304. Do not pass `--kv-unified` |
| `--n-gpu-layers all` | Every layer on the GPU. Default is `auto`. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | Default is `layer`. One visible GPU is already one device. `none` is the guard if a second device is visible |
| `--load-mode none` | This binary has no `--no-mmap`. `none` is the buffered read. This WSL box's working servers use that read. The Q4 file is 16.42 GiB, so the host needs that much free RAM |
| `--batch-size 256` `--ubatch-size 256` | Defaults are 2048 and 512. On this 24 GB card the prefill scratch is the peak. 256 is the batch this folder already runs. Drop both to 128 if prefill runs out of memory |
| `q8_0` KV | Default `f16` for those eight layers at this context is 3.00 GiB. `q8_0` is 1.59 GiB |
| `--n-predict 16384` | Default is `-1`, which does not stop. Pi `maxTokens` on this box is 16384 |
| Sampling | This Q4 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

Images, only after `nvidia-smi` shows room: add `--mmproj ~/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf` and download that filename too. It is 639 MB decimal, 0.60 GiB.

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 98304
n_ctx_seq             = 98304
n_slots = 1, n_ctx_slot = 98304, kv_unified = 'false'
creating non-SWA KV cache, size = 98304 cells
creating     SWA KV cache, size = 1536 cells
n_ctx_seq (98304) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
flash_attn            = auto
loading model tensors, this can take a while... (load_mode = none)
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
          "name": "Inkling 2 Mini UD-Q4 (98304 q8/q8) - RTX 4090",
          "contextWindow": 98304,
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

## See also

- This machine, Qwen3.6: [Windows-RTX4090-Qwen3.6.md](Windows-RTX4090-Qwen3.6.md)
- This machine, LFM2.5: [Windows-RTX4090-LFM2.5-2.6B.md](Windows-RTX4090-LFM2.5-2.6B.md)

**Last Updated:** 2026-11-21 (llama.cpp master, UD-Q4_K_XL @ 98304 q8/q8, one slot, sm_89, untested)
