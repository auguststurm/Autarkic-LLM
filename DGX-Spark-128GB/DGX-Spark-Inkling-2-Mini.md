# DGX Spark Founders Edition (128 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this box (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

NVIDIA GB10, compute capability **12.1**, `aarch64`. The 128 GB pool is unified LPDDR5X shared with the Grace CPU. `uname -m` prints `aarch64`. Build on this machine. Port **8080**. The other guides in this folder also use 8080. Before starting, `ss -ltn | grep ':8080'` should print nothing.

Paths: `~/Documents/AIML/models` and `~/Documents/GitHub/llama.cpp-inkling`.

| Pin | Value |
| --- | --- |
| **Weights** | Q8, one copy, for four streams and for one stream. `Inkling-2-Mini-UD-Q8_K_XL.gguf_file` (22.87 GB decimal, 21.30 GiB). The baked uncensored file is Q4, 15.39 GiB |
| **Projector** | `mmproj-F16.gguf` (639 MB decimal, 0.60 GiB) |
| **Context** | `--ctx-size 2097152`, `--parallel 4 --no-kv-unified`. Each slot is **524288** |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 8192` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | ggml-org **master**, `CMAKE_CUDA_ARCHITECTURES=121` |

## Weights

The weights load once. Four streams do not load four copies. What grows is the KV cache: 34.00 GiB for four private 524288 windows, or 17.00 GiB for one 1048576 window. Q8 is 21.30 GiB, and it is the largest file Unsloth ships for this model. It fits both ways on this 128 GB pool. There is no higher quant to switch to for the single stream.

1. **Censored.** `Inkling-2-Mini-UD-Q8_K_XL.gguf_file` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). Q8, 21.30 GiB. Use this for four streams and for one stream.
2. **Censored plus the LoRA.** The same Q8 file, plus `Inkling-2-Mini-Uncensored-LoRA.gguf`. The adapter is about 42 MB. This is how to stay on Q8 and run the uncensored edit, for either stream count.
3. **Uncensored file.** `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` from [AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF). Q4, 15.39 GiB. The edit is already in the file. Leave `--lora` off. This is the lower quant, used when you want one uncensored GGUF instead of Q8 plus the adapter. It also fits four streams and one stream.

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

Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` on this machine. Do not copy a binary built on another computer. Do not point this command at `llama-cpp-turboquant` or `llama.cpp-prism`.

```bash
cd ~/Documents/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="121"
cmake --build build --config Release -j"$(nproc)" --target llama-server
```

If this directory is already on an older tag:

```bash
cd ~/Documents/GitHub/llama.cpp-inkling
git switch master
git pull --ff-only
```

`"121"` is this GPU. This source rewrites it to `121a`. The configure log should contain `Replacing 121 in CMAKE_CUDA_ARCHITECTURES with 121a`. `-DCMAKE_BUILD_TYPE=Release` is required because Unix Makefiles ignore `--config`. Use the `nvcc` already on this Spark. If `nvcc` rejects `121` or `121a`, that toolkit does not know this GPU. `no kernel image` means the `sm_121` image is missing. `unknown model architecture: 'inkling2_loop'` means the tree is too old.

## Server

`nvidia-smi` first. One GB10 is device 0. If a second device is listed, set `CUDA_VISIBLE_DEVICES` to the GB10's index.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

./llama-server \
  --model ~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  --mmproj ~/Documents/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 2097152 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 4 --no-kv-unified \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
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
  --ctx-size 2097152 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 4 --no-kv-unified \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
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
  --ctx-size 2097152 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 4 --no-kv-unified \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
```


Checked against llama.cpp master source and the Q8 file. Not a measured load on this Spark.

| Flag | Why |
| --- | --- |
| `--ctx-size 2097152` | Total of four private caches. With the shared pool off, `n_ctx_seq` is `--ctx-size / 4`, so each slot is 524288. 524288 divides by 256. Weights are 21.30 GiB, the projector is 0.60 GiB, and eight full-context layers at `q8_0` are 34.00 GiB for that total. Four sliding-window streams are 510 MiB. Known payload is about 56.4 GiB. That leaves about 70 GB of the 128 GB pool for the OS and the compute graph. The compute graph is not measured on this Spark. Four private copies of the full 1048576 window would be 68 GiB of full-context KV, about 90 GiB with the weights and the projector, which does not leave that room, so they are not this command |
| `--parallel 4 --no-kv-unified` | Four agents, four private caches. `--parallel 4` does not turn the shared pool on. `--no-kv-unified` keeps it off, so one agent cannot take cells from another. One stream, below, is the choice when a single session needs the full 1048576 window |
| `--n-gpu-layers all` | Every layer on the GPU. Default is `auto`. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | Default is `layer`. One visible GPU is already one device. `none` is the guard if a second device is visible |
| `q8_0` KV | Default `f16` for those eight layers at this total is 64 GiB. `q8_0` is 34.00 GiB |
| `--n-predict 8192` | Default is `-1`, which does not stop. The tested guide in this folder caps Pi at 8192. If a reply ends with `finish_reason: length`, raise this and Pi `maxTokens` together to 16384 |
| Sampling | This Q8 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

Text-only: drop `--mmproj` and skip the projector download.

### One stream

Use this when one session should hold the full window. Keep the Q8 file. The change is the context, not the quant: `--ctx-size 1048576` and `--parallel 1`. The GB10 then spends every decode step on that one request. Set Pi `contextWindow` to `1048576`. The LoRA command and the baked Q4 command take those same two flag changes.

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
  --n-predict 8192
```

Eight full-context layers are then 17.00 GiB. With the weights and the projector, the known payload is about 39.0 GiB. Confirm `n_seq_max = 1`, `kv_unified = false`, `n_ctx = 1048576`, `n_ctx_seq = 1048576`.

## Confirm

This block is the four-stream command.

```text
n_seq_max             = 4
kv_unified            = false
n_ctx                 = 2097152
n_ctx_seq             = 524288
n_slots = 4, n_ctx_slot = 524288, kv_unified = 'false'
creating non-SWA KV cache, size = 524288 cells
creating     SWA KV cache, size = 1536 cells
n_ctx_seq (524288) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
flash_attn            = auto
```

The capacity line is this pin. Each slot is half the GGUF window. The offload log lists one CUDA device. If `nvidia-smi` prints `Memory-Usage: Not Supported`, that is the unified pool, not a failed load.

## Pi `models.json`

`id` must match `--alias`. `~/.pi/agent/models.json`. Restart Pi. This JSON is the four-stream command. For one stream, set `contextWindow` to `1048576` and the name to the 1048576 window.

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
          "name": "Inkling 2 Mini UD-Q8 (524288 x4 q8/q8) - DGX Spark",
          "contextWindow": 524288,
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

- This box, Qwen3.6: [DGX-Spark-Qwen3.6.md](DGX-Spark-Qwen3.6.md)
- This box, Qwen3.8: [DGX-Spark-Qwen3.8.md](DGX-Spark-Qwen3.8.md)

**Last Updated:** 2026-11-22 (four private streams at 524288; one stream at 1048576 is the alternative, untested)
