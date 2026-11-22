# DGX Spark Founders Edition (128 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this box (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

NVIDIA GB10, compute capability **12.1**, `aarch64`. The 128 GB pool is unified LPDDR5X shared with the Grace CPU. `uname -m` prints `aarch64`. Build on this machine. Port **8080**. The other guides in this folder also use 8080. Before starting, `ss -ltn | grep ':8080'` should print nothing.

Paths: `~/Documents/AIML/models` and `~/Documents/GitHub/llama.cpp-inkling`.

| Pin | Value |
| --- | --- |
| **Weights** | `~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` (22.87 GB decimal, 21.30 GiB) |
| **Projector** | `mmproj-F16.gguf` (639 MB decimal, 0.60 GiB) |
| **Context** | `--ctx-size 1048576`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 8192` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | ggml-org **master**, `CMAKE_CUDA_ARCHITECTURES=121` |

## Weights

Skip this when that `.gguf_file` is already under `~/Documents/AIML/models/Inkling-2-Mini-GGUF`.

Download the `Mainline_Support` file. The repo-root `Inkling-2-Mini-UD-Q8_K_XL.gguf` is a different file. Leave the Hub name. Pass both paths as filenames: a token after `--include` is a filename, and `hf` then ignores `--include` and downloads only the projector.

```bash
mkdir -p ~/Documents/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models/Inkling-2-Mini-GGUF
```

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
  --ctx-size 1048576 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
```

Checked against llama.cpp master source and the Q8 file. Not a measured load on this Spark.

| Flag | Why |
| --- | --- |
| `--ctx-size 1048576` | The GGUF `context_length`. 1048576 divides by 256, so the context pad does not change it. Weights are 21.30 GiB, the projector is 0.60 GiB, and eight full-context layers at `q8_0` are 17.00 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. Known payload is about 39.0 GiB. The OS shares the 128 GB pool. The compute buffer is not measured. Four private copies of this window would be 68 GiB of full-context KV and are not the pin |
| `--parallel 1` | One request, one slot. Continuous batching is on by default, and every live slot is one step on this GPU. `n_seq_max` 1 keeps `n_ctx_seq` at 1048576. Do not pass `--kv-unified` |
| `--n-gpu-layers all` | Every layer on the GPU. Default is `auto`. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | Default is `layer`. One visible GPU is already one device. `none` is the guard if a second device is visible |
| `q8_0` KV | Default `f16` for those eight layers at this context is 32 GiB. `q8_0` is 17.00 GiB |
| `--n-predict 8192` | Default is `-1`, which does not stop. The tested guide in this folder caps Pi at 8192. If a reply ends with `finish_reason: length`, raise this and Pi `maxTokens` together to 16384 |
| Sampling | This Q8 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

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

`n_ctx_seq` matches the GGUF `context_length`. A line that says the full capacity will not be utilized means the context is below this pin. The offload log lists one CUDA device. If `nvidia-smi` prints `Memory-Usage: Not Supported`, that is the unified pool, not a failed load.

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
          "name": "Inkling 2 Mini UD-Q8 (1048576 q8/q8) - DGX Spark",
          "contextWindow": 1048576,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

## Out of memory

Keep `--parallel 1` and `--n-gpu-layers all`. Drop `--ctx-size` to `524288`, then `262144`, and set Pi `contextWindow` to that number.

`--fit on` does not shrink a context you set. If the pinned context does not fit, this binary aborts with `n_gpu_layers already set by user to -2`. A CUDA out-of-memory after that estimate is the same ladder. Do not pass `--cpu-moe` or `--n-cpu-moe`.

## See also

- This box, Qwen3.6: [DGX-Spark-Qwen3.6.md](DGX-Spark-Qwen3.6.md)
- This box, Qwen3.8: [DGX-Spark-Qwen3.8.md](DGX-Spark-Qwen3.8.md)

**Last Updated:** 2026-11-21 (llama.cpp master, UD-Q8_K_XL @ 1048576 q8/q8, one slot, sm_121, untested)
