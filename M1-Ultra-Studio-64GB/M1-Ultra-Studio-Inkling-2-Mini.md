# M1 Ultra Mac Studio (64 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this machine (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn. Metal can load and then run out of memory on the first prompt.

**Metal, not CUDA.** 64 GB unified memory, shared with macOS. `uname -m` prints `arm64`. Build on this Mac. Port **8080**. The other guide in this folder also uses 8080. Before starting, `lsof -nP -iTCP:8080 -sTCP:LISTEN` should print nothing.

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

Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` on this Mac. Do not point this command at `llama.cpp-prism` or `llama-cpp-turboquant`.

```bash
cd ~/Documents/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON
cmake --build build --config Release -j"$(sysctl -n hw.ncpu)" --target llama-server
```

Metal is the default backend on macOS in this source. The flag keeps the build from coming up CPU-only. The Metal library is embedded by default. If this directory is already on an older tag:

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

The list should show the Apple GPU. If it shows only CPU, the Metal backend did not build.

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

Checked against llama.cpp master source and the Q8 file. Not a measured load on this Mac.

| Flag | Why |
| --- | --- |
| `--ctx-size 1048576` | The GGUF `context_length`. 1048576 divides by 256, so the context pad does not change it. Weights are 21.30 GiB, the projector is 0.60 GiB, and eight full-context layers at `q8_0` are 17.00 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. Known payload is about 39.0 GiB on the 64 GB unified pool. macOS shares that pool. The Metal scratch buffer is not measured |
| `--parallel 1` | One request, one slot. Continuous batching is on by default, and every live slot is one step on this GPU. `n_seq_max` 1 keeps `n_ctx_seq` at 1048576. Do not pass `--kv-unified` |
| `--n-gpu-layers all` | Every layer on the GPU. Default `auto` also offloads every layer when a Metal device is present. `all` stops `--fit` from moving layers back to the CPU. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | One GPU. Default `layer` is for splitting across devices |
| `q8_0` KV | Default `f16` for those eight layers at this context is 32 GiB. `q8_0` is 17.00 GiB. `f16` plus the weights is about 54 GiB before macOS |
| `--n-predict 16384` | Default is `-1`, which does not stop. Pi `maxTokens` on this machine's other guide is 16384 |
| Sampling | This Q8 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--load-mode none` | Default mmap. On Metal, the GPU buffer is the mapped file. `none` would copy the 21.30 GiB file into RAM as well |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. Those reports are CUDA, not this Mac. `--flash-attn` stays `auto` |
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

`n_ctx_seq` matches the GGUF `context_length`. A line that says the full capacity will not be utilized means the context is below this pin. The offload log names the Metal device. Activity Monitor → Memory, after load and after a short decode: one server, not a second copy of the weights.

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
          "name": "Inkling 2 Mini UD-Q8 (1048576 q8/q8) - M1 Ultra",
          "contextWindow": 1048576,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

## Out of memory

Keep `--parallel 1`, `--n-gpu-layers all`, and the default mmap. A load that succeeds and a first prompt that dies is the Metal failure on this machine. Drop `--ctx-size` to `524288`, then `262144`, and set Pi `contextWindow` to that number. If the failure is still the first prompt, set `--batch-size 512` (ubatch is already 512), then both to `256`.

`--fit on` does not shrink a context you set. If the pinned context does not fit, this binary aborts with `n_gpu_layers already set by user to -2`. Do not pass `--cpu-moe` or `--n-cpu-moe`.

## See also

- This machine, Bonsai 2: [M1-Ultra-Studio-Bonsai-2-27B.md](M1-Ultra-Studio-Bonsai-2-27B.md)

**Last Updated:** 2026-11-21 (llama.cpp master, Metal, UD-Q8_K_XL @ 1048576 q8/q8, one slot, untested)
