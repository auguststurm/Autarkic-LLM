# M5 MacBook Pro (48 GB) — Inkling 2 Mini

> ⚠️ **Not yet tested** on this machine (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

**Metal, not CUDA.** 48 GB unified memory, shared with macOS. Close other heavy apps. One server. `uname -m` prints `arm64`. Build on this Mac. Port **8080**. The other guides in this folder also use 8080. Before starting, `lsof -nP -iTCP:8080 -sTCP:LISTEN` should print nothing.

Paths: `~/Documents/AIML/models` and `~/Documents/GitHub/llama.cpp-inkling`.

| Pin | Value |
| --- | --- |
| **Weights** | `~/Documents/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` (22.87 GB decimal, 21.30 GiB) |
| **Projector** | `mmproj-F16.gguf` (639 MB decimal, 0.60 GiB) |
| **Context** | `--ctx-size 524288`, `--parallel 1` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 8192` |
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

Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master` on this Mac. Do not point this command at `llama-cpp-turboquant`.

```bash
cd ~/Documents/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON
cmake --build build --config Release -j"$(sysctl -n hw.logicalcpu)" --target llama-server
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
  --ctx-size 524288 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 512 \
  --ubatch-size 512 \
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
  --ctx-size 524288 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 512 \
  --ubatch-size 512 \
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
  --ctx-size 524288 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --batch-size 512 \
  --ubatch-size 512 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 8192
```


Checked against llama.cpp master source and the Q8 file. Not a measured load on this Mac.

| Flag | Why |
| --- | --- |
| `--ctx-size 524288` | 524288 divides by 256, so the context pad does not change it. The GGUF `context_length` is 1048576. Weights are 21.30 GiB, the projector is 0.60 GiB, and eight full-context layers at `q8_0` are 8.50 GiB. Forty sliding-window layers are 1536 cells (1024 + 512), 127.5 MiB. Known payload is about 30.5 GiB, leaving about 17.5 GiB of the 48 GB pool before macOS and the Metal scratch buffer. That scratch is not measured. The native window is about 39.0 GiB and leaves about 9 GiB, so it is not the pin |
| `--parallel 1` | One request, one slot. Continuous batching is on by default, and every live slot is one step on this GPU. `n_seq_max` 1 keeps `n_ctx_seq` at 524288. Do not pass `--kv-unified` |
| `--n-gpu-layers all` | Every layer on the GPU. Default `auto` also offloads every layer when a Metal device is present. `all` stops `--fit` from moving layers back to the CPU. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--split-mode none` | One GPU. Default `layer` is for splitting across devices |
| `--batch-size 512` `--ubatch-size 512` | Defaults are 2048 and 512. This machine's tested Metal server uses 512 for both. Drop both to 256, then 128, if the first prompt runs out of memory |
| `q8_0` KV | Default `f16` for those eight layers at this context is 16 GiB. `q8_0` is 8.50 GiB |
| `--n-predict 8192` | Default is `-1`, which does not stop. Pi `maxTokens` on this machine's tested guide is 8192 |
| Sampling | This Q8 file stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--load-mode none` | Default mmap. On Metal, the GPU buffer is the mapped file |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. Those reports are CUDA, not this Mac. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

Text-only: drop `--mmproj` and skip the projector download.

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 524288
n_ctx_seq             = 524288
n_slots = 1, n_ctx_slot = 524288, kv_unified = 'false'
creating non-SWA KV cache, size = 524288 cells
creating     SWA KV cache, size = 1536 cells
n_ctx_seq (524288) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
flash_attn            = auto
```

The capacity line is this pin. The offload log names the Metal device. Activity Monitor → Memory, after load and after a short decode: one server.

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
          "name": "Inkling 2 Mini UD-Q8 (524288 q8/q8) - M5 Pro",
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

- This machine, Qwen3.6: [M5-MacBook-Pro-Qwen3.6.md](M5-MacBook-Pro-Qwen3.6.md)
- This machine, Qwen3.8: [M5-MacBook-Pro-Qwen3.8.md](M5-MacBook-Pro-Qwen3.8.md)

**Last Updated:** 2026-11-21 (llama.cpp master, Metal, UD-Q8_K_XL @ 524288 q8/q8, one slot, untested)
