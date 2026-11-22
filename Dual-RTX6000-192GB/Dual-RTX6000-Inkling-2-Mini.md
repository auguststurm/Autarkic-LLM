# Dual RTX 6000 Pro Max-Q — Inkling 2 Mini

> ⚠️ **Not yet tested** on this card (written 2026-11-21). Confirm load, then first decode, before a Pi tool turn.

GPU index **2**, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, **97887 MiB**, compute capability **12.0**, bus `00000000:C1:00.0`. Port **8084**, alias `inkling-2-mini-gpu2`. The other 6000 is the same command with `CUDA_VISIBLE_DEVICES=3`, port **8085**, alias `inkling-2-mini-gpu3` (index **3**, bus `00000000:E1:00.0`, same 97887 MiB). Index 0 is the desktop card. Do not use mask 0.

Ports 8080–8082 belong to Forge Trinity. The 5000 Inkling server is port 8083. Before starting, `ss -ltn | grep -E ':808[0-5]'` should print nothing.

| Pin | Value |
| --- | --- |
| **Weights** | `/data/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` (22.87 GB decimal, 21.30 GiB) |
| **Projector** | `mmproj-F16.gguf` (639 MB decimal, 0.60 GiB), repo root of that download |
| **Context** | `--ctx-size 2097152`, `--parallel 4`, no shared KV pool. Each slot is `n_ctx_seq` **524288** |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 16384` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | `~/Documents/GitHub/llama.cpp-inkling/build/bin/llama-server`, build **12625** (`cbb1c1c50`) |

## Weights

Skip this when `Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` is already under `/data/AIML/models/Inkling-2-Mini-GGUF`.

Download the `Mainline_Support` file. The repo-root `Inkling-2-Mini-UD-Q8_K_XL.gguf` is a different file. Leave the Hub name. Pass both paths as filenames: a token after `--include` is a filename, and `hf` then ignores `--include` and downloads only the projector.

```bash
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  mmproj-F16.gguf \
  --local-dir /data/AIML/models/Inkling-2-Mini-GGUF
```

## Build

Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master`. One tree serves both 6000s. This box already has build **12625** (`cbb1c1c50`). Skip the clone and the build when `llama-server --version` prints that.

```bash
cd ~/Documents/GitHub
gh repo clone ggml-org/llama.cpp llama.cpp-inkling
cd llama.cpp-inkling
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build --config Release -j"$(nproc)" --target llama-server
```

If this directory is already on an older tag:

```bash
cd ~/Documents/GitHub/llama.cpp-inkling
git switch master
git pull --ff-only
```

`"120"` is this GPU. CMake rewrites it to `120a`. The configure log should contain `Replacing 120 in CMAKE_CUDA_ARCHITECTURES with 120a`. `-DCMAKE_BUILD_TYPE=Release` is required because Unix Makefiles ignore `--config`. Blackwell needs CUDA 12.8 or newer. `no kernel image` means the arch rewrite did not happen. `unknown model architecture: 'inkling2_loop'` means the tree is too old.

## Server

Check `nvidia-smi` first. If this card is no longer index 2, change the mask. Do not `pkill` `llama-server`. That kills every server on the machine.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

CUDA_VISIBLE_DEVICES=2 ./llama-server \
  --model /data/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  --mmproj /data/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini-gpu2 \
  --host 127.0.0.1 --port 8084 \
  --ctx-size 2097152 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 4 --no-kv-unified \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

Checked against this binary, build 12625 (`cbb1c1c50`), and the GGUF on disk. Not a measured load.

| Flag | Why |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=2` | GPU 2, RTX PRO 6000 Blackwell Max-Q, 97887 MiB, CC 12.0, bus `00000000:C1:00.0`. Mask 0 is the desktop GPU. The other 6000 is index 3, bus `00000000:E1:00.0` |
| `--split-mode none` | Default is `layer`. This mask leaves one visible GPU, so `layer` and `none` place the model on that GPU. `none` is the guard if the mask is dropped |
| `--n-gpu-layers all` | This binary's value for every layer. Default is `auto`. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--ctx-size 2097152` | Total of the four private caches. With the shared pool off, this binary sets `n_ctx_seq = n_ctx / n_seq_max`, and 2097152 / 4 = 524288. Both divide by 256, so the context pad does not change them. Physical layers 5, 11, 17, and 23 are full attention. Layers 5 and 23 run once. Layers 11 and 17 run on each of the 3 passes. That is 8 full-context layers. At `q8_0` those 8 layers are 34.00 GiB for this total. The other 40 layers are the sliding window: 1536 cells by 4 streams, 510 MiB. Weights are 21.30 GiB and the projector is 0.60 GiB. Known payload is about 56.4 GiB on this 97887 MiB card, leaving about 39 GiB before the compute buffer. That buffer is not measured. Four private 1048576 caches are 68 GiB of full-context KV and are not the pin |
| `--parallel 4 --no-kv-unified` | One private cache per concurrent agent. The struct default is the pool off, and `--parallel 4` does not turn it on. `--kv-unified` puts all four slots in one cache of `--ctx-size` cells. A second live agent then takes cells from the first, and a full cache returns `Context size has been exceeded` and releases every processing slot |
| `q8_0` KV | Default `f16` for the same eight layers at this total is 64 GiB. `q8_0` is 34.00 GiB. This card's other servers use `q8_0` |
| `--n-predict 16384` | Default is `-1`, which does not stop. Pi `maxTokens` is the same number |
| Sampling | The GGUF stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. Those reports are other GPUs, not this 6000. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

Text-only: drop `--mmproj` and skip the projector download.

## Confirm

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
nvidia-smi   # index 2 grows after load and after a short decode
```

The capacity line is this pin. Each slot is half the GGUF `context_length`. Raising a slot to 1048576 is the four-private-1M case in the table above. The offload log lists one CUDA device. A second device means the mask did not take. Order: load, first decode, then a new Pi session with a real `ls` / `read`.

## Pi `models.json`

`id` must match `--alias`, or Pi calls a different model. `~/.pi/agent/models.json`. Restart Pi. One provider entry has one `baseUrl`. This block is port 8084. `contextWindow` for this server is the slot, 524288. It is not `--ctx-size` 2097152 and it is not the GGUF `context_length` 1048576.

The other 6000 uses the same block with `baseUrl` port **8085** and `id` `inkling-2-mini-gpu3`.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8084/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "inkling-2-mini-gpu2",
          "name": "Inkling 2 Mini UD-Q8 (524288 x4 q8/q8) - RTX PRO 6000",
          "contextWindow": 524288,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

## Out of memory

Keep `--parallel 4`, `--no-kv-unified`, and `--n-gpu-layers all`. Drop `--ctx-size` to `1048576` (each slot 262144), then `524288` (each slot 131072). Set Pi `contextWindow` to that per-slot number, not to `--ctx-size`.

`--fit on` does not shrink a context you set. If the pinned context does not fit, this binary aborts with `n_gpu_layers already set by user to -2`. A CUDA out-of-memory after that estimate is the same ladder. Do not pass `--cpu-moe` or `--n-cpu-moe` to free VRAM. That is the open NaN path on layers 6–17.

## See also

- Same model on the RTX PRO 5000: [RTX-PRO-5000-Inkling-2-Mini.md](../RTX-PRO-5000-48GB/RTX-PRO-5000-Inkling-2-Mini.md)
- Qwen on these cards: [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-Qwen3.8.md)

**Last Updated:** 2026-11-21 (llama-server build 12625, UD-Q8_K_XL, four private slots at 524288, `CUDA_VISIBLE_DEVICES=2`)
