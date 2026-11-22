# RTX PRO 5000 Blackwell (48 GB) — Inkling 2 Mini

> ✅ **Tested** on this card (2026-11-22). The censored Q8 command below loaded at `--ctx-size 262144` and `--ubatch-size 256`. The LoRA command and the baked Q4 command use those same server flags.

GPU index **1**, NVIDIA RTX PRO 5000 Blackwell, **48935 MiB**, compute capability **12.0**, bus `00000000:21:00.0`. Port **8083**. Stop the Qwen server on 8081 before starting this one. Both use this GPU. Ports 8080–8082 belong to Forge Trinity. Before starting, `ss -ltn | grep -E ':808[0-3]'` should print nothing.

| Pin | Value |
| --- | --- |
| **Weights** | `/data/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file` (22.87 GB decimal, 21.30 GiB) |
| **Projector** | `mmproj-F16.gguf` (639 MB decimal, 0.60 GiB), repo root of that download |
| **Context** | `--ctx-size 262144`, `--parallel 1`, `--ubatch-size 256` |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 16384` |
| **Sampling** | temp **1.0** · top_p **1.0** · top_k **0** · min_p **0.0** |
| **Binary** | `~/Documents/GitHub/llama.cpp-inkling/build/bin/llama-server`, build **12625** (`cbb1c1c50`) |

## Weights

Three ways to run this model. The three server commands use them in this order.

1. **Censored.** `Inkling-2-Mini-UD-Q8_K_XL.gguf_file` from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF). The Unsloth file is Q8, 21.30 GiB. That is the censored weights, and it is the quality file on this machine.
2. **Censored plus the LoRA.** The same Unsloth file, plus `Inkling-2-Mini-Uncensored-LoRA.gguf`. The AtomicChat uncensored release is a Q4 file, 15.39 GiB. To stay on Q8 and still remove the refusals, put their LoRA on this Unsloth file. The LoRA is the edit by itself. The baked Q4 file is the same edit already merged, at the lower quant.
3. **Uncensored file.** `Inkling-2-Mini-Uncensored-AD-Q4_K.gguf` from [AtomicChat/Inkling-2-Mini-Uncensored-GGUF](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF). Leave `--lora` off. The edit is already in the file.

```bash
mkdir -p /data/AIML/models
hf download unsloth/Inkling-2-Mini-GGUF \
  Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  mmproj-F16.gguf \
  --local-dir /data/AIML/models/Inkling-2-Mini-GGUF
```

```bash
mkdir -p /data/AIML/models
hf download AtomicChat/Inkling-2-Mini-Uncensored-GGUF \
  Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  Inkling-2-Mini-Uncensored-LoRA.gguf \
  mmproj-Inkling-2-Mini-Uncensored-F16.gguf \
  --local-dir /data/AIML/models/Inkling-2-Mini-Uncensored-GGUF
```

`mmproj-Inkling-2-Mini-Uncensored-F16.gguf` has the same vision weights as Unsloth `mmproj-F16.gguf`. Either projector works. A command that already has `--mmproj` can keep the Unsloth path.

## Build

Clone [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `master`. Skip the clone and the build when `llama-server --version` prints build **12625** (`cbb1c1c50`).

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

## Power limit

GPU **1** only: NVIDIA RTX PRO 5000 Blackwell, bus `00000000:21:00.0`. Every command below passes that bus to `-i`. A bare `nvidia-smi -pl` writes every GPU. The server mask is the index; this cap follows the bus, so a later index change does not move the cap.

This card’s VBIOS accepts **250–300 W**. Default limit is **300 W** (driver 595.99.02, VBIOS `98.02.92.00.05`). Set **270 W**. A 72 GB card with the same 300 W rating ran Qwen3.8-27B serving at that lock: peak **244 W**, **73 °C**, under the cap ([Deep-AI-Evo](https://github.com/Deep-AI-Evo/qwen3.8-27b-q6k-fp8-rtx-pro5000-serving-benchmark)). This 48 GB card has not repeated that run.

`-pl` and `-pm` clear when the driver unloads, including a reboot. `nvidia-persistenced` on this box starts with `--no-persistence-mode`, so it does not apply the cap. The unit below does, on the next boot.

```bash
nvidia-smi -i 00000000:21:00.0 --query-gpu=index,name,pci.bus_id,power.limit,power.min_limit,power.max_limit --format=csv
sudo nvidia-smi -i 00000000:21:00.0 -pm 1
sudo nvidia-smi -i 00000000:21:00.0 -pl 270
```

Before the cap, that query returned `1, NVIDIA RTX PRO 5000 Blackwell, 00000000:21:00.0, 300.00 W, 250.00 W, 300.00 W`. If the name is anything else, stop. Applied 2026-11-21: persistence mode enabled via the daemon, and the limit changed from 300.00 W to 270.00 W. The unit file is installed, enabled, and active. A query on this bus reads `1, NVIDIA RTX PRO 5000 Blackwell, Enabled, 270.00 W`. A reboot runs that unit. Without the unit, a reboot returns the limit to 300 W.

`/etc/systemd/system/nvidia-rtx-pro-5000-power.service`:

```ini
[Unit]
Description=Cap RTX PRO 5000 power limit
After=nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -i 00000000:21:00.0 -pm 1
ExecStart=/usr/bin/nvidia-smi -i 00000000:21:00.0 -pl 270
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-rtx-pro-5000-power.service
systemctl is-enabled nvidia-rtx-pro-5000-power.service
nvidia-smi -i 00000000:21:00.0 --query-gpu=index,name,persistence_mode,power.limit --format=csv
```

Expect `enabled` and `1, NVIDIA RTX PRO 5000 Blackwell, Enabled, 270.00 W`. After a restart, run the same query. `journalctl -b -u nvidia-rtx-pro-5000-power.service` is the boot record.

Under a decode, same GPU:

```bash
nvidia-smi -i 00000000:21:00.0 --query-gpu=index,name,temperature.gpu,power.draw,power.limit,clocks_event_reasons.active --format=csv
```

If that field is rejected, use `clocks_throttle_reasons.active`.

`Unable to determine the device handle for GPU1: 0000:21:00.0` means the card left the bus. Confirm with `journalctl -k -b | grep -E 'NVRM: Xid|fallen off the bus'` (`Xid 79`). A power cycle clears it. The watt cap does not.

## Server

Check `nvidia-smi` first. If this card is no longer index 1, change the mask. Do not `pkill` `llama-server`. That kills every server on the machine.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model /data/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  --mmproj /data/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini-gpu1 \
  --host 127.0.0.1 --port 8083 \
  --ctx-size 262144 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --ubatch-size 256 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

### Original with the uncensored LoRA

Same command. `--lora` applies [AtomicChat's adapter](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF) to the weights above. Scale is 1.0.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model /data/AIML/models/Inkling-2-Mini-GGUF/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf_file \
  --lora /data/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-LoRA.gguf \
  --mmproj /data/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini-gpu1 \
  --host 127.0.0.1 --port 8083 \
  --ctx-size 262144 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --ubatch-size 256 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```

### Uncensored model

Same command. `--model` is the baked file. Leave `--lora` off.

```bash
cd ~/Documents/GitHub/llama.cpp-inkling/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model /data/AIML/models/Inkling-2-Mini-Uncensored-GGUF/Inkling-2-Mini-Uncensored-AD-Q4_K.gguf \
  --mmproj /data/AIML/models/Inkling-2-Mini-GGUF/mmproj-F16.gguf \
  --alias inkling-2-mini-gpu1 \
  --host 127.0.0.1 --port 8083 \
  --ctx-size 262144 \
  --n-gpu-layers all \
  --split-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 \
  --ubatch-size 256 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --n-predict 16384
```


Checked against this binary, build 12625 (`cbb1c1c50`), and the GGUF on disk. The censored command loaded on this card at `--ubatch-size 256`.

| Flag | Why |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=1` | GPU 1, RTX PRO 5000 Blackwell, 48935 MiB, CC 12.0, bus `00000000:21:00.0`, power limit 270.00 W. Index 0 is the desktop GPU. If the index changes, change this mask and leave the power-limit bus alone |
| `--split-mode none` | Default is `layer`. This mask leaves one visible GPU, so `layer` and `none` place the model on that GPU. `none` is the guard if the mask is dropped |
| `--n-gpu-layers all` | This binary's value for every layer. Default is `auto`. The file has `block_count` 24 and loop `[6, 18)` × 3, so `n_layer_all` is 48 |
| `--ctx-size 262144` | File `context_length` is 1048576. 262144 divides by 256, so the context pad does not change it. The KV for this window was not the buffer that failed |
| `--ubatch-size 256` | The prompt graph is reserved for `min(n_ctx, n_ubatch)` tokens. This is the size that loaded. The default 512 asked for 25532 MiB and did not fit |
| `--parallel 1` | One request, one slot. This binary's continuous batching is on by default, and every live slot is one step on this GPU. Another slot does not add a GPU. It shares the 270 W budget, so each request is slower. `n_seq_max` 1 keeps `n_ctx_seq` at 262144. Do not pass `--kv-unified`. At one slot it does not change the window |
| `q8_0` KV | Default is `f16`. Eight full-context layers at 262144 are 4.25 GiB of tensor payload; `f16` is 8 GiB. Physical layers 5, 11, 17, and 23 are full attention. Layers 5 and 23 run once. Layers 11 and 17 run on each of the 3 passes. The other 40 layers are the sliding window: 1280 cells (1024 + the 256 micro-batch), 106.25 MiB at `q8_0` |
| `--n-predict 16384` | Default is `-1`, which does not stop. Pi `maxTokens` is the same number |
| Sampling | The GGUF stores temp 1.0, top_k 0, top_p 1.0, min_p 0.0, and this binary copies those when the flags are omitted. CLI defaults otherwise are 0.80, 40, 0.95, 0.05 |
| no `--cpu-moe` | [Issue 32710](https://github.com/ggml-org/llama.cpp/issues/32710) is open. Experts of looped layers 6–17 on the host return NaN. Those reports are other GPUs, not this 5000. `--flash-attn` stays `auto` |
| no `--reasoning off` | `--reasoning` is the chat template (default `auto`). It does not turn the 3-pass layer loop on or off |

Text-only: drop `--mmproj` and skip the projector download.

## Confirm

```text
n_seq_max             = 1
kv_unified            = false
n_ctx                 = 262144
n_ctx_seq             = 262144
n_slots = 1, n_ctx_slot = 262144, kv_unified = 'false'
creating non-SWA KV cache, size = 262144 cells
creating     SWA KV cache, size = 1280 cells
n_ctx_seq (262144) < n_ctx_train (1048576) -- the full capacity of the model will not be utilized
flash_attn            = auto
nvidia-smi   # index 1 grows after load and after a short decode
```

The capacity line is this pin. The GGUF `context_length` is 1048576. This card stays at 262144. The offload log lists one CUDA device. A second device means the mask did not take. Order: load, first decode, then a new Pi session with a real `ls` / `read`.

## Pi `models.json`

`id` must match `--alias`, or Pi calls a different model. `~/.pi/agent/models.json`. Restart Pi.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8083/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "inkling-2-mini-gpu1",
          "name": "Inkling 2 Mini UD-Q8 (262k q8/q8) - RTX PRO 5000",
          "contextWindow": 262144,
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
  "defaultModel": "inkling-2-mini-gpu1"
}
```

```bash
PI_CODING_AGENT_DIR="$PWD/.pi/agent" pi
```

Models, settings, and sessions for that launch stay in `.pi/agent`. `/model` reloads `models.json`.

## See also

- Same weights on one 6000: [Dual-RTX6000-Inkling-2-Mini.md](../Dual-RTX6000-192GB/Dual-RTX6000-Inkling-2-Mini.md)
- Qwen on this card, same power unit: [RTX-PRO-5000-Qwen3.8.md](RTX-PRO-5000-Qwen3.8.md)

**Last Updated:** 2026-11-22 (censored Q8 @ 262144, ubatch 256, tested on this card)
