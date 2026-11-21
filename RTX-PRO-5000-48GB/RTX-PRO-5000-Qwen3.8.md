# RTX PRO 5000 Blackwell (48 GB) - Qwen3.8-27B

> ⚠️ **Not yet tested** through a full Pi tool turn on this card. The binary is the existing Forge `sm_120` build. No separate rebuild.

Forge, GPU index **1**. `nvidia-smi` reports **48935 MiB**, compute capability **12.0**, bus `0000:21:00.0`. The other cards are the RTX PRO 2000 (index 0, desktop) and two RTX PRO 6000 Max-Q (index 2 and 3). All four are CC 12.0. `libggml-cuda.so` in the current build already contains `sm_120` cubins.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (2026-10-24) |
| **Weights** | `/data/AIML/models/0bserverx-Qwen3.8-27B-RVN/RVN-Q8_0-multilingual-mtp.gguf` (28 GB) |
| **Projector** | `mmproj-Qwen3.8-27B-Q8_0.gguf` (601 MB), same directory |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` / `q8_0`, in VRAM for the life of the process |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` (Pi tools) |
| **Paths** | weights `/data/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant` |

**Pi:** `contextWindow` = 262144, `maxTokens` = 16384. Two limits, no DRY. [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Memory:** the Q8 file is 28 GB. On the Dual RTX `sm_120` pin, 262k q8/q8 KV is about 9–10 GB, so the load sits near 40 GB of the 48935 MiB on this card. The desktop framebuffer is on the RTX PRO 2000, not here.

The on-disk KV for a live server is VRAM. Do not create `bin/kv-cache`. `--slot-save-path` is what would use that directory, and this command does not set it. Hybrid Qwen does not restore those snapshots.

## Build

Use the binary already in `~/Documents/GitHub/llama-cpp-turboquant/build/bin`. CMake cache for that build is `CMAKE_CUDA_ARCHITECTURES=120`. A missing architecture fails at once with "no kernel image is available for execution on the device." This card does not need its own build.

Rebuild only when the tree itself changes. Same cmake, and do not wipe `build/` while a server is running:

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build build --config Release -j"$(nproc)"
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). `"120"` is compute capability (`sm_120`). `q8_0` / `q8_0` flash attention is in the default CUDA set.

## PRIMARY command

`CUDA_VISIBLE_DEVICES=1` is the 5000. Check `nvidia-smi` first. If that card's index changes, change the mask. Do not pass `--main-gpu`. Its default is already 0, and after the mask the 5000 is the only device the process can see. Do not pass `--threads`. One server lets llama.cpp pick the host count. Do not `pkill` — that kills every `llama-server` on this machine.

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model /data/AIML/models/0bserverx-Qwen3.8-27B-RVN/RVN-Q8_0-multilingual-mtp.gguf \
  --mmproj /data/AIML/models/0bserverx-Qwen3.8-27B-RVN/mmproj-Qwen3.8-27B-Q8_0.gguf \
  --alias qwen3.8-27b-gpu1 \
  --host 127.0.0.1 --port 8081 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --split-mode none \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --flash-attn on \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 2
```

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=1` | Selects the 5000. Index 0 is the 16 GB desktop card |
| `--split-mode none` | Keeps a forgotten mask from layer-splitting across the other three GPUs |
| `--cache-ram 0` | Hybrid Qwen does not restore the RAM prompt cache. The warm KV is the GPU cache of this process |
| `--load-mode none` | Buffered read. Needs ~28 GB free host RAM for the GGUF |
| Sampling / `--n-predict 16384` | Pi tools pin. Match Pi `maxTokens` |

Build `10878` (`643956891`) already defaults `--jinja` on and context shift off. `--kv-stream-arena-mib` stays unset (default 0). KV stays `q8_0` / `q8_0`; `turbo5` and `turbo6` are not part of this pin.

`cache-idle-slots requires --cache-ram, disabling` is the expected log line.

The GGUF name contains `mtp`. This command does not pass `--spec-type`. MTP stays off until a normal decode works.

### Confirm

```text
log: n_ctx_seq (262144)
log: load_mode = none
nvidia-smi   # index 1 grows after load and after a short decode; index 0 stays the desktop
```

Then: (1) load, (2) **first decode**, (3) new Pi session with real `ls` / `read`.

If the driver loses the card, `nvidia-smi` prints `Unable to determine the device handle for GPU1: 0000:21:00.0`. A launch then dies in `ggml_backend_cuda_buffer_set_tensor` with `CUDA error: unspecified launch failure`. That is a lost GPU, not a missing flag. Leave index 1 alone until `nvidia-smi` shows 48935 MiB again. The 96 GB cards are index 2 and 3.

## Pi `models.json`

`--alias` must match `id`. Port is 8081. Save to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8081/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.8-27b-gpu1",
          "name": "Qwen3.8-27B RVN Q8 (262k q8/q8) - RTX PRO 5000",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

## This box

**First-decode OOM** (load succeeded, generate did not), in order:

1. `--ubatch-size 512 --batch-size 512`
2. Drop `--ctx-size` to `196608`, then `131072`, and set Pi `contextWindow` to the same number

Keep `--fit off`. A bare `--fit on` can shrink the window under the Pi pin.

**Optional turbo V** after Q8 @ 262k decodes cleanly: `--cache-type-k q8_0 --cache-type-v turbo4`. New Pi session; real `ls` / `read` before trusting it.

## See also

- Measured sm_120 footprint: [Dual-RTX6000-Qwen3.8-localmaxing.md](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8-localmaxing.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-10-24 (build 10878: drop `--jinja` and `--no-context-shift`, both defaults)
