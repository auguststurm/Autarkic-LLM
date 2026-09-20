# M4 Mac Mini (16 GB) - Qwen3.6-35B-A3B (experimental)

> ⚠️ **Untested. Tight fit.** Daily driver: **[Gemma 4 E2B](M4-Mac-Mini-Gemma-4-E2B.md)** (~3 GB). Air IQ4_NL (~18 GB) **cannot load**.

MoE 35B total / ~3B active — **full weights must still fit**. Only IQ2/IQ1-class quants load. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested experiment |
| **Weights** | `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (~11.5 GB) |
| **Catalog** | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| **Context** | `--ctx-size 8192` (`--fit off`) — starting estimate; raise only after first decode |
| **KV** | `q8_0` K / **turbo2** V |
| **Wired GPU** | `sudo sysctl iogpu.wired_limit_mb=13000` (not persistent) |

On 16 GB the weights already consume most of usable memory. **Any useful context only fits if V is compressed** (`q8_0` K / **turbo2** V, `--flash-attn on`). Do not use bare `--fit on` for Pi. Tiers: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md#2-turboquant-kv-cache).

## Memory reality (this box)

Usable budget after macOS overhead (~3–4 GB) and the default GPU wired-memory cap (~⅔ of RAM ≈ 10.7 GB) is small. Actual GGUF sizes for `unsloth/Qwen3.6-35B-A3B-GGUF`:

```text
Quant        Size      Fits 16 GB?
-----------  --------  -------------------------------------------
UD-IQ1_M     10.0 GB   yes, with headroom (lowest quality)
UD-IQ2_XXS   10.8 GB   marginal
UD-IQ2_M     11.5 GB   marginal  <- recommended starting point
UD-Q2_K_XL   12.3 GB   tight; needs raised wired limit
UD-Q3_K_S    15.4 GB   no
UD-IQ4_NL    ~18 GB    no (Air quant — does not fit)
UD-Q4_K_XL   22.4 GB   no
```

Raise the Metal GPU memory limit before launching (hands ~13 GB to the GPU; revert with a reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=13000
```

> ⚠️ **This leaves only ~3 GB for macOS** (13 GB of 16 GB handed to the GPU) — below Apple’s recommended headroom. Watch **Activity Monitor → Memory → Memory Pressure** and back off the moment it turns yellow/red or the system starts swapping: lower the wired limit, drop context, or switch to `UD-IQ1_M`. The `iogpu.wired_limit_mb` change is **not** persistent — a reboot restores the default (~⅔ of RAM). `13000` is sized for ~11.5 GB `IQ2_M` plus KV/compute; for `UD-IQ1_M` (~10 GB), try `12000` for more OS headroom.

- **Pinned context below is a starting estimate (untested), not a verified decode matrix.** Confirm **first decode**, not only load (Air lesson: process can report `n_ctx` and still Metal-OOM on first token).
- Close heavy apps. Prefer **one** long-lived `llama-server` — avoid rapid stop/start thrash on unified memory.

Expect a **small** context on 16 GB even with turbo2. Gemma is the better Pi daily driver. `--reasoning off` for Pi; pin `contextWindow` to real `n_ctx_seq`.

## Download

- **Model:** `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (~11.5 GB MoE; expect a noticeable quality drop vs Q4+/IQ4)
- **Path:** `~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- **Fallback if OOM:** `UD-IQ1_M` (~10 GB)

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  --local-dir ~/Documents/AIML/models
```

## Build

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

# TheTom TurboQuant fork — not ggml-org/llama.cpp
# https://github.com/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)

cd bin
mkdir -p ./kv-cache
```

Confirm the binary accepts turbo types:

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> **Fork version:** tip that includes Metal turbo4 `rnorm` fix (**`b01afefed` / PR #200 content or later**). No manual Metal shader edit on current TheTom tip. Rebuild after `git pull`.

## PRIMARY command

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`. Raise wired limit first (see above).

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 8192 \
  --fit off \
  --cache-type-k q8_0 --cache-type-v turbo2 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 64 \
  --batch-size 64 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 0 --temp 0.65 --top-p 0.90 \
  --n-predict 4096 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 8192` | Conservative **untested** pin so Pi sees a real window; raise only after decode works |
| `--cache-type-k q8_0 --cache-type-v turbo2` | TurboQuant on V so context can exist after ~11.5 GB weights |
| `--flash-attn on` | Required for turbo KV types |
| `--ubatch-size` / `--batch-size` **64** | Matches Air’s tight-batch lesson; drop to 32 if peak OOM |
| `--fit off` | Avoid silent context collapse; pin deliberately |
| Thinking off | Agent-friendly non-thinking Qwen3.6 |
| `--host 127.0.0.1` | Local-only default (use `0.0.0.0` only if you intend LAN exposure — no auth) |
| No checkpoint flags | Qwen3.6 hybrid attention often won’t restore checkpoints usefully — see [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

Confirm **`n_ctx` / `n_ctx_seq`** in the log or `GET /v1/models`, then **run a short decode**.

### Fallbacks if you Metal-OOM

1. Confirm `sudo sysctl iogpu.wired_limit_mb=13000` and close other apps.
2. Drop batch to `32`.
3. Drop `--ctx-size` to `4096`.
4. Switch weights to `UD-IQ1_M`.
5. Do **not** switch to IQ4_NL / Q4_K_XL — TurboQuant compresses **KV**, not weights.

### If 8k is stable (raise carefully)

```bash
# Same flags, try in order and confirm decode each time:
#   --ctx-size 12288
#   --ctx-size 16384
# Stop at the first Metal OOM; report the max that worked.
```

Watch logs for `kIOGPUCommandBufferCallbackErrorOutOfMemory` and Memory Pressure. Everyday Pi: prefer [Gemma 4 E2B](M4-Mac-Mini-Gemma-4-E2B.md).

## Pi Coding Agent `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi.

`maxTokens` ≤ `--n-predict` (4096). `contextWindow` = the **`--ctx-size` you actually ran** (default start: 8192).

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.6-35B-A3B",
          "name": "Qwen3.6-35B-A3B IQ2_M turbo2 (8k) - M4 Mini",
          "contextWindow": 8192,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

> **Provisional until measured.** After your first stable run, set `contextWindow` to the effective `n_ctx_seq` from the log. Report results via issue/PR.

**Last Updated:** 2026-09-20 (recipe density; ⚠️ untested)
