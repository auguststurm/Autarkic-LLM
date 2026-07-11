# M4 Mac Mini (16 GB) - Qwen3.6-35B-A3B (experimental)

Experimental setup for Mac Mini M4 with 16 GB unified memory, using **llama-cpp-turboquant**. Structure follows the [M4 MacBook Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md), but **16 GB cannot run the Air’s IQ4_NL (~18 GB) quant**. This guide only makes sense if you want to push a heavily compressed MoE; for daily use prefer **[Gemma 4 E2B](M4-Mac-Mini-Gemma-4-E2B.md)** (~3 GB, lots of headroom).

> ⚠️ **Untested on hardware. 16 GB is a tight fit for this model.** Qwen3.6-35B-A3B is a 35B-parameter MoE; even though only ~3B params are active per token, the *full* weights must be resident. 4-bit quants (21–22 GB) **cannot load**. Only IQ2/IQ1-class quants fit, with a raised GPU wired limit and a modest pinned context. Treat this as an experiment for people who want to push the limit and report back.

## Why TurboQuant matters here

On 16 GB the weights already consume most of usable memory. **Any useful context only fits if the KV cache is compressed.** The Air guide proved that on 24 GB with IQ4; here the same idea applies with a smaller quant and smaller window.

| Tier | ~bits / value | Role on this Mini |
| --- | --- | --- |
| `turbo4` | ~4.25 | Milder; less context |
| `turbo3` | ~3.25 | Middle ground |
| **`turbo2`** | ~2.0 | **Starting V-cache** — max context under extreme pressure |

Asymmetric config below: **K = `q8_0`**, **V = `turbo2`**. **Requires** `--flash-attn on` and this fork (upstream rejects `turbo*`).

**Do not use bare `--fit on` for Pi.** The Air lesson: default fit can crush context toward ~4096. Pin `--ctx-size` and set `--fit off`; if you OOM, lower the pin.

## Memory reality (read this)

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

## Pi Coding Agent: read this first

Pi needs a realistic `contextWindow` matching the server’s real `n_ctx_seq`.

**Always pin `--ctx-size` and set `--fit off`.**

**Thinking / empty replies** (Qwen3.6):

- `--reasoning off`
- `--chat-template-kwargs '{"enable_thinking":false}'`

Expect a **small** context on 16 GB even with turbo2. Gemma is the better Pi daily driver on this machine.

## Recommended model

- **Model:** `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (~11.5 GB MoE; expect a noticeable quality drop vs Q4+/IQ4)
- **Path:** `~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- **Fallback if OOM:** `UD-IQ1_M` (~10 GB)

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  --local-dir ~/Documents/AIML/models
```

## Build instructions (TurboQuant fork)

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

# TheTom TurboQuant fork — not ggml-org/llama.cpp
# https://github.com/TheTom/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
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

## Optimized llama-server command (IQ2_M + TurboQuant @ 8k start)

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

## Performance notes

- 16 GB is the binding constraint, not compute. Quant choice is driven entirely by the memory budget above.
- Expect IQ2-level quality (noticeably below Q5/Q6 or Air IQ4). For everyday use, prefer [Gemma 4 E2B](M4-Mac-Mini-Gemma-4-E2B.md).
- turbo2 V costs some decode speed vs `q8_0`/`turbo4` — that is the trade for any usable context.
- Watch logs for `kIOGPUCommandBufferCallbackErrorOutOfMemory` and Memory Pressure.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Pattern reference: [M4 Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).

## Pi Coding Agent `models.json` snippet

`maxTokens` ≤ `--n-predict` (4096). `contextWindow` = the **`--ctx-size` you actually ran** (default start: 8192).

```json
{
  "id": "qwen3.6-35B-A3B",
  "name": "Qwen3.6-35B-A3B IQ2_M turbo2 (8k) - M4 Mini",
  "contextWindow": 8192,
  "maxTokens": 4096
}
```

> **Provisional until measured.** After your first stable run, set `contextWindow` to the effective `n_ctx_seq` from the log.

Nest in the full `providers` wrapper from [`local-setup.md`](../local-setup.md#6-pi-coding-agent--hermes-integration). Point Pi at `http://127.0.0.1:8080/v1`.

## Report your results

This config is untested on real hardware. If you run it on a 16 GB M4 Mac Mini, please open an issue with:

- **Quant used** and whether it loaded or OOM’d
- **Largest `--ctx-size` that decoded** (not just loaded), with KV types and batch
- **`iogpu.wired_limit_mb`** value needed
- **Peak memory** (startup log + Activity Monitor)
- **Speed** (prefill / decode tok/s)
- **Subjective quality** at IQ2 vs Gemma for your tasks
- **llama-cpp-turboquant commit** built

**Last Updated:** July 2026
