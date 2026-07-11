# M4 MacBook Air (24 GB) - Qwen3.6-35B-A3B

Optimized setup for MacBook Air M4 with 24 GB unified memory, using **llama-cpp-turboquant** (TurboQuant KV) to maximize context.

> ✅ **Tested by the maintainer on this hardware** (July 2026, TheTom `feature/turboquant-kv-cache` @ `c3e6dbb13`+). Settings below come from real decode runs (including a clean reboot re-check); expect variance with thermals and background load.

## Why TurboQuant matters here

This guide is built around the **TurboQuant fork**, not stock llama.cpp. On 24 GB the weights already eat most of unified memory; **long context only fits if the KV cache is compressed**.

| Tier | ~bits / value | Role on this Mac |
| --- | --- | --- |
| `turbo4` | ~4.25 | Mildest; good quality; fewer tokens of context |
| `turbo3` | ~3.25 | Fork “sweet spot” on bigger GPUs — **does not free enough** for 61k on this Air |
| **`turbo2`** | ~2.0 | **Recommended V-cache here** — most compression; enables max verified context |

Asymmetric config used below: **K = `q8_0`** (quality), **V = `turbo2`** (memory). That is the standard TurboQuant pattern; the fork may also auto-upgrade K if you request turbo K on high-GQA models (`TURBO_AUTO_ASYMMETRIC`).

**Requires** `--flash-attn on` (quantized KV needs FA kernels). **Requires** this fork — upstream llama.cpp rejects `turbo2`/`turbo3`/`turbo4`.

## Memory reality (read this)

- **Model train context:** `n_ctx_train = 262144` (262k). Full train window is **not** feasible on 24 GB with this weight size (weights + OS + compute still dominate).
- **Weights:** use **UD-IQ4_NL (~18 GB)**. **UD-Q4_K_XL (~22 GB)** Metal-OOMs on first decode on this machine with current builds — do not use it here.
- **Largest context verified (decode + short chat):**

| `--ctx-size` | KV | Batch | Result |
| --- | --- | --- | --- |
| 49152 | `q8_0` / `turbo4` | 128 | OK (conservative) |
| **61440** | **`q8_0` / `turbo2`** | **64** | **OK — recommended** |
| 61440 | `q8_0` / `turbo3` | 64 | Metal OOM |
| 61440 | `q8_0` / `turbo2` | 128 | Metal OOM (batch too large) |
| 65536 | any turbo tier tried | ≤16 | Metal OOM on **first decode** (process may still report `n_ctx=65536` after load) |

So TurboQuant is what gets you from “tiny fit context” to **~61k** (~23% of train length) — not 262k, but the max this box has proven stable. Clean reboot does **not** unlock 65k; the wall is peak Metal working set at graph compute (weights + hybrid/compute scratch + KV), not “forgot to use turbo.”

Close heavy apps before launch. Prefer **one** long-lived `llama-server` process — avoid rapid stop/start cycles on 24 GB unified memory (each load spikes Metal residency and can thrash the whole machine).

## Pi Coding Agent: read this first

Pi needs a large `contextWindow`. Match it to the server’s real `n_ctx_seq`.

**Always pin `--ctx-size` and set `--fit off`.** Default `--fit on` can crush context toward ~4096 and break Pi.

**Thinking / empty replies:** disable thinking so Pi gets `message.content`:

- `--reasoning off`
- `--chat-template-kwargs '{"enable_thinking":false}'`

## Recommended model

- **Model:** `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` (~18 GB)
- **Tested path:** `~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
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

## Optimized llama-server command (IQ4_NL + TurboQuant @ 61k)

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 61440 \
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
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 61440` | Max context verified with TurboQuant on this hardware |
| `--cache-type-k q8_0 --cache-type-v turbo2` | **TurboQuant working hard on V** so 61k fits; K stays higher precision |
| `--flash-attn on` | Required for turbo KV types |
| `--ubatch-size` / `--batch-size` **64** | 128 peak-OOMs at 61k; 64 verified |
| `--fit off` | Do not let fit shrink the context you paid for with turbo |
| Thinking off | Agent-friendly non-thinking Qwen3.6 |
| `--host 127.0.0.1` | Local-only default (use `0.0.0.0` only if you intend LAN exposure — no auth) |
| No checkpoint flags | Qwen3.6 hybrid attention often won’t restore checkpoints usefully — see [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

Confirm **`n_ctx` / `n_ctx_seq (61440)`** in the log or `GET /v1/models`.

### Fallbacks if you Metal-OOM

1. Close other apps (browsers, IDEs, other local servers).  
2. Drop batch to `32`.  
3. Drop context to `49152` and optionally relax V to `turbo4` (higher quality KV, less context).  
4. Do **not** switch to Q4_K_XL on this machine.

### Conservative alternate (more KV quality, less context)

```bash
# Same flags as above, but:
#   --ctx-size 49152
#   --cache-type-k q8_0 --cache-type-v turbo4
#   --ubatch-size 128 --batch-size 128
```

## Not recommended on this hardware: Q4_K_XL

`Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` (~22 GB) exceeds practical Metal working set here and fails first decode even at small context with current builds. TurboQuant compresses **KV**, not weights — it cannot fix an oversized quant. Use IQ4_NL.

Older “it worked if only two terminals were open” runs were oversubscribed and thrashy; that is not a supported config on this guide.

## Performance notes

- MoE keeps active compute modest; **turbo2 V costs some decode speed** vs `q8_0`/`turbo4` — that is the trade for ~61k context.
- `n_ctx_seq (61440) < n_ctx_train (262144)` is expected.
- Watch logs for `kIOGPUCommandBufferCallbackErrorOutOfMemory` and `recommended max working set`.
- After rebuilds, re-check actual `n_ctx` and keep Pi’s `contextWindow` in sync.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## Pi Coding Agent `models.json`

Save this **entire** file to `~/.pi/agent/models.json` (copy-paste as-is — do not assemble a wrapper). Create parent dirs if needed: `mkdir -p ~/.pi/agent`.

`maxTokens` ≤ `--n-predict` (8192). `contextWindow` = `--ctx-size`.

**IQ4_NL + TurboQuant @ 61k (recommended):**

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
          "name": "Qwen3.6-35B-A3B IQ4_NL turbo2 (61k) - M4 Air",
          "contextWindow": 61440,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

**Fallback @ 48k / turbo4:**

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
          "name": "Qwen3.6-35B-A3B IQ4_NL turbo4 (48k) - M4 Air",
          "contextWindow": 49152,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```


**Last Updated:** July 2026
