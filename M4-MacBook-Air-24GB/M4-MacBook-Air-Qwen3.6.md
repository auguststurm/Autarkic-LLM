# M4 MacBook Air (24 GB) - Qwen3.6-35B-A3B

> ✅ **Tested** on this hardware (July 2026, TheTom `feature/turboquant-kv-cache` @ `c3e6dbb13`+), including a clean-reboot decode re-check.

24 GB unified · **Metal** · llama-cpp-turboquant. Weights eat most of RAM; **long context only fits if V is compressed**. Upstream llama.cpp rejects `turbo*`. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Turbo tiers: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md#2-turboquant-kv-cache).

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested (decode + short chat) |
| **Weights** | `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` (~18 GB) — **not** Q4_K_XL (~22 GB) |
| **Catalog** | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) |
| **Context** | `--ctx-size 61440` (`--fit off`) |
| **KV** | `q8_0` K / **turbo2** V |
| **Output** | `--n-predict 8192` |
| **Sampling** | temp **0.65** · top_p **0.90** · repeat **1.10** |
| **Thinking** | `--reasoning off` |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

## Memory reality (this box)

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

Need the engine? [local-setup.md](../local-setup.md).

## Download

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --local-dir ~/Documents/AIML/models
```

## Build

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). Tip `b01afefed` / [PR #200](https://github.com/TheTom/llama-cpp-turboquant/pull/200)+ for Metal turbo4 `rnorm`. Confirm: `./llama-server --help | grep -A2 cache-type-v`.

## PRIMARY command

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

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. `contextWindow` = 61440, `maxTokens` = 8192.

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
