# M5 MacBook Pro (48 GB) - Qwen3.6-27B

> ✅ **Tested** on this hardware. Qwen3.8 port (same Metal knobs, ⚠️ untested): [M5-MacBook-Pro-Qwen3.8.md](M5-MacBook-Pro-Qwen3.8.md). **Metal, not CUDA.** Tighter Metal: [M4 Air](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).

48 GB unified · llama-cpp-turboquant. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested |
| **Weights** | `Qwen3.6-27B-UD-Q5_K_XL.gguf` (~20 GB) |
| **Catalog** | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) |
| **Context** | `--ctx-size 196608` (`--fit off`) — not full 262k |
| **KV** | `q8_0` / `q8_0` (Air needs turbo2 on 24 GB; here q8/q8 fits) |
| **Output** | `--n-predict 8192` |
| **Sampling** | temp **0.65** · top_p **0.90** · repeat **1.10** |
| **Thinking** | `--reasoning off` |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

Close heavy apps; one long-lived server. Need the engine? [local-setup.md](../local-setup.md).

## Download

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q5_K_XL.gguf \
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

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). Metal turbo4 `rnorm` fix: tip `b01afefed` / [PR #200](https://github.com/TheTom/llama-cpp-turboquant/pull/200)+. Optional turbo: `./llama-server --help | grep -A2 cache-type-v`.

## PRIMARY command

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-27B-UD-Q5_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 196608 \
  --fit off \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 512 \
  --batch-size 512 \
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
| `--ctx-size 196608` | Large agent window verified on this hardware with Q5 |
| `--cache-type-k q8_0 --cache-type-v q8_0` | Quality baseline at 196k — room enough without turbo V |
| `--flash-attn on` | Fast path; required if you later switch V to turbo* |
| `--ubatch-size` / `--batch-size` **512** | Comfortable on 48 GB; drop if you Metal-OOM |
| `--fit off` | Do not let fit shrink the context you pinned |
| Thinking off | Agent-friendly non-thinking Qwen3.6 |
| `--host 127.0.0.1` | Local-only default (use `0.0.0.0` only if you intend LAN exposure — no auth) |
| No checkpoint flags | Qwen3.6 hybrid attention often won’t restore checkpoints usefully — see [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing) |

Confirm **`n_ctx` / `n_ctx_seq (196608)`** in the log or `GET /v1/models`.

### Fallbacks if you Metal-OOM

1. Close other apps (browsers, IDEs, other local servers).
2. Drop batch to `256` or `128`.
3. Drop context (e.g. `131072` or `65536`) rather than enabling bare `--fit on`.
4. Optional: keep context high and compress V with turbo (see below).

### Optional: more context via TurboQuant V

If you want headroom beyond 196k (or a heavier quant) and accept some decode cost:

```bash
# Same flags as above, but e.g.:
#   --ctx-size 229376   # try upward carefully; confirm decode, not just load
#   --cache-type-k q8_0 --cache-type-v turbo4   # safer first turbo step
#   # or turbo3 / turbo2 if still tight
```

Verify output quality after enabling turbo on Metal (see [TurboQuant notes](../llama-cpp-turboquant.md#2-turboquant-kv-cache)).

## Performance notes

- Excellent quality/speed balance on 48 GB unified memory for dense Q5.
- Strong agentic performance with Pi / Hermes and MCP workflows.
- Metal backend is efficient on Apple Silicon; thermals on sustained load still matter on a laptop.
- `n_ctx_seq (196608) < n_ctx_train (262144)` is expected.
- After rebuilds, re-check actual `n_ctx` and keep Pi’s `contextWindow` in sync.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Pattern reference: [M4 Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. `contextWindow` = 196608, `maxTokens` = 8192.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.6-27b",
          "name": "Qwen3.6-27B Q5_K_XL (196k) - M5 Pro",
          "contextWindow": 196608,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```


## Alternate: MoE upgrade (untested)

Primary config above is the **tested baseline**. For stronger agentic quality at similar weight size (~18 GB), try the same MoE quant proven on the Air:

- **Model:** `Qwen3.6-35B-A3B-UD-IQ4_NL.gguf` (~18 GB MoE)
- **Path:** `~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf`

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --local-dir ~/Documents/AIML/models
```

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-IQ4_NL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 65536 \
  --fit off \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --chat-template-kwargs '{"enable_thinking":false}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 512 \
  --batch-size 512 \
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

On 48 GB you can likely push MoE context well past 65k (the Air reaches ~61k with turbo2 on **24 GB**). Start at 65k with `q8_0`/`q8_0`, then raise `--ctx-size` or switch V to `turbo4`/`turbo2` if you need more. Confirm **decode**, not only load.

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
          "name": "Qwen3.6-35B-A3B IQ4_NL (64k) - M5 Pro",
          "contextWindow": 65536,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

**Last Updated:** July 2026
