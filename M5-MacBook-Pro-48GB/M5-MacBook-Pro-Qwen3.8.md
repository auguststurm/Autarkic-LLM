# M5 MacBook Pro (48 GB) - Qwen3.8-27B

> ⚠️ **Not yet tested** on this hardware with Qwen3.8. Port of the **tested** [Qwen3.6 M5](M5-MacBook-Pro-Qwen3.6.md) pin (Q5 @ 196k · q8/q8 Metal). Confirm load → **first decode** (Metal can load then OOM) → Pi tools, then report via issue/PR.
>
> **Metal, not CUDA.** Do not paste Dual RTX / DGX cmake or `--n-gpu-layers` here. Qwen3.8 extras (MTP, sampling, thinking, vision): [Dual RTX Qwen3.8](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md). Tighter Metal pattern: [M4 Air](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).

48 GB unified · llama-cpp-turboquant **Metal**. Same box, same agent-scale pin as 3.6; only the weights file and arch tag (`qwen35`) change.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (ported from tested 3.6 Metal on this box) |
| **Weights** | `Qwen3.8-27B-UD-Q5_K_XL.gguf` (~20.2 GB) |
| **Catalog** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| **Context** | `--ctx-size 196608` (`--fit off`) — not full 262k train length |
| **KV** | `q8_0` / `q8_0` (no turbo required on 48 GB) |
| **Output** | `--n-predict 8192` (raise toward 16384 + Pi if reports truncate) |
| **Sampling** | temp **0.65** · top_p **0.90** · repeat **1.10** (this box’s tested 3.6 command) |
| **Thinking** | `--reasoning off` (Pi tools) |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant` |

**Pi:** `contextWindow` = 196608, `maxTokens` = 8192. Path-heavy tools misbehaving? Agent profile temp 0.6 / top_p 0.95 / top_k 20 / presence 0 on a **new** session. [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). GGUF names: [local-setup](../local-setup.md#understanding-gguf-quants-why-so-many-files).

**Memory:** Q5 ~20 GB leaves room for macOS + KV at 196k. Q8 (~31.5 GB) fights the laptop budget at this pin. Air *needs* turbo2 V on 24 GB; here q8/q8 was the tested 3.6 quality baseline. Close heavy apps; one long-lived `llama-server`.

Need the engine built first? [local-setup.md](../local-setup.md).

## Download

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

Fresh **Metal** turboquant build (arch `qwen35`). Optional step-up: `UD-Q6_K_XL` (~25.9 GB) if decode stays happy with apps closed. Ladder: [local-setup](../local-setup.md#q8-vs-q6-vs-q5-vs-q4-quality-vs-speed).

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

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). Tip with Metal turbo4 `rnorm` fix (`b01afefed` / [PR #200](https://github.com/TheTom/llama-cpp-turboquant/pull/200) or later) — no manual shader edit. Optional turbo: `./llama-server --help | grep -A2 cache-type-v`.

## PRIMARY command

Port of the M5 Qwen3.6 tested Metal baseline. Run from `build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q5_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 196608 \
  --fit off \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
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

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 196608` | Agent window verified on this hardware with Q5 on 3.6 (`n_ctx_seq` < 262k train is expected) |
| `q8_0` / `q8_0` | Quality baseline at 196k — room enough without turbo V |
| Batch 512 / 512 | Comfortable on 48 GB; drop if Metal-OOM |
| `--threads 0` | Let the runtime pick host threads on Apple Silicon |
| No `--n-gpu-layers` | Metal unified memory — do not paste CUDA flags |
| Sampling | Matches tested 3.6 on this box |
| `--n-predict 8192` | Matches tested M5 3.6 + Pi. Thin if you later turn thinking **on** |

### Confirm

```text
log: n_ctx_seq (196608)
# model loads as qwen35 / Qwen3.8-27B
```

Then: (1) load, (2) **first decode**, (3) new Pi session with real `ls` / `read`.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B Q5_K_XL (196k q8/q8) - M5 Pro",
          "contextWindow": 196608,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

## This box

**Metal-OOM:** (1) close apps, (2) drop batch to 256 or 128, (3) drop ctx (131072 or 65536) — never bare `--fit on`, (4) optional turbo V (`--cache-type-v turbo4`, then turbo3 / turbo2). Verify quality after turbo on Metal: [TurboQuant notes](../llama-cpp-turboquant.md#2-turboquant-kv-cache).

**Optional Q6:** `Qwen3.8-27B-UD-Q6_K_XL.gguf` (~25.9 GB), same command, swap `--model`. Confirm **decode** at 196k.

No mid-size Qwen3.8 MoE twin yet — stay on dense 27B.

## Qwen3.8 optionals

MTP (smaller Metal gain than CUDA), Unsloth sampling, thinking / preserve, vision: **[Dual RTX Qwen3.8 — optionals](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md#qwen38-optionals)**.

## See also

- Twin 3.6 (tested): [M5-MacBook-Pro-Qwen3.6.md](M5-MacBook-Pro-Qwen3.6.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-08-20 (recipe overlay on tested M5 3.6 Metal; 3.8 still ⚠️ untested)
