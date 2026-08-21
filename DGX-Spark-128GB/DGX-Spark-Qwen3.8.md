# DGX Spark Founders Edition (128 GB) - Qwen3.8-27B

> ⚠️ **Not yet tested** on this hardware with Qwen3.8. Port of the **tested** [Qwen3.6 DGX Spark](DGX-Spark-Qwen3.6.md) pin (Q6 @ 262k · q8/turbo4). Confirm load → first decode → Pi tools, then report via issue/PR.
>
> CUDA Qwen3.8 twin (✅ tested): [Dual RTX 6000 Qwen3.8](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md) — MTP, sampling, thinking, vision live there.

CUDA CC **12.1** (GB10) · llama-cpp-turboquant. Same box, same agent-scale pin as 3.6; only the weights file and arch tag (`qwen35`) change.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (ported from tested 3.6 on this box) |
| **Weights** | `Qwen3.8-27B-UD-Q6_K_XL.gguf` (~25.9 GB) |
| **Catalog** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` K / **turbo4** V |
| **Output** | `--n-predict 8192` (raise toward 16384 + Pi if reports truncate) |
| **Sampling** | temp **0.65** · top_p **0.90** · repeat **1.10** (this box’s tested 3.6 command) |
| **Thinking** | `--reasoning off` (Pi tools) |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant` |

**Pi:** `contextWindow` = 262144, `maxTokens` = 8192. Path-heavy tools misbehaving? Switch to the Dual RTX / 4090 agent profile (temp 0.6 / top_p 0.95 / top_k 20 / presence 0) on a **new** session. [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). GGUF names: [local-setup](../local-setup.md#understanding-gguf-quants-why-so-many-files).

Need the engine built first? [local-setup.md](../local-setup.md).

## Download

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

Fresh turboquant build (arch `qwen35`). Qwen3.8 Q6 is **~25.9 GB** vs ~22 GB listed for 3.6 Q6 on this box — still comfortable on 128 GB. Optional max-fidelity: `UD-Q8_K_XL` (~31.5 GB), same command, swap `--model`. Ladder: [local-setup](../local-setup.md#q8-vs-q6-vs-q5-vs-q4-quality-vs-speed).

## Build

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121"
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). GB10 = CC 12.1. Confirm turbo types: `./llama-server --help | grep -A2 cache-type-v`.

## PRIMARY command

Port of the DGX Spark Qwen3.6 tested baseline. Run from `build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v turbo4 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 28 --temp 0.65 --top-p 0.90 \
  --n-predict 8192 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 262144` | Full native window; stable on this box with turbo V on 3.6 |
| `q8_0` K / **turbo4** V | CUDA quality-leaning turbo V (not Metal turbo2); keep K precise |
| `--load-mode none` | Buffered read; needs enough **system RAM** for the GGUF during load |
| `--threads 28` | Spark host CPU pairing from the 3.6 guide |
| Sampling | Matches tested 3.6 on this box — not Unsloth’s table |
| `--n-predict 8192` | Matches tested DGX 3.6 + Pi. Thin if you later turn thinking **on** — raise with `maxTokens` |

`--ctx-size` is a request: confirm `n_ctx_seq (262144)` and `load_mode = none` in the log. 3.6 ballpark on Spark: ~45–65 t/s prefill, 90–120+ t/s decode — re-bench 3.8.

### Confirm

```text
log: n_ctx_seq (262144)
log: load_mode = none
# model loads as qwen35 / Qwen3.8-27B
```

Then: (1) load, (2) short decode, (3) new Pi session with real `ls` / `read`.

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
          "name": "Qwen3.8-27B Q6_K_XL (262k q8/turbo4) - DGX Spark",
          "contextWindow": 262144,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

## This box

**Optional q8/q8** if tools feel soft and memory allows (Dual RTX primary): `--cache-type-v q8_0` instead of turbo4. Need more capacity: keep K at `q8_0`, step V turbo4 → turbo3 → turbo2.

**Optional Q8 weights:** `Qwen3.8-27B-UD-Q8_K_XL.gguf` (~31.5 GB), same command, swap `--model` and Pi `name`.

## Qwen3.8 optionals

MTP, Unsloth sampling, thinking / preserve, vision: **[Dual RTX Qwen3.8 — optionals](../Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md#qwen38-optionals)**. On CUDA, MTP is the first speed lever (`--spec-type draft-mtp --spec-draft-n-max 2`); smoke-test Pi tools after.

## See also

- Twin 3.6 (tested): [DGX-Spark-Qwen3.6.md](DGX-Spark-Qwen3.6.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-08-20 (recipe overlay on tested DGX 3.6; 3.8 still ⚠️ untested)
