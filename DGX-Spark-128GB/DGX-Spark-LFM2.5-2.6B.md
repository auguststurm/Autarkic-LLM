# DGX Spark Founders Edition (128 GB) - LFM2.5-2.6B

> ⚠️ **Not yet tested** on this hardware with LFM2.5. GB10 cmake (`"121"`, FA **on**), Q8_0, Liquid sampling. Confirm load → first decode → Pi tools, then report via issue/PR.

CUDA CC **12.1** (GB10) · llama-cpp-turboquant. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (cmake from tested 3.6 on this box) |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · Pi `contextWindow` **131072** |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 16384` · Pi `maxTokens` **16384** |
| **Sampling** | temp **0.1** · top_k **50** · repeat **1.1** (Liquid card) |
| **Thinking** | Omit server `--reasoning off`. **Pi skills/tools:** `reasoning` **false**, no `thinkingLevelMap`. **Pi traces:** `reasoning` **true**, `thinkingLevelMap.off` **null** |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need the engine? [local-setup.md](../local-setup.md). Reuse this folder’s [Qwen3.6](DGX-Spark-Qwen3.6.md) binary if it was built with the cmake below.

## Download

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Optional F16 (5.4 GB): same command, swap `--model` to `LFM2.5-2.6B-F16.gguf`. Skip Q6_K (not needed on 128 GB).

## Build

Same cmake as [Qwen3.6 on this box](DGX-Spark-Qwen3.6.md#build). GB10 = CC 12.1. ARM host: build from source (`uname -m` is `aarch64`).

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="121"
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). Do not launch the [PrismML](DGX-Spark-Bonsai-2-27B.md) binary for this GGUF.

## PRIMARY command

Run from `build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  --alias lfm2.5-2.6b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --repeat-penalty 1.1 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 28 --temp 0.1 --top-k 50 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

Omit `--reasoning off` — the [chat template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja) always opens `<think>`. Omit `--cache-ram 0` and `--main-gpu`.

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 131072` | Native train length. KV is **1,088 MiB** q8/q8 at 128k |
| `q8_0` / `q8_0` | Only 8 of 30 layers are GQA |
| `--flash-attn on` | This box’s CUDA pin |
| `--load-mode none` | This box’s CUDA pin |
| `--threads 28` | Spark host CPU pairing |
| Batch 1024 | Drop to 256 if prefill OOMs, then `--ctx-size 65536` |
| `--n-predict 16384` | Pi default. Thinking counts against the cap. Do not drop to 4096 |
| Sampling | Liquid card: `temp 0.1` / `top_k 50` / `repeat-penalty 1.1` |

### Confirm

```text
log: n_ctx_seq (131072)
log: load_mode = none
log: kv_cache_init ≈ 1088 MiB
```

Then: (1) load, (2) short decode, (3) new Pi session with real `ls` / `read`.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Thinking lands in `reasoning_content`, answer in `content`. Restart **both** llama-server and Pi after pin or JSON changes. Status bar must match `131072` / `16384`.

## Pi Coding Agent `models.json`

Save **one** of these to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). `id` matches `--alias`. `/model` to reload; `/new` after a pin or reasoning-shape change.

### Skills / tools (start here)

`reasoning` **false**, no `thinkingLevelMap`. Does not strip `<think>` from the GGUF.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "lfm2.5-2.6b",
          "name": "LFM2.5-2.6B Q8_0 (128k, tools) - DGX Spark",
          "reasoning": false,
          "contextWindow": 131072,
          "maxTokens": 16384,
          "compat": {
            "supportsReasoningEffort": false
          }
        }
      ]
    }
  }
}
```

### Thinking on

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "lfm2.5-2.6b",
          "name": "LFM2.5-2.6B Q8_0 (128k, think) - DGX Spark",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null
          },
          "contextWindow": 131072,
          "maxTokens": 16384,
          "compat": {
            "supportsReasoningEffort": false
          }
        }
      ]
    }
  }
}
```

Empty `content` with `finish_reason: length` → tools JSON first, not a higher cap. Do not add server `--reasoning off`.

## This box

**Half-window:** `--ctx-size 65536` and Pi `contextWindow` 65536.

**DSpark** (Liquid draft sidecar, not this hardware): only if `./llama-server --help | grep -i dspark` shows `draft-dspark`. Download `LFM2.5-2.6B-DSpark-Q8_0.gguf` from [LFM2.5-2.6B-DSpark-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF), then `--model-draft … --spec-type draft-dspark --spec-draft-n-max 10 --spec-draft-n-min 0`. Skip if the flag is missing.

## See also

- This box, Qwen3.6 (✅ tested): [DGX-Spark-Qwen3.6.md](DGX-Spark-Qwen3.6.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent)

**Last Updated:** 2026-09-20 (GB10 `"121"`, 128k q8/q8 FA on, ⚠️ untested)
