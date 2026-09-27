# Windows RTX 4090 (24 GB) - LFM2.5-2.6B

> ⚠️ **Not yet tested** as a Pi daily driver on this SKU. Confirm load → short decode → Pi tools, then report via issue/PR.

**WSL2** (not native Windows) · CUDA sm_**89** · llama-cpp-turboquant. Paths on this box: **`~/AIML`** (models) · **`~/GitHub`** (engine) — WSL2 convention, not `~/Documents/AIML`. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent).

Model flags follow the [Forge Trinity LFM2.5](../Forge-Trinity/Forge-Trinity.md) recipe (Q8_0 @ 128k q8/q8, Liquid sampling, no `--reasoning`). Load, cmake, host, and threads follow this 4090 WSL2 pin.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested decode |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · Pi `contextWindow` **131072** |
| **KV** | `q8_0` / `q8_0` |
| **Load** | `--no-mmap` (this WSL2 box’s working pin) |
| **CMake** | `GGML_CUDA=ON` · `CMAKE_CUDA_ARCHITECTURES="89"` · `GGML_CUDA_F16=ON` · `GGML_CUDA_FA_ALL_QUANTS=ON` |
| **Output** | `--n-predict 16384` · Pi `maxTokens` **16384** |
| **Sampling** | temp **0.1** · top_k **50** · top_p **1.0** · repeat **1.1** (Liquid card) |
| **Thinking** | Template always opens `<think>`. Omit server `--reasoning`. **Pi skills/tools:** `reasoning` **false**. **Pi traces:** `reasoning` **true**, `thinkingLevelMap.off` **null** |
| **Paths** | `~/AIML/models` · `~/GitHub/llama-cpp-turboquant` (WSL2) |

Need the engine? [local-setup.md](../local-setup.md) (WSL2 + Ubuntu). Reuse this folder’s [Qwen3.6](Windows-RTX4090-Qwen3.6.md) binary if it was built with the cmake below. `pkill -9 llama-server` before this command — one GPU, one server.

## Download

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/AIML/models
```

## Build (Ada / sm_89)

Same cmake as [Qwen3.6 on this box](Windows-RTX4090-Qwen3.6.md). Skip this section when that binary already exists.

```bash
cd ~/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache && git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). `"89"` is Ada. `FA_ALL_QUANTS` is this box’s Qwen pin (the 3090 `ptxas` failure does not apply here).

## PRIMARY command

Run from `~/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

cd ~/GitHub/llama-cpp-turboquant/build/bin

./llama-server \
  --model ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  --alias lfm2.5-2.6b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --no-mmap \
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
  --threads 0 --temp 0.1 --top-k 50 --top-p 1.0 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

The [chat template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja) always opens `<think>`. `--jinja` selects that template. Omit `--reasoning off` / `--reasoning-budget 0`. Omit `--cache-ram 0`. Omit `--load-mode` — this box uses `--no-mmap` (do not combine the two).

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 131072` `--fit off` | Native train length, pinned. q8_0 KV for 8 KV layers at 128k is about **1,088 MiB** |
| `q8_0` / `q8_0` | Weights 2.87 GB + about 1.1 GB KV at 128k. Keep q8 V |
| `--flash-attn on` | This box’s CUDA pin |
| `--n-gpu-layers 99` `--main-gpu 0` | All layers on the WSL2 GPU, device 0 |
| `--no-mmap` | Buffered read. This WSL2 box’s working load flag. File is 2.87 GB — a few GB free **host RAM** is enough |
| Batch 1024 | Same physical batch as the Forge LFM recipe (16 GB card). Drop to **256**, then **128**, if prefill OOMs |
| `--threads 0` | The process picks the CPU thread count |
| `--n-predict 16384` | Pi reply cap. Think tokens count against it. 4096 ends the first Pi turn with `finish_reason: length` |
| Sampling | Liquid card: `temp 0.1` / `top_k 50` / `top_p 1.0` / `repeat-penalty 1.1`. Presence, frequency, and min-p are 0 |
| `--host 127.0.0.1` | Loopback on this machine |

`--ctx-size` is a request: confirm `n_ctx_seq (131072)`.

### Confirm

```text
log: n_ctx_seq (131072)
log: kv_cache_init ≈ 1088 MiB
nvidia-smi   # inside WSL; MiB after load and after a short decode
```

A `--no-mmap` deprecation line that mentions `--load-mode` is expected. Parser maps `--no-mmap` to load mode `none`. Do not add `--load-mode none` on the same command.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Thinking is `reasoning_content`. The answer is `content`. Then: new Pi session with real `ls` / `read`. Restart **both** llama-server and Pi after pin or JSON changes. Status bar must match `131072` / `16384`.

## Pi `models.json`

Save **one** of these to `~/.pi/agent/models.json` inside WSL (`mkdir -p ~/.pi/agent`). Pi on Windows uses `%USERPROFILE%\.pi\agent\models.json` and the same `baseUrl`. `id` matches `--alias`. `/model` reloads the file. `/new` after a pin or reasoning-shape change.

### Skills / tools

`reasoning` **false**. The template’s `<think>` prefix stays in the GGUF output.

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
          "name": "LFM2.5-2.6B Q8_0 (128k, tools) - RTX 4090",
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
          "name": "LFM2.5-2.6B Q8_0 (128k, think) - RTX 4090",
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

Empty `content` with `finish_reason: length` means the next session uses the tools JSON. The server cap stays 16384. Do not add server `--reasoning off`.

## This box

**OOM on load / first decode:** (1) close other GPU apps / check `nvidia-smi` **inside** WSL, (2) `--ubatch-size 256 --batch-size 256`, (3) `--ctx-size 65536` + Pi `contextWindow` 65536.

**Half-window:** `--ctx-size 65536` and Pi `contextWindow` 65536.

`--n-predict` is already 16384. Loopback `--host 127.0.0.1` is reachable from Windows programs on this machine.

## See also

- This box, 27B Qwen (✅ tested): [Windows-RTX4090-Qwen3.6.md](Windows-RTX4090-Qwen3.6.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent)
- LFM recipe source: [Forge-Trinity.md](../Forge-Trinity/Forge-Trinity.md)

**Last Updated:** 2026-09-26 (⚠️ untested; `--no-mmap`, Ada `"89"`, Liquid `top_p 1.0`)
