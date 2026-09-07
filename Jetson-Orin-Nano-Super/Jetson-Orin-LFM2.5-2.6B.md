# Nvidia Jetson Orin Nano Super - LFM2.5-2.6B

> ⚠️ **Not yet tested** on this hardware with LFM2.5-2.6B (researched **2026-09-07**). Confirm load → first decode → Pi tools before relying on it. Tested daily driver on this box: [Gemma 4 E2B](Jetson-Orin-Gemma4-E2B.md).

8 GB LPDDR5 (~7.3 Gi usable) · Ampere sm_**87** · llama-cpp-turboquant. **Paths:** `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant`. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent). If you OOM, drop `--ctx-size`, never bare `--fit on`.

The [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B) does **not** recommend this model for agentic coding.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (Liquid card + this box’s Gemma 8 GB pin) |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 16384` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 4096` (thinking counts against this) |
| **Sampling** | temp **0.1** · top_k **50** · repeat **1.1** (Liquid card) |
| **Thinking** | Template always opens `<think>`. Omit `--reasoning off` |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need the engine? [local-setup.md](../local-setup.md) (JetPack / CUDA).

## Download

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Q8_0 (2.87 GB) is smaller than the tested Gemma Q4_K_S (~3 GB) on this box. Q6_K (2.22 GB) is the [headroom swap](#q6_k-alternative) if Q8 does not load.

## Build

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="87" \
  -DGGML_CUDA_F16=ON \
  -DLLAMA_CURL=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). `FA_ALL_QUANTS` lengthens compile but covers quantized KV + flash-attn.

## PRIMARY command

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`. **MAXN SUPER** first: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  --alias lfm2.5-2.6b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 16384 \
  --fit off \
  --n-gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 64 \
  --batch-size 128 \
  --repeat-penalty 1.1 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 0 --temp 0.1 --top-k 50 \
  --n-predict 4096 \
  --kv-unified \
  --log-verbosity 1
```

Omit `--load-mode none` on this 8 GB box (default **mmap** so the OS can page the GGUF). Omit `--reasoning off` / `--reasoning-budget 0` — the chat template always starts the assistant with `<think>`.

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 16384` | Daily Pi window on 8 GB; same pin as tested Gemma on this box. 32k is a stretch after decode is stable |
| `--n-predict 4096` | Thinking counts against the output cap. Raise **both** this and Pi `maxTokens` if replies clip |
| `--ubatch-size` / `--batch-size` **64 / 128** | Prefill vs peak on 8 GB; 256 is the first OOM lever |
| `--cache-type-k/v q8_0` | Quality default; turbo V only if raising context OOMs |
| `--n-gpu-layers 99` | Full GPU offload |
| `--fit off` | Keep pinned context agent-visible |
| `--host 127.0.0.1` | Local-only default on an edge device |
| `--alias lfm2.5-2.6b` | Stable id for Pi / curl (matches `models.json`) |
| LFM sampling | `temp 0.1` / `top_k 50` / `repeat-penalty 1.1` — [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |

Confirm **`n_ctx` / `n_ctx_seq (16384)`** in the log or `GET /v1/models`, then one short decode (not only load).

Thinking should land in `reasoning_content`, answer in `content`:

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Then: new Pi session with real `ls` / `read`.

### Fallbacks if you OOM

1. Run **MAXN SUPER** and free other processes (`sudo nvpmodel -m 2 && sudo jetson_clocks`).
2. Keep batch at `64` / `128`.
3. Drop `--ctx-size` (and Pi `contextWindow`) to `8192` or `4096`.
4. Swap weights to **Q6_K** (2.22 GB) with the same flags — [below](#q6_k-alternative).
5. Optional: `--cache-type-v turbo4` if you need more context than quality at the KV.
6. Do **not** rely on bare `--fit on` for Pi — pin a smaller context instead.

Keep Pi `contextWindow` = `--ctx-size` and `maxTokens` ≤ `--n-predict`. Restart **both** the server and Pi after changing either side. Status bar must match the pin.

## Q6_K alternative

Same PRIMARY flags. Only the file changes.

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q6_K.gguf \
  --local-dir ~/Documents/AIML/models
```

```bash
  --model ~/Documents/AIML/models/LFM2.5-2.6B-Q6_K.gguf \
```

## Pi Coding Agent `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi.

`maxTokens` ≤ `--n-predict` (4096). `contextWindow` = `--ctx-size`.

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
          "name": "LFM2.5-2.6B Q8_0 (16k) - Jetson Orin Nano Super",
          "contextWindow": 16384,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```

## Performance notes

- Q8_0 is the quality pin (2.87 GB vs tested Gemma ~3 GB on this box). Q6_K is the headroom swap, not the first download.
- Run in **MAXN SUPER** power mode; monitor with `jtop`.
- Enable zram if `free -h` shows swap 0 — unified memory spikes on prefill will otherwise SIGKILL the server.
- Empty `content` with `finish_reason: length` → raise `--n-predict` / `maxTokens`, not `--reasoning off`.
- After rebuilds, re-check actual `n_ctx` and keep Pi’s `contextWindow` in sync.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

**Last Updated:** 2026-09-07
