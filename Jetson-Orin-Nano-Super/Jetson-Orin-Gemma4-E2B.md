# Nvidia Jetson Orin Nano Super - Gemma 4 E2B

> ✅ **Tested** on this hardware (**2026-09-07**) with **Pi Coding Agent**. Q4_K_S @ 16k q8/q8 on 8 GB unified; drop `--ctx-size` if you OOM.

8 GB LPDDR5 (~7.3 Gi usable) · Ampere sm_**87** · llama-cpp-turboquant. **Paths:** `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant`. Pi: [agentic harnesses](../agentic-harnesses.md). If you OOM, drop `--ctx-size`, never bare `--fit on`.

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested 2026-09-07 (Pi, 16k q8/q8) |
| **Weights** | `gemma-4-E2B-it-Q4_K_S.gguf` (~3 GB) |
| **Catalog** | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF) |
| **Context** | `--ctx-size 16384` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 2048` |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need the engine? [local-setup.md](../local-setup.md) (JetPack / CUDA).

## Download

```bash
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_S.gguf \
  --local-dir ~/Documents/AIML/models
```

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
  --model ~/Documents/AIML/models/gemma-4-E2B-it-Q4_K_S.gguf \
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
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 0 --temp 0.75 --top-p 0.92 \
  --n-predict 2048 \
  --kv-unified \
  --log-verbosity 1
```

Omit `--load-mode none` on this 8 GB box (default **mmap** so the OS can page the GGUF).

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 16384` | Daily Pi window on 8 GB; 32k is a stretch after decode is stable |
| `--n-predict 2048` | One-reply cap. Raise **both** this and Pi `maxTokens` (e.g. 4096) if replies clip |
| `--ubatch-size` / `--batch-size` **64 / 128** | Prefill vs peak on 8 GB; 256 is the first OOM lever |
| `--cache-type-k/v q8_0` | Quality default; turbo V only if raising context OOMs |
| `--n-gpu-layers 99` | Full GPU offload |
| `--fit off` | Keep pinned context agent-visible |
| `--host 127.0.0.1` | Local-only default on an edge device |
| Gemma sampling | `temp 0.75` / `top-p 0.92` — repo Gemma baseline |

Confirm **`n_ctx` / `n_ctx_seq (16384)`** in the log or `GET /v1/models`, then one short decode (not only load).

### Fallbacks if you OOM

1. Run **MAXN SUPER** and free other processes (`sudo nvpmodel -m 2 && sudo jetson_clocks`).
2. Keep batch at `64` / `128`.
3. Drop `--ctx-size` (and Pi `contextWindow`) to `8192` or `4096`.
4. Optional: `--cache-type-v turbo4` if you need more context than quality at the KV.
5. Do **not** rely on bare `--fit on` for Pi — pin a smaller context instead.

Keep Pi `contextWindow` = `--ctx-size` and `maxTokens` ≤ `--n-predict`. Restart **both** the server and Pi after changing either side. Status bar must match the pin.

## Pi Coding Agent `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi.

`maxTokens` ≤ `--n-predict` (2048). `contextWindow` = `--ctx-size`.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "gemma-4-e2b",
          "name": "Gemma 4 E2B Q4_K_S (16k) - Jetson Orin Nano Super",
          "contextWindow": 16384,
          "maxTokens": 2048
        }
      ]
    }
  }
}
```

## Performance notes

- Q4_K_S is the sweet spot for the 8 GB memory limit.
- Run in **MAXN SUPER** power mode; monitor with `jtop`.
- Enable zram if `free -h` shows swap 0 — unified memory spikes on prefill will otherwise SIGKILL the server.
- After rebuilds, re-check actual `n_ctx` and keep Pi’s `contextWindow` in sync.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

**Last Updated:** 2026-09-07
