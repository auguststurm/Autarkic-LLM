# M4 Mac Mini (16 GB) - Gemma 4 E2B

Recommended lightweight multimodal setup for Mac Mini M4 with 16 GB unified memory, using **llama-cpp-turboquant**. Command shape and Pi integration follow the [M4 MacBook Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md); this model is small enough that TurboQuant V is optional, not required to fit.

> ⚠️ **Not yet tested on this hardware.** Best-effort starting config. At ~3 GB the model leaves comfortable headroom on 16 GB, so risk is low. If you run it, please report results via an issue. For a hard 35B MoE experiment on this box, see [M4-Mac-Mini-Qwen3.6.md](M4-Mac-Mini-Qwen3.6.md).

## Memory reality (read this)

- **Weights:** Gemma 4 E2B `Q4_K_S` is ~3 GB — easily fits with macOS overhead on 16 GB.
- **Context:** this guide pins **32768** with `--fit off`. You can often raise context further; if you Metal-OOM, drop `--ctx-size` rather than enabling bare `--fit on` (fit can crush agent context — lesson from the Air guide).
- **KV:** primary uses **`q8_0` / `q8_0`** for quality. Optional turbo V only if you push a much larger window.
- Close heavy apps before launch. Prefer **one** long-lived `llama-server` process.

## Pi Coding Agent: read this first

Pi needs a large `contextWindow`. Match it to the server’s real `n_ctx_seq`.

**Always pin `--ctx-size` and set `--fit off`.** Default `--fit on` can shrink context and break long-agent sessions.

## Recommended model

- **Model:** `gemma-4-E2B-it-Q4_K_S.gguf` (~3 GB)
- **Path:** `~/Documents/AIML/models/gemma-4-E2B-it-Q4_K_S.gguf`

```bash
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_S.gguf \
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

Confirm the binary accepts turbo types (optional path):

```bash
./llama-server --help | grep -A2 cache-type-v
# must list turbo2, turbo3, turbo4
```

> **Fork version:** tip that includes Metal turbo4 `rnorm` fix (**`b01afefed` / PR #200 content or later**). No manual Metal shader edit on current TheTom tip. Rebuild after `git pull`.

## Optimized llama-server command (Q4_K_S @ 32k)

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/gemma-4-E2B-it-Q4_K_S.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 \
  --fit off \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 256 \
  --batch-size 256 \
  --reasoning off \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 0 --temp 0.75 --top-p 0.92 \
  --n-predict 4096 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values

| Flag / value | Why |
| --- | --- |
| `--ctx-size 32768` | Solid agent window with lots of headroom on 16 GB |
| `--cache-type-k q8_0 --cache-type-v q8_0` | Quality default; model is tiny so turbo is optional |
| `--flash-attn on` | Fast path; required if you later use turbo* V |
| `--ubatch-size` / `--batch-size` **256** | Comfortable for this size; drop if OOM |
| `--fit off` | Keep the pinned context agent-visible |
| `--reasoning off` | Prefer direct content for agent harnesses |
| `--host 127.0.0.1` | Local-only default (use `0.0.0.0` only if you intend LAN exposure — no auth) |
| Gemma sampling | `temp 0.75` / `top-p 0.92` — Gemma baseline in this repo |

Confirm **`n_ctx` / `n_ctx_seq (32768)`** in the log or `GET /v1/models`.

### Fallbacks if you Metal-OOM

1. Close other apps.
2. Drop batch to `128` or `64`.
3. Drop `--ctx-size` (e.g. `16384`).
4. Do **not** rely on bare `--fit on` for Pi — pin a smaller context instead.

## Enable multimodal (optional)

Gemma 4 E2B is natively multimodal (text + image + audio), but llama-server only loads the vision/audio projector if you pass one:

```bash
hf download unsloth/gemma-4-E2B-it-GGUF \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models
```

Add to the `llama-server` command:

```bash
  --mmproj ~/Documents/AIML/models/mmproj-F16.gguf \
```

Without `--mmproj` the server runs text-only.

## Performance notes

- Extremely lightweight (~3 GB loaded): the recommended daily driver on 16 GB Minis.
- Multimodal once you pass `--mmproj`; text-only otherwise.
- Fast and responsive for agentic tasks compared with a tight 35B MoE experiment on the same box.
- After rebuilds, re-check actual `n_ctx` and keep Pi’s `contextWindow` in sync.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md). Pattern reference: [M4 Air guide](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).

## Pi Coding Agent `models.json`

Save this **entire** file as Pi’s `models.json` (copy-paste as-is — do not assemble a wrapper).

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
          "id": "gemma-4-e2b",
          "name": "Gemma 4 E2B Q4_K_S (32k) - M4 Mini",
          "contextWindow": 32768,
          "maxTokens": 4096
        }
      ]
    }
  }
}
```


**Last Updated:** July 2026
