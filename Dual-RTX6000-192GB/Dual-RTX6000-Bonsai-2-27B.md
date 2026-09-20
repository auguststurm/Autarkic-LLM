# Dual RTX 6000 Pro Max-Q (192 GB) - Ternary Bonsai 2 27B

> ⚠️ **Not yet tested** on this hardware with Bonsai 2 (researched **2026-09-18**). Confirm load → first decode → Pi tools before relying on it. This is a **different engine** from the rest of this folder: **not** llama-cpp-turboquant.

Blackwell **sm_120** · [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, **prism-b10658+**) · Ubuntu. Qwen3.8-27B hybrid backbone packed as true ternary weights (~6–7 GB). Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Stock llama.cpp and llama-cpp-turboquant will not run these files.** They refuse `PQ2_0` / `PTQ1_0` as unknown types. Do not load a Bonsai 2 `Q2_0` on a stock build either — that file can load silently and emit garbage (Hadamard runtime missing). Use the PrismML fork.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (PrismML card + this box’s Dual RTX CUDA pin) |
| **PRIMARY weights** | `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.2 GB, 2.13 bpw) |
| **A/B weights** | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.9 GB, 1.75 bpw) |
| **Catalog** | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` · single GPU (`--main-gpu 0`) |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` (Pi tools). Model thinks by default if you leave it on |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama.cpp-prism` |

Need CUDA prereqs? [local-setup.md](../local-setup.md). This page’s cmake is the PrismML fork, not the turboquant clone. GGUF names (`PQ2_0` vs `PTQ1_0`): [local-setup](../local-setup.md#model-catalog-hugging-face).

## Download

Both packs. Same ternary g128 weights; different on-disk packing. Skip `Ternary-Bonsai-2-27B-F16.gguf` (~54 GB) unless you want an FP16 baseline.

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf \
  Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Confirm both files exist under `~/Documents/AIML/models` before building.

**Why two files, and which first.** On this Blackwell box, start with **PQ2_0**, then swap to **PTQ1_0** and [bench both](#try-both-packs). Keep whichever wins here.

| Pack | Size | Role on this box |
| --- | --- | --- |
| **PQ2_0** | ~7.2 GB | **PRIMARY** — cheaper unpack; expected faster PP; demo default |
| **PTQ1_0** | ~5.9 GB | **A/B** — denser trits; try second and keep whichever wins here |

## Build

Separate clone so the Dual RTX turboquant tree stays untouched. `prism-b10658` or newer.

```bash
cd ~/Documents/GitHub
git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git llama.cpp-prism
cd llama.cpp-prism
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j$(nproc)
cd bin
```

Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) branch **`prism`**. `"120"` is GPU compute capability (`sm_120`), not the CUDA toolkit version. PrismML’s fat-binary script uses `120a`; this box pins `"120"`. Omit `-DCMAKE_CUDA_ARCHITECTURES="120"` to autodetect.

Confirm before debugging flags:

```bash
./llama-server --version          # prism-b10658 or higher (releases have been prism-b10687+)
./llama-server --help | grep -E 'PTQ1|PQ2' || true
```

If load later fails with unknown type `PTQ1_0` / `PQ2_0`, you launched **turboquant** or stock llama.cpp. Check `pwd` is `~/Documents/GitHub/llama.cpp-prism/build/bin`.

**Optional shortcut (prebuilt):** [PrismML CUDA 12.8 Linux x64](https://github.com/PrismML-Eng/llama.cpp/releases/latest) (`llama-prism-*-bin-linux-cuda-12.8-x64.tar.gz`). Native `sm_120` is more reliable from the source build above on this Blackwell card.

## PRIMARY command

Research baseline — Dual RTX CUDA pin + Bonsai 2 **PQ2_0**. No mmproj. Thinking **off** for Pi. Run from `llama.cpp-prism/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --alias bonsai-2-27b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 32 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values (this box + this model)

| Flag | Why |
| --- | --- |
| `--model …-PQ2_0.gguf` | Blackwell-leaning pack; swap the filename to try PTQ1_0 |
| `--ctx-size 262144` | Native Qwen3.8 window. ~7 GB weights + hybrid KV fit one 96 GB card easily |
| `q8_0` / `q8_0` | This fork has **no** `turbo*` cache types |
| `--cache-ram 0` | Same hybrid Qwen / DeltaNet multi-turn issue as 3.8 ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| `--main-gpu 0` | Single-GPU primary; second 96 GB stays free (or holds the Q8 host — [below](#this-box)) |
| `--reasoning off` | Pi needs `message.content` / tools. Bonsai 2 **thinks by default** if you leave this on |
| Sampling | Pi tools pin — **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)). Official thinking/instruct rows: [optionals](#bonsai-2-optionals) |
| `--n-predict 16384` | Match Pi `maxTokens`. Thinking tokens also count against this if you turn thinking on |
| `--jinja` | Native OpenAI-style `tool_calls` (PrismML 27B server profile) |
| `--alias bonsai-2-27b` | Matches the Pi JSON `id` |

Universal flags (`--fit off`, loopback, no checkpoints): [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) still describes them; this binary is the **PrismML** fork, so ignore turbo V / TQ weight types.

### Confirm

`--ctx-size 262144` is a request. Trust the load log:

```text
log: n_ctx_seq (262144)
log: load_mode = none
# model loads; types are PQ2_0 (or PTQ1_0), not a refused unknown type
nvidia-smi   # MiB after load and after a short decode — expect ~16–20 GB class, not Q8’s ~40
```

Then: (1) load, (2) first decode, (3) new Pi session with real `ls` / `read`.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-2-27b",
       "messages":[{"role":"user","content":"Reply with exactly: BONSAI_OK"}]}'
```

## Try both packs

Same command. Only `--model` changes. `pkill` between runs.

```bash
# Same PRIMARY, only:
#   --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

From `build/bin`, numbers PrismML reports as `tg128` / `pp512`:

```bash
./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128

./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128
```

Keep the faster pack for interactive Pi; keep the other on disk. Quality should be close (same ternary assignment; packing differs). Smoke Pi `ls`/`read` on **each** before treating a tok/s winner as the daily driver.

Re-bench on this box. Q8 27B on this hardware is ~51 tok/s — Bonsai 2 is the speed/footprint experiment, not a quality upgrade over Q8.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi so the status bar matches. Same JSON for either pack (one `llama-server` at a time).

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "bonsai-2-27b",
          "name": "Ternary Bonsai 2 27B PQ2_0 (262k q8/q8) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

If you settle on PTQ1_0, change `name` only. `--alias` must still match `id`.

## This box

**Single GPU on purpose.** ~6–7 GB weights + 262k q8/q8 hybrid KV is a ~16–20 GB object on a 96 GB card. Do not `--split-mode layer` this model.

This is **not** a replacement for the [Q8 27B Pi primary](Dual-RTX6000-Qwen3.8.md) (tested 2026-08-14). Bonsai 2 is the same architecture at ~9× smaller language weights (PrismML: 98.2% of Qwen3.8-27B FP16 on their thinking suite). Go back to Q8 if tool quality drops.

**Two cards.** You can park Bonsai 2 on GPU 0 while the Q8 primary stays on GPU 1 — two **different** `llama-server` binaries (`llama.cpp-prism` vs `llama-cpp-turboquant`), `CUDA_VISIBLE_DEVICES`, different ports. Isolation pattern: [Localmaxing](Dual-RTX6000-Qwen3.8-localmaxing.md). Do not load these GGUFs in turboquant.

**Optional: both packs at once** (A/B without restarting). GPU 1 first (`:8081` PQ2_0), then GPU 0 (`:8080` PTQ1_0). `--threads 12` with two processes. `--alias` unique per port.

```bash
# GPU 1 (display) — start first
cd ~/Documents/GitHub/llama.cpp-prism/build/bin
CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --alias bonsai-2-27b-pq2 --port 8081 --threads 12 \
  --host 127.0.0.1 --ctx-size 262144 --fit off \
  --n-gpu-layers 99 --main-gpu 0 --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
  --jinja --flash-attn on --no-context-shift --parallel 1 \
  --ubatch-size 1024 --batch-size 1024 \
  --reasoning off --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --frequency-penalty 0.0 --repeat-last-n 64 \
  --n-predict 16384 --kv-unified --log-verbosity 1
```

```bash
# GPU 0
cd ~/Documents/GitHub/llama.cpp-prism/build/bin
CUDA_VISIBLE_DEVICES=0 ./llama-server \
  --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --alias bonsai-2-27b-ptq1 --port 8080 --threads 12 \
  --host 127.0.0.1 --ctx-size 262144 --fit off \
  --n-gpu-layers 99 --main-gpu 0 --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
  --jinja --flash-attn on --no-context-shift --parallel 1 \
  --ubatch-size 1024 --batch-size 1024 \
  --reasoning off --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --frequency-penalty 0.0 --repeat-last-n 64 \
  --n-predict 16384 --kv-unified --log-verbosity 1
```

```bash
for p in 8080 8081; do echo -n ":$p "; curl -s http://127.0.0.1:$p/health || echo down; echo; done
nvidia-smi
```

Two Pi provider keys (`llama-cpp-8080` / `llama-cpp-8081`) if you want `/model` hop — copy the Localmaxing JSON shape and set `id` to the `--alias` values above.

## Bonsai 2 optionals

### Sampling (leave Pi for these)

Keep the pin’s Pi row for tool/coding agents. PrismML / Qwen3.8 `generation_config` when you are **not** in path-heavy tool loops:

| Mode | temp | top_p | presence | Notes |
| --- | --- | --- | --- | --- |
| Thinking | **1.0** | **0.95** | 0.0 | Bonsai 2 demo default; `--reasoning on` |
| Instruct (non-thinking) | 0.7 | 0.80 | **1.5** | Chat only — presence 1.5 warps reused paths in Pi |
| **This repo’s Pi tools** | **0.6** | **0.95** | **0.0** | This box’s agent pin |

### Thinking on (not the Pi default)

Primary stays `--reasoning off`. The model **reasons by default** if you omit that. If you turn thinking **on**:

- Default effort is **`xhigh`**. Use **`medium`** for shorter traces. PrismML: **`low` is not supported** and behaves close to `xhigh` — do not use `low` as a speed knob. Bound length with `--reasoning-budget N` or a per-request thinking budget.
- **Thinking tokens count against `--n-predict`.** XHIGH can fill the page and truncate the answer. Raise `--n-predict` / Pi `maxTokens` together if traces cut off.
- **Leave `--reasoning-preserve` off** for Pi and long sessions.

```bash
# Deltas only — not the Pi primary:
#   --reasoning on
#   --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.0
#   --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

### Vision (`mmproj`)

Not required for Pi text/agent work. Demo uses the Q8_0 pack.

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Add `--mmproj ~/Documents/AIML/models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. BF16 reference: `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` (~0.93 GB). Images bill as prompt tokens.

### Other stacks

[prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) is Apple Silicon MLX, not this CUDA box. Demo wrapper scripts: [PrismML-Eng/Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) (source of truth if this page and their docs disagree). Hadamard runtime is [not upstream yet](https://github.com/ggml-org/llama.cpp/pull/27779).

## See also

- Tested Q8 Pi primary: [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-Qwen3.8.md)
- Two cards, one model each: [Dual-RTX6000-Qwen3.8-localmaxing.md](Dual-RTX6000-Qwen3.8-localmaxing.md)
- Model card: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) · docs: [Ternary Bonsai 2 27B](https://docs.prismml.com/bonsai-2-27b)
- Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) · Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-09-20 (recipe density; ⚠️ untested on this box)
