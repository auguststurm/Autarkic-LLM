# M1 Ultra Mac Studio (64 GB) - Ternary Bonsai 2 27B

> ⚠️ **Not yet tested** on this hardware with Bonsai 2 (researched **2026-09-18**). Confirm load → **first decode** (Metal can load then OOM) → Pi tools before relying on it. **Metal, not CUDA.** **Not** llama-cpp-turboquant.

64 GB unified · [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, **prism-b10658+**) · **Metal**. Paths: `~/Documents/AIML/models` · `~/Documents/GitHub/llama.cpp-prism` — same macOS layout as [M5 Pro](../M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md). Tighter Metal pattern: [M4 Air](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).

Same Qwen3.8-27B hybrid backbone as Dual RTX Bonsai 2, packed as true ternary weights (~6–7 GB). M5 Pro 48 GB holds Q5 27B (~20 GB) at **196k** q8/q8. This Studio has **64 GB** and Bonsai 2 is ~7 GB — **262k** is the research pin; Metal still has to survive **first decode**. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Stock llama.cpp and llama-cpp-turboquant will not run these files.** They refuse `PQ2_0` / `PTQ1_0` as unknown types. Do not load a Bonsai 2 `Q2_0` on a stock build either — that file can load silently and emit garbage (Hadamard runtime missing). Use the PrismML fork. **Do not paste CUDA flags** (`--n-gpu-layers`, `-DCMAKE_CUDA_ARCHITECTURES`).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (PrismML Metal + this repo’s 48–64 GB unified pin) |
| **PRIMARY weights** | `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.2 GB, 2.13 bpw) |
| **A/B weights** | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.9 GB, 1.75 bpw) |
| **Catalog** | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Context** | `--ctx-size 262144` (`--fit off`) · Metal-OOM fallback [196k / 131k](#this-box) |
| **KV** | `q8_0` / `q8_0` · this fork has **no** `turbo*` types |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` (Pi tools). Model thinks by default if you leave it on |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama.cpp-prism` |

Need Metal prereqs? [local-setup.md](../local-setup.md). This page’s cmake is the PrismML fork, not the turboquant clone. GGUF names (`PQ2_0` vs `PTQ1_0`): [local-setup](../local-setup.md#model-catalog-hugging-face).

Close heavy apps; one long-lived `llama-server`. Unified memory is the same pool as macOS.

## Download

Both packs. Same ternary g128 weights; different on-disk packing. Skip `Ternary-Bonsai-2-27B-F16.gguf` (~54 GB).

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf \
  Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Confirm both files exist under `~/Documents/AIML/models` before building.

**Why two files, and which first.** PrismML’s Apple numbers are **PQ2_0** only (M5 Max ~47 tok/s, M5 Pro ~28, M4 Pro ~18 — **not this M1 Ultra**). **PQ2_0** has Metal kernels. Start there, then [bench PTQ1_0](#try-both-packs). Keep whichever wins *here*.

| Pack | Size | Role on this box |
| --- | --- | --- |
| **PQ2_0** | ~7.2 GB | **PRIMARY** — demo default; measured Apple pack; Metal kernels documented |
| **PTQ1_0** | ~5.9 GB | **A/B** — smaller; no M-series row on the card |

## Build

Separate clone so a turboquant Metal tree stays untouched. `prism-b10658` or newer. Metal is the default backend on macOS in this fork; set it explicitly anyway.

```bash
cd ~/Documents/GitHub
git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git llama.cpp-prism
cd llama.cpp-prism
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)
cd bin
```

Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) branch **`prism`**. Same Metal cmake shape as [M5 Pro](../M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md#build), different repo. The turboquant Metal `rnorm` shader note does **not** apply here.

Confirm before debugging flags:

```bash
uname -m                  # arm64
./llama-server --version  # prism-b10658 or higher (releases have been prism-b10709+)
./llama-server --help | grep -E 'PTQ1|PQ2' || true
```

If load later fails with unknown type `PTQ1_0` / `PQ2_0`, you launched **turboquant** or stock llama.cpp. Check `pwd` is `~/Documents/GitHub/llama.cpp-prism/build/bin`.

**Optional shortcut (prebuilt):** [PrismML macOS Apple Silicon](https://github.com/PrismML-Eng/llama.cpp/releases/latest) (`llama-prism-*-bin-macos-arm64.tar.gz`). This **is** a valid path on Studio (unlike DGX Spark CUDA tarballs). Prefer source if the archive lags `prism`.

## PRIMARY command

Research baseline — Metal unified 64 GB + Bonsai 2 **PQ2_0**. No mmproj. No `--n-gpu-layers`. Thinking **off** for Pi. Run from `llama.cpp-prism/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --alias bonsai-2-27b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
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
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 0 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--model …-PQ2_0.gguf` | Apple-measured packing; swap the filename to try PTQ1_0 |
| `--ctx-size 262144` | Native window. ~7 GB weights + ~9–10 GB hybrid q8 KV leaves tens of GB for macOS + Metal scratch on 64 GB. M5 Pro Q5 27B stopped at **196k** because *weights* were ~20 GB |
| `q8_0` / `q8_0` | Same quality KV as M5 Pro 48 GB. This fork has **no** `turbo*` — if Metal-OOM, drop ctx or batch, not K |
| Batch 512 | Same as M5 Pro Metal. Drop to 256 / 128 on first-decode OOM (Air uses 64 on 24 GB) |
| `--threads 0` | Let the runtime pick host threads on Apple Silicon (M5 pin) |
| No `--n-gpu-layers` | Metal unified memory — CUDA paste will not help |
| No `--load-mode none` | Metal guides keep default **mmap** |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| `--reasoning off` | Pi needs `message.content` / tools. Bonsai 2 **thinks by default** if you leave this on |
| Sampling | Dual RTX / Bonsai **Pi tools** pin — **no DRY**. M5 Qwen uses temp **0.65** / top_p **0.90** / repeat **1.10** — not this PRIMARY |
| `--n-predict 16384` | Match Pi `maxTokens`. M5 Qwen uses **8192** |
| `--alias bonsai-2-27b` | Matches the Pi JSON `id` |

`--ctx-size` is a request. Trust `n_ctx_seq`. Then **decode** — Metal often dies on the first prompt, not at load ([M4 Air](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md)).

Universal flags (`--fit off`, loopback, no checkpoints): [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) still describes them; this binary is the **PrismML** fork, so ignore turbo V / TQ weight types.

### Confirm

```text
log: n_ctx_seq (262144)
# types are PQ2_0 (or PTQ1_0), not a refused unknown type
```

Then: (1) load, (2) **first decode**, (3) new Pi session with real `ls` / `read`.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-2-27b",
       "messages":[{"role":"user","content":"Reply with exactly: BONSAI_OK"}]}'
```

Activity Monitor → GPU / Memory: after load expect a modest footprint (~weights + reserved KV), not a 20 GB Q5. macOS still sits in the same 64 GB.

## Try both packs

Same command. Only `--model` changes. `pkill` between runs.

```bash
# Same PRIMARY, only:
#   --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

From `build/bin`:

```bash
./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128

./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128
```

`-ngl 99` is fine on `llama-bench` here; do **not** add it to the **server** command. Keep the faster pack for interactive Pi. Smoke Pi `ls`/`read` on **each**.

Vendor Apple ballpark (PQ2_0, not this SKU): M5 Max ~47 tok/s, M5 Pro ~28, M4 Pro ~18. M1 Ultra is older / wider — expect **below M5 Max**, not a Dual RTX CUDA number.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. Status bar must show **262k** (or **196k** / **131k** if you drop the pin).

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
          "name": "Ternary Bonsai 2 27B PQ2_0 (262k q8/q8) - M1 Ultra Studio",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

If you settle on PTQ1_0, change `name` only. If you take a [fallback ctx](#this-box), change **both** the server pin and `contextWindow`. `--alias` must still match `id`.

## This box

**64 GB unified.** ~7 GB weights + 262k q8/q8 hybrid KV (~9–10 GB) + Metal scratch should fit with macOS — **if first decode succeeds**. Do not `--split-mode layer`. Do not load these GGUFs in a turboquant Metal `llama-server`.

**Metal-OOM on first decode:** (1) close apps / one server only, (2) batch 256 then 128, (3) `--ctx-size 196608` + Pi 196608 (M5 Pro 27B pin), (4) `--ctx-size 131072` + Pi 131072. Never bare `--fit on`. This fork cannot turbo V.

**Half-window:** `65536` is only if 131k still OOMs — unlikely on 64 GB with 7 GB weights.

llama-server usually **reserves the full KV for `--ctx-size` at startup**. Estimates (`PQ2_0`, q8/q8, `--parallel 1`, no mmproj):

| `--ctx-size` | After load (est.) | Use when |
| --- | --- | --- |
| **262144** (PRIMARY) | **~18–20 GB** + Metal scratch | Default if first decode lives |
| **196608** | a bit under 262k | M5 Pro 27B window; first Metal fallback |
| **131072** | **~13–15 GB** | Comfortable; same as 24 GB Bonsai PRIMARY |

**A) Turn-1 garbage:** new Pi session; q8/q8; no DRY.

**B) Path-heavy tools:** PRIMARY already uses the Dual RTX agent profile. Do not mix M5 Qwen `repeat 1.10` on the same session.

| | M5 Pro Q5 27B (tested 3.6) | This Bonsai 2 pin | Dual RTX Bonsai |
| --- | --- | --- | --- |
| RAM | 48 GB unified | **64 GB unified** | 96 GB discrete |
| Weights | Q5 ~20 GB | **~7 GB** | ~7 GB |
| PRIMARY pin | **196k** q8/q8 | **262k** q8/q8 | **262k** q8/q8 |
| Engine | turboquant Metal | PrismML Metal | PrismML CUDA |
| Batch | 512 | **512** | 1024 |

## Bonsai 2 optionals

### Sampling (leave Pi for these)

| Mode | temp | top_p | presence | Notes |
| --- | --- | --- | --- | --- |
| Thinking | **1.0** | **0.95** | 0.0 | Bonsai 2 demo default; `--reasoning on` |
| Instruct (non-thinking) | 0.7 | 0.80 | **1.5** | Chat only |
| M5 Qwen (this repo) | 0.65 | 0.90 | 0.0 | repeat **1.10** — not the Bonsai Pi default |
| **This repo’s Pi tools** | **0.6** | **0.95** | **0.0** | Dual RTX / Bonsai agent pin |

### Thinking on (not the Pi default)

Primary stays `--reasoning off`. Default effort is **`xhigh`**. Use **`medium`** for shorter traces. PrismML: **`low` is not supported**. Thinking tokens count against `--n-predict`. Leave `--reasoning-preserve` off for Pi.

```bash
# Deltas only — not the Pi primary:
#   --reasoning on
#   --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.0
#   --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

### Vision (`mmproj`)

Not required for Pi text/agent work. Large images are downscaled on Metal by default (~1,024 vision tokens in PrismML’s demo).

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Add `--mmproj ~/Documents/AIML/models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. Images bill as prompt tokens.

### Other stack (MLX)

[prism-ml/Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) (~8.5 GB, vision included) is the Apple-native pack. PrismML: Bonsai 2 MLX runs on **stock MLX**. That is **not** this `llama-server` + Pi JSON primary. Demo: [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo) `start_mlx_server.sh`. Stay on GGUF + PrismML Metal if you want the same OpenAI endpoint as the other Bonsai guides.

## See also

- Dual RTX Bonsai 2 (CUDA, 262k): [Dual-RTX6000-Bonsai-2-27B.md](../Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md)
- M5 Pro 48 GB Qwen (Metal, 196k, turboquant): [M5-MacBook-Pro-Qwen3.8.md](../M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md)
- Tight Metal (24 GB, turbo2): [M4-MacBook-Air-Qwen3.6.md](../M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md)
- Model card: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) · docs: [Ternary Bonsai 2 27B](https://docs.prismml.com/bonsai-2-27b) · [run llama.cpp](https://docs.prismml.com/run/llamacpp)
- Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) · MLX pack: [Ternary-Bonsai-2-27B-mlx-2bit](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
- Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-09-18 (researched; ⚠️ untested on this box)
