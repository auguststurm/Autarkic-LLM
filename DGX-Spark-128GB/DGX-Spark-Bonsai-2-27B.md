# DGX Spark Founders Edition (128 GB) - Ternary Bonsai 2 27B

> ⚠️ **Not yet tested** on this hardware with Bonsai 2 (researched **2026-09-18**). Confirm load → first decode → Pi tools before relying on it. **Not** llama-cpp-turboquant.

CUDA CC **12.1** (GB10) · [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, **prism-b10658+**). Paths: `~/Documents/AIML/models` · `~/Documents/GitHub/llama.cpp-prism` — same Linux layout as this folder’s [Qwen3.6](DGX-Spark-Qwen3.6.md) / [Qwen3.8](DGX-Spark-Qwen3.8.md).

Same Qwen3.8-27B hybrid backbone as Dual RTX Bonsai 2, packed as true ternary weights (~6–7 GB). This box already holds **262k** with Q6 27B (~22–26 GB). Bonsai 2 is a small object on 128 GB unified memory. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Twin Dual RTX recipe: [Dual-RTX6000-Bonsai-2-27B.md](../Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md).

**Stock llama.cpp and llama-cpp-turboquant will not run these files.** They refuse `PQ2_0` / `PTQ1_0` as unknown types. Do not load a Bonsai 2 `Q2_0` on a stock build either — that file can load silently and emit garbage (Hadamard runtime missing). Use the PrismML fork.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (PrismML card + this box’s GB10 CUDA pin) |
| **PRIMARY weights** | `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.2 GB, 2.13 bpw) |
| **A/B weights** | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.9 GB, 1.75 bpw) |
| **Catalog** | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` · this fork has **no** `turbo*` types |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` (Pi tools). Model thinks by default if you leave it on |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama.cpp-prism` |

Need CUDA prereqs? [local-setup.md](../local-setup.md). This page’s cmake is the PrismML fork (`"121"`), not the turboquant clone. GGUF names (`PQ2_0` vs `PTQ1_0`): [local-setup](../local-setup.md#model-catalog-hugging-face).

## Download

Both packs. Same ternary g128 weights; different on-disk packing. Skip `Ternary-Bonsai-2-27B-F16.gguf` (~54 GB).

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf \
  Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Confirm both files exist under `~/Documents/AIML/models` before building.

**Why two files, and which first.** PrismML has **no GB10 / Spark row**. Closest: **Blackwell** cards (RTX PRO 6000, 5090) prefer **PQ2_0** for decode *and* prompt processing; **Ada** prefers **PTQ1_0** for decode. GB10 is Blackwell. Start with **PQ2_0**, then [bench both](#try-both-packs).

| Pack | Size | Role on this box |
| --- | --- | --- |
| **PQ2_0** | ~7.2 GB | **PRIMARY** — Blackwell-leaning; cheaper unpack; faster PP on PrismML’s table |
| **PTQ1_0** | ~5.9 GB | **A/B** — denser trits; try second and keep whichever wins here |

## Build

Separate clone so this folder’s turboquant tree stays untouched. `prism-b10658` or newer. Spark is **GB10 + Grace (ARM)**. PrismML’s CUDA release tarballs are **x86_64** — **build from source** on this box. Pin `"121"` (this folder’s other guides); a 12.8 x64 fat binary would miss `sm_121` even if the ISA matched.

```bash
cd ~/Documents/GitHub
git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git llama.cpp-prism
cd llama.cpp-prism
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="121"
cmake --build . --config Release -j$(nproc)
cd bin
```

Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) branch **`prism`**. `"121"` is GPU compute capability (`sm_121` / CC 12.1), same as [Qwen3.8 on this box](DGX-Spark-Qwen3.8.md#build) — not the CUDA toolkit version. Omit `-DCMAKE_CUDA_ARCHITECTURES="121"` to autodetect.

Confirm before debugging flags:

```bash
uname -m                  # aarch64 on Spark
./llama-server --version  # prism-b10658 or higher
./llama-server --help | grep -E 'PTQ1|PQ2' || true
```

If load later fails with unknown type `PTQ1_0` / `PQ2_0`, you launched **turboquant** or stock llama.cpp. Check `pwd` is `~/Documents/GitHub/llama.cpp-prism/build/bin`.

## PRIMARY command

Research baseline — this box’s CUDA pin (262k, `--threads 28`, batch 1024) + Bonsai 2 **PQ2_0**. No mmproj. Thinking **off** for Pi. KV is **q8/q8** (PrismML has no `turbo4`). Run from `llama.cpp-prism/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --alias bonsai-2-27b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
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
  --threads 28 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--model …-PQ2_0.gguf` | Blackwell-leaning pack; swap the filename to try PTQ1_0 |
| `--ctx-size 262144` | Full native window. This box already ran 262k with Q6 ~22–26 GB. Bonsai 2 weights + hybrid q8 KV is ~18–20 GB class on 128 GB unified |
| `q8_0` / `q8_0` | This folder’s Qwen uses **turbo4 V** on turboquant. PrismML has **no** `turbo*` — keep q8 V; 128 GB does not need the capacity lever |
| `--threads 28` | Same Spark host CPU pairing as this folder’s Qwen 3.6 / 3.8 |
| Batch 1024 | Same as this folder’s Qwen and Dual RTX Bonsai |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| `--load-mode none` | Buffered read. Unified memory: the GGUF is small (~7 GB) |
| `--reasoning off` | Pi needs `message.content` / tools. Bonsai 2 **thinks by default** if you leave this on |
| Sampling | Dual RTX / Bonsai **Pi tools** pin — **no DRY**. This folder’s tested Qwen 3.6 is temp **0.65** / top_p **0.90** / repeat **1.10** — use that only if you want to match the Spark Qwen command, on a **new** session |
| `--n-predict 16384` | Match Pi `maxTokens`. This folder’s Qwen uses **8192**; raise both if you keep that |
| `--alias bonsai-2-27b` | Matches the Pi JSON `id` |
| No `--main-gpu` | Single GB10, same as this folder’s Qwen |

Universal flags (`--fit off`, loopback, no checkpoints): [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) still describes them; this binary is the **PrismML** fork, so ignore turbo V / TQ weight types.

### Confirm

```text
log: n_ctx_seq (262144)
log: load_mode = none
# types are PQ2_0 (or PTQ1_0), not a refused unknown type
nvidia-smi   # unified pool; expect ~18–20 GB class, not Q6’s ~30+
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

From `build/bin`:

```bash
./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128

./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128
```

Keep the faster pack for interactive Pi. Smoke Pi `ls`/`read` on **each**. This folder’s Qwen 3.6 ballpark is ~45–65 t/s prefill / 90–120+ decode — Bonsai 2 will be a different number; re-bench. Dual RTX Q8 27B (~51 tok/s on that box) is a different quality/speed class.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. Status bar must show **262k**.

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
          "name": "Ternary Bonsai 2 27B PQ2_0 (262k q8/q8) - DGX Spark",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

If you settle on PTQ1_0, change `name` only. `--alias` must still match `id`. If you drop `--n-predict` to this folder’s Qwen **8192**, set `maxTokens` to 8192 too.

## This box

**128 GB unified.** ~7 GB weights + 262k q8/q8 hybrid KV (~9–10 GB) + scratch is an **~18–20 GB** object. Do not `--split-mode layer`. Do not load these GGUFs in the turboquant `llama-server` from the Qwen guides.

This is **not** a replacement for the [Q6 27B Qwen3.8 port](DGX-Spark-Qwen3.8.md) or the [tested 3.6](DGX-Spark-Qwen3.6.md). Those are the turboquant hosts. Bonsai 2 is the ternary Qwen3.8 compression for speed/footprint on the PrismML fork.

**No turbo V.** This folder’s Qwen pin uses turbo4 V to buy 262k with Q6 weights. Bonsai 2 does not need that, and this fork cannot do it.

**A) OOM (unlikely):** drop batch to 256, then `--ctx-size 131072` + Pi 131072. Confirm you are on `llama.cpp-prism`, not turboquant.

**B) Path-heavy Pi tools feel off:** this PRIMARY already uses the Dual RTX agent profile. Do not mix in this folder’s Qwen `repeat 1.10` / `temp 0.65` on the same session.

| | This folder Qwen 3.6 (tested) | This Bonsai 2 pin | Dual RTX Bonsai |
| --- | --- | --- | --- |
| Weights | Q6 ~22 GB | **~7 GB** | ~7 GB |
| PRIMARY pin | **262k** q8/**turbo4** | **262k** q8/**q8** | **262k** q8/q8 |
| Engine | turboquant `"121"` | PrismML `"121"` | PrismML `"120"` |
| `--threads` | 28 | **28** | 32 |
| `--n-predict` | 8192 | **16384** | 16384 |

## Bonsai 2 optionals

### Sampling (leave Pi for these)

| Mode | temp | top_p | presence | Notes |
| --- | --- | --- | --- | --- |
| Thinking | **1.0** | **0.95** | 0.0 | Bonsai 2 demo default; `--reasoning on` |
| Instruct (non-thinking) | 0.7 | 0.80 | **1.5** | Chat only |
| This folder’s tested Qwen 3.6 | 0.65 | 0.90 | 0.0 | repeat **1.10** — Spark Qwen command, not the Bonsai Pi default |
| **This repo’s Pi tools** | **0.6** | **0.95** | **0.0** | Dual RTX / Bonsai agent pin |

### Thinking on (not the Pi default)

Primary stays `--reasoning off`. Default effort is **`xhigh`**. Use **`medium`** for shorter traces. PrismML: **`low` is not supported**. Thinking tokens count against `--n-predict` — this folder’s Qwen 8192 cap is easy to blow; PRIMARY is already 16384. Leave `--reasoning-preserve` off for Pi.

```bash
# Deltas only — not the Pi primary:
#   --reasoning on
#   --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.0
#   --chat-template-kwargs '{"reasoning_effort":"medium"}'
```

### Vision (`mmproj`)

Not required for Pi text/agent work.

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Add `--mmproj ~/Documents/AIML/models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`. CUDA/ROCm image token cap is uncapped in PrismML’s demo; images still bill as prompt tokens.

## See also

- Twin 3.6 (tested on this box): [DGX-Spark-Qwen3.6.md](DGX-Spark-Qwen3.6.md)
- Qwen3.8 port (⚠️ untested, turboquant): [DGX-Spark-Qwen3.8.md](DGX-Spark-Qwen3.8.md)
- Dual RTX Bonsai 2 (CUDA sm_120, 262k): [Dual-RTX6000-Bonsai-2-27B.md](../Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md)
- 24 GB CUDA WSL2 Bonsai 2 (131k): [Windows-RTX3090-Bonsai-2-27B.md](../Win-RTX3090-24GB/Windows-RTX3090-Bonsai-2-27B.md)
- Model card: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) · docs: [Ternary Bonsai 2 27B](https://docs.prismml.com/bonsai-2-27b)
- Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)
- Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-09-18 (researched; ⚠️ untested on this box)
