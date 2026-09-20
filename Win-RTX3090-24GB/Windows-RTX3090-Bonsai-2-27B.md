# Windows RTX 3090 (24 GB) - Ternary Bonsai 2 27B

> ⚠️ **Not yet tested** on this hardware with Bonsai 2 (researched **2026-09-18**). Confirm load → first decode → Pi tools before relying on it. **Not** llama-cpp-turboquant.

**WSL2** (not native Windows) · CUDA sm_**86** (Ampere) · [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, **prism-b10658+**). Paths: **`~/AIML`** · **`~/GitHub`** — not `~/Documents/AIML`. Do not copy this box’s LFM turboquant cmake here.

Same Qwen3.8-27B hybrid backbone, packed as true ternary weights (~6–7 GB). Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Stock llama.cpp and llama-cpp-turboquant will not run these files.** They refuse `PQ2_0` / `PTQ1_0` as unknown types. Do not load a Bonsai 2 `Q2_0` on a stock build either — that file can load silently and emit garbage (Hadamard runtime missing). Use the PrismML fork.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (PrismML card + this repo’s 24 GB CUDA / WSL2 pin) |
| **PRIMARY weights** | `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.2 GB, 2.13 bpw) |
| **A/B weights** | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.9 GB, 1.75 bpw) |
| **Catalog** | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Context** | `--ctx-size 131072` (`--fit off`) · half-window **65536** · try [262k](#this-box--only-if-primary-loads) if VRAM allows |
| **KV** | `q8_0` / `q8_0` · this fork has **no** `turbo*` types |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` (Pi tools). Model thinks by default if you leave it on |
| **Paths** | `~/AIML/models` · `~/GitHub/llama.cpp-prism` (WSL2) |

Need CUDA prereqs? [local-setup.md](../local-setup.md) (WSL2 + Ubuntu). This page’s cmake is the PrismML fork, not the turboquant clone. GGUF names (`PQ2_0` vs `PTQ1_0`): [local-setup](../local-setup.md#model-catalog-hugging-face).

## Download

Both packs. Same ternary g128 weights; different on-disk packing. Skip `Ternary-Bonsai-2-27B-F16.gguf` (~54 GB).

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf \
  Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --local-dir ~/AIML/models
```

Confirm both files exist under `~/AIML/models` before building.

**Why two files, and which first.** PrismML did **not** publish a 3090 row. Closest numbers on the [model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf): **A100 (Ampere)** prefers **PQ2_0** for decode *and* prompt processing; **RTX 4090 (Ada)** prefers **PTQ1_0** for decode and **PQ2_0** for PP. This card is Ampere + GDDR6X. Start with **PQ2_0** (demo default), then swap to **PTQ1_0** and [bench both](#try-both-packs). Keep whichever wins *here*.

| Pack | Size | Role on this box |
| --- | --- | --- |
| **PQ2_0** | ~7.2 GB | **PRIMARY** — Ampere A100-leaning; cheaper unpack; faster PP everywhere in PrismML’s table |
| **PTQ1_0** | ~5.9 GB | **A/B** — smaller; Ada 4090 decode winner — try second on this 3090 |

## Build (Ampere / sm_86)

Separate clone so a turboquant tree under `~/GitHub/llama-cpp-turboquant` stays untouched. `prism-b10658` or newer.

```bash
cd ~/GitHub
git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git llama.cpp-prism
cd llama.cpp-prism
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build . --config Release -j$(nproc)
cd bin
```

Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) branch **`prism`**. `"86"` is GPU compute capability (`sm_86`), not the CUDA toolkit version. The [4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) uses `"89"` (Ada). Omit `-DCMAKE_CUDA_ARCHITECTURES="86"` to autodetect.

Confirm before debugging flags:

```bash
./llama-server --version          # prism-b10658 or higher (releases have been prism-b10687+)
./llama-server --help | grep -E 'PTQ1|PQ2' || true
```

If load later fails with unknown type `PTQ1_0` / `PQ2_0`, you launched **turboquant** or stock llama.cpp. Check `pwd` is `~/GitHub/llama.cpp-prism/build/bin`.

**Optional shortcut (prebuilt):** [PrismML CUDA 12.8 Linux x64](https://github.com/PrismML-Eng/llama.cpp/releases/latest) (`llama-prism-*-bin-linux-cuda-12.8-x64.tar.gz`). Native `sm_86` is more reliable from the source build above.

## PRIMARY command

Research baseline — 24 GB WSL2 CUDA pin + Bonsai 2 **PQ2_0**. No mmproj. Thinking **off** for Pi. Run from `~/GitHub/llama.cpp-prism/build/bin`.

```bash
pkill -9 llama-server

cd ~/GitHub/llama.cpp-prism/build/bin

./llama-server \
  --model ~/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --alias bonsai-2-27b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
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
  --ubatch-size 256 \
  --batch-size 256 \
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
| `--model …-PQ2_0.gguf` | Ampere-leaning start; swap the filename to try PTQ1_0 |
| `--ctx-size 131072` | Native-class window with margin on 24 GB + WSL desktop VRAM. Dual RTX pins **262k** on 96 GB; try that [below](#this-box--only-if-primary-loads) if `nvidia-smi` has headroom. Half-window is **65536** ([VRAM](#context-sizes-vram)) |
| `q8_0` / `q8_0` | Same 24 GB CUDA agent KV policy as the 4090. This fork has **no** `turbo*` — if you OOM, drop `--ctx-size` or batch, not K |
| Batch 256 | Same prefill-vs-peak as the 4090; Dual RTX uses 1024 on 96 GB. Drop to 128 on OOM |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| `--threads 0` | Auto CPU threads (4090 WSL2 pin). Dual RTX uses 32 on a 24-core Linux box |
| `--reasoning off` | Pi needs `message.content` / tools. Bonsai 2 **thinks by default** if you leave this on |
| Sampling | Dual RTX / 4090 Qwen **Pi tools** pin — **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 16384` | Match Pi `maxTokens`. Thinking tokens also count against this if you turn thinking on |
| `--load-mode none` | Buffered read. Weights are ~7 GB (easier host RAM than the 4090’s ~18 GB Q4) |
| `--alias bonsai-2-27b` | Matches the Pi JSON `id` |

`--ctx-size` is a request. Trust `n_ctx_seq`. Hybrid Qwen3.8 KV is only the full-attention layers (~9–10 GB at **262k** q8/q8 in Dual RTX notes) — weights are ~7 GB, so 131k should be comfortable; 262k is the tight stretch on 24 GB.

Universal flags (`--fit off`, loopback, no checkpoints): [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) still describes them; this binary is the **PrismML** fork, so ignore turbo V / TQ weight types.

### Confirm

```text
log: n_ctx_seq (131072)
log: load_mode = none
# model loads; types are PQ2_0 (or PTQ1_0), not a refused unknown type
nvidia-smi   # inside WSL; MiB after load and after a short decode
```

Then: (1) load, (2) first decode, (3) new Pi session with real `ls` / `read`.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-2-27b",
       "messages":[{"role":"user","content":"Reply with exactly: BONSAI_OK"}]}'
```

Normal logs: `cache-idle-slots requires --cache-ram, disabling` (follows `--cache-ram 0`).

## Try both packs

Same command. Only `--model` changes. `pkill` between runs.

```bash
# Same PRIMARY, only:
#   --model ~/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

From `build/bin`, numbers PrismML reports as `tg128` / `pp512`:

```bash
./llama-bench \
  -m ~/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128

./llama-bench \
  -m ~/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128
```

Keep the faster pack for interactive Pi; keep the other on disk. Quality should be close (same ternary assignment; packing differs). Smoke Pi `ls`/`read` on **each** before treating a tok/s winner as the daily driver.

Vendor ballpark (not this SKU): RTX 4090 PQ2_0 ~81 tok/s decode / ~3124 pp; PTQ1_0 ~91 / ~1645. Expect the 3090 a bit under the 4090. Dual RTX Q8 27B is a different quality/speed class (~51 tok/s on that box).

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. Status bar must show **~131k** (or **64k** / **262k** if you change the pin). Same JSON for either pack (one `llama-server` at a time).

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
          "name": "Ternary Bonsai 2 27B PQ2_0 (131k q8/q8) - RTX 3090",
          "contextWindow": 131072,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

If you settle on PTQ1_0, change `name` only. If you take **64k** (`65536`) or [262k](#this-box--only-if-primary-loads), change **both** the server pin and `contextWindow`. `--alias` must still match `id`.

## This box — only if primary loads

**Single GPU.** ~6–7 GB weights + 131k q8/q8 hybrid KV should land well under 24 GB. Do not `--split-mode layer`. Windows desktop + WSL still steal some VRAM — check `nvidia-smi` **inside** WSL.

### Context sizes (VRAM)

`--ctx-size 131072` is the PRIMARY. Half of that is **`65536` (64k)** — same command, only the pin and Pi `contextWindow` change. Weights stay ~7.2 GB (`PQ2_0`); KV is what halves. Estimates for this guide’s pin (`q8_0`/`q8_0`, `--parallel 1`, no mmproj), not a measured 3090 `nvidia-smi`:

| `--ctx-size` | KV (q8) | After load (est.) | Use when |
| --- | ---: | --- | --- |
| **131072** (PRIMARY) | ~4–5 GB | **~13–15 GB** (~15–17 GB if the window is full) | Default. Plenty of 24 GB left |
| **65536** (half) | ~2–2.5 GB | **~11–13 GB** | Extra headroom (desktop/WSL VRAM, mmproj, or a second process). Match Pi `contextWindow` |

llama-server usually **reserves the full KV for `--ctx-size` at startup**, so load already shows most of that number. `PTQ1_0` is ~1.3 GB less on weights. 262k roughly **doubles** the 131k KV (~9–10 GB) → **~18–21 GB** total — tight on 24 GB WSL; that stretch is **B)** below.

```bash
# Half-window — same PRIMARY, only:
#   --ctx-size 65536
# Pi contextWindow: 65536
```

**OOM on load / first decode:** (1) batch 128, (2) `--ctx-size 98304` + Pi 98304, (3) `--ctx-size 65536` + Pi 65536, (4) free desktop GPU apps / check `nvidia-smi` **inside** WSL. Last resort on this fork (no turbo V): `--cache-type-k q4_0 --cache-type-v q4_0` at the same ctx — re-smoke Pi tools.

**262k stretch** if `n_ctx_seq` matches and VRAM has headroom: `--ctx-size 262144` and Pi `contextWindow` 262144. If prefill OOMs, keep 262k and drop batch to 128.

`nvidia-smi` **inside** WSL. Loopback `--host 127.0.0.1` is reachable from Windows clients on the same machine.

Sampling, thinking on, vision: **[Dual RTX Bonsai 2 — optionals](../Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md#bonsai-2-optionals)**. mmproj path on this box is `~/AIML/models`.

## See also

- Dual RTX Bonsai 2 (96 GB, 262k PRIMARY): [Dual-RTX6000-Bonsai-2-27B.md](../Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md)
- 24 GB RDNA3 HIP twin: [7900-XTX-Bonsai-2-27B.md](../AMD-7900-XTX/7900-XTX-Bonsai-2-27B.md)
- This box, LFM2.5-2.6B (turboquant, ⚠️ untested @ 128k / FA off): [Windows-RTX3090-LFM2.5-2.6B.md](Windows-RTX3090-LFM2.5-2.6B.md)
- 24 GB CUDA WSL2 twin (Qwen3.6 Q4, turboquant): [Windows-RTX4090-Qwen3.6.md](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md)
- Model card: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) · docs: [Ternary Bonsai 2 27B](https://docs.prismml.com/bonsai-2-27b) · [run llama.cpp](https://docs.prismml.com/run/llamacpp)
- Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) · demo: [Bonsai-demo](https://github.com/PrismML-Eng/Bonsai-demo)
- Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-09-20 (recipe density; ⚠️ untested on this box)
