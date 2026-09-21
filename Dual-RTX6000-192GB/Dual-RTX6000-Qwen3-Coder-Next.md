# Dual RTX 6000 Pro Max-Q (192 GB) - Qwen3-Coder-Next

> ⚠️ **Not yet tested** on this hardware with Qwen3-Coder-Next (researched **2026-09-20**). Confirm load → first decode before relying on it.

Blackwell **sm_120** · llama-cpp-turboquant · Ubuntu. **One 96 GB card.** 80B MoE / **3B active**, arch `qwen3next`, native **262k**, **non-thinking** (no `<think>` blocks). Catalog: [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF).

This page is the **coder model** on GPU 0. The other 96 GB card stays free for the tested [Qwen3.8-27B primary](Dual-RTX6000-Qwen3.8.md) if you want it later — that dual-process pattern is [Localmaxing](Dual-RTX6000-Qwen3.8-localmaxing.md), not this file.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (Unsloth + Qwen card + this box’s Dual RTX CUDA pin) |
| **PRIMARY weights** | `Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf` (~73.1 GB, **3 shards**) |
| **Catalog** | [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) · [Qwen/Qwen3-Coder-Next](https://huggingface.co/Qwen/Qwen3-Coder-Next) |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` · **one** GPU (`CUDA_VISIBLE_DEVICES=0` · `--split-mode none` · `--main-gpu 0`) |
| **Output** | `--n-predict 32768` |
| **Sampling** | temp **1.0** · top_p **0.95** · top_k **40** · min_p **0.01** · presence **0** · repeat **1.0** |
| **Thinking** | Instruct-only (no `<think>`). Keep `--reasoning off` + budget **0** like other Dual RTX Qwen pins |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant` |

Need the engine built first? [local-setup.md](../local-setup.md). GGUF names: [local-setup](../local-setup.md#understanding-gguf-quants-why-so-many-files). Unsloth run notes: [Qwen3-Coder-Next](https://unsloth.ai/docs/models/qwen3-coder-next).

**3B active is decode speed, not a VRAM discount.** All 512 experts still load. This GGUF **replaces a card** — it is not a sidecar next to a 262k 27B on the **same** GPU ([Localmaxing](Dual-RTX6000-Qwen3.8-localmaxing.md#vram-remaining-is-not-idle-gpu)).

## Download

**One pack:** `UD-Q6_K_XL` (~73.1 GB). Unsloth Q6 is **sharded** (three files). `llama-server` takes `00001-of-00003`; keep the other two **in the same folder**. `00001` is a **~5.9 MB metadata shard**, not the model.

```bash
hf download unsloth/Qwen3-Coder-Next-GGUF \
  --include "UD-Q6_K_XL/*" \
  --local-dir ~/Documents/AIML/models
```

```bash
ls -lh ~/Documents/AIML/models/UD-Q6_K_XL/
# expect ~5.9M + ~50G + ~23G. Point --model at 00001-of-00003 only.
```

Skip `BF16` (~159 GB), `imatrix_*`, and other quants. Q8 / Q5 are [later swaps](#later-swaps) if Q6 is clean and you have a reason — not part of this download.

On this 96 GB card Q6 is the realistic primary (full 262k with headroom). Q8 (~86.3 GB) is a tight stretch; Q5 (~59.5 GB) is extra headroom. Prefer `UD-…_K_XL` over plain `Q8_0` / `Q6_K` / `Q5_K_M`. Ladder: [local-setup](../local-setup.md#q8-vs-q6-vs-q5-vs-q4-quality-vs-speed).

Unsloth: 4-bit wants **>45 GB**; 8-bit wants **~85 GB** (weights-class, short context). Hybrid **attention** KV at 262k q8/q8 is a **geometry estimate** (~3–4 GB: 12 full-attention layers, 2 KV heads, head dim 256) — not measured on this box. Naive “48 layers × 2 KV heads” calculators quote ~12 GB at f16; only **12/48** layers keep a full cache. CUDA scratch still scales with `--ubatch-size` and **first decode can OOM after a successful load**. That is why Q8 is not the primary.

Need a **fresh** turboquant (arch `qwen3next`). Unsloth re-uploaded these GGUFs after a **2026-02-04** llama.cpp `key_gdiff` fix (looping / bad output) and a **2026-02-19** tool-call parse fix. This box already serves **qwen35** (Qwen3.8); `qwen3next` landed in llama.cpp **earlier**. Still confirm **this** binary (below). `git pull` and rebuild if load says unknown architecture.

## Build

Same cmake as the [Qwen3.8 primary](Dual-RTX6000-Qwen3.8.md#build):

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). `"120"` is GPU compute capability (`sm_120`), not the CUDA toolkit version.

```bash
./llama-server --version
./llama-server --list-devices
# CUDA0 / CUDA1 should each show ~96 GB. PRIMARY hides CUDA1 at runtime.
grep -R LLM_ARCH_QWEN3NEXT ../../src ../../include 2>/dev/null | head
# expect at least one hit (file layout moved in some llama.cpp revs)
```

If load fails with `unknown model architecture: 'qwen3next'`, you launched an old binary — pull, rebuild, check `pwd` is `llama-cpp-turboquant/build/bin`. Do not add YaRN (`--rope-scaling yarn`) at 262k — that is Qwen’s **>262k** recipe.

## PRIMARY command

Research baseline — Dual RTX CUDA pin + Unsloth / Qwen **Coder-Next** sampling. **Q6_K_XL**, no MTP, no split. Run from `build/bin`.

```bash
pkill -9 llama-server

cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=0 ./llama-server \
  --model ~/Documents/AIML/models/UD-Q6_K_XL/Qwen3-Coder-Next-UD-Q6_K_XL-00001-of-00003.gguf \
  --alias qwen3-coder-next \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 262144 \
  --fit off \
  --n-gpu-layers 99 \
  --split-mode none \
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
  --temp 1.0 --top-p 0.95 --top-k 40 --min-p 0.01 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 32 \
  --n-predict 32768 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values (this box + this model)

| Flag | Why |
| --- | --- |
| `--model …-00001-of-00003.gguf` | Sharded pack; llama.cpp finds `00002` / `00003` next to it |
| `CUDA_VISIBLE_DEVICES=0` + `--split-mode none` + `--main-gpu 0` | **Required on this dual-GPU box.** llama.cpp default `--split-mode layer` stripes layers across **every visible GPU**. `--main-gpu 0` alone does **not** confine an MoE. Qwen’s `-sm row` is multi-GPU expert split — not this pin |
| `--ctx-size 262144` | Native window. Attention KV is a geometry estimate (~3–4 GB q8/q8), not measured. Q6 weights (~73 GB) should leave room; first decode is the real test |
| `q8_0` / `q8_0` | High-precision KV while VRAM allows; turbo V is a later capacity lever (not needed for Q6) |
| `--cache-ram 0` | Same Gated-DeltaNet family as Qwen3.8; multi-turn restore issues ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| `--load-mode none` | Dual RTX CUDA pin (buffered read). **Host RAM must cover the GGUF during load** (~73 GB for Q6, ~86 GB for Q8). If the host OOMs or load is painfully slow, omit it (default mmap) |
| `--ubatch-size` / `--batch-size` **1024** | Dual RTX CUDA pin. Hidden size is only **2048**, so scratch is not 27B-dense-sized — still drop to **512** if first decode OOMs |
| `--reasoning off` + budget **0** | Instruct-only model (no `<think>`). Not a Muse-style no-op; keep the pin so clients get `message.content`. Do not pass `enable_thinking` kwargs |
| Sampling | Unsloth Coder-Next defaults (**min_p 0.01** — llama.cpp’s default is 0.05). Qwen’s own llama.cpp snippet uses **min_p 0**; stay on 0.01 with these GGUFs. **No DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 32768` | Long patches / agent turns (Qwen llama.cpp example). Match the client output cap |
| `--jinja` + `--alias qwen3-coder-next` | Embedded coder template (tool calls). Alias matches Pi `id` if you wire it later |

Universal flags (`--fit off`, `--flash-attn on`, loopback, no checkpoints): [llama-cpp-turboquant.md](../llama-cpp-turboquant.md).

### Confirm

`--ctx-size 262144` is a request. Trust the load log:

```text
log: n_ctx_seq (262144)
log: load_mode = none
# model loads as qwen3next / Qwen3-Coder-Next
# split / device: only CUDA0. If you see CUDA1 in the offload table, isolation failed
nvidia-smi   # after load AND after first decode — GPU 1 should stay ~display (~1 GB)
```

If `n_ctx_seq` is **32768** (or anything below 262144), the GGUF metadata capped it — stop and report; do **not** add YaRN to “fix” it.

Then: (1) load, (2) **first decode** (CUDA can load then OOM), (3) a real tool-shaped prompt.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-coder-next",
       "messages":[{"role":"user","content":"Reply with exactly: CODER_NEXT_OK"}]}'
```

The string reply only proves decode. Tool parse (the Feb 19 llama.cpp fix) needs a `tools=` request before you trust an agent loop.

## Later swaps

Stay on Q6 until load + first decode are clean. Do **not** fetch Q8 or Q5 “just in case” (~86 GB + ~60 GB extra). Same PRIMARY flags; only `--model` changes. `pkill` between runs. If **Q6** first-decode OOMs, drop `--ubatch-size 512 --batch-size 512` **before** dropping context or changing quant.

### 8-bit (`UD-Q8_K_XL`) — optional stretch

~86.3 GB. Tight on one 96 GB card. Download only after Q6 works and you want the quality experiment:

```bash
hf download unsloth/Qwen3-Coder-Next-GGUF \
  --include "UD-Q8_K_XL/*" \
  --local-dir ~/Documents/AIML/models
```

```bash
# Same PRIMARY, only:
#   --model ~/Documents/AIML/models/UD-Q8_K_XL/Qwen3-Coder-Next-UD-Q8_K_XL-00001-of-00003.gguf
```

Keep this process on **GPU 0**. If **load** or **first decode** OOMs, change **one** axis at a time:

1. `--ubatch-size 512 --batch-size 512`
2. `--ctx-size 131072` (then the client window must match)
3. Stay on Q6

Do not also pass `--fit on`. Skip plain `Q8_0` unless UD-Q8 is the last straw.

### 5-bit (`UD-Q5_K_XL`) — optional headroom

~59.5 GB. Same 262k q8/q8. Only if Q6 is hotter than you like or you need VRAM back. Quality step Q6 → Q5 is small; skip Q4 on this card.

```bash
hf download unsloth/Qwen3-Coder-Next-GGUF \
  --include "UD-Q5_K_XL/*" \
  --local-dir ~/Documents/AIML/models
```

```bash
# Same PRIMARY, only:
#   --model ~/Documents/AIML/models/UD-Q5_K_XL/Qwen3-Coder-Next-UD-Q5_K_XL-00001-of-00003.gguf
```

**Optional turbo V** (capacity only — Q6/Q5 should not need it):

```bash
# Same PRIMARY, only:
#   --cache-type-k q8_0 --cache-type-v turbo4
# Keep K at q8_0. Re-smoke a tool call before trusting it.
```

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`) when you point Pi at this server. Restart Pi. `contextWindow` = 262144, `maxTokens` = 32768. If a later Q8 swap drops ctx to 131k, change **both** the server pin and `contextWindow`.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3-coder-next",
          "name": "Qwen3-Coder-Next Q6_K_XL (262k q8/q8) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

`--alias` **must** match `id`. Connect rules (two limits, no DRY): [agentic harnesses](../agentic-harnesses.md).

## This box

**Single GPU on purpose.** Q6 + 262k q8/q8 is meant to fill **one** 96 GB card as the coding endpoint. PRIMARY already sets `--split-mode none`. Do not add `--split-mode layer --tensor-split 96,96` unless Q8 will not load on one GPU — that is still **one** model, striped, and it occupies both cards.

The second card is idle until you start a **second** `llama-server`. Tested 27B host: [Qwen3.8](Dual-RTX6000-Qwen3.8.md). Isolation recipe: [Localmaxing](Dual-RTX6000-Qwen3.8-localmaxing.md) (`pkill` **once**, start GPU 1 first). PRIMARY already uses `CUDA_VISIBLE_DEVICES=0`; keep that if you add a second process.

Do not park Coder-Next and a 262k 27B on the **same** GPU — they queue.

**Sampling vs the 27B Pi row.** This model’s published defaults are temp **1.0** / top_k **40** / min_p **0.01**. Do not copy the dense-27B Pi row (temp 0.6 / top_k 20) unless tool paths warp; keep **presence 0** and **no DRY** either way.

No `mmproj`. No thinking toggle. FP8 / NVFP4 via vLLM or SGLang is a different stack ([Unsloth](https://unsloth.ai/docs/models/qwen3-coder-next#fp8-qwen3-coder-next-in-vllm)).

## See also

- Unsloth: [Qwen3-Coder-Next](https://unsloth.ai/docs/models/qwen3-coder-next) · GGUF: [unsloth/Qwen3-Coder-Next-GGUF](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
- Tested 27B on the other card: [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-Qwen3.8.md)
- Two processes, one per GPU: [Dual-RTX6000-Qwen3.8-localmaxing.md](Dual-RTX6000-Qwen3.8-localmaxing.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses](../agentic-harnesses.md)

**Last Updated:** 2026-09-20 (PRIMARY isolation: `CUDA_VISIBLE_DEVICES=0` + `--split-mode none`; ⚠️ untested)
