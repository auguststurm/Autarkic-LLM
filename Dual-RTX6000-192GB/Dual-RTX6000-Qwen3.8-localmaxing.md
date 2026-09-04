# Dual RTX 6000 Pro Max-Q (192 GB) — Localmaxing

> This machine has **two 96 GB cards**. The [Q8 27B primary](Dual-RTX6000-Qwen3.8.md) is the tested single-model Pi setup — it uses one card. This page is how to keep **more than one Qwen resident** so both cards work.
>
> One isolated `llama-server` per model (`CUDA_VISIBLE_DEVICES`). Not llama.cpp **router** mode (load/unload one process). Not `--split-mode layer` (still one model, striped).

Blackwell **sm_120** · llama-cpp-turboquant · Ubuntu. Paths: model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant`.

| Pin | Value |
| --- | --- |
| **Status** | ✅ Two Q6 27B load + parallel decode **2026-09-03**. 27B + specialist ⚠️ not load-tested as a pair |
| **27B** | `--ctx-size 262144` (`--fit off`) · `q8_0`/`q8_0` · `--n-predict 16384` · `--parallel 1` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** · `--reasoning off` |
| **Isolation** | `CUDA_VISIBLE_DEVICES` · `--main-gpu 0` inside each process · `--host 127.0.0.1` |
| **CPU** | `--threads 12` (two processes) |

Build: [Q8 primary](Dual-RTX6000-Qwen3.8.md#build). Pi: `contextWindow` = that process’s `--ctx-size`; `maxTokens` = 16384.

## What this page is for

Every guide in this repo is a per-machine recipe: download, cmake, `llama-server`, Pi `models.json`. On a 24 GB card that means **one** model. On this box the Q8 27B primary leaves ~96 GB idle.

Localmaxing is the second recipe for the same hardware: **two (or more) GGUFs loaded at once**, each on its own `llama-server`, so a local harness can use more than one endpoint without swapping weights.

Pi Coding Agent, `pi-dynamic-workflows`, and `pi-subagents` are why multiple endpoints matter — they can `/model` hop and route `small` / `medium` / `big` at different `baseUrl`s. Package install, Tavily, and skills are **not** this page: [Pi graphs](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md).

## Combinations that use both cards

Two cards can **decode two streams at once**. Two processes on the **same** GPU queue. GPU 1 holds the display (~1 GB). Put the 27B you will actually talk to on GPU 1 so a specialist on GPU 0 cannot stall it.

One model per card. The **second stream** is what you are choosing. GPU 1 is always a 27B you can run Pi on. If you only have one job, stay on the [Q8 primary](Dual-RTX6000-Qwen3.8.md) — this page is for when both cards should generate.

| Pack | GPU 0 `:8080` | GPU 1 `:8081` | Use when the second stream is… | Status |
| --- | --- | --- | --- | --- |
| **Two 27B** | 3.8-27B Q6 262k (~35 GB) | 3.8-27B Q6 262k (~35 GB) | **Another 27B** — same quality, same 262k | ✅ 2026-09-03 |
| **27B + Coder** | Coder-30B-A3B Q6 64k (~29 GB) | 3.8-27B Q8 262k (~40 GB) | **Fast code** (3B active) while you stay on a Q8 host | ⚠️ Pair untested |
| **27B + 35B-A3B** | 3.6-35B-A3B Q8 262k (~50 GB) | 3.8-27B Q8 262k (~40 GB) | **Fast general** work (3B active), not repo edits | ⚠️ Pair untested |

Q8 27B on **both** cards should also fit (~40 GB class each). Q6 is what was measured. Do not drop the interactive 27B below Q8 unless you are running two 27Bs.

### Why pick a pack

Workflows only use GPU 0 if `small` (or `{ model: "llama-cpp-8080/…" }`) actually points there. If every `agent()` is `medium` on the 27B, the second card is idle no matter which pack you loaded.

**Two 27B** — both endpoints are the Pi-quality coder.

- **Use for:** two interactive Pi sessions (two terminals, two `/model` ids); or `parallel()` subagents that must be as good as the host (hard review, implement, architecture) and may need 262k.
- **Not for:** cheap fan-out (list files, classify, short extract). You pay 27B latency for work a 3B-active MoE would finish faster. The interactive session is Q6, not the tested Q8 host.

**27B + Coder** — Q8 27B host on GPU 1; Coder-30B is a coding MoE on GPU 0.

- **Use for:** you on Pi (tools, 262k, planning) while workflow/subagent **code** jobs run on the other card — file audit, patch, review of a slice of the repo. Map `small` → Coder, `medium`/`big` → 27B.
- **Not for:** a second human who needs a full 27B session; non-code graphs (ingest, classify, writing); using Coder as the Pi host (64k, not the tested agent).

**27B + 35B-A3B** — same Q8 host; 35B-A3B is a general MoE (still 3B active, 262k in this pin).

- **Use for:** you on Pi while the other card does **non-code** fan-out — scan, classify, ingest, skeptic, short summary. Faster than a second 27B for those jobs; 262k on the fast side if the prompt is large.
- **Not for:** code-heavy subagents (use Coder); two streams that both need 27B quality (use two 27Bs).

Coder-Next / Flash-Next replace a *card*, they are not a fourth pack — [VRAM remaining](#vram-remaining-is-not-idle-gpu).

### VRAM remaining is not idle GPU

Localmaxing here is **two full-speed generations at once** (one per card), not 100% occupancy. A Q6 27B at 262k is a **~35 GB** object on a **96 GB** card. That is the model, not a timid pin.

Weights are ~26 GB (Q6) / ~32 GB (Q8). The rest is KV + scratch. Qwen3.8-27B is hybrid: only **16** layers keep full attention KV, so 262k costs ~9–10 GB instead of blowing the card. The native window is already 262k — you cannot spend the other ~60 GB on “more context” for this model. Higher quant is Q8 (~40 GB). MTP is 1–2 GB. Display on GPU 1 is ~1 GB. None of that fills a 96 GB card.

What *is* maxed: **decode**. One 27B already holds a card at ~99% / ~300 W / ~51 tok/s. Remaining is empty **weight storage**, not unused FLOPs. Filling it with a second `llama-server` on the **same** GPU does not add tok/s. Those two processes **queue**. Pi `parallel()` of four agents on four endpoints still only generates **two** at a time; the two that share a GPU split that card’s ~51 tok/s (~25 each) and the session feels stuck. Two processes on the **host** GPU is worse: your interactive 27B waits behind a specialist.

| Spend remaining on | Occupancy | What Pi feels |
| --- | --- | --- |
| Nothing extra (these packs) | ~35–50 GB / card | Card already at full tok/s. Second GPU is a second stream |
| Second GGUF on the **same** GPU | Higher | Queue. Same tok/s, more models waiting. Fine only if they almost never generate together |
| 2+2 Q6 27B (two processes per card) | ~70 GB / card | Four endpoints, still two concurrent streams. ~25 tok/s per model if all four generate — a bad Pi experience |
| Bigger model **instead** (Coder-Next ~52 GB, Flash-Next Q2/Q3 ~79–90 GB) | Most of one card | Still **one** stream on that GPU. Different quality; Flash-Next is not a Pi host |

So the Remaining column is large because **this Qwen 27B does not need 96 GB**, and stuffing the rest with more copies does not make Pi faster. Two cards → two streams. That is the maximum that still feels like a coding agent. Occupancy is a different knob: bigger model on a card, or extra GGUFs you accept will queue.

### Qwen that belong on this box

| Model | Job on a 96 GB card | Skip if |
| --- | --- | --- |
| [Qwen3.8-27B](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | **The** local Pi/coding host. Dense VLM, 262k hybrid KV is cheap. ✅ Tested on this hardware | You need the card for a specialist *and* you already have a 27B on the other GPU |
| [Qwen3-Coder-30B-A3B](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF) | Fast code specialist (3B active). Second card next to 27B | You want two 27Bs instead |
| [Qwen3.6-35B-A3B](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) | Fast general specialist (3B active). Second card next to 27B | You want two 27Bs, or the second job is code (use Coder-30B) |
| [Qwen3-Coder-Next](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF) 80B-A3B | Larger coding MoE on **one** card (Q4 ~50 GB, ⚠️ untested here) | You need it beside a 262k 27B on the **same** GPU |
| [Qwen3.8-Flash-Next](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 125B on **one** card (Q2 ~79 GB). Arch `qwen4exp`. Not a Pi host | You want it *and* 27B on the same card (Q4 is 111 GB) |

**3B active is speed, not a VRAM discount** — every expert still loads. It is also **not a free second lane.** While 35B-A3B or Coder-30B is generating, that GPU belongs to that process; a second `llama-server` on the same card still queues. Each token only *reads* ~3B of weights, so that one stream is faster and may not peg bandwidth the way 27B dense does. Spare silicon shows up as **more tok/s for this model**, not as a 27B running beside it at full speed.

Do not park [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) next to the 27B host (it queues the session you care about). Do not load Qwen3.8-2.4T-A95B. Skip older 3.5/3.6 **dense** 27B copies; 3.8-27B replaces them as the host. Extra Qwen3-VL is unnecessary — 27B is already a VLM ([mmproj](Dual-RTX6000-Qwen3.8.md#vision-mmproj)).

A third process only makes sense on GPU 0, and only if you accept that it **queues** with the specialist already there. That is leftover **storage**, not a fourth concurrent generator — [VRAM remaining](#vram-remaining-is-not-idle-gpu).

## 1. Download

**Two 27B** (Q6 is on this box as of 2026-09-03):

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

**27B + Coder** — keep the tested Q8 host; add the coding MoE:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models

hf download unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF \
  Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

**27B + 35B-A3B** — same Q8 host; general MoE instead of Coder:

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models

hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

Confirm filenames on Hugging Face before download. Catalog: [local-setup](../local-setup.md#model-catalog-hugging-face).

## 2. Build

Same cmake as the [Q8 primary](Dual-RTX6000-Qwen3.8.md#build). From `build/bin`:

```bash
./llama-server --list-devices
# CUDA0: ~97249 MiB  (~96 GB free)
# CUDA1: ~97246 MiB  (~95.6 GB free — desktop lives here)
```

## 3. Start the servers

`pkill` **once**. Two terminals. Start **GPU 1** (`:8081`) first; wait until `/health` is ok; then GPU 0. A second `pkill` kills every `llama-server`.

```bash
pkill -9 llama-server
```

`CUDA_VISIBLE_DEVICES` makes the visible card `CUDA0`, so `--main-gpu 0` is correct on **both**. `--threads 12` is llama-server’s **CPU** thread count (24 cores, two processes), not GPU compute. `--alias` **must** match the Pi JSON `id`.

Copy **one** pack below — both servers, then that pack’s `models.json` and `model-tiers.json`.

### Two 27B

Two Pi-quality 262k endpoints — [why](#why-pick-a-pack).

| GPU | Port | Alias | File | ctx | VRAM | Remaining |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| **0** | 8080 | `qwen3.8-27b-gpu0` | `Qwen3.8-27B-UD-Q6_K_XL.gguf` | 262144 | **~35 GB** | **~61 GB** |
| **1** (display) | 8081 | `qwen3.8-27b-gpu1` | same | 262144 | **~35 GB** | **~61 GB** |

| | |
| --- | --- |
| **VRAM** | **~70 GB** of 192 GB (~35 + ~35). **~122 GB** remaining |
| **Decode** | **~51 + 51 tok/s** when both generate (same as one instance alone). 27B dense on each card |
| **For** | Two interactive sessions, or parallel subagents that must match the host |
| **Status** | ✅ Load + parallel decode 2026-09-03 (35099 MiB / 35612 MiB after load) |

**GPU 1** (`:8081`) — start this first:

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --alias qwen3.8-27b-gpu1 \
  --port 8081 \
  --ctx-size 262144 --threads 12 \
  --host 127.0.0.1 --fit off \
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

**GPU 0** (`:8080`):

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=0 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --alias qwen3.8-27b-gpu0 \
  --port 8080 \
  --ctx-size 262144 --threads 12 \
  --host 127.0.0.1 --fit off \
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

`~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`):

```json
{
  "providers": {
    "llama-cpp-8080": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.8-27b-gpu0",
          "name": "Qwen3.8-27B Q6 GPU0",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    },
    "llama-cpp-8081": {
      "baseUrl": "http://127.0.0.1:8081/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.8-27b-gpu1",
          "name": "Qwen3.8-27B Q6 GPU1",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

`~/.pi/workflows/model-tiers.json`:

```json
{
  "tiers": {
    "small": "llama-cpp-8080/qwen3.8-27b-gpu0",
    "medium": "llama-cpp-8081/qwen3.8-27b-gpu1",
    "big": "llama-cpp-8081/qwen3.8-27b-gpu1"
  }
}
```

```text
/model llama-cpp-8081/qwen3.8-27b-gpu1
```

Second Pi session (other terminal): `/model llama-cpp-8080/qwen3.8-27b-gpu0`.

### 27B + Coder

Q8 Pi host + fast coding MoE — [why](#why-pick-a-pack).

| GPU | Port | Alias | File | ctx | VRAM | Remaining |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| **0** | 8080 | `qwen3-coder-30b` | `Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf` | 65536 | **~29 GB** | **~67 GB** |
| **1** (display) | 8081 | `qwen3.8-27b-gpu1` | `Qwen3.8-27B-UD-Q8_K_XL.gguf` | 262144 | **~40 GB** | **~55 GB** |

| | |
| --- | --- |
| **VRAM** | **~69 GB** of 192 GB (~29 + ~40). **~122 GB** remaining (~67 + ~55) |
| **Decode** | GPU 0 Coder **3B active** ~100–150 tok/s (forecast) + GPU 1 27B Q8 dense ~46 tok/s class. Both at once if they generate together |
| **For** | You on 27B; code subagents (audit / patch / review) on Coder |
| **Status** | ⚠️ VRAM and tok/s estimated. 27B Q8 is the tested Pi host |

**GPU 1** (`:8081`) — start this first:

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --alias qwen3.8-27b-gpu1 \
  --port 8081 \
  --ctx-size 262144 --threads 12 \
  --host 127.0.0.1 --fit off \
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

**GPU 0** (`:8080`):

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=0 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3-Coder-30B-A3B-Instruct-UD-Q6_K_XL.gguf \
  --alias qwen3-coder-30b \
  --port 8080 \
  --ctx-size 65536 --threads 12 \
  --host 127.0.0.1 --fit off \
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

`~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`):

```json
{
  "providers": {
    "llama-cpp-8080": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3-coder-30b",
          "name": "Qwen3-Coder-30B-A3B Q6 GPU0",
          "reasoning": false,
          "contextWindow": 65536,
          "maxTokens": 16384
        }
      ]
    },
    "llama-cpp-8081": {
      "baseUrl": "http://127.0.0.1:8081/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.8-27b-gpu1",
          "name": "Qwen3.8-27B Q8 GPU1",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

`~/.pi/workflows/model-tiers.json`:

```json
{
  "tiers": {
    "small": "llama-cpp-8080/qwen3-coder-30b",
    "medium": "llama-cpp-8081/qwen3.8-27b-gpu1",
    "big": "llama-cpp-8081/qwen3.8-27b-gpu1"
  }
}
```

```text
/model llama-cpp-8081/qwen3.8-27b-gpu1
```

### 27B + 35B-A3B

Q8 Pi host + fast general MoE — [why](#why-pick-a-pack).

| GPU | Port | Alias | File | ctx | VRAM | Remaining |
| ---: | ---: | --- | --- | ---: | ---: | ---: |
| **0** | 8080 | `qwen3.6-35b-a3b` | `Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf` | 262144 | **~50 GB** | **~46 GB** |
| **1** (display) | 8081 | `qwen3.8-27b-gpu1` | `Qwen3.8-27B-UD-Q8_K_XL.gguf` | 262144 | **~40 GB** | **~55 GB** |

| | |
| --- | --- |
| **VRAM** | **~90 GB** of 192 GB (~50 + ~40). **~101 GB** remaining (~46 + ~55) |
| **Decode** | GPU 0 35B-A3B **3B active** ~100–150 tok/s (forecast) + GPU 1 27B Q8 dense ~46 tok/s class. Both at once if they generate together |
| **For** | You on 27B; non-code fan-out (scan / classify / ingest / summary) on 35B-A3B |
| **Status** | ⚠️ VRAM and tok/s estimated. 27B Q8 is the tested Pi host |

**GPU 1** (`:8081`) — start this first:

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --alias qwen3.8-27b-gpu1 \
  --port 8081 \
  --ctx-size 262144 --threads 12 \
  --host 127.0.0.1 --fit off \
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

**GPU 0** (`:8080`):

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=0 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --alias qwen3.6-35b-a3b \
  --port 8080 \
  --ctx-size 262144 --threads 12 \
  --host 127.0.0.1 --fit off \
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

`~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`):

```json
{
  "providers": {
    "llama-cpp-8080": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.6-35b-a3b",
          "name": "Qwen3.6-35B-A3B Q8 GPU0",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    },
    "llama-cpp-8081": {
      "baseUrl": "http://127.0.0.1:8081/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "qwen3.8-27b-gpu1",
          "name": "Qwen3.8-27B Q8 GPU1",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

`~/.pi/workflows/model-tiers.json`:

```json
{
  "tiers": {
    "small": "llama-cpp-8080/qwen3.6-35b-a3b",
    "medium": "llama-cpp-8081/qwen3.8-27b-gpu1",
    "big": "llama-cpp-8081/qwen3.8-27b-gpu1"
  }
}
```

```text
/model llama-cpp-8081/qwen3.8-27b-gpu1
```

### Confirm

```bash
for p in 8080 8081; do echo -n ":$p "; curl -s http://127.0.0.1:$p/health || echo down; echo; done
# /v1/models on each port → n_ctx matches the table and the --alias you set
nvidia-smi
```

Load log per process: `n_ctx_seq` matches that group, `load_mode = none`. Short decode on **each** port. `nvidia-smi` should match the pack VRAM row. Stop all: `pkill -9 llama-server`. Stop one: `pkill -9 -f 'port 8080'`.

## 4. Pi notes

JSON for each pack is in [§3](#3-start-the-servers). Paths: `~/.pi/agent/models.json` and `~/.pi/workflows/model-tiers.json`. **One provider key per `baseUrl`.** Dummy `apiKey` is required or Pi hides the models in `/model`. `compat` keeps Pi from sending a `developer` role / `reasoning_effort` these local Qwen servers are not running. `--alias` **must** match the JSON `id`. `contextWindow` = that server’s `--ctx-size`. Open `/model` to reload.

Without a valid **medium**, workflow agents fail. A workflow that puts every `agent()` on `medium` never calls GPU 0. `parallel()` work has to name `small` (or `{ model: "llama-cpp-8080/…" }`) or the second card stays idle — [why](#why-pick-a-pack).

Status bar must show **262k** on the 27B host. New session after sampler/server changes (`/new`). Two humans / two heavy sessions: two `pi` processes, **different** `/model` ids on **different GPUs**.

Cloud providers (Grok / xAI via `/login`) can sit beside these keys. Optional: `pi install npm:pi-llama-cpp` for `/models` browse — **not required**. Connect rules: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Packages and research skills: [Pi graphs](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md).

## This box

- GPU 1 holds the display. 27B host there; specialist (if any) on GPU 0.
- One 27B saturates decode on a card. Remaining VRAM is empty weight storage — [why it is half the card](#vram-remaining-is-not-idle-gpu).
- Hybrid 27B KV is only the 16 full-attention layers — that is why 262k is cheap. KV does **not** shrink with weight quant. Specialists drop `--ctx-size` when a second GGUF must fit; the 27B host keeps 262k.
- MTP: dense 27B ~1.4–2.2×; MoE ~1.15–1.25×. Do not stack on first dual/MoE smoke. [Qwen3.8 optionals](Dual-RTX6000-Qwen3.8.md#qwen38-optionals).
- `--split-mode layer --tensor-split 96,96` is still **one** model — [Q8 alternate](Dual-RTX6000-Qwen3.8.md#this-box). `--parallel 2` on one server splits `--ctx-size` on **one** GPU. `--models-dir` router unloads; both weights would share whatever GPUs that process sees. `--n-cpu-moe` is a 24 GB offload trick.

| Approach | Use? |
| --- | --- |
| One `llama-server` per model + `CUDA_VISIBLE_DEVICES` | **This guide** |
| `--split-mode layer` | One model bigger than a card |
| `--parallel N` on one server | Extra slots on one GPU, not a second model |
| `--models-dir` router | Not concurrent resident models |

## See also

- Tested single-GPU Pi primary: [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-Qwen3.8.md)
- 3.6 MoE **split across both cards** (different layout): [Dual-RTX6000-Qwen3.6.md](Dual-RTX6000-Qwen3.6.md#alternate-multi-gpu-layer-split)
- Pi connect: [agentic harnesses](../agentic-harnesses.md)
- Workflows, Tavily, skills: [pi-coding-agent-graphs](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md)
- Unsloth: [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8) · [Qwen3.6](https://unsloth.ai/docs/models/qwen3.6) · [Coder-30B](https://unsloth.ai/docs/models/tutorials/qwen3-coder-how-to-run-locally) · [Coder-Next](https://unsloth.ai/docs/models/qwen3-coder-next) · [Flash-Next](https://unsloth.ai/docs/models/qwen3.8-next)

**Last Updated:** 2026-09-04
