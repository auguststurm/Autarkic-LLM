# Dual RTX 6000 Pro Max-Q (192 GB) — multiple models for Pi

> Two **isolated** `llama-server` processes, one per 96 GB card. That is how Pi Coding Agent gets two live endpoints (`/model`, workflows, subagents). This is **not** llama.cpp **router** mode (load/unload one process) and **not** `--split-mode layer` (one model across both GPUs).
>
> Single-model quality default remains the [Q8 27B primary](Dual-RTX6000-Qwen3.8.md) (✅ Pi, 2026-08-14). Use this file when the second card should *do a job*.

CUDA CC **12.0** · llama-cpp-turboquant · Ubuntu. Paths: model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant`.

| Pin (both recipes) | Value |
| --- | --- |
| **Context / KV / out** | `--ctx-size 262144` · `--fit off` · `q8_0`/`q8_0` · `--n-predict 16384` · `--parallel 1` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** · `--reasoning off` |
| **CPU** | `--threads 12` per process (24 physical cores) |
| **Isolation** | `CUDA_VISIBLE_DEVICES` · `--main-gpu 0` inside each process · `--host 127.0.0.1` |

## Which recipe?

| | GPU 0 (`:8080`) | GPU 1 (`:8081`, display) | Pi gets | Status |
| --- | --- | --- | --- | --- |
| **A — two coding agents** | 3.8-27B **Q6** | 3.8-27B **Q6** | Two identical 262k coders | ✅ Load + parallel decode **2026-09-03** (~51 tok/s **each**) |
| **B — Pi + Tavily / graphs** | 3.6-35B-A3B **Q8** (MoE, 3B active) | 3.8-27B **Q8** | Fast `small` + best local coder as host / `medium` / `big` | ⚠️ Pairing not load-tested. 27B Q8 **is** the tested Pi host |

**Start with B** if you use [search-topic-research](../_Pi-Coding-Agent-Graphs/example-skills/search-topic-research/) or `parallel()` specialists. **Use A** for two humans or two heavy coding sessions that both need 27B. Do not run A “so Tavily can use two models” — the shipped skill is sequential and will ignore the second 27B.

Pi official docs describe a **`--models-dir` router**. That unloads/loads on demand in **one** process. Two ~35–40 GB models will not stay resident, and both would see both GPUs. Skip it here.

## 1. Download

**Recipe A** (Q6 is already on this box as of 2026-09-03):

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

**Recipe B** (Q8 27B is the tested primary file; add the MoE):

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models

hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

Q8 27B ~31.5 GB · Q6 27B ~25.9 GB · 35B-A3B Q8 ~38.5 GB. Catalog: [local-setup](../local-setup.md#model-catalog-hugging-face).

## 2. Build

Same cmake as the [Q8 primary](Dual-RTX6000-Qwen3.8.md#build). From `build/bin`:

```bash
./llama-server --list-devices
# CUDA0: ~97249 MiB  (~96 GB free)
# CUDA1: ~97246 MiB  (~95.6 GB free — desktop lives here)
```

## 3. Start both servers

`pkill` **once**. Start GPU 0; wait until `http://127.0.0.1:8080/health` is ok; then GPU 1. A second `pkill` kills GPU 0.

Shared flags (both recipes, both processes): `--fit off --n-gpu-layers 99 --main-gpu 0 --load-mode none --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 --jinja --flash-attn on --no-context-shift --parallel 1 --ubatch-size 1024 --batch-size 1024 --reasoning off --reasoning-budget 0 --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0 --frequency-penalty 0.0 --repeat-last-n 64 --threads 12 --n-predict 16384 --kv-unified --log-verbosity 1 --ctx-size 262144 --host 127.0.0.1`

`CUDA_VISIBLE_DEVICES` makes the visible card `CUDA0`, so `--main-gpu 0` is correct on **both**.

### Recipe A — two Q6 27B (measured)

**Terminal A — GPU 0**

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=0 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q6_K_XL.gguf \
  --alias qwen3.8-27b-gpu0 \
  --port 8080 \
  --host 127.0.0.1 --ctx-size 262144 --fit off \
  --n-gpu-layers 99 --main-gpu 0 --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
  --jinja --flash-attn on --no-context-shift --parallel 1 \
  --ubatch-size 1024 --batch-size 1024 \
  --reasoning off --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --frequency-penalty 0.0 --repeat-last-n 64 \
  --threads 12 --n-predict 16384 --kv-unified --log-verbosity 1
```

**Terminal B — GPU 1** — same command except `CUDA_VISIBLE_DEVICES=1`, `--alias qwen3.8-27b-gpu1`, `--port 8081`.

### Recipe B — MoE `small` + 27B Q8 host (Pi + Tavily)

**Terminal A — GPU 0 (fan-out)**

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=0 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf \
  --alias qwen3.6-35b-a3b \
  --port 8080 \
  --host 127.0.0.1 --ctx-size 262144 --fit off \
  --n-gpu-layers 99 --main-gpu 0 --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
  --jinja --flash-attn on --no-context-shift --parallel 1 \
  --ubatch-size 1024 --batch-size 1024 \
  --reasoning off --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --frequency-penalty 0.0 --repeat-last-n 64 \
  --threads 12 --n-predict 16384 --kv-unified --log-verbosity 1
```

**Terminal B — GPU 1 (you + coding + Findings/Report)**

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant/build/bin

CUDA_VISIBLE_DEVICES=1 ./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --alias qwen3.8-27b-gpu1 \
  --port 8081 \
  --host 127.0.0.1 --ctx-size 262144 --fit off \
  --n-gpu-layers 99 --main-gpu 0 --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
  --jinja --flash-attn on --no-context-shift --parallel 1 \
  --ubatch-size 1024 --batch-size 1024 \
  --reasoning off --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --frequency-penalty 0.0 --repeat-last-n 64 \
  --threads 12 --n-predict 16384 --kv-unified --log-verbosity 1
```

### Confirm

```bash
curl -s http://127.0.0.1:8080/health; echo
curl -s http://127.0.0.1:8081/health; echo
# /v1/models  →  n_ctx 262144 and the --alias you set
nvidia-smi   # Recipe A: ~35 GB each (measured). Recipe B: MoE ~50 GB class + 27B Q8 ~40 GB class
```

Load log per process: `n_ctx_seq (262144)`, `load_mode = none`. Then a short decode on **each** port. Stop both: `pkill -9 llama-server`. Stop one: `pkill -9 -f 'port 8081'`.

## 4. Pi `models.json`

Save to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). **One provider key per `baseUrl`.** Dummy `apiKey` is required or Pi hides the models in `/model`. `compat` keeps Pi from sending a `developer` role / `reasoning_effort` these local Qwen servers are not running. Open `/model` to reload (Pi rereads this file then).

**Recipe B** (copy this if you are doing Tavily / graphs):

```json
{
  "providers": {
    "llama-cpp-gpu0": {
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
          "name": "Qwen3.6-35B-A3B Q8 GPU0 (fan-out)",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    },
    "llama-cpp-gpu1": {
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
          "name": "Qwen3.8-27B Q8 GPU1 (host / coding)",
          "reasoning": false,
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

**Recipe A:** same shape, both ids `qwen3.8-27b-gpu0` / `qwen3.8-27b-gpu1`, both names Q6, both `baseUrl`s 8080/8081.

`--alias` on the server **must** match the JSON `id` (that string is what Pi puts in `"model"`).

Built-in cloud providers (Grok / xAI via `/login`) can sit **beside** these keys. `/model` lists local and cloud together. Do **not** point workflow `small`/`medium` at Grok — Tavily already did search; Grok is a manual `/model` hop for “are we solving the right problem.”

Optional: `pi install npm:pi-llama-cpp` and `llamaSettings.servers` for `/models` browse across 8080/8081. **Not required** — `models.json` is what `/model` and `/workflows-models` use in this repo.

## 5. Workflow tiers

```text
/workflows-models
```

Writes `~/.pi/workflows/model-tiers.json`. Ids are `providerKey/modelId`.

**Recipe B**

```json
{
  "tiers": {
    "small": "llama-cpp-gpu0/qwen3.6-35b-a3b",
    "medium": "llama-cpp-gpu1/qwen3.8-27b-gpu1",
    "big": "llama-cpp-gpu1/qwen3.8-27b-gpu1"
  }
}
```

**Recipe A** — point all three at gpu0, or `medium`→gpu0 and `big`→gpu1 if you later `parallel()` two 27Bs.

Without a valid **medium**, [search-topic-research](../_Pi-Coding-Agent-Graphs/example-skills/search-topic-research/) fails even when Tavily succeeded.

## 6. Use Pi every day

Packages (once): [graphs doc](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md).

```bash
pi install npm:@tavily/pi-extension
pi install npm:@quintinshaw/pi-dynamic-workflows
pi install npm:pi-subagents
```

Tavily key in the **process** env (`source .env` then `TAVILY_API_KEY=… pi`). The extension does not read `.env` itself.

### Coding (one session)

```text
/model llama-cpp-gpu1/qwen3.8-27b-gpu1
```

That is GPU 1, Recipe B (or either GPU on Recipe A). Status bar must show **262k**. New session after sampler/server changes (`/new`). Real `ls` / `read` before trusting tools. [Agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

### Two agents at once

Two terminals, two `pi` processes, **different** `/model` ids — one per GPU. That is the only way two generations run at full tok/s. One Pi session = one model at a time; `/model` only hops the *next* turns.

### Tavily research — [Best blend for Pi + Tavily](#best-blend-for-pi--tavily-the-decision)

Host owns **all** `web_search` / `web_fetch`. Workflow agents are file-only (so they cannot invent URLs). The **example skill as shipped** sets `concurrency: 1` and every `agent()` on **medium** — GPU 0 will sit idle until you fork the four calls (copy the skill first):

```javascript
await agent(`Librarian…`, { label: "ingest", tier: "small" });
await agent(`Findings…`, { label: "findings", tier: "medium" });
await agent(`Hostile review…`, { label: "skeptic", tier: "small" });
await agent(`Write the FINAL report…`, { label: "report-writer", tier: "big" });
```

Then `/skill:search-topic-research <topic>` from a **27B** host session (`/new` per report).

| Who | Model | Why |
| --- | --- | --- |
| Host + Tavily + pack | 27B Q8 GPU1 | Query quality, paths, 70k+ packs. Only Pi-tested coder on this box |
| Ingest (≤40 lines) | 35B-A3B GPU0 | Fast librarian |
| Findings (claim+URL) | 27B Q8 | Where cheap models invent |
| Skeptic | 35B-A3B GPU0 | Different weights than Findings |
| Report (8k–16k file) | 27B Q8 | Isolated long decode |

### `parallel()` / pi-subagents

`agent("…", { tier: "small" })` or `{ model: "llama-cpp-gpu0/qwen3.6-35b-a3b" }` is what hits the second card. If every `agent()` uses medium, you bought VRAM and still serialize. Tiers only have three names; a fourth endpoint needs explicit `model:`.

## 7. What “25 GB” and two GPUs actually mean

The **Q6 GGUF is ~25.9 GB on disk** — weights on **one** card, not a split. Measured **2026-09-03** (Recipe A, 262k q8/q8):

| State | GPU 0 | GPU 1 |
| --- | --- | --- |
| After load | **35099 MiB** | **35612 MiB** |
| Parallel decode | ~51 tok/s each, both ~99% / ~300 W | same 51 tok/s as one instance alone |
| Remaining | **~62 GB** | **~61 GB** |

Hybrid 27B KV is only the 16 full-attention layers — that is why 262k is cheap. One 27B already pegs a card. A **second process on the same GPU** adds a queue, not tok/s. A **second GPU** doubles throughput. Recipe B: ~46 tok/s coding (Q8 forecast) **plus** ~100–150 tok/s fan-out (MoE forecast, 3B active) when both fire.

## Why not the other dual-GPU knobs

| Approach | Use? |
| --- | --- |
| Two `llama-server`s + `CUDA_VISIBLE_DEVICES` | **This guide** |
| `--split-mode layer --tensor-split 96,96` | One model bigger than a card — [Q8 alternate](Dual-RTX6000-Qwen3.8.md#this-box) |
| `--parallel 2` on one server | Splits `--ctx-size` on **one** GPU |
| `--models-dir` router | Not concurrent; both weights would share whatever GPUs that process sees |
| `--n-cpu-moe` | 24 GB offload trick. You have 96 GB |

## Best blend for Pi + Tavily (the decision)

This is Recipe B plus the skill fork in §6. Full job table and anti-patterns:

- Do not 2× Q6 for this workload (host blocked, all phases on medium).
- Do not put Tavily on the MoE.
- Do not give workflow agents `web_search`.
- Do not replace GPU 1’s 27B with Flash-Next Q2.
- Qwen3.8-2.4T-A95B does not fit — that is Grok.
- Escalate to Grok **after** a local pack/report, for system design / HLE-shaped questions, not for another SERP.

## MoE menu (this box)

**“3B active” is compute, not VRAM** — every expert still loads.

| Model | Active | Weights | Beside 27B Q8? | Pi |
| --- | ---: | --- | --- | --- |
| **Qwen3.6-35B-A3B** | 3B | Q8 **38.5 GB** | **Yes** | Recipe B `small` |
| Older 30B/35B-A3B | 3B | similar | Yes | Skip |
| Qwen3.8-Flash-Next | 6B + 51B n-gram | Q2 79 · Q3 90 · **Q4 111** | Q4 **no**; Q2/Q3 razor; arch `qwen4exp` | Weekend A/B, not daily |
| 2.4T-A95B | 95B | 1-bit ~397 GB | No | Grok |

Two 35B-A3Bs and no 27B = two librarians, no coder.

## How many 27B? (throughput vs concurrency)

Identical copies only. KV does **not** shrink with quant. Keep 262k for Pi (graphs overflow 32k/64k).

| Setup | Decode when it matters | Pi | Fit |
| --- | --- | --- | --- |
| 1× Q8 | ~46 tok/s, GPU1 idle | One session | Yes — [primary](Dual-RTX6000-Qwen3.8.md) |
| 2× Q6 1+1 | **~51+51** measured | Two coding sessions | **Recipe A** |
| 2× Q8 1+1 | ~46+46 | Same, max quality | Likely |
| 1× Q8 + 1× 35B-A3B | coding + fast fan-out | **Recipe B** | The useful blend |
| 2+2 Q6 | still ~102 tok/s **box** when all four generate | Only if `parallel()` is actually 4-wide | Likely VRAM, half tok/s per model |
| 6× Q4 or 64k ctx | crawl | Graphs overflow | No |

Solo Q6 decode **~51 tok/s** measured (80-token completion). Q8/Q5/Q4 are bandwidth forecasts. 16k Report ≈ **5.4 min** at Q6 isolated, **~11 min** if that 27B shares a GPU.

## This box

- PRIMARY isolation is one process per card. Packed 2-per-GPU is the table above, not Recipe A/B.
- GPU 1 holds the display (~1 GB).
- MTP: dense 27B ~1.4–2.2×; MoE ~1.15–1.25×. Do not stack on first dual/MoE smoke. [Qwen3.8 optionals](Dual-RTX6000-Qwen3.8.md#qwen38-optionals).

## See also

- Tested single-GPU Pi primary: [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-Qwen3.8.md)
- 3.6 MoE **split across both cards** (different layout): [Dual-RTX6000-Qwen3.6.md](Dual-RTX6000-Qwen3.6.md#alternate-multi-gpu-layer-split)
- Pi connect: [agentic harnesses](../agentic-harnesses.md) · Tavily + skills: [pi-coding-agent-graphs](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md)
- Unsloth: [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8) · [Qwen3.6](https://unsloth.ai/docs/models/qwen3.6)

**Last Updated:** 2026-09-03 (multi-model Pi recipes A/B; Q6 dual measured; Tavily blend)
