# Windows RTX 3090 (24 GB) - LFM2.5-2.6B

> ⚠️ **Not yet tested** on this hardware with LFM2.5 (ported **2026-09-20** from the ✅ Jetson 64k pin). Confirm load → first decode → Pi tools before relying on it. Twin: [Jetson Orin LFM2.5](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md) (✅ Tested 2026-09-08 @ **64k**). Sibling on this box: [Ternary Bonsai 2 27B](Windows-RTX3090-Bonsai-2-27B.md) (⚠️ untested, **PrismML fork**) — do not copy those pins here.

**WSL2** (not native Windows) · CUDA sm_**86** (Ampere) · llama-cpp-turboquant. Paths on this box: **`~/AIML`** (models) · **`~/GitHub`** (engine) — same WSL2 convention as the [RTX 4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md), not `~/Documents/AIML`.

24 GB discrete GDDR6X vs the Jetson’s 8 GB unified. Weights are **2.87 GB**; q8/q8 KV at native **131072** is **1,088 MiB** (measured on the Jetson — KV geometry, not this SKU’s `nvidia-smi`). Native train length is the PRIMARY here because that KV is cheap, **not** because 128k was run on a 3090. The Jetson’s 2026-09-08 test stopped at 64k; 128k was a stretch there too. Confirm **load → first decode → a long prefill**, then Pi tools. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent). If you OOM, drop **batch** first, then `--ctx-size`; never bare `--fit on`. Do **not** drop `--n-predict` / `maxTokens` to 4096 — that is the first-turn Pi `length` stop ([below](#pi-truncation-on-the-first-turn)).

The [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B) does **not** recommend this model for agentic coding. On the Jetson it is a strong Pi daily driver anyway; this 24 GB pin is the same recipe with a native window. The GGUF template always opens `<think>`; for Pi **skills / tool calls** start with `reasoning` **false** ([below](#pi-coding-agent-modelsjson)).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (ported from Jetson ✅ 2026-09-08 **64k** q8/q8; 128k was untested there too) |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · Pi `contextWindow` **131072** · half-window **65536** |
| **KV** | `q8_0` / `q8_0` (do **not** turbo V — KV ~1.1 Gi at 128k) |
| **Output** | `--n-predict 16384` · Pi `maxTokens` **16384** (thinking counts against this) |
| **Sampling** | temp **0.1** · top_k **50** · repeat **1.1** (Liquid card) |
| **Thinking** | Template always opens `<think>` (omit server `--reasoning off`). **Pi skills/tools:** `reasoning` **false**, no `thinkingLevelMap`. **Pi traces:** `reasoning` **true**, `thinkingLevelMap.off` **null** |
| **Paths** | `~/AIML/models` · `~/GitHub/llama-cpp-turboquant` (WSL2) |

Need CUDA in WSL first? [local-setup.md](../local-setup.md) (WSL2 + Ubuntu). That file’s clone path is `~/Documents/GitHub` — **this box uses `~/GitHub`** (clone command in Build below). Hardware not in the table? [ai-assisted-setup.md](../ai-assisted-setup.md).

## Download

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/AIML/models
```

Q8_0 (2.87 GB) is the quality pin tested on the Jetson. **Q6_K** (2.22 GB) is that board’s headroom swap — skip it on 24 GB. Optional full-precision: [F16](#f16-optional) (5.4 GB).

## Build (Ampere / sm_86)

First time only (WSL2 path — not `~/Documents/GitHub` from [local-setup.md](../local-setup.md)):

```bash
mkdir -p ~/GitHub && cd ~/GitHub
git clone --depth 1 --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp-turboquant
```

If this box already has the [Bonsai PrismML](Windows-RTX3090-Bonsai-2-27B.md) tree under `~/GitHub/llama.cpp-prism`, leave it alone and clone turboquant next to it.

```bash
cd ~/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86" \
  -DGGML_CUDA_F16=ON \
  -DLLAMA_CURL=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). `"86"` is GPU compute capability (`sm_86`), not the CUDA toolkit version. The [4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) uses `"89"` (Ada); the [Jetson](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md) uses `"87"`. Omit `-DCMAKE_CUDA_ARCHITECTURES="86"` to autodetect. `FA_ALL_QUANTS` lengthens compile but covers quantized KV + flash-attn.

`-DLLAMA_CURL=ON` needs `libcurl4-openssl-dev` (`sudo apt install libcurl4-openssl-dev`). Not required for a local `--model` path — drop the flag if cmake fails on curl.

This is the **turboquant** tree (`~/GitHub/llama-cpp-turboquant`). Do not launch the [Bonsai PrismML](Windows-RTX3090-Bonsai-2-27B.md) binary for this GGUF. Confirm `nvidia-smi` **inside** WSL before debugging flags.

## PRIMARY command

Run from `~/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

cd ~/GitHub/llama-cpp-turboquant/build/bin

./llama-server \
  --model ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  --alias lfm2.5-2.6b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 256 \
  --batch-size 256 \
  --repeat-penalty 1.1 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 0 --temp 0.1 --top-k 50 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

Omit `--reasoning off` / `--reasoning-budget 0` — the chat template always starts the assistant with `<think>`. Omit `--cache-ram 0` (LFM is hybrid short-conv + GQA, not Qwen Gated-DeltaNet). Omit `--load-mode none` — mmap is the Jetson-tested path and this GGUF is 2.87 GB (the 4090 uses `none` for ~18 GB). Add it only if WSL mmap of the file is slow.

### Why these values (this box)

| Flag / value | Why |
| --- | --- |
| `--ctx-size 131072` | Native train length ([model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B); Liquid’s OpenClaw/Hermes examples also use 131072). Jetson PRIMARY is **64k** because 8 GB unified is tight; KV here is **1,088 MiB** at 128k ([budget](#context-budget), Jetson measurement). Liquid’s 32k is the *memory-constrained* example. Only **8 of 30** layers are GQA, so KV is cheap. **128k prefill on sm_86 is unmeasured** — if first decode OOMs, drop batch before ctx. Half-window **65536** if you share the GPU ([below](#this-box--only-if-primary-loads)) |
| `--n-predict 16384` | Pi’s published default (24 GB CUDA convention). Liquid’s OpenClaw example and the Jetson test use **8192**. Thinking counts against the cap — a higher budget can also let `<think>` run longer before the answer, not only give more room for tools. Start at 16384; if traces eat the turn, use the [tools JSON](#skills--tools-start-here) before raising further. Do not drop to 4096 |
| `--ubatch-size` / `--batch-size` **256 / 256** | Copied from the 4090 / this box’s Bonsai **27B** pin, not measured on LFM. Jetson used 64/128 because unified-memory prefill spikes SIGKILL; 256 was that board’s first OOM lever. On 24 GB discrete, 256 should be comfortable — still drop to **128** if 128k prefill OOMs, before cutting ctx |
| `--cache-type-k/v q8_0` | Quality default. Turbo V is the wrong lever (KV ~1.1 Gi at 128k); keep q8/q8 |
| `--n-gpu-layers 99` | Full GPU offload |
| `--main-gpu 0` | Discrete card (WSL2 CUDA convention; Jetson omits) |
| mmap (no `--load-mode`) | Jetson-tested for this file. `--load-mode none` is the 4090’s ~18 GB WSL lever — add it only if mmap is slow |
| `--fit off` | Keep pinned context agent-visible |
| `--host 127.0.0.1` | Local-only default. WSL2 localhost forwarding reaches Windows clients on the same machine |
| `--alias lfm2.5-2.6b` | Stable id for Pi / curl (matches `models.json`) |
| `--threads 0` | Auto CPU threads (4090 / Bonsai WSL2 pin) |
| LFM sampling | `temp 0.1` / `top_k 50` / `repeat-penalty 1.1` — [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B). Do not copy Qwen Pi (`0.6` / `top_k 20`) onto this template |

Confirm **`n_ctx` / `n_ctx_seq (131072)`** in the log or `GET /v1/models`, then one short decode (not only load). Expect `kv_cache_init` ≈ **1088 MiB** for q8/q8 at 128k (Jetson figure). `n_ctx_seq (131072) == n_ctx_train (131072)` is expected on this pin. Then `nvidia-smi` **after that decode** — prefill scratch is what OOMs, not idle KV. A 17×23 prompt does not prove 128k prefill.

Thinking should land in `reasoning_content`, answer in `content`:

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Then: new Pi session with real `ls` / `read`. Restart **both** the server and Pi after changing `--ctx-size` / `--n-predict` or `models.json`.

```text
log: n_ctx_seq (131072)
nvidia-smi   # inside WSL; MiB after load and after a short decode
```

### Context budget

Architecture from the card / `config.json`: 30 layers (22 short-conv + **8 GQA**), 8 KV heads, `head_dim` 64, `max_position_embeddings` **131072**. Conv state is ~0.3 MB and does **not** grow with context. llama.cpp q8_0 KV for this geometry is **1,088 MiB at 128k** (measured on the Jetson; scales linearly).

Weights **2.87 GB** + KV **~1.1 GB** is ~4 GB of *known* tensors. CUDA graphs, compute buffer, and **128k prefill scratch** are **not** measured on this SKU — the 6–8 GB row below is a guess, not `nvidia-smi`. WSL + Windows desktop also steal VRAM. llama.cpp pre-allocates the full KV at start; prefill can still spike above that.

| `--ctx-size` | q8/q8 KV (Jetson) | After load (est., unmeasured here) | Role |
| --- | --- | --- | --- |
| 65536 | ~544 MiB | **~5–7 GB** | Half-window — Jetson PRIMARY; use if sharing the GPU |
| **131072** | **1,088 MiB** | **~6–8 GB** (prefill may be higher) | **PRIMARY** — native train length |

f16 KV at 128k is ~2 Gi (still fine on 24 GB). turbo V saves little here and costs decode speed. Do not raise past **131072**.

Pi compaction default `reserveTokens` is **16384**. Compaction threshold is `contextWindow - 16384`:

| Pi `contextWindow` | Compacts after |
| --- | --- |
| 65536 | ~48k |
| **131072** | **~114k** |

Do **not** raise `reserveTokens` when you raise the window — that would eat the extra session length. Leave `--n-predict` / `maxTokens` at **16384** unless a turn still ends `length`.

Keep **q8/q8**.

### Pi truncation on the first turn

Pi’s **“Response was truncated before completion.”** is the UI for `stopReason === "length"` ([pi#7540](https://github.com/earendil-works/pi/pull/7540)) — generation hit the output allowance, not a crash.

Three documented constraints stacked on the old 16k / 4096 pin (the Jetson’s first failure mode). This 128k / 16384 pin removes the *window* and *clamp* parts; it does **not** guarantee a first turn will finish — always-on thinking can still hit `length` at 16384.

1. **Always-on thinking.** The Jinja template always starts `<think>`. Those tokens count against `--n-predict` / Pi `maxTokens` (same rule as [Qwen thinking](../agentic-harnesses.md#two-different-limits) and [Muse](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent)).
2. **Pi clamps the request.** It sends `max_tokens = min(model.maxTokens, contextWindow - promptTokens - 4096)` ([pi#7540](https://github.com/earendil-works/pi/pull/7540)). At `contextWindow` 16384 that is `12288 - promptTokens`. A coding-agent system prompt + tools can leave little room; `maxTokens` 4096 then caps it again. At **131072** the clamp is not the problem.
3. **Pi compaction default** `reserveTokens` is **16384** ([compaction.md](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md)). With `contextWindow` 16384 the threshold is `16384 - 16384 = 0`. At **131072** it is ~114k.

Confirm in the llama-server log: `tokens_predicted` / eval token count equal to the cap, and the chat chunk `finish_reason: length`. That is this bug, not an OOM.

`--n-predict` is only the **default** when the request omits `max_tokens`; Pi’s `max_tokens` overrides it ([llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [pr#22873](https://github.com/ggml-org/llama.cpp/pull/22873)). Raise **both** sides anyway so they stay aligned.

## This box — only if primary loads

**Single GPU.** ~2.9 GB weights + 128k q8/q8 KV *should* land well under 24 GB. That is KV math, not a 3090 measurement. Do not `--split-mode layer`. Windows desktop + WSL still steal some VRAM — check `nvidia-smi` **inside** WSL (the Windows Task Manager number is not the WSL view).

```bash
# Half-window — same PRIMARY, only:
#   --ctx-size 65536
# Pi contextWindow: 65536
```

**A) OOM on load / first decode:** (1) free desktop GPU apps / check WSL VRAM, (2) **batch 128** (128k prefill is the likely spike), (3) `--ctx-size 65536` + Pi 65536. Status bar must match. Do **not** swap to Q6_K first — that is the Jetson 8 GB lever.

**B) `n_ctx_seq` matches and VRAM has headroom:** you are already at native **131072**. Do not raise past train length. Optional: [F16](#f16-optional) or [DSpark](#dspark-optional) at the same pin.

**C) Turn-1 garbage / flaky tools:** not fixed by more context. New Pi session; use the [tools JSON](#skills--tools-start-here) (`reasoning` false); confirm PRIMARY is still q8/q8 and Liquid sampling; no DRY / no client sampling override. Pi tools were field-tested on the **Jetson**, not this SKU. Liquid’s native tool markup is Pythonic `<|tool_call_start|>` — Pi’s OpenAI tool loop is a different shape that happened to work on that board.

**D) Max output token limit / empty `content` with `finish_reason: length`:** primary is already 16384. Confirm Pi restarted. If thinking ate the budget, switch to the tools JSON rather than raising further. Prefer writing `reports/*.md` and a short chat summary. Do not add server `--reasoning off` to “fix” truncation (the template still opens `<think>`). Liquid’s OpenClaw example stays at **8192** — same command, only `--n-predict` / `maxTokens` if you want that pin.

| | Jetson Orin Nano Super 8 GB | This RTX 3090 24 GB | RTX 4090 Q4 27B |
| --- | --- | --- | --- |
| Weights | LFM Q8 **2.87 GB** | LFM Q8 **2.87 GB** | Qwen3.6 Q4 ~17.6 GB |
| PRIMARY pin | **64k q8/q8** (128k stretch) | **128k q8/q8** | **96k q8/q8** |
| `--n-predict` | **8192** | **16384** | **16384** |
| Batch | 64 / 128 | **256 / 256** | 256 / 256 |
| Engine | turboquant sm_87 | turboquant sm_**86** | turboquant sm_89 |
| Paths | `~/Documents/AIML` | **`~/AIML`** (WSL2) | `~/AIML` (WSL2) |

Same LFM2.5 Q8_0 file as the Jetson. Different **KV budget**, **batch**, and **WSL2 paths**. Same mmap load as the Jetson (not the 4090’s `--load-mode none`).

## F16 optional

Same PRIMARY flags. Only the file changes. 5.4 GB weights still leave most of 24 GB free. Q8_0 is the **Jetson-tested** quality pin; F16 is an untested bump on this box. Liquid also ships `LFM2.5-2.6B-BF16.gguf` at the same size — llama.cpp will convert either pack; do not treat F16 vs BF16 as a measured Ampere win. F16 is the file Liquid pairs with [DSpark](#dspark-optional).

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-F16.gguf \
  --local-dir ~/AIML/models
```

```bash
  --model ~/AIML/models/LFM2.5-2.6B-F16.gguf \
```

Change the Pi `name` suffix if you settle on F16. Smoke load → decode → Pi `ls`/`read` before treating it as the daily driver.

## DSpark optional

Liquid’s speculative-decoding sidecar for this model ([LFM2.5-2.6B-DSpark-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF); llama.cpp `#25173`). **Not part of PRIMARY.** Needs `--spec-type draft-dspark` on the **running binary** — confirm `./llama-server --help | grep -i dspark` on this turboquant build before downloading. The Muse Glimmer guides use `draft-dflash`; that is a different spec type. Untested on this 3090 / this fork.

```bash
hf download LiquidAI/LFM2.5-2.6B-DSpark-GGUF \
  LFM2.5-2.6B-DSpark-Q8_0.gguf \
  --local-dir ~/AIML/models
```

Add to the PRIMARY command (Q8_0 target + Q8_0 draft; Liquid’s example uses F16/F16):

```bash
  --model-draft ~/AIML/models/LFM2.5-2.6B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 10 \
  --spec-draft-n-min 0 \
```

Draft file is ~349 MB (Q8_0) / ~664 MB (F16). Speculative decode is exact under greedy verify — sampling still applies on the target. If the flag is missing, stay on PRIMARY without it.

## Pi Coding Agent `models.json`

Save **one** of these files to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Replace any earlier 16k / 32k / 64k copy. Open `/model` to reload (Pi reloads this file there; restart if the status bar is stale). `/new` after a pin or reasoning-shape change.

`maxTokens` ≤ `--n-predict` (16384). `contextWindow` = `--ctx-size` (131072). Both blocks use `id` **`lfm2.5-2.6b`** to match `--alias`. `compat.supportsReasoningEffort: false` is Liquid’s Pi note for this endpoint. Same llama-server command either way — do not add `--reasoning off`; the [chat template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja) always ends the generation prompt with `<|im_start|>assistant\n<think>`.

### Skills / tools (start here)

Field-tested on the Jetson while first building Pi **skills** and calling specific tools. Set `reasoning` **false** and **omit** `thinkingLevelMap`. Pi then does not run the reasoning-model path, so the agent invokes the tool instead of thinking about invoking the tool. Too much thinking for things that needed to just work.

This does **not** strip `<think>` from the GGUF. The model may still spend think tokens (they still count against `maxTokens` / `--n-predict`). What changes is Pi: no `thinkingLevelMap.off: null` lock, no reasoning-harness handling of `reasoning_content` before the tool loop.

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
          "name": "LFM2.5-2.6B Q8_0 (128k, tools) - RTX 3090",
          "reasoning": false,
          "contextWindow": 131072,
          "maxTokens": 16384,
          "compat": {
            "supportsReasoningEffort": false
          }
        }
      ]
    }
  }
}
```

### Thinking on

Pi’s documented shape for a model that cannot disable thinking ([models.md](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)): `reasoning` **true**, `thinkingLevelMap.off` **null** (hides Off in the UI). Use this when you want traces in `reasoning_content` and a clean `content` / tool split. Worse for early skill work: the model burns the output budget thinking about the skill instead of calling it.

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
          "name": "LFM2.5-2.6B Q8_0 (128k, think) - RTX 3090",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null
          },
          "contextWindow": 131072,
          "maxTokens": 16384,
          "compat": {
            "supportsReasoningEffort": false
          }
        }
      ]
    }
  }
}
```

If you take the [half-window](#this-box--only-if-primary-loads), only change `contextWindow` to **65536** and the `name` suffix. `/new` after switching between the two JSON shapes.

## WSL2

- `nvidia-smi` **inside** WSL. One long-lived server; don’t share the GPU heavily with Windows games / browsers. WSL may not see all 24 GB if the Windows desktop is using the card.
- Loopback `--host 127.0.0.1` is reachable from Windows browsers/clients on the same machine (WSL2 localhost forwarding).
- This guide’s Pi JSON path is **`~/.pi/agent/models.json` inside WSL** (Pi running in the Ubuntu distro). If Pi is installed on Windows itself, that file is `%USERPROFILE%\.pi\agent\models.json` — still point `baseUrl` at `http://127.0.0.1:8080/v1`.
- Workflows: [pi-coding-agent-graphs.md](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)

## Performance notes

- ⚠️ **Untested** on this SKU. Ported from the Jetson ✅ **64k / 8192** pin (128k was a stretch there, never recorded). This box’s daily pin is **128k / 16384** by VRAM math. For **skills / specific tool calls**, use the [tools JSON](#skills--tools-start-here) (`reasoning` false, no `thinkingLevelMap`). Keep the [think JSON](#thinking-on) when you want traces. Not Gemma’s 16k / 2048 and not Liquid’s 32k memory-constrained example. A 16k `contextWindow` plus `maxTokens` 4096 produces Pi’s **“Response was truncated before completion.”** on the first turn ([above](#pi-truncation-on-the-first-turn)).
- `n_ctx_seq (131072) == n_ctx_train (131072)` is expected on the daily pin.
- Q8_0 is the quality pin (tested on the Jetson). F16 is the optional bump, not the first download. Q6_K is the 8 GB headroom swap — skip it here. DSpark is a speed sidecar, not a quality lever.
- No tok/s number for this SKU — do not copy Jetson or H100 figures. Record `timings` from a real decode after `nvidia-smi` is stable.
- After pin changes, restart **llama-server and Pi** so the status bar matches `131072` / `16384` (or `65536` / `16384` on the half-window).
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## See also

- Jetson Orin Nano Super (8 GB, ✅ 64k): [Jetson-Orin-LFM2.5-2.6B.md](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md)
- This box, 27B ternary (PrismML fork, ⚠️ untested): [Windows-RTX3090-Bonsai-2-27B.md](Windows-RTX3090-Bonsai-2-27B.md)
- 24 GB CUDA WSL2 twin (Qwen3.6 Q4, turboquant): [Windows-RTX4090-Qwen3.6.md](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md)
- Model card: [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) · GGUF: [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
- Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent)

**Last Updated:** 2026-09-20 (128k / 16384 WSL2 pin; estimates labeled; ⚠️ untested on this box)
