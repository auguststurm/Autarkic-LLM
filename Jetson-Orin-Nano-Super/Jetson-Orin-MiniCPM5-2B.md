# Nvidia Jetson Orin Nano Super - MiniCPM5-2B

> ⚠️ **Not yet tested** on this hardware with MiniCPM5-2B. Nothing in PRIMARY is a MiniCPM measurement on this board. Numbers below are tagged: **OpenBMB**, **geometry** (`config.json`), **this Jetson (LFM/Gemma)**, **other GPU (X)**, or **estimate**. Confirm **load → first decode → Pi tools**, then report via issue/PR.

8 GB LPDDR5 (~7.3 Gi usable — **this Jetson**, LFM Q8 @ 32k snapshot 2026-09-08) · Ampere sm_**87** · llama-cpp-turboquant. **Paths:** `~/Documents/AIML/models/minicpm5` · `~/Documents/GitHub/llama-cpp-turboquant`. Pi: [agentic harnesses — MiniCPM5-2B](../agentic-harnesses.md#minicpm5-2b--pi-coding-agent). If you OOM, drop `--ctx-size` or swap to Q4_K_M, never bare `--fit on`. Do **not** copy Gemma sampling (temp 0.75) or LFM sampling (temp 0.1). LFM’s **64k** on this box is real — MiniCPM does **not** get that window for free ([why](#lfm-64k-vs-this-pin)).

Dense **LlamaForCausalLM** (~2.52B, 42 layers, GQA 16 Q / 2 KV, `head_dim` 128, native **131072**). Official GGUF is **text-only** — no `mmproj` in [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF). OpenBMB documents thinking as a **toggle** (`--reasoning on|off`), unlike LFM/Muse; that switch is **untested** on this Jetson build. PRIMARY below **forces off** for Pi skills/tools. Do **not** pass `--chat-template-kwargs '{"enable_thinking":…}'` — current llama-server deprecates that in favor of `--reasoning`.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (OpenBMB llama.cpp + this box’s CUDA) |
| **Weights** | `MiniCPM5-2B-Q8_0.gguf` (2.68 GB) |
| **Catalog** | [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) · [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) |
| **Context** | `--ctx-size 32768` (`--fit off`) · Pi `contextWindow` **32768** · 64k is [D1](#d-one-axis-follow-ups-only-after-ab), not a copy of LFM |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 8192` · Pi `maxTokens` **8192** (thinking counts against this if you turn it on) |
| **Sampling** | temp **1.0** · top_p **0.95** · min_p **0.0** (OpenBMB MiniCPM5-2B Think table / `generation_config.json`) |
| **Thinking** | **PRIMARY off** for Pi tools: `--reasoning off` + `--reasoning-budget 0`. Think-on recipe [below](#thinking-on-optional). |
| **Vision** | **Off** — this GGUF has no projector ([below](#vision-not-this-checkpoint)) |
| **Paths** | `~/Documents/AIML/models/minicpm5` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need the engine? [local-setup.md](../local-setup.md) (JetPack / CUDA). CMake below is the **same command that built Gemma/LFM on this box** — MiniCPM is stock **llama** (OpenBMB; no extra arch tag), so that build should load it, but that load is **untested**.

## Download

Official files on [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) (confirmed 2026-09-14):

| File | Size | Role |
| --- | --- | --- |
| `MiniCPM5-2B-Q8_0.gguf` | 2.68 GB | **PRIMARY** — OpenBMB: “very small quality drop vs F16” |
| `MiniCPM5-2B-Q4_K_M.gguf` | 1.56 GB | OpenBMB **edge / mobile** rec; [headroom swap](#q4_k_m-alternative) if Q8 OOMs or you want more ctx |
| `MiniCPM5-2B-F16.gguf` | 5.04 GB | Reference; **not** a daily pin on 8 GB unified |

```bash
hf download openbmb/MiniCPM5-2B-GGUF \
  MiniCPM5-2B-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models/minicpm5
```

If the three quants are already in that directory, skip the download. There is **no** `mmproj-*.gguf` in this repo — do not invent one.

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

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). `FA_ALL_QUANTS` lengthens compile; it is what the other Jetson guides use for q8 KV + flash-attn. OpenBMB’s path is stock llama.cpp; this repo stays on the fork so `turbo*` types remain available **if** a later stretch needs them (untested on MiniCPM).

## PRIMARY command

Run from `~/Documents/GitHub/llama-cpp-turboquant/build/bin`. **MAXN SUPER** first: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

Think **off**, vision **off**, Q8_0 @ 32k. OpenBMB’s GGUF card smoke is:

```bash
llama-server -m MiniCPM5-2B-F16.gguf -a MiniCPM5-2B --port 8080 -ngl 99 -c 8192 --jinja
```

(curl `temperature=1.0, top_p=0.95, min_p=0.0`, `max_tokens=256`.) That `-c 8192` is below Pi’s `reserveTokens` **16384**. Flags below are only what this **model card**, this **42-layer GQA-2 KV**, or this **8 GB unified** board actually requires. Defaults that already match (`--host 127.0.0.1`, `--port 8080`, `--top-p 0.95`, `--jinja` on, `--context-shift` off, `--threads -1`) are omitted except `--jinja` / `--host` (OpenBMB + autarky). **`--threads 0` is not auto** (`--help` default is **-1**) — do not copy that from the other Jetson files.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/minicpm5/MiniCPM5-2B-Q8_0.gguf \
  --alias minicpm5-2b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 \
  --fit off \
  --n-gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --parallel 1 \
  --ubatch-size 64 \
  --batch-size 128 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 1.0 --top-p 0.95 --min-p 0.0 \
  --n-predict 8192 \
  --kv-unified
```

Omit `--load-mode none` / `--no-mmap` (default mmap pages the GGUF on 8 GB). Omit `--mmproj`. Omit `--chat-template-kwargs`. Omit `--log-verbosity` (default **3** = info). Omit `--no-context-shift` (already the default). Omit `--threads` (default **-1** = all cores).

### Why these values

Checked against `./llama-server --help` on llama-cpp-turboquant. “Default” below is that help text.

| Flag / value | Default | Why this value |
| --- | --- | --- |
| `MiniCPM5-2B-Q8_0.gguf` (2.68 GB) | — | Highest official quant (**OpenBMB**: Q8 ≈ F16 quality). File size is in the same class as LFM Q8 (2.87 GB) which **loaded** here — that is **not** a MiniCPM load proof. Q4_K_M is OpenBMB’s edge rec / [headroom swap](#q4_k_m-alternative) if Q8 OOMs. F16 (5.04 GB) + even 32k q8 KV (~0.7 Gi, **geometry**) + OS does not fit ~7.3 Gi. |
| `--ctx-size 32768` | `0` = train length **131072** | Unset would request 131k. f16 KV at 131k is **5.64 GB** (geometry + [3060 measurement](#field-reports-x)); Q8+that cache is an 8 GB **discrete** card at the limit, not unified with an OS. `--c 8192` (OpenBMB smoke) and `16384` make Pi compaction threshold ≤ 0 (`reserveTokens` 16384 — [measured on this board](Jetson-Orin-LFM2.5-2.6B.md#pi-truncation-on-the-first-turn)). **32768** is OpenBMB’s own long-context step ([Ollama](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/ollama.md) `num_ctx 32768`, [vLLM](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/vllm.md) 8192/32768). q8 KV ≈ **0.70 Gi** → weights+KV ≈ **3.4 GB** before scratch. 64k (KV ≈ 1.40 Gi, total ≈ 4.1 GB) is the first stretch **after** 32k decode is clean — not PRIMARY without a measurement. |
| `--cache-type-k/v q8_0` | **f16 / f16** | f16 KV at 32k is **~1.41 Gi** (**geometry**; +0.7 Gi vs q8). That 0.7 Gi is prefill margin. Prefill SIGKILL on this board is **measured on Gemma/LFM**, not MiniCPM. q8_0 is a llama.cpp type, not turbo. If 32k q8 is stable and tools feel soft, try **V f16** (keep K q8). Turbo V is a later capacity lever (**untested** on MiniCPM). |
| `--n-predict 8192` | **-1** (unlimited) | Unlimited plus a loop can fill the window. OpenBMB smoke is 256/2048. Pi `maxTokens` must not exceed this. **8192 is a Pi-agent choice, not a MiniCPM measurement.** Raise **both** sides to 16384 if a turn ends `length`. |
| `--ubatch-size 64` `--batch-size 128` | **512 / 2048** | Scratch scales with ubatch. 2048/1024 is a [3060 12 GB](#field-reports-x) recipe. **This board SIGKILLed on prefill at large batch with Gemma/LFM** (~3 GB weights). MiniCPM at 2048 is **untested**. 256 is the first raise if 64/128 is stable. |
| `--n-gpu-layers 99` | **auto** | 42 layers. OpenBMB: **99 = all**. Auto on unified memory **can** leave layers on CPU (general llama.cpp; not MiniCPM-measured). |
| `--fit off` | **on** | `--fit on` will shrink an unset/oversized ctx. Pi needs the pin you typed. |
| `--parallel 1` | **-1** (auto) | Auto can open extra slots and **split** `-c`. One Pi session = one slot. |
| `--kv-unified` | on only if `-np` is auto | We set `-np 1`, so the auto-on default does **not** apply. One shared KV buffer for that single slot. |
| `--flash-attn on` | **auto** | Other Jetson guides + the 3060 MiniCPM run use FA with quantized KV. Confirm **FA on** in the load log. Do not leave auto. |
| `--jinja` | **enabled** | OpenBMB’s server line. Chat template + tools. Harmless if already default. |
| `--reasoning off` `--reasoning-budget 0` | **auto** / **-1** | OpenBMB’s published 2B mode is Think; they also document a No-think toggle. **`--reasoning auto` on this GGUF+server is unverified** — PRIMARY forces off. [B. curl](#b-first-decode-curl-thinking-off) is the check (`reasoning` chars = 0). Unlike LFM, OpenBMB says this checkpoint **can** switch. |
| `--temp 1.0` `--top-p 0.95` `--min-p 0.0` | **0.80 / 0.95 / 0.05** | Official MiniCPM5-2B triplet. **min_p 0.05 is the llama.cpp default OpenBMB says locks repetition loops.** top_p already matches default; keep it with the triplet. Do **not** set `--top-k` (OpenBMB doesn’t; default 40 stands). Do **not** copy Gemma 0.75 or LFM 0.1. Repeat penalty stays **1.0** (default/off) until [loops](#repetition). |
| `--alias minicpm5-2b` | filename | Pi `id` / curl `model`. |
| `--host 127.0.0.1` | already loopback | Explicit autarky. |

Confirm **`n_ctx_seq (32768)`** in the server log (`grep n_ctx_seq`) or:

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/models
```

Then one short decode (not only load). q8/q8 KV **~0.7 Gi at 32k** is **geometry**, not a measured `kv_cache_init` — write the log line in [A](#a-load-primary-no-other-changes).

Then: new Pi session with real `ls` / `read`. Restart **both** the server and Pi after changing `--ctx-size` / `--n-predict` or `models.json`.

## On-box test log (fill this in)

Status stays ⚠️ until **A + B + C** all pass on this Jetson. Change **one** axis per row after A is green. `pkill -9 llama-server` between rows. Copy log lines, do not paraphrase.

### What this PRIMARY is vs other recipes

| | OpenBMB GGUF smoke | 3060 field (X) | **This Jetson PRIMARY** |
| --- | --- | --- | --- |
| Weights | F16 (card) / Q4_K_M (skill) | Q8_0 or Q4_K_M | **Q8_0** |
| `-c` | **8192** | **131072** | **32768** |
| KV | default f16 | **f16/f16** | **q8/q8** |
| Batch | default 2048/512 | **2048/1024** | **128/64** |
| Thinking | unset (auto → Think) | unset | **`--reasoning off`** |
| `--fit` | default **on** | unset | **off** |
| `-np` | default auto | **1** | **1** |
| Sampling | curl 1.0 / 0.95 / **min_p 0.0** | 1.0 / 0.95 (no min_p) | **1.0 / 0.95 / min_p 0.0** |
| `--threads` | default **-1** | unset | **omit** (do not use `0`) |
| `--no-context-shift` | default already off | unset | **omit** |

### A. Load (PRIMARY, no other changes)

```bash
sudo nvpmodel -m 2 && sudo jetson_clocks
free -h    # note swap; if 0, enable zram before stretching ctx
# run PRIMARY from build/bin
```

| Check | Expect | Yours |
| --- | --- | --- |
| `n_ctx_seq` | **32768** (not 8192, not 131072) | |
| mmap | present; not `load_mode = none` (wording **untested** on this build) | |
| flash attn | **on** | |
| `kv_cache_init` | record actual; **geometry** ≈ 700 MiB q8/q8 @ 32k | |
| `free -h` after load | still some available; no SIGKILL | |
| `./llama-server --version` | write it | |

Load-only is not a pass. If this row fails: MAXN SUPER, close apps, then drop `-c` to **16384** (and Pi `contextWindow`) before touching quant.

### B. First decode (curl, thinking off)

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minicpm5-2b","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}],"max_tokens":64}' \
| python3 -c "import json,sys; r=json.load(sys.stdin); m=r['choices'][0]['message']; \
print('content  :', (m.get('content') or '')[:200]); \
print('reasoning:', len(m.get('reasoning_content') or ''), 'chars'); \
print('finish   :', r['choices'][0].get('finish_reason')); \
print('usage    :', r.get('usage'))"
```

| Check | Expect | Yours |
| --- | --- | --- |
| `content` | 17×23 is **391**. Temp 1.0 may add prose — fail only if the math is wrong or `content` is empty | |
| `reasoning` chars | **0** if `--reasoning off` worked. Non-zero = off did not take (**do not assume** it works until this row) | |
| `finish` | `stop`, not `length` | |
| first decode | no OOM / SIGKILL | |
| tok/s if logged | record | |

If `reasoning` is non-zero, `--reasoning off` did not take on **this** build — stop and fix that before Pi (do not assume OpenBMB’s toggle). If `finish=length` with empty content, raise **both** `--n-predict` and (later) Pi `maxTokens`.

### C. Pi tools (skills JSON)

Install the [tools `models.json`](#skills--tools-start-here). `/model` reload, **`/new`**. Ask for a real `ls` then `read` of a file.

| Check | Expect | Yours |
| --- | --- | --- |
| status bar | 32768 / 8192 | |
| `ls` | actual tool call, not a narrated plan | |
| `read` | file bytes back | |
| first-turn truncate? | no “Response was truncated before completion.” | |

**A+B+C green → mark this guide ✅ Tested** (date + `--version` + `kv_cache_init`). Until then leave the banner ⚠️.

### D. One-axis follow-ups (only after A+B)

Same PRIMARY, **one** edit per row. Re-run B (and C if you will daily-drive that row).

| # | Change | Why | Load | Decode | Pi tools | Keep? |
| --- | --- | --- | --- | --- | --- | --- |
| **D1** | `--ctx-size 65536` + Pi `contextWindow` **65536** (same `maxTokens` 8192) | LFM’s daily window on this box. MiniCPM KV ≈ **1.4 Gi** (**geometry**; LFM ~544 MiB measured). Expect `n_ctx_seq (65536)`; write actual `kv_cache_init`. zram if swap is 0. If load or first decode SIGKILLs, stay on 32k (or D4 then 64k). | | | | **promote to PRIMARY if A+B+C pass at 64k** |
| D2 | `--reasoning on`, drop `--reasoning-budget 0` + [think JSON](#thinking-on) | Official Think mode | | | | optional traces |
| D3 | `--cache-type-v f16` (K stays q8) | Quality vs +0.7 Gi at 32k | | | | |
| D4 | Q4_K_M, same ctx as the row you kept | OpenBMB edge; buys KV headroom for 64k | | | | |
| D5 | `-b 256 -ub 128` | Faster prefill if 64/128 was easy | | | | |
| D6 | `--repeat-penalty 1.05` | Only if B/C **loops** | | | | |

Do **not** stack D-rows on the first try. Do **not** enable DSpark until PRIMARY is daily-stable (`./llama-server --help | grep draft-dspark` first — **may be absent** on this turboquant build).

### E. Record (for the ✅ banner)

```text
date:
llama-server --version:
n_ctx_seq:
kv_cache_init:
free -h after load:
free -h after first decode:
tok/s (if any):
Pi ls/read: pass / fail
notes:
```

### LFM 64k vs this pin

[LFM2.5 on this Jetson](Jetson-Orin-LFM2.5-2.6B.md) is ✅ **64k** on **Q8_0**; **Q6_K** (2.22 GB) is that guide’s headroom swap. That still does **not** copy onto MiniCPM. Same board; different **weights file** and different **KV**. Leftover-RAM rows below are **estimates** from the LFM Q8 @ 32k `free -h` snapshot, not MiniCPM measurements.

**Words used here** (longer decoder: [glossary](../glossary.md) · [GGUF names](../local-setup.md#understanding-gguf-quants-why-so-many-files)):

| Term | Meaning |
| --- | --- |
| **Weights / GGUF** | The model file on disk (`*.gguf`). `llama-server --model` loads **one** of these. |
| **Quant (Q8, Q6, Q4)** | How hard that file is compressed. **Q8** ≈ 8 bits/weight, closest to full quality. **Q6** ≈ 6 bits, smaller/faster, slight quality drop. **Q4** ≈ 4 bits, smallest of the official MiniCPM files. The number is *about* bits, not a quality score out of 10. |
| **`Q8_0` vs `Q6_K` vs `Q4_K_M`** | Recipe inside that bit level. `_0` is a simple 8-bit pack. `_K` is a grouped recipe. `_M` = medium/balanced mix. **Q8 weights ≠ q8 KV** (next row). |
| **KV cache** | Working memory for the *conversation so far* (attention Keys and Values). Grows with **context length**, not with which Q8/Q6/Q4 **weights** file you picked. Switching LFM Q8 → Q6 saves weight bytes only; the 64k KV stays ~544 MiB. |
| **q8/q8 KV** | `--cache-type-k q8_0 --cache-type-v q8_0` — compress that working memory to ~8-bit. Default is **f16** (about 2× the RAM). |
| **Context / 32k / 64k** | How many tokens of history+prompt the server keeps. `32k` = 32768, `64k` = 65536. `--ctx-size` on the server must match Pi `contextWindow`. |
| **GQA** | Grouped-query attention: fewer KV heads than query heads, which is why KV is smaller than a full-attention model. LFM only *grows* KV on **8** of 30 layers; MiniCPM grows it on **all 42**. |
| **Gi / MiB / GB** | Memory sizes. Roughly 1 Gi ≈ 1.07 GB. `free -h` uses Gi. |
| **Prefill** | The expensive step when a long prompt is first ingested (not the per-token “decode”). On this board, prefill is what **SIGKILL**s `llama-server` if RAM spikes and swap is 0. |
| **zram** | Compressed RAM used as swap. Enable if `free -h` shows swap 0 before stretching context. |
| **PRIMARY** | The copy-paste daily command in this file. Stretch rows are experiments, not the default. |

**Q6 vs Q8 (LFM):** `LFM2.5-2.6B-Q6_K.gguf` is **2.22 GB**; Q8_0 is **2.87 GB** (~**0.65 GB** less on disk). MiniCPM’s official GGUFs have **no Q6** — only Q8_0 (2.68 GB), Q4_K_M (1.56 GB), F16 (5.04 GB).

| | LFM Q8 | LFM **Q6** (headroom swap) | MiniCPM Q8 (this PRIMARY) |
| --- | --- | --- | --- |
| Weights file | 2.87 GB | **2.22 GB** | 2.68 GB |
| Layers that grow KV | **8** GQA | **8** GQA (same) | **all 42** GQA |
| KV @ 32k (q8/q8) | ~272 MiB | ~272 MiB (KV unchanged) | ~700 MiB |
| KV @ **64k** (q8/q8) | ~544 MiB | **~544 MiB** (tested window, cheaper weights) | **~1.4 Gi** |
| KV @ 128k (q8/q8) | 1088 MiB (measured) | 1088 MiB | ~2.8 Gi |
| Weights + 64k KV | ~3.4 GB | **~2.8 GB** | **~4.1 GB** |

This board snapshot (2026-09-08, LFM **Q8** @ 32k + Pi, swap 0): **7.3 Gi total, 1.4 Gi available**. llama.cpp pre-allocates the full KV at start. LFM Q6 is ~**0.65 GB** smaller than that Q8 file — extra headroom, same KV.

**Estimate only** if you swap to MiniCPM Q8 from that snapshot (not measured):

- MiniCPM **32k** ≈ **~1.2 Gi** free.
- MiniCPM **64k** ≈ **~0.5 Gi** free — tighter than LFM Q8 @ 64k (~1.1 Gi), and much tighter than **LFM Q6 @ 64k**. Same band as LFM’s **128k** stretch (zram; prefill SIGKILL if swap is 0).

PRIMARY stays **32k** until [D1](#d-one-axis-follow-ups-only-after-ab) (64k) load+decode is clean. Enable zram before D1 if swap is 0. If 64k OOMs on Q8, [D4](#d-one-axis-follow-ups-only-after-ab) Q4_K_M then retry 64k (MiniCPM’s analogue of LFM Q6: smaller weights, **same** KV).

### Context budget

Architecture from [config.json](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json): 42 layers, GQA 16 Q / **2 KV**, `head_dim` 128, `max_position_embeddings` **131072**.

f16 KV at native 131072 is **5.637 GB** (42 × 2 KV heads × `head_dim` 128 × 2 bytes × 2 of K/V) — matches an RTX 3060 Q8_0 @ 131k f16 KV report (~8.2 GB total VRAM, ~5.5 GB implied cache). q8_0 is ~half. Prefill scratch is extra; unified-memory spikes SIGKILL when swap is 0.

| `--ctx-size` | q8/q8 KV (est.) | f16/f16 KV | Role |
| --- | --- | --- | --- |
| 8192 | ~0.18 Gi | ~0.35 Gi | OpenBMB GGUF / llama.cpp smoke — **too small for Pi** |
| 16384 | ~0.36 Gi | ~0.70 Gi | First drop if 32k OOMs |
| **32768** | **~0.70 Gi** | ~1.41 Gi | **PRIMARY** until 64k is measured. Q8 weights + q8 KV ≈ 3.4 GB before OS/scratch |
| 65536 | ~1.40 Gi | ~2.82 Gi | **Same Pi window as LFM** — [D1](#d-one-axis-follow-ups-only-after-ab). ~0.5 Gi leftover vs LFM’s ~1.1 Gi. zram first |
| 131072 | ~2.80 Gi | **5.64 GB** (measured class) | Native; not a daily pin on 8 GB unified |

Pi compaction default `reserveTokens` is **16384**. Compaction threshold is `contextWindow - 16384`:

| Pi `contextWindow` | Compacts after |
| --- | --- |
| 16384 | 0 (immediate) |
| **32768** | **~16k** of history |
| 65536 | ~48k |
| 131072 | ~114k |

Do **not** raise `reserveTokens` when you raise the window. Keep `--n-predict` / `maxTokens` at **8192** unless a turn still ends `length`.

## Thinking on (optional)

OpenBMB’s **published** MiniCPM5-2B mode is Think ([transformers.md](https://github.com/OpenBMB/MiniCPM/blob/main/docs/deployment/transformers.md): “only supported mode” at temp 1.0 / top_p 0.95). llama-server’s switch is **`--reasoning on|off`** (default `auto`). MiniCPM5-**1B** has a separate No-think sampling row (temp **0.7**) — **do not** apply that 1B row to 2B.

PRIMARY above is think **off** for Pi skills/tools. To turn thinking **on** (math / code / traces), keep the same sampling and change only:

```bash
# Same PRIMARY, only:
#   --reasoning on
#   # drop --reasoning-budget 0  (default -1 = unrestricted)
```

Use the [thinking-on Pi JSON](#thinking-on). Thinking tokens count against `--n-predict` / `maxTokens` — if `content` is empty with `finish_reason: length`, raise **both** to **16384**. `/new` after switching.

Do **not** add `--reasoning-format none` (dumps `<think>` into `content` and confuses Pi). Do **not** add `--reasoning-preserve` for Pi tool sessions.

## Repetition

OpenBMB: if generations loop, add **`repetition_penalty=1.05`** (same temp / top_p / min_p). In llama.cpp that is `--repeat-penalty 1.05`.

Community report on official Q8_0 / Q4_K_M GGUFs with thinking **on** ([OpenBMB/MiniCPM#374](https://github.com/OpenBMB/MiniCPM/issues/374)): `--repeat-penalty 1.15` cut runaway rate further on an RTX 3060. That is **not** the card default — try **1.05** first, then 1.15 if think-on still loops. Change this flag only; smoke-test decode again.

## Vision (not this checkpoint)

**This GGUF has no vision file.** The [GGUF files tab](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/tree/main) is only `F16` / `Q8_0` / `Q4_K_M` — no `mmproj`. [Discussion #1](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/discussions/1) asks for vision later. Do **not** pass `--mmproj`. A MiniCPM-V projector is a **different model**; do not assume it will load (not tried here).

| Want | Do |
| --- | --- |
| Text / Pi tools (this guide) | Omit `--mmproj` (PRIMARY) |
| Images on **this Jetson** | Gemma 4 E2B is the VLM sibling — [Jetson Gemma](Jetson-Orin-Gemma4-E2B.md) is ✅ **text**. `mmproj` usage is documented on the Mac Mini Gemma guide, **not tested on this Jetson** |
| OpenBMB vision models | Separate **MiniCPM-V** / **MiniCPM-o** line, not this checkpoint |

## Q4_K_M alternative

OpenBMB’s **recommended** llama.cpp quant for edge / minimal VRAM. Same PRIMARY flags. Only the file changes. Prefer this if Q8 does not load, or before stretching ctx toward 64k / 128k.

```bash
hf download openbmb/MiniCPM5-2B-GGUF \
  MiniCPM5-2B-Q4_K_M.gguf \
  --local-dir ~/Documents/AIML/models/minicpm5
```

```bash
  --model ~/Documents/AIML/models/minicpm5/MiniCPM5-2B-Q4_K_M.gguf \
```

Update the Pi `name` suffix to `Q4_K_M`. Confirm load → first decode again. Quality drop vs Q8 is expected (OpenBMB: “small drop, ideal for laptops”).

**F16** (5.04 GB) is the reference file. On 8 GB unified it leaves little room for OS + KV + prefill — do not start there.

## Pi Coding Agent `models.json`

Save **one** of these files to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Open `/model` to reload (restart if the status bar is stale). `/new` after a pin or reasoning-shape change.

`maxTokens` ≤ `--n-predict` (8192). `contextWindow` = `--ctx-size` (32768). Both blocks use `id` **`minicpm5-2b`** to match `--alias`.

### Skills / tools (start here)

Use with the PRIMARY server command (thinking **off**). Pi `reasoning` **false** is the LFM-on-this-box lesson for skills. MiniCPM also has a **server** off switch — whether **both** are required is **untested**. Start with both; if tools work, leave them.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "minicpm5-2b",
          "name": "MiniCPM5-2B Q8_0 (32k, tools) - Jetson Orin Nano Super",
          "reasoning": false,
          "contextWindow": 32768,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

### Thinking on

Match the [think-on server flags](#thinking-on-optional). Pi’s documented shape for a reasoning model ([models.md](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)): `reasoning` **true**, `thinkingLevelMap.off` **null**. That JSON shape is from Pi’s docs / the LFM guide — **untested on MiniCPM**.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "minicpm5-2b",
          "name": "MiniCPM5-2B Q8_0 (32k, think) - Jetson Orin Nano Super",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null
          },
          "contextWindow": 32768,
          "maxTokens": 8192
        }
      ]
    }
  }
}
```

If 8192 still ends `length`, set **both** `--n-predict` and `maxTokens` to **16384**. For a 64k stretch, only change `contextWindow` to **65536** and the `name` suffix. `/new` after switching JSON shapes.

MiniCPM5’s template can emit **XML-style** tool tags (`<tool_call>` — [vLLM recipe](https://recipes.vllm.ai/openbmb/MiniCPM5-2B)). Upstream llama.cpp has a MiniCPM5 parser ([#24889](https://github.com/ggml-org/llama.cpp/pull/24889)); **this turboquant build has not been checked** for it. Pi’s own tool loop is separate — [C](#c-pi-tools-skills-json) is the only proof.

## Fallbacks if you OOM

Change **one** axis at a time; confirm decode after each.

1. Run **MAXN SUPER** and free other processes (`sudo nvpmodel -m 2 && sudo jetson_clocks`).
2. Keep batch at `64` / `128`.
3. Drop `--ctx-size` (and Pi `contextWindow`) to `16384`.
4. Swap to [Q4_K_M](#q4_k_m-alternative) (same ctx).
5. Optional quality: `--cache-type-v f16` (keep K `q8_0`) if 32k q8 is stable and tools feel soft (**untested**).
6. Optional capacity: `--cache-type-v turbo4` only if you are raising ctx and still OOM. Keep K at `q8_0`. **turbo\* is fork-only and untested on MiniCPM.**
7. Do **not** rely on bare `--fit on` for Pi — pin a smaller context instead.

If `free -h` shows swap **0**, enable zram before stretching ctx (LFM 128k on this board needed it). Jetson images often ship `nvzramconfig` (`systemctl status nvzramconfig`; `swapon --show`). This file is not a zram how-to.

## Field reports (X)

Not this Jetson — desktop llama.cpp runs of the **same official GGUFs**. Useful for VRAM math; do not copy their batch or 131k pin onto 8 GB unified.

- **[Dogukan @DogukanUrker](https://x.com/DogukanUrker/status/2097007347902128352)** (RTX **3060** 12 GB, 2026-09-07), same command except the GGUF path:

  ```bash
  llama-server -m MiniCPM5-2B-Q8_0.gguf -ngl 99 -c 131072 -fa on --jinja \
    -np 1 -ctk f16 -ctv f16 -b 2048 -ub 1024 --temp 1.0 --top-p 0.95
  ```

  Q4_K_M: **~163 tok/s** decode, **7.2 GB** VRAM. Q8_0: **~113 tok/s**, **8.2 GB**. Prefill ~5800 tok/s both. Most of that VRAM is **131k f16 KV**, not weights. Quote: *“drop the context size and MiniCPM5-2B becomes a very comfortable fit on an 8GB card.”*

- **[AJ @ItsmeAjayKV](https://x.com/ItsmeAjayKV/status/2097019939509158019)** (RTX **3090**): Q8_0 + **f16 KV** at full 128k ~**7 GB**; F16 weights + f16 KV ~**10 GB**. DSpark draft GGUF ([aj9o9](https://huggingface.co/aj9o9/MiniCPM5-2B-DSpark-GGUF), n-max 7, **q8 KV**): +**1.8 GB**; 32k decode **70 → 85 tok/s**. Stock llama.cpp, no custom arch.

- **[bwayne @bwmcn](https://x.com/bwmcn/status/2099235632945979564)** (RTX **4060 8 GB**): daily driver **Q8_0** among models that fit that card.

- **[TeksEdge](https://x.com/TeksEdge/status/2097180629477859690)** (Windows **CPU**, Q4_K_M, thinking **off**): ~36 tok/s but their harness scored it far below Qwen3.5-4B. Treat Q4 as the **speed/headroom** swap, not a free quality match for Q8.

- **[atomic.chat](https://x.com/atomic_chat_hq/status/2097639898266116377)** (Q4_K_M, same tasks): MiniCPM5-2B **3/3** in 23 s vs Gemma 4 E2B **2/3** in 29.5 s (Gemma broke a JSON-only rule). Sibling on this Jetson, not a pin to copy.

OpenBMB: *“one checkpoint that can think or answer fast, you pick per request”* ([quoted in](https://x.com/ShubhamMal72313/status/2097816948927254878)). No X report of a MiniCPM5-2B `mmproj`.

## Performance notes

- ⚠️ **Untested** starting pin: **Q8_0 @ 32k q8/q8**, think **off**, `--n-predict` **8192**. Prove load → first decode → Pi `ls`/`read` on this box before calling it daily.
- Official OpenBMB GGUF / llama.cpp smoke is **`-ngl 99 -c 8192 --jinja`** plus curl `temp 1.0 / top_p 0.95 / min_p 0.0` — fine for `1+1=?`, not a Pi agent window. [HF discussions](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/discussions) (#1 vision-later, #2 praise, #3 CPU-only i3 eval) do **not** change server flags.
- `n_ctx_seq (32768) < n_ctx_train (131072)` is expected on this pin **if** load honors `--ctx-size` + `--fit off`.
- Sampling is **1.0 / 0.95 / min_p 0.0**, not Gemma and not LFM.
- **DSpark** is a **separate** GGUF ([openbmb/MiniCPM5-2B-DSpark-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark-GGUF)). Some llama.cpp builds have `--spec-type draft-dspark` + `-md`; field reports add ~1.8 GB. **Leave it off.** Confirm the flag in `--help` on **this** binary before trying.
- Run in **MAXN SUPER**; monitor with `jtop`.
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## See also

- Sibling on this box (✅): [Gemma 4 E2B](Jetson-Orin-Gemma4-E2B.md) · [LFM2.5-2.6B](Jetson-Orin-LFM2.5-2.6B.md) — do not copy pins
- Pi: [agentic harnesses — MiniCPM5](../agentic-harnesses.md#minicpm5-2b--pi-coding-agent)
- Catalog / GGUF names: [local-setup.md](../local-setup.md)
- Terms: [glossary](../glossary.md)

**Last Updated:** 2026-09-14 (⚠️ untested; claims tagged measured / OpenBMB / geometry / estimate)
