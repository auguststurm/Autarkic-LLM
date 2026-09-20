# Nvidia Jetson Orin Nano Super - MiniCPM5-2B

> ⚠️ **Not yet tested** on this hardware with MiniCPM5-2B. Confirm **load → first decode → Pi tools**, then report via issue/PR.

8 GB LPDDR5 (~7.3 Gi usable — LFM Q8 @ 32k snapshot 2026-09-08) · Ampere sm_**87** · llama-cpp-turboquant. Pi: [agentic harnesses — MiniCPM5-2B](../agentic-harnesses.md#minicpm5-2b--pi-coding-agent).

Dense **LlamaForCausalLM** (~2.52B, 42 layers, GQA 16 Q / 2 KV, `head_dim` 128, native **131072**). Official GGUF is **text-only**. OpenBMB documents thinking as a **toggle** (`--reasoning on|off`); that switch is untested on this build. PRIMARY **forces off** for Pi skills/tools. Do not pass `--chat-template-kwargs '{"enable_thinking":…}'`.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (OpenBMB llama.cpp + this box’s CUDA) |
| **Weights** | `MiniCPM5-2B-Q8_0.gguf` (2.68 GB) |
| **Catalog** | [openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF) · [openbmb/MiniCPM5-2B](https://huggingface.co/openbmb/MiniCPM5-2B) |
| **Context** | `--ctx-size 32768` (`--fit off`) · Pi `contextWindow` **32768** |
| **KV** | `q8_0` / `q8_0` |
| **Output** | `--n-predict 8192` · Pi `maxTokens` **8192** |
| **Sampling** | temp **1.0** · top_p **0.95** · min_p **0.0** (OpenBMB Think table / `generation_config.json`) |
| **Thinking** | **PRIMARY off:** `--reasoning off` + `--reasoning-budget 0` |
| **Vision** | **Off** — this GGUF has no projector |
| **Paths** | `~/Documents/AIML/models/minicpm5` · `~/Documents/GitHub/llama-cpp-turboquant` |

Need the engine? [local-setup.md](../local-setup.md) (JetPack / CUDA). Reuse this folder’s Gemma/LFM binary if it was built with the cmake below.

## Download

| File | Size | Role |
| --- | --- | --- |
| `MiniCPM5-2B-Q8_0.gguf` | 2.68 GB | **PRIMARY** — OpenBMB: “very small quality drop vs F16” |
| `MiniCPM5-2B-Q4_K_M.gguf` | 1.56 GB | OpenBMB edge rec; headroom swap if Q8 OOMs or you want more ctx |
| `MiniCPM5-2B-F16.gguf` | 5.04 GB | Reference; **not** a daily pin on 8 GB unified |

```bash
hf download openbmb/MiniCPM5-2B-GGUF \
  MiniCPM5-2B-Q8_0.gguf \
  --local-dir ~/Documents/AIML/models/minicpm5
```

There is **no** `mmproj-*.gguf` in this repo — do not invent one.

## Build

Same cmake as [Gemma](Jetson-Orin-Gemma4-E2B.md#build) / [LFM](Jetson-Orin-LFM2.5-2.6B.md#build) on this box.

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

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). MiniCPM is stock **llama** (no extra arch tag).

## PRIMARY command

Run from `build/bin`. **MAXN SUPER** first: `sudo nvpmodel -m 2 && sudo jetson_clocks`.

Think **off**, vision **off**, Q8_0 @ 32k. OpenBMB’s GGUF smoke is `-c 8192` — below Pi’s `reserveTokens` **16384**. **`--threads 0` is not auto** (`--help` default is **-1**) — omit `--threads`.

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

Omit `--load-mode none` (mmap on 8 GB). Omit `--mmproj`. Omit `--chat-template-kwargs`. Omit `--log-verbosity` (default **3**). Omit `--no-context-shift` (already default). Omit `--threads` (default **-1**). Do **not** set `--top-k` (OpenBMB doesn’t; default 40 stands). Repeat penalty stays **1.0** until loops.

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 32768` | Unset would request 131k. OpenBMB long-context step (Ollama / vLLM `32768`). 8192 / 16384 make Pi compaction threshold ≤ 0. q8 KV ≈ **0.70 Gi** at 32k (geometry). 64k is a stretch after decode is clean |
| `q8_0` / `q8_0` | Default KV is f16 (~1.41 Gi at 32k). Prefill SIGKILL on this board is measured on Gemma/LFM. If 32k q8 is stable and tools feel soft, try **V f16** (keep K q8) |
| `--n-predict 8192` | Pi-agent choice, not a MiniCPM measurement. Raise **both** sides to 16384 if a turn ends `length` |
| Batch 64 / 128 | Scratch scales with ubatch. This board SIGKILLed on large-batch Gemma/LFM prefill. 256 is the first raise if 64/128 is stable |
| `--n-gpu-layers 99` | 42 layers; OpenBMB: 99 = all. Auto on unified memory can leave layers on CPU |
| `--fit off` | llama.cpp default is **on** and will shrink an oversized ctx |
| `--parallel 1` | Auto can open extra slots and split `-c` |
| `--flash-attn on` | Do not leave `auto` |
| `--reasoning off` | `--reasoning auto` on this GGUF+server is unverified. Confirm curl `reasoning` chars = 0 |
| Sampling | Official MiniCPM5-2B triplet. **min_p 0.05** (llama.cpp default) is what OpenBMB says locks repetition loops |

### Confirm

```text
log: n_ctx_seq (32768)
# flash attn on; mmap (not load_mode = none)
# kv_cache_init ≈ 700 MiB (geometry, not measured here)
```

Then: (1) load, (2) short decode with reasoning chars **0**, (3) new Pi session with real `ls` / `read`.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"minicpm5-2b","messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}],"max_tokens":64}' \
| python3 -c "import json,sys; r=json.load(sys.stdin); m=r['choices'][0]['message']; \
print('content  :', (m.get('content') or '')[:200]); \
print('reasoning:', len(m.get('reasoning_content') or ''), 'chars'); \
print('finish   :', r['choices'][0].get('finish_reason'))"
```

17×23 is **391**. Temp 1.0 may add prose — fail only if the math is wrong or `content` is empty. Non-zero `reasoning` means `--reasoning off` did not take on **this** build — fix that before Pi.

Restart **both** llama-server and Pi after pin or JSON changes. Status bar must match `32768` / `8192`.

## Pi Coding Agent `models.json`

Save **one** of these to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). `id` matches `--alias`. `/model` to reload; `/new` after a pin or reasoning-shape change.

### Skills / tools (start here)

PRIMARY is think **off**. Pi `reasoning` **false**. Whether both are required is untested — start with both.

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

Match the think-on server flags. `reasoning` **true**, `thinkingLevelMap.off` **null**. Untested on MiniCPM.

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

MiniCPM5’s template can emit XML-style tool tags (`<tool_call>`). Upstream llama.cpp has a MiniCPM5 parser ([#24889](https://github.com/ggml-org/llama.cpp/pull/24889)); **this turboquant build has not been checked** for it. Pi’s own tool loop is separate — real `ls` / `read` is the only proof.

## This box

**64k stretch** (after 32k decode is clean): `--ctx-size 65536` and Pi `contextWindow` 65536. MiniCPM q8 KV ≈ **1.40 Gi** at 64k (geometry; all 42 layers grow KV). Enable zram if `free -h` shows swap 0. If 64k OOMs on Q8, swap to Q4_K_M then retry 64k.

**Q4_K_M:** same PRIMARY, swap `--model` to `MiniCPM5-2B-Q4_K_M.gguf` (1.56 GB). OpenBMB: “small drop, ideal for laptops.”

**OOM:** (1) MAXN SUPER, (2) keep batch 64/128, (3) `--ctx-size 16384` + Pi 16384, (4) Q4_K_M. Do not rely on bare `--fit on`. Optional: `--cache-type-v f16` (K stays q8) if 32k q8 is stable and tools feel soft. `--cache-type-v turbo4` only if raising ctx still OOMs — untested on MiniCPM.

**Thinking on:** same PRIMARY, `--reasoning on`, drop `--reasoning-budget 0`, use the think JSON. OpenBMB’s published 2B mode is Think. MiniCPM5-**1B** No-think sampling (temp **0.7**) does **not** apply to 2B. Do not add `--reasoning-format none` or `--reasoning-preserve` for Pi.

**Repetition:** OpenBMB `repetition_penalty=1.05` → `--repeat-penalty 1.05`. If think-on still loops, try **1.15** ([OpenBMB/MiniCPM#374](https://github.com/OpenBMB/MiniCPM/issues/374)).

**Vision:** this GGUF has no `mmproj` ([files](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/tree/main); [discussion #1](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF/discussions/1) asks for vision later). A MiniCPM-V projector is a different model.

**DSpark:** separate GGUF ([openbmb/MiniCPM5-2B-DSpark-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-DSpark-GGUF)). Leave it off until PRIMARY is daily-stable. Confirm `./llama-server --help | grep draft-dspark` on **this** binary first.

### Context budget

Architecture from [config.json](https://huggingface.co/openbmb/MiniCPM5-2B/blob/main/config.json): 42 layers, GQA 16 Q / **2 KV**, `head_dim` 128, `max_position_embeddings` **131072**. f16 KV at 131072 is **5.637 GB**. q8_0 is ~half. Prefill scratch is extra.

| `--ctx-size` | q8/q8 KV (est.) | Role |
| --- | --- | --- |
| 16384 | ~0.36 Gi | First drop if 32k OOMs |
| **32768** | **~0.70 Gi** | **PRIMARY** until 64k is measured |
| 65536 | ~1.40 Gi | Stretch after 32k decode |
| 131072 | ~2.80 Gi | Native; not a daily pin on 8 GB unified |

Pi compaction default `reserveTokens` is **16384**. At `contextWindow` 32768, compaction starts after ~16k of history. Do not raise `reserveTokens` when you raise the window.

## See also

- This box: [Gemma 4 E2B](Jetson-Orin-Gemma4-E2B.md) · [LFM2.5-2.6B](Jetson-Orin-LFM2.5-2.6B.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses — MiniCPM5](../agentic-harnesses.md#minicpm5-2b--pi-coding-agent)

**Last Updated:** 2026-09-20 (recipe density; ⚠️ untested; KV figures are geometry unless noted)
