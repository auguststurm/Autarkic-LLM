# Dual RTX 6000 Pro Max-Q (192 GB) - Qwen3.8-27B

> ✅ **Tested** on this hardware (**2026-08-14**, Qwen3.8 release day) with **Pi Coding Agent**. Same knobs as the [Qwen3.6 Dual RTX](Dual-RTX6000-Qwen3.6.md) primary; only the weights file changed.

CUDA CC **12.0** · llama-cpp-turboquant · Ubuntu. This file is also the **Qwen3.8 appendix** other 3.8 ports link to (MTP, sampling, thinking, vision).

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested 2026-08-14 (Pi tools, full train window) |
| **Weights** | `Qwen3.8-27B-UD-Q8_K_XL.gguf` (~31.5 GB) |
| **Catalog** | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| **Context** | `--ctx-size 262144` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` · single GPU (`--main-gpu 0`) |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` (Pi tools) |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant` |

**Pi:** pin `contextWindow` = 262144 and `maxTokens` = 16384. Two limits, no DRY, no `--reasoning-preserve`. Cross-hardware: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). GGUF names (`UD-`, `_K_XL`): [local-setup](../local-setup.md#understanding-gguf-quants-why-so-many-files).

Need the engine built first? [local-setup.md](../local-setup.md). Hardware not in the README table? [ai-assisted-setup.md](../ai-assisted-setup.md).

## Download

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  Qwen3.8-27B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

Fresh turboquant build required (arch tag `qwen35`). If load fails with unknown architecture, `git pull` and rebuild before changing flags.

On this 192 GB box: **Q8** is the tested primary. Q6 (~25.9 GB) / Q5 (~20.2 GB) / Q4 (~17.9 GB) all fit; drop only for multi-instance or speed A/Bs. Ladder: [local-setup](../local-setup.md#q8-vs-q6-vs-q5-vs-q4-quality-vs-speed).

## Build

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="120"
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (not `ggml-org/llama.cpp`). Omit `-DCMAKE_CUDA_ARCHITECTURES="120"` to autodetect; set it when cross-compiling (Blackwell = CC 12.0). Prereqs and other backends: [local-setup](../local-setup.md).

Optional (only if you try turbo V): `./llama-server --help | grep -A2 cache-type-v` must list `turbo2`, `turbo3`, `turbo4`.

## PRIMARY command

Tested baseline with Pi. Run from `build/bin`. Use this first (no MTP).

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Qwen3.8-27B-UD-Q8_K_XL.gguf \
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

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 262144` | Full native window; one 96 GB card holds Q8 + KV — confirmed with Pi |
| `q8_0` / `q8_0` | High-precision KV while VRAM allows; turbo V is a later capacity lever |
| `--main-gpu 0` | Single-GPU primary on a dual-card box (second 96 GB stays free) |
| `--load-mode none` | Buffered read (old `--no-mmap`). Confirmed: log `load_mode = none`, no mmap deprecation warning |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn restore issues ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| `--reasoning off` | Pi needs `message.content` / tools — not thinking traces |
| Sampling | Dual RTX / 4090 agent profile; **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 16384` | Report-length agent output (match Pi `maxTokens`) |

Universal flags (`--fit off`, `--jinja`, `--flash-attn on`, loopback, no checkpoints): [llama-cpp-turboquant.md](../llama-cpp-turboquant.md).

### Confirm

`--ctx-size 262144` is a request. Trust the load log:

```text
log: n_ctx_seq (262144)
log: load_mode = none
# model loads as qwen35 / Qwen3.8-27B
nvidia-smi   # MiB after load and after a short decode
```

Then: (1) load, (2) first decode, (3) new Pi session with real `ls` / `read`.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi so the status bar matches.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.8-27b",
          "name": "Qwen3.8-27B Q8_K_XL (262k q8/q8) - Dual RTX 6000",
          "contextWindow": 262144,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

## This box

**Single GPU on purpose.** ~31.5 GB weights + 262k q8/q8 KV fit one 96 GB card. The other card stays idle unless you take the multi-GPU alternate.

**Optional turbo V** (capacity only — not the quality default):

```bash
# Same PRIMARY, only:
#   --cache-type-k q8_0 --cache-type-v turbo4
# Keep K at q8_0. New Pi session; real ls/read before trusting it.
```

**Alternate: multi-GPU layer split** — ⚠️ not part of the 2026-08-14 primary test. Same 27B dense weights (no 3.8 MoE twin). PCIe sync overhead; bench against single-GPU first.

```bash
# Same PRIMARY, plus:
#   --split-mode layer --tensor-split 96,96
```

## Qwen3.8 optionals

### MTP (speed)

Unsloth GGUFs include MTP heads (~1.4–2.2× decode on this CUDA box, ~1–2 GB extra). Primary is **without** MTP (field-tested). After a build that lists `draft-mtp` in `--help`, add to the same command:

```bash
  --spec-type draft-mtp \
  --spec-draft-n-max 2
```

Start at `n-max 2` (try 1–6). **Correctness before tok/s:** re-smoke Pi `ls`/`read` on a new session. Do not stack turbo V + MTP and trust a speed screenshot. Docs: [MTP](https://unsloth.ai/docs/models/mtp).

### Sampling (leave Pi for these)

Keep the pin’s Pi row for tool/coding agents. Unsloth’s [table](https://unsloth.ai/docs/models/qwen3.8#recommended-settings) when you are **not** in path-heavy tool loops:

| Mode | temp | top_p | presence | Notes |
| --- | --- | --- | --- | --- |
| Thinking | 1.0 | 0.95 | 0.0 | With `--reasoning on` |
| Instruct (non-thinking) | 0.7 | 0.80 | **1.5** | Chat only — presence 1.5 warps reused paths in Pi |
| **This repo’s Pi tools** | **0.6** | **0.95** | **0.0** | Field-validated on this box |

### Thinking on (not the Pi default)

Primary stays `--reasoning off`. If you turn thinking **on**:

- Unset `reasoning_effort` renders as **XHIGH** on the Unsloth/open chat template these GGUFs ship — not a middle setting. Set it on purpose. Hosted APIs can map labels differently.
- On that template, **`low` and `xhigh` inject dedicated instructions**; `medium` is accepted but often has **no dedicated MEDIUM branch**. Do not treat medium as the balanced default here.
- **Thinking tokens count against `--n-predict`.** XHIGH can fill the page and truncate the answer. Start at **`low`**; use **xhigh** only with a large output cap. Want a short answer? Turn thinking **off** — do not starve it with a tiny `maxTokens`.
- **Leave `--reasoning-preserve` off** for Pi and long sessions. Preserve re-injects prior traces (thousands of tokens per turn). Enable only for short thinking-on chat.

```bash
# Deltas only — not the Pi primary:
#   --reasoning on
#   --temp 1.0 --top-p 0.95 --top-k 20 --presence-penalty 0.0
#   --chat-template-kwargs '{"reasoning_effort":"low"}'
# Raise --n-predict / Pi maxTokens together if traces truncate.
```

### Vision (`mmproj`)

Not required for Pi text/agent work.

```bash
hf download unsloth/Qwen3.8-27B-GGUF \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/models
```

Add `--mmproj ~/Documents/AIML/models/mmproj-F16.gguf` only if this `llama-server` build exposes multimodal flags for `qwen35`.

### NVFP4 (different stack)

[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) is for **vLLM / SGLang**, not this GGUF + turboquant + Pi primary. Viable on this Blackwell card if you leave the repo stack. [Unsloth NVFP4](https://unsloth.ai/docs/basics/nvfp4).

## See also

- Twin 3.6 (tested 2026-08-08): [Dual-RTX6000-Qwen3.6.md](Dual-RTX6000-Qwen3.6.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)
- Unsloth: [Qwen3.8](https://unsloth.ai/docs/models/qwen3.8) · [MTP](https://unsloth.ai/docs/models/mtp)

**Last Updated:** 2026-08-20 (recipe + canonical 3.8 optionals; primary still the 2026-08-14 tested command)
