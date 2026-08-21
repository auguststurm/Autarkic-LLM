# Dual RTX 6000 Pro Max-Q (192 GB) - Muse Glimmer 30B

> ⚠️ **Not yet tested** on this hardware with Muse Glimmer (researched **2026-08-14**). Confirm load → first decode → Pi tools before relying on it.
>
> Same box, tested Qwen path: [Qwen3.8 Dual RTX](Dual-RTX6000-Qwen3.8.md) (✅ 2026-08-14). **Do not** copy Qwen `--reasoning off` or temp 0.6 / top_k 20 onto Muse.

CUDA CC **12.0** · llama-cpp-turboquant · Ubuntu. Compute/memory house style matches Dual RTX Qwen (single GPU, q8/q8, pinned ctx). Agent and sampling knobs do **not**. Pi layout: [agentic harnesses — Muse](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent).

Need llama.cpp / turboquant **`b10353+`** (arch `muse-glimmer`). This box’s tip (`b10465` / `feature/turboquant-kv-cache`) already registers it and `draft-dflash`. Older builds refuse the file.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (Meta + Unsloth + this box’s CUDA pin) |
| **Weights** | `Muse-Glimmer-30B-UD-Q8_K_XL.gguf` (~32.3 GB) |
| **Catalog** | [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF) · [meta-models/Muse-Glimmer-30B](https://huggingface.co/meta-models/Muse-Glimmer-30B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · optional 262k below |
| **KV** | `q8_0` / `q8_0` · single GPU (`--main-gpu 0`) |
| **Output** | `--n-predict 32768` (thinking counts against this) |
| **Sampling** | temp **1.0** · top_p **0.95** · top_k **64** · presence **0** · repeat **1.0** |
| **Thinking** | **Cannot be switched off.** `reasoning_strength` **high** (Meta default for agents) |
| **Paths** | model `~/Documents/AIML/models` · engine `~/Documents/GitHub/llama-cpp-turboquant` |

**Pi (this model):** `--jinja` required. Keep `--parallel 1` (slots split `-c`; a 32k slot can fill on thinking and return **empty `content`**). Never stop on `<|eom|>`. Do not pass `--reasoning-format none` (dumps thinking into `content`). Details: [agentic harnesses — Muse](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent). GGUF names: [local-setup](../local-setup.md#understanding-gguf-quants-why-so-many-files).

| vs Dual RTX Qwen3.8 | Muse |
| --- | --- |
| Native ctx 262k · thinking off · top_k 20 · 16k out · optional MTP | Native **131k** · thinking **always on** · top_k **64** · **32k out** · optional **DFlash** |
| Hybrid Gated-DeltaNet (`qwen35`) | Dense + gated attention + **SWA 2048** on 3/4 layers (`muse-glimmer`) — KV is cheap |

Need the engine built first? [local-setup.md](../local-setup.md).

## Download

```bash
hf download unsloth/Muse-Glimmer-30B-GGUF \
  Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  --local-dir ~/Documents/AIML/models
```

If load fails with `unknown model architecture: 'muse-glimmer'`, `git pull` and rebuild before changing flags.

On this box: **Q8** primary. Q6 (~26.3 GB) / Q5 (~21.8 GB) / Q4 (~15.9 GB) all fit. Meta measured ~0.2% mean degradation on their Dynamic 4-bit vs full precision — this box does not need that trade. Ladder: [local-setup](../local-setup.md#q8-vs-q6-vs-q5-vs-q4-quality-vs-speed).

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

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). Confirm before debugging flags:

```bash
./llama-server --version          # 10353 or higher (this box has been 10465+)
grep -c LLM_ARCH_MUSE_GLIMMER ../../src/llama-arch.cpp   # expect >= 1
./llama-server --help | grep -i spec-type               # need draft-dflash for optional DFlash
```

Stock llama.cpp `b10353+` also loads this GGUF with `q8_0`/`q8_0`; it will reject `turbo*` cache types.

## PRIMARY command

Research baseline — Dual RTX CUDA pin + **Meta / Unsloth Muse defaults**. No DFlash, no mmproj. Run from `build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Muse-Glimmer-30B-UD-Q8_K_XL.gguf \
  --alias muse-glimmer-30b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --chat-template-kwargs '{"reasoning_strength":"high"}' \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 1024 \
  --batch-size 1024 \
  --temp 1.0 --top-p 0.95 --top-k 64 --min-p 0.0 \
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
| `--ctx-size 131072` | Official native window. One 96 GB card has lots of room. Optional [262k](#optional-262k-context) |
| `q8_0` / `q8_0` | KV is cheap (GQA 2 KV heads + SWA on 3/4 layers). turbo V almost never required |
| **No `--cache-ram 0`** | That is a **hybrid Qwen / DeltaNet** workaround ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) — not this backbone |
| `--jinja` + `--alias muse-glimmer-30b` | Embedded Muse template (required). Alias matches Pi `id` |
| `reasoning_strength: high` | Meta default for coding / agents. **`--reasoning off` is a no-op** |
| Sampling | Meta + Unsloth published defaults; **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 32768` | Thinking eats the output cap. Mid-thought ceiling → empty `content`, `finish_reason: length` |
| No `--swa-full` | Would expand local layers to full-length KV and throw away cheap cache |

Universal flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md).

### Confirm

```text
log: n_ctx_seq (131072)     # n_ctx_slot = 131072 with n_slots = 1
# model loads as muse-glimmer / Muse-Glimmer-30B
nvidia-smi
```

Thinking should land in `reasoning_content`, answer in `content`:

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"muse-glimmer-30b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

If `content` begins with `to=self<|message|>`, the chat parser is too old — rebuild (`b10353+`). Then: new Pi session with real `ls` / `read`.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. If you take [262k](#optional-262k-context), change **both** the server pin and `contextWindow`.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "muse-glimmer-30b",
          "name": "Muse Glimmer 30B Q8_K_XL (131k q8/q8) - Dual RTX 6000",
          "contextWindow": 131072,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

## This box

**Single GPU on purpose.** Q8 + 131k/262k KV + optional DFlash + mmproj fit one 96 GB card. Multi-GPU (`--split-mode layer --tensor-split 96,96`) is unused for this 30B pin.

**Reasoning strength:** `high` (primary) · `xhigh` for hardest problems (raise `--n-predict` if you still hit `length`) · `medium` / `low` for snappier tools. Per request: `"chat_template_kwargs": {"reasoning_strength":"low"}`. The OpenAI spelling `reasoning_effort` is **not** what this template reads.

## Optionals

### DFlash (speed)

Block-diffusion drafter (~1.6 GB). Different flag from Qwen MTP. Primary is without it.

```bash
hf download unsloth/Muse-Glimmer-30B-GGUF \
  dflash-kquant.gguf \
  --local-dir ~/Documents/AIML/models
```

Add to PRIMARY (build must list `draft-dflash`):

```bash
  --spec-draft-model ~/Documents/AIML/models/dflash-kquant.gguf \
  --spec-type draft-dflash \
  --spec-draft-ngl 99 \
  --spec-draft-n-max 4
```

Start at `n-max 4` (try 4–15; flag is clamped to the trained block of 16). Harmless: `[spec] failed to measure draft model memory`. Re-smoke Pi tools. Meta filename: `dflash-Muse-Glimmer-30B-Q4_K_M.gguf` in [meta-models/Muse-Glimmer-30B-GGUF](https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF). Docs: [DFlash](https://dev.meta.ai/docs/muse-glimmer/spec-decode).

### Optional 262k context

GGUF metadata often caps at 131072. Override:

```bash
# Same PRIMARY, plus:
#   --ctx-size 262144
#   --override-kv "muse-glimmer.context_length=int:262144"
# With DFlash, also: dflash.context_length=int:262144
```

Then Pi `contextWindow` **262144**. If the log still caps, set metadata with `gguf_set_metadata.py` from the turboquant clone. Quality of the extended half is less documented than native 131k.

### Vision (`mmproj`)

```bash
hf download unsloth/Muse-Glimmer-30B-GGUF \
  mmproj-Muse-Glimmer-30B-BF16.gguf \
  --local-dir ~/Documents/AIML/models
```

Add `--mmproj …/mmproj-Muse-Glimmer-30B-BF16.gguf`. Leaner: `mmproj-Muse-Glimmer-30B-Q8_0.gguf`, `mmproj-kquant.gguf`. Images bill as prompt tokens (up to 4096 visual tokens). Keep text/agent primary without mmproj.

### Other stacks

vLLM / SGLang (BF16 / FP8 / NVFP4 + native DFlash) are viable on this Blackwell card **if you leave GGUF + turboquant + Pi**. [Meta deploy](https://dev.meta.ai/docs/muse-glimmer/).

## See also

- [Meta llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/) · [Unsloth Muse](https://unsloth.ai/docs/models/muse-glimmer) · [prompting](https://dev.meta.ai/docs/muse-glimmer/prompting)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses — Muse](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent)
- Twin Qwen3.8 (tested): [Dual-RTX6000-Qwen3.8.md](Dual-RTX6000-Qwen3.8.md)

**Last Updated:** 2026-08-20 (recipe shape; still ⚠️ untested on this box)
