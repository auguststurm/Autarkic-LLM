# M2 Mac Mini (16 GB) - Qwen3.6-35B-A3B

Setup for Mac Mini M2 with 16 GB unified memory.

> ⚠️ **Untested on hardware. 16 GB is a tight fit for this model.** Qwen3.6-35B-A3B is a 35B-parameter MoE; even though only ~3B params are active per token, the *full* weights must be resident in memory. The 4-bit quants are 21–22 GB and **cannot load on a 16 GB machine**. Only the heavily-compressed IQ2/IQ1 quants fit, and only with a reduced context and a raised GPU memory limit (see below). If you just want a reliable model on 16 GB, use the **[Gemma 4 E2B config](M2-Mac-Mini-Gemma-4-E2B.md)** in this folder (~3 GB, lots of headroom). Treat this Qwen guide as an experiment for people who want to push the limit and report back. On the base M2's ~100 GB/s memory bandwidth (vs ~120 GB/s on the M4 Mini), expect slower decode than the [M4 Mac Mini config](../M4-Mac-Mini-16GB/M4-Mac-Mini-Qwen3.6.md).

## Memory Budget (16 GB unified)

Usable budget after macOS overhead (~3–4 GB) and the default GPU wired-memory cap (~⅔ of RAM ≈ 10.7 GB) is small. The model weights + KV cache + compute buffers must all fit. Actual GGUF sizes for `unsloth/Qwen3.6-35B-A3B-GGUF`:

```text
Quant        Size      Fits 16 GB?
-----------  --------  -------------------------------------------
UD-IQ1_M     10.0 GB   yes, with headroom (lowest quality)
UD-IQ2_XXS   10.8 GB   marginal
UD-IQ2_M     11.5 GB   marginal  <- recommended starting point
UD-Q2_K_XL   12.3 GB   tight; needs raised wired limit
UD-Q3_K_S    15.4 GB   no
UD-Q4_K_XL   22.4 GB   no
```

Raise the Metal GPU memory limit before launching (hands ~13 GB to the GPU; revert with a reboot):

```bash
sudo sysctl iogpu.wired_limit_mb=13000
```

> ⚠️ **This leaves only ~3 GB for macOS** (13 GB of 16 GB handed to the GPU) — below Apple's recommended headroom and at the bottom of this machine's own ~3–4 GB OS overhead, so expect memory pressure. Watch **Activity Monitor → Memory → Memory Pressure** and back off the moment it turns yellow/red or the system starts swapping: lower the limit or drop to `UD-IQ1_M` (with `--fit`, the context auto-shrinks to match). The `iogpu.wired_limit_mb` change is **not** persistent — a reboot restores the default (~⅔ of RAM). The `13000` value is sized for the ~11.5 GB `IQ2_M` plus its KV/compute buffers; if you run the smaller `UD-IQ1_M` (~10 GB), a lower limit (e.g. `12000`) leaves more OS headroom.

## Recommended Model

- **Model**: `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (MoE; IQ2 chosen so it fits 16 GB: expect a noticeable quality drop vs. Q4+)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- **Fallback if it OOMs**: drop to `UD-IQ1_M` (10 GB); `--fit` will size the context to whatever memory remains.

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

# Apple Silicon: first apply the Metal rnorm patch from ../local-setup.md (step 2), or the shader fails to compile when llama-server starts
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)

cd bin
mkdir -p ./kv-cache
```

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --fit on \
  --fit-target 256 \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 128 \
  --batch-size 128 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 1024 \
  --threads 0 --temp 0.65 --top-p 0.90 \
  --n-predict 4096 \
  --cache-ram 1024 \
  --slot-save-path ./kv-cache \
  --checkpoint-min-step 8192 \
  --ctx-checkpoints 4 \
  --log-verbosity 2
```

> **Checkpoint flags on Qwen3.6:** `--slot-save-path`, `--checkpoint-min-step`, and `--ctx-checkpoints` are included here, but may be **no-ops on Qwen3.6** — its hybrid Gated-DeltaNet attention hits a known llama.cpp bug where context checkpoints aren't restored, forcing full prompt reprocessing each turn. On this memory-tight config, dropping them also frees the `--cache-ram` reservation. Details: [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing).

## Performance Notes

- 16 GB is the binding constraint here, not compute: the MoE's low *active* parameter count helps speed, but the full weights still have to be resident, so quant choice is driven entirely by the memory budget above.
- The base M2's memory bandwidth (~100 GB/s) is lower than the M4 Mini's (~120 GB/s), and decode on a memory-bound MoE scales with bandwidth — expect generation a bit slower than the M4 Mini numbers. If decode feels too slow at IQ2, the Gemma config is the better daily driver.
- With `--ctx-size` and `--n-gpu-layers` omitted, `--fit on --fit-target 256` auto-sizes the context to whatever fits after the ~11.5 GB weights — expect a small context on 16 GB; check the startup log for the number it chose. Raising the wired limit (below) gives fit more room.
- Close other apps and run `sudo sysctl iogpu.wired_limit_mb=13000` before launching, or the GPU allocation cap will reject the model.
- `--fit on --fit-target 256` manages the load: we **omit `--ctx-size`/`--n-gpu-layers`** so fit can auto-tune context and offload (pinning either makes it abort on the current fork). It **cannot shrink the weights**, though — a 22 GB Q4 still won't load on 16 GB no matter the context, which is why this guide uses an IQ2 quant.
- Expect IQ2-level quality (noticeably below the Q5/Q6 configs on larger machines). For everyday use on 16 GB, the [Gemma 4 E2B config](M2-Mac-Mini-Gemma-4-E2B.md) is the better choice; this guide is for testers who want to see how far a 35B MoE can be pushed on 16 GB.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6-35B-A3B IQ2_M (fit-sized) - M2 Mini",
  "contextWindow": 4096,
  "maxTokens": 2048
}
```

> **`contextWindow` is provisional (untested).** `--fit` chooses the real context at launch; the `4096` above is a conservative placeholder. After your first run, set `contextWindow` to the effective `n_ctx` from the startup log (and `maxTokens` to no more than half of it).

## Measured Results

> 📝 **Placeholder — pending a real run on a 16 GB M2 Mac Mini.** Replace each *TBD* once measured.

| Metric | Value |
| --- | --- |
| Quant used | *TBD* |
| Largest `--ctx-size` that loaded | *TBD* |
| `iogpu.wired_limit_mb` needed | *TBD* |
| Peak memory (startup log + Activity Monitor) | *TBD* |
| Prefill / prompt-eval (tok/s) | *TBD* |
| Decode / generation (tok/s) | *TBD* |
| Subjective quality at IQ2 | *TBD* |
| llama-cpp-turboquant commit built | *TBD* |

## Report Your Results

This config is untested on real hardware. If you run it on a 16 GB M2 Mac Mini, please open an issue with:

- **Quant used** (e.g. `UD-IQ2_M`) and whether it loaded or OOM'd.
- **Context that actually fit**: the largest `--ctx-size` that loaded without OOM.
- **Peak memory**: from the `llama-server` startup log (model buffer + KV cache + compute buffer sizes) and Activity Monitor's "Memory Used" while generating.
- **`iogpu.wired_limit_mb`** value you needed (or whether the default worked).
- **Speed**: prompt eval (prefill) and generation (decode) tokens/sec, reported by `llama-server` per request.
- **Subjective quality** at IQ2 for your tasks, and whether dropping to `UD-IQ1_M` or moving up to `UD-Q2_K_XL` worked better.

That feedback lets us replace the estimates above with measured numbers.

**Last Updated:** June 2026
