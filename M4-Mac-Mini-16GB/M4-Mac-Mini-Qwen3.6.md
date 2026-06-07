# M4 Mac Mini (16 GB) - Qwen3.6-35B-A3B

Setup for Mac Mini M4 with 16 GB unified memory.

> ⚠️ **Untested on hardware. 16 GB is a tight fit for this model.** Qwen3.6-35B-A3B is a 35B-parameter MoE; even though only ~3B params are active per token, the *full* weights must be resident in memory. The 4-bit quants are 21–22 GB and **cannot load on a 16 GB machine**. Only the heavily-compressed IQ2/IQ1 quants fit, and only with a reduced context and a raised GPU memory limit (see below). If you just want a reliable model on 16 GB, use the **[Gemma 4 E2B config](M4-Mac-Mini-Gemma-4-E2B.md)** in this folder (~3 GB, lots of headroom). Treat this Qwen guide as an experiment for people who want to push the limit and report back.

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

> ⚠️ **This leaves only ~3 GB for macOS** (13 GB of 16 GB handed to the GPU) — below Apple's recommended headroom and at the bottom of this machine's own ~3–4 GB OS overhead, so expect memory pressure. Watch **Activity Monitor → Memory → Memory Pressure** and back off the moment it turns yellow/red or the system starts swapping: lower the limit, drop to `UD-IQ1_M`, or reduce `--ctx-size`. The `iogpu.wired_limit_mb` change is **not** persistent — a reboot restores the default (~⅔ of RAM). The `13000` value is sized for the ~11.5 GB `IQ2_M` plus its KV/compute buffers; if you run the smaller `UD-IQ1_M` (~10 GB), a lower limit (e.g. `12000`) leaves more OS headroom.

## Recommended Model

- **Model**: `Qwen3.6-35B-A3B-UD-IQ2_M.gguf` (MoE; IQ2 chosen so it fits 16 GB: expect a noticeable quality drop vs. Q4+)
- **Path**: `~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-IQ2_M.gguf`
- **Fallback if it OOMs**: drop to `UD-IQ1_M` (10 GB) and/or lower `--ctx-size`.

## Build Instructions

```bash
cd ~/Documents/GitHub/llama-cpp-turboquant

rm -rf build
mkdir build && cd build

cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)

cd ../bin
mkdir -p ./kv-cache
```

## Optimized llama-server Command

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/Models/unsloth/Qwen/LLM/Qwen3.6-35B-A3B-UD-IQ2_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 16384 \
  --n-gpu-layers 99 \
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
  --checkpoint-every-n-tokens 8192 \
  --ctx-checkpoints 4 \
  --log-verbosity 2
```

> **Checkpoint flags on Qwen3.6:** `--slot-save-path`, `--checkpoint-every-n-tokens`, and `--ctx-checkpoints` are included here, but may be **no-ops on Qwen3.6** — its hybrid Gated-DeltaNet attention hits a known llama.cpp bug where context checkpoints aren't restored, forcing full prompt reprocessing each turn. On this memory-tight config, dropping them also frees the `--cache-ram` reservation. Details: [checkpointing caveat](../llama-cpp-turboquant.md#prompt-cache--checkpointing).

## Performance Notes

- 16 GB is the binding constraint here, not compute: the MoE's low *active* parameter count helps speed, but the full weights still have to be resident, so quant choice is driven entirely by the memory budget above.
- `--ctx-size 16384` keeps the KV cache (~0.8 GB at q8_0) small enough to leave room for the weights. Lower it further if you hit OOM; raise it only if you have headroom to spare.
- Close other apps and run `sudo sysctl iogpu.wired_limit_mb=13000` before launching, or the GPU allocation cap will reject the model.
- `--fit on --fit-target 256` (in the command above) lets llama-server auto-tune context/offload to fit available memory, but note it **cannot shrink the weights**: a 22 GB Q4 still won't load on 16 GB no matter the context, which is why this guide uses an IQ2 quant.
- Expect IQ2-level quality (noticeably below the Q5/Q6 configs on larger machines). For everyday use on 16 GB, the [Gemma 4 E2B config](M4-Mac-Mini-Gemma-4-E2B.md) is the better choice; this guide is for testers who want to see how far a 35B MoE can be pushed on 16 GB.

## Pi Coding Agent models.json Snippet

```json
{
  "id": "qwen3.6-35b-a3b",
  "name": "Qwen3.6-35B-A3B IQ2_M (16k) - M4 Mini",
  "contextWindow": 16384,
  "maxTokens": 8192
}
```

## Report Your Results

This config is untested on real hardware. If you run it on a 16 GB M4 Mac Mini, please open an issue with:

- **Quant used** (e.g. `UD-IQ2_M`) and whether it loaded or OOM'd.
- **Context that actually fit**: the largest `--ctx-size` that loaded without OOM.
- **Peak memory**: from the `llama-server` startup log (model buffer + KV cache + compute buffer sizes) and Activity Monitor's "Memory Used" while generating.
- **`iogpu.wired_limit_mb`** value you needed (or whether the default worked).
- **Speed**: prompt eval (prefill) and generation (decode) tokens/sec, reported by `llama-server` per request.
- **Subjective quality** at IQ2 for your tasks, and whether dropping to `UD-IQ1_M` or moving up to `UD-Q2_K_XL` worked better.

That feedback lets us replace the estimates above with measured numbers.

**Last Updated:** June 2026
