# M2 Mac Mini (16 GB) - Gemma 4 E2B

Recommended lightweight multimodal setup for Mac Mini M2 with 16 GB unified memory.

> ⚠️ **Not yet tested on this hardware.** This is a best-effort starting config, but at ~3 GB the model leaves comfortable headroom on 16 GB, so it should be low-risk. The base M2's lower memory bandwidth (~100 GB/s vs the M4 Mini's ~120 GB/s) makes this small dense model the most comfortable daily driver on this machine. If you run it, please report your results via an issue.

## Recommended Model

- **Model**: `gemma-4-E2B-it-Q4_K_S.gguf`
- **Path**: `~/Documents/AIML/Models/unsloth/gemma-4-E2B/gemma-4-E2B-it-Q4_K_S.gguf`

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
  --model ~/Documents/AIML/Models/unsloth/gemma-4-E2B/gemma-4-E2B-it-Q4_K_S.gguf \
  --host 0.0.0.0 --port 8080 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --no-mmap \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 256 \
  --batch-size 256 \
  --reasoning on \
  --reasoning-budget 0 \
  --repeat-penalty 1.10 \
  --presence-penalty 0.0 \
  --frequency-penalty 0.0 \
  --min-p 0.0 \
  --repeat-last-n 512 \
  --threads 0 --temp 0.75 --top-p 0.92 \
  --n-predict 4096 \
  --cache-ram 1024 \
  --slot-save-path ./kv-cache \
  --checkpoint-min-step 8192 \
  --ctx-checkpoints 4 \
  --log-verbosity 2
```

## Enable Multimodal (optional)

Gemma 4 E2B is natively multimodal (text + image + audio), but llama-server only loads the vision/audio projector if you pass one. Download an mmproj file and add the flag:

```bash
hf download unsloth/gemma-4-E2B-it-GGUF \
  mmproj-F16.gguf \
  --local-dir ~/Documents/AIML/Models/unsloth/gemma-4-E2B
```

Then add to the `llama-server` command above:

```bash
  --mmproj ~/Documents/AIML/Models/unsloth/gemma-4-E2B/mmproj-F16.gguf \
```

Without `--mmproj` the server runs text-only.

## Performance Notes

- Extremely lightweight (~3 GB loaded): easily fits with macOS overhead on 16 GB.
- Multimodal capable (text + image + audio) once you pass `--mmproj` (see above); text-only otherwise.
- On the base M2's ~100 GB/s bandwidth this small dense model still stays responsive for agentic tasks; it is the recommended choice on this machine over the tight [Qwen3.6-35B-A3B experiment](M2-Mac-Mini-Qwen3.6.md).
- **No `--fit` here (intentional):** at ~3 GB the model leaves plenty of room, so `--ctx-size 32768` is pinned and guaranteed. `--fit` is omitted because the current fork aborts it when `--n-gpu-layers`/`--ctx-size` are set. If you ever want a bigger context and hit an OOM, lower `--ctx-size`.

## Measured Results

> 📝 **Placeholder — pending a real run on a 16 GB M2 Mac Mini.** Replace each *TBD* once measured.

| Metric | Value |
| --- | --- |
| Largest `--ctx-size` that loaded | *TBD* |
| Peak memory (startup log + Activity Monitor) | *TBD* |
| Prefill / prompt-eval (tok/s) | *TBD* |
| Decode / generation (tok/s) | *TBD* |
| Multimodal tested (`--mmproj`)? | *TBD* |
| llama-cpp-turboquant commit built | *TBD* |

## Pi Coding Agent models.json Snippet

```json
{
  "id": "gemma-4-e2b",
  "name": "Gemma 4 E2B Q4_K_S (32k) - M2 Mini",
  "contextWindow": 32768,
  "maxTokens": 4096
}
```

**Last Updated:** July 2026
