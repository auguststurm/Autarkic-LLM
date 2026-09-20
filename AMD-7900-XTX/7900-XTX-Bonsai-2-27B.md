# AMD 7900 XTX (24 GB) - Ternary Bonsai 2 27B

> ⚠️ **Not yet tested** on this hardware with Bonsai 2 (researched **2026-09-18**). Confirm load → first decode → Pi tools before relying on it. **Not** llama-cpp-turboquant. **Not CUDA.**

24 GB · RDNA3 **gfx1100** · [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) (`prism` branch, **prism-b10658+**). Paths: `~/Documents/AIML/models` · `~/Documents/GitHub/llama.cpp-prism` — same Linux layout as this folder’s [Qwen3.6 27B](7900-XTX-Qwen3.6-27b.md).

This folder’s Qwen recipes are **Vulkan**. Bonsai 2 is a **different engine**, and **Vulkan is the wrong first backend today**: `PQ2_0` has **no Vulkan kernels** (CPU fallback), and `PTQ1_0` Vulkan decode is a slow generic shader until [PR #188](https://github.com/PrismML-Eng/llama.cpp/pull/188) lands. **PRIMARY is HIP / ROCm** (`-DGGML_HIP=ON`), where PrismML documents `PQ2_0` kernels. Do not pass `-DGGML_CUDA=ON` on this GPU.

Same Qwen3.8-27B hybrid backbone as Dual RTX / 3090 Bonsai 2, packed as true ternary weights (~6–7 GB). This folder’s tested Qwen 27B spends ~18–19 GB on IQ4_NL and pins **128k**. Bonsai 2 leaves most of the 24 GB for KV. Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

**Stock llama.cpp and llama-cpp-turboquant will not run these files.** They refuse `PQ2_0` / `PTQ1_0` as unknown types. Do not load a Bonsai 2 `Q2_0` on a stock build either — that file can load silently and emit garbage (Hadamard runtime missing). Use the PrismML fork.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested (PrismML backends + this folder’s 24 GB RDNA3 pin) |
| **Backend** | **HIP / ROCm** (PRIMARY). Vulkan only as [alternate](#alternate-vulkan--expect-slow) |
| **PRIMARY weights** | `Ternary-Bonsai-2-27B-PQ2_0.gguf` (~7.2 GB, 2.13 bpw) |
| **A/B weights** | `Ternary-Bonsai-2-27B-PTQ1_0.gguf` (~5.9 GB, 1.75 bpw) |
| **Catalog** | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| **Context** | `--ctx-size 131072` (`--fit off`) · half-window **65536** · try [262k](#this-box) if VRAM allows |
| **KV** | `q8_0` / `q8_0` · this fork has **no** `turbo*` types |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` (Pi tools). Model thinks by default if you leave it on |
| **Paths** | `~/Documents/AIML/models` · `~/Documents/GitHub/llama.cpp-prism` |

Need ROCm / Vulkan prereqs? [local-setup.md](../local-setup.md). This page’s cmake is the PrismML fork, not the turboquant clone. GGUF names (`PQ2_0` vs `PTQ1_0`): [local-setup](../local-setup.md#model-catalog-hugging-face).

## Download

Both packs. Same ternary g128 weights; different on-disk packing. Skip `Ternary-Bonsai-2-27B-F16.gguf` (~54 GB).

```bash
hf download prism-ml/Ternary-Bonsai-2-27B-gguf \
  Ternary-Bonsai-2-27B-PQ2_0.gguf \
  Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  --local-dir ~/Documents/AIML/models
```

Confirm both files exist under `~/Documents/AIML/models` before building.

**Why two files, and which first.** PrismML has **no 7900 XTX row**. `PQ2_0` is the demo default and the packing with **ROCm kernels**. `PTQ1_0` is smaller. Start with **PQ2_0** on HIP, then [bench both](#try-both-packs). On **Vulkan**, use **PTQ1_0 only** until PQ2_0 shaders exist — `PQ2_0` on current Vulkan falls back to CPU.

| Pack | Size | HIP / ROCm | Vulkan (current `prism`) |
| --- | --- | --- | --- |
| **PQ2_0** | ~7.2 GB | **PRIMARY** — documented kernels | CPU fallback — do not use |
| **PTQ1_0** | ~5.9 GB | **A/B** | Only Vulkan file that stays on GPU; decode still slow pending [PR #188](https://github.com/PrismML-Eng/llama.cpp/pull/188) |

## Build (HIP / ROCm)

Separate clone so this folder’s turboquant Vulkan tree stays untouched. `prism-b10658` or newer. Needs the ROCm toolkit (`hipcc`).

```bash
cd ~/Documents/GitHub
git clone -b prism https://github.com/PrismML-Eng/llama.cpp.git llama.cpp-prism
cd llama.cpp-prism
git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="gfx1100"
cmake --build . --config Release -j$(nproc)
cd bin
```

Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) branch **`prism`**. `"gfx1100"` is Navi 31 (7900 XTX), not a CUDA sm. Omit `-DAMDGPU_TARGETS` to autodetect. **Do not** add `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` to this HIP build.

Confirm before debugging flags:

```bash
./llama-server --version          # prism-b10658 or higher (releases have been prism-b10709+)
./llama-server --help | grep -E 'PTQ1|PQ2' || true
./llama-server --list-devices     # expect a ROCm/HIP GPU, not CPU-only
```

If load later fails with unknown type `PTQ1_0` / `PQ2_0`, you launched **turboquant** or stock llama.cpp. Check `pwd` is `~/Documents/GitHub/llama.cpp-prism/build/bin`.

**Optional shortcut (prebuilt):** [PrismML Ubuntu x64 ROCm 7.2](https://github.com/PrismML-Eng/llama.cpp/releases/latest) (`llama-prism-*-bin-ubuntu-rocm-7.2-x64.tar.gz`). Match the ROCm major on the box.

## PRIMARY command

Research baseline — 24 GB RDNA3 **HIP** + Bonsai 2 **PQ2_0**. No mmproj. Thinking **off** for Pi. Run from `llama.cpp-prism/build/bin`.

```bash
pkill -9 llama-server

./llama-server \
  --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  --alias bonsai-2-27b \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 131072 \
  --fit off \
  --n-gpu-layers 99 \
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
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values (this box)

| Flag | Why |
| --- | --- |
| HIP binary, not Vulkan turboquant | `PQ2_0` kernels are on ROCm; this folder’s Qwen Vulkan build cannot load these types |
| `--model …-PQ2_0.gguf` | Demo default + ROCm kernels; swap the filename to try PTQ1_0 |
| `--ctx-size 131072` | Same 24 GB Bonsai pin as the [3090](../Win-RTX3090-24GB/Windows-RTX3090-Bonsai-2-27B.md). This folder’s Qwen 27B used **128k** with ~18 GB weights; Bonsai 2 has ~7 GB. Half-window **65536**. Native **262k** is the stretch |
| `q8_0` / `q8_0` | Same KV as this folder’s tested Qwen. This fork has **no** `turbo*` — if you OOM, drop `--ctx-size` or batch |
| Batch 1024 | Same ubatch as this folder’s Qwen 27B (already tested on 24 GB with *larger* weights). Drop to 256 on prefill OOM |
| `--cache-ram 0` | Hybrid Qwen / DeltaNet multi-turn ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| `--reasoning off` | Pi needs `message.content` / tools. Bonsai 2 **thinks by default** if you leave this on |
| Sampling | Dual RTX / this folder’s Qwen **Pi tools** pin — **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 16384` | Match Pi `maxTokens`. This folder’s Qwen 27B uses 65536; keep that only if you want it, and raise Pi `maxTokens` together |
| No `--main-gpu` | Single AMD card |
| `--alias bonsai-2-27b` | Matches the Pi JSON `id` |

Universal flags (`--fit off`, loopback, no checkpoints): [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) still describes them; this binary is the **PrismML** fork, so ignore turbo V / TQ weight types.

### Confirm

```text
log: n_ctx_seq (131072)
log: load_mode = none
# types are PQ2_0 (or PTQ1_0), not a refused unknown type
# offload is HIP/ROCm, not "CPU" for the matmuls
rocm-smi   # or amd-smi; MiB after load and after a short decode
```

Then: (1) load, (2) first decode, (3) new Pi session with real `ls` / `read`.

If decode is **< ~5 tok/s**, the weights are on **CPU**. You are on Vulkan `PQ2_0`, a CPU-only HIP build, or the wrong binary. Stop and fix the backend before changing flags.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"bonsai-2-27b",
       "messages":[{"role":"user","content":"Reply with exactly: BONSAI_OK"}]}'
```

## Try both packs

HIP only. Same command. Only `--model` changes. `pkill` between runs.

```bash
# Same PRIMARY, only:
#   --model ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

From `build/bin`:

```bash
./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PQ2_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128

./llama-bench \
  -m ~/Documents/AIML/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
  -ngl 99 -fa 1 -p 512 -n 128
```

Keep the faster pack for interactive Pi. Smoke Pi `ls`/`read` on **each**. PrismML has no 7900 XTX tok/s row; this folder’s Qwen 27B MTP is ~60–65 t/s on Vulkan — Bonsai 2 HIP will be a different number. Do not compare until the log shows GPU offload.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. Status bar must show **~131k** (or **64k** / **262k** if you change the pin).

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "bonsai-2-27b",
          "name": "Ternary Bonsai 2 27B PQ2_0 (131k q8/q8) - 7900 XTX HIP",
          "contextWindow": 131072,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

If you settle on PTQ1_0, change `name` only. If you take **64k** (`65536`) or [262k](#this-box), change **both** the server pin and `contextWindow`. `--alias` must still match `id`. This folder’s Qwen 27B JSON uses `maxTokens` 65536 to match that guide’s `--n-predict`; Bonsai 2 PRIMARY is 16384.

## This box

**Single GPU.** ~6–7 GB weights + 131k q8/q8 hybrid KV should land well under 24 GB. Do not `--split-mode layer`. Do not load these GGUFs in the turboquant Vulkan `llama-server` from the Qwen guides.

### Context sizes (VRAM)

`--ctx-size 131072` is the PRIMARY. Half of that is **`65536` (64k)**. Weights stay ~7.2 GB (`PQ2_0`); KV is what halves. Same 24 GB math as the 3090 Bonsai pin (`q8_0`/`q8_0`, `--parallel 1`, no mmproj) — estimates, not a measured 7900 XTX `rocm-smi`:

| `--ctx-size` | KV (q8) | After load (est.) | Use when |
| --- | ---: | --- | --- |
| **131072** (PRIMARY) | ~4–5 GB | **~13–15 GB** (~15–17 GB if the window is full) | Default. This folder’s Qwen 27B was tight at 128k because *weights* were ~18 GB |
| **65536** (half) | ~2–2.5 GB | **~11–13 GB** | Extra headroom. Match Pi `contextWindow` |
| **262144** (stretch) | ~9–10 GB | **~18–21 GB** | If 131k load is comfortable. This folder’s **35B-A3B** already ran 262k on 24 GB |

llama-server usually **reserves the full KV for `--ctx-size` at startup**. `PTQ1_0` is ~1.3 GB less on weights.

```bash
# Half-window — same PRIMARY, only:
#   --ctx-size 65536
# Pi contextWindow: 65536
```

**OOM on load / first decode:** (1) batch 256, (2) `--ctx-size 65536` + Pi 65536, (3) confirm HIP not Vulkan-CPU. Last resort on this fork (no turbo V): `--cache-type-k q4_0 --cache-type-v q4_0`.

**262k stretch** if 131k load is comfortable: `--ctx-size 262144` + Pi `contextWindow` 262144. If prefill OOMs, keep 262k and drop batch to 256.

## Alternate: Vulkan — expect slow

Only if HIP is unavailable. Build a **second** tree or `build-vulkan/` — do not mix HIP and Vulkan in one cmake.

```bash
cd ~/Documents/GitHub/llama.cpp-prism
cmake -B build-vulkan \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_VULKAN=ON
cmake --build build-vulkan --config Release -j$(nproc)
cd build-vulkan/bin
```

Use **`Ternary-Bonsai-2-27B-PTQ1_0.gguf` only**. Current `prism` Vulkan: PTQ1_0 stays on GPU but decode is a generic trit shader (~5 tok/s class on smaller RDNA; [issue #185](https://github.com/PrismML-Eng/llama.cpp/issues/185) / [#186](https://github.com/PrismML-Eng/llama.cpp/issues/186)). **`PQ2_0` falls back to CPU.** Same PRIMARY flags otherwise. After [PR #188](https://github.com/PrismML-Eng/llama.cpp/pull/188) merges, rebuild and re-bench both packs — then Vulkan can match this folder’s usual backend.

Prebuilt: `llama-prism-*-bin-ubuntu-vulkan-x64.tar.gz` from [releases](https://github.com/PrismML-Eng/llama.cpp/releases/latest).

Sampling, thinking on, vision: **[Dual RTX Bonsai 2 — optionals](../Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md#bonsai-2-optionals)**. HIP image tokens should follow PrismML’s ROCm path (uncapped); Vulkan/CPU downscale large images.

## See also

- This folder’s tested Qwen 27B (Vulkan, turboquant, MTP): [7900-XTX-Qwen3.6-27b.md](7900-XTX-Qwen3.6-27b.md)
- This folder’s tested 35B-A3B (Vulkan, 262k): [7900-XTX-Qwen3.6-35b-a3b.md](7900-XTX-Qwen3.6-35b-a3b.md)
- Dual RTX Bonsai 2 (CUDA, 262k): [Dual-RTX6000-Bonsai-2-27B.md](../Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md)
- 24 GB CUDA WSL2 Bonsai 2: [Windows-RTX3090-Bonsai-2-27B.md](../Win-RTX3090-24GB/Windows-RTX3090-Bonsai-2-27B.md)
- Model card: [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) · docs: [Ternary Bonsai 2 27B](https://docs.prismml.com/bonsai-2-27b) · [run llama.cpp](https://docs.prismml.com/run/llamacpp)
- Fork: [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp) · Vulkan kernel PR: [#188](https://github.com/PrismML-Eng/llama.cpp/pull/188)
- Pi: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware)

**Last Updated:** 2026-09-20 (recipe density; HIP PRIMARY; Vulkan incomplete; ⚠️ untested)
