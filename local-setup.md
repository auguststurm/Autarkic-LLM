# local-setup.md

Common setup steps for all platforms in **Autarkic-LLM**.

## 1. Prerequisites

### Linux (Ubuntu 24.04+ recommended)

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl python3 python3-pip python3-venv ninja-build

# For CUDA (DGX Spark, RTX cards)
# Install CUDA Toolkit from NVIDIA (JetPack for Jetson)
```

### macOS (Apple Silicon)

```bash
xcode-select --install
brew install cmake git python@3.11
```

### Windows (WSL2 recommended for best results)

Use **WSL2** with Ubuntu. Native Windows builds are possible but less stable for CUDA.

## 2. Clone & Build llama.cpp-turboquant

```bash
cd ~/Documents/GitHub

git clone --depth 1 --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp-turboquant

cd llama-cpp-turboquant

# Clean previous build (recommended when updating)
rm -rf build

mkdir build && cd build
```

**Platform-specific CMake commands** (see hardware-specific guides for exact flags):

**Linux / CUDA:**

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build . --config Release -j$(nproc)
```

**macOS (Metal):**

> ⚠️ **Apple Silicon build fix (required as of fork commit `0c8fcfe73`, June 2026).** The fork dropped the dead `rnorm` field from the `block_turbo4_0` CPU struct (`ggml/src/ggml-common.h`, now 66 bytes) but left two writes to it in the Metal shader. macOS compiles that shader at `llama-server` **startup**, so the build links fine and then crashes on launch with `no member named 'rnorm'`. Delete the two offending lines first — in `ggml/src/ggml-metal/ggml-metal.metal`, remove the line `dst.rnorm = half(0.0f);` and the line `blk.rnorm = half(0.0f);  // reserved field, unused in 4-bit mode` (search for `rnorm`; both are zero-writes to a now-removed field, safe to delete). A fix is proposed upstream ([PR #200](https://github.com/TheTom/llama-cpp-turboquant/pull/200)) but not yet merged as of July 2026 — apply this patch until it lands (or until the fork otherwise gates these writes behind `#if !TURBO4_USE_4BIT`).

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
cmake --build . --config Release -j$(sysctl -n hw.logicalcpu)
```

## 3. Model Download (Unsloth GGUF)

### Model catalog (Hugging Face)

All configs use [Unsloth](https://huggingface.co/unsloth) GGUF builds (Dynamic 2.0 "UD" quants) so they run on [llama.cpp](https://github.com/ggml-org/llama.cpp). Quant suffixes (`Q4`, `Q6`, `IQ2`, `_K_XL`, …) trade file size/memory for quality: see the [Glossary](glossary.md) and [Unsloth's quant guide](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs).

| Model | Type | GGUF files & downloads | Original weights |
| --- | --- | --- | --- |
| Qwen3.6-27B | Dense, 262K ctx | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/tree/main) | [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) |
| Qwen3.6-35B-A3B | MoE (3B active) | [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main) | [QwenLM/Qwen3.6](https://github.com/QwenLM/Qwen3.6) |
| Gemma 4 E2B | Dense edge (PLE) | [unsloth/gemma-4-E2B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/tree/main) | [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) |

Collections: [Qwen3.6 (Unsloth)](https://huggingface.co/collections/unsloth/qwen36) · [Gemma 4 (Unsloth)](https://huggingface.co/collections/unsloth/gemma-4). MTP variants (e.g. `*-MTP-GGUF`) offer ~1.5–2× faster decode via multi-token prediction. Pick the largest quant that fits your memory budget: each hardware guide names the exact file, and the [M4 Mac Mini guide](M4-Mac-Mini-16GB/M4-Mac-Mini-Qwen3.6.md) shows how to reason about the budget.

### Download

> **Disk space:** GGUFs are large. The Qwen3.6-27B `Q6_K_XL` is ~22 GB, the 35B-A3B `Q4_K_XL` ~22 GB (down to ~11.5 GB for the `IQ2_M` used on 16 GB Macs), and Gemma 4 E2B `Q4_K_S` ~3 GB. Make sure you have the room — and note `hf_transfer` downloads can momentarily use extra space.

```bash
pip install -U huggingface_hub hf_transfer

# Qwen example (hf is the current CLI; older docs use `huggingface-cli download`)
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q6_K_XL.gguf \
  --local-dir ~/Documents/AIML/Models/unsloth/Qwen/LLM

# Gemma example (for the Jetson / Mac Mini guides)
hf download unsloth/gemma-4-E2B-it-GGUF \
  gemma-4-E2B-it-Q4_K_S.gguf \
  --local-dir ~/Documents/AIML/Models/unsloth/gemma-4-E2B
```

> **Gated models:** the Unsloth GGUF repos above are generally open, but the *original* Google/Qwen weights (and some mirrors) may be gated. If a download 401s, run `hf auth login` and accept the model's license on its Hugging Face page first. Confirm the exact quant filename on the repo's "Files" tab: available quants vary per model.

## 4. Create KV Cache Directory

```bash
# From the build directory; the llama-server binary is in build/bin
cd bin
mkdir -p ./kv-cache
```

## 5. Common Best Practices

- **Network exposure (read this).** Every command in these guides binds `--host 0.0.0.0`, which serves the model to *every device on your network with no authentication* — and typically with a shell-capable [agentic harness](agentic-harnesses.md) behind it. That is at odds with the "nothing leaves the box" goal. For a truly self-contained machine, use `--host 127.0.0.1` (loopback only). Keep `0.0.0.0` **only** on a trusted LAN where you deliberately reach the server from other machines, and put it behind a firewall.
- Always run `pkill -9 llama-server` before starting a new instance. The `-9` (SIGKILL) is deliberate, not lazy: llama-server's graceful shutdown can hang on SIGTERM/SIGINT (upstream issues [#11742](https://github.com/ggml-org/llama.cpp/issues/11742), [#20921](https://github.com/ggml-org/llama.cpp/issues/20921)), and it does **not** auto-save KV state on exit — slot saves to `--slot-save-path` happen only on explicit API calls, and checkpoints (`--cache-ram`) are ephemeral — so a hard kill loses nothing persistent.
- Use `--no-mmap` on systems with sufficient RAM.
- On memory-constrained machines, add `--fit on --fit-target <MiB>` (e.g. `256`): llama-server auto-fits the model to free memory by adjusting GPU-layer offload and choosing an effective context. **Leave `--n-gpu-layers` and `--ctx-size` unset when using `--fit`** — on the current fork it aborts if `--n-gpu-layers` is set and won't shrink a pinned `--ctx-size`, so the two approaches are mutually exclusive. Check the startup log for the context it allocated. (Used in the M4 MacBook Air and 16 GB Mini Qwen guides; roomier configs pin context explicitly and skip `--fit`.)
- Monitor resources:
  - Linux: `htop`, `nvidia-smi -l 1`
  - macOS: `htop` + Activity Monitor
  - Jetson: `jtop`
- Update procedure: `git pull origin feature/turboquant-kv-cache` → delete `build/` → rebuild.

## 6. Pi Coding Agent / Hermes Integration

Create a `models.json` file pointing to your local server:

> **Important:** match these settings to the model you actually loaded with `llama-server`. Mismatched values cause truncation, errors, or wasted memory.
>
> - **`id`** / **`name`**: identify the real model you launched.
> - **`contextWindow`**: must not exceed your `--ctx-size` (and the effective context after `--fit` — check the startup log).
> - **`maxTokens`**: must not exceed your `--n-predict`.
> - Each hardware guide lists the exact snippet for its model — copy from there.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "none",
      "models": [
        {
          "id": "your-model",
          "name": "Your Model Name",
          "contextWindow": 262144,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

## Next Steps

- Go to your hardware-specific folder (e.g. `DGX-Spark-128GB/`) for exact build flags, model recommendations, and optimized `llama-server` commands.
- For what the TurboQuant fork adds and what every `llama-server` flag does (and when *not* to use it), see [`llama-cpp-turboquant.md`](llama-cpp-turboquant.md).
