# Windows RTX 3090 (24 GB) - LFM2.5-2.6B

> ⚠️ **Not yet tested** as a Pi daily driver on this SKU. Port of the ✅ [Jetson LFM2.5](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md) pin (Q8_0, Liquid sampling, Pi JSON) onto this box’s Ampere cmake (`FA=OFF`, `86-real`). Confirm load → first decode → Pi tools, then report via issue/PR.
>
> Do not paste the [4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) or Jetson cmake (`FA_ALL_QUANTS`, `"89"` / `"87"`). Do not launch the [Bonsai PrismML](Windows-RTX3090-Bonsai-2-27B.md) binary for this GGUF.

**WSL2** (not native Windows) · CUDA sm_**86** (Ampere GA102) · llama-cpp-turboquant. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested decode (Jetson ✅ 64k). CMake: `FA_ALL_QUANTS` **fails** on this SKU — PRIMARY is FA **off** |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · Pi `contextWindow` **131072** |
| **KV** | `q8_0` / `q8_0` (no `turbo*` — turbo KV needs flash-attn) |
| **Flash-attn** | **off** (`-DGGML_CUDA_FA=OFF`) |
| **CMake** | `86-real` · `FA=OFF` · `NO_VMM` · graphs off · **no** `FA_ALL_QUANTS` |
| **Output** | `--n-predict 16384` · Pi `maxTokens` **16384** |
| **Sampling** | temp **0.1** · top_k **50** · repeat **1.1** (Liquid card) |
| **Thinking** | Omit server `--reasoning off`. **Pi skills/tools:** `reasoning` **false**, no `thinkingLevelMap`. **Pi traces:** `reasoning` **true**, `thinkingLevelMap.off` **null** |
| **Paths** | `~/AIML/models` · `~/GitHub/llama-cpp-turboquant` (WSL2) |

Need CUDA in WSL first? Do **not** skip **First-time WSL** below on a machine that has never had WSL. [local-setup.md](../local-setup.md) clone path is `~/Documents/GitHub` — **this box uses `~/GitHub`**.

## First-time WSL (Win11 + Ubuntu)

Field notes from an HP OMEN 30L (Win11, RTX 3090) that had **never** had WSL. Driver stays on **Windows**. Toolkit + compiler live in Ubuntu.

**Windows (PowerShell, Admin once):** `wsl --install` (Ubuntu), reboot, confirm the Game Ready / Studio NVIDIA driver is current. Then everything below is **inside** the Ubuntu distro.

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl ninja-build \
  python3-full python3-pip python3-pip-whl python3-venv python3-dev
```

System Python has no usable `pip` module until those packages land. **Do not** `pip install` into `/usr` (PEP 668). Hugging Face’s installer (`curl … hf.co/cli/install.sh`) makes `~/.hf-cli/venv` and then runs `$venv/bin/python -m pip` — if that venv was created *before* `python3-venv` worked, you get `No module named pip`. Wipe and recreate:

```bash
rm -rf ~/.hf-cli
python3 -m venv ~/.hf-cli
~/.hf-cli/bin/python -m pip install -U pip huggingface_hub
mkdir -p ~/.local/bin
ln -sf ~/.hf-cli/bin/hf ~/.local/bin/hf
export PATH="$HOME/.local/bin:$PATH"
# add the PATH line to ~/.bashrc
hf --help    # command is `hf`, not `huggingface-cli`
```

**CUDA toolkit (WSL-Ubuntu repo only):**

```bash
# Do NOT: apt install cuda, cuda-drivers, nvidia-driver-*, or nvidia-cuda-toolkit
# Those pull a Linux NVIDIA driver and break the Windows↔WSL libcuda stub.

wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
apt-cache search '^cuda-toolkit-'   # pick the latest cuda-toolkit-XX-Y
sudo apt-get -y install cuda-toolkit-13-2   # example; use what search listed
```

```bash
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
# add both to ~/.bashrc
nvcc --version
nvidia-smi                  # inside WSL; uses the Windows driver
# nvidia-smi --query-gpu=compute_cap often prints nothing under WSL. 3090 is still 86.
```

Keep clones and GGUFs under **`~/` on the WSL filesystem**. mmap / CUDA of files on `/mnt/c/...` is a known segfault source.

## Download

```bash
mkdir -p ~/AIML/models
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/AIML/models
```

Q8_0 (2.87 GB) is the quality pin tested on the Jetson. Skip Q6_K (Jetson 8 GB lever). Optional F16 (5.4 GB): same command, swap `--model` to `LFM2.5-2.6B-F16.gguf`.

## Build (Ampere / sm_86)

`FA_ALL_QUANTS` + D=512 turbo FA vec fails `ptxas` here (`0x10100` bytes vs `0xc000` max). `"86"` (not `86-real`) can keep Pascal `61-virtual` from a dirty cache. LFM `head_dim` is **64** — this box does not need those kernels.

First time only (WSL2 path — not `~/Documents/GitHub`):

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
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=OFF \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86-real
grep CMAKE_CUDA_ARCHITECTURES build/CMakeCache.txt   # must be only 86-real
cmake --build build --config Release -j$(nproc)
mkdir -p build/bin/kv-cache
```

| Flag | Why on this box |
| --- | --- |
| `86-real` | Ampere GA102 only. Not `"86"` (can keep `61-virtual`). Not the 4090’s `"89"` |
| `GGML_NATIVE=OFF` | Stop the host from adding extra virtual archs |
| `GGML_CUDA_FA=OFF` | Skip FA kernels, including the D=512 turbo vec `ptxas` rejects. PRIMARY then uses `--flash-attn off` |
| `GGML_CUDA_GRAPHS=OFF` + `GGML_CUDA_NO_VMM=ON` | WSL CUDA graphs / VMM are a likely segfault source. Do not re-enable until a small GPU run is clean |
| No `FA_ALL_QUANTS` | Instantiates turbo×head-dim combos this card cannot compile |
| No `GGML_CUDA_F16` | Not in the OMEN 30L recipe. Add `-DGGML_CUDA_F16=ON` only after FA=OFF already links |

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (`feature/turboquant-kv-cache`). This recipe stays on that fork with FA off. Do not launch the PrismML binary for this GGUF.

**If `ptxas` still mentions `flash_attn_ext_vec` / `0xc000`:** `build/` was not wiped, or `CMakeCache.txt` still lists `61-virtual` / `FA_ALL_QUANTS`. Delete `build/` and rerun the cmake above.

**If `llama-server` segfaults** after a failed FA compile, treat the binary as bad — rebuild with `FA=OFF`. From `~/GitHub/llama-cpp-turboquant`:

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
# CPU first (proves the GGUF / binary)
./build/bin/llama-server -m ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  -ngl 0 -c 2048 --flash-attn off --port 8080
# then GPU, small window
./build/bin/llama-server -m ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  -ngl 99 -c 4096 --flash-attn off \
  --cache-type-k q8_0 --cache-type-v q8_0 --port 8080
```

GGUF must be under `~/AIML`, not `/mnt/c`. `ldd build/bin/llama-server` should see `libcuda` from `/usr/lib/wsl/lib`. Do not “fix” a segfault by turning `--flash-attn on` — this binary has no FA kernels.

## PRIMARY command

Run from `~/GitHub/llama-cpp-turboquant/build/bin`. Use **`--flash-attn off`**, not `-fa 1`. Do not pass `--cache-type-v turbo*`.

```bash
pkill -9 llama-server

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

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
  --flash-attn off \
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

Omit `--reasoning off` — the [chat template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja) always opens `<think>`. Omit `--cache-ram 0`. Omit `--load-mode none` — mmap is the Jetson-tested path and this GGUF is 2.87 GB. Add `--load-mode none` only if WSL mmap of the file is slow.

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 131072` | Native train length. KV is **1,088 MiB** q8/q8 at 128k (Jetson-measured). **128k prefill on sm_86 is unmeasured** — if first decode OOMs, drop batch before ctx |
| `q8_0` / `q8_0` | Only 8 of 30 layers are GQA; turbo KV needs FA, which this build does not have |
| `--flash-attn off` | Matches `-DGGML_CUDA_FA=OFF`. `auto` may still try FA |
| mmap (no `--load-mode`) | Jetson-tested for this file. `--load-mode none` is the 4090 ~18 GB WSL lever |
| `--main-gpu 0` | Discrete card (WSL2 CUDA convention) |
| Batch 256 | Copied from this box’s 27B pin, not measured on LFM. Drop to **128** if 128k prefill OOMs, then `--ctx-size 65536` |
| `--n-predict 16384` | Pi default. Thinking counts against the cap. Do not drop to 4096 ([Jetson truncation notes](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md#pi-truncation-on-the-first-turn)) |
| `--threads 0` | Auto CPU threads (this box’s WSL2 pin) |
| Sampling | Liquid card: `temp 0.1` / `top_k 50` / `repeat-penalty 1.1` |

### Confirm

```text
log: n_ctx_seq (131072)
log: kv_cache_init ≈ 1088 MiB
nvidia-smi   # inside WSL; MiB after load and after a short decode
```

Then: (1) load, (2) short decode, (3) new Pi session with real `ls` / `read`. A 17×23 prompt does not prove 128k prefill.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Thinking lands in `reasoning_content`, answer in `content`. Restart **both** llama-server and Pi after pin or JSON changes. Status bar must match `131072` / `16384`.

## Pi Coding Agent `models.json`

Save **one** of these to `~/.pi/agent/models.json` **inside WSL** (`mkdir -p ~/.pi/agent`). If Pi is installed on Windows itself, that file is `%USERPROFILE%\.pi\agent\models.json` — still point `baseUrl` at `http://127.0.0.1:8080/v1`. `id` matches `--alias`. `/model` to reload; `/new` after a pin or reasoning-shape change.

### Skills / tools (start here)

`reasoning` **false**, no `thinkingLevelMap`. Field-tested on the Jetson for skill/tool work. Does not strip `<think>` from the GGUF.

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

Empty `content` with `finish_reason: length` → tools JSON first, not a higher cap. Do not add server `--reasoning off`.

## This box

**Half-window:** `--ctx-size 65536` and Pi `contextWindow` 65536.

**OOM on load / first decode:** (1) free desktop GPU apps / check `nvidia-smi` **inside** WSL, (2) batch **128**, (3) `--ctx-size 65536` + Pi 65536. Do not swap to Q6_K first. Do not turn `--flash-attn on` or `turbo*`.

**DSpark** (Liquid draft sidecar): skip on this FA-off binary. Liquid’s sample uses `-fa on`. Confirm `./llama-server --help | grep -i dspark` shows `draft-dspark` only if you later rebuild with FA on — not this cmake.

Loopback `--host 127.0.0.1` is reachable from Windows clients on the same machine (WSL2 localhost forwarding).

## See also

- Jetson (✅ 64k): [Jetson-Orin-LFM2.5-2.6B.md](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md)
- DGX Spark (⚠️ 128k / FA on): [DGX-Spark-LFM2.5-2.6B.md](../DGX-Spark-128GB/DGX-Spark-LFM2.5-2.6B.md)
- This box, 27B ternary (PrismML fork, ⚠️ untested): [Windows-RTX3090-Bonsai-2-27B.md](Windows-RTX3090-Bonsai-2-27B.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent)

**Last Updated:** 2026-09-20 (recipe density; FA off, 86-real, 128k q8/q8, ⚠️ decode untested)
