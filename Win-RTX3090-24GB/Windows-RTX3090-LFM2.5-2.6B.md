# Windows RTX 3090 (24 GB) - LFM2.5-2.6B

> ⚠️ **Not yet tested** as a Pi daily driver on this SKU. Run the sections in order. Confirm the build links, then load → short decode → Pi tools, and report via issue/PR.

**WSL2 Ubuntu 26.04 on Win11** · CUDA **13.2** · sm_**86** (Ampere GA102) · llama-cpp-turboquant. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested decode |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · Pi `contextWindow` **131072** |
| **KV** | `q8_0` / `q8_0` |
| **Flash-attn** | **on** (build default) |
| **CMake** | `GGML_CUDA=ON` · `CMAKE_CUDA_ARCHITECTURES="86"` |
| **Toolkit** | `cuda-toolkit-13-2` from the WSL-Ubuntu repo · `/usr/local/cuda-13.2` |
| **Output** | `--n-predict 16384` · Pi `maxTokens` **16384** |
| **Sampling** | temp **0.1** · top_k **50** · repeat **1.1** (Liquid card) |
| **Thinking** | Template always opens `<think>`. **Pi skills/tools:** `reasoning` **false**. **Pi traces:** `reasoning` **true**, `thinkingLevelMap.off` **null** |
| **Paths** | `~/AIML/models` · `~/GitHub/llama-cpp-turboquant` |

## 1. First-time machine setup

Skip this section when `nvcc --version` already prints `release 13.2` and `hf --help` runs.

**Windows (PowerShell, Admin once):** `wsl --install` (Ubuntu), reboot, confirm the Windows NVIDIA driver is current. The driver stays on Windows. The toolkit stays in Ubuntu. Packages `cuda`, `cuda-drivers`, `nvidia-driver-*`, and `nvidia-cuda-toolkit` install a Linux driver over the WSL `libcuda` stub. `nvidia-smi` inside WSL must report CUDA **13.2** or newer (driver branch R595, 595.45+).

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl ninja-build \
  python3-full python3-pip python3-pip-whl python3-venv python3-dev
rm -rf ~/.hf-cli
python3 -m venv ~/.hf-cli
~/.hf-cli/bin/python -m pip install -U pip huggingface_hub
mkdir -p ~/.local/bin
ln -sf ~/.hf-cli/bin/hf ~/.local/bin/hf
export PATH="$HOME/.local/bin:$PATH"
hf --help
```

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-2
line='export PATH=/usr/local/cuda-13.2/bin${PATH:+:${PATH}}'
grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
export PATH=/usr/local/cuda-13.2/bin${PATH:+:${PATH}}
nvcc --version
nvidia-smi
```

That `PATH` line is the [post-install step](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-installation-guide-linux/index.html#environment-setup) for this toolkit. `LD_LIBRARY_PATH` is the runfile installer step, not this deb install. `nvcc --version` prints `release 13.2`. This card is compute capability 8.6.

Models and the clone live under `~/`. A GGUF opened from `/mnt/c/...` segfaults.

## 2. Download

```bash
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/AIML/models
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/AIML/models
```

Q8_0 (2.87 GB) is the weights file. Optional F16 (5.4 GB): same command, filename `LFM2.5-2.6B-F16.gguf`.

## 3. Build

CUDA 13.2 is the toolkit whose headers match Ubuntu 26.04’s `noexcept` `rsqrt`. LFM2.5 head size is 64. `q8_0` / `q8_0` flash attention is in the fork’s default CUDA set.

If [Bonsai](Windows-RTX3090-Bonsai-2-27B.md) already occupies `~/GitHub/llama.cpp-prism`, leave that tree.

```bash
mkdir -p ~/GitHub
cd ~/GitHub
if [ ! -d llama-cpp-turboquant ]; then
  git clone --depth 1 --branch feature/turboquant-kv-cache \
    https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp-turboquant
fi
cd ~/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build --config Release -j$(nproc)
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (`feature/turboquant-kv-cache`). `"86"` is this card’s compute capability, the value in [llama.cpp’s build doc](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md).

## 4. Check the binary

```bash
cd ~/GitHub/llama-cpp-turboquant/build/bin
ldd ./llama-server | grep -E 'cudart|libcuda'
./llama-server --version
```

`libcudart` resolves under `/usr/local/cuda-13.2`. `libcuda` resolves under `/usr/lib/wsl/lib`. `--version` prints `ggml_cuda_init` and `NVIDIA GeForce RTX 3090`. If `ldd` prints `libcudart.so.13 => not found`, the [install FAQ](https://docs.nvidia.com/cuda/archive/13.2.2/cuda-installation-guide-linux/index.html) says to set `LD_LIBRARY_PATH=/usr/local/cuda-13.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}`.

Short GPU load:

```bash
cd ~/GitHub/llama-cpp-turboquant/build/bin
./llama-server -m ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  -ngl 99 -c 4096 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --port 8080
```

Stop it with Ctrl-C before the primary command.

## 5. Server

```bash
pkill -9 llama-server
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
  --flash-attn on \
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

The [chat template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja) always opens `<think>`. `--jinja` selects that template. Loader is mmap. `--load-mode none` is the switch when WSL mmap of this file is slow.

| Flag | Why |
| --- | --- |
| `--ctx-size 131072` `--fit off` | Native train length, pinned. q8_0 KV for 8 KV layers at 128k is about **1,088 MiB**. On OOM: batch 128, then ctx 65536 |
| `q8_0` / `q8_0` | KV for this pin. Weights 2.87 GB + about 1.1 GB KV at 128k |
| `--flash-attn on` | Default CUDA flash-attention kernels |
| `--n-gpu-layers 99` `--main-gpu 0` | All layers on the WSL2 GPU, device 0 |
| `--parallel 1` `--kv-unified` | One slot, one KV buffer |
| `--no-context-shift` | A full window stops the request |
| Batch 256 | Physical batch for this pin |
| `--n-predict 16384` | Pi reply cap. Think tokens count against it. 4096 ends the first Pi turn with `finish_reason: length` |
| `--threads 0` | The process picks the CPU thread count |
| Sampling | Liquid card: `temp 0.1` / `top_k 50` / `repeat-penalty 1.1`. Presence, frequency, and min-p are 0 |

```text
log: n_ctx_seq (131072)
log: kv_cache_init ≈ 1088 MiB
```

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Thinking is `reasoning_content`. The answer is `content`. `nvidia-smi` inside WSL shows memory after load and after this curl. The Pi status bar reads `131072` / `16384`.

**Half-window:** `--ctx-size 65536` and Pi `contextWindow` 65536.

**OOM on load:** close other GPU apps, then `--ubatch-size 128 --batch-size 128`, then ctx `65536`.

Loopback `--host 127.0.0.1` is reachable from Windows programs on this machine.

## 6. Pi `models.json`

Save one file to `~/.pi/agent/models.json` inside WSL (`mkdir -p ~/.pi/agent`). Pi on Windows uses `%USERPROFILE%\.pi\agent\models.json` and the same `baseUrl`. `id` matches `--alias`. `/model` reloads the file. `/new` after a pin or reasoning-shape change.

### Skills / tools

`reasoning` **false**. The template’s `<think>` prefix stays in the GGUF output.

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

Empty `content` with `finish_reason: length` means the next session uses the tools JSON. The server cap stays 16384.

## See also

- This box, 27B ternary (PrismML fork): [Windows-RTX3090-Bonsai-2-27B.md](Windows-RTX3090-Bonsai-2-27B.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent)

**Last Updated:** 2026-09-26
