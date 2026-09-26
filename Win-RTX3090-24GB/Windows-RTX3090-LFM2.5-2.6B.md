# Windows RTX 3090 (24 GB) - LFM2.5-2.6B

> ⚠️ **Not yet tested** as a Pi daily driver on this SKU. Confirm the build links, then load → short decode → Pi tools, and report via issue/PR.

**WSL2 Ubuntu on Win11** · CUDA sm_**86** (Ampere GA102) · llama-cpp-turboquant. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested decode. FA **on**, `86-real` |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · Pi `contextWindow` **131072** |
| **KV** | `q8_0` / `q8_0` |
| **Flash-attn** | **on** |
| **CMake** | `86-real` · `GGML_CUDA_FA=ON` · `GGML_CUDA_F16=ON` · `GGML_CUDA_NO_VMM=ON` · `GGML_CUDA_GRAPHS=OFF` |
| **Output** | `--n-predict 16384` · Pi `maxTokens` **16384** |
| **Sampling** | temp **0.1** · top_k **50** · repeat **1.1** (Liquid card) |
| **Thinking** | Template always opens `<think>`. **Pi skills/tools:** `reasoning` **false**. **Pi traces:** `reasoning` **true**, `thinkingLevelMap.off` **null** |
| **Paths** | `~/AIML/models` · `~/GitHub/llama-cpp-turboquant` |

A machine that has never had WSL follows **First-time WSL** below before the build. Paths on this box are `~/AIML` and `~/GitHub`.

## First-time WSL (Win11 + Ubuntu)

Field notes from an HP OMEN 30L (Win11, RTX 3090) on its first WSL install. The NVIDIA driver stays the Windows Game Ready / Studio driver. The toolkit and compiler live in Ubuntu.

**Windows (PowerShell, Admin once):** `wsl --install` (Ubuntu), reboot, then confirm the Windows NVIDIA driver is current. Everything below is inside the Ubuntu distro.

```bash
sudo apt update
sudo apt install -y build-essential cmake git curl ninja-build \
  python3-full python3-pip python3-pip-whl python3-venv python3-dev
```

PEP 668 blocks `pip` into `/usr`. Install `hf` in its own venv. If `~/.hf-cli` was created before `python3-venv` was installed, `python -m pip` fails with `No module named pip` — delete that directory and recreate it:

```bash
rm -rf ~/.hf-cli
python3 -m venv ~/.hf-cli
~/.hf-cli/bin/python -m pip install -U pip huggingface_hub
mkdir -p ~/.local/bin
ln -sf ~/.hf-cli/bin/hf ~/.local/bin/hf
export PATH="$HOME/.local/bin:$PATH"
# add the PATH line to ~/.bashrc
hf --help    # the command is `hf`
```

**CUDA toolkit from the WSL-Ubuntu repo.** `nvidia-smi` inside WSL uses the Windows driver. Install **`cuda-toolkit-13-1`**. Toolkit **13.2** (`libcudart.so.13.2`) segfaults in `cuInit` on WSL2 during `ggml_cuda_init`, before `llama-server` prints a log line. The packages `cuda`, `cuda-drivers`, `nvidia-driver-*`, and `nvidia-cuda-toolkit` pull a Linux driver over the WSL `libcuda` stub.

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
apt-cache search '^cuda-toolkit-13-1'
sudo apt-get -y install cuda-toolkit-13-1
```

`cuda-toolkit-12-8` is the fallback when the 13.1 package is absent. A binary already linked to `libcudart.so.13` needs a rebuild to run on 12.8.

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
# add all three to ~/.bashrc
nvcc --version    # release 13.1
nvidia-smi
# nvidia-smi --query-gpu=compute_cap often prints nothing under WSL. This card is sm_86.
```

Clones and GGUFs live under `~/` on the WSL filesystem. mmap / CUDA of a file on `/mnt/c/...` segfaults.

## Download

```bash
mkdir -p ~/AIML/models
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-Q8_0.gguf \
  --local-dir ~/AIML/models
```

Q8_0 (2.87 GB) is the weights file for this pin. Optional F16 (5.4 GB): same command, filename `LFM2.5-2.6B-F16.gguf`.

## Build (Ampere / sm_86)

LFM2.5 has 30 layers: 22 short-conv and 8 GQA, head size **64**. The fork’s default CUDA flash-attn set includes `q8_0` / `q8_0` at head sizes 64, 128, and 256. `CMAKE_CUDA_ARCHITECTURES=86-real` emits SASS for this GA102.

`ptxas` rejects `flash_attn_ext_vec` when that kernel wants `0x10100` bytes of static shared memory. The cap on that error is `0xc000` (48 KiB), which is the head-dim 512 vec kernel. On this branch `DECL_FATTN_VEC_CASE_D512` is compiled for HIP. A cache that still lists `61-virtual` also compiles Pascal, whose shared-memory cap is 48 KiB. After cmake, the architecture line is `86-real` only and `GGML_CUDA_FA_ALL_QUANTS` is `OFF`.

First clone:

```bash
mkdir -p ~/GitHub && cd ~/GitHub
git clone --depth 1 --branch feature/turboquant-kv-cache \
  https://github.com/TheTom/llama-cpp-turboquant.git llama-cpp-turboquant
```

If [Bonsai](Windows-RTX3090-Bonsai-2-27B.md) already occupies `~/GitHub/llama.cpp-prism`, leave that tree. This GGUF uses `~/GitHub/llama-cpp-turboquant`.

```bash
cd ~/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
git pull
rm -rf build
which nvcc    # $CUDA_HOME/bin/nvcc, release 13.1
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86-real
grep CMAKE_CUDA_ARCHITECTURES build/CMakeCache.txt    # 86-real only
grep GGML_CUDA_FA_ALL_QUANTS build/CMakeCache.txt     # OFF
cmake --build build --config Release -j$(nproc)
```

| Flag | Why on this box |
| --- | --- |
| `86-real` | SASS for GA102. The cache check above catches a leftover `61-virtual` |
| `GGML_NATIVE=OFF` | CUDA arch list stays `86-real` |
| `GGML_CUDA_FA=ON` | Flash-attn. Head 64 `q8_0`/`q8_0` is in the default CUDA set |
| `GGML_CUDA_F16=ON` | FP16 compute on sm_86 tensor cores |
| `GGML_CUDA_GRAPHS=OFF` | WSL pin. Graphs stay off until a small GPU run is clean |
| `GGML_CUDA_NO_VMM=ON` | WSL pin. VMM stays off until that same run is clean |

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (`feature/turboquant-kv-cache`).

**If `ptxas` still prints `flash_attn_ext_vec` and `0xc000`:** delete `build/` and rerun the cmake. Read the two grep lines before building again. If they are already `86-real` and `OFF` and `ptxas` still fails, rebuild with `-DGGML_CUDA_FA=OFF` and run the server with `--flash-attn off`. That recovery build keeps the score buffer, so 128k prefill uses more VRAM.

**Segmentation fault and no llama log.** That is `cuInit` dying before `ggml_cuda_init` prints. `-ngl 0` still enters that path. From `~/GitHub/llama-cpp-turboquant/build/bin`, with `CUDA_HOME` set as above:

```bash
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
./llama-server --version
```

A good `--version` prints `ggml_cuda_init` and `NVIDIA GeForce RTX 3090`. If `nvcc --version` says **13.2** and `--version` is only `Segmentation fault`, install `cuda-toolkit-13-1`, point `CUDA_HOME` at `/usr/local/cuda-13.1`, and run `--version` again. The existing binary loads `libcudart.so.13`, so the 13.1 runtime satisfies it. When `--version` shows the 3090, rerun the primary command with that same `LD_LIBRARY_PATH`.

If `--version` is still a bare segfault after the 13.1 runtime is on `LD_LIBRARY_PATH`, wipe `build/` and reconfigure with `-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc` so the next link is against 13.1. `ldd build/bin/llama-server` then shows `libcudart` from `$CUDA_HOME/lib64` and `libcuda` from `/usr/lib/wsl/lib`.

**Small GPU window** once `--version` shows the card. Still in `build/bin`:

```bash
./llama-server -m ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  -ngl 99 -c 4096 --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --port 8080
```

## PRIMARY command

Run from `~/GitHub/llama-cpp-turboquant/build/bin`. `--flash-attn` is `on`. KV types are `q8_0` and `q8_0`.

```bash
pkill -9 llama-server

export CUDA_HOME=/usr/local/cuda-13.1
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"

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

The [chat template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja) always opens `<think>`. `--jinja` selects that template. Loader is mmap for this 2.87 GB file. `--load-mode none` is the switch when WSL mmap of the file is slow.

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 131072` `--fit off` | Native train length, pinned. q8_0 KV estimate for 8 KV layers at 128k is **1,088 MiB**. 128k prefill on this SKU is unmeasured — on OOM, batch 128, then ctx 65536 |
| `q8_0` / `q8_0` | KV precision for this pin. Weights 2.87 GB + about 1.1 GB KV at 128k |
| `--flash-attn on` | Same switch as `-DGGML_CUDA_FA=ON`. The score matrix stays inside the kernel |
| `--n-gpu-layers 99` `--main-gpu 0` | All layers on the WSL2 GPU, device 0 |
| `--parallel 1` `--kv-unified` | One slot, one KV buffer, so the 128k pin is the whole cache |
| `--no-context-shift` | A full window stops the request |
| Batch 256 | Starting physical batch while 128k prefill is still unmeasured |
| `--n-predict 16384` | Pi reply cap. Think tokens count against it. 4096 ends the first Pi turn with `finish_reason: length` |
| `--threads 0` | The process picks the CPU thread count |
| Sampling | Liquid card: `temp 0.1` / `top_k 50` / `repeat-penalty 1.1`. Presence, frequency, and min-p are 0 |

### Confirm

```text
log: n_ctx_seq (131072)
log: kv_cache_init ≈ 1088 MiB
nvidia-smi   # inside WSL; MiB after load and after a short decode
```

Order: (1) load, (2) short decode, (3) a new Pi session that actually runs `ls` / `read`. The curl below is the short decode. A 128k prefill is a separate run.

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Thinking is `reasoning_content`. The answer is `content`. After a pin or JSON change, restart llama-server and Pi. The Pi status bar reads `131072` / `16384`.

## Pi Coding Agent `models.json`

Save **one** of these to `~/.pi/agent/models.json` inside WSL (`mkdir -p ~/.pi/agent`). Pi installed on Windows uses `%USERPROFILE%\.pi\agent\models.json` and the same `baseUrl`, `http://127.0.0.1:8080/v1`. `id` matches `--alias`. `/model` reloads the file. `/new` after a pin or reasoning-shape change.

### Skills / tools (start here)

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

Empty `content` with `finish_reason: length` means the next session uses the tools JSON above. The server cap stays 16384.

## This box

**Half-window:** `--ctx-size 65536` and Pi `contextWindow` 65536.

**OOM on load or first decode:** (1) close other GPU apps and read `nvidia-smi` inside WSL, (2) `--ubatch-size 128 --batch-size 128`, (3) `--ctx-size 65536` and Pi `65536`. On the FA-off recovery binary, the server flag stays `--flash-attn off`.

**DSpark** (Liquid draft sidecar): optional, untried here. `./llama-server --help | grep -i dspark` lists `draft-dspark` when this binary has it. Liquid’s sample command uses `--flash-attn on`.

Loopback `--host 127.0.0.1` is reachable from Windows programs on this same machine (WSL2 localhost forwarding).

## See also

- This box, 27B ternary (PrismML fork): [Windows-RTX3090-Bonsai-2-27B.md](Windows-RTX3090-Bonsai-2-27B.md)
- Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md) · Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent)

**Last Updated:** 2026-09-26 (WSL toolkit is CUDA 13.1; 13.2 `libcudart` segfaults in `cuInit`)
