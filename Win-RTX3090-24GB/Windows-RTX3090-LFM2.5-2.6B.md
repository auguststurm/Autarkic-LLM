# Windows RTX 3090 (24 GB) - LFM2.5-2.6B

> ⚠️ **Not yet tested** as a Pi daily driver on this SKU. Ported **2026-09-20** from the ✅ Jetson 64k pin. **Build on a fresh Win11 + WSL Ubuntu RTX 3090 (OMEN 30L) does *not* take the 4090 / Jetson cmake** — `FA_ALL_QUANTS` + `"86"` fails `ptxas` (D=512 turbo FA vec, 64KB shared vs 48KB cap). Use the 3090 cmake in **Build** below (`FA=OFF`, `86-real`, no VMM/graphs). Confirm load → first decode → Pi tools before relying on it. Twin: [Jetson Orin LFM2.5](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md) (✅ Tested 2026-09-08 @ **64k**). Sibling on this box: [Ternary Bonsai 2 27B](Windows-RTX3090-Bonsai-2-27B.md) (⚠️ untested, **PrismML fork**) — do not copy those pins here.

**WSL2** (not native Windows) · CUDA sm_**86** (Ampere GA102) · llama-cpp-turboquant. Paths: **`~/AIML`** (models) · **`~/GitHub`** (engine) — WSL ext4, **not** `/mnt/c/...`. Same path convention as the [RTX 4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md); **not** that guide’s cmake (`"89"` + `FA_ALL_QUANTS`). 24 GB VRAM does not make Ada flash-attn kernels compile on Ampere.

24 GB discrete GDDR6X vs the Jetson’s 8 GB unified. Weights are **2.87 GB**; q8/q8 KV at native **131072** is **1,088 MiB** (measured on the Jetson — KV geometry, not this SKU’s `nvidia-smi`). Native train length is the PRIMARY here because that KV is cheap, **not** because 128k was run on a 3090. The Jetson’s 2026-09-08 test stopped at 64k; 128k was a stretch there too. Confirm **load → first decode → a long prefill**, then Pi tools. Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent). If you OOM, drop **batch** first, then `--ctx-size`; never bare `--fit on`. Do **not** drop `--n-predict` / `maxTokens` to 4096 — that is the first-turn Pi `length` stop ([below](#pi-truncation-on-the-first-turn)).

The [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B) does **not** recommend this model for agentic coding. On the Jetson it is a strong Pi daily driver anyway. This 24 GB pin keeps that **model, sampling, and Pi JSON shape**, with a native 128k window — **not** the Jetson/4090 cmake or `--flash-attn on`. The GGUF template always opens `<think>`; for Pi **skills / tool calls** start with `reasoning` **false** ([below](#pi-coding-agent-modelsjson)).

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested decode (Jetson ✅ 64k). CMake: FA_ALL_QUANTS **fails** on this SKU — PRIMARY is FA **off** |
| **Weights** | `LFM2.5-2.6B-Q8_0.gguf` (2.87 GB) |
| **Catalog** | [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF) · [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |
| **Context** | `--ctx-size 131072` (`--fit off`) · Pi `contextWindow` **131072** · half-window **65536** |
| **KV** | `q8_0` / `q8_0` (do **not** turbo V — FA is **off**; turbo KV needs flash-attn) |
| **Flash-attn** | **off** (`-DGGML_CUDA_FA=OFF`; Jetson/4090 stay `on`) |
| **CMake** | `86-real` · `FA=OFF` · `NO_VMM` · graphs off · **no** `FA_ALL_QUANTS` |
| **Output** | `--n-predict 16384` · Pi `maxTokens` **16384** (thinking counts against this) |
| **Sampling** | temp **0.1** · top_k **50** · repeat **1.1** (Liquid card) |
| **Thinking** | Template always opens `<think>` (omit server `--reasoning off`). **Pi skills/tools:** `reasoning` **false**, no `thinkingLevelMap`. **Pi traces:** `reasoning` **true**, `thinkingLevelMap.off` **null** |
| **Paths** | `~/AIML/models` · `~/GitHub/llama-cpp-turboquant` (WSL2) |

Need CUDA in WSL first? Do **not** skip **First-time WSL** below on a machine that has never had WSL. [local-setup.md](../local-setup.md) clone path is `~/Documents/GitHub` — **this box uses `~/GitHub`**. Hardware not in the table? [ai-assisted-setup.md](../ai-assisted-setup.md).

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

Q8_0 (2.87 GB) is the quality pin tested on the Jetson. **Q6_K** (2.22 GB) is that board’s headroom swap — skip it on 24 GB. Optional full-precision: [F16](#f16-optional) (5.4 GB).

## Build (Ampere / sm_86)

Do **not** paste the [4090](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md) or [Jetson](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md) cmake. Those set `-DGGML_CUDA_FA_ALL_QUANTS=ON` and `"89"` / `"87"`. On this card that compiles `flash_attn_ext_vec<512, …>` TurboQuant types (`fattn-vec-instance-turbo4_0-f16.cu.o`, `q8_0-turbo{2,3,4}_0.cu.o`). `ptxas` then dies: `uses too much shared data (0x10100 bytes, 0xc000 max)` — 64KB static shared vs a 48KB cap. Wiping `build/` and setting `"86"` is **not** enough: the fork’s default arch list can still emit `61-virtual`, and D=512 turbo FA vec fails even as `86-real`. LFM’s `head_dim` is **64**; this box does not need those kernels.

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
| `86-real` | Ampere GA102 only. Not `"86"` (can keep Pascal `61-virtual` from a dirty cache / fork default list). Not the 4090’s `"89"` |
| `GGML_NATIVE=OFF` | Stop the host from adding extra virtual archs |
| `GGML_CUDA_FA=OFF` | Skip all FA kernels, including the D=512 turbo vec that `ptxas` rejects. PRIMARY then uses `--flash-attn off` |
| `GGML_CUDA_GRAPHS=OFF` + `GGML_CUDA_NO_VMM=ON` | WSL CUDA graphs / VMM are a likely segfault source (incomplete FA binary or WSL VMM). Do not re-enable until a small GPU run is clean |
| No `FA_ALL_QUANTS` | Jetson/4090 extra. Instantiates turbo×head-dim combos this card cannot compile |
| No `GGML_CUDA_F16` | Not in the OMEN 30L recipe. Add `-DGGML_CUDA_F16=ON` only after FA=OFF already links |

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (`feature/turboquant-kv-cache`). TheTom is Metal-leaning; CUDA FA+turbo on 3090 is the painful path. This LFM pin does **not** need turbo V (KV ~1.1 Gi at 128k) — FA off + q8/q8 is the compile that matches the card. A CUDA-first fork ([spiritbuun/llama-cpp-turboquant-cuda](https://github.com/spiritbuun/llama-cpp-turboquant-cuda)) is only worth switching if you later want FA+turbo on this SKU; it is **not** this repo’s engine.

Do not launch the [Bonsai PrismML](Windows-RTX3090-Bonsai-2-27B.md) binary for this GGUF.

**If `ptxas` still mentions `flash_attn_ext_vec` / `0xc000`:** `build/` was not wiped, or `CMakeCache.txt` still lists `61-virtual` / `FA_ALL_QUANTS`. Delete `build/` and rerun the cmake above. Do not “fix” it by copying the 4090 flags.

**If `llama-server` segfaults** after a failed FA compile, treat the binary as bad — rebuild with `FA=OFF` as above. From the **repo root** (`~/GitHub/llama-cpp-turboquant`):

```bash
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
# CPU first (proves the GGUF / binary)
./build/bin/llama-server -m ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  -ngl 0 -c 2048 --flash-attn off --port 8080
# then GPU, small window
./build/bin/llama-server -m ~/AIML/models/LFM2.5-2.6B-Q8_0.gguf \
  -ngl 99 -c 4096 --flash-attn off \
  --cache-type-k q8_0 --cache-type-v q8_0 --port 8080
# gdb -q -ex run -ex bt -ex quit --args ./build/bin/llama-server ...
```

GGUF must be under `~/AIML`, not `/mnt/c`. `ldd build/bin/llama-server` should see `libcuda` from `/usr/lib/wsl/lib`. Do not “fix” a segfault by turning `--flash-attn on` — this binary has no FA kernels.

## PRIMARY command

Run from `~/GitHub/llama-cpp-turboquant/build/bin`. FA is **off** because this binary was built with `-DGGML_CUDA_FA=OFF`. Values are `on|off|auto` (default `auto`) — use **`--flash-attn off`**, not `-fa 1`. Do **not** pass `--cache-type-v turbo4` (or turbo2/3) with FA off; this fork’s turbo KV path expects flash-attn.

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

Omit `--reasoning off` / `--reasoning-budget 0` — the chat template always starts the assistant with `<think>`. Omit `--cache-ram 0` (LFM is hybrid short-conv + GQA, not Qwen Gated-DeltaNet). Omit `--load-mode none` — mmap is the Jetson-tested path and this GGUF is 2.87 GB (the 4090 uses `none` for ~18 GB). Add it only if WSL mmap of the file is slow.

### Why these values (this box)

| Flag / value | Why |
| --- | --- |
| `--ctx-size 131072` | Native train length ([model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B); Liquid’s OpenClaw/Hermes examples also use 131072). Jetson PRIMARY is **64k** because 8 GB unified is tight; KV here is **1,088 MiB** at 128k ([budget](#context-budget), Jetson measurement). Liquid’s 32k is the *memory-constrained* example. Only **8 of 30** layers are GQA, so KV is cheap. **128k prefill on sm_86 is unmeasured** — if first decode OOMs, drop batch before ctx. Half-window **65536** if you share the GPU ([below](#this-box--only-if-primary-loads)) |
| `--n-predict 16384` | Pi’s published default (24 GB CUDA convention). Liquid’s OpenClaw example and the Jetson test use **8192**. Thinking counts against the cap — a higher budget can also let `<think>` run longer before the answer, not only give more room for tools. Start at 16384; if traces eat the turn, use the [tools JSON](#skills--tools-start-here) before raising further. Do not drop to 4096 |
| `--ubatch-size` / `--batch-size` **256 / 256** | Copied from the 4090 / this box’s Bonsai **27B** pin, not measured on LFM. Jetson used 64/128 because unified-memory prefill spikes SIGKILL; 256 was that board’s first OOM lever. On 24 GB discrete, 256 should be comfortable — still drop to **128** if 128k prefill OOMs, before cutting ctx |
| `--flash-attn off` | Matches `-DGGML_CUDA_FA=OFF`. Jetson/4090 use `on`. `auto` may still try FA |
| `--cache-type-k/v q8_0` | Quality default. **No turbo\*** — turbo KV needs FA, which this build does not have. KV ~1.1 Gi at 128k anyway |
| `--n-gpu-layers 99` | Full GPU offload |
| `--main-gpu 0` | Discrete card (WSL2 CUDA convention; Jetson omits) |
| mmap (no `--load-mode`) | Jetson-tested for this file. `--load-mode none` is the 4090’s ~18 GB WSL lever — add it only if mmap is slow |
| `--fit off` | Keep pinned context agent-visible |
| `--host 127.0.0.1` | Local-only default. WSL2 localhost forwarding reaches Windows clients on the same machine |
| `--alias lfm2.5-2.6b` | Stable id for Pi / curl (matches `models.json`) |
| `--threads 0` | Auto CPU threads (4090 / Bonsai WSL2 pin) |
| LFM sampling | `temp 0.1` / `top_k 50` / `repeat-penalty 1.1` — [model card](https://huggingface.co/LiquidAI/LFM2.5-2.6B). Do not copy Qwen Pi (`0.6` / `top_k 20`) onto this template |

Confirm **`n_ctx` / `n_ctx_seq (131072)`** in the log or `GET /v1/models`, then one short decode (not only load). Expect `kv_cache_init` ≈ **1088 MiB** for q8/q8 at 128k (Jetson figure). `n_ctx_seq (131072) == n_ctx_train (131072)` is expected on this pin. Then `nvidia-smi` **after that decode** — prefill scratch is what OOMs, not idle KV. A 17×23 prompt does not prove 128k prefill.

Thinking should land in `reasoning_content`, answer in `content`:

```bash
curl -s --noproxy '*' http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2.5-2.6b",
       "messages":[{"role":"user","content":"What is 17 * 23? Reply with just the number."}]}' \
| python3 -c "import json,sys; m=json.load(sys.stdin)['choices'][0]['message']; \
print('content  :', m.get('content')); print('reasoning chars:', len(m.get('reasoning_content') or ''))"
```

Then: new Pi session with real `ls` / `read`. Restart **both** the server and Pi after changing `--ctx-size` / `--n-predict` or `models.json`.

```text
log: n_ctx_seq (131072)
nvidia-smi   # inside WSL; MiB after load and after a short decode
```

### Context budget

Architecture from the card / `config.json`: 30 layers (22 short-conv + **8 GQA**), 8 KV heads, `head_dim` 64, `max_position_embeddings` **131072**. Conv state is ~0.3 MB and does **not** grow with context. llama.cpp q8_0 KV for this geometry is **1,088 MiB at 128k** (measured on the Jetson; scales linearly).

Weights **2.87 GB** + KV **~1.1 GB** is ~4 GB of *known* tensors. Compute buffer and **128k prefill scratch** are **not** measured on this SKU — the 6–8 GB row below is a guess, not `nvidia-smi`. This build has **CUDA graphs off**, so do not budget for graphs. WSL + Windows desktop still steal VRAM. llama.cpp pre-allocates the full KV at start; prefill can still spike above that.

| `--ctx-size` | q8/q8 KV (Jetson) | After load (est., unmeasured here) | Role |
| --- | --- | --- | --- |
| 65536 | ~544 MiB | **~5–7 GB** | Half-window — Jetson PRIMARY; use if sharing the GPU |
| **131072** | **1,088 MiB** | **~6–8 GB** (prefill may be higher) | **PRIMARY** — native train length |

f16 KV at 128k is ~2 Gi (still fine on 24 GB) if q8 misbehaves with FA off (untested). Do **not** use `turbo*` on this binary. Do not raise past **131072**.

Pi compaction default `reserveTokens` is **16384**. Compaction threshold is `contextWindow - 16384`:

| Pi `contextWindow` | Compacts after |
| --- | --- |
| 65536 | ~48k |
| **131072** | **~114k** |

Do **not** raise `reserveTokens` when you raise the window — that would eat the extra session length. Leave `--n-predict` / `maxTokens` at **16384** unless a turn still ends `length`.

Keep **q8/q8**.

### Pi truncation on the first turn

Pi’s **“Response was truncated before completion.”** is the UI for `stopReason === "length"` ([pi#7540](https://github.com/earendil-works/pi/pull/7540)) — generation hit the output allowance, not a crash.

Three documented constraints stacked on the old 16k / 4096 pin (the Jetson’s first failure mode). This 128k / 16384 pin removes the *window* and *clamp* parts; it does **not** guarantee a first turn will finish — always-on thinking can still hit `length` at 16384.

1. **Always-on thinking.** The Jinja template always starts `<think>`. Those tokens count against `--n-predict` / Pi `maxTokens` (same rule as [Qwen thinking](../agentic-harnesses.md#two-different-limits) and [Muse](../agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent)).
2. **Pi clamps the request.** It sends `max_tokens = min(model.maxTokens, contextWindow - promptTokens - 4096)` ([pi#7540](https://github.com/earendil-works/pi/pull/7540)). At `contextWindow` 16384 that is `12288 - promptTokens`. A coding-agent system prompt + tools can leave little room; `maxTokens` 4096 then caps it again. At **131072** the clamp is not the problem.
3. **Pi compaction default** `reserveTokens` is **16384** ([compaction.md](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/compaction.md)). With `contextWindow` 16384 the threshold is `16384 - 16384 = 0`. At **131072** it is ~114k.

Confirm in the llama-server log: `tokens_predicted` / eval token count equal to the cap, and the chat chunk `finish_reason: length`. That is this bug, not an OOM.

`--n-predict` is only the **default** when the request omits `max_tokens`; Pi’s `max_tokens` overrides it ([llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [pr#22873](https://github.com/ggml-org/llama.cpp/pull/22873)). Raise **both** sides anyway so they stay aligned.

## This box — only if primary loads

**Single GPU.** ~2.9 GB weights + 128k q8/q8 KV *should* land well under 24 GB. That is KV math, not a 3090 measurement. Do not `--split-mode layer`. Windows desktop + WSL still steal some VRAM — check `nvidia-smi` **inside** WSL (the Windows Task Manager number is not the WSL view).

```bash
# Half-window — same PRIMARY, only:
#   --ctx-size 65536
# Pi contextWindow: 65536
```

**A) OOM on load / first decode:** (1) free desktop GPU apps / check WSL VRAM, (2) **batch 128** (128k prefill is the likely spike), (3) `--ctx-size 65536` + Pi 65536. Status bar must match. Do **not** swap to Q6_K first — that is the Jetson 8 GB lever. Do **not** turn `--flash-attn on` or `turbo*` to “save VRAM” — this binary has no FA kernels.

**B) `n_ctx_seq` matches and VRAM has headroom:** you are already at native **131072**. Do not raise past train length. Optional: [F16](#f16-optional) at the same pin. [DSpark](#dspark-optional) is **not** a VRAM stretch here (needs FA + a flag this build may lack).

**C) Turn-1 garbage / flaky tools:** not fixed by more context. New Pi session; use the [tools JSON](#skills--tools-start-here) (`reasoning` false); confirm PRIMARY is still q8/q8 and Liquid sampling; no DRY / no client sampling override. Pi tools were field-tested on the **Jetson**, not this SKU. Liquid’s native tool markup is Pythonic `<|tool_call_start|>` — Pi’s OpenAI tool loop is a different shape that happened to work on that board.

**D) Max output token limit / empty `content` with `finish_reason: length`:** primary is already 16384. Confirm Pi restarted. If thinking ate the budget, switch to the tools JSON rather than raising further. Prefer writing `reports/*.md` and a short chat summary. Do not add server `--reasoning off` to “fix” truncation (the template still opens `<think>`). Liquid’s OpenClaw example stays at **8192** — same command, only `--n-predict` / `maxTokens` if you want that pin.

| | Jetson Orin Nano Super 8 GB | This RTX 3090 24 GB | RTX 4090 Q4 27B |
| --- | --- | --- | --- |
| Weights | LFM Q8 **2.87 GB** | LFM Q8 **2.87 GB** | Qwen3.6 Q4 ~17.6 GB |
| PRIMARY pin | **64k q8/q8** (128k stretch) | **128k q8/q8** | **96k q8/q8** |
| `--n-predict` | **8192** | **16384** | **16384** |
| Batch | 64 / 128 | **256 / 256** | 256 / 256 |
| Flash-attn | **on** (`FA_ALL_QUANTS`) | **off** (`GGML_CUDA_FA=OFF`) | **on** (`FA_ALL_QUANTS`) |
| CMake arch | `"87"` | **`86-real`** | `"89"` |
| Engine | turboquant sm_87 | turboquant sm_**86** | turboquant sm_89 |
| Paths | `~/Documents/AIML` | **`~/AIML`** (WSL2) | `~/AIML` (WSL2) |

Same LFM2.5 Q8_0 file as the Jetson. Different **KV budget**, **batch**, **FA**, and **WSL2 paths**. Do **not** copy the 4090 cmake because VRAM is also 24 GB — Ada compiles D=512 FA vec; Ampere on this fork does not.

## F16 optional

Same PRIMARY flags. Only the file changes. 5.4 GB weights still leave most of 24 GB free. Q8_0 is the **Jetson-tested** quality pin; F16 is an untested bump on this box. Liquid also ships `LFM2.5-2.6B-BF16.gguf` at the same size — llama.cpp will convert either pack; do not treat F16 vs BF16 as a measured Ampere win. F16 is the file Liquid pairs with [DSpark](#dspark-optional).

```bash
hf download LiquidAI/LFM2.5-2.6B-GGUF \
  LFM2.5-2.6B-F16.gguf \
  --local-dir ~/AIML/models
```

```bash
  --model ~/AIML/models/LFM2.5-2.6B-F16.gguf \
```

Change the Pi `name` suffix if you settle on F16. Smoke load → decode → Pi `ls`/`read` before treating it as the daily driver.

## DSpark optional

Liquid’s speculative-decoding sidecar ([LFM2.5-2.6B-DSpark-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark-GGUF); llama.cpp `#25173`). **Not part of PRIMARY.** Liquid’s sample uses **`-fa on`**. This 3090 binary is **`FA=OFF`**, so skip DSpark until a different build has flash-attn. Confirm `./llama-server --help | grep -i dspark` first — TheTom turboquant may not have `--spec-type draft-dspark` yet. Muse Glimmer’s `draft-dflash` is a different spec type.

```bash
hf download LiquidAI/LFM2.5-2.6B-DSpark-GGUF \
  LFM2.5-2.6B-DSpark-Q8_0.gguf \
  --local-dir ~/AIML/models
```

Only if `--help` shows `draft-dspark` **and** you rebuilt with FA on (not this guide’s cmake):

```bash
  --model-draft ~/AIML/models/LFM2.5-2.6B-DSpark-Q8_0.gguf \
  --spec-type draft-dspark \
  --spec-draft-n-max 10 \
  --spec-draft-n-min 0 \
```

Draft is ~349 MB (Q8_0) / ~664 MB (F16). Speculative decode is exact under greedy verify. If the flag is missing or FA is off, stay on PRIMARY.

## Pi Coding Agent `models.json`

Save **one** of these files to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Replace any earlier 16k / 32k / 64k copy. Open `/model` to reload (Pi reloads this file there; restart if the status bar is stale). `/new` after a pin or reasoning-shape change.

`maxTokens` ≤ `--n-predict` (16384). `contextWindow` = `--ctx-size` (131072). Both blocks use `id` **`lfm2.5-2.6b`** to match `--alias`. `compat.supportsReasoningEffort: false` is Liquid’s Pi note for this endpoint. Same llama-server command either way — do not add `--reasoning off`; the [chat template](https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/chat_template.jinja) always ends the generation prompt with `<|im_start|>assistant\n<think>`.

### Skills / tools (start here)

Field-tested on the Jetson while first building Pi **skills** and calling specific tools. Set `reasoning` **false** and **omit** `thinkingLevelMap`. Pi then does not run the reasoning-model path, so the agent invokes the tool instead of thinking about invoking the tool. Too much thinking for things that needed to just work.

This does **not** strip `<think>` from the GGUF. The model may still spend think tokens (they still count against `maxTokens` / `--n-predict`). What changes is Pi: no `thinkingLevelMap.off: null` lock, no reasoning-harness handling of `reasoning_content` before the tool loop.

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

Pi’s documented shape for a model that cannot disable thinking ([models.md](https://github.com/earendil-works/pi/blob/main/packages/coding-agent/docs/models.md)): `reasoning` **true**, `thinkingLevelMap.off` **null** (hides Off in the UI). Use this when you want traces in `reasoning_content` and a clean `content` / tool split. Worse for early skill work: the model burns the output budget thinking about the skill instead of calling it.

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

If you take the [half-window](#this-box--only-if-primary-loads), only change `contextWindow` to **65536** and the `name` suffix. `/new` after switching between the two JSON shapes.

## WSL2

- First install: **First-time WSL** above. Driver on Windows; toolkit from the **wsl-ubuntu** repo; `LD_LIBRARY_PATH` includes `/usr/lib/wsl/lib`.
- `nvidia-smi` **inside** WSL. Compute-cap query is often empty; 3090 is still **86**. One long-lived server; don’t share the GPU heavily with Windows games / browsers. WSL may not see all 24 GB if the Windows desktop is using the card.
- GGUFs and the build tree under `~/` (ext4). Not `/mnt/c`.
- Loopback `--host 127.0.0.1` is reachable from Windows browsers/clients on the same machine (WSL2 localhost forwarding).
- This guide’s Pi JSON path is **`~/.pi/agent/models.json` inside WSL** (Pi running in the Ubuntu distro). If Pi is installed on Windows itself, that file is `%USERPROFILE%\.pi\agent\models.json` — still point `baseUrl` at `http://127.0.0.1:8080/v1`.
- Workflows: [pi-coding-agent-graphs.md](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)

## Performance notes

- ⚠️ **Untested** as a Pi daily driver. Ported from the Jetson ✅ **64k / 8192** pin (128k was a stretch there). Fresh Win11 WSL on a 3090: **do not use the 4090/Jetson cmake**. PRIMARY is **128k / 16384 / FA off / q8/q8**. For **skills / specific tool calls**, use the [tools JSON](#skills--tools-start-here) (`reasoning` false, no `thinkingLevelMap`). Keep the [think JSON](#thinking-on) when you want traces.
- `n_ctx_seq (131072) == n_ctx_train (131072)` is expected on the daily pin.
- Q8_0 is the quality pin (tested on the Jetson). F16 is the optional bump, not the first download. Q6_K is the 8 GB headroom swap — skip it here. DSpark needs FA + `draft-dspark`; skip it on this FA-off binary.
- No tok/s number for this SKU — FA off will be slower than the Jetson’s FA-on pin; do not copy Jetson or H100 figures. Record `timings` after `nvidia-smi` is stable.
- After pin changes, restart **llama-server and Pi** so the status bar matches `131072` / `16384` (or `65536` / `16384` on the half-window).
- Flag deep-dive: [`llama-cpp-turboquant.md`](../llama-cpp-turboquant.md).

## See also

- Jetson Orin Nano Super (8 GB, ✅ 64k): [Jetson-Orin-LFM2.5-2.6B.md](../Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md)
- This box, 27B ternary (PrismML fork, ⚠️ untested): [Windows-RTX3090-Bonsai-2-27B.md](Windows-RTX3090-Bonsai-2-27B.md)
- 24 GB CUDA WSL2 twin (Qwen3.6 Q4, turboquant, **Ada cmake — do not paste here**): [Windows-RTX4090-Qwen3.6.md](../Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md)
- Model card: [LiquidAI/LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) · GGUF: [LiquidAI/LFM2.5-2.6B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF)
- Pi: [agentic harnesses — LFM2.5](../agentic-harnesses.md#lfm25-26b--pi-coding-agent)

**Last Updated:** 2026-09-20 (OMEN 30L WSL: FA off, 86-real, no VMM/graphs; DSpark skipped; ⚠️ decode untested)
