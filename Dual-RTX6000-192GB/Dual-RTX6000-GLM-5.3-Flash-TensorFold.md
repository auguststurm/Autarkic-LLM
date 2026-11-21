# Dual RTX 6000 Pro Max-Q (192 GB) — GLM-5.3-Flash with TensorFold

> Two 96 GB cards, one GLM-5.3-Flash. TensorFold runs **tensor parallel 2**: one process per GPU, half the experts on each, one OpenAI-compatible API. That is the unify step.
>
> The Huihui tree at [huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF/tree/main) is **GGUF**. TensorFold’s CUDA GLM server loads **safetensors** checkpoints (EXL3, NVFP4, FP4-Local, XSFP4). The start on this box is [Mia’s 2× RTX PRO 6000 recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold). The refusal-reduced TensorFold weights are that recipe’s gated Ablit EXL3. Huihui’s GGUF is a later, separate llama.cpp path, and only the smaller packs fit 192 GB.

Blackwell **sm_120** · Ubuntu. Weights: `/data/AIML/models/glm53-tensorfold`. Recipe checkout: `~/Documents/GitHub/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold`.

| Pin | Value |
| --- | --- |
| **Status** | ⚠️ Untested here (researched 2026-11-17). Mia’s tok/s figures are her 400 W measurement |
| **Model** | GLM-5.3-Flash, 320B total / 18B active, native window **1,048,576**, MIT |
| **Engine** | TensorFold native (`tensorfold-native`) at commit `f350cd3e`, Zig 0.17.0, Docker, 93 engine patches in the recipe |
| **Weights** | [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold) `@76c0b517`, about **164 GiB** |
| **Ablit** | [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit), gated, `ABLIT=1` |
| **GPUs** | Both. `GPUS=0,1` puts rank 0 on GPU 0. On this box GPU 1 holds the display |
| **API** | `127.0.0.1:8020`, model id `GLM-5.3-Flash` |
| **Serve** | `PARALLEL=4`, `CONTEXT=auto`, `KV=fp8`, `DENSE=q8`, `MEMORY_RESERVE_GIB=4` |
| **Drafter** | `dflash2` (Inco AI, CC BY-NC-ND 4.0). Commercial use: `DRAFTER=mtp` or `DRAFTER=none` |

Both cards stay in this server until `./stop.sh`. A Qwen `llama-server` on either card has to be stopped first. [Localmaxing](Dual-RTX6000-Qwen3.8-localmaxing.md) is the other way to use the pair: one model per card, two APIs.

## What the two ranks do

One container (`glm53-zig`) runs both ranks. Rank 1 starts first and follows rank 0. Rank 0 serves HTTP and holds the vision tower (about 1.8 GiB). Each rank holds half of the experts, the attention and KDA heads, and the vocabulary. NCCL uses loopback on this host (`NCCL_SOCKET_IFNAME=lo`). Peer-to-peer is used when NVML reports it. A `W_P2P` warning means the copy falls back to host memory and the server still comes up.

Weights stay on the GPUs. The loader stages each layer through host RAM and warns below 16 GiB available. Host RAM does not hold a second copy of the 164 GiB checkpoint, so every start reads it from disk.

`KV=fp8` with `KV_PAGES=on` puts every stream on one pool of 512-row pages. `CONTEXT` is then one request’s cap. A request that does not fit waits. `CONTEXT=auto` asks for 1,048,576 and, if the engine refuses, retries at the window the engine’s own figures allow. That window is logged in `rank0.log` and kept in `.state/context-fit.tsv`. Quote the number from that log.

Mia’s published run, both cards capped at **400 W**: 245.5 tok/s for one prose request (temperature 0.7, thinking on), 436.6 tok/s combined at four streams, prefill about 6,033 tok/s at a 32K prompt. This repo’s cards are **Max-Q**. Read `nvidia-smi` for the power limit on this host and measure here before treating 245.5 as this box.

## Which weights

| Checkpoint | Format | Disk | This pair |
| --- | --- | --- | --- |
| Mia EXL3 4 bpw `@76c0b517` | EXL3 safetensors | ~164 GiB | **Primary.** Recipe default |
| Mia Ablit EXL3 4 bpw | EXL3 safetensors, gated | ~164 GiB beside the primary | Refusal-reduced TensorFold weights. `ABLIT=1` |
| Huihui UD-IQ4_XS | GGUF, 5 shards | **156.82 GB** (146.05 GiB), summed 2026-11-17 | llama.cpp only. Highest Huihui pack in the size class a 2× Max-Q llama.cpp run has published |
| Huihui UD-Q3_K_XL | GGUF, 4 shards | **147.54 GB** (137.40 GiB), summed 2026-11-17 | llama.cpp only. More room left for KV than IQ4 |
| Huihui UD-Q4_K_XL / Q5 / Q6 | GGUF | Unsloth lists the source packs at 200 / 240 / 292 GB | Over 192 GB before KV and compute buffers |

Huihui’s card says the GGUFs come from [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) and that **only layers 15–35** (0-based) were ablated. Expert tensors were left as Unsloth quantized them. Mia’s Ablit is an abliteration of the EXL3 4 bpw checkpoint, served by the same engine as the primary. The two are different weight trees.

TensorFold 1.2.2 added `core/gguf.zig` as a first reader for Apple Silicon (standard K-quants repacked to affine weights). The CUDA GLM recipe still serves the safetensors formats above. The Mac converter `tools/glm5_q8_0_gguf_to_mlx.py` reads a **Q8_0** GGUF (Unsloth lists that pack at 341 GB) into an MLX checkpoint for Metal.

Smaller Huihui packs (UD-IQ1_S through UD-IQ3_XXS, UD-Q2_K_XL) also fit. Unsloth’s catalog sizes for those source packs run from about 93 GB to 120 GB. Download one pack. IQ4 is the llama.cpp quality pick below. Q3 is the fallback if IQ4 fails at load.

## 1. Free both GPUs

`start.sh` requires at least **88,000 MiB free** on each selected GPU and under **4,096 MiB** already in use. A desktop on GPU 1 is inside that allowance. A resident Qwen server is not.

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,power.limit --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
```

Stop each listed `llama-server` from the terminal that owns it, or `kill <pid>` for those PIDs. Re-run the first command. GPU 0 should be nearly empty. GPU 1 should show only the display.

Disk, on the volume that will hold the checkpoint:

```bash
df -h /data
```

Leave about **180 GiB** for the EXL3 checkpoint, **10 GiB** for DFlash2, and **40 GiB** for the image build. Ablit is another **164 GiB** beside the primary. `df` below that and the download will stop mid-shard; the next `./start.sh` resumes it.

Docker with the NVIDIA Container Toolkit, BuildKit, and your user in the `docker` group. The host also needs `curl`, `python3`, `flock`, `tar`, `gzip`, `sha256sum`, `nvidia-smi`, and the Hugging Face CLI:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
```

## 2. Clone and pin this host

```bash
git clone https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold.git \
  ~/Documents/GitHub/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold
cd ~/Documents/GitHub/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold
cp scripts/local.sh.example scripts/local.sh
```

`scripts/local.sh` is gitignored. Put this in it:

```bash
GPUS=0,1
PORT=8020
HOST=127.0.0.1
DATA_DIR=/data/AIML/models/glm53-tensorfold
DRAFTER=dflash2
PARALLEL=4
CONTEXT=auto
KV=fp8
DENSE=q8
MEMORY_RESERVE_GIB=4
MAX_GPU_USED_MIB=4096
```

`GPUS` is rank order. The first index is rank 0 (HTTP and the vision tower), which is the heavier rank. This box’s display is GPU 1, so `0,1` keeps rank 0 off that card. If `nvidia-smi` shows the display on GPU 0, set `GPUS=1,0`.

`HOST=127.0.0.1` keeps the API on loopback. The recipe’s own default is `0.0.0.0`. Use that only on a trusted LAN, and export `TENSORFOLD_API_KEY` so every route except `/health` requires `Authorization: Bearer`.

`DRAFTER=dflash2` downloads [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) at the recipe’s pinned revision. That drafter is **research and evaluation only**. For any other use:

```bash
DRAFTER=mtp     # the checkpoint’s own MTP head
# or
DRAFTER=none
```

Settings layer as `scripts/config.sh`, then `.env`, then `scripts/local.sh`, then exported variables. An exported variable wins. `./start.sh restart` applies a change.

## 3. Start

```bash
cd ~/Documents/GitHub/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold
DRY_RUN=1 ./start.sh
./start.sh
```

`DRY_RUN=1` prints the Docker build, both rank command lines, the `docker run`, and the smoke test. `./start.sh` checks the host, builds the image once, downloads the checkpoint once, starts both ranks, waits until `/health` answers, checks `/v1/models` for `GLM-5.3-Flash`, and sends a smoke completion.

The first start is the long one. The image build is a few minutes with BuildKit’s caches. Each rank then hashes every shard once (`WAIT_TIMEOUT` is 2400 s). Later starts reuse `DATA_DIR/zig-home` and only read and load the weights. The recipe prints `GLM-5.3-Flash-EXL3 is now LIVE! on port 8020` when the smoke test passes.

```bash
curl -s http://127.0.0.1:8020/health
curl -s http://127.0.0.1:8020/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"Hello"}]}'
```

Thinking is off by default. A request turns it on with `"chat_template_kwargs": {"enable_thinking": true}`. Image input is on. A user message can carry an `image_url` part. `VISION=0` drops the tower and returns that ~1.8 GiB on rank 0.

Logs for the running session are `logs/current/rank0.log` and `logs/current/rank1.log`. The startup line in `rank0.log` is the measured fit: model GiB, stream count, and the window `CONTEXT=auto` kept.

```bash
./start.sh restart
./stop.sh
docker logs -f glm53-zig
```

`./stop.sh` stops this recipe’s container, gives rank 0 20 s to finish open requests, and archives the logs under `logs/`. It leaves any other container alone.

### If start.sh stops

| Code or line | What to do |
| --- | --- |
| `E_GPU_MEMORY` / `E_GPU_BUSY` | A compute process still holds a GPU. Stop it. Raise `MAX_GPU_USED_MIB` only when the extra MiB is the display |
| `E_DOWNLOAD` | Free space or the network. Run `./start.sh` again; the download resumes |
| `E_PORT` | 8020 or 29583–29584 is taken. Change `PORT` / `NCCL_PORT`, or stop the holder |
| `E_HF_TOKEN` | Ablit only. See below |
| `--parallel N needs ... GiB` | Lower `PARALLEL` or `CONTEXT`. `CONTEXT=auto` already retries once |
| Out of memory mid-request | Raise `MEMORY_RESERVE_GIB`, or lower `PARALLEL` / `CONTEXT`. `VISION=0` frees rank 0 |
| `W_P2P` | NVML reports no peer-to-peer (IOMMU, ACS, or BIOS). NCCL continues over host memory |
| First start still hashing | Expected once per checkpoint. Raise `WAIT_TIMEOUT` if 2400 s is short |

## 4. Ablit (TensorFold, refusal-reduced)

Open [Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit](https://huggingface.co/Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold-Ablit), log in, and accept the gate with the account that will own the token. Create a read token. In `scripts/local.sh`:

```bash
ABLIT=1
HF_TOKEN=hf_...
```

```bash
./start.sh restart
```

The token is used for the download. It is not passed to the ranks and it is not printed. Weights land in `DATA_DIR/model-ablit` (~164 GiB) beside the published checkpoint. After that download, the server starts with no token. `ABLIT=0` and `./start.sh restart` return to the published checkpoint, which stays on disk.

Mia’s note on this checkpoint: thinking off is how these weights behave best, and they keep the stock DFlash2 drafter, which accepts a little less than on the published checkpoint. One measured exception, from the Spark recipe: a schema-constrained JSON reply dropped a digit of an id, with drafting on or off. The same request copied the id when thinking was on, and the published weights copied it with thinking off. For ids, hashes, or other exact strings, use the published checkpoint or send that request with thinking on.

`scripts/graft.py` can copy named `self_attn.o_proj` tensors from the Ablit tree into a copy of the published tree. The recipe’s README has the command. The full Ablit serve above is the path to start with.

## 5. Pi

Write `~/.pi/agent/models.json` after the server is up. `id` is the recipe’s model name. `maxTokens` matches `MAX_TOKENS` (32768). Set `contextWindow` to the window in `rank0.log` / `.state/context-fit.tsv`. Thinking is off in the pin above, so `reasoning` is false. Dummy `apiKey` `1337` is there so Pi shows the provider; this server does not check it unless `TENSORFOLD_API_KEY` is set.

```json
{
  "providers": {
    "tensorfold-glm53": {
      "baseUrl": "http://127.0.0.1:8020/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "GLM-5.3-Flash",
          "name": "GLM-5.3-Flash EXL3 (TensorFold, both RTX 6000)",
          "reasoning": false,
          "contextWindow": 131072,
          "maxTokens": 32768
        }
      ]
    }
  }
}
```

Replace `131072` with the fitted window before trusting a long Pi session. Restart Pi after the edit. This block is untested against TensorFold’s tool-call parser.

## 6. Huihui GGUF on llama.cpp

Use this section when the requirement is the Huihui files themselves. It is a second engine. It does not get the recipe’s tensor-parallel kernels, DFlash2, or the shared FP8 page pool. Layer split still puts the model on both GPUs: llama.cpp’s default `--split-mode layer` places whole layers on each card. `--split-mode row` fails at load on this model with `device CUDA0 does not support split buffers` (reported on 4× H100, same GGUF family).

Build **upstream** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), in its own directory. The GGUF architecture string is `glm5-next` (llama.cpp PR #27773). The turboquant tree used for Qwen on this box is the wrong binary. A log line `unknown model architecture: 'glm5-next'` means that checkout predates the arch.

```bash
git clone https://github.com/ggml-org/llama.cpp.git ~/Documents/GitHub/llama.cpp
cd ~/Documents/GitHub/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j --target llama-server
```

Download **one** pack. IQ4 is the pick. Q3 is the retry if IQ4 OOMs at load.

```bash
hf download huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF \
  --include 'UD-IQ4_XS/*' \
  --local-dir /data/AIML/models/Huihui-GLM-5.3-Flash-abliterated-GGUF
```

Point `--model` at shard `00001-of-00005`. The other four files stay in that directory. The first shard is metadata (about 9 MB).

Stop the TensorFold container first (`./stop.sh` in the recipe directory). Both GPUs must be visible to this process, so do not set `CUDA_VISIBLE_DEVICES` to a single card and do not pass `--split-mode none`.

```bash
cd ~/Documents/GitHub/llama.cpp
./build/bin/llama-server \
  --model /data/AIML/models/Huihui-GLM-5.3-Flash-abliterated-GGUF/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
  --alias glm53-flash-huihui \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 32768 --n-gpu-layers 99 --fit off \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 --cache-ram 0 \
  --parallel 1 --batch-size 512 --ubatch-size 512 \
  --jinja
```

`--cache-ram 0` matches the hybrid KDA layers: a disk or RAM slot snapshot does not restore that recurrent state. Start at 32k. The native window is 1,048,576. Raise `--ctx-size` only after a load and a first decode succeed. If the first decode OOMs, drop `--ubatch-size` to 256 before dropping context. If load OOMs on the display GPU, bias weight toward the other card with `--tensor-split` and retry. Vision needs the repo’s `mmproj-model-bf16.gguf` (1.16 GB) and `--mmproj`; leave it off until text decode is clean.

Q3 retry, same flags, different file:

```bash
hf download huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF \
  --include 'UD-Q3_K_XL/*' \
  --local-dir /data/AIML/models/Huihui-GLM-5.3-Flash-abliterated-GGUF
# --model .../UD-Q3_K_XL/GLM-5.3-Flash-UD-Q3_K_XL-00001-of-00004.gguf
```

A public 2× RTX PRO 6000 Max-Q llama.cpp comparison (UD-IQ3_XXS ~120.4 GB and UD-IQ4_XS ~156.8 GB, weights and KV on the GPUs) reported 32K-input median decode of 66.2 tok/s (IQ3) and 62.1 tok/s (IQ4) with MTP off, and 72.4 / 72.8 tok/s with a 3-token MTP draft. That video is a different host. It is the size-class evidence that IQ4 can load on this card pair, and the speed band to expect from layer split.

Confirm:

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-flash-huihui","messages":[{"role":"user","content":"Hello"}]}'
```

Huihui’s card describes the ablation as a proof of concept with safety filtering reduced. Read that warning before pointing an agent at it.

## Back to one model per card

`./stop.sh` in the recipe directory, then start the Qwen servers again. TensorFold and a per-card `llama-server` do not share a GPU. The tested single-card host is [Qwen3.8](Dual-RTX6000-Qwen3.8.md). Two resident models, one per card: [Localmaxing](Dual-RTX6000-Qwen3.8-localmaxing.md).

## See also

- Recipe: [MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold) · write-up: [mia-ai.net](https://mia-ai.net/models/GLM-5.3-Flash-2x-RTX-PRO-6000-TensorFold)
- Engine: [ashhart/TensorFold](https://github.com/ashhart/TensorFold) · CUDA GLM recipe: [glm-5.3-flash-cuda.md](https://github.com/ashhart/TensorFold/blob/main/docs/recipes/glm-5.3-flash-cuda.md)
- Base model: [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) · source GGUF: [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)
- Huihui: [Huihui-GLM-5.3-Flash-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-GLM-5.3-Flash-abliterated-GGUF/tree/main)

**Last Updated:** 2026-11-17 (TensorFold EXL3 primary, Ablit switch, Huihui GGUF sizes; ⚠️ untested on this host)
