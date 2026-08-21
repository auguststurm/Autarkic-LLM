# Windows RTX 4090 (24 GB) - Qwen3.6-27B

> ✅ **Tested** on this hardware with Pi Coding Agent / multi-agent research (stable tools + report-length output).

**WSL2** (not native Windows) · CUDA sm_**89** · llama-cpp-turboquant. Paths on this box: **`~/AIML`** (models) · **`~/GitHub`** (engine) — WSL2 convention, not `~/Documents/AIML`.

K vs V, two token limits, no DRY: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware). Flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md).

| Pin | Value |
| --- | --- |
| **Status** | ✅ Tested (Pi tools + report-length output) |
| **Weights** | `Qwen3.6-27B-UD-Q4_K_XL.gguf` (~17.6 GB) |
| **Catalog** | [unsloth/Qwen3.6-27B-GGUF](https://huggingface.co/unsloth/Qwen3.6-27B-GGUF) |
| **Context** | `--ctx-size 98304` (`--fit off`) |
| **KV** | `q8_0` / `q8_0` (turbo V only if raising ctx OOMs) |
| **Output** | `--n-predict 16384` |
| **Pi sampling** | temp **0.6** · top_p **0.95** · top_k **20** · presence **0** · repeat **1.0** |
| **Thinking** | `--reasoning off` |
| **Paths** | `~/AIML/models` · `~/GitHub/llama-cpp-turboquant` (WSL2) |

**This 24 GB box:** weights ~17.6 GB fixed; KV grows with the pin; compute scratch peaks on prefill. Field VRAM often ~19–21 GB at modest pins — room exists, not M5-class 48 GB. Prompt overflows seen **>32k, >64k, ~70k** one request → primary **96k** with margin. Reports need **16k** out. Raise `--ctx-size` and Pi `contextWindow` together; raising one limit does not fix the other.

Need the engine built first? [local-setup.md](../local-setup.md) (WSL2 + Ubuntu). Hardware not in the table? [ai-assisted-setup.md](../ai-assisted-setup.md).

## Download

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --local-dir ~/AIML/models
```

## Build (Ada / sm_89)

```bash
cd ~/GitHub/llama-cpp-turboquant
git checkout feature/turboquant-kv-cache && git pull
rm -rf build && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build . --config Release -j$(nproc)
cd bin && mkdir -p ./kv-cache
```

Fork: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant). Optional turbo: `./llama-server --help | grep -A2 cache-type-v`.

## PRIMARY command

Run from `~/GitHub/llama-cpp-turboquant/build/bin`.

```bash
pkill -9 llama-server

cd ~/GitHub/llama-cpp-turboquant/build/bin

./llama-server \
  --model ~/AIML/models/Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --host 127.0.0.1 --port 8080 \
  --ctx-size 98304 \
  --fit off \
  --n-gpu-layers 99 \
  --main-gpu 0 \
  --load-mode none \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-ram 0 \
  --jinja \
  --flash-attn on \
  --no-context-shift \
  --parallel 1 \
  --ubatch-size 256 \
  --batch-size 256 \
  --reasoning off \
  --reasoning-budget 0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 \
  --repeat-penalty 1.0 \
  --frequency-penalty 0.0 \
  --repeat-last-n 64 \
  --threads 0 \
  --n-predict 16384 \
  --kv-unified \
  --log-verbosity 1
```

### Why these values (this box)

| Flag | Why |
| --- | --- |
| `--ctx-size 98304` | ≥70k prompts + margin; pin stays agent-visible |
| `q8_0` / `q8_0` | Stable K (routing) + V; turbo V only when this pin OOMs |
| Batch 256 | Prefill vs peak; drop to 128 on OOM |
| `--cache-ram 0` | Hybrid DeltaNet multi-turn ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)) |
| Sampling | Qwen tool/coding profile; **no DRY** ([#20837](https://github.com/ggml-org/llama.cpp/issues/20837)) |
| `--n-predict 16384` | Full reports (match Pi `maxTokens`) |
| `--load-mode none` | Needs free **host RAM** (~18 GB+) for load on WSL2 |

`--ctx-size` is a request: confirm `n_ctx_seq (98304)`. `n_ctx_seq < n_ctx_train (262144)` is expected on 24 GB.

### Confirm

```text
log: n_ctx_seq (98304)
nvidia-smi   # inside WSL; MiB after load and after a short decode
```

Normal logs: `cache-idle-slots requires --cache-ram, disabling` (follows `--cache-ram 0`). Do not pass `enable_thinking` kwargs.

## Pi `models.json`

Save this entire file to `~/.pi/agent/models.json` (`mkdir -p ~/.pi/agent`). Restart Pi. Status bar must show **~96k**.

```json
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:8080/v1",
      "api": "openai-completions",
      "apiKey": "1337",
      "models": [
        {
          "id": "qwen3.6-27b",
          "name": "Qwen3.6-27B Q4 agent (96k q8/q8) - RTX 4090",
          "contextWindow": 98304,
          "maxTokens": 16384
        }
      ]
    }
  }
}
```

## This box — only if primary fails (one axis at a time)

**A) OOM on load / first decode:** (1) batch 128, (2) `--ctx-size 65536` + Pi 65536, (3) free desktop GPU apps / check WSL VRAM.

**B) `request (N) exceeds … (98304)`:** keep q8/q8 first if VRAM allows — `--ctx-size 131072` + Pi 131072. If that OOMs: same ctx, `--cache-type-v turbo4`, batch 128, **new** session + real `ls`/`read`. Further: 196608 turbo4 batch 128, then 262144 only if stable.

**C) Turn-1 garbage (fake paths, STAMP loops):** not fixed by more context. New Pi session; confirm PRIMARY is still q8/q8; no DRY / no client sampling override. A/B only V → turbo4; if garbage returns, stay q8 V.

**D) Max output token limit:** primary is already 16384. Confirm Pi restarted. Prefer writing `reports/*.md` and a short chat summary.

| | M5 MBP 48 GB | This RTX 4090 24 GB |
| --- | --- | --- |
| Weights | Q5 ~20 GB | Q4 ~17.6 GB |
| Tested large pin | **196k q8/q8** | **96k q8/q8** primary |
| Turbo V | Optional | **Only when raising past what q8 V fits** |

Same model family. Different **KV budget**.

## WSL2

- `nvidia-smi` **inside** WSL. One long-lived server; don’t share the GPU heavily.
- Workflows: [pi-coding-agent-graphs.md](../_Pi-Coding-Agent-Graphs/pi-coding-agent-graphs.md)

**Last Updated:** 2026-08-20 (recipe shape; WSL2 paths unchanged)
