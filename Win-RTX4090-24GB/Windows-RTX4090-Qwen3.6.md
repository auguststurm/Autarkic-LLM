# Windows RTX 4090 (24 GB) - Qwen3.6-27B

**WSL2 · CUDA sm_89 · llama-cpp-turboquant · Pi / pi-workflows**

Paths: `~/AIML`, `~/GitHub`.

> ✅ **Tested** on this hardware; **field-validated working** with Pi Coding Agent / multi-agent research (stable tools + report-length output). Baseline from **first principles** (weights vs KV growth, asymmetric K/V, hybrid Qwen + Pi). Cross-hardware Pi notes: [agentic harnesses](../agentic-harnesses.md#qwen36-27b--pi-coding-agent-cross-hardware).

---

## First principles (why this baseline)

### 1. Memory has two fixed budgets

| Piece | Scales with | On this box |
| --- | --- | --- |
| **Weights** | Quant only | Q4_K_XL ≈ **17.6 GB** (fixed) |
| **KV cache** | **ctx × K/V precision × layers** | Grows with every token of window you **pin** |
| **Compute scratch** | batch / ubatch | Peaks on prefill |
| **Hard cap** | — | **24 GB VRAM** (+ Windows/WSL slice) |

Field: total VRAM often sits **~19–21 GB** at modest pins → **room exists**, but not M5-class (48 GB unified).

### 2. K and V are not equal

| Cache | Job | Quant rule |
| --- | --- | --- |
| **K (keys)** | Routing — *which* past tokens matter | **Keep high:** `q8_0` (or f16). Do **not** turbo K first. |
| **V (values)** | Payload — *what* you read | Compressible: `q8_0` → `turbo4` → `turbo3` → `turbo2` only as capacity needs |

Wrong K → wrong attention → fake paths, wrong country, planning loops.  
Aggressive V → more context on 24 GB; can hurt quality if pushed too hard too early.

### 3. Token limits are two different knobs

| Knob | Limits | Field failures |
| --- | --- | --- |
| **`--ctx-size` / Pi `contextWindow`** | **Input** (history + tools + Tavily + skills) | `request (70743) exceeds … (65536)` |
| **`--n-predict` / Pi `maxTokens`** | **One reply’s length** | `maximum output token limit` |

Raising one does not fix the other.

### 4. Hybrid Qwen + Pi constraints

- **Thinking off:** `--reasoning off` (+ budget 0). Prefer this over deprecated `enable_thinking` kwargs.
- **No DRY** — path/name corruption on Qwen3.6 tools ([llama.cpp #20837](https://github.com/ggml-org/llama.cpp/issues/20837)).
- **Tool sampling:** `temp 0.6`, `top_p 0.95`, `top_k 20`, **`presence_penalty 0`**, `repeat_penalty 1.0`.
- **`--cache-ram 0`** — hybrid DeltaNet multi-turn cache restore issues ([#21681](https://github.com/ggml-org/llama.cpp/issues/21681)).
- **New session** after any garbage thread (compaction ≠ clean KV/history).
- TurboQuant **fork** ≠ must use turbo **types**. M5 primary is **q8/q8** at huge ctx on 48 GB. On 24 GB, turbo **V** is the capacity lever when q8 V no longer fits.

### 5. Baseline sizing (this hardware + this workload)

| Requirement | Number |
| --- | --- |
| Prompt overflows seen | **>32k**, **>64k**, **~70k** single request |
| Minimum useful pin | **≥ 98304 (96k)** with margin |
| Report output | Often **>4k–8k** tokens → **16384** out |
| Stability | **K=q8_0, V=q8_0** until OOM forces V turbo |
| Capacity if OOM or still short | Same K, **V=turbo4**, raise ctx |

**Primary = 96k · q8/q8 · 16k out.**  
Covers real ~70k prompts, prioritizes stable K/V, uses free VRAM for window without jumping to turbo-first.

---

## PRIMARY command (baseline — use this)

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
  --no-mmap \
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

| Flag group | Setting | Principle |
| --- | --- | --- |
| Model | Q4_K_XL, `ngl 99`, `main-gpu 0` | Full offload; weights leave ~6 GB-class room for KV |
| Context | **98304**, `--fit off` | ≥70k prompts + margin; pin stays agent-visible |
| KV | **q8_0 / q8_0** | Stable routing (K) + payload (V); no turbo until needed |
| Hybrid | `--cache-ram 0`, no checkpoints | Correct multi-turn over cache speed |
| Attention | `--flash-attn on` | Required for quantized KV |
| Agent | reasoning off, no DRY, presence 0 | Pi tools / paths |
| Sampling | 0.6 / 0.95 / top_k 20 | Qwen tool/coding profile |
| Output | **n-predict 16384** | Full reports |
| Batch | 256 | Prefill vs peak; drop to 128 on OOM |

**Confirm**

```text
log: n_ctx_seq (98304)
nvidia-smi   # note MiB after load and after a short decode
```

**Pi** — write entire file, restart Pi, **new session**:

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

```bash
mkdir -p ~/.pi/agent
# → ~/.pi/agent/models.json
```

Status bar must show **~96k**. Server and Pi numbers must match.

---

## Only if primary fails (ordered — change one axis)

### A) OOM on load / first decode

1. `--ubatch-size 128 --batch-size 128`  
2. Still OOM → `--ctx-size 65536` (Pi `contextWindow` **65536**) — quality held, window smaller  
3. Still OOM → free desktop GPU apps / check WSL VRAM

### B) `request (N) exceeds … (98304)` (real overflow)

Keep **q8/q8** first if VRAM allows:

```bash
# PRIMARY + only:
#   --ctx-size 131072
# Pi contextWindow: 131072
```

If that **OOMs**, then use TurboQuant **V** (K stays q8):

```bash
#   --ctx-size 131072
#   --cache-type-k q8_0 --cache-type-v turbo4
#   --ubatch-size 128 --batch-size 128
# Pi: 131072
# Smoke-test: new session, real ls/read, before multi-agent runs
```

Further: **196608** with turbo4 (batch 128), then **262144** only if stable — same K/V rules.

### C) Turn-1 garbage (fake paths, STAMP loops, topic drift)

**Not fixed by more context.**

1. **New Pi session** (mandatory).  
2. Confirm PRIMARY is **q8/q8** (not leftover turbo4 process).  
3. Confirm no DRY / no client sampling override.  
4. A/B: only flip V to turbo4 — if garbage returns, stay q8 V.

### D) Max output token limit

Primary already **16384**. Confirm Pi restarted with matching `maxTokens`. Prefer agent **writes `reports/*.md`** and short chat summary for huge reports.

---

## Model download

```bash
hf download unsloth/Qwen3.6-27B-GGUF \
  Qwen3.6-27B-UD-Q4_K_XL.gguf \
  --local-dir ~/AIML/models
```

## Build (Ada)

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
./llama-server --help | grep -A2 cache-type-v   # turbo* listed if you need capacity profile
```

---

## Startup warnings (normal)

| Log | Meaning |
| --- | --- |
| `n_ctx_seq (98304) < n_ctx_train (262144)` | Pin &lt; train max — expected on 24 GB |
| `cache-idle-slots requires --cache-ram, disabling` | Follows `--cache-ram 0` |
| `enable_thinking` via chat-template-kwargs deprecated | Do not pass kwargs; use `--reasoning off` |

---

## M5 48 GB vs this box

| | M5 MBP 48 GB | RTX 4090 24 GB |
| --- | --- | --- |
| Weights | Q5 ~20 GB | Q4 ~17.6 GB |
| Tested large pin | **196k q8/q8** | **96k q8/q8** primary |
| Why different | ~2× memory pool | Discrete 24 GB hard cap |
| Turbo V | Optional | **Only when raising past what q8 V fits** |

Same model family. Different **KV budget**.

---

## WSL2

- `nvidia-smi` inside WSL.  
- `--no-mmap` needs free host RAM (~18 GB+) for load.  
- One long-lived server; don’t share the GPU heavily.  
- Workflows: [pi-coding-agent-graphs.md](../pi-coding-agent-graphs.md) · flags: [llama-cpp-turboquant.md](../llama-cpp-turboquant.md)

**Last Updated:** August 2026
