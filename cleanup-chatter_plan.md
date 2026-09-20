# Hardware-guide chatter cleanup

Gold standard: [`DGX-Spark-128GB/DGX-Spark-LFM2.5-2.6B.md`](DGX-Spark-128GB/DGX-Spark-LFM2.5-2.6B.md) and [`DGX-Spark-128GB/DGX-Spark-Qwen3.8.md`](DGX-Spark-128GB/DGX-Spark-Qwen3.8.md).

Target shape: pin table → Download → Build → PRIMARY command → short **this-box** why table → Confirm → Pi JSON → short This box → See also.

## Rules

**Keep** (do not drop or invent):

- This box’s pins, cmake, PRIMARY command, paths, sampling, Pi JSON.
- This-box measurements and unique failure modes (Jetson `free -h`, 3090 `ptxas` / first-time WSL, Dual RTX tested dates).
- Canonical long-form on the **tested source** page (Jetson LFM truncation / context budget; Dual RTX Qwen3.8 optionals; Dual RTX Bonsai optionals). Other pages **link**, they do not copy.
- One-line “do not paste X cmake / binary” when that mix-up is a real trap.
- Sibling links in **See also**.

**Cut** (cross-guide chatter, not pins):

- Cross-SKU comparison tables.
- Repeated “do not copy the other guide” essays in the body.
- Port write-ups, cmake “why not these flags” tables about **other** hardware.
- Full Pi truncation / context-budget lectures copied onto ports.
- A/B/C/D fallback trees; fill-in worksheets; glossary tables inside a hardware guide.
- Performance notes that only restate the pin table.
- Optionals duplicated on every Bonsai / Qwen3.8 port.

Essays stay in `local-setup.md`, `llama-cpp-turboquant.md`, `agentic-harnesses.md`.

README / `local-setup.md` / `agentic-harnesses.md` / skill folders are **out of scope**.

## Tracker

| # | Guide | Before | After | Status |
| --- | --- | ---: | ---: | --- |
| 1 | `DGX-Spark-128GB/DGX-Spark-LFM2.5-2.6B.md` | 194 | 194 | **Done** (gold; left as-is) |
| 2 | `DGX-Spark-128GB/DGX-Spark-Qwen3.8.md` | 148 | 148 | **Done** (gold twin; date only) |
| 3 | `Win-RTX3090-24GB/Windows-RTX3090-LFM2.5-2.6B.md` | 429 | 299 | **Done** — kept First-time WSL + Ampere FA=OFF cmake; cut comparison table, truncation copy, A/B/C/D |
| 4 | `Jetson-Orin-Nano-Super/Jetson-Orin-LFM2.5-2.6B.md` | 287 | 286 | **Done** — canonical LFM kept (budget, truncation, 128k stretch, Q6_K); cut Spark/3090 cmake banner |
| 5 | `Jetson-Orin-Nano-Super/Jetson-Orin-MiniCPM5-2B.md` | 465 | 234 | **Done** — cut glossary, A–E worksheets, LFM-vs-MiniCPM tables; kept OpenBMB pins + field-report numbers |
| 6 | `DGX-Spark-128GB/DGX-Spark-Bonsai-2-27B.md` | 262 | 222 | **Done** — cut Qwen-vs-Bonsai table; optionals → Dual RTX |
| 7 | `Dual-RTX6000-192GB/Dual-RTX6000-Bonsai-2-27B.md` | 305 | 300 | **Done** — canonical Bonsai optionals stay; trimmed twin list |
| 8 | `Win-RTX3090-24GB/Windows-RTX3090-Bonsai-2-27B.md` | 309 | 245 | **Done** — kept WSL/`"86"`/131k; cut 4090 table + duplicated optionals |
| 9 | `AMD-7900-XTX/7900-XTX-Bonsai-2-27B.md` | 296 | 256 | **Done** — kept HIP + Vulkan-slow alternate; cut comparison table |
| 10 | `M1-Ultra-Studio-64GB/M1-Ultra-Studio-Bonsai-2-27B.md` | 278 | 234 | **Done** — kept Metal 262k + MLX one-liner; cut CUDA comparison |
| 11 | `Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.6.md` | 218 | 211 | **Done** — kept layer-split alternate; cut performance-note restates |
| 12 | `Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md` | 224 | 224 | **Done** — canonical 3.8 optionals stay; light intro trim |
| 13 | `Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8-localmaxing.md` | 630 | 612 | **Done** — kept two-process commands; cut repo-lecture + model catalog |
| 14 | `Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md` | 219 | 217 | **Done** — kept Muse pins + DFlash |
| 15 | `Win-RTX4090-24GB/Windows-RTX4090-Qwen3.6.md` | 160 | 148 | **Done** — cut M5 table + A/B/C/D tree |
| 16 | `DGX-Spark-128GB/DGX-Spark-Qwen3.6.md` | 158 | 158 | **Done** — kept Q8 alternate; trimmed performance notes |
| 17 | `M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.6.md` | 209 | 202 | **Done** — kept 196k Metal + MoE alternate |
| 18 | `M5-MacBook-Pro-48GB/M5-MacBook-Pro-Qwen3.8.md` | 147 | 146 | **Done** |
| 19 | `M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md` | 186 | 178 | **Done** — kept measured decode matrix |
| 20 | `M4-Mac-Mini-16GB/M4-Mac-Mini-Qwen3.6.md` | 214 | 183 | **Done** — cut turbo-tier essay + report worksheet |
| 21 | `M2-Mac-Mini-16GB/M2-Mac-Mini-Qwen3.6.md` | 231 | 183 | **Done** — kept wired-limit + M2 bandwidth; cut TBD table |
| 22 | `M4-Mac-Mini-16GB/M4-Mac-Mini-Gemma-4-E2B.md` | 154 | 148 | **Done** |
| 23 | `M2-Mac-Mini-16GB/M2-Mac-Mini-Gemma-4-E2B.md` | 167 | 148 | **Done** — cut TBD measured table |
| 24 | `Jetson-Orin-Nano-Super/Jetson-Orin-Gemma4-E2B.md` | 140 | 139 | **Done** |
| 25 | `AMD-7900-XTX/7900-XTX-Qwen3.6-27b.md` | 124 | 122 | **Done** |
| 26 | `AMD-7900-XTX/7900-XTX-Qwen3.6-35b-a3b.md` | 122 | 121 | **Done** |

Hardware-guide total: **6276 → 5558** lines.

## Canonical pages (do not flatten)

- Jetson LFM: truncation + measured 64k budget + 128k stretch.
- Dual RTX Qwen3.8: MTP / sampling / thinking / vision optionals.
- Dual RTX Bonsai 2: sampling / thinking / vision optionals.
- Dual RTX Localmaxing: two `llama-server` commands (one per card).
