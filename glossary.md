# Glossary & Further Reading

Plain-language definitions for readers new to local inference (skip if you live in this stuff), plus curated links for going deeper.

## Glossary

- **GGUF**: the single-file model format llama.cpp loads. One `.gguf` per model+quant.
- **Quantization (`Q2`–`Q8`, `IQ*`)**: lower bits = smaller/faster, lower quality. `IQ` = "importance-aware" low-bit quants; `_K_S/_K_M/_K_XL` are size tiers within a level. Full naming decoder + Q4–Q8 ladder: [local-setup — Understanding GGUF quants](local-setup.md#understanding-gguf-quants-why-so-many-files).
- **UD- (Unsloth Dynamic)**: per-layer mixed-precision quants that keep sensitive layers higher-precision for better quality at the same size. Prefer the `_XL` variants when they fit.
- **MoE vs dense**: a Mixture-of-Experts model (e.g. `35B-A3B` = 35B total, **3B active** per token) is fast like a small model but the _full_ weights must still fit in memory. Dense models activate all parameters.
- **E2B (Gemma)**: "edge/effective 2B", a small _dense_ model using Per-Layer Embeddings, not MoE.
- **KV cache**: stored attention keys/values for the context; grows with context length. Quantizing it (`--cache-type-k/v`, TurboQuant) saves memory. See the [TurboQuant deep dive](llama-cpp-turboquant.md#2-turboquant-kv-cache).
- **Flash attention (`--flash-attn`)**: a faster, lower-memory attention kernel.
- **Auto-fit (`--fit on --fit-target <MiB>`)**: llama-server detects free memory and auto-adjusts layer offload and the _effective_ context so the model loads. Can **silently crush context** (sometimes toward ~4096), which breaks coding agents. This repo’s hardware guides prefer **pinned `--ctx-size` + `--fit off`** and lower context or quant if you OOM. See the [M4 Air guide](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).
- **Context window**: max tokens (prompt + output) the model can attend to; set with `--ctx-size`. Match Pi’s `contextWindow` to the server’s real `n_ctx_seq`.
- **Thinking / reasoning (Qwen3.6 / Qwen3.8)**: hybrid models can emit “thinking” instead of normal chat content. For agents, guides prefer **`--reasoning off`** (+ budget 0) so Pi gets `message.content`. Avoid relying only on deprecated `--chat-template-kwargs '{"enable_thinking":false}'` on current llama-server builds.
- **Thinking / reasoning (Muse Glimmer)**: the template **always** opens a private `to=self` channel. **`--reasoning off` does nothing.** Control depth with `reasoning_strength` (`low` / `medium` / `high` / `xhigh`). `llama-server` puts traces in `reasoning_content` so `content` stays clean. See [agentic harnesses — Muse](agentic-harnesses.md#muse-glimmer-30b--pi-coding-agent).
- **DFlash**: Muse Glimmer’s speculative-decode companion (~1.6 GB GGUF). Block-diffusion draft (`--spec-type draft-dflash`), not Qwen MTP.
- **Qwen3.8 (released 2026-08-14)**: dense ~27B vision-language model (GGUF arch `qwen35`), native 262k context. **✅ Field-tested** on Dual RTX 6000 with Pi ([guide](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md)); DGX Spark / M5 Pro ports still untested — see [README — Qwen3.8](README.md#qwen38-2026-08-14). GGUF names + Q8→Q4 ladder: [local-setup](local-setup.md#understanding-gguf-quants-why-so-many-files). MTP / thinking / vision: [Dual RTX Qwen3.8 optionals](Dual-RTX6000-192GB/Dual-RTX6000-Qwen3.8.md#qwen38-optionals).
- **Muse Glimmer 30B (released August 2026)**: Meta Superintelligence Labs dense ~30B VLM (GGUF arch `muse-glimmer`, Apache 2.0), native 131k context. Dual RTX guide ⚠️ untested: [Dual-RTX6000-Muse-Glimmer.md](Dual-RTX6000-192GB/Dual-RTX6000-Muse-Glimmer.md). Needs llama.cpp `b10353+`.

## Further reading

### Models & quantization

- [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs): what the `UD-` prefix means and how to pick a quant
- [Muse Glimmer — How to Run Locally (Unsloth)](https://unsloth.ai/docs/models/muse-glimmer): sampling, hardware table, llama.cpp
- [Meta — Deploy Muse Glimmer with llama.cpp](https://dev.meta.ai/docs/muse-glimmer/llama-cpp/): `b10353+`, DFlash, reasoning_strength, ATEM tools
- [unsloth/Muse-Glimmer-30B-GGUF](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF): GGUF catalog (confirm filenames before download)
- [Qwen3.8 — How to Run Locally (Unsloth)](https://unsloth.ai/docs/models/qwen3.8): day-zero run notes, sampling, hardware table
- [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF): GGUF catalog (confirm filenames before download)
- [GGUF format (Hugging Face docs)](https://huggingface.co/docs/hub/gguf): the on-disk model format
- [Gemma 4 overview (Google)](https://ai.google.dev/gemma/docs/core): E2B/E4B/26B-A4B/31B architecture

### Engine & inference

- [llama-cpp-turboquant deep dive](llama-cpp-turboquant.md): fork internals + full flag reference (in this repo)
- [llama-cpp-turboquant (upstream fork)](https://github.com/TheTom/llama-cpp-turboquant) · [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) · [TurboQuant design discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)

### Tooling

- [`hf` CLI / huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/cli): downloading GGUFs (see [`local-setup.md`](local-setup.md))
