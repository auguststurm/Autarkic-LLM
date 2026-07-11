# Glossary & Further Reading

Plain-language definitions for readers new to local inference (skip if you live in this stuff), plus curated links for going deeper.

## Glossary

- **GGUF**: the single-file model format llama.cpp loads. One `.gguf` per model+quant.
- **Quantization (`Q2`–`Q8`, `IQ*`)**: lower bits = smaller/faster, lower quality. `IQ` = "importance-aware" low-bit quants; `_K_S/_K_M/_K_XL` are size tiers within a level.
- **UD- (Unsloth Dynamic)**: per-layer mixed-precision quants that keep sensitive layers higher-precision for better quality at the same size. Prefer the `_XL` variants when they fit.
- **MoE vs dense**: a Mixture-of-Experts model (e.g. `35B-A3B` = 35B total, **3B active** per token) is fast like a small model but the _full_ weights must still fit in memory. Dense models activate all parameters.
- **E2B (Gemma)**: "edge/effective 2B", a small _dense_ model using Per-Layer Embeddings, not MoE.
- **KV cache**: stored attention keys/values for the context; grows with context length. Quantizing it (`--cache-type-k/v`, TurboQuant) saves memory. See the [TurboQuant deep dive](llama-cpp-turboquant.md#2-turboquant-kv-cache).
- **Flash attention (`--flash-attn`)**: a faster, lower-memory attention kernel.
- **Auto-fit (`--fit on --fit-target <MiB>`)**: llama-server detects free memory and auto-adjusts layer offload and the _effective_ context so the model loads. Can **silently crush context** (sometimes toward ~4096), which breaks coding agents. This repo’s hardware guides prefer **pinned `--ctx-size` + `--fit off`** and lower context or quant if you OOM. See the [M4 Air guide](M4-MacBook-Air-24GB/M4-MacBook-Air-Qwen3.6.md).
- **Context window**: max tokens (prompt + output) the model can attend to; set with `--ctx-size`. Match Pi’s `contextWindow` to the server’s real `n_ctx_seq`.
- **Thinking / reasoning (Qwen3.6)**: models can emit “thinking” instead of normal chat content. For agents, guides use `--reasoning off` and `--chat-template-kwargs '{"enable_thinking":false}'` so Pi gets `message.content`.

## Further reading

### Models & quantization

- [Unsloth Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs): what the `UD-` prefix means and how to pick a quant
- [GGUF format (Hugging Face docs)](https://huggingface.co/docs/hub/gguf): the on-disk model format
- [Gemma 4 overview (Google)](https://ai.google.dev/gemma/docs/core): E2B/E4B/26B-A4B/31B architecture

### Engine & inference

- [llama-cpp-turboquant deep dive](llama-cpp-turboquant.md): fork internals + full flag reference (in this repo)
- [llama-cpp-turboquant (upstream fork)](https://github.com/TheTom/llama-cpp-turboquant) · [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) · [TurboQuant design discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)

### Tooling

- [`hf` CLI / huggingface_hub](https://huggingface.co/docs/huggingface_hub/guides/cli): downloading GGUFs (see [`local-setup.md`](local-setup.md))
