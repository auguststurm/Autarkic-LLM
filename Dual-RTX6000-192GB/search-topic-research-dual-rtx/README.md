# Search Topic Research (Dual RTX)

Same Tavily research as the [generic skill](../../_Pi-Coding-Agent-Graphs/example-skills/search-topic-research/), built for this machine’s two cards. One folder, one install.

**Need:** Localmaxing pack running, `small` → `:8080`, `medium`/`big` → `:8081` ([guide](Dual-RTX6000-Qwen3.8-localmaxing.md)). Tavily key in the Pi process env.

```bash
mkdir -p ~/.pi/agent/skills
cp -a Dual-RTX6000-192GB/search-topic-research-dual-rtx \
      ~/.pi/agent/skills/search-topic-research-dual-rtx
```

`/reload`, then:

```text
/skill:search-topic-research-dual-rtx EU AI Act enforcement timeline 2025-2026
```

| Phase | Cards |
| --- | --- |
| Host Tavily (pack) | Your `/model` — one GPU |
| Findings | GPU 0 + GPU 1 at the same time |
| Skeptic | GPU 0 + GPU 1 at the same time |
| Report | `big` — one GPU |

One-GPU machines: use the [generic skill](../../_Pi-Coding-Agent-Graphs/example-skills/search-topic-research/), not this folder.
