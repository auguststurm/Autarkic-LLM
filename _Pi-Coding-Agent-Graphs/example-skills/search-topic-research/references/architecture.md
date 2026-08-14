# Architecture

```text
Host Tavily
  → notes/<slug>/<YYYY-MM-DD>-research-pack.md     (pack-template)
  → workflow: Ingest → Findings → Skeptic → Report
  → reports/<slug>/<YYYY-MM-DD>-research-report.md (report-template)
```

`$ROOT` = `$HOME/.pi/agent/skills/search-topic-research`. Day (`YYYY-MM-DD`) in filename.

Workflow agents: coding tools only (no Tavily). Model = tiers **medium**.
