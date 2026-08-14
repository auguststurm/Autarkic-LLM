# Tavily knobs (host only)

Tools: `web_search`, `web_fetch`. snake_case. Local runs: prefer **basic** + small `max_results`; fetch 1–3 primaries.

## web_search

| Param | Use |
|-------|-----|
| `search_depth: "basic"` | Default. 1 credit. |
| `search_depth: "advanced"` | High-stakes; multi-chunk. 2 credits. Enables `chunks_per_source` 1–3. |
| `search_depth: "fast"` / `"ultra-fast"` | Low-latency scan. 1 credit. |
| `topic: "general"` | Default; **required** for `country`. |
| `topic: "news"` | Breaking / current events. |
| `topic: "finance"` | Markets, filings, rates. |
| `time_range` | Optional: `day`/`week`/`month`/`year` (or `d`/`w`/`m`/`y`). **Omit** if historical/evergreen. |
| `start_date` / `end_date` | `YYYY-MM-DD` when enums too coarse. |
| `max_results` | 5–8 coverage (local); max 20. |
| `include_answer` | `true`/`basic`/`advanced` — orient only; verify URLs. |
| `include_domains` / `exclude_domains` | Trust list / drop farms (max 300 / 150). |
| `country` | Enum lowercase only (`"japan"`, `"united states"`, …) + `topic: "general"`. Not ISO codes. |
| `exact_match` | Quoted phrases in `query`. |
| `auto_parameters` | Optional; may upgrade to advanced (cost). |
| `include_raw_content` | Rare — prefer `web_fetch`. |
| `include_images` / `include_favicon` / `include_usage` | Usually off. |

## web_fetch

| Param | Use |
|-------|-----|
| `urls` | String or array ≤20. **Not** stringified JSON. |
| `extract_depth` | `basic` default; `advanced` for long docs. |
| `format` | `markdown` default; `text` if markdown noisy. |
| `query` | Focus extract; required for `chunks_per_source` (1–5). |
| `timeout` | Seconds 1–60 for slow sites. |

## Constraints

- Search `chunks_per_source` → needs `search_depth: "advanced"`.
- `country` → `topic: "general"` + valid enum.
- Fetch `chunks_per_source` → needs `query`.
- Pattern: search discover → fetch only pages that change the report.
