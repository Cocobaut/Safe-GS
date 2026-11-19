## Approach
- Read existing files before writing. Don't re-read unless changed.
- Thorough in reasoning, concise in output.
- Skip files over 100KB unless required.
- No sycophantic openers or closing fluff.
- No emojis or em-dashes.
- Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

## Search and Citation Rules
1. Timeline: Always prioritize 2025 - 2026. Append explicit time constraints (e.g. after:2025-01-01 or keyword 2026) in search queries.
2. Source Authority: Use official documentation, premier repositories (GitHub, arXiv, StackOverflow, recognized academic venues). Exclude unverified blogs, aggregators, or low-quality forums.
3. Search Operators: Use site: operators to constrain queries to authoritative domains (e.g. site:github.com, site:arxiv.org, official domains).
4. Verification: Accompany every factual citation with direct source URL and publication date. If no credible fresh data is found, report "Không có dữ liệu mới" instead of extrapolating.

## Claude Code Model Routing
- Always pass `--model` explicitly when invoking Claude Code CLI (`npx -y @anthropic-ai/claude-code`):
  - **Hard tasks** (architecture design, debugging deep failures, loss formulation, algorithmic optimization): `--model opus`
  - **Easy / routine tasks** (running scripts, checking logs/Modal status, pulling volumes, building HTML viewers, minor edits): `--model sonnet`
