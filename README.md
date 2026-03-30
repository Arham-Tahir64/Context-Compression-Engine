# Context Compression Engine

Local FastAPI sidecar for long-running coding sessions. It stores project-scoped memory, reconstructs a briefing for the next LLM call, and keeps the HTTP boundary stable so you can feed it context from Codex, Claude Code, or any other tool.

## Minimum Local Setup

1. Copy `.env.example` to `.env`.
2. Set `DATA_DIR` to a writable local path. For repo-local development, `.cce-data` is the safest default.
3. Make sure the embedding model `all-MiniLM-L6-v2` is already available locally. The server now runs in offline-only mode for embeddings and will fail fast if the model is missing.
4. Start LM Studio if you want real WM/LTM summarization. If LM Studio is down, queued compression falls back to extractive truncation.
5. Start the server:

```bash
source .venv/bin/activate
./scripts/run_dev.sh
```

## Real Usage Flow

The engine does not crawl your repo by itself. `project_hint` only identifies the project. It remembers whatever your adapter sends in `recent_context`.

For a real project, feed it:
- the current user prompt
- the files you want remembered
- the current `git diff`
- notes, logs, or prior decisions

### Compress a prompt with project context

```bash
python -m cce.cli compress \
  --project /absolute/path/to/repo \
  --prompt "Continue fixing the auth middleware bug" \
  --file app/auth.py \
  --file tests/test_auth.py \
  --include-git-diff \
  --notes-file notes/session.txt
```

This prints the `optimized_prompt` by default. Add `--json` to inspect the full API response, including warnings and memory-hit counts.

### Recall project memory after a reset

```bash
python -m cce.cli recall \
  --project /absolute/path/to/repo \
  --query "current architecture decisions and recent changes"
```

This prints the briefing from `/recall`, which you can paste into the next session when your tool loses context.

## What The CLI Sends

The CLI helper turns each selected source into a deterministic `recent_context` entry:
- `[NOTES] ...`
- `[FILE] relative/path`
- `[GIT DIFF]`

That gives the engine durable project memory without requiring a deep tool-specific integration yet.
