# Implementation Plan — Context Compression Engine

## Project Structure

```
context-compression-engine/
├── pyproject.toml
├── requirements.txt
├── .env.example
├── docs/
│   ├── PRODUCT_SCOPE.md
│   └── IMPLEMENTATION_PLAN.md
│
├── cce/
│   ├── main.py                     # FastAPI app + lifespan
│   ├── settings.py                 # Pydantic Settings (env vars + defaults)
│   ├── dependencies.py             # DI wiring
│   │
│   ├── api/
│   │   ├── routes.py               # /compress, /recall, /health, /project/*
│   │   └── schemas.py              # Pydantic request/response models
│   │
│   ├── pipeline/
│   │   ├── chunker.py              # Text → List[Chunk]
│   │   ├── scorer.py               # Heuristic importance scoring
│   │   ├── router.py               # Routes chunks → STM / WM / LTM
│   │   └── assembler.py            # Builds optimized PromptPackage
│   │
│   ├── memory/
│   │   ├── base.py                 # MemoryStore ABC
│   │   ├── stm.py                  # In-process deque ring buffer
│   │   ├── wm.py                   # Top-K heap + SQLite
│   │   └── ltm.py                  # FAISS + SQLite + LM Studio summaries
│   │
│   ├── compression/
│   │   ├── queue.py                # asyncio.Queue wrapper
│   │   └── compressor.py           # LM Studio HTTP calls (httpx.AsyncClient)
│   │
│   ├── embeddings/
│   │   ├── base.py                 # EmbeddingProvider ABC
│   │   ├── sentence_transformer.py # Default: all-MiniLM-L6-v2
│   │   └── lm_studio.py            # Swap-in: nomic-embed-text
│   │
│   ├── identity/
│   │   └── resolver.py             # git root → config → named workspace
│   │
│   ├── storage/
│   │   ├── db.py                   # aiosqlite connection pool
│   │   ├── schema.sql              # DDL
│   │   └── faiss_store.py          # Per-project index load/save/query
│   │
│   └── models/
│       └── types.py                # All core dataclasses
│
├── tests/
│   ├── unit/
│   └── integration/
│
└── scripts/
    ├── run_dev.sh
    └── bench_latency.py
```

---

## Build Order (Critical Path)

```
1. models/types.py          ← everything depends on data shapes         [DONE]
2. settings.py              ← all modules need config                   [DONE]
3. embeddings/base.py + sentence_transformer.py                         [DONE]
4. storage/db.py + schema.sql + faiss_store.py                         [DONE]
5. pipeline/chunker.py                                                  [PHASE 1]
6. pipeline/scorer.py                                                   [PHASE 1]
7. memory/stm + wm + ltm                                               [PHASE 2]
8. pipeline/router.py                                                   [PHASE 2]
9. compression/queue + compressor      ← async, off critical path      [PHASE 3]
10. pipeline/assembler.py                                               [PHASE 4]
11. api/schemas + routes + main.py     ← wire everything               [PHASE 4]
12. identity/resolver.py                                                [PHASE 5]
```

---

## Pipeline Overview

```
Incoming Context
      ↓
[1] Chunker          — splits turns/edits/logs into 200–500 token units + metadata
      ↓
[2] Importance Scorer — heuristic: embedding similarity + recency + entity overlap
      ↓
[3] Memory Router     — STM (verbatim) / WM (lightly compressed) / LTM (FAISS + summaries)
      ↓
[4] Async Compressor  — local LLM via LM Studio, runs out-of-band, not on critical path
      ↓
[5] Prompt Assembler  — retrieves top-K from WM+LTM, mixes with STM, fits token budget
      ↓
Optimized Prompt → LLM (Claude / Codex)
```

---

## API Contract

### `POST /compress`
Called before every LLM request.

**Request:**
```json
{
  "project_hint": "/path/to/repo",
  "current_message": "user's current prompt",
  "recent_context": [{ "role": "user", "content": "...", "turn_index": 0 }],
  "metadata": { "tool": "cursor", "max_context_tokens": 8192 }
}
```

**Response:**
```json
{
  "optimized_prompt": "...",
  "original_token_estimate": 4200,
  "compressed_token_estimate": 1100,
  "compression_ratio": 3.8,
  "memory_hits": { "stm": 3, "wm": 5, "ltm": 2 },
  "latency_ms": { "total": 87, "retrieval": 12, "assembly": 8 }
}
```

### `POST /recall`
Called on context reset events.

**Request:**
```json
{ "project_hint": "/path/to/repo", "query": "optional focus", "max_tokens": 2048 }
```

**Response:**
```json
{ "briefing": "...", "token_estimate": 850, "memory_sources": { "stm_records": 0, "wm_records": 8, "ltm_records": 4 } }
```

### Other endpoints
- `GET /health`
- `GET /project/{project_id}/stats`
- `DELETE /project/{project_id}/memory`

---

## Importance Scoring Formula

```
composite = 0.45 × cosine_similarity(chunk, query)
          + 0.35 × recency_decay(turns_ago)        # 1 / (1 + decay_rate * turns_ago)
          + 0.20 × keyword_jaccard(chunk, query)
```

Weights exposed in `Settings`. Default decay rate: 0.15.

## Routing Thresholds

| Score | Destination |
|---|---|
| ≥ 0.85 | STM (verbatim) |
| 0.50 – 0.85 | WM (light compression queued) |
| 0.15 – 0.50 | LTM (heavy compression queued) |
| < 0.15 | DISCARD |

---

## Memory Tiers

| Tier | Storage | Capacity | Compression | Persistence |
|---|---|---|---|---|
| STM | In-process deque | 20 turns | None (verbatim) | Lost on restart |
| WM | SQLite | 50 records | Light (1–3 sentence) | Survives restart |
| LTM | FAISS + SQLite | 10,000 records | Heavy (key facts only) | Survives restart |

---

## Key Risks

| Risk | Mitigation |
|---|---|
| LM Studio latency | Compression is always async — `/recall` returns existing summaries only |
| FAISS corruption on crash | Atomic write via `os.replace()` on temp file |
| Token count miscalculation | Configurable padding factor; swap in `tiktoken` later |
| Embedding dimension mismatch on provider swap | Store model + dimension in SQLite; raise clear error with re-index instruction |
| Code blocks split mid-function | Structural chunking treats unclosed fences as extending to EOF |

---

## Phase Status

| Phase | What | Status |
|---|---|---|
| 0 | Foundation: models, settings, embeddings, storage, FastAPI skeleton | ✅ Done |
| 1 | Chunker + Scorer | ✅ Done |
| 2 | Memory tiers (STM, WM, LTM) + Router + Identity resolver | ✅ Done |
| 3 | Async compression (queue + LM Studio compressor) | ✅ Done |
| 4 | Prompt assembler + wire API endpoints | ✅ Done |
| 5 | FAISS integrity check, error handling, .gitignore, bench script | ✅ Done |
| 6 | Integration testing + latency benchmarks | 🔄 Next |
