# Product Scope — Context Compression Engine

## What it is
A fully local sidecar service that acts as middleware between the developer and an LLM (Claude, Codex). It intercepts outgoing context, manages a 3-tier memory store, and returns an optimized prompt — all without blocking the LLM call in the critical path.

## Primary Use Case
Prevent long-running coding agents (Cursor, Claude Code, Codex) from degrading as their context grows across multi-hour or multi-day sessions. The agent should behave as if it remembers everything even after most raw context is removed.

## Target User
Developers building or using agent-based workflows who hit context limits and see quality degrade. Not a non-technical end-user product.

## Architecture
- Local sidecar service (process runs alongside the dev's tooling)
- Intercepts context before it reaches the LLM
- Returns a reconstructed, compressed prompt
- Project-based local persistence (sessions survive process restarts across days)

## LLM Integration
- Claude and Codex at launch
- Clean abstraction layer to add more models later
- Compression/summarization done via a **locally hosted model (LM Studio)** — no external API calls for memory or compression

## Scale / "Long Context" Definition
- Multi-hour to multi-day coding sessions
- Tens to hundreds of thousands of tokens
- Content types: iterative code edits, logs, reasoning traces, prior decisions

## Latency Budget
- End-to-end overhead: <200–500ms (amortized)
- Retrieval: 10–50ms
- Compression: async / precomputed — must NOT block LLM calls in critical path

## Data Constraints
- Fully local. No external API calls required for memory or compression.
- Privacy-first: suitable for private codebases

## V1 Scope — IN
- Chunking + metadata
- FAISS-based embedding retrieval
- Heuristic importance scoring (embedding similarity, recency, entity/goal overlap)
- 3-tier memory: STM (verbatim recent), WM (lightly compressed, high-score), LTM (heavily compressed + embeddings)
- Basic summarization via local LLM (LM Studio)
- Prompt reconstruction (assembles optimized context for outgoing call)

## V1 Scope — OUT (explicitly excluded)
- Learned compression models (no preference optimization, no fine-tuning)
- Reinforcement-based or usage-history scoring
- Multimodal memory
- Meta-memory controllers
- Multi-agent debate schemes
- Complex structured extraction pipelines

## Success Criteria
**Quality**: Outputs match or exceed truncation baseline; semantic similarity to full-context answers; code correctness preserved
**Efficiency**: 3–10x token compression ratio; <500ms latency overhead
**Stability**: No degradation over multi-day sessions

## Technical Decisions (locked)

### Interception Mechanism
Local HTTP API (primary). Thin client adapters per tool (Cursor, Claude Code). stdin/stdout deferred to post-v1. API accepts: current prompt + recent context + project identity + metadata. Returns: optimized prompt package.

### Embedding Model
`all-MiniLM-L6-v2` via Python (sentence-transformers). Fast, lightweight, no LM Studio dependency. Embedding provider abstracted behind an interface so `nomic-embed-text` can be swapped in later.

### Project Identity
Keyed by **Git repository root** (auto-detected). Config file or explicit session ID can override. Non-git projects fall back to a named local workspace.

## North Star
> "You can remove most of the raw context and the agent still behaves as if it remembers everything."

The agent correctly recalls prior decisions, constraints, and code structure after hours or days without needing the full context window.
