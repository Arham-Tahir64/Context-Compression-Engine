-- Project registry: one row per project, tracks embedding model for dimension safety.
CREATE TABLE IF NOT EXISTS projects (
    project_id      TEXT PRIMARY KEY,
    display_name    TEXT NOT NULL,
    git_root        TEXT,
    embedding_model TEXT NOT NULL,
    embedding_dim   INTEGER NOT NULL,
    created_at      REAL NOT NULL,
    last_accessed_at REAL NOT NULL
);

-- Working Memory: top-scored chunks, lightly compressed, persisted across restarts.
CREATE TABLE IF NOT EXISTS wm_records (
    record_id               TEXT PRIMARY KEY,
    project_id              TEXT NOT NULL REFERENCES projects(project_id),
    content                 TEXT NOT NULL,
    original_token_count    INTEGER NOT NULL,
    compressed_token_count  INTEGER NOT NULL,
    source_chunk_ids        TEXT NOT NULL,   -- JSON array of chunk_ids
    embedding               BLOB NOT NULL,   -- numpy float32 bytes
    importance_score        REAL NOT NULL,
    created_at              REAL NOT NULL,
    last_accessed_at        REAL NOT NULL,
    metadata                TEXT NOT NULL DEFAULT '{}'  -- JSON
);

CREATE INDEX IF NOT EXISTS idx_wm_project ON wm_records(project_id);
CREATE INDEX IF NOT EXISTS idx_wm_score   ON wm_records(project_id, importance_score DESC);

-- Long-Term Memory: heavily compressed summaries, associated with FAISS integer IDs.
CREATE TABLE IF NOT EXISTS ltm_records (
    record_id               TEXT PRIMARY KEY,
    project_id              TEXT NOT NULL REFERENCES projects(project_id),
    faiss_id                INTEGER NOT NULL,  -- index in the per-project FAISS index
    content                 TEXT NOT NULL,
    original_token_count    INTEGER NOT NULL,
    compressed_token_count  INTEGER NOT NULL,
    source_chunk_ids        TEXT NOT NULL,   -- JSON array
    importance_score        REAL NOT NULL,
    created_at              REAL NOT NULL,
    last_accessed_at        REAL NOT NULL,
    metadata                TEXT NOT NULL DEFAULT '{}'  -- JSON
);

CREATE INDEX IF NOT EXISTS idx_ltm_project  ON ltm_records(project_id);
CREATE INDEX IF NOT EXISTS idx_ltm_faiss_id ON ltm_records(project_id, faiss_id);

-- Raw chunk log: original chunks before compression, used for re-indexing.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,
    project_id      TEXT NOT NULL REFERENCES projects(project_id),
    content         TEXT NOT NULL,
    content_type    TEXT NOT NULL,
    token_count     INTEGER NOT NULL,
    turn_index      INTEGER NOT NULL,
    created_at      REAL NOT NULL,
    tier_assigned   TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project_id, created_at DESC);
