from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LM Studio ---
    lm_studio_base_url: str = "http://localhost:1234/v1"
    lm_studio_model: str = "local-model"
    lm_studio_timeout_seconds: float = 30.0

    # --- Memory tier capacities ---
    stm_max_turns: int = 20
    wm_max_records: int = 50
    ltm_max_records: int = 10_000

    # --- Importance scoring weights (must sum to 1.0) ---
    score_weight_relevance: float = 0.45
    score_weight_recency: float = 0.35
    score_weight_keyword: float = 0.20
    recency_decay_rate: float = 0.15  # exponential decay per turn

    # --- Routing thresholds ---
    router_stm_threshold: float = 0.85
    router_wm_threshold: float = 0.50
    router_discard_threshold: float = 0.15

    # --- Chunking ---
    max_chunk_tokens_prose: int = 256
    max_chunk_tokens_code: int = 512

    # --- Assembly ---
    default_max_context_tokens: int = 8192
    token_count_padding_factor: float = 1.1
    assembly_stm_reserved_tokens: int = 1024  # always reserved for STM
    assembly_response_reserved_tokens: int = 1024  # reserved for LLM response

    # --- Storage ---
    data_dir: Path = Path("~/.cce/data").expanduser()

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8765

    # --- Compression queue ---
    compression_queue_maxsize: int = 500

    def model_post_init(self, __context: object) -> None:
        self.data_dir = Path(self.data_dir).expanduser()
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        p = self.data_dir / project_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def db_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "memory.db"

    def faiss_index_path(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "ltm.faiss"
