from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class FaissStore:
    """Per-project FAISS index for LTM vector search.

    Uses IndexFlatIP (inner product) on L2-normalised vectors, which
    is equivalent to cosine similarity. Vectors must be normalised
    before insertion — the embedding providers guarantee this.

    Atomic writes: saves to a .tmp file then os.replace() to avoid
    corruption on crash.
    """

    def __init__(self, index_path: Path, dimension: int):
        self._index_path = index_path
        self._dimension = dimension
        self._index = None  # loaded lazily or on connect()

    def load(self) -> None:
        import faiss
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        else:
            self._index = faiss.IndexFlatIP(self._dimension)

    def verify_integrity(self, expected_count: int) -> tuple[bool, str]:
        """Check FAISS ntotal matches SQLite record count.

        Returns (ok, message). Called at project init to catch corruption
        from a previous unclean shutdown.
        """
        if self._index is None:
            return False, "Index not loaded"
        actual = self._index.ntotal
        if actual != expected_count:
            return False, (
                f"FAISS index has {actual} vectors but SQLite has "
                f"{expected_count} ltm_records — index may be stale. "
                f"Run the re-index script to rebuild."
            )
        return True, "ok"

    def unload(self) -> None:
        self._index = None

    @property
    def ntotal(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def add(self, vectors: list[list[float]]) -> list[int]:
        """Add vectors and return their assigned integer IDs (sequential)."""
        import faiss
        start_id = self._index.ntotal
        arr = np.array(vectors, dtype=np.float32)
        self._index.add(arr)
        return list(range(start_id, start_id + len(vectors)))

    def search(self, query: list[float], top_k: int) -> list[tuple[int, float]]:
        """Return list of (faiss_id, score) sorted by score descending."""
        if self._index is None or self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        q = np.array([query], dtype=np.float32)
        scores, ids = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]

    def save(self) -> None:
        import faiss
        tmp = Path(str(self._index_path) + ".tmp")
        faiss.write_index(self._index, str(tmp))
        os.replace(tmp, self._index_path)  # atomic on POSIX
