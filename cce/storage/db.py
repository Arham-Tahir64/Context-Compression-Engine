from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Thin async SQLite wrapper. One instance per project."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._apply_schema()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def _apply_schema(self) -> None:
        schema = _SCHEMA_PATH.read_text()
        await self._conn.executescript(schema)
        await self._conn.commit()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Context manager that commits on exit or rolls back on exception."""
        try:
            yield self._conn
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    async def execute(self, sql: str, params: tuple = ()) -> aiosqlite.Cursor:
        return await self._conn.execute(sql, params)

    async def executemany(self, sql: str, params_seq: list[tuple]) -> None:
        await self._conn.executemany(sql, params_seq)
        await self._conn.commit()

    async def fetchall(self, sql: str, params: tuple = ()) -> list[aiosqlite.Row]:
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchall()

    async def fetchone(self, sql: str, params: tuple = ()) -> aiosqlite.Row | None:
        cursor = await self._conn.execute(sql, params)
        return await cursor.fetchone()


# --- Serialization helpers used by memory stores ---

def encode_embedding(embedding: list[float]) -> bytes:
    import numpy as np
    return np.array(embedding, dtype=np.float32).tobytes()


def decode_embedding(blob: bytes) -> list[float]:
    import numpy as np
    return np.frombuffer(blob, dtype=np.float32).tolist()


def encode_json(obj: object) -> str:
    return json.dumps(obj)


def decode_json(s: str) -> object:
    return json.loads(s)
