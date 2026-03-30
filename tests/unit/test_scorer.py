import time
import pytest
import numpy as np
from cce.models.types import Chunk, ChunkType
from cce.pipeline.scorer import ImportanceScorer
from cce.settings import Settings


@pytest.fixture
def scorer():
    return ImportanceScorer(Settings())


def _make_chunk(content: str, turn_index: int = 0, embedding: list[float] | None = None) -> Chunk:
    return Chunk(
        chunk_id="test-id",
        project_id="proj",
        content=content,
        content_type=ChunkType.PROSE,
        token_count=10,
        char_offset=0,
        turn_index=turn_index,
        created_at=time.time(),
        embedding=embedding,
    )


def _unit_vec(dim: int = 384, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


# --- Recency ---

def test_recency_current_turn_scores_1(scorer):
    chunk = _make_chunk("text", turn_index=5)
    score = scorer._recency(chunk, current_turn_index=5)
    assert score == pytest.approx(1.0)


def test_recency_decays_with_distance(scorer):
    chunk_recent = _make_chunk("text", turn_index=9)
    chunk_old = _make_chunk("text", turn_index=0)
    s_recent = scorer._recency(chunk_recent, current_turn_index=10)
    s_old = scorer._recency(chunk_old, current_turn_index=10)
    assert s_recent > s_old


def test_recency_never_negative(scorer):
    chunk = _make_chunk("text", turn_index=0)
    score = scorer._recency(chunk, current_turn_index=1000)
    assert score >= 0.0


# --- Keyword Jaccard ---

def test_keyword_identical_text(scorer):
    chunk = _make_chunk("function calculate total items")
    score = scorer._keyword(chunk, frozenset({"function", "calculate", "total", "items"}))
    assert score == pytest.approx(1.0)


def test_keyword_no_overlap(scorer):
    chunk = _make_chunk("authentication login password")
    score = scorer._keyword(chunk, frozenset({"database", "query", "index"}))
    assert score == pytest.approx(0.0)


def test_keyword_partial_overlap(scorer):
    chunk = _make_chunk("function calculates total items count")
    score = scorer._keyword(chunk, frozenset({"function", "database", "query"}))
    assert 0.0 < score < 1.0


def test_keyword_empty_query(scorer):
    chunk = _make_chunk("some content here")
    assert scorer._keyword(chunk, frozenset()) == 0.0


# --- Relevance ---

def test_relevance_identical_vectors(scorer):
    vec = _unit_vec()
    chunk = _make_chunk("text", embedding=vec)
    q = np.array(vec, dtype=np.float32)
    score = scorer._relevance(chunk, q)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_relevance_orthogonal_vectors(scorer):
    v1 = np.zeros(384, dtype=np.float32)
    v1[0] = 1.0
    v2 = np.zeros(384, dtype=np.float32)
    v2[1] = 1.0
    chunk = _make_chunk("text", embedding=v1.tolist())
    score = scorer._relevance(chunk, v2)
    assert score == pytest.approx(0.0, abs=1e-5)


def test_relevance_no_embedding(scorer):
    chunk = _make_chunk("text", embedding=None)
    q = np.array(_unit_vec(), dtype=np.float32)
    assert scorer._relevance(chunk, q) == 0.0


# --- Composite score ---

def test_score_returns_sorted_descending(scorer):
    vec = _unit_vec(seed=1)
    chunks = [
        _make_chunk("authentication login", turn_index=0, embedding=_unit_vec(seed=2)),
        _make_chunk("function calculate total", turn_index=9, embedding=vec),
        _make_chunk("database query index", turn_index=5, embedding=_unit_vec(seed=3)),
    ]
    scored = scorer.score(chunks, vec, "function calculate", current_turn_index=10)
    scores = [s.composite_score for s in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_composite_in_range(scorer):
    vec = _unit_vec()
    chunk = _make_chunk("some text content here", turn_index=1, embedding=vec)
    scored = scorer.score([chunk], vec, "some text", current_turn_index=1)
    assert 0.0 <= scored[0].composite_score <= 1.0
