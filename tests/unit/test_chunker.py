import time
import pytest
from cce.models.types import ChunkType, ConversationTurn, Role
from cce.pipeline.chunker import Chunker, estimate_tokens, extract_keywords
from cce.settings import Settings


@pytest.fixture
def chunker():
    return Chunker(Settings())


def _turns(*contents: str) -> list[ConversationTurn]:
    return [
        ConversationTurn(role=Role.USER, content=c, turn_index=i, timestamp=time.time())
        for i, c in enumerate(contents)
    ]


# --- estimate_tokens ---

def test_estimate_tokens_basic():
    assert estimate_tokens("hello world") == 2  # int(2 * 1.3) = 2
    # More important: non-zero
    assert estimate_tokens("") == 1


def test_estimate_tokens_scales():
    short = estimate_tokens("one two three")
    long = estimate_tokens("one two three " * 10)
    assert long > short


# --- extract_keywords ---

def test_extract_keywords_filters_stopwords():
    kw = extract_keywords("the function returns a list of items")
    assert "the" not in kw
    assert "function" in kw
    assert "returns" in kw
    assert "list" in kw
    assert "items" in kw


def test_extract_keywords_empty():
    assert extract_keywords("") == frozenset()


def test_extract_keywords_code():
    kw = extract_keywords("def calculate_total(items): return sum(items)")
    assert "calculate_total" in kw
    assert "items" in kw


# --- Chunker: prose ---

def test_chunk_plain_prose(chunker):
    text = "This is a simple sentence. It should become one chunk."
    chunks = chunker.chunk_text(text, project_id="proj1")
    assert len(chunks) >= 1
    assert all(c.content_type == ChunkType.PROSE for c in chunks)
    assert all(c.project_id == "proj1" for c in chunks)


def test_chunk_multi_paragraph(chunker):
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph."
    chunks = chunker.chunk_text(text, project_id="p")
    assert len(chunks) == 3


# --- Chunker: code fences ---

def test_chunk_code_block(chunker):
    text = "Some prose.\n\n```python\ndef foo():\n    return 42\n```\n\nMore prose."
    chunks = chunker.chunk_text(text, project_id="p")
    types = [c.content_type for c in chunks]
    assert ChunkType.CODE in types
    assert ChunkType.PROSE in types


def test_chunk_unclosed_fence(chunker):
    """Unclosed code fence should not crash — treated as code to EOF."""
    text = "Intro.\n\n```python\ndef foo():\n    pass\n"
    chunks = chunker.chunk_text(text, project_id="p")
    code_chunks = [c for c in chunks if c.content_type == ChunkType.CODE]
    assert len(code_chunks) >= 1


def test_chunk_code_content_preserved(chunker):
    code = "def bar(x):\n    return x * 2"
    text = f"```\n{code}\n```"
    chunks = chunker.chunk_text(text, project_id="p")
    code_chunks = [c for c in chunks if c.content_type == ChunkType.CODE]
    assert any("bar" in c.content for c in code_chunks)


# --- Chunker: size enforcement ---

def test_oversized_prose_is_split(chunker):
    # Create prose that clearly exceeds max_chunk_tokens_prose (256)
    sentence = "This is a sentence that contains information. "
    big_text = sentence * 60  # ~60 * 8 words = 480 words >> 256 tokens
    chunks = chunker.chunk_text(big_text, project_id="p")
    assert len(chunks) > 1


# --- Chunker: turns ---

def test_chunk_turns_assigns_turn_index(chunker):
    turns = _turns("First turn content.", "Second turn content.")
    chunks = chunker.chunk_turns(turns, project_id="p")
    indices = [c.turn_index for c in chunks]
    assert 0 in indices
    assert 1 in indices


def test_chunk_turns_metadata_has_role(chunker):
    turns = _turns("Hello from user.")
    chunks = chunker.chunk_turns(turns, project_id="p")
    assert chunks[0].metadata["role"] == "user"


def test_chunk_ids_are_unique(chunker):
    turns = _turns("A", "B", "C")
    chunks = chunker.chunk_turns(turns, project_id="p")
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
