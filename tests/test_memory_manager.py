"""
Unit tests for MemoryManager (store / retrieve / delete / update).
"""

from __future__ import annotations

import pytest

from memory_module import MemoryManager, SimpleHashEmbedder


@pytest.fixture()
def memory() -> MemoryManager:
    """Return a fresh MemoryManager backed by the offline SimpleHashEmbedder."""
    return MemoryManager(embedder=SimpleHashEmbedder())


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

class TestStore:
    def test_store_returns_string_id(self, memory: MemoryManager) -> None:
        mid = memory.store("Hello world")
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_store_increments_length(self, memory: MemoryManager) -> None:
        assert len(memory) == 0
        memory.store("First memory")
        assert len(memory) == 1
        memory.store("Second memory")
        assert len(memory) == 2

    def test_store_with_metadata(self, memory: MemoryManager) -> None:
        mid = memory.store("Fact with metadata", {"source": "test", "importance": 5})
        record = memory.get(mid)
        assert record is not None
        assert record.metadata["source"] == "test"
        assert record.metadata["importance"] == 5

    def test_store_preserves_content(self, memory: MemoryManager) -> None:
        content = "The sky is blue and the sun is bright."
        mid = memory.store(content)
        record = memory.get(mid)
        assert record is not None
        assert record.content == content

    def test_store_empty_string_raises(self, memory: MemoryManager) -> None:
        with pytest.raises(ValueError):
            memory.store("")

    def test_store_whitespace_only_raises(self, memory: MemoryManager) -> None:
        with pytest.raises(ValueError):
            memory.store("   ")

    def test_store_unique_ids(self, memory: MemoryManager) -> None:
        ids = {memory.store(f"Memory number {i}") for i in range(10)}
        assert len(ids) == 10


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------

class TestRetrieve:
    def test_retrieve_empty_store_returns_empty(self, memory: MemoryManager) -> None:
        results = memory.retrieve("anything")
        assert results == []

    def test_retrieve_returns_most_relevant(self, memory: MemoryManager) -> None:
        memory.store("The capital of France is Paris.")
        memory.store("Python is a programming language.")
        memory.store("FAISS is used for similarity search.")

        results = memory.retrieve("What is the capital of France?", top_k=1)
        assert len(results) == 1
        record, score = results[0]
        assert "France" in record.content or "Paris" in record.content
        # Cosine similarity of L2-normalised vectors is in [-1, 1]; allow a tiny
        # float rounding tolerance above 1.0.
        COSINE_SIMILARITY_TOLERANCE = 1.05
        assert 0 <= score <= COSINE_SIMILARITY_TOLERANCE

    def test_retrieve_respects_top_k(self, memory: MemoryManager) -> None:
        for i in range(10):
            memory.store(f"Fact number {i}: some information here.")
        results = memory.retrieve("some fact", top_k=3)
        assert len(results) <= 3

    def test_retrieve_top_k_larger_than_store(self, memory: MemoryManager) -> None:
        memory.store("Only one memory.")
        results = memory.retrieve("memory", top_k=100)
        assert len(results) == 1

    def test_retrieve_top_k_zero_raises(self, memory: MemoryManager) -> None:
        with pytest.raises(ValueError):
            memory.retrieve("query", top_k=0)

    def test_retrieve_scores_in_descending_order(self, memory: MemoryManager) -> None:
        for i in range(5):
            memory.store(f"Document {i} about various topics.")
        results = memory.retrieve("document topics", top_k=5)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_returns_tuples(self, memory: MemoryManager) -> None:
        memory.store("Some content here.")
        results = memory.retrieve("content")
        assert len(results) == 1
        record, score = results[0]
        assert hasattr(record, "content")
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_returns_true_for_existing(self, memory: MemoryManager) -> None:
        mid = memory.store("Memory to delete")
        assert memory.delete(mid) is True

    def test_delete_reduces_length(self, memory: MemoryManager) -> None:
        mid = memory.store("Memory to delete")
        assert len(memory) == 1
        memory.delete(mid)
        assert len(memory) == 0

    def test_delete_returns_false_for_unknown_id(self, memory: MemoryManager) -> None:
        assert memory.delete("nonexistent-id") is False

    def test_deleted_memory_not_retrieved(self, memory: MemoryManager) -> None:
        mid = memory.store("Unique phrase: xylophone jazz cats")
        memory.delete(mid)
        results = memory.retrieve("xylophone jazz cats", top_k=5)
        ids = [r.memory_id for r, _ in results]
        assert mid not in ids

    def test_delete_only_removes_target(self, memory: MemoryManager) -> None:
        mid1 = memory.store("First memory")
        mid2 = memory.store("Second memory")
        memory.delete(mid1)
        assert len(memory) == 1
        record = memory.get(mid2)
        assert record is not None


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_update_returns_true_for_existing(self, memory: MemoryManager) -> None:
        mid = memory.store("Original content")
        assert memory.update(mid, "Updated content") is True

    def test_update_changes_content(self, memory: MemoryManager) -> None:
        mid = memory.store("Old text about apples")
        memory.update(mid, "New text about oranges")
        record = memory.get(mid)
        assert record is not None
        assert "oranges" in record.content

    def test_update_preserves_memory_id(self, memory: MemoryManager) -> None:
        mid = memory.store("Some text")
        memory.update(mid, "Different text")
        assert memory.get(mid) is not None

    def test_update_changes_metadata(self, memory: MemoryManager) -> None:
        mid = memory.store("Content", {"version": 1})
        memory.update(mid, "Content v2", {"version": 2})
        record = memory.get(mid)
        assert record is not None
        assert record.metadata["version"] == 2

    def test_update_nonexistent_returns_false(self, memory: MemoryManager) -> None:
        assert memory.update("no-such-id", "replacement") is False

    def test_update_empty_content_raises(self, memory: MemoryManager) -> None:
        mid = memory.store("Content")
        with pytest.raises(ValueError):
            memory.update(mid, "")


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_returns_record(self, memory: MemoryManager) -> None:
        mid = memory.store("Hello")
        record = memory.get(mid)
        assert record is not None
        assert record.memory_id == mid
        assert record.content == "Hello"

    def test_get_missing_id_returns_none(self, memory: MemoryManager) -> None:
        assert memory.get("missing-id") is None
