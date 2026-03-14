"""
FAISS-backed vector storage for the MoFA persistent memory module.

Manages a FAISS flat inner-product index alongside an in-memory dict that
maps integer IDs to (content, metadata) records.  Because embeddings are
L2-normalised before insertion, inner-product search is equivalent to
cosine-similarity search.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import faiss
import numpy as np


@dataclass
class MemoryRecord:
    """A single persisted memory entry.

    Attributes
    ----------
    memory_id:
        Unique string identifier for this memory (UUID4 by default).
    content:
        The raw text that was stored.
    metadata:
        Arbitrary key/value data attached to the memory (e.g. timestamps,
        session IDs, speaker labels).
    faiss_index:
        Internal integer index used by FAISS; not exposed to callers.
    """

    memory_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    faiss_index: int = -1


class VectorStore:
    """FAISS-based vector store that supports add / search / remove operations.

    Parameters
    ----------
    dimension:
        Dimensionality of the embedding vectors to be stored.
    """

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension

        # IndexFlatIP performs exact brute-force inner-product search.
        # With L2-normalised vectors this equals cosine similarity.
        self._index: faiss.IndexIDMap = faiss.IndexIDMap(
            faiss.IndexFlatIP(dimension)
        )

        # Maps FAISS integer IDs → MemoryRecord
        self._records: dict[int, MemoryRecord] = {}

        # Reverse mapping: memory_id (UUID str) → FAISS integer ID
        # Kept in sync with _records to provide O(1) lookups by memory_id.
        self._id_map: dict[str, int] = {}

        # Monotonically increasing counter used as FAISS integer IDs
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, content: str, embedding: np.ndarray, metadata: dict[str, Any]) -> str:
        """Store a new memory and return its memory_id.

        Parameters
        ----------
        content:
            Text to remember.
        embedding:
            Pre-computed, L2-normalised float32 vector.
        metadata:
            Arbitrary metadata dict to attach to this record.

        Returns
        -------
        str
            A UUID4 string that uniquely identifies this memory.
        """
        faiss_id = self._next_id
        self._next_id += 1

        memory_id = str(uuid.uuid4())
        record = MemoryRecord(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
            faiss_index=faiss_id,
        )
        self._records[faiss_id] = record

        # Maintain reverse mapping for O(1) lookup by memory_id
        self._id_map[memory_id] = faiss_id

        # FAISS expects shape (1, dim) and int64 IDs
        vector = embedding.reshape(1, -1).astype(np.float32)
        ids = np.array([faiss_id], dtype=np.int64)
        self._index.add_with_ids(vector, ids)

        return memory_id

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[MemoryRecord, float]]:
        """Return the *top_k* most similar memories to *query_embedding*.

        Parameters
        ----------
        query_embedding:
            L2-normalised float32 query vector.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list of (MemoryRecord, score)
            Records sorted by descending cosine similarity.  Score is in
            ``[-1, 1]``; 1 means identical, −1 means opposite.
        """
        if len(self._records) == 0:
            return []

        k = min(top_k, len(self._records))
        vector = query_embedding.reshape(1, -1).astype(np.float32)
        scores, faiss_ids = self._index.search(vector, k)

        results: list[tuple[MemoryRecord, float]] = []
        for score, fid in zip(scores[0], faiss_ids[0]):
            if fid == -1:
                # FAISS uses −1 as a sentinel for "not found"
                continue
            record = self._records.get(int(fid))
            if record is not None:
                results.append((record, float(score)))

        return results

    def remove(self, memory_id: str) -> bool:
        """Delete a memory by its *memory_id*.

        Parameters
        ----------
        memory_id:
            The UUID returned by :meth:`add`.

        Returns
        -------
        bool
            ``True`` if the memory was found and removed, ``False`` otherwise.
        """
        # O(1) lookup via reverse mapping
        faiss_id: int | None = self._id_map.get(memory_id)

        if faiss_id is None:
            return False

        # Remove from FAISS index and internal dicts
        ids_to_remove = np.array([faiss_id], dtype=np.int64)
        self._index.remove_ids(ids_to_remove)
        del self._records[faiss_id]
        del self._id_map[memory_id]
        return True

    def update(self, memory_id: str, new_content: str, new_embedding: np.ndarray, new_metadata: dict[str, Any]) -> bool:
        """Replace the content and embedding of an existing memory.

        Implemented as a remove-then-re-add so that the FAISS index stays
        consistent.  The *memory_id* of the updated entry is preserved.

        Parameters
        ----------
        memory_id:
            ID of the memory to update.
        new_content:
            Replacement text.
        new_embedding:
            New L2-normalised float32 vector for *new_content*.
        new_metadata:
            Replacement metadata dict.

        Returns
        -------
        bool
            ``True`` if the memory was found and updated, ``False`` otherwise.
        """
        if not self.remove(memory_id):
            return False

        # Re-add with the same memory_id
        faiss_id = self._next_id
        self._next_id += 1
        record = MemoryRecord(
            memory_id=memory_id,
            content=new_content,
            metadata=new_metadata,
            faiss_index=faiss_id,
        )
        self._records[faiss_id] = record
        self._id_map[memory_id] = faiss_id
        vector = new_embedding.reshape(1, -1).astype(np.float32)
        ids = np.array([faiss_id], dtype=np.int64)
        self._index.add_with_ids(vector, ids)
        return True

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Retrieve a :class:`MemoryRecord` by its *memory_id*.

        Returns ``None`` if the ID does not exist.
        O(1) via the reverse mapping maintained alongside the FAISS index.
        """
        faiss_id = self._id_map.get(memory_id)
        if faiss_id is None:
            return None
        return self._records.get(faiss_id)

    def __len__(self) -> int:
        return len(self._records)
