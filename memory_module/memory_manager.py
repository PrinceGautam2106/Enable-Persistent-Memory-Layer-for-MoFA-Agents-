"""
MemoryManager — the central public interface for the MoFA persistent memory layer.

Typical usage::

    from memory_module import MemoryManager

    memory = MemoryManager()

    # Store a memory
    mid = memory.store("The capital of France is Paris.", {"source": "geography"})

    # Retrieve relevant memories
    results = memory.retrieve("What is the capital of France?", top_k=3)
    for record, score in results:
        print(f"[{score:.3f}] {record.content}")

    # Delete a memory
    memory.delete(mid)
"""

from __future__ import annotations

from typing import Any

from .embeddings import BaseEmbedder, get_default_embedder
from .storage import MemoryRecord, VectorStore


class MemoryManager:
    """High-level memory API for MoFA agents.

    Combines a :class:`~memory_module.embeddings.BaseEmbedder` with a
    :class:`~memory_module.storage.VectorStore` to provide persistent,
    semantically-searchable memory across agent interactions.

    Parameters
    ----------
    model_name:
        sentence-transformers model identifier used when *embedder* is not
        provided.  Defaults to ``all-MiniLM-L6-v2``.
    embedder:
        Optional custom embedder instance.  Pass any
        :class:`~memory_module.embeddings.BaseEmbedder` subclass to override
        the default.  Useful for testing (e.g. :class:`SimpleHashEmbedder`)
        or for using alternative embedding models.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self._embedder = embedder if embedder is not None else get_default_embedder(model_name)
        self._store = VectorStore(self._embedder.dimension)

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def store(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Embed *content* and persist it in the vector store.

        Parameters
        ----------
        content:
            Text to remember (e.g. a user utterance, agent response, or fact).
        metadata:
            Optional dictionary of key/value annotations to attach to this
            memory (e.g. ``{"timestamp": "...", "session_id": "abc"}``).

        Returns
        -------
        str
            A UUID4 string that can later be passed to :meth:`delete` or
            :meth:`update`.
        """
        if not content or not content.strip():
            raise ValueError("content must be a non-empty string")

        meta = metadata or {}
        embedding = self._embedder.embed_one(content)
        return self._store.add(content, embedding, meta)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[MemoryRecord, float]]:
        """Return the most semantically similar memories to *query*.

        Parameters
        ----------
        query:
            Free-text query used to search stored memories.
        top_k:
            Maximum number of memories to return (default: 5).

        Returns
        -------
        list of (MemoryRecord, score)
            Ordered from most to least similar.  *score* is cosine similarity
            in the range ``[-1, 1]``.
        """
        if top_k < 1:
            raise ValueError("top_k must be a positive integer")

        query_embedding = self._embedder.embed_one(query)
        return self._store.search(query_embedding, top_k)

    def delete(self, memory_id: str) -> bool:
        """Remove a memory from the store.

        Parameters
        ----------
        memory_id:
            The ID returned by :meth:`store`.

        Returns
        -------
        bool
            ``True`` if the memory was found and deleted.
        """
        return self._store.remove(memory_id)

    def update(self, memory_id: str, new_content: str, new_metadata: dict[str, Any] | None = None) -> bool:
        """Replace the content (and optionally metadata) of an existing memory.

        The existing *memory_id* is preserved after the update.

        Parameters
        ----------
        memory_id:
            ID of the memory to update.
        new_content:
            Replacement text.  A new embedding is computed automatically.
        new_metadata:
            Replacement metadata dict.  Pass ``None`` to keep the original.

        Returns
        -------
        bool
            ``True`` if the memory was found and updated.
        """
        if not new_content or not new_content.strip():
            raise ValueError("new_content must be a non-empty string")

        # Preserve old metadata if not provided
        if new_metadata is None:
            existing = self._store.get(memory_id)
            new_metadata = existing.metadata if existing else {}

        new_embedding = self._embedder.embed_one(new_content)
        return self._store.update(memory_id, new_content, new_embedding, new_metadata)

    def get(self, memory_id: str) -> MemoryRecord | None:
        """Fetch a single :class:`~memory_module.storage.MemoryRecord` by ID.

        Returns ``None`` if the ID does not exist.
        """
        return self._store.get(memory_id)

    def __len__(self) -> int:
        """Return the total number of stored memories."""
        return len(self._store)
