"""
Embedding utilities for the MoFA persistent memory module.

Provides a common :class:`BaseEmbedder` interface and two implementations:

* :class:`SentenceTransformerEmbedder` — wraps a sentence-transformers model
  for high-quality semantic embeddings (requires internet access on first use
  to download model weights).
* :class:`SimpleHashEmbedder` — deterministic numpy-based embedder that works
  fully offline; suitable for testing or environments without internet access.

Use :func:`get_default_embedder` to obtain the best available embedder
automatically.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Abstract base class for all embedder implementations.

    Sub-classes must implement :meth:`embed` and expose :attr:`dimension`.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalised embeddings for a list of texts.

        Parameters
        ----------
        texts:
            One or more strings to embed.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(len(texts), dimension)``.
        """

    def embed_one(self, text: str) -> np.ndarray:
        """Convenience wrapper that embeds a single string.

        Returns
        -------
        numpy.ndarray
            Float32 array of shape ``(dimension,)``.
        """
        return self.embed([text])[0]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors produced by this embedder."""


# ---------------------------------------------------------------------------
# Sentence-transformers implementation (production quality)
# ---------------------------------------------------------------------------

# Default lightweight model — balances speed and quality.
_DEFAULT_MODEL = "all-MiniLM-L6-v2"


class SentenceTransformerEmbedder(BaseEmbedder):
    """Wraps a sentence-transformers model for high-quality semantic embeddings.

    Requires internet access on first use to download model weights from
    HuggingFace, or a locally cached model.

    Parameters
    ----------
    model_name:
        sentence-transformers model identifier.
        Defaults to ``all-MiniLM-L6-v2``.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL) -> None:
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Hash-based offline implementation (testing / no-network environments)
# ---------------------------------------------------------------------------

class SimpleHashEmbedder(BaseEmbedder):
    """Deterministic, offline embedder based on character-level hashing.

    Produces vectors of a fixed dimensionality by hashing word n-grams and
    accumulating their contributions.  Vectors are L2-normalised so that they
    are compatible with the cosine-similarity search in the vector store.

    This implementation is **not** semantically meaningful beyond very crude
    word-overlap similarity, but it is reproducible, fast, and requires no
    external dependencies beyond NumPy.  Use it for unit tests or offline
    environments.

    Parameters
    ----------
    dimension:
        Size of the embedding vectors (default: 128).
    """

    def __init__(self, dimension: int = 128) -> None:
        self._dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        result = np.stack([self._embed_text(t) for t in texts])
        return result.astype(np.float32)

    def _embed_text(self, text: str) -> np.ndarray:
        """Convert *text* to a normalised float32 vector via hashing."""
        vector = np.zeros(self._dimension, dtype=np.float64)
        tokens = text.lower().split()
        for token in tokens:
            # Use blake2b (fast, non-cryptographic use case — no security requirement)
            digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
            # Spread each byte's contribution across the vector space
            for i, byte in enumerate(digest):
                idx = (byte + i * 31) % self._dimension
                vector[idx] += byte / 255.0
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.astype(np.float32)

    @property
    def dimension(self) -> int:
        return self._dimension


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

# Legacy alias kept for backward compatibility
EmbeddingModel = SentenceTransformerEmbedder


def get_default_embedder(model_name: str = _DEFAULT_MODEL) -> BaseEmbedder:
    """Return the best available embedder for the current environment.

    Attempts to instantiate :class:`SentenceTransformerEmbedder`.  If that
    fails (e.g. no internet access, model not cached), falls back to
    :class:`SimpleHashEmbedder` and emits a warning.
    """
    import warnings
    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"Could not load sentence-transformers model '{model_name}' "
            f"({exc}). Falling back to SimpleHashEmbedder. "
            "Semantic search quality will be reduced.",
            RuntimeWarning,
            stacklevel=2,
        )
        return SimpleHashEmbedder()
