"""
MoFA Persistent Memory Module

Provides a lightweight, reusable memory system for AI agents to store and
retrieve knowledge from previous interactions using semantic similarity search.
"""

from .memory_manager import MemoryManager
from .embeddings import BaseEmbedder, SentenceTransformerEmbedder, SimpleHashEmbedder, get_default_embedder

__all__ = [
    "MemoryManager",
    "BaseEmbedder",
    "SentenceTransformerEmbedder",
    "SimpleHashEmbedder",
    "get_default_embedder",
]
