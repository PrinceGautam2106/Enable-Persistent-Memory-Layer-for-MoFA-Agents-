"""
MemoryAgent — a simple MoFA agent demonstrating the persistent memory workflow.

Workflow for each user query:
  1. Retrieve relevant memories from the vector store.
  2. Build an augmented prompt that includes those memories as context.
  3. Generate a response (a deterministic echo-style response is used here so
     that no external LLM API key is required; swap in any LLM call you like).
  4. Store the (query → response) interaction as a new memory.

This design keeps the agent stateless between calls while the MemoryManager
holds all long-term context.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memory_module import MemoryManager
from memory_module.storage import MemoryRecord


class MemoryAgent:
    """A MoFA agent enhanced with persistent semantic memory.

    Parameters
    ----------
    memory:
        An optional :class:`~memory_module.MemoryManager` instance.  If not
        provided, a new one is created with the default embedding model.
    top_k:
        Number of past memories to surface when building each prompt.
    session_id:
        Identifier attached to memories created by this agent instance.
    """

    def __init__(
        self,
        memory: MemoryManager | None = None,
        top_k: int = 3,
        session_id: str = "default",
    ) -> None:
        self.memory = memory if memory is not None else MemoryManager()
        self.top_k = top_k
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def chat(self, user_query: str) -> str:
        """Process a user query using the full memory workflow.

        Steps:
          1. Retrieve relevant memories.
          2. Build an augmented prompt.
          3. Generate a response.
          4. Persist the interaction.

        Parameters
        ----------
        user_query:
            The message sent by the user.

        Returns
        -------
        str
            The agent's response.
        """
        # Step 1: retrieve
        relevant_memories = self._retrieve_memories(user_query)

        # Step 2: build prompt
        prompt = self._build_prompt(user_query, relevant_memories)

        # Step 3: generate response
        response = self._generate_response(prompt, user_query, relevant_memories)

        # Step 4: store interaction
        self._store_interaction(user_query, response)

        return response

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _retrieve_memories(self, query: str) -> list[tuple[MemoryRecord, float]]:
        """Fetch the most relevant past memories for *query*."""
        return self.memory.retrieve(query, top_k=self.top_k)

    def _build_prompt(self, user_query: str, memories: list[tuple[MemoryRecord, float]]) -> str:
        """Construct an augmented prompt by prepending retrieved memories.

        The memory context is injected before the user query so that an LLM
        can use it to generate a more informed answer.
        """
        if not memories:
            return f"User: {user_query}"

        context_lines = ["Relevant past context:"]
        for record, score in memories:
            context_lines.append(f"  [{score:.2f}] {record.content}")

        context = "\n".join(context_lines)
        return f"{context}\n\nUser: {user_query}"

    def _generate_response(
        self,
        prompt: str,
        user_query: str,
        memories: list[tuple[MemoryRecord, float]],
    ) -> str:
        """Generate a response to the user query.

        This default implementation produces a simple, deterministic reply
        that summarises the retrieved memories.  Replace this method with a
        real LLM call (e.g. OpenAI, Anthropic, a local model) in production.
        """
        if memories:
            top_content = memories[0][0].content
            return (
                f"Based on what I remember, I can tell you: {top_content}\n"
                f"(Your question was: '{user_query}')"
            )
        return (
            f"I don't have any prior context about '{user_query}' yet, "
            "but I'll remember this for next time."
        )

    def _store_interaction(self, user_query: str, response: str) -> tuple[str, str]:
        """Persist the query and response as separate memory entries.

        Returns
        -------
        tuple[str, str]
            The memory IDs for the stored query and response.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        base_meta: dict[str, Any] = {"session_id": self.session_id, "timestamp": timestamp}

        query_id = self.memory.store(
            f"User asked: {user_query}",
            {**base_meta, "role": "user"},
        )
        response_id = self.memory.store(
            f"Agent answered: {response}",
            {**base_meta, "role": "agent"},
        )
        return query_id, response_id
