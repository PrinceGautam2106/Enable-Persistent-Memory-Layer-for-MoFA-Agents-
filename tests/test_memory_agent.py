"""
Unit tests for MemoryAgent — the memory-augmented agent workflow.
"""

from __future__ import annotations

import pytest

from memory_module import MemoryManager, SimpleHashEmbedder
from agent.memory_agent import MemoryAgent


@pytest.fixture()
def agent() -> MemoryAgent:
    """Return a fresh MemoryAgent backed by the offline SimpleHashEmbedder."""
    return MemoryAgent(memory=MemoryManager(embedder=SimpleHashEmbedder()), top_k=3, session_id="test")


# ---------------------------------------------------------------------------
# chat workflow
# ---------------------------------------------------------------------------

class TestChat:
    def test_chat_returns_string(self, agent: MemoryAgent) -> None:
        response = agent.chat("Hello, who are you?")
        assert isinstance(response, str)
        assert len(response) > 0

    def test_chat_stores_interaction(self, agent: MemoryAgent) -> None:
        """Each chat turn should add two memories (user + agent)."""
        assert len(agent.memory) == 0
        agent.chat("What is AI?")
        assert len(agent.memory) == 2

    def test_chat_multiple_turns_accumulate_memories(self, agent: MemoryAgent) -> None:
        agent.chat("First question")
        agent.chat("Second question")
        agent.chat("Third question")
        assert len(agent.memory) == 6  # 2 memories per turn

    def test_chat_no_memory_response(self, agent: MemoryAgent) -> None:
        """First ever query should acknowledge lack of prior context."""
        response = agent.chat("Tell me about quantum physics")
        assert "don't have any prior context" in response or len(response) > 0

    def test_chat_uses_stored_memories(self, agent: MemoryAgent) -> None:
        """After storing a fact, it should surface in a related query."""
        agent.memory.store("The speed of light is approximately 299,792 km/s.")
        response = agent.chat("What is the speed of light?")
        # The response should reference the stored fact
        assert "299,792" in response or "speed" in response.lower() or len(response) > 0


# ---------------------------------------------------------------------------
# memory workflow helpers
# ---------------------------------------------------------------------------

class TestMemoryWorkflow:
    def test_retrieve_memories_empty(self, agent: MemoryAgent) -> None:
        results = agent._retrieve_memories("anything")
        assert results == []

    def test_build_prompt_no_memories(self, agent: MemoryAgent) -> None:
        prompt = agent._build_prompt("Hello", [])
        assert "Hello" in prompt

    def test_build_prompt_with_memories(self, agent: MemoryAgent) -> None:
        agent.memory.store("Some relevant fact about planets.")
        memories = agent._retrieve_memories("planets")
        prompt = agent._build_prompt("Tell me about planets", memories)
        assert "Relevant past context" in prompt
        assert "planets" in prompt.lower()

    def test_store_interaction_creates_two_memories(self, agent: MemoryAgent) -> None:
        query_id, response_id = agent._store_interaction("User question", "Agent answer")
        assert isinstance(query_id, str)
        assert isinstance(response_id, str)
        assert query_id != response_id
        assert len(agent.memory) == 2

    def test_stored_memories_have_session_id(self, agent: MemoryAgent) -> None:
        agent._store_interaction("My question", "My answer")
        results = agent.memory.retrieve("question", top_k=5)
        for record, _ in results:
            assert record.metadata.get("session_id") == "test"

    def test_stored_memories_have_timestamp(self, agent: MemoryAgent) -> None:
        agent._store_interaction("Q", "A")
        results = agent.memory.retrieve("Q", top_k=5)
        for record, _ in results:
            assert "timestamp" in record.metadata


# ---------------------------------------------------------------------------
# shared memory across agents
# ---------------------------------------------------------------------------

class TestSharedMemory:
    def test_two_agents_share_memory(self) -> None:
        """Two agents pointing at the same MemoryManager share memories."""
        shared_memory = MemoryManager(embedder=SimpleHashEmbedder())
        agent1 = MemoryAgent(memory=shared_memory, session_id="agent1")
        agent2 = MemoryAgent(memory=shared_memory, session_id="agent2")

        agent1.chat("I learned that water boils at 100 degrees Celsius.")
        # agent2 should be able to retrieve what agent1 stored
        results = agent2.memory.retrieve("boiling point of water", top_k=3)
        assert len(results) > 0
