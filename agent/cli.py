"""
CLI demo for the MoFA Persistent Memory Agent.

Run with:
    python -m agent.cli

Or:
    python agent/cli.py

The script walks through a short scripted conversation to demonstrate:
  - storing facts as memories
  - retrieving relevant context in later turns
  - deleting a specific memory
"""

from __future__ import annotations

import sys

from memory_module import MemoryManager
from agent.memory_agent import MemoryAgent


SEPARATOR = "-" * 60


def print_section(title: str) -> None:
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


def demo_scripted() -> None:
    """Run a self-contained scripted demo without user interaction."""
    memory = MemoryManager()
    agent = MemoryAgent(memory=memory, top_k=3, session_id="demo")

    print_section("MoFA Persistent Memory Agent — Scripted Demo")

    # --- Pre-load some facts directly into memory ---
    print("\n[1] Pre-loading facts into memory...\n")
    facts = [
        ("The MoFA framework enables multi-agent workflows.", {"source": "docs"}),
        ("Paris is the capital of France.", {"source": "geography"}),
        ("Python is a popular programming language for AI.", {"source": "tech"}),
        ("FAISS is a library for efficient similarity search.", {"source": "tech"}),
        ("Sentence-transformers produce dense semantic embeddings.", {"source": "tech"}),
    ]
    stored_ids = []
    for content, meta in facts:
        mid = memory.store(content, meta)
        stored_ids.append(mid)
        print(f"  Stored [{mid[:8]}…]: {content}")

    # --- Conversation turn 1 ---
    print_section("Turn 1 — Asking about France")
    query1 = "What do you know about France?"
    print(f"User: {query1}")
    response1 = agent.chat(query1)
    print(f"Agent: {response1}")

    # --- Conversation turn 2 ---
    print_section("Turn 2 — Asking about AI tools")
    query2 = "Which tools are useful for building AI memory systems?"
    print(f"User: {query2}")
    response2 = agent.chat(query2)
    print(f"Agent: {response2}")

    # --- Show retrieved memories for a custom query ---
    print_section("Direct retrieval — 'multi-agent framework'")
    results = memory.retrieve("multi-agent framework", top_k=3)
    print("Top-3 memories:\n")
    for record, score in results:
        print(f"  [{score:.3f}] {record.content}")

    # --- Delete one memory and verify ---
    target_id = stored_ids[1]  # "Paris is the capital…"
    print_section(f"Deleting memory {target_id[:8]}…")
    deleted = memory.delete(target_id)
    print(f"  Deleted: {deleted}")
    print(f"  Total memories remaining: {len(memory)}")

    # --- Update a memory ---
    target_id = stored_ids[0]
    print_section(f"Updating memory {target_id[:8]}…")
    updated = memory.update(
        target_id,
        "The MoFA framework supports persistent memory for multi-agent AI systems.",
        {"source": "docs", "updated": True},
    )
    print(f"  Updated: {updated}")
    record = memory.get(target_id)
    if record:
        print(f"  New content: {record.content}")

    print_section("Demo complete")
    print(f"  Total memories in store: {len(memory)}\n")


def demo_interactive() -> None:
    """Run an interactive REPL where the user types queries."""
    memory = MemoryManager()
    agent = MemoryAgent(memory=memory, top_k=3, session_id="interactive")

    print_section("MoFA Persistent Memory Agent — Interactive Demo")
    print("Type your message and press Enter.  Commands:")
    print("  /store <text>          — store a fact directly")
    print("  /retrieve <query>      — show top-5 memories for a query")
    print("  /list                  — show memory count")
    print("  /quit                  — exit\n")

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not raw:
            continue

        if raw.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if raw.startswith("/store "):
            text = raw[len("/store "):].strip()
            mid = memory.store(text, {"source": "user_input"})
            print(f"  [Stored {mid[:8]}…]\n")

        elif raw.startswith("/retrieve "):
            query = raw[len("/retrieve "):].strip()
            results = memory.retrieve(query, top_k=5)
            if not results:
                print("  No memories found.\n")
            else:
                for record, score in results:
                    print(f"  [{score:.3f}] {record.content}")
                print()

        elif raw == "/list":
            print(f"  Total memories: {len(memory)}\n")

        else:
            response = agent.chat(raw)
            print(f"Agent: {response}\n")


def main() -> None:
    if "--interactive" in sys.argv or "-i" in sys.argv:
        demo_interactive()
    else:
        demo_scripted()


if __name__ == "__main__":
    main()
