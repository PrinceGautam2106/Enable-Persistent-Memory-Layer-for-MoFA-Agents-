# Enable-Persistent-Memory-Layer-for-MoFA-Agents-

A lightweight, modular **Persistent Memory Module** for [MoFA](https://github.com/moxin-org/mofa) AI agents.  It lets agents store and retrieve knowledge from previous interactions using semantic similarity search, giving them long-term context across sessions.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Agent Memory Workflow](#agent-memory-workflow)
6. [CLI Demo](#cli-demo)
7. [Running the Tests](#running-the-tests)
8. [Configuration & Integration](#configuration--integration)

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    MemoryAgent                      │
│  (agent/memory_agent.py)                            │
│                                                     │
│  1. Receive user query                              │
│  2. Retrieve relevant memories ──────────────┐     │
│  3. Build augmented prompt                   │     │
│  4. Generate response                        │     │
│  5. Store interaction as new memory ─────────┤     │
└──────────────────────┬──────────────────────┘     │
                       │                             │
                       ▼                             │
┌─────────────────────────────────────────────────────┐
│                  MemoryManager                      │
│  (memory_module/memory_manager.py)                  │
│                                                     │
│  store(content, metadata)  → memory_id             │
│  retrieve(query, top_k)    → [(record, score)]     │
│  delete(memory_id)         → bool                  │
│  update(memory_id, …)      → bool                  │
└──────────┬────────────────────────────┬─────────────┘
           │                            │
           ▼                            ▼
┌─────────────────────┐    ┌───────────────────────────┐
│    BaseEmbedder     │    │       VectorStore          │
│  (embeddings.py)    │    │  (storage.py)              │
│                     │    │                            │
│  SentenceTransformer│    │  FAISS IndexFlatIP         │
│    Embedder         │    │  (inner-product / cosine)  │
│  ─────────────────  │    │                            │
│  SimpleHashEmbedder │    │  {faiss_id → MemoryRecord} │
│  (offline/tests)    │    │                            │
└─────────────────────┘    └───────────────────────────┘
```

### How It Improves AI Agents

| Without memory | With persistent memory |
|---|---|
| Each conversation starts from scratch | Prior context is available in every session |
| Agent cannot recall user preferences | Preferences and facts accumulate over time |
| Repeated questions get identical generic answers | Answers improve as the agent learns |
| Multi-agent systems cannot share knowledge | Agents can read/write a shared memory store |

### Key Design Decisions

- **Pluggable embedders** — `BaseEmbedder` is an abstract interface.  Swap in any embedding model without touching the rest of the code.
- **FAISS inner-product index with L2-normalised vectors** — equivalent to cosine similarity search; exact and fast for the memory sizes typical in agent workflows.
- **Deterministic offline embedder** — `SimpleHashEmbedder` works with no network access, making the module fully testable in CI environments.
- **Automatic fallback** — `get_default_embedder()` attempts to load a sentence-transformers model and silently falls back to `SimpleHashEmbedder` if the model is unavailable.
- **UUID-based memory IDs** — callers get a stable opaque handle to each memory for targeted deletes/updates.

---

## Project Structure

```
.
├── memory_module/               # Core memory library
│   ├── __init__.py              # Public exports
│   ├── embeddings.py            # BaseEmbedder, SentenceTransformerEmbedder, SimpleHashEmbedder
│   ├── storage.py               # VectorStore (FAISS) + MemoryRecord dataclass
│   └── memory_manager.py        # MemoryManager — the main public API
│
├── agent/                       # Example agent using the memory module
│   ├── __init__.py
│   ├── memory_agent.py          # MemoryAgent with full memory workflow
│   └── cli.py                   # CLI demo (scripted + interactive modes)
│
├── tests/                       # Unit tests
│   ├── __init__.py
│   ├── test_memory_manager.py   # Tests for MemoryManager
│   └── test_memory_agent.py     # Tests for MemoryAgent workflow
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs:

| Package | Purpose |
|---|---|
| `sentence-transformers` | High-quality semantic embeddings |
| `faiss-cpu` | Fast vector similarity search |
| `numpy` | Numeric array operations |

### 2. Use the memory module directly

```python
from memory_module import MemoryManager

memory = MemoryManager()

# Store facts
mid1 = memory.store("The Eiffel Tower is in Paris.", {"source": "geography"})
mid2 = memory.store("Python was created by Guido van Rossum.", {"source": "tech"})
mid3 = memory.store("FAISS is great for nearest-neighbour search.", {"source": "tech"})

# Retrieve relevant memories
results = memory.retrieve("What is in Paris?", top_k=2)
for record, score in results:
    print(f"[{score:.3f}] {record.content}")
# [0.842] The Eiffel Tower is in Paris.
# [0.312] FAISS is great for nearest-neighbour search.

# Update a memory
memory.update(mid1, "The Eiffel Tower is a landmark in Paris, France.")

# Delete a memory
memory.delete(mid2)

print(f"Total memories: {len(memory)}")  # 2
```

### 3. Use the memory-augmented agent

```python
from agent.memory_agent import MemoryAgent

agent = MemoryAgent(top_k=3, session_id="my-session")

# Turn 1 — no prior context yet
response = agent.chat("What do you know about the Eiffel Tower?")
print(response)

# Turn 2 — the agent now has memory of the previous exchange
response = agent.chat("Where is that tower located?")
print(response)
```

### 4. Use an offline embedder (no internet required)

```python
from memory_module import MemoryManager, SimpleHashEmbedder

memory = MemoryManager(embedder=SimpleHashEmbedder())
```

---

## API Reference

### `MemoryManager`

```python
MemoryManager(model_name="all-MiniLM-L6-v2", embedder=None)
```

#### `memory.store(content, metadata=None) → str`

Embed and persist *content*.  Returns a UUID memory ID.

```python
mid = memory.store("Water boils at 100 °C.", {"source": "chemistry"})
```

#### `memory.retrieve(query, top_k=5) → list[tuple[MemoryRecord, float]]`

Return up to *top_k* memories most similar to *query*, ordered by descending cosine similarity score.

```python
results = memory.retrieve("boiling point", top_k=3)
for record, score in results:
    print(record.memory_id, record.content, record.metadata, score)
```

#### `memory.delete(memory_id) → bool`

Remove the memory with the given ID.  Returns `True` on success.

```python
memory.delete(mid)
```

#### `memory.update(memory_id, new_content, new_metadata=None) → bool`

Replace content (and optionally metadata) of an existing memory.  The ID is preserved.

```python
memory.update(mid, "Water boils at 100 °C at sea level.", {"source": "chemistry", "updated": True})
```

#### `memory.get(memory_id) → MemoryRecord | None`

Fetch a single record by ID without doing a similarity search.

#### `len(memory) → int`

Total number of stored memories.

---

### `MemoryRecord`

```python
@dataclass
class MemoryRecord:
    memory_id: str        # UUID4 string
    content:   str        # The stored text
    metadata:  dict       # Arbitrary key/value annotations
    faiss_index: int      # Internal FAISS integer ID (do not modify)
```

---

### Embedders

| Class | Description |
|---|---|
| `SentenceTransformerEmbedder(model_name)` | Production-quality semantic embeddings via sentence-transformers |
| `SimpleHashEmbedder(dimension=128)` | Deterministic, offline, numpy-only — for testing |
| `get_default_embedder(model_name)` | Returns `SentenceTransformerEmbedder` or falls back to `SimpleHashEmbedder` |

---

## Agent Memory Workflow

`MemoryAgent.chat(user_query)` follows this pipeline on every call:

```
User query
    │
    ▼
1. retrieve(query, top_k)          ← find relevant past memories
    │
    ▼
2. build_prompt(query, memories)   ← prepend context to the prompt
    │
    ▼
3. generate_response(prompt)       ← call LLM (or default echo impl.)
    │
    ▼
4. store("User asked: …")          ← persist query as new memory
   store("Agent answered: …")      ← persist response as new memory
    │
    ▼
Response returned to caller
```

### Replacing the LLM backend

Override `MemoryAgent._generate_response` to plug in any LLM:

```python
import openai
from agent.memory_agent import MemoryAgent

class GPTMemoryAgent(MemoryAgent):
    def _generate_response(self, prompt, user_query, memories):
        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content
```

---

## CLI Demo

### Scripted demo (no interaction required)

```bash
python -m agent.cli
# or
python agent/cli.py
```

### Interactive REPL

```bash
python -m agent.cli --interactive
# or
python agent/cli.py -i
```

Interactive commands:

| Command | Description |
|---|---|
| `<text>` | Chat with the agent |
| `/store <text>` | Store a fact directly |
| `/retrieve <query>` | Show top-5 memories for a query |
| `/list` | Show total memory count |
| `/quit` | Exit |

---

## Running the Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Tests use `SimpleHashEmbedder` and require no internet access or GPU.

```
tests/test_memory_manager.py   27 tests — store, retrieve, delete, update, get
tests/test_memory_agent.py     12 tests — chat workflow, prompt building, shared memory
```
