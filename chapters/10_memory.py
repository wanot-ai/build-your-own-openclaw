#!/usr/bin/env python3
"""
Chapter 10: Long-Term Memory
==============================

NanoClaw's memory: file-based long-term storage.

NanoClaw stores memory in the group's filesystem:
- groups/{folder}/CLAUDE.md — per-group personality (auto-loaded by Claude Code)
- groups/{folder}/ — any files the agent creates

The agent (Claude Code) has natural file operations, so memory is just files.
It can create notes, save research, maintain TODO lists — all as regular files
in its mounted workspace.

Our simplified version:
- workspace/memory/ — global memory directory
- Key-value storage as markdown files
- Simple keyword search across all memory files

Run: uv run --with anthropic python chapters/10_memory.py
"""

import os
import time

MEMORY_DIR = os.path.join(
    os.path.dirname(__file__), "..", "workspace", "memory"
)


def save_memory(key: str, content: str):
    """
    Saves memory as a file.

    In NanoClaw, the agent uses Claude Code's file writing capability.
    We implement it as a tool.
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)
    filepath = os.path.join(MEMORY_DIR, f"{key}.md")
    with open(filepath, "w") as f:
        f.write(content)
    return f"Saved to memory: {key}"


def search_memory(query: str) -> str:
    """
    Searches memory.

    Simple keyword search. OpenClaw also supports vector search,
    but keyword search alone is quite useful.
    """
    if not os.path.exists(MEMORY_DIR):
        return "No memories found."

    query_words = query.lower().split()
    results = []

    for fname in sorted(os.listdir(MEMORY_DIR)):
        if not fname.endswith(".md"):
            continue
        filepath = os.path.join(MEMORY_DIR, fname)
        with open(filepath, "r") as f:
            content = f.read()

        # Keyword matching (added to results if any word matches)
        content_lower = content.lower()
        if any(word in content_lower for word in query_words):
            results.append(f"--- {fname} ---\n{content}")

    return "\n\n".join(results) if results else "No matching memories found."


def list_memories() -> list[str]:
    """List all memory files."""
    if not os.path.exists(MEMORY_DIR):
        return []
    return [f for f in sorted(os.listdir(MEMORY_DIR)) if f.endswith(".md")]


def read_memory(key: str) -> str | None:
    """Reads a specific memory."""
    filepath = os.path.join(MEMORY_DIR, f"{key}.md")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return f.read()
    return None


def delete_memory(key: str) -> str:
    """Deletes a memory."""
    filepath = os.path.join(MEMORY_DIR, f"{key}.md")
    if os.path.exists(filepath):
        os.remove(filepath)
        return f"Deleted memory: {key}"
    return f"Memory not found: {key}"


# Tool definitions (to be added to TOOLS in Chapter 4)
MEMORY_TOOLS = [
    {
        "name": "save_memory",
        "description": "Save important information to long-term memory. Use for user preferences, key facts, project notes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Short label, e.g. 'user-preferences', 'project-notes'",
                },
                "content": {
                    "type": "string",
                    "description": "The information to remember",
                },
            },
            "required": ["key", "content"],
        },
    },
    {
        "name": "search_memory",
        "description": "Search long-term memory for relevant information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for",
                }
            },
            "required": ["query"],
        },
    },
]


def main():
    print("=" * 50)
    print("Chapter 10: Long-Term Memory")
    print("=" * 50)
    print()

    # Save demo memories
    print("Saving memories...\n")

    save_memory(
        "user-preferences",
        """# User Preferences

- Favorite restaurant: Euljiro Golbaengi House
- Coffee: Americano, no sugar
- Timezone: Asia/Seoul
- Language: Korean + English mix
- Work hours: 10am - 7pm
""",
    )
    print("  ✅ Saved: user-preferences")

    save_memory(
        "project-nanoclaw",
        """# NanoClaw Project Notes

## Architecture
- Single Node.js process
- WhatsApp via baileys
- SQLite for all data
- Agents in Apple Container / Docker

## Key Files
- src/index.ts — main loop
- src/container-runner.ts — agent execution
- src/db.ts — database
- src/group-queue.ts — concurrency

## TODO
- [ ] Add Telegram channel
- [ ] Implement vector search for memory
- [x] Per-group CLAUDE.md
""",
    )
    print("  ✅ Saved: project-nanoclaw")

    save_memory(
        "meeting-notes-2025-02",
        """# Meeting Notes - February 2025

## Feb 10 - Team Standup
- Discussed NanoClaw architecture rewrite
- Decided to use SQLite instead of JSONL
- Next: implement container isolation

## Feb 12 - Design Review
- Reviewed group queue design
- Approved per-group CLAUDE.md approach
- Action: write tutorial based on NanoClaw
""",
    )
    print("  ✅ Saved: meeting-notes-2025-02")

    # List memories
    print(f"\n📋 All memories:")
    for fname in list_memories():
        print(f"  - {fname}")

    # Test search
    print(f"\n🔍 Search: 'restaurant coffee'")
    result = search_memory("restaurant coffee")
    print(f"{result}\n")

    print(f"🔍 Search: 'SQLite architecture'")
    result = search_memory("SQLite architecture")
    print(f"{result}\n")

    print(f"🔍 Search: 'nonexistent topic'")
    result = search_memory("nonexistent topic xyz123")
    print(f"{result}\n")

    # Read a specific memory
    print(f"📖 Read: 'user-preferences'")
    content = read_memory("user-preferences")
    if content:
        print(content)

    print("\n" + "=" * 50)
    print("\nNanoClaw memory model:")
    print("  - Each group has its own isolated filesystem")
    print("  - Agent creates files naturally (notes, code, data)")
    print("  - CLAUDE.md = personality, regular files = knowledge")
    print("  - No special memory API needed — it's just files!")
    print()
    print("Our simplified version:")
    print("  - workspace/memory/*.md for key-value storage")
    print("  - Keyword search across all memory files")
    print("  - Tools: save_memory, search_memory")


if __name__ == "__main__":
    main()
