#!/usr/bin/env python3
"""
Chapter 2: SQLite Database
==========================

NanoClaw uses SQLite instead of JSONL files.

The original "Build Your Own OpenClaw" used JSONL files for sessions.
NanoClaw uses SQLite for everything: messages, sessions, router state,
scheduled tasks, registered groups.

Why SQLite?
- Atomic writes (no corruption on crash)
- Indexed queries (fast lookups by timestamp, chat_jid)
- Single file (easy to backup, move, inspect)
- Built into Python (no dependencies)

This chapter builds the database layer that stores messages and sessions.

Run: uv run python chapters/02_database.py
"""

import os
import sqlite3
import time
import threading

DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "workspace", "messages.db"
)

# Thread-local connections (SQLite connections aren't thread-safe)
_local = threading.local()


def get_db() -> sqlite3.Connection:
    """Gets a per-thread DB connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        if DB_PATH != ":memory:":
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
        if DB_PATH != ":memory:":
            _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn


def init_database():
    """
    Initialize database schema. Equivalent to NanoClaw's createSchema().

    NanoClaw (TypeScript):
        database.exec(`
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT, chat_jid TEXT, sender TEXT, sender_name TEXT,
                content TEXT, timestamp TEXT, is_from_me INTEGER,
                PRIMARY KEY (id, chat_jid)
            );
            CREATE TABLE IF NOT EXISTS router_state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS sessions (group_folder TEXT PRIMARY KEY, session_id TEXT NOT NULL);
        `);
    """
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT,
            chat_jid TEXT,
            sender TEXT,
            sender_name TEXT,
            content TEXT,
            timestamp TEXT,
            is_from_me INTEGER DEFAULT 0,
            PRIMARY KEY (id, chat_jid)
        );
        CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_jid, timestamp);

        CREATE TABLE IF NOT EXISTS sessions (
            group_id TEXT PRIMARY KEY,
            messages TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS router_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS groups (
            group_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            folder TEXT NOT NULL UNIQUE,
            soul_path TEXT,
            added_at TEXT NOT NULL
        );
    """)
    db.commit()


def store_message(msg: dict):
    """
    Stores a message. Equivalent to NanoClaw's storeMessage().

    NanoClaw (TypeScript):
        db.prepare(`INSERT OR REPLACE INTO messages (...) VALUES (?, ?, ?, ?, ?, ?, ?)`)
          .run(msg.id, msg.chat_jid, msg.sender, msg.sender_name, msg.content, msg.timestamp, msg.is_from_me ? 1 : 0);
    """
    db = get_db()
    db.execute(
        """INSERT OR REPLACE INTO messages (id, chat_jid, sender, sender_name, content, timestamp, is_from_me)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            msg["id"],
            msg["chat_jid"],
            msg["sender"],
            msg["sender_name"],
            msg["content"],
            msg["timestamp"],
            1 if msg.get("is_from_me") else 0,
        ),
    )
    db.commit()


def get_new_messages(
    chat_jids: list[str], since_timestamp: str, bot_name: str
) -> tuple[list[dict], str]:
    """
    Retrieves new messages. Equivalent to NanoClaw's getNewMessages().

    NanoClaw (TypeScript):
        const sql = `SELECT ... FROM messages
            WHERE timestamp > ? AND chat_jid IN (...) AND content NOT LIKE ?
            ORDER BY timestamp`;
    """
    if not chat_jids:
        return [], since_timestamp

    db = get_db()
    placeholders = ",".join("?" for _ in chat_jids)
    rows = db.execute(
        f"""SELECT id, chat_jid, sender, sender_name, content, timestamp
            FROM messages
            WHERE timestamp > ? AND chat_jid IN ({placeholders})
              AND content NOT LIKE ?
            ORDER BY timestamp""",
        [since_timestamp, *chat_jids, f"{bot_name}:%"],
    ).fetchall()

    messages = [dict(r) for r in rows]
    new_ts = since_timestamp
    for m in messages:
        if m["timestamp"] > new_ts:
            new_ts = m["timestamp"]

    return messages, new_ts


def get_messages_since(
    chat_jid: str, since_timestamp: str, bot_name: str
) -> list[dict]:
    """Retrieves new messages for a specific group. Equivalent to NanoClaw's getMessagesSince()."""
    db = get_db()
    rows = db.execute(
        """SELECT id, chat_jid, sender, sender_name, content, timestamp
           FROM messages
           WHERE chat_jid = ? AND timestamp > ? AND content NOT LIKE ?
           ORDER BY timestamp""",
        (chat_jid, since_timestamp, f"{bot_name}:%"),
    ).fetchall()
    return [dict(r) for r in rows]


def get_router_state(key: str) -> str | None:
    """Gets the router state."""
    db = get_db()
    row = db.execute(
        "SELECT value FROM router_state WHERE key = ?", (key,)
    ).fetchone()
    return row["value"] if row else None


def set_router_state(key: str, value: str):
    """Saves the router state."""
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO router_state (key, value) VALUES (?, ?)",
        (key, value),
    )
    db.commit()


def get_session_messages(group_id: str) -> list[dict]:
    """Gets the session's conversation history."""
    import json

    db = get_db()
    row = db.execute(
        "SELECT messages FROM sessions WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row:
        return json.loads(row["messages"])
    return []


def save_session_messages(group_id: str, messages: list[dict]):
    """Saves the session's conversation history."""
    import json

    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO sessions (group_id, messages) VALUES (?, ?)",
        (group_id, json.dumps(messages)),
    )
    db.commit()


# --- Demo ---


def main():
    print("=" * 50)
    print("Chapter 2: SQLite Database")
    print("=" * 50)
    print()

    # Test with an in-memory DB
    global DB_PATH
    DB_PATH = ":memory:"
    _local.conn = None

    init_database()
    print("✅ Database initialized\n")

    # Test storing messages
    messages = [
        {
            "id": "msg_001",
            "chat_jid": "group1@local",
            "sender": "user1",
            "sender_name": "Alice",
            "content": "Hello everyone!",
            "timestamp": "2025-01-01T09:00:00",
        },
        {
            "id": "msg_002",
            "chat_jid": "group1@local",
            "sender": "user2",
            "sender_name": "Bob",
            "content": "@Nano what's the weather?",
            "timestamp": "2025-01-01T09:00:05",
        },
        {
            "id": "msg_003",
            "chat_jid": "group2@local",
            "sender": "user1",
            "sender_name": "Alice",
            "content": "Different group message",
            "timestamp": "2025-01-01T09:00:10",
        },
    ]

    for msg in messages:
        store_message(msg)
    print(f"✅ Stored {len(messages)} messages\n")

    # Test retrieving new messages
    new_msgs, new_ts = get_new_messages(
        ["group1@local", "group2@local"],
        "2025-01-01T08:59:59",
        "Nano",
    )
    print(f"📨 New messages since 08:59:59: {len(new_msgs)}")
    for m in new_msgs:
        print(f"   [{m['sender_name']}] {m['content']}")

    # Retrieve messages for a specific group
    group1_msgs = get_messages_since("group1@local", "2025-01-01T08:59:59", "Nano")
    print(f"\n📨 Group 1 messages: {len(group1_msgs)}")

    # Test router state
    set_router_state("last_timestamp", "2025-01-01T09:00:10")
    ts = get_router_state("last_timestamp")
    print(f"\n🔄 Router state - last_timestamp: {ts}")

    # Test sessions
    save_session_messages(
        "group1",
        [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ],
    )
    session = get_session_messages("group1")
    print(f"\n💾 Session for group1: {len(session)} messages")
    for m in session:
        print(f"   [{m['role']}] {m['content']}")

    print("\n✅ All database operations working!")
    print("\nCompare with NanoClaw's db.ts:")
    print("  - storeMessage() → store_message()")
    print("  - getNewMessages() → get_new_messages()")
    print("  - getMessagesSince() → get_messages_since()")
    print("  - getRouterState() / setRouterState() → get/set_router_state()")
    print("  - getAllSessions() → get/save_session_messages()")


if __name__ == "__main__":
    main()
