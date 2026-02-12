#!/usr/bin/env python3
"""
Chapter 11: Full System
========================

Everything combined: a complete implementation of the NanoClaw architecture.

This chapter combines all previous concepts into a working system:
- Polling loop (Ch 1)
- SQLite database (Ch 2)
- Per-group personality (Ch 3)
- Tool-using agent (Ch 4)
- Process isolation (Ch 5)
- Group queue (Ch 6)
- Task scheduler (Ch 7)
- IPC (Ch 8)
- Multi-channel gateway (Ch 9)
- Long-term memory (Ch 10)

For the complete standalone version, see nanoclaw.py in the project root.

Run: uv run --with anthropic python chapters/11_full_system.py
"""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler

import anthropic

# --- Configuration ---

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "workspace")
DB_PATH = os.path.join(WORKSPACE, "system.db")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
GROUPS_DIR = os.path.join(WORKSPACE, "groups")
POLL_INTERVAL = 2.0
BOT_NAME = "Nano"
DEFAULT_SOUL = "You are Nano, a helpful AI assistant. Be concise and direct."


# --- Database (Ch 2) ---

_local = threading.local()


def get_db() -> sqlite3.Connection:
    if not hasattr(_local, "conn") or _local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT, chat_jid TEXT, sender TEXT, sender_name TEXT,
            content TEXT, timestamp TEXT, is_from_me INTEGER DEFAULT 0,
            PRIMARY KEY (id, chat_jid)
        );
        CREATE INDEX IF NOT EXISTS idx_msg_ts ON messages(timestamp);
        CREATE INDEX IF NOT EXISTS idx_msg_chat ON messages(chat_jid, timestamp);
        CREATE TABLE IF NOT EXISTS sessions (
            group_id TEXT PRIMARY KEY, messages TEXT DEFAULT '[]'
        );
        CREATE TABLE IF NOT EXISTS router_state (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY, group_id TEXT NOT NULL, prompt TEXT NOT NULL,
            schedule_type TEXT NOT NULL, schedule_value TEXT NOT NULL,
            next_run TEXT, last_run TEXT, status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL
        );
    """)
    db.commit()


def store_message(msg: dict):
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO messages (id,chat_jid,sender,sender_name,content,timestamp,is_from_me) VALUES (?,?,?,?,?,?,?)",
        (msg["id"], msg["chat_jid"], msg["sender"], msg["sender_name"],
         msg["content"], msg["timestamp"], 1 if msg.get("is_from_me") else 0),
    )
    db.commit()


def get_new_messages(chat_jids: list[str], since: str) -> tuple[list[dict], str]:
    if not chat_jids:
        return [], since
    db = get_db()
    ph = ",".join("?" for _ in chat_jids)
    rows = db.execute(
        f"SELECT * FROM messages WHERE timestamp > ? AND chat_jid IN ({ph}) AND content NOT LIKE ? ORDER BY timestamp",
        [since, *chat_jids, f"{BOT_NAME}:%"],
    ).fetchall()
    msgs = [dict(r) for r in rows]
    new_ts = since
    for m in msgs:
        if m["timestamp"] > new_ts:
            new_ts = m["timestamp"]
    return msgs, new_ts


# --- Memory (Ch 10) ---

def save_memory(key: str, content: str) -> str:
    os.makedirs(MEMORY_DIR, exist_ok=True)
    with open(os.path.join(MEMORY_DIR, f"{key}.md"), "w") as f:
        f.write(content)
    return f"Saved: {key}"


def search_memory(query: str) -> str:
    if not os.path.exists(MEMORY_DIR):
        return "No memories."
    words = query.lower().split()
    results = []
    for fn in sorted(os.listdir(MEMORY_DIR)):
        if not fn.endswith(".md"):
            continue
        with open(os.path.join(MEMORY_DIR, fn)) as f:
            content = f.read()
        if any(w in content.lower() for w in words):
            results.append(f"--- {fn} ---\n{content}")
    return "\n\n".join(results) if results else "No matching memories."


# --- Tools + Agent Loop (Ch 4) ---

TOOLS = [
    {"name": "run_command", "description": "Run a shell command",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read a file from workspace",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write a file to workspace",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "save_memory", "description": "Save to long-term memory",
     "input_schema": {"type": "object", "properties": {"key": {"type": "string"}, "content": {"type": "string"}}, "required": ["key", "content"]}},
    {"name": "search_memory", "description": "Search long-term memory",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def execute_tool(name: str, inp: dict) -> str:
    if name == "run_command":
        try:
            r = subprocess.run(inp["command"], shell=True, capture_output=True, text=True, timeout=30, cwd=WORKSPACE)
            return (r.stdout + r.stderr).strip() or "(no output)"
        except Exception as e:
            return f"Error: {e}"
    elif name == "read_file":
        try:
            with open(os.path.join(WORKSPACE, inp["path"])) as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"
    elif name == "write_file":
        try:
            fp = os.path.join(WORKSPACE, inp["path"])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w") as f:
                f.write(inp["content"])
            return f"Written: {inp['path']}"
        except Exception as e:
            return f"Error: {e}"
    elif name == "save_memory":
        return save_memory(inp["key"], inp["content"])
    elif name == "search_memory":
        return search_memory(inp["query"])
    return f"Unknown tool: {name}"


def run_agent(messages: list[dict], system: str) -> tuple[str, list[dict]]:
    client = anthropic.Anthropic()
    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929", max_tokens=4096,
            system=system, tools=TOOLS, messages=messages,
        )
        content = []
        for b in resp.content:
            if hasattr(b, "text"):
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})

        if resp.stop_reason == "end_turn":
            messages.append({"role": "assistant", "content": content})
            return "".join(b.text for b in resp.content if hasattr(b, "text")), messages

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    print(f"  🔧 {b.name}: {json.dumps(b.input)[:80]}")
                    r = execute_tool(b.name, b.input)
                    print(f"     → {r[:100]}")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(r)})
            messages.append({"role": "user", "content": results})


# --- Group Personality (Ch 3) ---

def get_soul(group_id: str) -> str:
    soul_path = os.path.join(GROUPS_DIR, group_id, "CLAUDE.md")
    if os.path.exists(soul_path):
        with open(soul_path) as f:
            return f.read()
    # Check global SOUL.md
    global_soul = os.path.join(os.path.dirname(__file__), "..", "SOUL.md")
    if os.path.exists(global_soul):
        with open(global_soul) as f:
            return f.read()
    return DEFAULT_SOUL


# --- Group Queue (Ch 6, simplified) ---

group_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


# --- Message Format (Ch 1) ---

def format_messages(msgs: list[dict]) -> str:
    lines = []
    for m in msgs:
        s = m["sender_name"].replace("&", "&amp;").replace("<", "&lt;")
        c = m["content"].replace("&", "&amp;").replace("<", "&lt;")
        lines.append(f'<message sender="{s}" time="{m["timestamp"]}">{c}</message>')
    return "<messages>\n" + "\n".join(lines) + "\n</messages>"


# --- Gateway (Ch 9, simplified) ---

incoming_messages: list[dict] = []
msg_lock = threading.Lock()


def on_message(msg: dict):
    store_message(msg)
    with msg_lock:
        incoming_messages.append(msg)


# --- Main ---

def main():
    print("=" * 50)
    print("Chapter 11: Full System")
    print("=" * 50)
    print()

    os.makedirs(WORKSPACE, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    init_db()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Set ANTHROPIC_API_KEY for full agent functionality.")
        print("   Running in echo mode.\n")
        use_echo = True
    else:
        use_echo = False

    # Terminal input thread
    def stdin_reader():
        while True:
            try:
                line = input()
                if line.strip():
                    on_message({
                        "id": f"msg_{int(time.time()*1000)}",
                        "chat_jid": "terminal@local",
                        "sender": "user", "sender_name": "You",
                        "content": line.strip(),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    })
            except EOFError:
                break

    threading.Thread(target=stdin_reader, daemon=True).start()

    print("Full NanoClaw-style system running.")
    print("Type messages below. Ctrl+C to quit.\n")

    last_ts = ""
    session_messages: list[dict] = []

    try:
        while True:
            with msg_lock:
                msgs = incoming_messages.copy()
                incoming_messages.clear()

            if msgs:
                for m in msgs:
                    if m["timestamp"] > last_ts:
                        last_ts = m["timestamp"]

                prompt = format_messages(msgs)
                group_id = msgs[0]["chat_jid"].split("@")[0]
                soul = get_soul(group_id)

                with group_locks[group_id]:
                    if use_echo:
                        print(f"\n🤖 {BOT_NAME}: [Echo] {msgs[-1]['content']}\n")
                    else:
                        session_messages.append({"role": "user", "content": prompt})
                        try:
                            reply, session_messages = run_agent(session_messages, soul)
                            print(f"\n🤖 {BOT_NAME}: {reply}\n")
                        except Exception as e:
                            print(f"\n❌ Error: {e}\n")

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\n👋 Bye!")


if __name__ == "__main__":
    main()
