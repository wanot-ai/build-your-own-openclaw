#!/usr/bin/env python3
"""
NanoClaw Mini — Build Your Own AI Assistant
=============================================

A complete mini system implementing the NanoClaw architecture in Python.
All core NanoClaw features in approximately 500 lines.

Features:
  - Polling loop (NanoClaw's startMessageLoop)
  - SQLite database (messages, sessions, tasks, router state)
  - Per-group CLAUDE.md personality
  - Tool-using agent loop
  - Per-group concurrency control (GroupQueue)
  - Cron-based task scheduler
  - File-based long-term memory
  - Multi-channel: terminal + HTTP
  - IPC pattern for message routing

Run:
  uv run --with anthropic --with schedule python nanoclaw.py

Or without API key (echo mode):
  python nanoclaw.py
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

# Optional imports
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import schedule
    HAS_SCHEDULE = True
except ImportError:
    HAS_SCHEDULE = False

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE = os.path.join(BASE_DIR, "workspace")
DB_PATH = os.path.join(WORKSPACE, "nanoclaw.db")
MEMORY_DIR = os.path.join(WORKSPACE, "memory")
GROUPS_DIR = os.path.join(WORKSPACE, "groups")
POLL_INTERVAL = 2.0       # NanoClaw: 2000ms
SCHEDULER_INTERVAL = 10   # NanoClaw: 60000ms (10s for demo)
BOT_NAME = "Nano"
MAX_CONCURRENT = 3        # NanoClaw: 5
HTTP_PORT = 5555

DEFAULT_SOUL = """You are Nano, a helpful AI assistant.
- Be concise and direct
- Use tools when they help
- Save important facts to memory
- Feel free to use Korean comments naturally"""


# ============================================================
# Database (SQLite — NanoClaw's db.ts)
# ============================================================

_db_local = threading.local()


def get_db() -> sqlite3.Connection:
    """Per-thread DB connection. NanoClaw uses better-sqlite3."""
    if not hasattr(_db_local, "conn") or _db_local.conn is None:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        _db_local.conn = sqlite3.connect(DB_PATH)
        _db_local.conn.row_factory = sqlite3.Row
        _db_local.conn.execute("PRAGMA journal_mode=WAL")
    return _db_local.conn


def init_database():
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
            group_id TEXT PRIMARY KEY,
            messages TEXT DEFAULT '[]'
        );

        CREATE TABLE IF NOT EXISTS router_state (
            key TEXT PRIMARY KEY, value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY,
            group_id TEXT NOT NULL,
            chat_jid TEXT NOT NULL,
            prompt TEXT NOT NULL,
            schedule_type TEXT NOT NULL,
            schedule_value TEXT NOT NULL,
            next_run TEXT,
            last_run TEXT,
            last_result TEXT,
            status TEXT DEFAULT 'active',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_next ON scheduled_tasks(next_run);
    """)
    db.commit()


def store_message(msg: dict):
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO messages VALUES (?,?,?,?,?,?,?)",
        (msg["id"], msg["chat_jid"], msg["sender"], msg["sender_name"],
         msg["content"], msg["timestamp"], 1 if msg.get("is_from_me") else 0),
    )
    db.commit()


def get_new_messages(jids: list[str], since: str) -> tuple[list[dict], str]:
    if not jids:
        return [], since
    db = get_db()
    ph = ",".join("?" for _ in jids)
    rows = db.execute(
        f"""SELECT id, chat_jid, sender, sender_name, content, timestamp
            FROM messages WHERE timestamp > ? AND chat_jid IN ({ph})
            AND content NOT LIKE ? ORDER BY timestamp""",
        [since, *jids, f"{BOT_NAME}:%"],
    ).fetchall()
    msgs = [dict(r) for r in rows]
    ts = since
    for m in msgs:
        if m["timestamp"] > ts:
            ts = m["timestamp"]
    return msgs, ts


def get_messages_since(jid: str, since: str) -> list[dict]:
    db = get_db()
    return [dict(r) for r in db.execute(
        "SELECT * FROM messages WHERE chat_jid=? AND timestamp>? AND content NOT LIKE ? ORDER BY timestamp",
        (jid, since, f"{BOT_NAME}:%"),
    ).fetchall()]


def get_router_state(key: str) -> str | None:
    row = get_db().execute("SELECT value FROM router_state WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_router_state(key: str, value: str):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO router_state VALUES (?,?)", (key, value))
    db.commit()


def get_session(gid: str) -> list[dict]:
    row = get_db().execute("SELECT messages FROM sessions WHERE group_id=?", (gid,)).fetchone()
    return json.loads(row["messages"]) if row else []


def save_session(gid: str, msgs: list[dict]):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?)", (gid, json.dumps(msgs)))
    db.commit()


# ============================================================
# Message Format (NanoClaw's router.ts)
# ============================================================

def escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def format_messages(msgs: list[dict]) -> str:
    lines = [
        f'<message sender="{escape_xml(m["sender_name"])}" time="{m["timestamp"]}">{escape_xml(m["content"])}</message>'
        for m in msgs
    ]
    return "<messages>\n" + "\n".join(lines) + "\n</messages>"


# ============================================================
# Memory (File-based long-term memory)
# ============================================================

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
            txt = f.read()
        if any(w in txt.lower() for w in words):
            results.append(f"--- {fn} ---\n{txt}")
    return "\n\n".join(results) if results else "No matching memories."


# ============================================================
# 도구 + 에이전트 루프 (Agent loop with tools)
# ============================================================

TOOLS = [
    {"name": "run_command", "description": "Run a shell command in the workspace",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string", "description": "Shell command"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read a file from workspace",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string", "description": "Path relative to workspace"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "save_memory", "description": "Save important information to long-term memory",
     "input_schema": {"type": "object", "properties": {"key": {"type": "string", "description": "Short label"}, "content": {"type": "string", "description": "Info to remember"}}, "required": ["key", "content"]}},
    {"name": "search_memory", "description": "Search long-term memory",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
]


def execute_tool(name: str, inp: dict) -> str:
    try:
        if name == "run_command":
            r = subprocess.run(inp["command"], shell=True, capture_output=True, text=True, timeout=30, cwd=WORKSPACE)
            return (r.stdout + r.stderr).strip() or "(no output)"
        elif name == "read_file":
            with open(os.path.join(WORKSPACE, inp["path"])) as f:
                return f.read()
        elif name == "write_file":
            fp = os.path.join(WORKSPACE, inp["path"])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            with open(fp, "w") as f:
                f.write(inp["content"])
            return f"Written: {inp['path']}"
        elif name == "save_memory":
            return save_memory(inp["key"], inp["content"])
        elif name == "search_memory":
            return search_memory(inp["query"])
    except Exception as e:
        return f"Error: {e}"
    return f"Unknown tool: {name}"


def run_agent_turn(session_msgs: list[dict], system: str) -> tuple[str, list[dict]]:
    """에이전트 턴 실행. 도구 호출 루프 포함."""
    if not HAS_ANTHROPIC or not os.environ.get("ANTHROPIC_API_KEY"):
        # 에코 모드
        last = session_msgs[-1]["content"] if session_msgs else ""
        text = last if isinstance(last, str) else str(last)
        return f"[Echo] {text[:200]}", session_msgs

    client = anthropic.Anthropic()
    while True:
        resp = client.messages.create(
            model="claude-sonnet-4-5-20250929", max_tokens=4096,
            system=system, tools=TOOLS, messages=session_msgs,
        )
        content = []
        for b in resp.content:
            if hasattr(b, "text"):
                content.append({"type": "text", "text": b.text})
            elif b.type == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})

        if resp.stop_reason == "end_turn":
            session_msgs.append({"role": "assistant", "content": content})
            return "".join(b.text for b in resp.content if hasattr(b, "text")), session_msgs

        if resp.stop_reason == "tool_use":
            session_msgs.append({"role": "assistant", "content": content})
            results = []
            for b in resp.content:
                if b.type == "tool_use":
                    print(f"  🔧 {b.name}: {json.dumps(b.input)[:80]}")
                    r = execute_tool(b.name, b.input)
                    print(f"     → {r[:120]}")
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": str(r)})
            session_msgs.append({"role": "user", "content": results})


# ============================================================
# 그룹 성격 (Per-group CLAUDE.md — NanoClaw's groups/{folder}/CLAUDE.md)
# ============================================================

def get_soul(group_id: str) -> str:
    # 그룹별 CLAUDE.md
    group_soul = os.path.join(GROUPS_DIR, group_id, "CLAUDE.md")
    if os.path.exists(group_soul):
        with open(group_soul) as f:
            return f.read()
    # 글로벌 SOUL.md
    global_soul = os.path.join(BASE_DIR, "SOUL.md")
    if os.path.exists(global_soul):
        with open(global_soul) as f:
            return f.read()
    return DEFAULT_SOUL


# ============================================================
# 그룹 큐 (Per-group concurrency — NanoClaw's GroupQueue)
# ============================================================

class GroupQueue:
    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self._locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._active = 0
        self._max = max_concurrent
        self._global_lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)

    def process(self, group_id: str, fn):
        """그룹별 락 + 전역 동시성 제한."""
        def worker():
            self._semaphore.acquire()
            with self._global_lock:
                self._active += 1
            try:
                with self._locks[group_id]:
                    fn()
            finally:
                with self._global_lock:
                    self._active -= 1
                self._semaphore.release()
        threading.Thread(target=worker, daemon=True).start()


# ============================================================
# 태스크 스케줄러 (Cron tasks — NanoClaw's task-scheduler.ts)
# ============================================================

def get_due_tasks() -> list[dict]:
    now = datetime.now().isoformat()
    rows = get_db().execute(
        "SELECT * FROM scheduled_tasks WHERE status='active' AND next_run IS NOT NULL AND next_run<=? ORDER BY next_run",
        (now,),
    ).fetchall()
    return [dict(r) for r in rows]


def create_task(task_id: str, group_id: str, chat_jid: str, prompt: str,
                stype: str, svalue: str, next_run: str | None = None):
    db = get_db()
    db.execute(
        "INSERT OR REPLACE INTO scheduled_tasks (id,group_id,chat_jid,prompt,schedule_type,schedule_value,next_run,status,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (task_id, group_id, chat_jid, prompt, stype, svalue, next_run, "active", datetime.now().isoformat()),
    )
    db.commit()


def update_task_after_run(task_id: str, next_run: str | None, result: str):
    db = get_db()
    now = datetime.now().isoformat()
    db.execute(
        "UPDATE scheduled_tasks SET next_run=?, last_run=?, last_result=?, status=CASE WHEN ? IS NULL THEN 'completed' ELSE status END WHERE id=?",
        (next_run, now, result, next_run, task_id),
    )
    db.commit()


def run_scheduled_task(task: dict, send_fn):
    """스케줄된 태스크 실행."""
    print(f"\n  ⏰ Running task: {task['id']}")
    soul = get_soul(task["group_id"])
    session = get_session(f"task:{task['id']}")
    session.append({"role": "user", "content": task["prompt"]})

    try:
        reply, session = run_agent_turn(session, soul)
        save_session(f"task:{task['id']}", session)
        send_fn(task["chat_jid"], f"{BOT_NAME}: {reply}")

        # 다음 실행 계산
        next_run = None
        if task["schedule_type"] == "interval":
            ms = int(task["schedule_value"])
            next_run = (datetime.now() + timedelta(milliseconds=ms)).isoformat()
        update_task_after_run(task["id"], next_run, reply[:200])
    except Exception as e:
        print(f"  ❌ Task error: {e}")
        update_task_after_run(task["id"], None, f"Error: {e}")


def start_scheduler(queue: GroupQueue, send_fn):
    """스케줄러 폴링 루프."""
    def loop():
        while True:
            try:
                due = get_due_tasks()
                for task in due:
                    queue.process(task["group_id"], lambda t=task: run_scheduled_task(t, send_fn))
            except Exception as e:
                print(f"  Scheduler error: {e}")
            time.sleep(SCHEDULER_INTERVAL)
    threading.Thread(target=loop, daemon=True).start()


# ============================================================
# 채널 (Terminal + HTTP — NanoClaw's Channel interface)
# ============================================================

class Gateway:
    def __init__(self):
        self._messages: list[dict] = []
        self._lock = threading.Lock()
        self._send_fns: dict[str, callable] = {}  # jid_suffix → send_fn

    def on_message(self, msg: dict):
        store_message(msg)
        with self._lock:
            self._messages.append(msg)

    def poll(self) -> list[dict]:
        with self._lock:
            msgs = self._messages.copy()
            self._messages.clear()
        return msgs

    def register_sender(self, jid_suffix: str, fn):
        self._send_fns[jid_suffix] = fn

    def send(self, jid: str, text: str):
        for suffix, fn in self._send_fns.items():
            if jid.endswith(suffix):
                fn(jid, text)
                return
        print(f"  ⚠️ No channel for: {jid}")


def start_terminal(gateway: Gateway):
    """터미널 입력 스레드."""
    def reader():
        while True:
            try:
                line = input()
                if not line.strip():
                    continue
                if line.strip().startswith("/"):
                    handle_command(line.strip(), gateway)
                    continue
                gateway.on_message({
                    "id": f"term_{int(time.time()*1000)}",
                    "chat_jid": "terminal@local",
                    "sender": "user",
                    "sender_name": "You",
                    "content": line.strip(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            except EOFError:
                break
    threading.Thread(target=reader, daemon=True).start()
    gateway.register_sender("@local", lambda jid, text: print(f"\n🤖 {text}\n"))


def start_http(gateway: Gateway, port: int = HTTP_PORT):
    """HTTP API 채널."""
    responses: dict[str, str] = {}
    events: dict[str, threading.Event] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/chat":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            jid = f"http_{body.get('user_id', 'anon')}@http"
            evt = threading.Event()
            events[jid] = evt
            gateway.on_message({
                "id": f"http_{int(time.time()*1000)}",
                "chat_jid": jid,
                "sender": body.get("user_id", "anon"),
                "sender_name": body.get("user_name", "HTTP User"),
                "content": body.get("message", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
            evt.wait(timeout=60)
            resp = responses.pop(jid, "No response")
            events.pop(jid, None)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"response": resp}).encode())

        def log_message(self, *args):
            pass

    def send_http(jid: str, text: str):
        responses[jid] = text
        evt = events.get(jid)
        if evt:
            evt.set()

    server = HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    gateway.register_sender("@http", send_http)
    return server


# ============================================================
# 커맨드 (Slash commands)
# ============================================================

def handle_command(cmd: str, gateway: Gateway):
    parts = cmd.split(maxsplit=1)
    verb = parts[0].lower()

    if verb == "/new":
        save_session("terminal", [])
        print("🔄 Session reset.\n")
    elif verb == "/memory":
        if os.path.exists(MEMORY_DIR):
            files = [f for f in os.listdir(MEMORY_DIR) if f.endswith(".md")]
            print(f"📋 Memories: {', '.join(files) if files else '(none)'}\n")
        else:
            print("📋 No memories yet.\n")
    elif verb == "/tasks":
        rows = get_db().execute("SELECT id, status, next_run, prompt FROM scheduled_tasks ORDER BY created_at DESC").fetchall()
        if rows:
            for r in rows:
                r = dict(r)
                print(f"  {r['id']} [{r['status']}] next={r['next_run']} — {r['prompt'][:50]}")
        else:
            print("  No tasks.")
        print()
    elif verb == "/schedule" and len(parts) > 1:
        # /schedule 60 <prompt>  — 매 60초마다 실행
        rest = parts[1].split(maxsplit=1)
        if len(rest) >= 2:
            interval_s = int(rest[0])
            prompt = rest[1]
            tid = f"task_{int(time.time())}"
            next_run = (datetime.now() + timedelta(seconds=interval_s)).isoformat()
            create_task(tid, "terminal", "terminal@local", prompt, "interval", str(interval_s * 1000), next_run)
            print(f"✅ Task '{tid}' created: every {interval_s}s\n")
        else:
            print("Usage: /schedule <seconds> <prompt>\n")
    elif verb == "/help":
        print("Commands:")
        print("  /new       — Reset session")
        print("  /memory    — List memories")
        print("  /tasks     — List scheduled tasks")
        print("  /schedule <sec> <prompt> — Create recurring task")
        print("  /quit      — Exit")
        print()
    elif verb == "/quit":
        sys.exit(0)
    else:
        print(f"Unknown command: {verb}. Type /help\n")


# ============================================================
# 메인 폴링 루프 (NanoClaw's startMessageLoop)
# ============================================================

def main():
    print()
    print("  ╔══════════════════════════════════════╗")
    print("  ║          NanoClaw Mini 🦞             ║")
    print("  ║  Build Your Own AI Assistant          ║")
    print("  ╚══════════════════════════════════════╝")
    print()

    # 초기화
    os.makedirs(WORKSPACE, exist_ok=True)
    os.makedirs(MEMORY_DIR, exist_ok=True)
    init_database()

    gateway = Gateway()
    queue = GroupQueue()

    # 채널 시작
    start_terminal(gateway)
    try:
        start_http(gateway)
        print(f"  Channels: terminal, HTTP (:{HTTP_PORT})")
    except OSError:
        print(f"  Channels: terminal (HTTP port {HTTP_PORT} in use)")

    has_api = HAS_ANTHROPIC and os.environ.get("ANTHROPIC_API_KEY")
    if not has_api:
        print("  Mode: echo (set ANTHROPIC_API_KEY for Claude)")
    else:
        print("  Mode: Claude (claude-sonnet-4-5-20250929)")

    # 스케줄러 시작
    start_scheduler(queue, gateway.send)

    print()
    print("  Type a message, or /help for commands.")
    print("  Ctrl+C to quit.\n")

    # 세션 상태
    sessions_cache: dict[str, list[dict]] = {}
    last_agent_ts: dict[str, str] = {}

    # 전역 타임스탬프
    global_ts = get_router_state("last_timestamp") or ""

    try:
        while True:
            msgs = gateway.poll()

            if msgs:
                # 그룹별로 분류
                by_group: dict[str, list[dict]] = defaultdict(list)
                for m in msgs:
                    by_group[m["chat_jid"]].append(m)

                for jid, group_msgs in by_group.items():
                    group_id = jid.split("@")[0]

                    def process(gid=group_id, gmsgs=group_msgs, gjid=jid):
                        soul = get_soul(gid)
                        prompt = format_messages(gmsgs)

                        # 세션 로드
                        if gid not in sessions_cache:
                            sessions_cache[gid] = get_session(gid)
                        session = sessions_cache[gid]
                        session.append({"role": "user", "content": prompt})

                        try:
                            reply, session = run_agent_turn(session, soul)
                            sessions_cache[gid] = session
                            save_session(gid, session)
                            gateway.send(gjid, f"{BOT_NAME}: {reply}")
                        except Exception as e:
                            print(f"\n  ❌ Error: {e}\n")

                    queue.process(group_id, process)

                # 타임스탬프 갱신
                for m in msgs:
                    if m["timestamp"] > global_ts:
                        global_ts = m["timestamp"]
                set_router_state("last_timestamp", global_ts)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n\n  👋 Bye! Memories saved in workspace/\n")
        set_router_state("last_timestamp", global_ts)


if __name__ == "__main__":
    main()
