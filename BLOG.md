# Build Your Own AI Assistant from Scratch: A Deep Dive into NanoClaw's Architecture

*A comprehensive guide to building a production-grade, multi-channel AI assistant in Python — inspired by NanoClaw, the minimal TypeScript alternative to OpenClaw.*

---

## Introduction

There's a moment every developer hits when tinkering with AI APIs. You've got a script that calls Claude or GPT. It works. You type something in, you get something back. It feels like magic. And then you think: *"What if this could run all the time? What if it could talk to my group chats? What if it remembered things? What if it could use tools?"*

And suddenly you're not building a script anymore. You're building an **AI assistant**.

This is the story of how to build one from scratch. Not a toy. Not a weekend hack that falls over when two people message it at the same time. A real, production-grade system with message queuing, database persistence, per-group personalities, tool execution, sandboxed code isolation, scheduled tasks, and multi-channel support.

We're going to build it in Python, but our north star is [NanoClaw](https://github.com/qwibitai/nanoclaw) — a beautifully minimal TypeScript project that strips the full-featured [OpenClaw](https://openclaw.ai) platform down to its architectural essence. NanoClaw is roughly 2,000 lines of TypeScript. It's small enough to read in an afternoon, but it contains every major pattern you need to understand how a real AI assistant works under the hood.

OpenClaw itself is a beast — a full platform with Discord integration, WhatsApp bridging, browser control, node pairing, canvas rendering, and more. You could spend weeks exploring it. NanoClaw takes the same architectural DNA and distills it into something you can hold in your head. Our job is to take those patterns and rebuild them in Python, explaining every decision along the way.

By the end of this post, you'll understand:

- Why polling beats webhooks for this kind of system
- How to store conversations in SQLite (and why not JSONL)
- How to give each group chat its own personality
- How the agent loop actually works (it's a while loop, not magic)
- Why you need container isolation and how to fake it with subprocesses
- How to prevent race conditions with per-group queuing
- How to build a cron-like task scheduler
- How file-based IPC works between sandboxed processes
- How to serve one agent across multiple channels
- How to give your agent long-term memory
- How to wire it all together

Let's build.

---

## Chapter 1: The Polling Loop — Your Assistant's Heartbeat

### The Problem

The first decision you face when building any messaging integration is: **push or pull?**

Webhooks (push) are the "proper" way. The messaging platform sends you an HTTP request whenever something happens. It's efficient, real-time, and what every API documentation recommends.

Polling (pull) is the "lazy" way. You ask the platform "got anything new?" every few seconds. It's wasteful in theory — most of those requests return nothing.

So why does NanoClaw use polling? Why should we?

Because **polling is operationally simpler by an order of magnitude**.

With webhooks, you need:
- A publicly accessible HTTPS endpoint
- SSL certificates
- A way to handle webhook registration and verification
- Retry logic for when your server is down
- Idempotency handling for duplicate deliveries
- A web server framework

With polling, you need:
- A while loop
- `time.sleep()`

That's it. No public IP. No SSL. No web framework. Your assistant can run on a Raspberry Pi behind three layers of NAT and it works perfectly. When it crashes, you restart it and it picks up where it left off — it just polls for messages it missed.

NanoClaw polls its WhatsApp bridge (whatsapp-web.js via Baileys) every 2 seconds. It's simple, reliable, and surprisingly performant. Two seconds of latency is imperceptible in a chat context — humans take longer than that to notice a typing indicator.

### The Code

Here's the skeleton of our polling loop:

```python
import time
import threading
from queue import Queue

message_queue = Queue()

def poll_messages():
    """Poll the messaging backend for new messages."""
    # This could be WhatsApp Web, Discord API, Telegram, whatever
    response = requests.get(f"{BRIDGE_URL}/messages", params={
        "since": get_last_seen_timestamp()
    })
    if response.status_code == 200:
        return response.json().get("messages", [])
    return []

def format_messages(messages):
    """Convert raw messages into the format our agent expects."""
    formatted = []
    for msg in messages:
        formatted.append({
            "role": "user",
            "content": f"[{msg['sender']}]: {msg['content']}"
        })
    return formatted

def send_reply(chat_id, reply_text):
    """Send a reply back through the messaging bridge."""
    requests.post(f"{BRIDGE_URL}/send", json={
        "chat_id": chat_id,
        "message": reply_text
    })

def polling_loop():
    """Main polling loop — the heartbeat of our assistant."""
    while True:
        messages = poll_messages()
        if messages:
            reply = send_to_agent(format_messages(messages))
            send_reply(reply)
        time.sleep(2.0)

def main():
    polling_thread = threading.Thread(target=polling_loop, daemon=True)
    polling_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
```

### How NanoClaw Does It

NanoClaw's polling loop lives in its WhatsApp bridge integration. Every 2 seconds, it checks for new messages from the whatsapp-web.js backend. When messages arrive, they're batched, deduplicated against what's already been processed (using timestamps stored in the database), and then dispatched to the agent.

The key architectural choice is that the polling loop is **the only entry point for external messages**. Everything flows through this single funnel. This makes the system easy to reason about — there's exactly one place where new work enters the system.

NanoClaw also implements a clever optimization: message batching. If someone sends three messages in quick succession ("hey", "can you", "look at this"), the polling loop catches all three in a single poll and sends them to the agent as one batch. This saves API calls and produces more coherent responses since the agent sees the full context.

### Key Insight

> **Polling isn't lazy engineering — it's resilient engineering.** In a system that needs to run unattended for weeks, the simplest approach wins. A polling loop has exactly one failure mode (the poll fails), and exactly one recovery strategy (try again in 2 seconds). Compare that to a webhook setup where you need to handle registration failures, missed events, duplicate deliveries, and server downtime — all while maintaining a public endpoint.

The 2-second sleep isn't magic either. It's a sweet spot. One second feels aggressive and wastes CPU and API quota. Five seconds feels sluggish in a chat context. Two seconds means your assistant responds within 2-4 seconds of receiving a message, which feels natural — like a human who's multitasking and glances at their phone.

---

## Chapter 2: The SQLite Database — Your Assistant's Memory Bank

### The Problem

Once messages start flowing through your polling loop, you need to store them. This isn't optional. You need storage for:

1. **Deduplication** — Don't process the same message twice after a restart
2. **Context** — Feed conversation history to the LLM
3. **Sessions** — Track which conversations are active
4. **Audit** — Know what happened and when

The naive approach is to append messages to a JSON file (JSONL — one JSON object per line). It's simple, human-readable, and works great until it doesn't. JSONL falls apart when:

- You need to query by group or timestamp (you have to scan the entire file)
- Two threads write simultaneously (file corruption)
- The file gets large (reading 100MB of JSONL on every request)
- You need to update or delete records

SQLite solves all of these problems with zero infrastructure. It's a single file on disk, it supports concurrent reads, it handles locking internally, and it gives you the full power of SQL for queries. Python's `sqlite3` module is built-in — no pip install required.

### The Code

```python
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DATABASE_PATH = "assistant.db"

def init_database():
    """Initialize the database schema."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_jid TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL,
                is_from_me INTEGER DEFAULT 0,
                processed INTEGER DEFAULT 0
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                group_id TEXT PRIMARY KEY,
                last_timestamp REAL NOT NULL,
                last_active REAL NOT NULL,
                message_count INTEGER DEFAULT 0
            )
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_chat_jid 
            ON messages(chat_jid, timestamp)
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_processed 
            ON messages(processed, timestamp)
        """)
        db.commit()

@contextmanager
def get_db():
    """Context manager for database connections."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better concurrency
    try:
        yield conn
    finally:
        conn.close()

def store_message(chat_jid, sender, content, timestamp, is_from_me=False):
    """Store a message in the database."""
    with get_db() as db:
        db.execute(
            "INSERT INTO messages (chat_jid, sender, content, timestamp, is_from_me) VALUES (?, ?, ?, ?, ?)",
            (chat_jid, sender, content, timestamp, int(is_from_me))
        )
        db.execute("""
            INSERT INTO sessions (group_id, last_timestamp, last_active, message_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(group_id) DO UPDATE SET
                last_timestamp = MAX(sessions.last_timestamp, excluded.last_timestamp),
                last_active = excluded.last_active,
                message_count = sessions.message_count + 1
        """, (chat_jid, timestamp, datetime.now().timestamp()))
        db.commit()

def get_conversation_history(chat_jid, limit=50):
    """Retrieve recent conversation history for a chat."""
    with get_db() as db:
        rows = db.execute(
            "SELECT sender, content, timestamp, is_from_me FROM messages WHERE chat_jid = ? ORDER BY timestamp DESC LIMIT ?",
            (chat_jid, limit)
        ).fetchall()
        # Return in chronological order
        return list(reversed(rows))

def get_unprocessed_messages():
    """Get messages that haven't been processed yet."""
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM messages WHERE processed = 0 ORDER BY timestamp ASC"
        ).fetchall()
        return rows

def mark_processed(message_ids):
    """Mark messages as processed."""
    with get_db() as db:
        placeholders = ",".join("?" * len(message_ids))
        db.execute(
            f"UPDATE messages SET processed = 1 WHERE id IN ({placeholders})",
            message_ids
        )
        db.commit()
```

### How NanoClaw Does It

NanoClaw uses SQLite as its primary data store, with a schema remarkably similar to what we've built above. It stores messages with their WhatsApp JID (Jabber ID — WhatsApp's identifier for chats), tracks session state per group, and uses timestamps for deduplication.

One particularly elegant detail in NanoClaw's approach is the `PRAGMA journal_mode=WAL` setting. WAL (Write-Ahead Logging) mode allows multiple readers to operate concurrently with a single writer. This is critical because our polling loop reads from the database at the same time the agent might be writing tool results back. Without WAL mode, those operations would block each other.

NanoClaw also stores message metadata — not just the text content, but sender names, timestamps, read receipts, and message types (text, image, audio). This metadata is crucial for building rich context. When the agent sees `[Alice @ 2:34 PM]: Can someone help me with this?`, it has much more context than just seeing the raw text.

### Key Insight

> **SQLite is the most underrated database in software engineering.** It handles gigabytes of data, supports complex queries, provides ACID transactions, requires zero configuration, and ships with every Python installation. For a single-machine application like an AI assistant, it's not a compromise — it's the optimal choice. You'd need to be handling thousands of concurrent writes per second before you'd need to consider PostgreSQL, and by that point you're running a very different kind of system.

The `UPSERT` pattern (INSERT ... ON CONFLICT DO UPDATE) in our `store_message` function is worth studying. It atomically updates session metadata every time a message arrives, without requiring a separate query to check if the session exists. This kind of atomic operation is impossible with JSONL and trivial with SQL.

---

## Chapter 3: Per-Group Personality — Your Assistant's Many Faces

### The Problem

If your assistant lives in multiple group chats, it needs to behave differently in each one. A coding help group wants a technical, precise assistant. A friend group wants something casual and funny. A work channel wants professional and concise.

This isn't just about tone. Different groups might need:
- Different tool sets (code execution in the dev group, not in the family chat)
- Different knowledge bases (project docs for the work team)
- Different behavioral rules ("never discuss work topics in the friends group")
- Different names or personas

The question is: where do you store this configuration?

Environment variables? Too rigid — you'd need to restart to change anything. A config file? Better, but it mixes all groups together. A database table? Possible, but hard to edit and version.

The answer, borrowed from the Claude Code convention, is: **a file per group**.

### The Code

```python
import os

WORKSPACE_ROOT = "workspace"

def get_group_workspace(group_id):
    """Get or create the workspace directory for a group."""
    safe_id = group_id.replace("/", "_").replace("@", "_")
    group_dir = os.path.join(WORKSPACE_ROOT, "groups", safe_id)
    os.makedirs(group_dir, exist_ok=True)
    return group_dir

def get_system_prompt(group_id):
    """Load the system prompt for a group, with fallback to default."""
    group_dir = get_group_workspace(group_id)
    claude_md_path = os.path.join(group_dir, "CLAUDE.md")
    
    if os.path.exists(claude_md_path):
        with open(claude_md_path, "r") as f:
            custom_prompt = f.read().strip()
        if custom_prompt:
            return custom_prompt
    
    # Fall back to default system prompt
    return load_default_prompt()

def load_default_prompt():
    """Load the default system prompt."""
    default_path = os.path.join(WORKSPACE_ROOT, "CLAUDE.md")
    if os.path.exists(default_path):
        with open(default_path, "r") as f:
            return f.read().strip()
    
    return """You are a helpful AI assistant in a group chat. 
Be concise, friendly, and helpful. 
When multiple people are talking, pay attention to who said what."""

def save_group_prompt(group_id, prompt_text):
    """Save a custom system prompt for a group."""
    group_dir = get_group_workspace(group_id)
    claude_md_path = os.path.join(group_dir, "CLAUDE.md")
    with open(claude_md_path, "w") as f:
        f.write(prompt_text)

# The agent can modify its own personality via a tool:
def tool_update_personality(group_id, new_prompt):
    """Tool that lets the agent update its own personality for a group."""
    save_group_prompt(group_id, new_prompt)
    return f"Personality updated for group {group_id}."
```

And here's what a group's `CLAUDE.md` might look like:

```markdown
# Dev Team Assistant

You are a senior software engineer assistant for the dev team.

## Behavior
- Be technical and precise
- Show code examples when explaining concepts
- Use proper terminology, don't oversimplify
- When reviewing code, be constructive but thorough

## Tools
- You have access to: web_search, code_execute, file_read
- Prefer showing code over describing code

## Context
- The team works primarily in Python and TypeScript
- Main project: internal dashboard (React + FastAPI)
- Git repo: git@company.com:team/dashboard.git

## Tone
Professional but not stuffy. Think "helpful senior colleague," not "corporate chatbot."
```

### How NanoClaw Does It

NanoClaw uses exactly this pattern. Each chat group gets its own directory under the workspace, and a `CLAUDE.md` file (named after the Anthropic convention used by Claude Code) serves as the system prompt. This file is loaded at the start of every agent turn and injected as the system message in the API call.

What makes this pattern particularly powerful is that **the agent itself can modify these files**. If a user says "hey, can you be more casual in this chat?", the agent can update its own `CLAUDE.md` to reflect that preference. The change persists across restarts because it's just a file on disk.

NanoClaw takes this further by also storing per-group tool configurations and memory files in the same directory structure. Each group's workspace becomes a self-contained unit — its personality, its memory, its tools, its files — all in one directory that's easy to back up, inspect, or migrate.

This is directly inspired by OpenClaw's `AGENTS.md` and `SOUL.md` pattern, where the assistant's identity and instructions are stored as workspace files that can be edited by both the user and the assistant itself.

### Key Insight

> **Configuration as files beats configuration as data.** When your system prompt lives in a markdown file, you can edit it with any text editor. You can version it with git. You can diff it to see what changed. You can copy it between groups. You can write documentation in it. You can have the AI read it, understand it, and improve it. None of this is possible when configuration lives in a database column or environment variable. Files are the universal interface.

The fallback chain (group-specific → workspace default → hardcoded) is also important. New groups start with sensible defaults, and customization is opt-in. This means your assistant works out of the box but can be tailored over time.

---

## Chapter 4: The Agent Loop — Where the Magic Happens

### The Problem

Here's the thing about LLMs that surprises most people when they first build with them: **a single API call is rarely enough**.

When you give Claude or GPT a set of tools, the response might not be a final answer. It might be: "I need to call the `web_search` tool with query 'weather in Seoul'." Your code needs to execute that tool, feed the result back to the LLM, and let it continue. The LLM might then say "Now I need to call `get_calendar` to check your schedule." Another round trip. Eventually, it produces a final text response.

This is the **agent loop** — the core pattern that transforms a stateless LLM API into something that feels like an intelligent agent. It's deceptively simple in structure but carries a lot of subtlety.

### The Code

```python
import anthropic
import json

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-20250514"

# Define available tools
TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for information",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "save_memory",
        "description": "Save information to long-term memory",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Memory key/topic"},
                "content": {"type": "string", "description": "Content to remember"}
            },
            "required": ["key", "content"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file"}
            },
            "required": ["path"]
        }
    }
]

def execute_tool(tool_name, tool_input):
    """Execute a tool and return the result."""
    try:
        if tool_name == "web_search":
            return do_web_search(tool_input["query"])
        elif tool_name == "save_memory":
            return do_save_memory(tool_input["key"], tool_input["content"])
        elif tool_name == "read_file":
            return do_read_file(tool_input["path"])
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Tool error: {str(e)}"

def extract_text(response):
    """Extract text content from an API response."""
    texts = []
    for block in response.content:
        if block.type == "text":
            texts.append(block.text)
    return "\n".join(texts)

def run_agent_turn(messages, system_prompt, max_iterations=10):
    """
    Run a complete agent turn with tool use.
    
    This is the core loop: call the LLM, check if it wants to use tools,
    execute them, feed results back, repeat until it's done.
    """
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Call the LLM
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages
        )
        
        # If the model is done talking, return the final text
        if response.stop_reason == "end_turn":
            final_text = extract_text(response)
            return final_text, messages
        
        # If the model wants to use tools, execute them
        if response.stop_reason == "tool_use":
            # Add the assistant's response (with tool_use blocks) to history
            messages.append({
                "role": "assistant",
                "content": [block.model_dump() for block in response.content]
            })
            
            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  🔧 Executing tool: {block.name}({json.dumps(block.input)[:100]}...)")
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(result)
                    })
            
            # Feed tool results back to the model
            messages.append({
                "role": "user",
                "content": tool_results
            })
        else:
            # Unexpected stop reason — return whatever we have
            print(f"  ⚠️ Unexpected stop reason: {response.stop_reason}")
            return extract_text(response), messages
    
    # Safety valve: too many iterations
    print(f"  ⚠️ Agent loop hit max iterations ({max_iterations})")
    return extract_text(response) if response else "I got stuck in a loop. Sorry about that.", messages
```

### How NanoClaw Does It

NanoClaw implements this exact pattern in TypeScript with a few additional refinements:

**Token tracking.** Every API call's input and output tokens are tracked. This is important for cost monitoring — agent loops can burn through tokens fast if the model gets into a tool-heavy workflow. NanoClaw logs cumulative token usage per conversation so you can see which groups or users are most expensive.

**Streaming support.** While our implementation waits for the complete response, NanoClaw supports streaming responses. This lets the assistant show a typing indicator while the LLM is generating, and even stream partial responses to the chat. The stream is collected into content blocks, and tool_use blocks are detected as they complete within the stream.

**Error recovery.** If a tool execution fails, NanoClaw wraps the error message in a clear format and feeds it back. Importantly, it adds `is_error: true` to the tool result, which tells the LLM "this tool failed, don't just repeat the same call — adapt." Without this flag, models sometimes get stuck in a failure loop, calling the same broken tool over and over.

**Content block handling.** The Anthropic API returns content as an array of typed blocks (text, tool_use, image, etc.). NanoClaw carefully preserves these blocks when building the conversation history, rather than flattening everything to text. This is crucial because tool_use blocks contain the tool call ID that must match the corresponding tool_result.

### Key Insight

> **The agent loop is a while loop. That's it.** All the magic of "AI agents" — tool use, multi-step reasoning, research workflows — boils down to: call the API, check if it wants tools, execute them, call the API again. The pattern is so simple it feels like it can't be right, but it is. The intelligence comes from the model, not the loop. Your job is to build a reliable loop that handles errors gracefully and knows when to stop.

The `max_iterations` safety valve is critical and often overlooked. Without it, a model that gets confused about a tool can loop forever, burning through your API budget. Ten iterations is a generous limit — most useful agent turns complete in 1-3 iterations (one initial response, maybe one or two tool calls). If you're hitting 10, something has gone wrong.

---

## Chapter 5: Container Isolation — Running Code Without Fear

### The Problem

The moment your assistant can execute code or write files, you have a security problem. What if the LLM decides to `rm -rf /`? What if a user tricks it into reading `/etc/passwd`? What if a tool execution goes into an infinite loop and pegs your CPU at 100%?

You need **isolation**. The agent's code execution environment should be sandboxed — limited file access, limited network access, limited resource consumption. Break the sandbox? Fine, you broke a disposable environment, not the host.

The gold standard is Linux containers (Docker, LXC, namespaces). The practical minimum is subprocess isolation. Let's explore both.

### The Code

```python
import subprocess
import tempfile
import os
import signal
import json

class SandboxedAgent:
    """Run agent code in an isolated subprocess."""
    
    def __init__(self, group_id, workspace_root="workspace"):
        self.group_id = group_id
        self.workspace = os.path.join(workspace_root, "groups", 
                                       group_id.replace("/", "_").replace("@", "_"))
        os.makedirs(self.workspace, exist_ok=True)
    
    def run_agent(self, messages, system_prompt, timeout=120):
        """
        Run the agent in a subprocess with restricted working directory.
        
        The agent code runs in a separate process that:
        - Has its cwd set to the group workspace
        - Can only access files within that workspace
        - Has a timeout to prevent infinite loops
        - Communicates results via stdout/stderr
        """
        # Prepare the agent payload
        payload = {
            "messages": messages,
            "system_prompt": system_prompt,
            "tools": get_available_tools(self.group_id),
            "workspace": self.workspace
        }
        
        # Write payload to a temp file
        payload_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.json', delete=False
        )
        json.dump(payload, payload_file)
        payload_file.close()
        
        try:
            # Run agent in subprocess
            result = subprocess.run(
                ["python3", "agent_worker.py", payload_file.name],
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={
                    **os.environ,
                    "AGENT_WORKSPACE": self.workspace,
                    "AGENT_GROUP_ID": self.group_id,
                    # Restrict the agent's view
                    "HOME": self.workspace,
                    "TMPDIR": os.path.join(self.workspace, "tmp"),
                }
            )
            
            if result.returncode == 0:
                response = json.loads(result.stdout)
                return response.get("reply", ""), response.get("tool_log", [])
            else:
                error_msg = result.stderr[:500] if result.stderr else "Unknown error"
                return f"Agent error: {error_msg}", []
                
        except subprocess.TimeoutExpired:
            return "I took too long to respond. Sorry about that!", []
        except Exception as e:
            return f"Sandbox error: {str(e)}", []
        finally:
            os.unlink(payload_file.name)


class ContainerSandbox:
    """
    For production use: run agent in a real container.
    This provides true isolation with Linux namespaces.
    """
    
    def __init__(self, image="agent-sandbox:latest"):
        self.image = image
    
    def run_agent(self, payload, timeout=120):
        """Run agent in a Docker container with restricted resources."""
        payload_json = json.dumps(payload)
        
        result = subprocess.run([
            "docker", "run",
            "--rm",                          # Remove container after exit
            "--network=none",                # No network access
            "--memory=512m",                 # Max 512MB RAM
            "--cpus=1",                      # Max 1 CPU core
            "--read-only",                   # Read-only filesystem
            "--tmpfs=/tmp:size=100m",        # Writable /tmp, 100MB max
            "-v", f"{self.workspace}:/workspace:rw",  # Mount workspace
            "-e", f"PAYLOAD={payload_json}",
            self.image,
            "python3", "/app/agent_worker.py"
        ], capture_output=True, text=True, timeout=timeout)
        
        return json.loads(result.stdout) if result.returncode == 0 else None
```

### How NanoClaw Does It

NanoClaw uses real Linux containers for isolation, which is one of its most impressive features for a "minimal" project. It leverages `lxc` (Linux Containers) to create lightweight sandboxes for each agent session.

The architecture is split into two processes:

1. **The host process** — Runs the polling loop, manages the database, handles messaging. This process has full system access.
2. **The agent process** — Runs inside a container with a restricted filesystem, limited resources, and controlled network access. This is where LLM calls happen and tools execute.

The two processes communicate via file-based IPC (which we'll cover in Chapter 8). The agent writes "I need to send a message" to a file, the host reads it and actually sends the message. This separation means even if the agent process is completely compromised, it can only affect its own container.

OpenClaw takes this even further with full container orchestration, network policies, and per-tool permission systems. But NanoClaw proves that even a minimal implementation of container isolation dramatically improves security.

### Key Insight

> **Isolation is not optional when your assistant can execute tools.** The LLM is not always predictable. Users can attempt prompt injection. Tools can have bugs. Without isolation, a single bad tool execution can compromise your entire system. Subprocess isolation is the minimum viable security. Container isolation is the production standard. Choose based on your threat model, but choose *something*.

Notice the timeout parameter. This is as important as the isolation itself. An agent stuck in an infinite loop inside a container only wastes container resources. An agent stuck in an infinite loop on the bare host can take down your system. Always set timeouts on subprocess and container execution.

---

## Chapter 6: The Group Queue — Preventing Conversation Collisions

### The Problem

Imagine this scenario: Two messages arrive almost simultaneously for the same group chat. Your polling loop picks both up and spawns two agent workers. Both read the conversation history from the database. Both generate responses. Both try to write their responses back. Now the conversation history is corrupted — the second response doesn't account for the first, and the chat gets confused responses that overlap or contradict each other.

This is a classic concurrency problem, and it's especially tricky because:

1. You **want** concurrent processing — if 5 different groups have pending messages, you should handle all 5 in parallel.
2. You **don't want** concurrent processing within a single group — messages for the same group must be processed sequentially to maintain conversation coherence.

The solution is **per-group locking with bounded concurrency**.

### The Code

```python
import asyncio
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

class GroupQueue:
    """
    Per-group message queue with bounded concurrency.
    
    Ensures:
    - Messages for the same group are processed sequentially
    - Different groups can be processed in parallel
    - Maximum N groups processed simultaneously (to manage resources)
    """
    
    def __init__(self, max_concurrent=3):
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.group_locks = {}  # per-group locks
        self.group_locks_lock = threading.Lock()  # lock for the locks dict
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.pending_messages = defaultdict(list)  # group_id -> [messages]
        self.queue_lock = threading.Lock()
    
    def _get_group_lock(self, group_id):
        """Get or create a lock for a specific group."""
        with self.group_locks_lock:
            if group_id not in self.group_locks:
                self.group_locks[group_id] = threading.Lock()
            return self.group_locks[group_id]
    
    def enqueue(self, group_id, messages):
        """Add messages for a group to the processing queue."""
        with self.queue_lock:
            self.pending_messages[group_id].extend(messages)
        
        # Try to process this group (non-blocking)
        self.executor.submit(self._process_group, group_id)
    
    def _process_group(self, group_id):
        """Process all pending messages for a group, sequentially."""
        group_lock = self._get_group_lock(group_id)
        
        # If this group is already being processed, skip
        # (the current processor will pick up new messages)
        if not group_lock.acquire(blocking=False):
            return
        
        try:
            # Acquire a concurrency slot
            self.semaphore.acquire()
            try:
                while True:
                    # Grab all pending messages for this group
                    with self.queue_lock:
                        messages = self.pending_messages.pop(group_id, [])
                    
                    if not messages:
                        break
                    
                    # Process them as a batch
                    self._handle_messages(group_id, messages)
            finally:
                self.semaphore.release()
        finally:
            group_lock.release()
    
    def _handle_messages(self, group_id, messages):
        """Actually process a batch of messages for a group."""
        try:
            # Load conversation history
            history = get_conversation_history(group_id)
            
            # Format new messages and add to history
            for msg in messages:
                store_message(group_id, msg["sender"], msg["content"], msg["timestamp"])
                history.append(format_message(msg))
            
            # Get group-specific system prompt
            system_prompt = get_system_prompt(group_id)
            
            # Run the agent
            reply, _ = run_agent_turn(history, system_prompt)
            
            if reply:
                send_reply(group_id, reply)
                store_message(group_id, "assistant", reply, time.time(), is_from_me=True)
                
        except Exception as e:
            print(f"Error processing group {group_id}: {e}")


# Integration with the polling loop:
group_queue = GroupQueue(max_concurrent=3)

def polling_loop():
    while True:
        messages = poll_messages()
        
        # Group messages by chat ID
        by_group = defaultdict(list)
        for msg in messages:
            by_group[msg["chat_jid"]].append(msg)
        
        # Enqueue each group's messages
        for group_id, group_messages in by_group.items():
            group_queue.enqueue(group_id, group_messages)
        
        time.sleep(2.0)
```

### How NanoClaw Does It

NanoClaw implements a similar per-group queuing system, but with some additional sophistication:

**Message coalescing.** If a user sends multiple messages before the agent starts processing, NanoClaw groups them into a single agent turn. This is more natural — instead of responding to "hey" / "can you help" / "with this code" as three separate requests, the agent sees all three at once and responds coherently.

**Cooldown timer.** After receiving a message, NanoClaw waits a brief period (usually 1-2 seconds) before processing. This catches the common pattern where someone sends a message, then immediately follows up with "wait, I meant..." or adds more context. The cooldown gives them time to finish their thought.

**Priority groups.** Some groups can be configured as high-priority (active work conversations) or low-priority (casual chats). When the concurrency limit is reached, high-priority groups are processed first.

**Stale message handling.** If messages have been queued for too long (more than a few minutes due to high load), NanoClaw can acknowledge the delay: "Sorry for the slow response, I was busy with other conversations." This small touch makes the assistant feel more human.

### Key Insight

> **Per-group sequential processing with cross-group parallelism is the fundamental concurrency pattern for chat assistants.** It's the same pattern that databases use (row-level locking) and that web servers use (per-session consistency). The `max_concurrent` limit is equally important — without it, a burst of activity across 50 groups would spawn 50 agent workers, each making expensive API calls simultaneously. Three concurrent groups is a sensible default that balances responsiveness with resource usage.

The message batching behavior (coalescing messages that arrive between polling cycles) is a natural consequence of the polling architecture. With webhooks, each message would trigger an immediate, separate processing request. With polling, you inherently batch. This is another hidden advantage of the polling approach.

---

## Chapter 7: The Task Scheduler — Your Assistant Never Sleeps

### The Problem

A truly useful assistant doesn't just respond to messages — it takes initiative. "Remind me at 3 PM." "Send a daily summary every morning." "Check the weather forecast every day at 7 AM and post it to the group."

For this, you need a **task scheduler**. Something that can store scheduled tasks persistently (surviving restarts) and check periodically for tasks that are due.

You could use system cron, but that introduces external dependencies and makes your tasks harder to manage. Better to build a simple scheduler into the application itself, using SQLite for persistence.

### The Code

```python
import time
import json
import threading
from datetime import datetime, timedelta

def init_scheduler_tables():
    """Create the scheduled tasks table."""
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                task_data TEXT NOT NULL,
                cron_expression TEXT,
                next_run REAL NOT NULL,
                last_run REAL,
                is_recurring INTEGER DEFAULT 0,
                is_active INTEGER DEFAULT 1,
                created_at REAL NOT NULL,
                description TEXT
            )
        """)
        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_tasks_next_run
            ON scheduled_tasks(is_active, next_run)
        """)
        db.commit()

def schedule_task(group_id, task_type, task_data, run_at, 
                  cron_expression=None, description=None):
    """Schedule a new task."""
    with get_db() as db:
        db.execute("""
            INSERT INTO scheduled_tasks 
            (group_id, task_type, task_data, next_run, cron_expression, 
             is_recurring, created_at, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            group_id, task_type, json.dumps(task_data), run_at,
            cron_expression, 1 if cron_expression else 0,
            time.time(), description
        ))
        db.commit()

def get_due_tasks():
    """Get all tasks that are due to run."""
    now = time.time()
    with get_db() as db:
        rows = db.execute("""
            SELECT * FROM scheduled_tasks 
            WHERE is_active = 1 AND next_run <= ?
            ORDER BY next_run ASC
        """, (now,)).fetchall()
        return rows

def parse_cron_next(cron_expression, after_timestamp):
    """
    Parse a simplified cron expression and return the next run time.
    
    Supports: "daily HH:MM", "hourly", "weekly DAY HH:MM", 
              "every Nm" (every N minutes), "every Nh" (every N hours)
    """
    now = datetime.fromtimestamp(after_timestamp)
    
    if cron_expression.startswith("daily"):
        parts = cron_expression.split()
        hour, minute = map(int, parts[1].split(":"))
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        return next_run.timestamp()
    
    elif cron_expression.startswith("every"):
        parts = cron_expression.split()
        value = parts[1]
        if value.endswith("m"):
            minutes = int(value[:-1])
            return after_timestamp + (minutes * 60)
        elif value.endswith("h"):
            hours = int(value[:-1])
            return after_timestamp + (hours * 3600)
    
    elif cron_expression.startswith("weekly"):
        parts = cron_expression.split()
        day_name = parts[1].lower()
        hour, minute = map(int, parts[2].split(":"))
        days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, 
                "fri": 4, "sat": 5, "sun": 6}
        target_day = days.get(day_name, 0)
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        days_ahead = target_day - now.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        next_run += timedelta(days=days_ahead)
        return next_run.timestamp()
    
    # Default: run once (no recurrence)
    return None

def execute_task(task):
    """Execute a scheduled task."""
    task_type = task["task_type"]
    task_data = json.loads(task["task_data"])
    group_id = task["group_id"]
    
    if task_type == "reminder":
        send_reply(group_id, f"⏰ Reminder: {task_data['message']}")
    
    elif task_type == "agent_prompt":
        # Run the agent with a specific prompt
        system_prompt = get_system_prompt(group_id)
        messages = [{"role": "user", "content": task_data["prompt"]}]
        reply, _ = run_agent_turn(messages, system_prompt)
        if reply:
            send_reply(group_id, reply)
    
    elif task_type == "summary":
        # Generate a conversation summary
        history = get_conversation_history(group_id, limit=100)
        summary_prompt = "Summarize the key points from today's conversation."
        messages = format_history_for_summary(history) + [
            {"role": "user", "content": summary_prompt}
        ]
        reply, _ = run_agent_turn(messages, get_system_prompt(group_id))
        if reply:
            send_reply(group_id, f"📋 Daily Summary:\n{reply}")

def update_task_after_run(task):
    """Update task timing after execution."""
    with get_db() as db:
        if task["is_recurring"] and task["cron_expression"]:
            next_run = parse_cron_next(task["cron_expression"], time.time())
            if next_run:
                db.execute("""
                    UPDATE scheduled_tasks 
                    SET last_run = ?, next_run = ?
                    WHERE id = ?
                """, (time.time(), next_run, task["id"]))
            else:
                db.execute(
                    "UPDATE scheduled_tasks SET is_active = 0, last_run = ? WHERE id = ?",
                    (time.time(), task["id"])
                )
        else:
            # One-shot task — deactivate after running
            db.execute(
                "UPDATE scheduled_tasks SET is_active = 0, last_run = ? WHERE id = ?",
                (time.time(), task["id"])
            )
        db.commit()

def scheduler_loop():
    """Background loop that checks for and executes due tasks."""
    while True:
        try:
            due_tasks = get_due_tasks()
            for task in due_tasks:
                try:
                    execute_task(task)
                    update_task_after_run(task)
                except Exception as e:
                    print(f"Task execution error (task {task['id']}): {e}")
                    # Don't re-run failed one-shot tasks
                    if not task["is_recurring"]:
                        with get_db() as db:
                            db.execute(
                                "UPDATE scheduled_tasks SET is_active = 0 WHERE id = ?",
                                (task["id"],)
                            )
                            db.commit()
        except Exception as e:
            print(f"Scheduler error: {e}")
        
        time.sleep(60)  # Check every 60 seconds
```

### How NanoClaw Does It

NanoClaw stores scheduled tasks in SQLite, similar to our approach. Its scheduler runs as a background timer that checks for due tasks at regular intervals.

One interesting addition in NanoClaw is **agent-initiated scheduling**. The agent itself has a `schedule_task` tool, so users can say things like "remind me every Monday at 9 AM to review the sprint board" and the agent creates the recurring task. The task's `task_data` includes the original user request so the agent can reconstruct what it needs to do when the task fires.

NanoClaw also implements **task management tools** — the agent can list, modify, and cancel scheduled tasks. This makes the scheduling system conversational: "What reminders do I have?" / "Cancel the Monday morning one" / "Change it to Tuesday instead."

OpenClaw's approach is more sophisticated with its cron system, which supports full cron expressions, timezone-aware scheduling, and the ability to spawn tasks with specific models and thinking levels. But the core pattern is identical: check for due tasks, execute them, schedule the next occurrence.

### Key Insight

> **A scheduler transforms your assistant from reactive to proactive.** Without scheduling, your assistant only speaks when spoken to. With it, the assistant becomes a presence that can remind, summarize, check, and report on its own schedule. The implementation is surprisingly simple — a table of "thing to do" and "when to do it," checked every minute. But the user experience difference is enormous. It's the difference between a tool you use and an assistant that works for you.

The 60-second check interval is deliberately coarse. For a chat assistant, minute-level precision is more than enough. No one notices if their "9 AM reminder" fires at 9:00:23. The coarse interval also means the scheduler barely consumes any resources.

---

## Chapter 8: Inter-Process Communication — Bridging the Sandbox Gap

### The Problem

In Chapter 5, we sandboxed the agent in a subprocess (or container). That creates a new challenge: the agent needs to interact with the outside world (send messages, access the internet, read shared resources), but the whole point of the sandbox is to limit that access.

We need a **controlled communication channel** between the sandboxed agent and the host process. The agent should be able to make requests ("send this message to the group"), and the host should be able to approve and execute those requests in a controlled manner.

This is classic inter-process communication (IPC), but with a security twist: the agent (inside the sandbox) is untrusted. The host (outside) enforces policy on every request.

### The Code

```python
import os
import json
import time
import uuid
import glob

class FileIPC:
    """
    File-based IPC between host and sandboxed agent.
    
    Protocol:
    1. Agent writes a request file: {ipc_dir}/requests/{uuid}.json
    2. Host reads the request, validates it, executes it
    3. Host writes a response file: {ipc_dir}/responses/{uuid}.json
    4. Agent reads the response
    
    File-based IPC is used because:
    - Files work across process boundaries, containers, and even machines
    - No special libraries or protocols needed
    - Easy to inspect and debug (just look at the files)
    - Naturally asynchronous
    """
    
    def __init__(self, ipc_dir):
        self.ipc_dir = ipc_dir
        self.request_dir = os.path.join(ipc_dir, "requests")
        self.response_dir = os.path.join(ipc_dir, "responses")
        os.makedirs(self.request_dir, exist_ok=True)
        os.makedirs(self.response_dir, exist_ok=True)
    
    # --- Agent side (runs inside sandbox) ---
    
    def send_request(self, action, data, timeout=30):
        """Send a request to the host and wait for a response."""
        request_id = str(uuid.uuid4())
        request = {
            "id": request_id,
            "action": action,
            "data": data,
            "timestamp": time.time()
        }
        
        # Write request file (atomically, using rename)
        tmp_path = os.path.join(self.request_dir, f"{request_id}.tmp")
        final_path = os.path.join(self.request_dir, f"{request_id}.json")
        with open(tmp_path, "w") as f:
            json.dump(request, f)
        os.rename(tmp_path, final_path)
        
        # Wait for response
        response_path = os.path.join(self.response_dir, f"{request_id}.json")
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(response_path):
                with open(response_path, "r") as f:
                    response = json.load(f)
                os.unlink(response_path)  # Clean up
                return response
            time.sleep(0.1)
        
        return {"error": "Request timed out", "id": request_id}
    
    # --- Host side (runs outside sandbox) ---
    
    def poll_requests(self):
        """Check for pending requests from the agent."""
        requests = []
        pattern = os.path.join(self.request_dir, "*.json")
        for request_path in glob.glob(pattern):
            try:
                with open(request_path, "r") as f:
                    request = json.load(f)
                os.unlink(request_path)  # Consume the request
                requests.append(request)
            except (json.JSONDecodeError, IOError):
                continue  # Skip corrupted files
        return requests
    
    def send_response(self, request_id, result):
        """Send a response back to the agent."""
        response = {
            "id": request_id,
            "result": result,
            "timestamp": time.time()
        }
        tmp_path = os.path.join(self.response_dir, f"{request_id}.tmp")
        final_path = os.path.join(self.response_dir, f"{request_id}.json")
        with open(tmp_path, "w") as f:
            json.dump(response, f)
        os.rename(tmp_path, final_path)


class HostIPCHandler:
    """Host-side handler that processes agent requests with policy enforcement."""
    
    ALLOWED_ACTIONS = {
        "send_message",
        "web_search",
        "read_shared_file",
        "get_time",
        "schedule_task",
    }
    
    def __init__(self, ipc):
        self.ipc = ipc
    
    def process_requests(self):
        """Process all pending IPC requests."""
        requests = self.ipc.poll_requests()
        for request in requests:
            try:
                result = self.handle_request(request)
                self.ipc.send_response(request["id"], result)
            except Exception as e:
                self.ipc.send_response(request["id"], {
                    "error": str(e)
                })
    
    def handle_request(self, request):
        """Handle a single request with policy checks."""
        action = request.get("action")
        data = request.get("data", {})
        
        # Policy check: is this action allowed?
        if action not in self.ALLOWED_ACTIONS:
            return {"error": f"Action '{action}' is not allowed"}
        
        # Execute the action
        if action == "send_message":
            return self._handle_send_message(data)
        elif action == "web_search":
            return self._handle_web_search(data)
        elif action == "read_shared_file":
            return self._handle_read_file(data)
        elif action == "get_time":
            return {"time": time.time(), "formatted": datetime.now().isoformat()}
        elif action == "schedule_task":
            return self._handle_schedule_task(data)
        
        return {"error": "Unimplemented action"}
    
    def _handle_send_message(self, data):
        """Send a message, with rate limiting and content checks."""
        chat_id = data.get("chat_id")
        message = data.get("message", "")
        
        # Rate limiting
        if len(message) > 4000:
            message = message[:4000] + "\n\n(Message truncated)"
        
        send_reply(chat_id, message)
        return {"status": "sent", "length": len(message)}
    
    def _handle_web_search(self, data):
        """Perform a web search on behalf of the agent."""
        query = data.get("query", "")
        if len(query) > 200:
            return {"error": "Query too long"}
        results = do_web_search(query)
        return {"results": results}
    
    def _handle_read_file(self, data):
        """Read a shared file, with path validation."""
        path = data.get("path", "")
        # Security: ensure the path is within allowed directories
        allowed_roots = ["workspace/shared/", "workspace/docs/"]
        if not any(path.startswith(root) for root in allowed_roots):
            return {"error": "Access denied: path not in allowed directories"}
        
        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read(100000)  # Cap at 100KB
            return {"content": content}
        return {"error": "File not found"}
    
    def _handle_schedule_task(self, data):
        """Schedule a task on behalf of the agent."""
        schedule_task(
            group_id=data["group_id"],
            task_type=data["task_type"],
            task_data=data["task_data"],
            run_at=data["run_at"],
            cron_expression=data.get("cron_expression"),
            description=data.get("description")
        )
        return {"status": "scheduled"}
```

### How NanoClaw Does It

NanoClaw implements file-based IPC almost exactly as described above. The container has a mounted directory that's shared with the host, and both processes use JSON files in that directory to communicate.

The key design decision is **atomic file writing using rename**. Both our implementation and NanoClaw use the two-step pattern: write to a `.tmp` file first, then rename to the final name. This prevents the reader from seeing a partially-written file. `os.rename()` is atomic on POSIX systems (Linux, macOS), so the response file either exists completely or doesn't exist at all.

NanoClaw adds a few features to the basic pattern:

**Request prioritization.** Some IPC requests (like sending a message) are higher priority than others (like a web search). The host processes them in priority order.

**Batch responses.** If the agent sends multiple requests before the host processes them, NanoClaw can batch the processing for efficiency.

**Heartbeat monitoring.** The host monitors whether the agent process is still alive by checking for periodic heartbeat files. If the agent dies, the host can clean up resources and report the error.

OpenClaw's IPC is more sophisticated, using a proper message bus, but the file-based approach has a charm to it: it's debuggable by literally looking at files in a directory. When something goes wrong, you can inspect the request and response files to understand exactly what happened.

### Key Insight

> **File-based IPC is an underappreciated pattern.** It requires no libraries, no protocols, no ports. It works across containers, across users, even across machines (with a shared filesystem). It's naturally persistent — if the host crashes mid-request, the request file is still there when it restarts. And it's transparent — you can debug issues by looking at files in a directory with `ls` and `cat`. The atomic rename trick makes it reliable. For low-throughput, high-reliability communication (which is exactly what an AI assistant needs), file-based IPC is hard to beat.

The security model is also worth highlighting. By routing all external actions through the host-side handler, we create a chokepoint where **every agent action can be inspected, rate-limited, and denied**. The agent can't directly send a message — it has to ask the host, and the host decides whether to allow it. This "capability-based" security model is much more robust than trying to restrict what code the agent can execute.

---

## Chapter 9: The Multi-Channel Gateway — One Brain, Many Mouths

### The Problem

So far, our assistant talks to one messaging platform. But what if you want it on WhatsApp *and* Discord? Or add a terminal interface for local testing? Or expose an HTTP API for web integration?

You could copy-paste the entire codebase for each platform. But then you'd have three polling loops, three databases, three agent instances — each with their own conversation context, each drifting apart as you update one and forget the others.

The better approach is the **gateway pattern**: a single agent core with multiple channel adapters. Each adapter translates between its platform's message format and the agent's internal format. The agent doesn't know or care whether a message came from WhatsApp, Discord, or a terminal. It just processes messages.

### The Code

```python
from abc import ABC, abstractmethod
import http.server
import json
import sys
import threading

class ChannelAdapter(ABC):
    """Base class for channel adapters."""
    
    def __init__(self, agent_core):
        self.agent_core = agent_core
    
    @abstractmethod
    def start(self):
        """Start the channel adapter."""
        pass
    
    @abstractmethod
    def send_message(self, chat_id, message):
        """Send a message to a specific chat."""
        pass
    
    @abstractmethod
    def get_channel_name(self):
        """Return the name of this channel (for logging)."""
        pass
    
    def on_message_received(self, chat_id, sender, content, timestamp):
        """Called when this channel receives a message. Routes to agent core."""
        self.agent_core.handle_message(
            channel=self.get_channel_name(),
            chat_id=chat_id,
            sender=sender,
            content=content,
            timestamp=timestamp,
            reply_callback=lambda msg: self.send_message(chat_id, msg)
        )


class TerminalAdapter(ChannelAdapter):
    """Terminal/CLI interface for local testing."""
    
    def get_channel_name(self):
        return "terminal"
    
    def start(self):
        thread = threading.Thread(target=self._input_loop, daemon=True)
        thread.start()
    
    def _input_loop(self):
        chat_id = "terminal_session"
        print("\n🤖 AI Assistant (type 'quit' to exit)")
        print("-" * 40)
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("quit", "exit"):
                    break
                if not user_input:
                    continue
                self.on_message_received(
                    chat_id=chat_id,
                    sender="user",
                    content=user_input,
                    timestamp=time.time()
                )
            except EOFError:
                break
    
    def send_message(self, chat_id, message):
        print(f"\n🤖 Assistant: {message}")


class HTTPAdapter(ChannelAdapter):
    """HTTP API interface for web integration."""
    
    def __init__(self, agent_core, port=8080):
        super().__init__(agent_core)
        self.port = port
        self.pending_responses = {}
    
    def get_channel_name(self):
        return "http"
    
    def start(self):
        adapter = self  # Capture for the handler class
        
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/message":
                    content_length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(content_length))
                    
                    chat_id = body.get("chat_id", "default")
                    message = body.get("message", "")
                    sender = body.get("sender", "api_user")
                    
                    # Synchronous: wait for the response
                    response_event = threading.Event()
                    response_holder = {}
                    
                    def on_reply(msg):
                        response_holder["reply"] = msg
                        response_event.set()
                    
                    adapter.agent_core.handle_message(
                        channel="http",
                        chat_id=chat_id,
                        sender=sender,
                        content=message,
                        timestamp=time.time(),
                        reply_callback=on_reply
                    )
                    
                    # Wait for response (with timeout)
                    response_event.wait(timeout=120)
                    
                    reply = response_holder.get("reply", "No response generated")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"reply": reply}).encode())
                else:
                    self.send_error(404)
            
            def log_message(self, format, *args):
                pass  # Suppress default logging
        
        server = http.server.HTTPServer(("0.0.0.0", self.port), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"  📡 HTTP adapter listening on port {self.port}")
    
    def send_message(self, chat_id, message):
        # For HTTP, responses are sent synchronously via the request handler
        pass


class WhatsAppAdapter(ChannelAdapter):
    """WhatsApp adapter using whatsapp-web.js bridge."""
    
    def __init__(self, agent_core, bridge_url="http://localhost:3001"):
        super().__init__(agent_core)
        self.bridge_url = bridge_url
        self.last_seen = {}
    
    def get_channel_name(self):
        return "whatsapp"
    
    def start(self):
        thread = threading.Thread(target=self._poll_loop, daemon=True)
        thread.start()
    
    def _poll_loop(self):
        while True:
            try:
                messages = self._fetch_new_messages()
                for msg in messages:
                    self.on_message_received(
                        chat_id=msg["chat_jid"],
                        sender=msg["sender"],
                        content=msg["content"],
                        timestamp=msg["timestamp"]
                    )
            except Exception as e:
                print(f"WhatsApp poll error: {e}")
            time.sleep(2.0)
    
    def _fetch_new_messages(self):
        response = requests.get(f"{self.bridge_url}/messages")
        if response.status_code == 200:
            return response.json().get("messages", [])
        return []
    
    def send_message(self, chat_id, message):
        requests.post(f"{self.bridge_url}/send", json={
            "chat_id": chat_id,
            "message": message
        })


class AgentCore:
    """The central agent that all channels route through."""
    
    def __init__(self):
        self.group_queue = GroupQueue(max_concurrent=3)
    
    def handle_message(self, channel, chat_id, sender, content, 
                       timestamp, reply_callback):
        """Handle a message from any channel."""
        # Create a unified chat ID that includes the channel
        unified_id = f"{channel}:{chat_id}"
        
        # Store in database
        store_message(unified_id, sender, content, timestamp)
        
        # Enqueue for processing
        self.group_queue.enqueue(unified_id, [{
            "sender": sender,
            "content": content,
            "timestamp": timestamp,
            "reply_callback": reply_callback
        }])


class Gateway:
    """The gateway that wires everything together."""
    
    def __init__(self):
        self.core = AgentCore()
        self.adapters = []
    
    def add_adapter(self, adapter_class, **kwargs):
        adapter = adapter_class(self.core, **kwargs)
        self.adapters.append(adapter)
        return adapter
    
    def start(self):
        print("🚀 Starting AI Assistant Gateway")
        for adapter in self.adapters:
            adapter.start()
            print(f"  ✅ {adapter.get_channel_name()} adapter started")
        print("  All adapters running. Press Ctrl+C to stop.\n")


# Usage:
def main():
    gateway = Gateway()
    gateway.add_adapter(TerminalAdapter)
    gateway.add_adapter(HTTPAdapter, port=8080)
    # gateway.add_adapter(WhatsAppAdapter, bridge_url="http://localhost:3001")
    gateway.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
```

### How NanoClaw Does It

NanoClaw is primarily focused on WhatsApp, but its architecture cleanly separates the messaging layer from the agent layer, making it straightforward to add additional channels. The key abstraction is that **the agent never deals with platform-specific message formats** — it only sees a normalized representation (sender, content, timestamp, chat_id).

OpenClaw, being a production system, implements exactly the gateway pattern we've described — but at a much larger scale. It supports Discord, WhatsApp (via multiple bridging strategies), HTTP APIs, terminal interfaces, and even direct browser control. Each channel is a plugin that conforms to a standard interface.

The crucial architectural insight that both NanoClaw and OpenClaw share is the **unified chat ID**. By prefixing channel names to chat IDs (`whatsapp:123456@g.us`, `discord:channel_789`), you can store all conversations in the same database table and use the same processing pipeline. The channel prefix prevents collisions (a WhatsApp group and a Discord channel could both have the ID "123456") and makes it easy to route replies back to the correct channel.

### Key Insight

> **The gateway pattern decouples your agent's intelligence from its communication channels.** This is the same pattern that makes web applications work — your business logic doesn't know if the request came from a mobile app, a browser, or an API client. For an AI assistant, this decoupling means you can add new channels without touching the agent code, test locally via terminal while deploying to WhatsApp, and even bridge conversations across platforms if needed. Build the agent once, expose it everywhere.

The `reply_callback` pattern is also worth noting. Instead of the agent core knowing how to send messages on each platform, the channel adapter passes a callback function. This is dependency inversion — the high-level module (agent core) doesn't depend on low-level modules (channel adapters). It only depends on the abstract concept of "a function I can call to reply."

---

## Chapter 10: Long-Term Memory — Your Assistant Remembers

### The Problem

LLMs are stateless. Every API call starts fresh. Your assistant's conversation history gives it short-term memory (the current chat), but what about long-term memory?

"Remember that I prefer dark mode." "I told you last week that my project deadline is March 15th." "You said you'd look into that issue — did you?"

Without long-term memory, your assistant has the memory of a goldfish. Every conversation starts from zero. Users have to repeat context endlessly.

The solution is to give the assistant tools for saving and retrieving information. Not in the database (that's for messages), but in a structured knowledge store that persists across conversations.

### The Code

```python
import os
import json
import re
from datetime import datetime
from pathlib import Path

MEMORY_DIR = "workspace/memory"

def init_memory():
    """Initialize the memory directory structure."""
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(os.path.join(MEMORY_DIR, "topics"), exist_ok=True)
    os.makedirs(os.path.join(MEMORY_DIR, "people"), exist_ok=True)
    os.makedirs(os.path.join(MEMORY_DIR, "daily"), exist_ok=True)

def tool_save_memory(key, content, category="topics"):
    """
    Save information to long-term memory.
    
    Tool definition for the LLM:
    - name: save_memory
    - description: Save important information for later recall
    - params: key (topic name), content (what to remember), category (topics/people/daily)
    """
    safe_key = re.sub(r'[^a-zA-Z0-9_-]', '_', key.lower())
    category_dir = os.path.join(MEMORY_DIR, category)
    os.makedirs(category_dir, exist_ok=True)
    
    file_path = os.path.join(category_dir, f"{safe_key}.md")
    
    # If the file exists, append instead of overwrite
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"\n\n## [{timestamp}]\n{content}"
    
    if os.path.exists(file_path):
        with open(file_path, "a") as f:
            f.write(entry)
    else:
        with open(file_path, "w") as f:
            f.write(f"# {key}\n{entry}")
    
    return f"Saved to memory: {key}"

def tool_search_memory(query, category=None):
    """
    Search long-term memory for relevant information.
    
    Tool definition for the LLM:
    - name: search_memory
    - description: Search saved memories for relevant information
    - params: query (search text), category (optional filter)
    """
    results = []
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    # Determine which directories to search
    if category:
        search_dirs = [os.path.join(MEMORY_DIR, category)]
    else:
        search_dirs = [
            os.path.join(MEMORY_DIR, d) 
            for d in os.listdir(MEMORY_DIR)
            if os.path.isdir(os.path.join(MEMORY_DIR, d))
        ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        for filename in os.listdir(search_dir):
            if not filename.endswith(".md"):
                continue
            file_path = os.path.join(search_dir, filename)
            with open(file_path, "r") as f:
                content = f.read()
            
            # Simple relevance scoring: count matching words
            content_lower = content.lower()
            score = sum(1 for word in query_words if word in content_lower)
            
            # Bonus for filename match
            if query_lower in filename.lower():
                score += 3
            
            if score > 0:
                results.append({
                    "file": filename,
                    "category": os.path.basename(search_dir),
                    "content": content[:500],  # Truncate for context window
                    "score": score
                })
    
    # Sort by relevance
    results.sort(key=lambda x: x["score"], reverse=True)
    
    if results:
        formatted = []
        for r in results[:5]:  # Top 5 results
            formatted.append(f"**{r['file']}** ({r['category']}):\n{r['content']}")
        return "\n\n---\n\n".join(formatted)
    else:
        return "No memories found matching that query."

def tool_list_memories(category=None):
    """
    List all saved memories.
    
    Tool definition for the LLM:
    - name: list_memories
    - description: List all topics in memory
    - params: category (optional filter)
    """
    memories = []
    
    if category:
        dirs = [os.path.join(MEMORY_DIR, category)]
    else:
        dirs = [
            os.path.join(MEMORY_DIR, d) 
            for d in sorted(os.listdir(MEMORY_DIR))
            if os.path.isdir(os.path.join(MEMORY_DIR, d))
        ]
    
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            continue
        cat_name = os.path.basename(dir_path)
        files = sorted(f for f in os.listdir(dir_path) if f.endswith(".md"))
        if files:
            memories.append(f"**{cat_name}/**: {', '.join(f.replace('.md', '') for f in files)}")
    
    return "\n".join(memories) if memories else "No memories saved yet."

def tool_daily_log(content, group_id=None):
    """
    Add an entry to today's daily log.
    
    Automatically creates a daily file and appends timestamped entries.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    daily_dir = os.path.join(MEMORY_DIR, "daily")
    os.makedirs(daily_dir, exist_ok=True)
    
    file_path = os.path.join(daily_dir, f"{today}.md")
    timestamp = datetime.now().strftime("%H:%M")
    
    prefix = f"[{group_id}] " if group_id else ""
    entry = f"\n- **{timestamp}** {prefix}{content}"
    
    if os.path.exists(file_path):
        with open(file_path, "a") as f:
            f.write(entry)
    else:
        with open(file_path, "w") as f:
            f.write(f"# Daily Log - {today}\n{entry}")
    
    return f"Logged to daily notes."

# Add these tools to the agent's tool list:
MEMORY_TOOLS = [
    {
        "name": "save_memory",
        "description": "Save important information to long-term memory for later recall. Use this when you learn something worth remembering about users, projects, preferences, or decisions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Topic/name for this memory (e.g., 'user_preferences', 'project_deadlines')"},
                "content": {"type": "string", "description": "The information to remember"},
                "category": {"type": "string", "enum": ["topics", "people", "daily"], "description": "Memory category"}
            },
            "required": ["key", "content"]
        }
    },
    {
        "name": "search_memory",
        "description": "Search your long-term memory for previously saved information. Use this when you need to recall something you were told before.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for"},
                "category": {"type": "string", "description": "Optional category filter"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "list_memories",
        "description": "List all topics stored in memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Optional category filter"}
            }
        }
    }
]
```

### How NanoClaw Does It

NanoClaw's memory system uses the same file-based approach. Memory files are stored in the workspace directory and organized by category. The agent has tools to save, search, and list memories.

One area where NanoClaw adds sophistication is **automatic memory extraction**. After certain conversations (particularly ones where the user shares personal information, preferences, or important facts), the agent can proactively save relevant information to memory without being explicitly asked. This mimics how a good human assistant would think: "Oh, they mentioned their daughter's birthday is next week — I should remember that."

OpenClaw goes even further with its `MEMORY.md` file — a curated, long-term memory document that the assistant periodically reviews and updates. It distinguishes between raw daily logs (the `memory/YYYY-MM-DD.md` files) and curated long-term memory (the distilled insights that survive past the daily noise). This mirrors how human memory works: you remember the important stuff, not every detail of every day.

The search implementation in our version is deliberately simple — keyword matching with basic scoring. For a more sophisticated system, you could add:
- TF-IDF scoring
- Embedding-based semantic search (using an embedding model)
- Date-aware relevance (recent memories score higher)
- Cross-reference tracking (linking related memories)

But the simple version works surprisingly well. Most memory queries are direct ("what did I say about the project deadline?") and keyword matching handles them fine.

### Key Insight

> **File-based memory is simultaneously the simplest and most flexible approach.** By storing memories as markdown files, you get human-readable data (you can inspect your assistant's memories with any text editor), easy backups (just copy the directory), natural organization (directories as categories, files as topics), and version control compatibility (git diff your assistant's memories over time). The alternative — storing memories in a vector database — adds complexity, requires additional infrastructure, and produces memories that are opaque to human inspection. Start with files. You'd be surprised how far they take you.

The "append, don't overwrite" pattern for memory files is important. When the assistant learns something new about a topic, it adds to the existing file rather than replacing it. This creates a timestamped log of evolving knowledge. The topic file for "user_preferences" might have entries from six months ago and yesterday, and the agent can see how preferences have changed over time.

---

## Chapter 11: Wiring It All Together — The Full System

### The Problem

We've built eleven separate components. Now we need to make them work together as a cohesive system. This is where software architecture matters — not as an abstract exercise, but as the practical question of "what starts first, what depends on what, and how do errors propagate?"

### The Code

```python
#!/usr/bin/env python3
"""
AI Assistant — A production-grade multi-channel AI assistant.
Inspired by NanoClaw (github.com/qwibitai/nanoclaw).
"""

import os
import sys
import time
import signal
import logging
import threading
import argparse

# --- Configuration ---
CONFIG = {
    "model": os.environ.get("MODEL", "claude-sonnet-4-20250514"),
    "max_concurrent_groups": int(os.environ.get("MAX_CONCURRENT", "3")),
    "poll_interval": float(os.environ.get("POLL_INTERVAL", "2.0")),
    "scheduler_interval": float(os.environ.get("SCHEDULER_INTERVAL", "60")),
    "database_path": os.environ.get("DATABASE_PATH", "assistant.db"),
    "workspace_root": os.environ.get("WORKSPACE_ROOT", "workspace"),
    "http_port": int(os.environ.get("HTTP_PORT", "8080")),
    "whatsapp_bridge_url": os.environ.get("WHATSAPP_BRIDGE", "http://localhost:3001"),
    "log_level": os.environ.get("LOG_LEVEL", "INFO"),
}

# --- Logging ---
logging.basicConfig(
    level=getattr(logging, CONFIG["log_level"]),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("assistant")

# --- Graceful Shutdown ---
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    logger.info("Shutdown signal received. Stopping...")
    shutdown_event.set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- Component Assembly ---
def create_components():
    """Initialize all system components in dependency order."""
    
    # 1. Database (everything depends on this)
    logger.info("Initializing database...")
    init_database()
    init_scheduler_tables()
    
    # 2. Memory system
    logger.info("Initializing memory...")
    init_memory()
    
    # 3. Group queue (depends on database for history)
    logger.info("Initializing group queue...")
    group_queue = GroupQueue(max_concurrent=CONFIG["max_concurrent_groups"])
    
    # 4. Agent core (depends on group queue)
    logger.info("Initializing agent core...")
    agent_core = AgentCore()
    agent_core.group_queue = group_queue
    
    # 5. Gateway (depends on agent core)
    logger.info("Initializing gateway...")
    gateway = Gateway()
    gateway.core = agent_core
    
    return {
        "group_queue": group_queue,
        "agent_core": agent_core,
        "gateway": gateway,
    }

def start_background_services(components):
    """Start all background threads and services."""
    threads = []
    
    # Scheduler thread
    def scheduler_worker():
        logger.info("Scheduler started (checking every %ds)", CONFIG["scheduler_interval"])
        while not shutdown_event.is_set():
            try:
                due_tasks = get_due_tasks()
                for task in due_tasks:
                    try:
                        execute_task(task)
                        update_task_after_run(task)
                        logger.info("Executed task %d: %s", task["id"], task.get("description", "unknown"))
                    except Exception as e:
                        logger.error("Task %d failed: %s", task["id"], e)
            except Exception as e:
                logger.error("Scheduler error: %s", e)
            shutdown_event.wait(CONFIG["scheduler_interval"])
    
    scheduler_thread = threading.Thread(target=scheduler_worker, name="scheduler", daemon=True)
    scheduler_thread.start()
    threads.append(scheduler_thread)
    
    # IPC handler thread (for sandboxed agents)
    def ipc_worker():
        ipc = FileIPC("workspace/ipc")
        handler = HostIPCHandler(ipc)
        logger.info("IPC handler started")
        while not shutdown_event.is_set():
            try:
                handler.process_requests()
            except Exception as e:
                logger.error("IPC error: %s", e)
            shutdown_event.wait(0.5)
    
    ipc_thread = threading.Thread(target=ipc_worker, name="ipc", daemon=True)
    ipc_thread.start()
    threads.append(ipc_thread)
    
    return threads

def main():
    """Main entry point — wire everything together and run."""
    parser = argparse.ArgumentParser(description="AI Assistant")
    parser.add_argument("--terminal", action="store_true", help="Enable terminal interface")
    parser.add_argument("--http", action="store_true", help="Enable HTTP API")
    parser.add_argument("--whatsapp", action="store_true", help="Enable WhatsApp bridge")
    parser.add_argument("--port", type=int, default=CONFIG["http_port"], help="HTTP port")
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("AI Assistant starting up")
    logger.info("Model: %s", CONFIG["model"])
    logger.info("Max concurrent groups: %d", CONFIG["max_concurrent_groups"])
    logger.info("=" * 50)
    
    # Initialize components
    components = create_components()
    gateway = components["gateway"]
    
    # Add channel adapters based on arguments
    if args.terminal:
        gateway.add_adapter(TerminalAdapter)
    
    if args.http:
        gateway.add_adapter(HTTPAdapter, port=args.port)
    
    if args.whatsapp:
        gateway.add_adapter(WhatsAppAdapter, bridge_url=CONFIG["whatsapp_bridge_url"])
    
    # If no adapters specified, default to terminal
    if not (args.terminal or args.http or args.whatsapp):
        logger.info("No adapters specified, defaulting to terminal")
        gateway.add_adapter(TerminalAdapter)
    
    # Start background services
    bg_threads = start_background_services(components)
    
    # Start the gateway (this starts all channel adapters)
    gateway.start()
    
    # Wait for shutdown
    logger.info("System ready. Waiting for messages...")
    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(1.0)
    except KeyboardInterrupt:
        pass
    
    logger.info("Shutdown complete. Goodbye! 👋")

if __name__ == "__main__":
    main()
```

### How NanoClaw Does It

NanoClaw's `main.ts` follows a remarkably similar pattern: initialize the database, set up the WhatsApp bridge, start the polling loop, start the scheduler, and wait. The TypeScript version uses `async/await` instead of threading, which gives it a slightly cleaner structure (no daemon threads, no shutdown events), but the logical flow is identical.

One thing NanoClaw does well is **startup validation**. Before starting the main loop, it checks:
- Is the database accessible?
- Is the WhatsApp bridge running?
- Are API keys configured?
- Is the workspace directory writable?

If any check fails, it logs a clear error message and exits instead of starting up in a broken state. This is a small investment that saves hours of debugging.

NanoClaw also implements **graceful shutdown** — when it receives SIGTERM, it finishes processing any in-flight agent turns before exiting. This prevents corrupted conversation state and ensures tool executions complete cleanly.

### Key Insight

> **The architecture of the final system is the architecture we've been building all along.** There's no surprise integration step, no glue code, no framework. The polling loop polls. The group queue queues. The agent loop loops. The scheduler schedules. Each component does one thing, communicates through clear interfaces, and can be tested independently. This is not accidental — it's the result of building each component with clear boundaries from the start.

Notice the initialization order matters: database first (everything depends on it), then memory (which writes to the filesystem), then queue (which reads from the database), then agent core (which uses the queue), then gateway (which routes to the agent core). Reversing any of these would cause startup failures. Make dependencies explicit.

---

## Comparison: NanoClaw vs Our Python Version vs OpenClaw

Now that we've built the complete system, let's see how it stacks up:

| Feature | NanoClaw (TypeScript) | Our Version (Python) | OpenClaw |
|---|---|---|---|
| **Language** | TypeScript | Python | TypeScript/Node.js |
| **Lines of Code** | ~2,000 | ~1,500 | 50,000+ |
| **Messaging** | WhatsApp (Baileys) | Multi-channel (gateway) | Discord, WhatsApp, HTTP, Terminal |
| **Database** | SQLite | SQLite | SQLite + structured files |
| **Agent Loop** | Streaming + tool use | Tool use (non-streaming) | Full streaming + thinking + tools |
| **Isolation** | Linux containers (LXC) | Subprocess (with container option) | Full container orchestration |
| **Concurrency** | Per-group locks | Per-group queue + semaphore | Per-group with priority queues |
| **Scheduling** | SQLite-based cron | SQLite-based cron | Full cron with timezone support |
| **Memory** | File-based | File-based (categorized) | File-based + MEMORY.md curation |
| **IPC** | File-based JSON | File-based JSON | Message bus + file-based |
| **Per-group config** | CLAUDE.md files | CLAUDE.md files | AGENTS.md + SOUL.md + USER.md |
| **Browser control** | No | No | Full Playwright integration |
| **Node pairing** | No | No | Yes (phone, desktop nodes) |
| **Canvas/UI** | No | No | Yes (web canvas rendering) |
| **TTS/Voice** | No | No | Yes (ElevenLabs integration) |
| **Production ready?** | MVP | Learning/MVP | Yes |
| **Setup complexity** | npm install | pip install (stdlib only) | Full installation + config |

**NanoClaw** is the architectural blueprint. It proves these patterns work in a real system that handles actual WhatsApp conversations. It's minimal by design — the point is to show the essential architecture without drowning in features.

**Our Python version** takes the same patterns and makes them accessible to Python developers. We've added the gateway pattern for multi-channel support and categorized memory, but kept everything in the standard library (or close to it). It's a learning platform and a foundation for customization.

**OpenClaw** is the production platform. It takes every pattern we've discussed and adds the polish, security, and feature breadth needed for real-world deployment. Browser control, node pairing, voice synthesis, canvas rendering — these are all features you'd eventually want but don't need to understand the fundamentals.

---

## Closing Thoughts: What We've Actually Built

Let's take a step back and appreciate what we've created. Starting from nothing — not even a framework — we built:

1. **A reliable message ingestion system** (polling loop) that handles network failures gracefully and batches messages naturally.

2. **A persistent data layer** (SQLite) that stores conversations, sessions, and scheduled tasks with proper indexing and concurrent access.

3. **A customizable personality system** (per-group CLAUDE.md) that lets each conversation space have its own character, tools, and knowledge.

4. **An intelligent agent core** (agent loop) that can use tools, chain multiple steps, and handle errors without getting stuck.

5. **A security boundary** (container/subprocess isolation) that prevents the agent from damaging the host system, even if it makes mistakes.

6. **A concurrency model** (group queue) that processes multiple conversations in parallel while maintaining per-conversation consistency.

7. **A task scheduling system** (cron-like scheduler) that lets the assistant take proactive action on a timed basis.

8. **A secure communication protocol** (file-based IPC) that bridges the sandboxed agent and the host with full policy enforcement.

9. **A multi-channel architecture** (gateway pattern) that decouples the agent's intelligence from its communication channels.

10. **A long-term memory system** (file-based storage with search) that gives the assistant persistent knowledge across conversations and restarts.

11. **A clean startup and shutdown sequence** that initializes components in dependency order and shuts down gracefully.

Each of these components is individually simple. The polling loop is a while loop with sleep. The database is a few CREATE TABLE statements. The agent loop is a while loop that calls an API. The scheduler is a table with timestamps.

But together? Together they form something that feels genuinely intelligent. An assistant that's always available, remembers your preferences, takes initiative, handles multiple conversations without getting confused, runs code safely, and keeps working even when things go wrong.

That's the real lesson of NanoClaw and of building your own assistant: **the magic isn't in any single component. It's in how they compose.** Each component is boring. Together, they're remarkable.

The code in this post is real, working Python. You can take it, assemble it into a single project, and run it. Start with the terminal adapter. Play with the agent loop. Add a tool. Schedule a reminder. Give a group a personality. Watch it remember things across restarts.

And when you're ready for production? That's where [OpenClaw](https://openclaw.ai) comes in — taking every pattern we've explored here and hardening it for real-world use. But now you'll understand what's happening under the hood.

Build the thing. Understand the thing. Then decide if you need the big version or if your version is enough.

Happy building. 🛠️

---

*This post was inspired by [NanoClaw](https://github.com/qwibitai/nanoclaw), a minimal open-source AI assistant framework by QwibitAI. NanoClaw is a great starting point if you want to see these patterns implemented in TypeScript. Star it on GitHub if you found this useful.*

*For the full-featured platform, check out [OpenClaw](https://openclaw.ai) — the production AI assistant framework that takes these architectural patterns to their logical conclusion.*
