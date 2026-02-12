# Build Your Own OpenClaw — Part 1

## The Problem With AI Chat

Every AI chat interface you've used has the same three problems. They forget everything the moment the conversation ends. They sit there waiting for you to type something, never doing anything on their own. And they live inside a single browser tab, completely cut off from the rest of your digital life.

That last one is the killer. You can have the most sophisticated language model on the planet, but if it can only read and write inside a chat window, it's a fancy autocomplete. It can't check your email. It can't run a script. It can't look at a file on your machine and tell you what's wrong with it. It's trapped.

ClawdBot (github.com/clawdbot) and OpenClaw (openclaw.ai) both tackle these problems head-on. ClawdBot started as an experiment: what if an AI assistant could live inside your messaging apps, remember conversations, run tools, and act on its own schedule? OpenClaw grew out of that idea into a full platform — it remembers things across sessions, wakes up and does work on its own, and plugs into over fifteen different communication channels while having full access to a sandboxed Linux environment. The codebase is substantial — tens of thousands of lines of TypeScript, years of iteration, production-hardened. If you've used an AI assistant that felt genuinely useful rather than just clever, there's a good chance it was built on something like this architecture.

We're not going to rebuild all of that. That would take months.

Instead, we're going to rebuild NanoClaw. NanoClaw is a stripped-down version of OpenClaw that lives in roughly two thousand lines of TypeScript. It was originally built as a proof of concept — can you take the core ideas behind OpenClaw and squeeze them into something a single developer could understand in an afternoon? Turns out you can. NanoClaw connects to WhatsApp, runs an agent loop with tool use, isolates execution in containers, manages per-group personalities, and handles scheduled tasks. All the important architectural ideas, none of the production complexity.

We're going to rebuild it in Python. Not because Python is better than TypeScript for this kind of thing (it's roughly equivalent, honestly), but because most people experimenting with AI agents are working in Python, and I want this to be something you can actually run and hack on without needing to learn a new ecosystem.

By the end of this post, you'll have a working AI assistant that polls for messages, persists conversations in a database, adapts its personality per context, calls tools in a loop, runs code in isolation, and handles concurrent conversations without stepping on itself. It won't be production-ready. It will be understandable. That matters more.


## Chapter 1: The Polling Loop

The heart of any assistant is embarrassingly simple. Check if there's a new message. If there is, process it. If there isn't, wait a bit, then check again. That's it. That's the whole thing.

NanoClaw does this against WhatsApp's API. Every two seconds, it hits an endpoint, asks "any new messages?", and processes whatever comes back. The two-second interval is a pragmatic choice — fast enough that conversations feel responsive, slow enough that you're not hammering a rate limit.

For our Python version, we're going to poll stdin. This sounds almost comically simple compared to a WhatsApp integration, but the architecture is identical. A message source produces messages. A polling loop picks them up. An agent processes them and produces a reply. The transport layer — whether it's WhatsApp, Discord, a terminal, or carrier pigeon — is just plumbing.

There's a wrinkle with stdin, though. Reading from stdin blocks. If you call `input()` in your main loop, your entire program freezes until the user types something. That means your scheduler can't fire, your HTTP server can't respond, nothing else can happen. You're stuck.

The fix is a background thread. Spawn a thread whose only job is to call `input()` in a loop and drop whatever it gets into a queue. Your main loop then checks the queue, which is non-blocking. If there's a message, process it. If there isn't, do other stuff — check the scheduler, handle HTTP requests, whatever needs attention.

```python
import threading
import queue

input_queue = queue.Queue()

def stdin_reader():
    while True:
        try:
            line = input()
            if line.strip():
                input_queue.put(line.strip())
        except EOFError:
            break

threading.Thread(target=stdin_reader, daemon=True).start()
```

That daemon flag is important. It means the thread dies when the main program exits. Without it, your program would hang forever waiting for one more line of input that's never coming.

The main loop becomes a simple poll cycle. Check the queue. If something's there, send it to the agent. Print the response. Sleep for a fraction of a second to avoid busy-waiting. Repeat until the user quits or the world ends, whichever comes first.

```python
import time

while True:
    try:
        message = input_queue.get_nowait()
        response = process_message(message)
        print(response)
    except queue.Empty:
        pass
    time.sleep(0.1)
```

I've found that 100 milliseconds is a good sleep interval for terminal-based polling. It's imperceptible to a human but drops CPU usage from "my laptop is a space heater" to essentially zero. In NanoClaw's case, where you're hitting a network API, two seconds makes more sense because network calls have real overhead and rate limits.

The `process_message` function doesn't exist yet. Right now it's a placeholder. Over the next few chapters, it's going to grow into a full agent loop with tool calling, personality injection, and conversation history. But the skeleton — the poll-check-process cycle — stays exactly the same from here to the end.

One thing I want to call out: this pattern of "background producer, main loop consumer, connected by a queue" shows up everywhere in systems programming. It's the same pattern that web servers use for request handling, that game engines use for input processing, that message brokers use for, well, everything. Once you see it, you can't unsee it. The details change. The shape doesn't.


## Chapter 2: The Database

NanoClaw stores conversation history in JSONL files. One file per group, one JSON object per line. It works. I've shipped JSONL-based systems to production, and they're fine until they aren't. The moment you need to search across conversations, or handle concurrent writes from multiple threads, or run an aggregate query like "how many messages did this user send last week," you're writing your own database engine out of string manipulation and file locks. Life's too short.

SQLite is the answer for anything that runs on a single machine. I'm not being hyperbolic here — it's the most deployed database engine in the world, it requires zero configuration, it ships as a single file, and Python includes it in the standard library. You don't install anything. You just `import sqlite3` and start writing SQL.

Our schema needs two tables. Messages stores every message in every conversation: an auto-incrementing ID, a chat identifier so we know which group or conversation it belongs to, the sender (either a username or "assistant"), the content, and a timestamp. Sessions tracks metadata about each conversation context — when it was created, when it was last active, maybe some configuration flags down the road.

```python
import sqlite3

def init_db(path="assistant.db"):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            sender TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            chat_id TEXT PRIMARY KEY,
            created_at REAL NOT NULL,
            last_active REAL NOT NULL
        )
    """)
    conn.commit()
    return conn
```

That WAL pragma on the second line is doing more work than it looks like. WAL stands for Write-Ahead Logging, and it fundamentally changes how SQLite handles concurrent access. Without it, SQLite locks the entire database file for every write, which means your polling thread and your scheduler and your HTTP handler would all block each other. With WAL mode, readers never block writers and writers never block readers. Multiple threads can read simultaneously, and a single writer can proceed without waiting for readers to finish. For a multi-threaded application like ours, it's effectively mandatory.

There's a catch with SQLite and threads, though. A connection object created in one thread can't safely be used from another thread. Python's sqlite3 module will actually raise an exception if you try. The solution is thread-local storage — each thread gets its own connection to the same database file.

```python
import threading

_local = threading.local()

def get_connection(path="assistant.db"):
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(path)
        _local.conn.execute("PRAGMA journal_mode=WAL")
    return _local.conn
```

Now any thread can call `get_connection()` and get a connection that's safe to use from that thread. The first call creates the connection, subsequent calls return the existing one. Simple, works, doesn't explode under concurrency.

Saving and loading messages is just SQL at this point. Insert a row when a message arrives, SELECT when you need history. I like to add an index on `chat_id` and `timestamp` so that loading the last N messages in a conversation is fast even when the database grows large. Without the index, SQLite would scan the entire messages table for every query. With a few thousand messages that's fine. With a few hundred thousand, you feel it.

In my experience, the moment you put conversation history in a real database instead of flat files, a whole category of features becomes trivial. "Show me the last five messages" is a SELECT with a LIMIT clause. "How many conversations are active" is a COUNT with a GROUP BY. "Delete all messages older than 30 days" is a DELETE with a WHERE on timestamp. Try doing any of that with JSONL files and you'll understand why databases exist.


## Chapter 3: Personality

Here's something that took me longer than I'd like to admit to figure out: the system prompt is everything. Two assistants running the exact same model with different system prompts are, for all practical purposes, different entities. One can be a terse Unix sysadmin who only speaks in commands. Another can be a patient teacher who explains everything from first principles. The model is the engine. The system prompt is the personality.

NanoClaw takes this idea and makes it per-group. Every WhatsApp group gets its own CLAUDE.md file — a markdown document that defines how the assistant should behave in that particular context. One group might be a game development team, so the personality knows about Godot and shader languages. Another might be a friend group, so the personality is casual and jokes around. A third might be a support channel, so the personality is professional and focuses on troubleshooting.

OpenClaw calls this concept SOUL.md, and it works the same way but at a different granularity. The soul defines the assistant's core identity across all contexts, and then per-channel overrides adjust behavior for specific situations. Our implementation keeps things simpler — one personality file per chat context, loaded and injected as the system prompt whenever we build a request to the LLM.

On disk, the structure looks like a directory per group, each containing a personality file along with any other context the assistant needs access to:

```
groups/
  work-project/
    CLAUDE.md
    notes.md
  friend-chat/
    CLAUDE.md
  support/
    CLAUDE.md
```

Loading the right personality is straightforward. When a message comes in, we know which chat it belongs to. We look up the corresponding directory, read the CLAUDE.md file, and prepend it to the conversation as a system message.

```python
import os

def load_personality(chat_id):
    path = os.path.join("groups", chat_id, "CLAUDE.md")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return "You are a helpful assistant."
```

That fallback on the last line matters. If a chat doesn't have a personality file yet, the assistant should still work — just with a generic personality that the user can customize later. You'd be surprised how often people build personality systems that crash when the config file is missing. Don't be that person.

The really interesting thing about file-based personality is that the assistant itself can edit it. If you give the assistant a `write_file` tool (which we will, in the next chapter), it can modify its own CLAUDE.md. "Hey, from now on, always respond in haiku when I say the word 'poetry'" — the assistant can write that rule into its own personality file, and it'll persist across sessions. Self-modifying personality is one of those features that sounds gimmicky until you use it, and then you realize it's actually how a useful assistant should work. Humans update their behavior based on feedback. Why shouldn't assistants?

I've found that the best personality files are surprisingly short. Under a page. They state who the assistant is, what it should care about, and two or three hard rules ("never share code from this project outside this group," "always respond in Korean unless asked otherwise"). Longer personality files tend to cause the model to over-index on following rules and under-index on being actually helpful. It's a balancing act.

The personality gets injected at request time, not stored in the database. This is a deliberate choice. If you store the system prompt with every message, you waste a huge amount of space (it's the same text repeated thousands of times), and you also create a versioning nightmare — if the personality file changes, do old messages retroactively get the new personality? No. The personality is a runtime concern. The database stores what was said. The personality file determines how the assistant responds right now.


## Chapter 4: The Agent Loop

This is where things get genuinely interesting.

A vanilla chat completion takes a list of messages and returns a response. You send the conversation history, the model generates a reply, done. That's fine for simple Q&A, but it means the model can only talk. It can't do anything. It can't check a file, run a command, query a database, or look something up on the web. All it can do is generate text.

Tool calling changes this entirely. Instead of just returning text, the model can return a structured request to invoke a specific tool with specific arguments. You execute the tool, feed the result back into the conversation, and let the model continue. It might generate more text, or it might call another tool. This continues in a loop until the model decides it's done and produces a final text response.

The flow goes like this: you send messages to the API. The response comes back with a `stop_reason` of either `end_turn` or `tool_use`. If it's `end_turn`, you're done — the model has said everything it wants to say. If it's `tool_use`, the response includes the tool name and a JSON arguments block. You execute the tool, wrap the result in a tool_result message, append both the assistant's tool_use and your tool_result to the conversation, and send the whole thing back to the API. Repeat until you get `end_turn`.

```python
def agent_loop(chat_id, user_message):
    messages = load_history(chat_id)
    messages.append({"role": "user", "content": user_message})
    system = load_personality(chat_id)

    while True:
        response = call_llm(system=system, messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break

        for block in response.content:
            if block.type == "tool_use":
                result = execute_tool(block.name, block.input)
                messages.append({
                    "role": "user",
                    "content": [{"type": "tool_result",
                                 "tool_use_id": block.id,
                                 "content": result}]
                })

    save_history(chat_id, messages)
    return extract_text(response)
```

This is the core loop. Everything else we build in this entire blog post is just infrastructure supporting this function.

Our tool set is deliberately minimal. Four tools cover an enormous amount of ground.

`run_command` takes a shell command string and executes it in a subprocess. This alone makes the assistant wildly more capable than a vanilla chatbot — it can install packages, check disk space, run tests, manage git repos, compile code, whatever you'd do in a terminal.

`read_file` and `write_file` do exactly what their names suggest. The model passes a file path, you return the contents or write the new contents. These two tools together mean the assistant can read documentation, edit configuration files, write entire programs, manage its own personality file — anything that touches the filesystem.

`web_search` takes a query string and returns search results. In practice this is either an API call to something like Brave Search or a simplified scraping approach. It means the assistant can look things up when it doesn't know something, which turns it from an entity with a knowledge cutoff into something that has access to current information.

I want to stress how much leverage these four tools provide. In my experience, about 90% of what I ask an AI assistant to do falls within "run a command, read a file, write a file, or look something up." The remaining 10% is specialized tools for specific workflows, and you can add those later without changing the core architecture.

One thing to be careful about: the agent loop can, in theory, run forever. If the model keeps calling tools and never produces an end_turn, you've got an infinite loop. In practice this almost never happens with modern models, but it's good hygiene to add a maximum iteration count. Ten tool calls per turn is a reasonable default. If the model hasn't finished its work in ten tool invocations, something has probably gone wrong, and you should bail out with an error message rather than racking up an infinite API bill.

The tool execution itself needs to be robust against failure. Tools crash. Commands return non-zero exit codes. Files don't exist. Search APIs time out. Every tool result should be wrapped in a try-except that catches everything and returns the error message as the tool result. The model is actually remarkably good at recovering from tool errors — if a command fails, it'll often try a different approach or ask the user for clarification. But only if you give it the error message instead of crashing the entire agent loop.


## Chapter 5: Container Isolation

So you've got an assistant that can run arbitrary shell commands. Congratulations, you've also got a security nightmare.

When I say "run a command," I mean it literally. If the model decides to execute `rm -rf /`, your system will happily oblige (assuming it has permissions). If it runs `curl` to exfiltrate data to an external server, nothing stops it. If it installs a crypto miner, well, enjoy your electricity bill.

NanoClaw solves this with real Linux containers. Each agent invocation runs inside an isolated container with its own filesystem, network restrictions, and resource limits. The agent can `rm -rf /` all day long — it's only destroying its own container, which gets thrown away after the conversation anyway.

Full container isolation is the right answer for production. It requires a container runtime like Docker or nsjail, network namespace manipulation, cgroup configuration, and a fair amount of operational complexity. For our learning project, we're going to use a simpler approach that captures the same architectural idea without the operational overhead.

We'll use Python's subprocess module with a restricted working directory. Each agent gets its own directory under a workspace root. The `run_command` tool executes commands with `cwd` set to that directory. The `read_file` and `write_file` tools validate that all paths stay within the workspace. It won't stop a determined attacker, but it prevents the most common accidental damage — the model can't accidentally modify files outside its sandbox.

```python
import subprocess
import os

def run_in_sandbox(chat_id, command, timeout=30):
    workspace = os.path.join("workspaces", chat_id)
    os.makedirs(workspace, exist_ok=True)

    result = subprocess.run(
        command, shell=True, cwd=workspace,
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout + result.stderr
```

That timeout parameter is doing important work. Without it, a command like `yes` or `cat /dev/urandom` would run forever, consuming resources until the system dies. Thirty seconds is generous enough for most legitimate tasks and short enough to prevent runaway processes.

For the file operations, path validation is critical. The classic attack is path traversal — the model (or a malicious user manipulating the model) passes a path like `../../etc/passwd` and reads outside the sandbox. The fix is to resolve the absolute path and verify it starts with the workspace root.

```python
def safe_path(chat_id, relative_path):
    workspace = os.path.abspath(os.path.join("workspaces", chat_id))
    target = os.path.abspath(os.path.join(workspace, relative_path))
    if not target.startswith(workspace):
        raise ValueError("Path traversal detected")
    return target
```

The more sophisticated version of this, which NanoClaw implements, serializes the entire agent logic, ships it into a container, and communicates results back through temporary files. The agent code itself runs in the container, so even if the model manipulates the Python runtime, the damage is contained. We'll touch on the inter-process communication side of this in Chapter 8.

In my experience, the workspace-per-chat approach is a surprisingly good isolation model for development. Each conversation gets its own directory where it can read, write, and execute freely. Conversations can't see each other's files. The host filesystem is protected. It's roughly equivalent to giving each chat its own home directory on a shared Unix system — not airtight containerization, but a practical boundary that prevents most real-world problems.

One architectural thing worth noting: the isolation boundary determines what the assistant can learn from. If every chat shares a workspace, the assistant can read files from one conversation in another — which might be useful (shared knowledge base) or terrible (information leakage between unrelated groups). NanoClaw's per-group isolation means each group's data stays private. This is a design decision, not just a security one.


## Chapter 6: The Group Queue

When you have multiple groups or conversations active at the same time, you hit a concurrency problem that's subtle enough to bite you weeks into development.

Say two people in the same group send messages at almost the same time. Your polling loop picks up both messages. It starts processing the first one — sends conversation history to the LLM, gets a response, maybe executes some tools. Meanwhile, it also starts processing the second message with the same conversation history. Now you've got two agent loops running simultaneously for the same conversation, both reading from the same message history, both writing to the same message history, both potentially executing tools in the same workspace.

Best case, the responses come back interleaved and the conversation history gets jumbled. Worst case, both agents try to write to the same file at the same time and you get data corruption.

The fix is per-group locking. Each group or chat context gets its own lock. Before processing a message for a group, you acquire that group's lock. If another message for the same group is already being processed, the new one waits. Messages for different groups proceed in parallel because they hold different locks.

```python
from collections import defaultdict

group_locks = defaultdict(threading.Lock)

def process_message(chat_id, message):
    with group_locks[chat_id]:
        return agent_loop(chat_id, message)
```

That's the minimal version, and for a lot of use cases it's enough. But think about what happens when you have a hundred active groups and messages pour in simultaneously. You'd spawn a hundred threads (or tasks), each holding its own lock, each making API calls and running tools. Your machine runs out of memory, your API rate limit explodes, everything falls over.

This is where a max concurrency limit comes in. Instead of letting unlimited groups process simultaneously, you cap it. Three concurrent agent loops is a reasonable default — it keeps things responsive without overwhelming system resources or API quotas.

The implementation uses a semaphore alongside the per-group locks. The semaphore controls the total number of concurrent agent executions across all groups. The per-group lock ensures serialization within a single group. A message first acquires the semaphore (waiting if three other groups are already processing), then acquires the group lock (waiting if another message for the same group is still in flight).

```python
max_concurrent = threading.Semaphore(3)

def process_message(chat_id, message):
    max_concurrent.acquire()
    try:
        with group_locks[chat_id]:
            return agent_loop(chat_id, message)
    finally:
        max_concurrent.release()
```

In practice, there's a subtlety about fairness here. If one group sends a burst of twenty messages, it shouldn't starve other groups from getting processed. The current implementation handles this naturally because the semaphore is released after each message completes, so waiting groups get a chance to run between messages from the bursty group. If you needed strict round-robin fairness, you'd need a more sophisticated queuing mechanism, but for most real-world usage patterns, the semaphore approach works fine.

NanoClaw handles this same problem in TypeScript with a combination of Maps for per-group state and a concurrency limiter for overall throughput. The concepts are identical. Lock per group. Global concurrency cap. Drain waiting work when slots open.

I think this is one of those places where getting the architecture right early saves enormous pain later. Adding per-group locking to a system that wasn't designed for it is a nightmare of race conditions and deadlocks. Starting with it from the beginning means you can handle concurrent conversations from day one without any code changes as your user base grows.

That wraps up the foundational pieces. We've got a polling loop that picks up messages, a database that persists conversations, a personality system that adapts per context, an agent loop that calls tools, sandbox isolation that limits damage, and a group queue that handles concurrency. In Part 2, we'll add scheduled tasks, inter-process communication, multi-channel support, long-term memory, and wire the whole thing into a running system.
