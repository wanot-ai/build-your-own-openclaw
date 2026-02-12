# Build Your Own OpenClaw — Part 2

## Chapter 7: The Task Scheduler

An assistant that only responds when you talk to it is like a coworker who only works when you're standing over their shoulder. Useful, sure. But a real assistant does things without being asked. It checks your calendar in the morning and tells you about conflicts. It monitors a log file and pings you when errors spike. It sends a daily summary at 6 PM whether you asked for one or not.

This is what the scheduler does. It takes prompts and runs them on a cron schedule.

NanoClaw stores scheduled tasks in its own configuration. OpenClaw uses a heartbeat model — the system wakes up on a regular interval and checks a file for pending work. Our approach splits the difference: tasks are stored as rows in SQLite with a cron expression, and a scheduler thread polls them every sixty seconds.

Each task needs three things: which group or context it belongs to, what prompt to send to the agent, and when to run it. The schedule follows standard cron syntax — five fields for minute, hour, day of month, month, and day of week. "0 9 * * 1" means nine in the morning every Monday. "*/30 * * * *" means every thirty minutes. The format is ancient, ugly, and universally understood by anyone who's ever administered a Unix system.

```python
def init_scheduler_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            schedule TEXT NOT NULL,
            last_run REAL,
            enabled INTEGER DEFAULT 1
        )
    """)
    conn.commit()
```

The scheduler thread wakes up every sixty seconds and iterates over enabled tasks. For each one, it parses the cron expression and checks whether the current time matches. If it does, and the task hasn't already run in the current minute, it fires. "Fires" means exactly what processing a regular message means — it calls the agent loop with the task's prompt as if a user had typed it, targeting the task's group context so the right personality and history apply.

```python
from croniter import croniter
import time

def scheduler_loop():
    while True:
        now = time.time()
        conn = get_connection()
        tasks = conn.execute(
            "SELECT id, chat_id, prompt, schedule, last_run "
            "FROM scheduled_tasks WHERE enabled = 1"
        ).fetchall()

        for task_id, chat_id, prompt, schedule, last_run in tasks:
            cron = croniter(schedule, now)
            prev = cron.get_prev(float)
            if last_run is None or prev > last_run:
                process_message(chat_id, prompt)
                conn.execute(
                    "UPDATE scheduled_tasks SET last_run = ? WHERE id = ?",
                    (now, task_id)
                )
                conn.commit()

        time.sleep(60)
```

Using `croniter` here saves us from parsing cron expressions ourselves, which is one of those problems that seems simple until you try to handle all the edge cases. The `get_prev` call asks "when was the most recent time this schedule should have fired?" If that time is after the last recorded run, we need to fire now.

The sixty-second poll interval means scheduled tasks might run up to a minute late. For an AI assistant sending a daily summary, nobody cares. If you needed sub-second precision, you'd use a different approach — a priority queue sorted by next-fire-time, with the thread sleeping until the next event. For our purposes, polling every minute is fine and much simpler to reason about.

I've found that scheduled tasks are one of those features that transform how people use an assistant. Without them, the assistant is reactive. With them, it becomes proactive. A simple "check for new emails every two hours and summarize anything important" task turns a chat-based assistant into something that genuinely reduces your cognitive load. You're not polling your inbox anymore. Your assistant is doing it for you.

One design decision worth highlighting: tasks go through the same `process_message` function as user-initiated messages. They hit the same group lock, the same concurrency semaphore, the same agent loop. A scheduled task and a user message are architecturally identical. This means every feature we add — tool calling, memory, isolation — automatically applies to scheduled tasks too, with zero additional code.


## Chapter 8: Inter-Process Communication

When you run agent code in a separate process (or a container, as NanoClaw does), you need a way for the agent and the host to talk to each other. The agent needs to send messages back to the user. The host needs to deliver follow-up messages to the agent. And somebody needs to signal when the conversation is done and the process should shut down.

There are a dozen ways to do IPC. Unix sockets, named pipes, shared memory, message queues, HTTP servers, gRPC, stdin/stdout. NanoClaw chose the simplest possible approach: files.

The workspace directory includes an `ipc` subdirectory with a specific structure. The agent writes outgoing message files to `ipc/messages/`. Each file is a JSON object with the message content, target channel, and any metadata. The host watches that directory, picks up new files, delivers the messages to the appropriate channel, and deletes the files.

Going the other direction, when the host receives a follow-up message from the user while the agent is running, it writes the message to `ipc/input/`. The agent polls that directory for new input, reads whatever shows up, and incorporates it into its current conversation.

Shutdown signaling uses a sentinel file. When the host wants the agent to stop — maybe the user said "cancel" or the timeout expired — it writes an empty file called `_close` to the IPC directory. The agent checks for this file on every polling cycle, and when it appears, it wraps up whatever it's doing and exits cleanly.

```python
import json
import glob

def send_ipc_message(workspace, content, channel="default"):
    os.makedirs(os.path.join(workspace, "ipc", "messages"), exist_ok=True)
    msg_path = os.path.join(
        workspace, "ipc", "messages",
        f"{time.time_ns()}.json"
    )
    with open(msg_path, "w") as f:
        json.dump({"content": content, "channel": channel}, f)

def check_ipc_input(workspace):
    input_dir = os.path.join(workspace, "ipc", "input")
    if not os.path.exists(input_dir):
        return []
    messages = []
    for path in sorted(glob.glob(os.path.join(input_dir, "*.json"))):
        with open(path) as f:
            messages.append(json.load(f))
        os.remove(path)
    return messages

def should_close(workspace):
    return os.path.exists(os.path.join(workspace, "ipc", "_close"))
```

Using nanosecond timestamps for filenames guarantees ordering and avoids collisions. Two messages created in the same nanosecond on the same machine is theoretically possible but practically never happens. If you're paranoid, add a random suffix. I've never needed to.

The beauty of file-based IPC is debuggability. When something goes wrong — and things always go wrong — you can look at the files. They're right there on disk. Open them in a text editor. See exactly what the agent tried to send. See exactly what the host delivered. Compare timestamps. No packet captures, no socket inspectors, no log correlation. Just files.

The downside is performance. File I/O is orders of magnitude slower than shared memory or even Unix domain sockets. For an AI assistant where "fast" means "responds within a few seconds," this doesn't matter at all. You're bottlenecked on the LLM API call, which takes seconds. File I/O takes milliseconds. The hot path is the API call, and the IPC mechanism is nowhere near it.

There's an architectural insight here that goes beyond the implementation details. The IPC layer creates a clean boundary between "the thing that talks to users" and "the thing that runs agent logic." Either side can be replaced independently. You could swap the file-based IPC for a WebSocket connection without changing the agent code. You could swap the Python agent for one written in Rust without changing the host code. This is the kind of separation that seems like over-engineering when you have one process and one channel, and becomes essential the moment you have two of either.


## Chapter 9: Multi-Channel Support

NanoClaw talks to WhatsApp. Just WhatsApp. If you want to use it from a terminal, you don't. If you want to use it from Discord, you write a whole new integration. The WhatsApp connection is woven into the core of the application, and extracting it would require significant refactoring.

OpenClaw talks to over fifteen channels. Discord, Slack, Telegram, WhatsApp, email, SMS, a web interface, terminal, HTTP API, and more. Adding a new channel takes an afternoon because channel integration is a separate concern from agent logic. The system was designed for this from the start.

The pattern is called a gateway, and it's the same pattern that API gateways, message brokers, and chat aggregation services use. You define a common interface that all channels must implement. Each channel adapter translates between the channel's native protocol and the common interface. The agent only speaks the common interface. It doesn't know or care whether a message came from Discord or a terminal.

```python
class Channel:
    def poll(self):
        """Return list of new messages, each with chat_id and content."""
        raise NotImplementedError

    def send(self, chat_id, content):
        """Send a response to the given chat."""
        raise NotImplementedError
```

That's the interface. Two methods. Poll for incoming messages, send outgoing ones. Every channel implements these two methods and nothing else.

A terminal channel wraps the stdin reader we built in Chapter 1. Poll checks the input queue, send prints to stdout. Simple.

```python
class TerminalChannel(Channel):
    def __init__(self):
        self.queue = queue.Queue()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while True:
            try:
                line = input()
                if line.strip():
                    self.queue.put(line.strip())
            except EOFError:
                break

    def poll(self):
        messages = []
        while not self.queue.empty():
            try:
                messages.append({
                    "chat_id": "terminal",
                    "content": self.queue.get_nowait()
                })
            except queue.Empty:
                break
        return messages

    def send(self, chat_id, content):
        print(f"Assistant: {content}")
```

An HTTP channel wraps a simple web server. Poll checks a request queue that the server's handler populates. Send stores the response where the handler can return it.

```python
class HttpChannel(Channel):
    def __init__(self, port=8080):
        self.incoming = queue.Queue()
        self.responses = {}
        self.server = start_http_server(port, self.incoming, self.responses)

    def poll(self):
        messages = []
        while not self.incoming.empty():
            messages.append(self.incoming.get_nowait())
        return messages

    def send(self, chat_id, content):
        self.responses[chat_id] = content
```

The main loop becomes channel-agnostic. It iterates over all registered channels, polls each one, processes any messages that come in, and sends responses back through the originating channel.

```python
channels = [TerminalChannel(), HttpChannel(8080)]

while True:
    for channel in channels:
        for msg in channel.poll():
            response = process_message(msg["chat_id"], msg["content"])
            channel.send(msg["chat_id"], response)
    time.sleep(0.1)
```

Notice something about this code. There's no if-statement that checks which channel type a message came from. There's no channel-specific logic in the main loop at all. Adding a Discord channel means writing a DiscordChannel class and appending it to the list. The main loop doesn't change. The agent loop doesn't change. The database doesn't change. The scheduler doesn't change. Nothing changes except the new channel adapter.

This is the power of getting your abstractions right. When you identify the correct seam in your architecture and define a clean interface at that boundary, new features become additive instead of invasive. You're adding code, not changing code. In a codebase that changes constantly — and any AI assistant codebase changes constantly, because the whole field is moving at breakneck speed — that distinction is the difference between a maintainable system and a ball of mud.

I want to be honest about a tradeoff, though. The simple polling interface works well for straightforward request-response patterns. Real channel integrations are messier. Discord has slash commands, reactions, thread replies, voice channels. Telegram has inline queries, callbacks, edited messages. WhatsApp has read receipts, media messages, group metadata changes. Our two-method interface can't capture all of that richness. In OpenClaw's production codebase, the channel interface is considerably more complex, with hooks for message editing, reactions, file uploads, typing indicators, and more. The simple version here gets the architecture right. A production version would need to expand the interface significantly.


## Chapter 10: Memory

Conversations end. Context windows have limits. Sessions reset. Without explicit memory, your assistant develops amnesia every time the conversation exceeds the token limit or the process restarts.

The database we built in Chapter 2 stores conversation history, which handles the "within a session" case. But what about the information that should persist across sessions? The user's preferences, project context, decisions that were made weeks ago, that one server IP address the user mentioned once and will ask about again in three months. Conversation history is short-term memory. We need long-term memory.

The simplest possible approach, and the one we'll implement, is file-based memory. The assistant gets two tools: `save_memory` and `search_memory`. When the assistant encounters something worth remembering — the user mentions their timezone, or a key project decision gets made, or a complex debugging session reaches a conclusion — it calls `save_memory` with a topic and content. This writes a markdown file to a memory directory, organized by date and topic.

```python
def save_memory(chat_id, topic, content):
    mem_dir = os.path.join("groups", chat_id, "memory")
    os.makedirs(mem_dir, exist_ok=True)
    filename = f"{topic.lower().replace(' ', '_')}.md"
    path = os.path.join(mem_dir, filename)

    with open(path, "a") as f:
        timestamp = time.strftime("%Y-%m-%d %H:%M")
        f.write(f"\n## {timestamp}\n\n{content}\n")

    return f"Saved memory: {topic}"
```

Using append mode means memories accumulate over time rather than overwriting each other. The assistant might save a "server-config" memory multiple times as the configuration evolves, and each entry preserves the full history of changes. You can scroll through the file and see how things evolved. This is surprisingly useful for debugging — "why is the server configured this way?" can be answered by looking at the memory file and tracing the decisions.

Searching memory uses keyword matching. The assistant calls `search_memory` with a query, and we scan all memory files in the chat's memory directory for matching content.

```python
def search_memory(chat_id, query):
    mem_dir = os.path.join("groups", chat_id, "memory")
    if not os.path.exists(mem_dir):
        return "No memories found."

    results = []
    query_lower = query.lower()
    for filename in os.listdir(mem_dir):
        path = os.path.join(mem_dir, filename)
        with open(path) as f:
            content = f.read()
        if query_lower in content.lower():
            topic = filename.replace("_", " ").replace(".md", "")
            results.append(f"[{topic}]\n{content[:500]}")

    if not results:
        return "No matching memories found."
    return "\n---\n".join(results)
```

Keyword matching is the dumbest possible search algorithm. It finds "server" but not "machine." It finds "Python" but not "programming language." It has zero understanding of semantics. And you know what? For a personal assistant with a few dozen memory files, it works. The files are small. The scan is fast. The user's query usually contains the exact keyword that was used when the memory was saved, because it's the same human talking about the same stuff.

OpenClaw's production memory system uses vector embeddings. Every memory gets embedded into a high-dimensional vector space using a model like OpenAI's text-embedding-3. Search queries get embedded the same way, and results come back ranked by cosine similarity. This means searching for "deployment pipeline" can find memories about "CI/CD configuration" even though the words don't overlap, because the concepts are close in embedding space. It's a massive improvement in recall quality, and for a system with thousands of memories across dozens of groups, it's basically required.

Getting from keyword search to vector search is a weekend project. You add an embedding step when memories are saved, store the vectors in something like ChromaDB or just a numpy array, and replace the search function with a nearest-neighbor lookup. The architecture doesn't change at all — the tools still have the same interface, the agent still calls them the same way. Only the implementation behind the search improves.

Memory creates an interesting feedback loop with the personality system from Chapter 3. The personality file defines who the assistant is. Memories define what it knows. Together, they create an assistant that has both identity and knowledge persistence. Restart the process, wipe the conversation history, and the assistant still knows your name, your preferences, your project's architecture, the decisions you made last month. It picks up where it left off because the important information lives outside the ephemeral conversation.

In my experience, the key to making memory useful is giving the model good guidance about when to save memories. Without guidance, models either save everything (filling up the memory directory with trivia) or nothing (never calling the tool at all). A line in the personality file like "save important facts, decisions, and user preferences to memory when they come up in conversation — but don't save routine chit-chat" goes a long way.

The other thing I've learned is that memory retrieval should be automatic, not just on-demand. At the start of each conversation, you can load the most recent memories and inject them into the system prompt alongside the personality. This way the assistant doesn't have to be asked "do you remember X?" — it already has the recent context available. OpenClaw does something similar, loading relevant memories based on the conversation topic. Our simplified version could just load the last few memory entries, which covers the common case where recent context is most relevant.


## Chapter 11: Wiring It All Together

We have eight independent components. A polling loop, a database, a personality system, an agent loop with tools, sandbox isolation, a group queue, a task scheduler, and a memory system. Each one works on its own. The question is: how do they compose into a running system?

The answer is maybe fifty lines of startup code.

Initialize the database. Start the stdin reader thread. Start the scheduler thread. Optionally start an HTTP server. Enter the main loop. The main loop polls all channels, processes messages, and sleeps. That's the complete architecture, running in a single Python file.

```python
import signal
import sys

def main():
    init_db()
    init_scheduler_table(get_connection())

    channels = [TerminalChannel()]

    http_port = int(os.environ.get("HTTP_PORT", "0"))
    if http_port:
        channels.append(HttpChannel(http_port))

    scheduler_thread = threading.Thread(
        target=scheduler_loop, daemon=True
    )
    scheduler_thread.start()

    print("Assistant ready. Type a message.")

    while True:
        for channel in channels:
            for msg in channel.poll():
                response = process_message(
                    msg["chat_id"], msg["content"]
                )
                channel.send(msg["chat_id"], response)
        time.sleep(0.1)

if __name__ == "__main__":
    main()
```

Look at how little code this is. The complexity lives in the individual components. The wiring is trivial because each component has a clean interface and minimal dependencies on the others.

The startup order matters slightly. The database needs to be initialized before anything tries to read or write messages. The scheduler needs the database tables to exist. The channels need to be set up before the main loop starts polling them. Beyond that, the ordering is flexible.

Everything runs in-process. The scheduler is a daemon thread. The stdin reader is a daemon thread. The HTTP server (if enabled) is a daemon thread. The main loop runs on the main thread. When the main thread exits — because the user hit Ctrl+C or the process received SIGTERM — all daemon threads die automatically. No cleanup required. No orphaned processes. No zombie threads.

This single-process architecture has real advantages during development. One process to start, one process to kill, one set of logs to read, one debugger to attach. If something goes wrong, you add a print statement and run it again. There's no distributed systems debugging, no container orchestration, no service mesh.

It also has real limitations. A single process means a single machine. You can't distribute the load across multiple servers. A single Python process, specifically, means you're subject to the GIL, so CPU-bound work can't truly parallelize across threads. For an AI assistant, neither of these matters — you're I/O bound on API calls, and one machine handles more conversations than you'll have. But it's worth knowing where the architecture stops scaling, in case you eventually need it to scale.

In production, OpenClaw runs as a more sophisticated service with proper signal handling, graceful shutdown, health checks, metrics, structured logging, and process supervision. The conceptual architecture is the same. Components initialize, threads start, a loop runs, messages flow. The production version just handles all the edge cases that the development version ignores — what happens when the database is corrupted, when the API key expires mid-conversation, when the disk fills up, when a tool execution hangs past its timeout.

Adding a new feature to this system follows a predictable pattern. Build the feature as an independent component with a clean interface. Test it in isolation. Wire it into main(). The isolation means your new feature can't break existing features (much), and the wiring is always a few lines. This is not accidental. The whole architecture was designed so that adding the next thing is easy.

I want to highlight one specific thing about the code above: there's no global state. The database connection is thread-local. The channels are local variables in main(). The scheduler runs independently. The group locks are created on demand. This means you could, if you wanted to, run two completely independent assistants in the same process by calling main() twice with different configurations. You probably wouldn't want to, but the fact that you could is a sign that the architecture has clean boundaries.


## Conclusion

Let's take stock of what we built.

Starting from nothing, we constructed an AI assistant that polls for messages from multiple channels, stores conversation history in SQLite, adapts its personality based on context, calls tools in a loop to accomplish real work, isolates execution in sandboxed workspaces, serializes concurrent access per conversation, runs scheduled tasks on cron expressions, communicates across process boundaries through files, supports multiple input channels through a gateway pattern, and persists long-term memories across sessions.

The whole thing is maybe a thousand lines of Python. Probably less, if you skip the comments. That's not a lot of code for something that does this much.

How does it compare to NanoClaw? Architecturally, they're siblings. Both have a polling loop, an agent loop with tools, per-context personality, sandbox isolation, and scheduled tasks. NanoClaw uses WhatsApp where we use stdin and HTTP. NanoClaw uses JSONL files where we use SQLite. NanoClaw uses real Linux containers where we use subprocess with directory restrictions. NanoClaw uses TypeScript where we use Python. The bones are the same; the flesh differs.

How does it compare to OpenClaw? OpenClaw is a production system with years of development behind it. The channel system supports fifteen-plus integrations with full feature coverage — reactions, threads, file uploads, voice, the works. The memory system uses vector embeddings for semantic search. The sandbox is a real containerized environment with network isolation and resource limits. The personality system has a layered architecture with souls, personas, and channel-specific overrides. The scheduler is more sophisticated, with the heartbeat model that lets the assistant decide what to do on each wake-up rather than running fixed prompts. There's multi-agent support, where multiple AI personas can interact with each other. There's a browser control system, camera integration, screen recording, node pairing with mobile devices. The list goes on.

Our version captures the architectural essence without the production complexity. If you understand how our system works, you understand how OpenClaw works at a conceptual level. The production version handles more edge cases, supports more integrations, and has been beaten into shape by real-world usage. But the fundamental patterns — poll, process, tool-call loop, isolate, schedule, remember — are identical.

So where do you go from here? There are a few obvious directions.

Replace the terminal channel with a real WhatsApp integration. The WhatsApp Business API gives you webhook-based message delivery, which means you can swap from polling to push — more efficient, lower latency, and the approach NanoClaw uses in production. Or go with Discord, which has excellent Python libraries and a generous free tier for bots.

Replace keyword-based memory search with vector embeddings. ChromaDB, Pinecone, Qdrant, or even just numpy with a flat index. The API stays the same — save and search — but the search quality improves dramatically once you have more than a handful of memories.

Replace subprocess isolation with real containers. Docker is the obvious choice. You'd build the agent code into a container image, mount the workspace as a volume, and use the Docker SDK for Python to manage container lifecycle. The IPC mechanism from Chapter 8 works exactly the same way whether the process is a subprocess or a container — it's just files on a shared volume.

Add multi-agent support. Instead of one personality per group, imagine multiple AI personas that can talk to each other. One does code review, another writes documentation, a third manages project planning. They share a workspace and communicate through the same message system that humans use. This is one of the more fascinating frontiers in AI assistant design, and the architecture we've built supports it almost out of the box — each agent is just a different personality with access to the same tools.

The real takeaway from this exercise, if I'm being honest, is that AI assistants aren't as complicated as they seem from the outside. The core loop — receive message, think, maybe use tools, respond — is simple. Everything else is infrastructure to make that loop run reliably, concurrently, and across different contexts. Once you see the pattern, you can build your own version of it in a weekend. It won't be production-ready. But it'll be yours, and you'll understand every line. That understanding is worth more than any framework.
