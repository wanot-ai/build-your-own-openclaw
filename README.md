# Build Your Own AI Assistant

**Based on [NanoClaw](https://github.com/gavrielc/nanoclaw)'s architecture.**

A step-by-step tutorial that builds a personal AI assistant from scratch, following NanoClaw's actual production architecture: polling loops, SQLite, container isolation, per-group concurrency, and cron-based scheduling.

Read the full tutorial: **[BLOG.md](./BLOG.md)**

## Quick Start

```bash
# Run the complete system
uv run --with anthropic --with schedule python nanoclaw.py

# Or without API key (echo mode — still demonstrates the architecture)
python nanoclaw.py
```

## What You'll Build

```
nanoclaw.py  (~500 lines)
├── Polling loop          — polls for messages every 2s
├── SQLite database       — messages, sessions, router state, tasks
├── Per-group CLAUDE.md   — different personality per group
├── Agent loop            — tool-using AI with structured execution
├── Group queue           — per-group concurrency control
├── Task scheduler        — cron-based scheduled agent runs
├── Long-term memory      — file-based persistent knowledge
└── Multi-channel gateway — terminal + HTTP, extensible
```

## Chapter-by-Chapter

Each chapter is independently runnable:

| Chapter | File | Concept |
|---------|------|---------|
| 1 | `chapters/01_polling_bot.py` | WhatsApp-style polling loop |
| 2 | `chapters/02_database.py` | SQLite for messages + sessions |
| 3 | `chapters/03_personality.py` | Per-group CLAUDE.md personality |
| 4 | `chapters/04_agent_loop.py` | Tool-using agent with structured execution |
| 5 | `chapters/05_container_isolation.py` | Subprocess/container isolation |
| 6 | `chapters/06_group_queue.py` | Per-group concurrency control |
| 7 | `chapters/07_task_scheduler.py` | Cron-based scheduled tasks |
| 8 | `chapters/08_ipc.py` | Host ↔ agent IPC |
| 9 | `chapters/09_multi_channel.py` | Gateway pattern (terminal + HTTP) |
| 10 | `chapters/10_memory.py` | Long-term file-based memory |
| 11 | `chapters/11_full_system.py` | Everything combined |

Run any chapter:
```bash
# Most chapters work without an API key
python chapters/02_database.py
python chapters/06_group_queue.py

# Chapters with Claude API calls
ANTHROPIC_API_KEY=sk-... uv run --with anthropic python chapters/04_agent_loop.py
```

## Commands (in nanoclaw.py)

| Command | Description |
|---------|-------------|
| `/help` | Show commands |
| `/new` | Reset session |
| `/memory` | List saved memories |
| `/tasks` | List scheduled tasks |
| `/schedule <sec> <prompt>` | Create recurring task |
| `/quit` | Exit |

## HTTP API

When running `nanoclaw.py`, you can also send messages via HTTP:

```bash
curl -X POST http://127.0.0.1:5555/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"alice", "message":"Hello Nano!"}'
```

## File Structure

```
build-your-own-openclaw/
├── README.md          ← you are here
├── BLOG.md            ← full technical blog (3000-5000 words)
├── pyproject.toml     ← dependencies
├── nanoclaw.py        ← complete system (~500 lines)
├── SOUL.md            ← example personality
├── chapters/          ← step-by-step tutorial files
│   ├── 01_polling_bot.py
│   ├── 02_database.py
│   ├── ...
│   └── 11_full_system.py
└── workspace/         ← agent workspace
    ├── memory/        ← long-term memory files
    └── groups/        ← per-group folders
```

## Dependencies

- **Required:** Python 3.10+
- **For Claude:** `anthropic` SDK + `ANTHROPIC_API_KEY`
- **For scheduling:** `schedule` library
- **Built-in:** `sqlite3`, `subprocess`, `threading`, `http.server`

## How This Relates to NanoClaw

| This Tutorial | NanoClaw | OpenClaw |
|---------------|----------|----------|
| `nanoclaw.py` polling loop | `src/index.ts` startMessageLoop | Event-driven gateway |
| `sqlite3` database | `better-sqlite3` (db.ts) | Multiple stores + JSONL |
| `subprocess` isolation | Apple Container / Docker | Application-level permissions |
| `GroupQueue` class | `src/group-queue.ts` | Lane-based command queue |
| `schedule` library | `cron-parser` (task-scheduler.ts) | Cron + heartbeats |
| File-based IPC | `src/ipc.ts` filesystem IPC | In-process tool calls |
| Per-group CLAUDE.md | `groups/*/CLAUDE.md` | SOUL.md + AGENTS.md |
| Terminal + HTTP | WhatsApp (baileys) | 15+ channel providers |

## License

MIT
