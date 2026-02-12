#!/usr/bin/env python3
"""
Chapter 7: Cron-Based Task Scheduler
=====================================

NanoClaw's task scheduler: cron-based scheduled tasks.

NanoClaw supports scheduled tasks stored in SQLite:
- 'cron' type: standard cron expressions (e.g., "30 7 * * *")
- 'interval' type: run every N milliseconds
- 'once' type: run once at a specific time

The scheduler polls the database every 60 seconds for due tasks,
then enqueues them through the GroupQueue (so they respect concurrency limits).

NanoClaw (task-scheduler.ts):
    const loop = async () => {
        const dueTasks = getDueTasks();
        for (const task of dueTasks) {
            deps.queue.enqueueTask(
                currentTask.chat_jid, currentTask.id,
                () => runTask(currentTask, deps)
            );
        }
        setTimeout(loop, SCHEDULER_POLL_INTERVAL);
    };

Run: uv run --with schedule python chapters/07_task_scheduler.py
"""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta

# Note: We use raw SQLite polling here, not the `schedule` library.
# The `schedule` library is used in nanoclaw.py for convenience.

DB_PATH = ":memory:"  # In-memory DB for demo


def get_db() -> sqlite3.Connection:
    """DB connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_task_db(db: sqlite3.Connection):
    """
    Initialize task tables.

    NanoClaw (db.ts):
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY,
            group_folder TEXT NOT NULL,
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
    """
    db.executescript("""
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY,
            group_folder TEXT NOT NULL,
            chat_jid TEXT NOT NULL,
            prompt TEXT NOT NULL,
            schedule_type TEXT NOT NULL CHECK(schedule_type IN ('cron', 'interval', 'once')),
            schedule_value TEXT NOT NULL,
            next_run TEXT,
            last_run TEXT,
            last_result TEXT,
            status TEXT DEFAULT 'active' CHECK(status IN ('active', 'paused', 'completed')),
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_next_run ON scheduled_tasks(next_run);
        CREATE INDEX IF NOT EXISTS idx_status ON scheduled_tasks(status);

        CREATE TABLE IF NOT EXISTS task_run_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            run_at TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            status TEXT NOT NULL,
            result TEXT,
            error TEXT
        );
    """)
    db.commit()


def create_task(
    db: sqlite3.Connection,
    task_id: str,
    group_folder: str,
    chat_jid: str,
    prompt: str,
    schedule_type: str,
    schedule_value: str,
    next_run: str | None = None,
):
    """Creates a task."""
    now = datetime.now().isoformat()
    db.execute(
        """INSERT INTO scheduled_tasks
           (id, group_folder, chat_jid, prompt, schedule_type, schedule_value, next_run, status, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)""",
        (task_id, group_folder, chat_jid, prompt, schedule_type, schedule_value, next_run, now),
    )
    db.commit()


def get_due_tasks(db: sqlite3.Connection) -> list[dict]:
    """
    Retrieves tasks that are due to run.

    NanoClaw (db.ts):
        const now = new Date().toISOString();
        return db.prepare(`
            SELECT * FROM scheduled_tasks
            WHERE status = 'active' AND next_run IS NOT NULL AND next_run <= ?
        `).all(now);
    """
    now = datetime.now().isoformat()
    rows = db.execute(
        """SELECT * FROM scheduled_tasks
           WHERE status = 'active' AND next_run IS NOT NULL AND next_run <= ?
           ORDER BY next_run""",
        (now,),
    ).fetchall()
    return [dict(r) for r in rows]


def update_task_after_run(
    db: sqlite3.Connection,
    task_id: str,
    next_run: str | None,
    result: str,
):
    """
    Updates a task after execution.

    NanoClaw (db.ts):
        db.prepare(`UPDATE scheduled_tasks
            SET next_run = ?, last_run = ?, last_result = ?,
                status = CASE WHEN ? IS NULL THEN 'completed' ELSE status END
            WHERE id = ?`).run(nextRun, now, lastResult, nextRun, id);
    """
    now = datetime.now().isoformat()
    db.execute(
        """UPDATE scheduled_tasks
           SET next_run = ?, last_run = ?, last_result = ?,
               status = CASE WHEN ? IS NULL THEN 'completed' ELSE status END
           WHERE id = ?""",
        (next_run, now, result, next_run, task_id),
    )
    db.commit()


def log_task_run(
    db: sqlite3.Connection,
    task_id: str,
    duration_ms: int,
    status: str,
    result: str | None = None,
    error: str | None = None,
):
    """Records a task run log entry."""
    now = datetime.now().isoformat()
    db.execute(
        """INSERT INTO task_run_logs (task_id, run_at, duration_ms, status, result, error)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (task_id, now, duration_ms, status, result, error),
    )
    db.commit()


def calculate_next_run(schedule_type: str, schedule_value: str) -> str | None:
    """Calculates the next run time."""
    if schedule_type == "interval":
        ms = int(schedule_value)
        return (datetime.now() + timedelta(milliseconds=ms)).isoformat()
    elif schedule_type == "once":
        return None  # One-time tasks have no next run
    elif schedule_type == "cron":
        # Simplified: using schedule library
        # Real NanoClaw uses cron-parser
        return (datetime.now() + timedelta(minutes=1)).isoformat()
    return None


def run_task(db: sqlite3.Connection, task: dict):
    """
    Runs a task.

    NanoClaw (task-scheduler.ts):
        async function runTask(task, deps) {
            const output = await runContainerAgent(group, {
                prompt: task.prompt, sessionId, groupFolder: task.group_folder,
                chatJid: task.chat_jid, isMain, isScheduledTask: true,
            }, ...);
        }
    """
    start = time.time()
    print(f"  ⏰ Running task '{task['id']}': {task['prompt'][:60]}...")

    try:
        # In practice, the agent would be run here
        # In NanoClaw, runContainerAgent() is called
        result = f"Task completed: {task['prompt'][:50]}"
        duration_ms = int((time.time() - start) * 1000)

        log_task_run(db, task["id"], duration_ms, "success", result=result)

        next_run = calculate_next_run(task["schedule_type"], task["schedule_value"])
        update_task_after_run(db, task["id"], next_run, result)

        print(f"  ✅ Task '{task['id']}' done ({duration_ms}ms)")
        if next_run:
            print(f"     Next run: {next_run}")
        else:
            print(f"     Task completed (no more runs)")

    except Exception as e:
        duration_ms = int((time.time() - start) * 1000)
        log_task_run(db, task["id"], duration_ms, "error", error=str(e))
        print(f"  ❌ Task '{task['id']}' failed: {e}")


def scheduler_loop(db: sqlite3.Connection, iterations: int = 0):
    """
    Scheduler polling loop. Equivalent to NanoClaw's startSchedulerLoop().

    NanoClaw (task-scheduler.ts):
        const loop = async () => {
            const dueTasks = getDueTasks();
            for (const task of dueTasks) {
                deps.queue.enqueueTask(...);
            }
            setTimeout(loop, SCHEDULER_POLL_INTERVAL);
        };
    """
    count = 0
    while iterations == 0 or count < iterations:
        due = get_due_tasks(db)
        if due:
            print(f"\n📋 Found {len(due)} due task(s)")
            for task in due:
                run_task(db, task)
        count += 1
        if iterations == 0 or count < iterations:
            time.sleep(2)  # 2-second interval for demo (real NanoClaw: 60s)


def main():
    print("=" * 50)
    print("Chapter 7: Cron-Based Task Scheduler")
    print("=" * 50)
    print()

    db = get_db()
    init_task_db(db)
    print("✅ Task database initialized\n")

    # Create test tasks
    now = datetime.now()

    # 1. Task to run immediately (next_run is in the past)
    create_task(
        db,
        task_id="morning-briefing",
        group_folder="personal",
        chat_jid="me@local",
        prompt="Give me a morning briefing: weather, calendar, and news highlights",
        schedule_type="interval",
        schedule_value="60000",  # 60초마다
        next_run=(now - timedelta(seconds=5)).isoformat(),
    )
    print("✅ Created 'morning-briefing' (interval: 60s)")

    # 2. One-time task
    create_task(
        db,
        task_id="reminder-meeting",
        group_folder="work",
        chat_jid="work-group@local",
        prompt="Remind everyone: team meeting at 3pm today",
        schedule_type="once",
        schedule_value="",
        next_run=(now - timedelta(seconds=1)).isoformat(),
    )
    print("✅ Created 'reminder-meeting' (once)")

    # 3. Task not yet due
    create_task(
        db,
        task_id="weekly-report",
        group_folder="work",
        chat_jid="work-group@local",
        prompt="Generate weekly project report from git commits",
        schedule_type="cron",
        schedule_value="0 9 * * MON",
        next_run=(now + timedelta(hours=24)).isoformat(),
    )
    print("✅ Created 'weekly-report' (cron, not due yet)")

    print(f"\n--- Running scheduler (3 iterations) ---\n")
    scheduler_loop(db, iterations=3)

    # Check results
    print(f"\n--- Task states ---\n")
    for row in db.execute("SELECT id, status, last_run, next_run FROM scheduled_tasks").fetchall():
        r = dict(row)
        print(f"  {r['id']}: status={r['status']}, last_run={r['last_run']}, next_run={r['next_run']}")

    print(f"\n--- Run logs ---\n")
    for row in db.execute("SELECT task_id, run_at, status, duration_ms FROM task_run_logs ORDER BY run_at").fetchall():
        r = dict(row)
        print(f"  {r['task_id']}: {r['status']} at {r['run_at']} ({r['duration_ms']}ms)")

    print("\n✅ Scheduler demo complete!")


if __name__ == "__main__":
    main()
