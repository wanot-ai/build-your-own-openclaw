#!/usr/bin/env python3
"""
Chapter 6: Per-Group Concurrency Control
=========================================

NanoClaw's GroupQueue: per-group concurrency control.

NanoClaw ensures only one agent runs per group at a time,
while allowing different groups to run in parallel.
It also limits total concurrent containers (MAX_CONCURRENT_CONTAINERS=5).

This prevents:
- Race conditions on the same group's session/files
- Resource exhaustion from too many containers

NanoClaw's GroupQueue (group-queue.ts):
- Per-group state: active flag, pending messages, pending tasks
- Global concurrency limit
- Automatic drain: when one group finishes, waiting groups get a slot
- Retry with exponential backoff on failures

Run: uv run python chapters/06_group_queue.py
"""

import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GroupState:
    """
    Per-group state. Equivalent to NanoClaw's GroupState interface.

    NanoClaw (TypeScript):
        interface GroupState {
            active: boolean;
            pendingMessages: boolean;
            pendingTasks: QueuedTask[];
            process: ChildProcess | null;
            containerName: string | null;
            retryCount: number;
        }
    """

    active: bool = False
    pending_messages: bool = False
    pending_tasks: list = field(default_factory=list)
    retry_count: int = 0


class GroupQueue:
    """
    Per-group concurrency control queue.

    Python implementation of NanoClaw's GroupQueue.
    - Only one agent runs per group
    - Global concurrent execution limit
    - Auto-drain: when a slot opens, waiting groups are executed

    NanoClaw (group-queue.ts):
        export class GroupQueue {
            private groups = new Map<string, GroupState>();
            private activeCount = 0;
            private waitingGroups: string[] = [];
        }
    """

    MAX_CONCURRENT = 3  # NanoClaw default: 5
    MAX_RETRIES = 5
    BASE_RETRY_MS = 5.0  # seconds

    def __init__(self):
        self._groups: dict[str, GroupState] = defaultdict(GroupState)
        self._active_count = 0
        self._waiting: list[str] = []
        self._lock = threading.Lock()
        self._process_fn: Callable[[str], bool] | None = None
        self._shutting_down = False

    def set_process_fn(self, fn: Callable[[str], bool]):
        """Register message processing function. NanoClaw's setProcessMessagesFn()."""
        self._process_fn = fn

    def enqueue_message(self, group_jid: str):
        """
        Enqueues a group's message processing.

        NanoClaw (group-queue.ts):
            enqueueMessageCheck(groupJid: string): void {
                if (state.active) { state.pendingMessages = true; return; }
                if (this.activeCount >= MAX_CONCURRENT_CONTAINERS) {
                    state.pendingMessages = true;
                    this.waitingGroups.push(groupJid);
                    return;
                }
                this.runForGroup(groupJid, 'messages');
            }
        """
        if self._shutting_down:
            return

        with self._lock:
            state = self._groups[group_jid]

            if state.active:
                state.pending_messages = True
                print(f"  📋 [{group_jid}] Queued (group active)")
                return

            if self._active_count >= self.MAX_CONCURRENT:
                state.pending_messages = True
                if group_jid not in self._waiting:
                    self._waiting.append(group_jid)
                print(
                    f"  📋 [{group_jid}] Queued (at limit: {self._active_count}/{self.MAX_CONCURRENT})"
                )
                return

        # Slot available — run immediately
        self._run_for_group(group_jid)

    def _run_for_group(self, group_jid: str):
        """Processes the group's messages. Runs in a separate thread."""

        def worker():
            with self._lock:
                state = self._groups[group_jid]
                state.active = True
                state.pending_messages = False
                self._active_count += 1

            print(
                f"  ▶️  [{group_jid}] Processing (active: {self._active_count}/{self.MAX_CONCURRENT})"
            )

            try:
                if self._process_fn:
                    success = self._process_fn(group_jid)
                    if success:
                        self._groups[group_jid].retry_count = 0
                    else:
                        self._schedule_retry(group_jid)
            except Exception as e:
                print(f"  ❌ [{group_jid}] Error: {e}")
                self._schedule_retry(group_jid)
            finally:
                with self._lock:
                    state = self._groups[group_jid]
                    state.active = False
                    self._active_count -= 1
                    print(
                        f"  ✅ [{group_jid}] Done (active: {self._active_count}/{self.MAX_CONCURRENT})"
                    )

                # Drain: run the next waiting group
                self._drain(group_jid)

        threading.Thread(target=worker, daemon=True).start()

    def _drain(self, finished_jid: str):
        """
        Drains waiting work after completion.

        NanoClaw (group-queue.ts):
            private drainGroup(groupJid: string): void {
                if (state.pendingMessages) { this.runForGroup(groupJid, 'drain'); return; }
                this.drainWaiting();
            }
        """
        with self._lock:
            # If the same group has pending messages, run again
            state = self._groups[finished_jid]
            if state.pending_messages:
                # Run after releasing lock
                pass_to_self = True
            else:
                pass_to_self = False

        if pass_to_self:
            self._run_for_group(finished_jid)
            return

        # If other groups are waiting, run them
        with self._lock:
            while (
                self._waiting
                and self._active_count < self.MAX_CONCURRENT
            ):
                next_jid = self._waiting.pop(0)
                next_state = self._groups[next_jid]
                if next_state.pending_messages:
                    # Run outside the lock
                    threading.Thread(
                        target=self._run_for_group,
                        args=(next_jid,),
                        daemon=True,
                    ).start()

    def _schedule_retry(self, group_jid: str):
        """
        Retry with exponential backoff on failure.

        NanoClaw (group-queue.ts):
            const delayMs = BASE_RETRY_MS * Math.pow(2, state.retryCount - 1);
            setTimeout(() => this.enqueueMessageCheck(groupJid), delayMs);
        """
        state = self._groups[group_jid]
        state.retry_count += 1

        if state.retry_count > self.MAX_RETRIES:
            print(f"  ⛔ [{group_jid}] Max retries exceeded, dropping")
            state.retry_count = 0
            return

        delay = self.BASE_RETRY_MS * (2 ** (state.retry_count - 1))
        print(
            f"  🔄 [{group_jid}] Retry {state.retry_count} in {delay:.1f}s"
        )
        threading.Timer(delay, lambda: self.enqueue_message(group_jid)).start()

    def shutdown(self):
        """Graceful shutdown."""
        self._shutting_down = True
        print("  GroupQueue shutting down")


def main():
    print("=" * 50)
    print("Chapter 6: Per-Group Concurrency Control")
    print("=" * 50)
    print()
    print("NanoClaw's GroupQueue ensures:")
    print("  - One agent per group at a time")
    print("  - Max 3 concurrent groups (configurable)")
    print("  - Automatic drain of waiting groups")
    print()

    queue = GroupQueue()

    # Processing function: simulates a 2-second task
    def process_messages(group_jid: str) -> bool:
        print(f"  🔨 [{group_jid}] Working...")
        time.sleep(2)
        return True

    queue.set_process_fn(process_messages)

    print("Enqueueing 6 groups (max concurrent: 3)...\n")

    # Enqueue messages from 6 groups at once
    for i in range(6):
        queue.enqueue_message(f"group-{i}")
        time.sleep(0.1)  # slight delay

    # Wait for all work to complete
    time.sleep(15)

    print("\n--- Same group, multiple messages ---\n")

    # Same group, multiple messages: second gets queued while first is processing
    queue.enqueue_message("group-A")
    time.sleep(0.5)
    queue.enqueue_message("group-A")  # Goes into waiting state

    time.sleep(5)

    print("\n✅ All done!")
    print("\nKey concepts from NanoClaw's GroupQueue:")
    print("  1. Per-group lock: one container per group")
    print("  2. Global limit: max concurrent containers")
    print("  3. Auto-drain: idle slots filled from waiting queue")
    print("  4. Retry with backoff: failed groups get retried")


if __name__ == "__main__":
    main()
