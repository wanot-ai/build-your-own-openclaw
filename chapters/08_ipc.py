#!/usr/bin/env python3
"""
Chapter 8: Host ↔ Agent IPC
=============================

NanoClaw's IPC: communication between host and container.

Since agents run in isolated containers, they can't directly
call WhatsApp APIs or access the host filesystem. Instead,
NanoClaw uses file-based IPC:

1. Agent writes a request file to /workspace/ipc/messages/
2. Host watches the IPC directory and processes the request
3. Host writes the result back

This is also how the host sends follow-up messages to running containers:
- Host writes to /workspace/ipc/input/{timestamp}.json
- Agent polls the input directory for new messages
- Agent writes _close sentinel to signal shutdown

NanoClaw (ipc.ts):
    - Watches data/ipc/{groupFolder}/messages/ for outbound message requests
    - Watches data/ipc/{groupFolder}/tasks/ for task creation requests
    - Writes to data/ipc/{groupFolder}/input/ for inbound messages

Run: uv run python chapters/08_ipc.py
"""

import json
import os
import tempfile
import threading
import time
from pathlib import Path

IPC_POLL_INTERVAL = 1.0  # NanoClaw uses 1s


class IpcChannel:
    """
    Host <-> Agent IPC channel.

    Implements NanoClaw's IPC patterns:
    - messages/: Agent requests to send messages
    - tasks/: Agent requests to create tasks
    - input/: Host delivers follow-up messages
    - _close: Host delivers shutdown signal

    NanoClaw (ipc.ts / group-queue.ts):
        // Agent → Host (message send request)
        agent writes: ipc/messages/{timestamp}.json
        host reads and sends via WhatsApp

        // Host → Agent (follow-up message)
        host writes: ipc/input/{timestamp}.json
        agent reads and processes

        // Host → Agent (shutdown signal)
        host writes: ipc/input/_close
        agent exits gracefully
    """

    def __init__(self, ipc_dir: str):
        self.ipc_dir = ipc_dir
        self.messages_dir = os.path.join(ipc_dir, "messages")
        self.tasks_dir = os.path.join(ipc_dir, "tasks")
        self.input_dir = os.path.join(ipc_dir, "input")

        for d in [self.messages_dir, self.tasks_dir, self.input_dir]:
            os.makedirs(d, exist_ok=True)

    # --- Agent side (runs inside container) ---

    def agent_send_message(self, chat_jid: str, text: str):
        """
        Agent requests to send a message.
        Writing a file to the IPC directory triggers the host to read and process it.

        In NanoClaw this is implemented as an MCP tool,
        but the underlying principle is the same: file-based IPC.
        """
        filename = f"{int(time.time() * 1000)}.json"
        filepath = os.path.join(self.messages_dir, filename)
        temp_path = f"{filepath}.tmp"

        # Atomic write: tmp -> rename (same pattern as NanoClaw)
        with open(temp_path, "w") as f:
            json.dump({"chat_jid": chat_jid, "text": text}, f)
        os.rename(temp_path, filepath)

        print(f"  📤 [Agent] Sent IPC message: {text[:50]}...")

    def agent_create_task(self, task_data: dict):
        """Agent requests to create a task."""
        filename = f"{int(time.time() * 1000)}.json"
        filepath = os.path.join(self.tasks_dir, filename)
        temp_path = f"{filepath}.tmp"

        with open(temp_path, "w") as f:
            json.dump(task_data, f)
        os.rename(temp_path, filepath)

        print(f"  📤 [Agent] Created task request: {task_data.get('prompt', '')[:50]}...")

    def agent_poll_input(self) -> list[dict]:
        """
        Agent polls for input from the host.

        NanoClaw (agent-runner in container):
            - Polls /workspace/ipc/input/ for new .json files
            - Checks for _close sentinel to exit gracefully
        """
        messages = []

        # Check for _close signal
        close_path = os.path.join(self.input_dir, "_close")
        if os.path.exists(close_path):
            os.remove(close_path)
            return [{"type": "close"}]

        # Check for new message files
        for filename in sorted(os.listdir(self.input_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.input_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                messages.append(data)
                os.remove(filepath)
            except (json.JSONDecodeError, OSError):
                continue

        return messages

    # --- Host side ---

    def host_send_input(self, text: str):
        """
        Host sends a follow-up message to the agent.

        NanoClaw (group-queue.ts):
            sendMessage(groupJid, text) {
                const filename = `${Date.now()}-${random}.json`;
                fs.writeFileSync(tempPath, JSON.stringify({ type: 'message', text }));
                fs.renameSync(tempPath, filepath);
            }
        """
        filename = f"{int(time.time() * 1000)}-{os.urandom(2).hex()}.json"
        filepath = os.path.join(self.input_dir, filename)
        temp_path = f"{filepath}.tmp"

        with open(temp_path, "w") as f:
            json.dump({"type": "message", "text": text}, f)
        os.rename(temp_path, filepath)

        print(f"  📥 [Host] Sent input to agent: {text[:50]}...")

    def host_send_close(self):
        """
        Host sends a shutdown signal to the agent.

        NanoClaw (group-queue.ts):
            closeStdin(groupJid) {
                fs.writeFileSync(path.join(inputDir, '_close'), '');
            }
        """
        close_path = os.path.join(self.input_dir, "_close")
        with open(close_path, "w") as f:
            f.write("")
        print("  🛑 [Host] Sent close signal")

    def host_poll_messages(self) -> list[dict]:
        """Host polls for message requests from the agent."""
        messages = []
        for filename in sorted(os.listdir(self.messages_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.messages_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                messages.append(data)
                os.remove(filepath)
            except (json.JSONDecodeError, OSError):
                continue
        return messages

    def host_poll_tasks(self) -> list[dict]:
        """Host polls for task requests from the agent."""
        tasks = []
        for filename in sorted(os.listdir(self.tasks_dir)):
            if not filename.endswith(".json"):
                continue
            filepath = os.path.join(self.tasks_dir, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                tasks.append(data)
                os.remove(filepath)
            except (json.JSONDecodeError, OSError):
                continue
        return tasks


def main():
    print("=" * 50)
    print("Chapter 8: Host ↔ Agent IPC")
    print("=" * 50)
    print()
    print("NanoClaw uses file-based IPC between host and containers.")
    print("Agents write request files, host watches and processes them.\n")

    # Create temporary IPC directory
    with tempfile.TemporaryDirectory() as tmpdir:
        ipc = IpcChannel(tmpdir)

        # --- Simulation: agent and host communicate ---

        print("=== Scenario 1: Agent sends a message ===\n")

        # Agent requests to send messages
        ipc.agent_send_message("family@group", "Good morning! ☀️")
        ipc.agent_send_message("family@group", "Today's weather is sunny")

        # Host polls for message requests
        time.sleep(0.1)
        msgs = ipc.host_poll_messages()
        for msg in msgs:
            print(f"  📬 [Host] Would send to WhatsApp: {msg['text']}")

        print(f"\n=== Scenario 2: Host sends follow-up to agent ===\n")

        # Host delivers new messages to the agent
        ipc.host_send_input("Alice: Thanks Nano!")
        ipc.host_send_input("Bob: What about tomorrow?")

        # Agent polls for input
        time.sleep(0.1)
        inputs = ipc.agent_poll_input()
        for inp in inputs:
            print(f"  📨 [Agent] Received: {inp}")

        print(f"\n=== Scenario 3: Agent creates a task ===\n")

        # Agent requests to create a scheduled task
        ipc.agent_create_task(
            {
                "prompt": "Send weather update every morning at 7am",
                "schedule_type": "cron",
                "schedule_value": "0 7 * * *",
                "chat_jid": "family@group",
            }
        )

        # Host polls for task requests
        time.sleep(0.1)
        tasks = ipc.host_poll_tasks()
        for task in tasks:
            print(f"  📋 [Host] Would create task: {task['prompt']}")

        print(f"\n=== Scenario 4: Host signals agent to close ===\n")

        # Host sends shutdown signal
        ipc.host_send_close()

        # Agent receives shutdown signal
        time.sleep(0.1)
        inputs = ipc.agent_poll_input()
        for inp in inputs:
            if inp.get("type") == "close":
                print("  🛑 [Agent] Received close signal, shutting down...")

    print("\n" + "=" * 50)
    print("\nNanoClaw IPC architecture:")
    print("  data/ipc/{group}/messages/ → Agent → Host (send WhatsApp message)")
    print("  data/ipc/{group}/tasks/    → Agent → Host (create scheduled task)")
    print("  data/ipc/{group}/input/    → Host → Agent (follow-up messages)")
    print("  data/ipc/{group}/input/_close → Host → Agent (shutdown signal)")
    print()
    print("Key patterns:")
    print("  - Atomic writes: write to .tmp, then rename")
    print("  - Polling-based: both sides poll their directories")
    print("  - Per-group namespace: prevents cross-group access")


if __name__ == "__main__":
    main()
