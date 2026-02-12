#!/usr/bin/env python3
"""
Chapter 1: The Polling Bot
==========================

The core architecture of NanoClaw: the polling loop.

NanoClaw doesn't use webhooks or event-driven callbacks.
It polls WhatsApp every 2 seconds for new messages, processes them,
and sends replies. Simple. Reliable. Easy to debug.

This chapter builds a terminal-based polling bot that:
- Accepts messages via stdin (simulating WhatsApp)
- Polls for new messages every 2 seconds
- Sends them to Claude for a response
- Prints the reply (simulating WhatsApp send)

Run: uv run --with anthropic python chapters/01_polling_bot.py
"""

import os
import sys
import time
import threading
import anthropic

# --- Message Queue (Simulating WhatsApp) ---
# In NanoClaw, messages come from WhatsApp via baileys.
# Here we simulate with a thread-safe queue fed by stdin.

message_queue: list[dict] = []
queue_lock = threading.Lock()

POLL_INTERVAL = 2.0  # NanoClaw uses 2s polling interval


def stdin_reader():
    """Background thread: reads messages from stdin and adds them to the queue."""
    while True:
        try:
            line = input()
            if line.strip():
                msg = {
                    "id": f"msg_{int(time.time() * 1000)}",
                    "sender": "user",
                    "sender_name": "You",
                    "content": line.strip(),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "chat_jid": "terminal@local",
                }
                with queue_lock:
                    message_queue.append(msg)
        except EOFError:
            break


def poll_messages() -> list[dict]:
    """Retrieves new messages from the queue. Equivalent to NanoClaw's getNewMessages()."""
    with queue_lock:
        messages = message_queue.copy()
        message_queue.clear()
    return messages


def format_messages(messages: list[dict]) -> str:
    """
    Formats messages in XML format. Same pattern as NanoClaw's router.ts.

    NanoClaw (TypeScript):
        const lines = messages.map((m) =>
            `<message sender="${escapeXml(m.sender_name)}" time="${m.timestamp}">${escapeXml(m.content)}</message>`
        );
        return `<messages>\\n${lines.join('\\n')}\\n</messages>`;
    """
    lines = []
    for m in messages:
        sender = m["sender_name"].replace("&", "&amp;").replace("<", "&lt;")
        content = m["content"].replace("&", "&amp;").replace("<", "&lt;")
        lines.append(
            f'<message sender="{sender}" time="{m["timestamp"]}">{content}</message>'
        )
    return f"<messages>\n" + "\n".join(lines) + "\n</messages>"


def send_to_agent(prompt: str) -> str:
    """Claude API call. The simplest form."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def send_reply(text: str):
    """Sends a reply. In NanoClaw this goes to WhatsApp; here it goes to the terminal."""
    print(f"\n🤖 Nano: {text}\n")


def polling_loop():
    """
    Main polling loop. Same structure as NanoClaw's startMessageLoop().

    NanoClaw (TypeScript):
        while (true) {
            const { messages, newTimestamp } = getNewMessages(jids, lastTimestamp, ASSISTANT_NAME);
            if (messages.length > 0) {
                // process messages...
            }
            await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));
        }
    """
    print("Polling loop started (type messages below)...\n")

    while True:
        messages = poll_messages()

        if messages:
            prompt = format_messages(messages)
            print("  ⏳ Thinking...")

            try:
                reply = send_to_agent(prompt)
                send_reply(reply)
            except Exception as e:
                print(f"  ❌ Error: {e}\n")

        time.sleep(POLL_INTERVAL)


def main():
    print("=" * 50)
    print("Chapter 1: The Polling Bot")
    print("=" * 50)
    print()
    print("NanoClaw-style polling loop.")
    print("Type a message and press Enter.")
    print("The bot polls every 2s for new messages.")
    print("Ctrl+C to quit.\n")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Set ANTHROPIC_API_KEY to use Claude.")
        print("   Running in echo mode instead.\n")

        global send_to_agent

        def send_to_agent(prompt: str) -> str:
            return f"[Echo] Received: {prompt[:100]}..."

    # Run stdin reader in the background
    reader = threading.Thread(target=stdin_reader, daemon=True)
    reader.start()

    try:
        polling_loop()
    except KeyboardInterrupt:
        print("\n👋 Bye!")


if __name__ == "__main__":
    main()
