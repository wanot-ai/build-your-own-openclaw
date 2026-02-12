#!/usr/bin/env python3
"""
Chapter 9: Multi-Channel Gateway
==================================

NanoClaw's gateway pattern: connecting multiple channels to a single agent.

NanoClaw uses WhatsApp as its primary channel, but the architecture
supports multiple channels. Each channel:
1. Receives messages from a platform
2. Stores them in SQLite
3. The polling loop picks them up regardless of source

The key abstraction is the Channel interface:

NanoClaw (types.ts):
    export interface Channel {
        name: string;
        connect(): Promise<void>;
        sendMessage(jid: string, text: string): Promise<void>;
        isConnected(): boolean;
        ownsJid(jid: string): boolean;
        disconnect(): Promise<void>;
    }

This chapter implements terminal + HTTP channels, both feeding
into the same SQLite database and processed by the same polling loop.

Run: uv run --with anthropic python chapters/09_multi_channel.py
"""

import json
import os
import sqlite3
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

# --- Channel Interface ---


class Channel:
    """
    Channel abstract interface. Equivalent to NanoClaw's Channel type.

    NanoClaw (types.ts):
        export interface Channel {
            name: string;
            connect(): Promise<void>;
            sendMessage(jid: string, text: string): Promise<void>;
            isConnected(): boolean;
            ownsJid(jid: string): boolean;
            disconnect(): Promise<void>;
        }
    """

    name: str = "base"

    def connect(self):
        pass

    def send_message(self, jid: str, text: str):
        raise NotImplementedError

    def is_connected(self) -> bool:
        return True

    def owns_jid(self, jid: str) -> bool:
        raise NotImplementedError

    def disconnect(self):
        pass


# --- Terminal Channel ---


class TerminalChannel(Channel):
    """Terminal I/O channel."""

    name = "terminal"

    def __init__(self, on_message=None):
        self._on_message = on_message
        self._running = False

    def connect(self):
        self._running = True
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        while self._running:
            try:
                line = input()
                if line.strip() and self._on_message:
                    msg = {
                        "id": f"term_{int(time.time() * 1000)}",
                        "chat_jid": "terminal@local",
                        "sender": "terminal_user",
                        "sender_name": "You",
                        "content": line.strip(),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "is_from_me": False,
                    }
                    self._on_message(msg)
            except EOFError:
                break

    def send_message(self, jid: str, text: str):
        print(f"\n🤖 Nano: {text}\n")

    def owns_jid(self, jid: str) -> bool:
        return jid.endswith("@local")

    def disconnect(self):
        self._running = False


# --- HTTP Channel ---


class HttpChannel(Channel):
    """HTTP API channel. Messages can be sent via curl."""

    name = "http"

    def __init__(self, port: int = 5555, on_message=None):
        self._port = port
        self._on_message = on_message
        self._server: HTTPServer | None = None
        self._responses: dict[str, str] = {}
        self._response_events: dict[str, threading.Event] = {}

    def connect(self):
        channel = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/chat":
                    length = int(self.headers.get("Content-Length", 0))
                    body = json.loads(self.rfile.read(length))

                    msg_id = f"http_{int(time.time() * 1000)}"
                    msg = {
                        "id": msg_id,
                        "chat_jid": f"http_{body.get('user_id', 'anon')}@http",
                        "sender": body.get("user_id", "anon"),
                        "sender_name": body.get("user_name", "HTTP User"),
                        "content": body.get("message", ""),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "is_from_me": False,
                    }

                    # Create response wait event
                    event = threading.Event()
                    channel._response_events[msg["chat_jid"]] = event

                    if channel._on_message:
                        channel._on_message(msg)

                    # Wait for response (max 30 seconds)
                    event.wait(timeout=30)

                    response = channel._responses.pop(msg["chat_jid"], "No response")
                    channel._response_events.pop(msg["chat_jid"], None)

                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps({"response": response}).encode()
                    )
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress logs

        self._server = HTTPServer(("127.0.0.1", self._port), Handler)
        threading.Thread(
            target=self._server.serve_forever, daemon=True
        ).start()
        print(f"  HTTP channel listening on http://127.0.0.1:{self._port}/chat")

    def send_message(self, jid: str, text: str):
        self._responses[jid] = text
        event = self._response_events.get(jid)
        if event:
            event.set()

    def owns_jid(self, jid: str) -> bool:
        return jid.endswith("@http")

    def disconnect(self):
        if self._server:
            self._server.shutdown()


# --- Gateway: unifying all channels ---


class Gateway:
    """
    Gateway: connects multiple channels to a single agent.

    The role of NanoClaw's index.ts:
    - Receives messages from the WhatsApp channel
    - Stores them in SQLite
    - Processes them in the polling loop
    - Routes responses to the correct channel

    NanoClaw (router.ts):
        export function routeOutbound(channels, jid, text) {
            const channel = channels.find(c => c.ownsJid(jid) && c.isConnected());
            return channel.sendMessage(jid, text);
        }
    """

    def __init__(self):
        self.channels: list[Channel] = []
        self._messages: list[dict] = []
        self._lock = threading.Lock()

    def add_channel(self, channel: Channel):
        self.channels.append(channel)

    def on_message(self, msg: dict):
        """Message receive callback. Called from all channels."""
        with self._lock:
            self._messages.append(msg)

    def poll_messages(self) -> list[dict]:
        """Polls for new messages."""
        with self._lock:
            msgs = self._messages.copy()
            self._messages.clear()
        return msgs

    def send_reply(self, jid: str, text: str):
        """
        Routes the response to the appropriate channel.

        NanoClaw (router.ts):
            export function routeOutbound(channels, jid, text) {
                const channel = channels.find(c => c.ownsJid(jid));
                return channel.sendMessage(jid, text);
            }
        """
        for channel in self.channels:
            if channel.owns_jid(jid) and channel.is_connected():
                channel.send_message(jid, text)
                return
        print(f"  ⚠️ No channel found for JID: {jid}")

    def connect_all(self):
        for ch in self.channels:
            ch.connect()

    def disconnect_all(self):
        for ch in self.channels:
            ch.disconnect()


def main():
    print("=" * 50)
    print("Chapter 9: Multi-Channel Gateway")
    print("=" * 50)
    print()

    gateway = Gateway()

    # Add terminal channel
    terminal = TerminalChannel(on_message=gateway.on_message)
    gateway.add_channel(terminal)

    # Add HTTP channel
    http = HttpChannel(port=5555, on_message=gateway.on_message)
    gateway.add_channel(http)

    gateway.connect_all()

    print("\nGateway running with 2 channels:")
    print("  1. Terminal: type messages below")
    print("  2. HTTP: curl -X POST http://127.0.0.1:5555/chat \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"user_id":"alice","message":"Hello!"}\'')
    print()
    print("Both channels feed into the same agent.\n")

    # Polling loop
    try:
        while True:
            messages = gateway.poll_messages()
            for msg in messages:
                channel_type = "HTTP" if msg["chat_jid"].endswith("@http") else "Terminal"
                print(
                    f"  📨 [{channel_type}] {msg['sender_name']}: {msg['content']}"
                )

                # Echo response (in practice, would call the agent)
                reply = f"[Echo from {channel_type}] {msg['content']}"
                gateway.send_reply(msg["chat_jid"], reply)

            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gateway...")
        gateway.disconnect_all()


if __name__ == "__main__":
    main()
