#!/usr/bin/env python3
"""
Chapter 4: Tool-Using Agent Loop
=================================

The agent loop: an AI that uses tools.

The agent loop is the core execution pattern:
1. Send messages + tools to Claude
2. If Claude wants to use a tool → execute it, feed result back
3. Repeat until Claude gives a final text response

NanoClaw delegates this to Claude Code (Claude Agent SDK) which handles
tool execution internally. We build our own simplified version.

Run: uv run --with anthropic python chapters/04_agent_loop.py
"""

import json
import os
import subprocess
import anthropic

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "workspace")

# --- Tool Definitions ---

TOOLS = [
    {
        "name": "run_command",
        "description": "Run a shell command. Returns stdout + stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                }
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to workspace",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to workspace",
                },
                "content": {"type": "string", "description": "File content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "web_search",
        "description": "Search the web. Returns search results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"],
        },
    },
]


def execute_tool(name: str, tool_input: dict) -> str:
    """Executes a tool and returns the result."""
    if name == "run_command":
        try:
            result = subprocess.run(
                tool_input["command"],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=WORKSPACE,
            )
            output = result.stdout + result.stderr
            return output if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out (30s)"
        except Exception as e:
            return f"Error: {e}"

    elif name == "read_file":
        try:
            filepath = os.path.join(WORKSPACE, tool_input["path"])
            with open(filepath, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error: {e}"

    elif name == "write_file":
        try:
            filepath = os.path.join(WORKSPACE, tool_input["path"])
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w") as f:
                f.write(tool_input["content"])
            return f"Written to {tool_input['path']}"
        except Exception as e:
            return f"Error: {e}"

    elif name == "web_search":
        # In practice, would use Brave Search API, etc.
        return f"[Mock search results for: {tool_input['query']}]"

    return f"Unknown tool: {name}"


def serialize_content(content) -> list[dict]:
    """Converts API response content blocks into serializable dicts."""
    serialized = []
    for block in content:
        if hasattr(block, "text"):
            serialized.append({"type": "text", "text": block.text})
        elif block.type == "tool_use":
            serialized.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                }
            )
    return serialized


def run_agent_turn(
    messages: list[dict], system_prompt: str
) -> tuple[str, list[dict]]:
    """
    Runs an agent turn. Repeats tool calls and returns the final text.

    In NanoClaw, the Claude Code SDK handles this loop internally.
    We implement the same pattern using the Anthropic API directly.
    """
    client = anthropic.Anthropic()

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        content = serialize_content(response.content)

        # Completed without tool calls -> return final text
        if response.stop_reason == "end_turn":
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
            messages.append({"role": "assistant", "content": content})
            return "\n".join(text_parts), messages

        # Tool calls present -> execute and feed results back
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  🔧 {block.name}: {json.dumps(block.input)}")
                    result = execute_tool(block.name, block.input)
                    print(f"     → {result[:200]}")
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})


def main():
    print("=" * 50)
    print("Chapter 4: Tool-Using Agent Loop")
    print("=" * 50)
    print()

    system_prompt = """You are Nano, a helpful AI assistant.
You have tools to run commands, read/write files, and search the web.
Use them when needed. Be concise."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("⚠️  Set ANTHROPIC_API_KEY to run this chapter.")
        print("   Example: ANTHROPIC_API_KEY=sk-... python chapters/04_agent_loop.py")
        print()
        print("The agent loop pattern:")
        print("  1. User message → Claude API (with tools)")
        print("  2. Claude returns tool_use → execute tool → feed result back")
        print("  3. Repeat until Claude returns end_turn")
        print("  4. Return final text to user")
        return

    messages: list[dict] = []

    print("Interactive agent REPL. Type messages, the agent can use tools.")
    print("Commands: /quit, /new (reset session)\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/new":
            messages = []
            print("🔄 Session reset.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            response_text, messages = run_agent_turn(messages, system_prompt)
            print(f"\n🤖 Nano: {response_text}\n")
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            # Remove the failed message
            if messages and messages[-1]["role"] == "user":
                messages.pop()


if __name__ == "__main__":
    main()
