#!/usr/bin/env python3
"""
Chapter 5: Container Isolation
===============================

NanoClaw's core security model: running agents in containers.

NanoClaw runs each agent invocation in an isolated container
(Apple Container on macOS, Docker on Linux). The agent can only
see files explicitly mounted into the container.

Why containers?
- The agent can run shell commands safely (they execute IN the container)
- Each group's files are isolated from each other
- No application-level permission checks needed — OS-level isolation

Since we can't assume Docker in a tutorial, we use subprocess isolation:
- Agent runs in a subprocess with limited environment
- Working directory restricted to group's folder
- Communication via stdin/stdout JSON (like NanoClaw's IPC)

NanoClaw (container-runner.ts):
    const container = spawn('container', containerArgs, {
        stdio: ['pipe', 'pipe', 'pipe'],
    });
    container.stdin.write(JSON.stringify(input));
    container.stdin.end();

Run: uv run --with anthropic python chapters/05_container_isolation.py
"""

import json
import os
import subprocess
import sys
import tempfile

WORKSPACE = os.path.join(os.path.dirname(__file__), "..", "workspace")

# --- Agent Runner Code (runs in subprocess) ---

AGENT_RUNNER_CODE = '''
import json
import sys
import os

# Read input from stdin (same as NanoClaw's container input)
input_data = json.loads(sys.stdin.read())
prompt = input_data["prompt"]
group_folder = input_data["group_folder"]
workspace = input_data.get("workspace", ".")

# Load CLAUDE.md (if it exists)
soul = "You are Nano, a helpful AI assistant. Be concise."
claude_md = os.path.join(workspace, "CLAUDE.md")
if os.path.exists(claude_md):
    with open(claude_md) as f:
        soul = f.read()

# Output result to stdout (same pattern as NanoClaw's OUTPUT_MARKER)
OUTPUT_START = "---NANOCLAW_OUTPUT_START---"
OUTPUT_END = "---NANOCLAW_OUTPUT_END---"

try:
    # In practice, the Claude API would be called here, but in the demo we echo
    result = {
        "status": "success",
        "result": f"[Agent in {group_folder}] Processed: {prompt[:100]}",
        "soul_loaded": len(soul),
    }
except Exception as e:
    result = {"status": "error", "result": None, "error": str(e)}

# Output using NanoClaw's sentinel marker pattern
print(f"{OUTPUT_START}")
print(json.dumps(result))
print(f"{OUTPUT_END}")
'''

OUTPUT_START_MARKER = "---NANOCLAW_OUTPUT_START---"
OUTPUT_END_MARKER = "---NANOCLAW_OUTPUT_END---"


def run_in_subprocess(
    prompt: str,
    group_folder: str,
    workspace_path: str | None = None,
) -> dict:
    """
    Runs the agent in an isolated subprocess.

    A simplified subprocess version of NanoClaw's runContainerAgent().

    NanoClaw (container-runner.ts):
        const container = spawn('container', containerArgs, {
            stdio: ['pipe', 'pipe', 'pipe'],
        });
        container.stdin.write(JSON.stringify(input));
        container.stdin.end();

    Our version:
        proc = subprocess.Popen(['python3', '-c', code], ...)
        proc.stdin.write(json.dumps(input))
    """
    if workspace_path is None:
        workspace_path = os.path.join(WORKSPACE, "groups", group_folder)
    os.makedirs(workspace_path, exist_ok=True)

    input_data = {
        "prompt": prompt,
        "group_folder": group_folder,
        "workspace": workspace_path,
    }

    # Run subprocess (instead of container)
    proc = subprocess.Popen(
        [sys.executable, "-c", AGENT_RUNNER_CODE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace_path,  # Restrict working directory to group folder
        env={
            "PATH": os.environ.get("PATH", ""),
            "HOME": workspace_path,  # HOME is also isolated
            "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        },
    )

    stdout, stderr = proc.communicate(
        input=json.dumps(input_data), timeout=30
    )

    if stderr:
        print(f"  [stderr] {stderr.strip()}")

    # Same as NanoClaw's sentinel marker parsing
    start_idx = stdout.find(OUTPUT_START_MARKER)
    end_idx = stdout.find(OUTPUT_END_MARKER)

    if start_idx != -1 and end_idx != -1:
        json_str = stdout[
            start_idx + len(OUTPUT_START_MARKER) : end_idx
        ].strip()
        return json.loads(json_str)

    # If no marker found, try the last line
    lines = stdout.strip().split("\n")
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass

    return {
        "status": "error",
        "result": None,
        "error": f"Failed to parse output. Exit code: {proc.returncode}",
    }


def main():
    print("=" * 50)
    print("Chapter 5: Container Isolation")
    print("=" * 50)
    print()
    print("NanoClaw runs agents in isolated containers.")
    print("We simulate this with subprocess isolation.\n")

    # Create per-group folders and CLAUDE.md
    groups = {
        "family": "You are a warm, friendly family assistant. Korean is OK too!",
        "work": "You are a professional work assistant. Be precise and technical.",
    }

    for group_id, soul in groups.items():
        group_dir = os.path.join(WORKSPACE, "groups", group_id)
        os.makedirs(group_dir, exist_ok=True)
        with open(os.path.join(group_dir, "CLAUDE.md"), "w") as f:
            f.write(soul)
        print(f"✅ Created workspace/groups/{group_id}/CLAUDE.md")

    print()

    # Run agent in each group
    for group_id in groups:
        print(f"\n🚀 Running agent in '{group_id}' subprocess...")
        result = run_in_subprocess(
            prompt="Hello! What group am I in?",
            group_folder=group_id,
        )
        print(f"   Status: {result['status']}")
        if result.get("result"):
            print(f"   Result: {result['result']}")
        if result.get("soul_loaded"):
            print(f"   CLAUDE.md loaded: {result['soul_loaded']} chars")
        if result.get("error"):
            print(f"   Error: {result['error']}")

    print("\n" + "=" * 50)
    print("\nNanoClaw's container isolation:")
    print("  1. Each group gets its own container")
    print("  2. Only that group's folder is mounted")
    print("  3. Communication via stdin/stdout JSON")
    print("  4. Sentinel markers for robust output parsing")
    print()
    print("Our subprocess simulation provides similar isolation:")
    print("  - Restricted working directory")
    print("  - Limited environment variables")
    print("  - JSON IPC via stdin/stdout")


if __name__ == "__main__":
    main()
