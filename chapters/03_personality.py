#!/usr/bin/env python3
"""
Chapter 3: Per-Group Personality (CLAUDE.md)
============================================

Core idea of NanoClaw: a different personality for each group.

In NanoClaw, each WhatsApp group gets its own CLAUDE.md file.
This is like SOUL.md but per-context. The family group gets a casual
assistant, the work group gets a professional one.

NanoClaw stores these in:
    groups/{group_folder}/CLAUDE.md

Each group folder is isolated — the agent in one group can't see
another group's files or CLAUDE.md.

This chapter demonstrates loading per-group personalities.

Run: uv run --with anthropic python chapters/03_personality.py
"""

import os
import anthropic

GROUPS_DIR = os.path.join(os.path.dirname(__file__), "..", "workspace", "groups")

# Default system prompt
DEFAULT_SOUL = """You are Nano, a helpful AI assistant.
Be concise and direct. Skip the pleasantries."""

# Per-group CLAUDE.md for demo
EXAMPLE_SOULS: dict[str, str] = {
    "family": """# Family Chat Assistant

You are Nano, the family's helpful AI.
- Feel free to mix Korean and English in conversation
- Be warm and friendly, use emoji sometimes
- Help with everyday questions: recipes, schedules, recommendations
- Keep responses short — this is a family chat, not an essay
- Never share info from other groups
""",
    "work-project": """# Work Project Assistant

You are Nano, the team's technical assistant.
- Be professional and precise
- When discussing code, always show examples
- Cite sources when making claims
- Keep responses focused on the task at hand
- You have access to the project files in /workspace/group/
""",
    "personal": """# Personal Assistant (Main Channel)

You are Nano, my personal AI assistant.
- This is the main channel — you have full access
- Be direct and honest, skip the corporate speak
- You can manage other groups from here
- Remember my preferences and habits
- Save important things to memory
""",
}


def ensure_group_dir(group_id: str) -> str:
    """Creates the group folder and returns its path."""
    group_dir = os.path.join(GROUPS_DIR, group_id)
    os.makedirs(group_dir, exist_ok=True)
    return group_dir


def get_soul_for_group(group_id: str) -> str:
    """
    Loads the group's CLAUDE.md. Same pattern as NanoClaw's per-group mount.

    In NanoClaw:
    - The groups/{folder}/CLAUDE.md file is mounted into the container
    - Claude Code automatically loads CLAUDE.md as the system prompt
    - We read it directly and pass it as the system prompt

    NanoClaw (container-runner.ts):
        mounts.push({
            hostPath: path.join(GROUPS_DIR, group.folder),
            containerPath: '/workspace/group',
            readonly: false,
        });
    """
    group_dir = ensure_group_dir(group_id)
    claude_md = os.path.join(group_dir, "CLAUDE.md")

    if os.path.exists(claude_md):
        with open(claude_md, "r") as f:
            return f.read()

    return DEFAULT_SOUL


def save_soul_for_group(group_id: str, content: str):
    """Saves the group's CLAUDE.md."""
    group_dir = ensure_group_dir(group_id)
    claude_md = os.path.join(group_dir, "CLAUDE.md")
    with open(claude_md, "w") as f:
        f.write(content)


def demo_personality(group_id: str, message: str):
    """Demo of responding with a specific group's personality."""
    soul = get_soul_for_group(group_id)
    print(f"\n--- Group: {group_id} ---")
    print(f"CLAUDE.md loaded ({len(soul)} chars)")
    print(f"User: {message}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"🤖 [Would respond with {group_id} personality]")
        return

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=512,
        system=soul,
        messages=[{"role": "user", "content": message}],
    )
    print(f"🤖 Nano: {response.content[0].text}")


def main():
    print("=" * 50)
    print("Chapter 3: Per-Group Personality (CLAUDE.md)")
    print("=" * 50)
    print()
    print("NanoClaw gives each group its own CLAUDE.md.")
    print("Each group gets a different personality.\n")

    # Create CLAUDE.md for demo groups
    for group_id, soul in EXAMPLE_SOULS.items():
        save_soul_for_group(group_id, soul)
        print(f"✅ Created CLAUDE.md for '{group_id}'")

    print()

    # Same question in each group yields different personality responses
    question = "How do I make good coffee?"

    for group_id in EXAMPLE_SOULS:
        demo_personality(group_id, question)

    print("\n" + "=" * 50)
    print("\nKey insight from NanoClaw:")
    print("  Each group = isolated folder + isolated CLAUDE.md")
    print("  The agent in 'family' can't see 'work-project' files")
    print("  Container isolation enforces this at the OS level")
    print()
    print("File structure:")
    for group_id in EXAMPLE_SOULS:
        group_dir = os.path.join(GROUPS_DIR, group_id)
        print(f"  {group_dir}/CLAUDE.md")


if __name__ == "__main__":
    main()
