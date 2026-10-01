"""The tools the model can call: definitions plus client-side execution,
sandboxed to the active project's workspace."""
import subprocess
from pathlib import Path

import storage

TOOLS = [
    {
        "type": "text_editor_20250728",
        "name": "str_replace_based_edit_tool",
        "max_characters": 10000,
    },
    {"type": "bash_20250124", "name": "bash"},
]


def resolve_workspace_path(path_str):
    """Resolve a path the model gives us, sandboxed to the workspace.
    Rejects anything that would escape it (e.g. '../../etc/passwd')."""
    workspace = storage.get_workspace_dir()
    p = Path(path_str)
    if p.is_absolute():
        p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(".")
    full = (workspace / p).resolve()
    workspace_root = workspace.resolve()
    if full != workspace_root and workspace_root not in full.parents:
        raise ValueError(f"Refused: '{path_str}' resolves outside the workspace directory.")
    return full


def handle_text_editor(tool_input):
    """Returns (output_text, is_error)."""
    command = tool_input.get("command")
    try:
        path = resolve_workspace_path(tool_input["path"])
    except ValueError as e:
        return str(e), True

    if command == "view":
        if path.is_dir():
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in path.iterdir())
            return "\n".join(entries) or "(empty directory)", False
        if not path.exists():
            return f"File not found: {path}", True
        text = path.read_text(errors="replace")
        view_range = tool_input.get("view_range")
        if view_range:
            lines = text.splitlines()
            start, end = view_range
            end = len(lines) if end == -1 else end
            text = "\n".join(lines[start - 1:end])
        return text, False

    elif command == "create":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tool_input.get("file_text", ""))
        return f"File created: {path}", False

    elif command == "str_replace":
        if not path.exists():
            return f"File not found: {path}", True
        text = path.read_text()
        old, new = tool_input["old_str"], tool_input.get("new_str", "")
        count = text.count(old)
        if count == 0:
            return "old_str not found in file.", True
        if count > 1:
            return f"old_str is not unique ({count} occurrences) — refusing to guess.", True
        path.write_text(text.replace(old, new, 1))
        return "Replacement applied.", False

    elif command == "insert":
        if not path.exists():
            return f"File not found: {path}", True
        lines = path.read_text().splitlines()
        insert_line = tool_input["insert_line"]
        # current tool uses "insert_text"; fall back to "new_str" for older tool versions
        new_text = tool_input.get("insert_text", tool_input.get("new_str"))
        if new_text is None:
            return "insert requires 'insert_text'.", True
        lines[insert_line:insert_line] = new_text.splitlines()
        path.write_text("\n".join(lines) + "\n")
        return "Insert applied.", False

    return f"Unknown text editor command: {command}", True


def handle_bash(tool_input):
    """Only 'ls' commands are permitted. Extend this allowlist deliberately,
    not by accident."""
    command = tool_input.get("command", "").strip()
    first_word = command.split(" ")[0] if command else ""
    forbidden_chars = [";", "&&", "||", "|", "`", "$(", ">", "<"]

    if first_word != "ls" or any(ch in command for ch in forbidden_chars):
        return "Refused: this agent only permits plain 'ls' commands.", True

    try:
        result = subprocess.run(
            command, shell=True, cwd=storage.get_workspace_dir(),
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out.", True

    output = (result.stdout + result.stderr).strip()
    return output or "(no output)", False


def execute_tool(name, tool_input):
    try:
        if name == "str_replace_based_edit_tool":
            return handle_text_editor(tool_input)
        if name == "bash":
            return handle_bash(tool_input)
        return f"Unknown tool: {name}", True
    except Exception as e:
        return f"Tool error: {type(e).__name__}: {e}", True