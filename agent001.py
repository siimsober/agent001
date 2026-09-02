import json
import argparse
import subprocess
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import anthropic

load_dotenv()  # reads .env and sets the environment variable
LOG_FILE = Path("log/messages.jsonl")
WORKSPACE_DIR = Path("workspace")
WORKSPACE_DIR.mkdir(exist_ok=True)

TOOLS = [
    {
        "type": "text_editor_20250728", 
        "name": "str_replace_based_edit_tool", 
        "max_characters": 10000
    },
    {"type": "bash_20250124", "name": "bash"},
]

def log_message(message, direction="response"):
    """Append a Claude API message object to a JSONL log file."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "message": (
            json.loads(message.model_dump_json())
            if hasattr(message, "model_dump_json")
            else message
        ),
    }
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")

def load_conversation(log_file=LOG_FILE):
    """Read the JSONL log and rebuild a list of {role, content} messages
    suitable for passing straight into messages.create(messages=...)."""
    messages = []
    if not log_file.exists():
        return messages

    with log_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            msg = entry["message"]
            # Both "request" (already {role, content}) and "response"
            # (full API response dump) entries have role + content,
            # so we just pull those two fields out.
            messages.append({
                "role": msg["role"],
                "content": msg["content"],
            })
    return messages

# --- pricing table: $ per 1M tokens (input, output) ---
# Verify current rates at https://claude.com/pricing before relying on this for real 
# budgeting.
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5":           (2.00, 10.00),
    "claude-opus-5":             (5.00, 25.00),
    "claude-fable-5":            (10.00, 50.00),
    "claude-mythos-5":           (10.00, 50.00),
}

def print_usage_summary(log_file=LOG_FILE):
    """Read the JSONL log and print a per-model token/cost summary table."""
    totals = {}  # model -> {"input": int, "output": int}

    if not log_file.exists():
        print("No log file found.")
        return

    with log_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("direction") != "response":
                continue
            msg = entry["message"]
            usage = msg.get("usage")
            model = msg.get("model")
            if not usage or not model:
                continue
            stats = totals.setdefault(model, {"input": 0, "output": 0})
            stats["input"] += usage.get("input_tokens", 0)
            stats["output"] += usage.get("output_tokens", 0)

    if not totals:
        print("No usage data found in log.")
        return

    header = f"{'MODEL':<30} {'INPUT':>10} {'OUTPUT':>10} {'COST ($)':>10}"
    print(header)
    print("-" * len(header))

    grand_total = 0.0
    for model, stats in sorted(totals.items()):
        in_tok, out_tok = stats["input"], stats["output"]
        if model in PRICING:
            in_rate, out_rate = PRICING[model]
            cost = (in_tok / 1_000_000) * in_rate + (out_tok / 1_000_000) * out_rate
        else:
            cost = float("nan")  # unknown model, can't price it
        grand_total += 0 if cost != cost else cost  # skip NaN
        print(f"{model:<30} {in_tok:>10} {out_tok:>10} {cost:>10.4f}")

    print("-" * len(header))
    print(f"{'TOTAL':<30} {'':>10} {'':>10} {grand_total:>10.4f}")

# ---------------- tool execution (client-side) ----------------

def resolve_workspace_path(path_str):
    """Resolve a path the model gives us, sandboxed to WORKSPACE_DIR.
    Rejects anything that would escape the workspace (e.g. '../../etc/passwd')."""
    p = Path(path_str)
    if p.is_absolute():
        p = Path(*p.parts[1:]) if len(p.parts) > 1 else Path(".")
    full = (WORKSPACE_DIR / p).resolve()
    workspace_root = WORKSPACE_DIR.resolve()
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
            if not path.exists():
                return f"Directory not found: {path}", True
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
        lines[insert_line:insert_line] = tool_input["new_str"].splitlines()
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
            command, shell=True, cwd=WORKSPACE_DIR,
            capture_output=True, text=True, timeout=10,
        )
    except subprocess.TimeoutExpired:
        return "Command timed out.", True

    output = (result.stdout + result.stderr).strip()
    return output or "(no output)", False

def execute_tool(name, tool_input):
    if name == "str_replace_based_edit_tool":
        return handle_text_editor(tool_input)
    if name == "bash":
        return handle_bash(tool_input)
    return f"Unknown tool: {name}", True

# ---------------- agent turn with tool loop ----------------

def run_agent_turn(
        client, history, user_message, model="claude-haiku-4-5-20251001", max_tokens=2000
    ):
    """Sends user_message, executes any tool_use requests, and loops until
    Claude stops asking for tools. Every request/response is logged."""
    log_message(user_message, direction="request")
    messages = history + [user_message]

    while True:
        response = client.messages.create(
            model=model, max_tokens=max_tokens, tools=TOOLS, messages=messages,
        )
        log_message(response)

        assistant_content = json.loads(response.model_dump_json())["content"]
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            return messages, response

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                output_text, is_error = execute_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output_text,
                    "is_error": is_error,
                })

        tool_result_message = {"role": "user", "content": tool_results}
        log_message(tool_result_message, direction="request")
        messages.append(tool_result_message)

def print_response(response):
    for block in response.content:
        if block.type == "text":
            print(block.text)
    print(f"\n[Input tokens: {response.usage.input_tokens} | Output tokens: {response.usage.output_tokens}]")


# ---------------- modes ----------------

def run_interactive(client, history):
    print("Interactive mode. Type 'exit' or 'quit' to stop.\n")
    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        message = {"role": "user", "content": user_input}
        history, response = run_agent_turn(client, history, message)
        print_response(response)
        print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Start an interactive prompt loop instead of sending the automatic progress message."
    )
    args = parser.parse_args()

    client = anthropic.Anthropic()
    history = load_conversation()

    if args.interactive:
        run_interactive(client, history)
        print_usage_summary()
        return

    if not history:
        message = {
            "role": "user",
            "content": (
                "Goal is to develop a breast cancer polygenic risk score calculation medical device software. "
                "The software needs to be developed, implemented, and confirm to IVDR regulation and sufficiently developed for notified body review. "
                "The plan may be gritiqued and improved. This agent will run once per day to make progress towards the goal.\n"
                "You have a text editor tool and a restricted 'ls' bash command, both scoped to the workspace/ directory, "
                "for reading and writing project files."
            ),
        }
    else:
        message = {
            "role": "user",
            "content": "Continue making progress on the plan from where you left off.",
        }

    print(message["content"])
    history, response = run_agent_turn(client, history, message)
    print_response(response)
    print_usage_summary()

if __name__ == "__main__":
    main()