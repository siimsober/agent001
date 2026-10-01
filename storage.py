"""Persistence: project selection, the JSONL message log, and reports
computed from it (usage/cost and per-turn history).

The active project's log file and workspace dir live here as module state.
Other modules must call get_log_file() / get_workspace_dir() rather than
importing the values directly, because `from storage import X` would capture
the value at import time (before set_project has run)."""
import json
from datetime import datetime, timezone
from pathlib import Path

STARTER_AGENTS_MD = """# AGENTS.md

This is the starting brief for a new project. Replace this file with the
actual plan/instructions for the agent before running it non-interactively.
"""

# $ per 1M tokens (input, output).
# Verify current rates at https://claude.com/pricing before relying on this
# for real budgeting.
PRICING = {
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-sonnet-5":           (2.00, 10.00),
    "claude-opus-5":             (5.00, 25.00),
    "claude-fable-5":            (10.00, 50.00),
    "claude-fable-5-1":          (10.00, 50.00),
    "claude-mythos-5":           (10.00, 50.00),
}


# ---------------- project selection ----------------

_log_file = None
_workspace_dir = None


def get_log_file():
    if _log_file is None:
        raise RuntimeError("No project selected: call set_project() first.")
    return _log_file


def get_workspace_dir():
    if _workspace_dir is None:
        raise RuntimeError("No project selected: call set_project() first.")
    return _workspace_dir


def set_project(name):
    """Point the log/workspace at <project>'s own dirs, creating them if this
    is a new project. Old projects are left exactly as they are on disk, so
    you can always come back to them with -p <name>."""
    global _log_file, _workspace_dir
    _workspace_dir = Path("workspace") / name
    _log_file = Path("log") / name / "messages.jsonl"

    _workspace_dir.mkdir(parents=True, exist_ok=True)
    _log_file.parent.mkdir(parents=True, exist_ok=True)

    agents_md = _workspace_dir / "AGENTS.md"
    if not agents_md.exists() and not _log_file.exists():
        # Brand-new project: seed a starter brief so the non-interactive
        # first run has something to read instead of crashing.
        agents_md.write_text(STARTER_AGENTS_MD)
        print(f"[new project '{name}': created {agents_md} — "
              f"edit it before running non-interactively]")

    return _log_file, _workspace_dir


def list_projects():
    """List projects that have either a workspace dir or a log dir."""
    names = set()
    for base in (Path("workspace"), Path("log")):
        if base.exists():
            names.update(p.name for p in base.iterdir() if p.is_dir())
    return sorted(names)


# ---------------- message log ----------------

def log_message(message, direction="response", log_file=None):
    """Append a Claude API message object to a JSONL log file."""
    log_file = log_file or get_log_file()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "direction": direction,
        "message": (
            json.loads(message.model_dump_json())
            if hasattr(message, "model_dump_json")
            else message
        ),
    }
    with log_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def iter_log_entries(log_file=None):
    """Yield each parsed entry from the log. Yields nothing if the log
    doesn't exist yet."""
    log_file = log_file or get_log_file()
    if not log_file.exists():
        return
    with log_file.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_conversation(log_file=None):
    """Rebuild a list of {role, content} messages suitable for passing
    straight into messages.create(messages=...).

    Both "request" (already {role, content}) and "response" (full API
    response dump) entries have role + content, so we pull just those."""
    return [
        {"role": e["message"]["role"], "content": e["message"]["content"]}
        for e in iter_log_entries(log_file)
    ]


# ---------------- reports ----------------

def print_usage_summary(log_file=None):
    """Print a per-model token/cost summary table."""
    log_file = log_file or get_log_file()
    if not log_file.exists():
        print("No log file found.")
        return

    totals = {}  # model -> {"input": int, "output": int}
    for entry in iter_log_entries(log_file):
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


def _collect_turns(log_file):
    """Group log entries into turns: one per real user message, with
    tool-use roundtrips collapsed. Turns with no response are dropped."""
    turns = []
    current = None
    for entry in iter_log_entries(log_file):
        msg = entry["message"]
        direction = entry.get("direction")

        if direction == "request":
            content = msg.get("content")
            is_tool_result = (
                isinstance(content, list)
                and content
                and all(b.get("type") == "tool_result" for b in content)
            )
            if is_tool_result:
                continue  # continuation of the current turn
            current = {
                "ts": datetime.fromisoformat(entry["timestamp"]),
                "model": None, "calls": 0, "in": 0, "out": 0, "stop": "",
            }
            turns.append(current)

        elif direction == "response" and current is not None:
            usage = msg.get("usage") or {}
            current["model"] = msg.get("model") or current["model"]
            current["calls"] += 1
            current["in"] += usage.get("input_tokens", 0)
            current["out"] += usage.get("output_tokens", 0)
            current["stop"] = msg.get("stop_reason", "")

    return [t for t in turns if t["calls"] > 0]


def print_model_history(n=20, log_file=None):
    """Print the last n turns with model and token totals, then per-model
    turn counts."""
    log_file = log_file or get_log_file()
    if not log_file.exists():
        print("No log file found.")
        return

    turns = _collect_turns(log_file)
    if not turns:
        print("No model calls found in log.")
        return

    header = f"{'TIME (UTC)':<17} {'MODEL':<28} {'CALLS':>5} {'IN':>9} {'OUT':>8}  STOP"
    print(header)
    print("-" * len(header))
    for t in turns[-n:]:
        print(
            f"{t['ts'].strftime('%Y-%m-%d %H:%M'):<17} {t['model']:<28} "
            f"{t['calls']:>5} {t['in']:>9} {t['out']:>8}  {t['stop']}"
        )

    print("\nTurns per model:")
    summary = {}
    for t in turns:
        s = summary.setdefault(t["model"], {"turns": 0, "last": t["ts"]})
        s["turns"] += 1
        s["last"] = max(s["last"], t["ts"])
    for model, s in sorted(summary.items(), key=lambda kv: kv[1]["last"], reverse=True):
        print(f"  {model:<28} {s['turns']:>5} turns, "
              f"last {s['last'].strftime('%Y-%m-%d %H:%M')} UTC")