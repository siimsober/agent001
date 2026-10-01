"""Claude agent: config, CLI, retry logic and the agent loop.

Usage:
  python agent.py                      # automatic run (default project)
  python agent.py -i -m sonnet         # interactive
  python agent.py -p myproj --history  # last 20 turns
"""
import argparse
import json
import random
import time
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv

import storage
from storage import load_conversation, log_message, print_model_history, print_usage_summary
from tools import TOOLS, execute_tool

MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-5",
    "fable": "claude-fable-5-1",
}
DEFAULT_MODEL = "haiku"
DEFAULT_PROJECT = "default"
MAX_TOKENS = 10000


# ---------------- retry ----------------

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_RETRIES = 6


def _retry_delay(response, attempt):
    """Seconds to wait: retry-after header if present, else backoff + jitter."""
    if response is not None:
        header = response.headers.get("retry-after")
        if header:
            try:
                return float(header) + random.uniform(0, 1)  # small jitter
            except ValueError:
                pass  # HTTP-date format; fall through to backoff
    return min(60, 2 ** attempt) + random.uniform(0, 1)


def create_with_retry(client, **kwargs):
    """messages.create with retry on rate limits (429), overloads (529),
    transient 5xx errors and connection problems. Honors the retry-after
    header when the server sends one, otherwise uses exponential backoff
    with jitter. Non-retryable errors (400, 401, 403, 404...) raise
    immediately."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.messages.create(**kwargs)

        except anthropic.APIStatusError as e:
            if e.status_code not in RETRYABLE_STATUS or attempt == MAX_RETRIES:
                raise
            wait = _retry_delay(e.response, attempt)
            print(f"[{e.status_code} {type(e).__name__} — retry {attempt + 1}/{MAX_RETRIES} "
                  f"in {wait:.1f}s]")
            time.sleep(wait)

        except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
            if attempt == MAX_RETRIES:
                raise
            wait = _retry_delay(None, attempt)
            print(f"[{type(e).__name__} — retry {attempt + 1}/{MAX_RETRIES} in {wait:.1f}s]")
            time.sleep(wait)


# ---------------- agent turn with tool loop ----------------

def run_agent_turn(
    client, history, user_message, model="claude-haiku-4-5-20251001",
    max_tokens=MAX_TOKENS,
):
    """Sends user_message, executes any tool_use requests, and loops until
    Claude stops asking for tools. Every request/response is logged."""
    log_message(user_message, direction="request")
    messages = history + [user_message]
    turn_usage = {"input_tokens": 0, "output_tokens": 0}

    while True:
        response = create_with_retry(
            client, model=model, max_tokens=max_tokens, tools=TOOLS, messages=messages,
            betas=["context-management-2025-06-27"],
            context_management={"edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 60000},
                "keep": {"type": "tool_uses", "value": 6},
            }]},
        )
        log_message(response)
        turn_usage["input_tokens"] += response.usage.input_tokens
        turn_usage["output_tokens"] += response.usage.output_tokens

        assistant_content = json.loads(response.model_dump_json())["content"]
        messages.append({"role": "assistant", "content": assistant_content})

        if response.stop_reason != "tool_use":
            return messages, response, turn_usage

        tool_results = []
        for block in response.content:
            if block.type == "thinking":
                if block.thinking:  # can be empty (redacted/omitted)
                    print(f"[thinking]\n{block.thinking}\n")
            elif block.type == "text":
                print(block.text)
            elif block.type == "tool_use":
                print(f"[tool_use: {block.name}] {json.dumps(block.input)[:100]}")
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


def print_response(response, show_thinking=True, turn_usage=None):
    """turn_usage, if given, is the {"input_tokens", "output_tokens"} total
    for the whole turn (all tool-use round-trips)."""
    has_text = False
    for block in response.content:
        if block.type == "thinking" and show_thinking:
            if block.thinking:  # can be empty string, e.g. redacted/truncated
                print(f"[thinking]\n{block.thinking}\n")
        elif block.type == "text":
            print(block.text)
            has_text = True

    if not has_text:
        print(f"[No text output — stop_reason: {response.stop_reason}]")
        if response.stop_reason == "max_tokens":
            print("[Hit max_tokens before producing a reply — consider raising max_tokens.]")

    if turn_usage is not None:
        print(
            f"\n[Turn total — Input tokens: {turn_usage['input_tokens']} | "
            f"Output tokens: {turn_usage['output_tokens']}]"
        )
    print(
        f"\n[Response input tokens: {response.usage.input_tokens} | "
        f"Response output tokens: {response.usage.output_tokens}]"
    )


# ---------------- CLI ----------------

def timestamp_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def run_interactive(client, history, model):
    print(f"Interactive mode (model: {model}). Type 'exit' or 'quit' to stop.\n")
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

        message = {"role": "user", "content": f"Time is {timestamp_str()}\n{user_input}"}
        history, response, turn_usage = run_agent_turn(client, history, message, model=model)
        print_response(response, turn_usage=turn_usage)
        print()


def run_automatic(client, history, model, project_name):
    """Non-interactive: send AGENTS.md on the first run, otherwise a
    'keep going' nudge."""
    if not history:
        brief = (storage.get_workspace_dir() / "AGENTS.md").read_text()
        content = f"Time is {timestamp_str()}\n{brief}"
    else:
        content = (
            f"Time is {timestamp_str()}\n"
            "Continue making progress on the plan from where you left off."
        )
    message = {"role": "user", "content": content}

    print(f"[project: {project_name} | model: {model}]")
    print(message["content"])
    history, response, turn_usage = run_agent_turn(client, history, message, model=model)
    print_response(response, turn_usage=turn_usage)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--interactive", action="store_true",
        help="Start an interactive prompt loop instead of sending the automatic "
             "progress message.",
    )
    parser.add_argument(
        "-m", "--model", choices=MODEL_ALIASES.keys(), default=DEFAULT_MODEL,
        help=f"Which model to use (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "-p", "--project", default=DEFAULT_PROJECT,
        help="Project name. Each project gets its own workspace/<name>/ and "
             "log/<name>/messages.jsonl, so different projects never mix "
             f"histories or files. (default: {DEFAULT_PROJECT})",
    )
    parser.add_argument(
        "--list-projects", action="store_true",
        help="List known projects (by workspace/log directory) and exit.",
    )
    parser.add_argument(
        "--history", nargs="?", const=20, type=int, metavar="N",
        help="Show the last N turns (default 20) for the selected project and exit.",
    )
    return parser


def main():
    load_dotenv()  # reads .env and sets the environment variables
    args = build_parser().parse_args()

    if args.list_projects:
        projects = storage.list_projects()
        if not projects:
            print("No projects found yet.")
        else:
            print("Known projects:")
            for name in projects:
                marker = " (default)" if name == DEFAULT_PROJECT else ""
                print(f"  - {name}{marker}")
        return

    model = MODEL_ALIASES[args.model]
    storage.set_project(args.project)

    if args.history is not None:
        print_model_history(args.history)
        return

    client = anthropic.Anthropic(max_retries=0)
    history = load_conversation()

    if args.interactive:
        run_interactive(client, history, model)
    else:
        run_automatic(client, history, model, args.project)
    print_usage_summary()


if __name__ == "__main__":
    main()