import json
import argparse
from datetime import datetime, timezone
from dotenv import load_dotenv
from pathlib import Path
import anthropic

load_dotenv()  # reads .env and sets the environment variable
LOG_FILE = Path("log/messages.jsonl")

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
# Verify current rates at https://claude.com/pricing before relying on this for real budgeting.
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

def send_message(client, history, message, model="claude-haiku-4-5-20251001", max_tokens=2000):
    """Log the outgoing message, call the API, log and return the response."""
    log_message(message, direction="request")
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=history + [message],
    )
    log_message(response)
    return response

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
        response = send_message(client, history, message)

        # keep growing history so later turns in this session have context
        history.append(message)
        history.append({"role": "assistant", "content": response.content[0].text})

        for block in response.content:
            if block.type == "text":
                print(block.text)
        print(f"\n[Input tokens: {response.usage.input_tokens} | Output tokens: {response.usage.output_tokens}]\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--interactive", action="store_true",
                         help="Start an interactive prompt loop instead of sending the automatic progress message.")
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
                "Try to respond in one token. Max is set to 2."
            ),
        }
    else:
        message = {
            "role": "user",
            "content": "Continue making progress on the plan from where you left off.",
        }

    print(message["content"])
    response = send_message(client, history, message)

    for block in response.content:
        if block.type == "text":
            print(block.text)
    print(f"\nInput tokens:  {response.usage.input_tokens}")
    print(f"Output tokens: {response.usage.output_tokens}")

    print_usage_summary()


if __name__ == "__main__":
    main()