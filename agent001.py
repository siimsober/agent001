import json
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

client = anthropic.Anthropic()  # automatically picks up ANTHROPIC_API_KEY

history = load_conversation()

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

log_message(message, direction = "request")
print(message)
response = client.messages.create(
    model = "claude-haiku-4-5-20251001",
    max_tokens =1000,
    messages = history + [message],
)

log_message(response)
print(response)

for block in response.content:
    if block.type == "text":
        print(block.text)
