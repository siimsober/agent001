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

client = anthropic.Anthropic()  # automatically picks up ANTHROPIC_API_KEY

message = {
    "role": "user",
    "content": (
        "Goal is to develop a breast cancer polygenic risk score calculation medical device software. "
        "The software needs to be developed, implemented, and confirm to IVDR regulation and sufficiently developed for notified body review. "
        "The plan may be gritiqued and improved. This agent will run once per day to make progress towards the goal.\n"
        "Try to respond in one token. Max is set to 2."
    ),
}
log_message(message, direction = "request")

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    messages=[message],
)

log_message(response)
print(message)

for block in response.content:
    if block.type == "text":
        print(block.text)
