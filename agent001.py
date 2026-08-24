from dotenv import load_dotenv
load_dotenv()  # reads .env and sets the environment variable

import anthropic
client = anthropic.Anthropic()  # automatically picks up ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1000,
    messages=[
        {
            "role": "user",
            "content": (
                "Goal is to develop a breast cancer polygenic risk score calculation medical device software. "
                "The software needs to be developed, implemented, and confirm to IVDR regulation and sufficiently developed for notified body review. "
                "The plan may be gritiqued and improved. This agent will run once per day to make progress towards the goal.\n"
                "Try to respond in one token. Max is set to 2."
                ),
        }
    ],
)

print(message)

for block in message.content:
    if block.type == "text":
        print(block.text)
