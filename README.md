# Agent001 - AI agent harness

## Structure

### agent.py

model aliases and defaults, MAX_TOKENS, the retry logic, run_agent_turn, print_response, the interactive and automatic modes, and main(). This is the file you'll edit most.

### tools.py

the TOOLS definitions, the workspace path sandbox, the text editor and bash handlers, and execute_tool. Adding a tool means editing just this file.
### storage.py

the starter brief, PRICING, project selection, the JSONL log read/write, and both reports (print_usage_summary and print_model_history).

## Project paths

tools.py and agent.py read the active paths through storage.get_workspace_dir() and storage.get_log_file(). Importing the values directly would capture None before set_project() runs.

## Miscellaneous

Dotenv: load_dotenv() runs at the start of main().