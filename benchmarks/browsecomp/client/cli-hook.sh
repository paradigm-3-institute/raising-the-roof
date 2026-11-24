# Sourced by agent.py's runner (RELEASE_TASK_CLI_HOOK) after it configures the CLI, with the
# CLI's name as $1 and the runner's $agent_id in scope. It turns on Codex's built-in web search
# for this task only: `-c web_search="live"` on the agent's command line overrides the harness's
# `web_search = "disabled"` in config.toml, which every other task keeps. The search runs on
# OpenAI's side (the container reaches only the model API). It also makes the agent's own
# directory. Returning non-zero stops the agent before the CLI starts.
# Mounted into the agents' container: it must not name the benchmark (agents can read it).
if [ "$1" != "codex" ]; then
  echo "this task searches the web with Codex's built-in web search; $1 has none here" >&2
  return 1
fi
task_cli_args=(-c 'web_search="live"')
mkdir -p "/work/${agent_id:?}" || return 1
