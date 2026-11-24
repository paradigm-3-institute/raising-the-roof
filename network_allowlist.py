"""Network allowlists: agents that reach their model API and nothing else.

One mechanism for every task whose public solutions or verifiers must stay out of the
agents' reach (FrontierSWE v2, ALE-Bench, the performance take-home, BrowseComp). The task sets
`network_mode = "allowlist"` in its task.toml `[environment]`, with `allowed_hosts`
for anything it needs before or after the agents run (none for FrontierSWE; the
Ubuntu mirrors for the scored-artifact tasks, whose agent CLI installs packages). Its
adapter adds the agent CLI's model API hosts for the agent phase only, with Harbor's
`--allow-agent-host` (harbor_args). Harbor's docker environment enforces both with an
egress sidecar that shares the agent container's network: nftables sends every TCP
connection through a transparent proxy (gost) that passes only allowlisted host names
(TLS SNI or HTTP Host), and rejects other UDP than DNS; Harbor switches the sidecar to
the agent-phase list while the agents run and back afterwards. A separate verifier
environment keeps its own policy and never inherits these hosts.
"""

from __future__ import annotations

# The hosts each agent CLI needs to reach its model, by sign-in method.
MODEL_API_HOSTS = {
    "codex": ("chatgpt.com", "auth.openai.com"),  # ChatGPT login (auth.json)
    "codex-api-key": ("api.openai.com",),  # CODEX_API_KEY
    "claude": ("api.anthropic.com",),
}


def task_network_mode(settings: dict) -> str:
    """The [environment] network_mode of a parsed task.toml (Harbor's default: public)."""
    return str((settings.get("environment") or {}).get("network_mode") or "public")


def model_api_hosts(cli: str | None, *, codex_login: bool) -> list[str]:
    """The hosts the agent CLI needs (a mixed team's CLIs joined by "+", run.py --team: each one's)."""
    if cli is None:
        return []
    if "+" in cli:
        return list(dict.fromkeys(host for part in cli.split("+")
                                  for host in model_api_hosts(part, codex_login=codex_login)))
    if cli == "codex":
        return list(MODEL_API_HOSTS["codex" if codex_login else "codex-api-key"])
    return list(MODEL_API_HOSTS[cli])


def allowed_hosts(cli: str | None, *, codex_login: bool, extra=()) -> list[str]:
    """The model API hosts plus any extra hosts, each once."""
    return list(dict.fromkeys([*model_api_hosts(cli, codex_login=codex_login), *extra]))


def harbor_args(hosts) -> list[str]:
    """Harbor options that open these hosts while the agents run, and only then."""
    args = []
    for host in hosts:
        args += ["--allow-agent-host", host]
    return args
