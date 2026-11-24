"""Checks for network allowlists (network_allowlist.py and its users). No Docker needed."""

from __future__ import annotations

import sys


from cli_render import ROOT

sys.path.insert(0, str(ROOT))
import network_allowlist  # noqa: E402


def test_model_api_hosts():
    assert network_allowlist.model_api_hosts("codex", codex_login=True) == ["chatgpt.com", "auth.openai.com"]
    assert network_allowlist.model_api_hosts("codex", codex_login=False) == ["api.openai.com"]
    assert network_allowlist.model_api_hosts("claude", codex_login=False) == ["api.anthropic.com"]
    assert network_allowlist.model_api_hosts(None, codex_login=False) == []
    assert network_allowlist.allowed_hosts("claude", codex_login=False, extra=["x.org", "api.anthropic.com"]) \
        == ["api.anthropic.com", "x.org"]
    # Opened for the agent phase only.
    assert network_allowlist.harbor_args(["a.org", "b.org"]) == [
        "--allow-agent-host", "a.org", "--allow-agent-host", "b.org"]


def test_one_host_list():
    import scored_artifact
    assert scored_artifact.MODEL_API_HOSTS is network_allowlist.MODEL_API_HOSTS


def test_task_network_mode():
    assert network_allowlist.task_network_mode({}) == "public"
    assert network_allowlist.task_network_mode({"environment": {"network_mode": "allowlist"}}) == "allowlist"


