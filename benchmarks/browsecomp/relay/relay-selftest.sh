#!/bin/bash
# Self-test of the search filter's relay outside Harbor: the relay container on its own Docker network, then one
# tiny `codex exec` (gpt-5.6-sol, effort low, one web search) through it, plus curl checks of the paths it must
# refuse. Prints the relay's log (no tokens, no login contents). Spends a few thousand tokens. Needs the relay image
# and TLS files (python browsecomp_task.py check ... with BROWSECOMP_SEARCH_FILTER=on makes them) and the Codex
# package in .cache/codex-0.162.1.
#   bash benchmarks/browsecomp/relay/relay-selftest.sh [dry-run|on] [debug]
set -u
MODE=${1:-dry-run}
TREE=$(cd "$(dirname "$0")/../../.." && pwd)
SRC=$TREE/benchmarks/browsecomp/relay
TLS=$TREE/.cache/browsecomp/relay-tls
PROT=$TREE/.cache/browsecomp/protected-questions
AUTH=${CODEX_AUTH_JSON:-$HOME/.codex/auth.json}
W=$(mktemp -d /tmp/relay-selftest.XXXX); chmod 700 "$W"; mkdir -p "$W/state" "$W/home"
NET=relay-selftest-$$
docker network create "$NET" >/dev/null
EXTRA=(); [ "$MODE" = dry-run ] && EXTRA+=(--dry-run)
DEBUG=(); [ "${2:-}" = debug ] && DEBUG+=(-e RELAY_DEBUG_FIELDS=1)
docker run -d --rm --name relay-selftest-$$ --network "$NET" --network-alias relay "${DEBUG[@]}" \
  -v "$SRC:/relay/app:ro" -v "$PROT:/relay/protected:ro" -v "$TLS/relay.pem:/relay/tls/relay.pem:ro" \
  -v "$TLS/relay-key.pem:/relay/tls/relay-key.pem:ro" -v "$W/state:/relay/state" \
  --entrypoint python3 ttc-browsecomp-relay:1 -I /relay/app/proxy.py --host 0.0.0.0 --port 8789 \
  --tls-cert /relay/tls/relay.pem --tls-key /relay/tls/relay-key.pem --block-file /relay/app/blocklist.txt \
  --protect /relay/protected --no-config-endpoint --run-as "$(id -u):$(id -g)" \
  --log /relay/state/search_filter.jsonl --dump-dir /relay/state/dump \
  --allow-path /backend-api/codex/ --allow-path /backend-api/wham/accounts/check \
  --allow-path /backend-api/wham/settings/user --allow-path /oauth/token \
  --allow-tool-type function --allow-tool-type custom --allow-tool-type namespace "${EXTRA[@]}" >/dev/null
for _ in $(seq 60); do grep -q '"kind": "start"' "$W/state/search_filter.jsonl" 2>/dev/null && break; sleep 1; done
echo "== curl checks (from a container on the relay's network)"
docker run --rm --network "$NET" -v "$TLS/ca.pem:/ca.pem:ro" --entrypoint bash python:3.12-slim-bookworm -c '
for u in /__relay/health /backend-api/codex/models /backend-api/conversation /backend-api/wham/tasks/list /__search_filter/config; do
  printf "%-36s %s\n" "$u" "$(python3 -c "import ssl,sys,urllib.request,urllib.error
c=ssl.create_default_context(cafile=\"/ca.pem\")
try: print(urllib.request.urlopen(\"https://relay:8789\"+sys.argv[1],context=c,timeout=20).status)
except urllib.error.HTTPError as e: print(e.code)
except Exception as e: print(type(e).__name__)" "$u")"
done
printf "%-36s %s\n" "POST /oauth/token (empty)" "$(python3 -c "import ssl,urllib.request,urllib.error
c=ssl.create_default_context(cafile=\"/ca.pem\")
r=urllib.request.Request(\"https://relay:8789/oauth/token\",data=b\"{}\",headers={\"Content-Type\":\"application/json\"})
try: print(urllib.request.urlopen(r,context=c,timeout=20).status)
except urllib.error.HTTPError as e: print(e.code, \"(from auth.openai.com)\" if e.headers.get(\"server\") else \"\")
except Exception as e: print(type(e).__name__)")"
'
echo "== codex exec through the relay"
docker run --rm --network "$NET" --user "$(id -u):$(id -g)" -e HOME=/codex-home -e CODEX_HOME=/codex-home \
  -e CODEX_CA_CERTIFICATE=/relay-ca.pem -e CODEX_REFRESH_TOKEN_URL_OVERRIDE=https://relay:8789/oauth/token \
  -v "$TREE/.cache/codex-0.162.1:/opt/codex:ro" -v "$AUTH:/mnt/codex-auth/auth.json" -v "$TLS/ca.pem:/relay-ca.pem:ro" \
  -v "$W/home:/codex-home" --entrypoint bash python:3.12-slim-bookworm -c '
ln -sfn /mnt/codex-auth/auth.json /codex-home/auth.json && cd /codex-home &&
echo "Use your web search tool once to look up the height of the Eiffel Tower in metres, then reply with just the number." |
/opt/codex/bin/codex exec --ephemeral --skip-git-repo-check --json -m gpt-5.6-sol -c model_reasoning_effort="low" \
  -c web_search="live" -c openai_base_url="https://relay:8789/backend-api/codex" \
  -c chatgpt_base_url="https://relay:8789/backend-api/" -c check_for_update_on_startup=false - 2>&1 |
python3 -c "import json,sys
for l in sys.stdin:
    try: e=json.loads(l)
    except ValueError: print(\"  text:\", l.strip()[:160]); continue
    t=e.get(\"type\"); it=e.get(\"item\") or {}
    if t in (\"error\",\"turn.failed\",\"turn.completed\") or it.get(\"type\") in (\"web_search\",\"agent_message\",\"error\"):
        print(\" \", t, it.get(\"type\") or \"\", json.dumps(it.get(\"action\") or it.get(\"text\") or e.get(\"usage\") or e.get(\"message\") or e.get(\"error\"))[:200])"'
docker rm -f relay-selftest-$$ >/dev/null 2>&1; docker network rm "$NET" >/dev/null
echo "== relay log"
python3 -c "import json,sys
for l in open(sys.argv[1]):
    e=json.loads(l); e.pop(\"t\",None)
    for k in (\"hosts\",\"blocked\"): e[k]=len(e.get(k) or [])
    print(\" \", json.dumps(e)[:900])" "$W/state/search_filter.jsonl"
ls "$W/state/dump" | wc -l
rm -rf "$W"
