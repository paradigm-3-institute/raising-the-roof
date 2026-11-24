"""Codex search filter: a proxy that keeps chosen sites and texts out of Codex's web search.

Codex has no `blocked_domains` setting (openai/codex#23656), but the search backend accepts
one. Point Codex's OpenAI provider at this proxy and it adds the filter to every search:

    codex -c openai_base_url="http://127.0.0.1:8789/backend-api/codex" ...

Codex searches two ways, and both pass through here:
  * hosted search: a `web_search` tool in each /responses request (HTTP or WebSocket);
    the proxy adds `filters.blocked_domains` to it.
  * standalone search: POST /alpha/search; the proxy adds `settings.filters.blocked_domains`
    and drops commands that name a blocked domain or open a blocked URL.

With --protect (question files, or folders of them), it also removes every search result
and opened page that quotes a protected text (the rule: Protector), before Codex hands the
search output to the model. Hosted search runs on OpenAI's side, where the proxy cannot see
results, so in that mode the hosted tool is removed instead. Block lists and protected texts
are re-read whenever their files change. --dry-run only logs what it would withhold.

Anything the proxy cannot read (an unknown body encoding, non-JSON on a search route) is
refused rather than forwarded. Every search the model makes is logged to --log with the
result URLs it got back, so you can check that none is on a blocked domain.

Sidecar mode (BrowseComp): the proxy runs in its own container next to the
agents' (the "relay" service of each trial), which is the only host the agents' network
reaches. --host binds it there; /oauth/ requests go to auth.openai.com (Codex's token
refresh, CODEX_REFRESH_TOKEN_URL_OVERRIDE), everything else to chatgpt.com; --allow-path
limits the paths it forwards; --allow-tool-type limits the tools a /responses request may
offer the model, and web URLs in a request's input (images, files the backend would fetch)
are refused. --run-as drops root after the state directory is handed to the host's user.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

UPSTREAM = "https://chatgpt.com"
AUTH_UPSTREAM = "https://auth.openai.com"  # /oauth/... (token refresh)
HOP_HEADERS = {
    "host", "connection", "keep-alive", "proxy-connection", "transfer-encoding", "upgrade",
    "te", "trailer", "content-length", "content-encoding", "accept-encoding",
    "sec-websocket-key", "sec-websocket-version", "sec-websocket-extensions",
    "sec-websocket-accept", "sec-websocket-protocol",
}
URL_RE = re.compile(r"https?://[^\s\"'<>\\)\]]+")
CONFIG_PATH = "/__search_filter/config"
HEALTH_PATH = "/__relay/health"
RESULT_SEP = "\n" + "-" * 80 + "\n"  # between results in a standalone search's output text
REF_RE = re.compile(r"turn\d+[a-z_]+\d+")  # a result's ref id, e.g. turn1search5, turn2view0
# Request fields whose web URL the backend would fetch itself (an image or a file), out of the filter's sight.
FETCHED_URL_KEYS = ("image_url", "file_url")
QUOTE_RES = [re.compile(p, re.S) for p in (
    r'"([^"]{3,}?)"', r"“([^”]{3,}?)”", r"‘([^’]{3,}?)’",
    r"(?<![A-Za-z])'([^']{3,}?)'(?![A-Za-z])")]


class Sources:
    """The text of every file in the given files and folders, re-read whenever one changes
    (checked at most every `recheck` seconds)."""

    def __init__(self, paths, recheck: float = 5.0):
        self.paths = [Path(p) for p in paths]
        missing = [str(p) for p in self.paths if not p.exists()]
        if missing:
            raise SystemExit(f"not found: {', '.join(missing)}")
        self.recheck = recheck
        self._key, self._texts, self._names, self._checked = None, [], [], 0.0

    def _files(self):
        files = []
        for p in self.paths:
            if p.is_dir():
                files += sorted(f for f in p.rglob("*") if f.is_file() and not f.name.startswith("."))
            elif p.is_file():
                files.append(p)
        return files

    def texts(self) -> list[str]:
        """The same list object until a file changes, so callers can cache on identity."""
        if self._key is not None and time.monotonic() - self._checked < self.recheck:
            return self._texts
        self._checked = time.monotonic()
        files = self._files()
        key = [(str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files]
        if key != self._key:
            self._key = key
            self._names = [f.name for f in files]
            self._texts = [f.read_text(errors="replace") for f in files]
        return self._texts

    def names(self) -> list[str]:
        self.texts()
        return self._names


def words(text: str) -> list[str]:
    """Lowercased letter runs and digit runs, so "x²+y^2" and "x2 + y 2" give the same words.
    LaTeX commands and the line markers of an opened page ("L12: ", "L39@P0: ") are dropped."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(?m)^l\d+\S{0,12}?: ?", " ", text)
    text = re.sub(r"\\[a-z]+", " ", text)
    return re.findall(r"[^\W\d_]+|\d+", text)


def quoted_positions(text: str, w: list[str]) -> set[int]:
    """Word positions of `w` (the words of `text`) inside a quoted span of the text: titles,
    lyrics, quotations, which an honest source can hold verbatim."""
    inside: set[int] = set()
    for rx in QUOTE_RES:
        for m in rx.finditer(text):
            span = words(m.group(1))
            if not span:
                continue
            for j in range(len(w) - len(span) + 1):
                if w[j] == span[0] and w[j:j + len(span)] == span:
                    inside.update(range(j, j + len(span)))
    return inside


class Protector:
    """Finds text that quotes a protected text.

    Base rule: min_words or more consecutive words of a protected text, or
    min_coverage of its 5-word phrases (which survives the stray line numbers and broken
    superscripts of extracted PDF text).

    Tuning (BrowseComp, from false-alarm checks):
      skip_quoted   a run or phrase that lies wholly inside a quoted span of the protected text
                    does not count (a source can quote the same title, lyric or line);
      stock_df      a phrase found in this many protected texts or more is stock phrasing
                    ("what is the name of the") and does not count;
      min_cov_hits  the coverage rule also needs this many distinct protected phrases found;
      min_quoted    the run rule needs this many words of the protected text quoted, in pieces of
                    min_words or more (one 11-word phrase a question shares with its source is not
                    enough at 15; a 15-word stretch, or two 10-word pieces, is).
    """

    K = 5

    def __init__(self, sources: Sources, min_words: int, min_coverage: float, *, skip_quoted: bool = False,
                 stock_df: int = 0, min_cov_hits: int = 0, min_quoted: int = 0):
        self.sources, self.n, self.coverage = sources, min_words, min_coverage
        self.skip_quoted, self.stock_df, self.min_cov_hits = skip_quoted, stock_df, min_cov_hits
        self.min_quoted = max(min_quoted, min_words)
        self._src = None

    def _load(self):
        texts = self.sources.texts()
        if texts is self._src:
            return
        names = self.sources.names()
        per_text = []
        df: dict[tuple, int] = {}
        for t in texts:
            w = words(t)
            inside = quoted_positions(t, w) if self.skip_quoted else set()
            runs = {}  # each min_words phrase -> its first position in the text
            for i in range(len(w) - self.n + 1):
                if not inside.issuperset(range(i, i + self.n)):
                    runs.setdefault(tuple(w[i:i + self.n]), i)
            phrases = {tuple(w[i:i + self.K]) for i in range(len(w) - self.K + 1)
                       if not inside.issuperset(range(i, i + self.K))}
            for g in set(runs) | phrases:
                df[g] = df.get(g, 0) + 1
            per_text.append((w, runs, phrases))
        stock = {g for g, d in df.items() if self.stock_df and d >= self.stock_df}
        self.grams, self.short, self.index, self.sizes, self.names = {}, [], {}, [], names
        for t, (w, runs, phrases) in enumerate(per_text):
            if len(w) >= self.n:
                for g, i in runs.items():
                    if g not in stock:
                        self.grams.setdefault(g, []).append((t, i))
            elif w:  # shorter than min_words: the whole text must appear
                self.short.append((" " + " ".join(w) + " ", t))
            kept = phrases - stock
            for g in kept:
                self.index.setdefault(g, []).append(t)
            self.sizes.append(len(kept))
        self.n_stock = len(stock)
        self._src = texts

    def count(self) -> int:
        self._load()
        return len(self._src)

    def explain(self, text: str) -> list[dict]:
        """Every protected text this text quotes, with the rule that found it."""
        self._load()
        w = words(text)
        found: dict[int, dict] = {}
        pos: dict[int, list[int]] = {}
        quoted: dict[int, set] = {}  # words of each protected text the page quotes (in min_words pieces)
        for i in range(len(w) - self.n + 1):
            for t, j in self.grams.get(tuple(w[i:i + self.n]), ()):
                pos.setdefault(t, []).append(i)
                quoted.setdefault(t, set()).update(range(j, j + self.n))
        for t, ps in pos.items():
            if len(quoted[t]) < self.min_quoted:
                continue
            best = run = 1
            start = best_start = ps[0]
            for a, b in zip(ps, ps[1:]):
                if b == a + 1:
                    run += 1
                else:
                    run, start = 1, b
                if run > best:
                    best, best_start = run, start
            found[t] = {"run": best + self.n - 1, "quoted": len(quoted[t]),
                        "excerpt": " ".join(w[best_start:best_start + min(best + self.n - 1, 12)])}
        joined = " " + " ".join(w) + " "
        for s, t in self.short:
            if s in joined:
                found.setdefault(t, {})["whole"] = True
        hits: dict[int, int] = {}
        for g in {tuple(w[i:i + self.K]) for i in range(len(w) - self.K + 1)}:
            for t in self.index.get(g, ()):
                hits[t] = hits.get(t, 0) + 1
        for t, h in hits.items():
            size = self.sizes[t]
            if size >= 4 and h >= self.min_cov_hits and h / size >= self.coverage:
                found.setdefault(t, {})["coverage"] = [h, size]
        return [{"text": self.names[t], **v} for t, v in sorted(found.items())]

    def matches(self, text: str) -> bool:
        return bool(self.explain(text))


def redact_search_response(resp: dict, protector: Protector, dry_run: bool = False,
                           policy: "Policy | None" = None) -> list[dict]:
    """Remove every result (its title, URL and text) that quotes a protected text, or whose URL
    wraps a blocked site (policy), in place (dry_run: only report them). Returns what was (or
    would be) removed: each block's title line and why."""
    output = resp["output"]
    if not isinstance(output, str):
        raise TypeError("search output is not text")
    results = resp.get("results") if isinstance(resp.get("results"), list) else []
    bad_refs, why_ref = set(), {}
    for r in results:
        if isinstance(r, dict) and r.get("ref_id"):
            m = protector.explain(" ".join(str(r.get(k) or "") for k in ("title", "snippet", "url")))
            if m:
                bad_refs.add(r.get("ref_id"))
                why_ref[r.get("ref_id")] = m
    kept, removed = [], []
    for block in output.split(RESULT_SEP):
        refs = set(REF_RE.findall(block))
        m = protector.explain(block)
        head_urls = URL_RE.findall(block.split("\n", 1)[0])
        if policy and any(policy.blocks_ref(u) for u in head_urls):
            m = [*m, "URL wraps a blocked site"]
        if m or refs & bad_refs:
            removed.append({"title": block.split("\n", 1)[0][:200], "refs": sorted(refs)[:5],
                            "matches": m or [x for r in sorted(refs & bad_refs) for x in why_ref.get(r, [])][:5]
                            or ["ref of a withheld result"]})
            bad_refs |= refs
        else:
            kept.append(block)
    if dry_run:
        return removed
    resp["output"] = RESULT_SEP.join(kept) if kept else "No results."
    if "results" in resp and isinstance(resp["results"], list):
        resp["results"] = [r for r in results if not (isinstance(r, dict) and r.get("ref_id") in bad_refs)]
    # Codex gives the model only `output` today; drop the encrypted copy in case that changes.
    resp["encrypted_output"] = None
    return removed


class Policy:
    def __init__(self, blocked: list[str], block_files: Sources | None = None, drop_hosted=False,
                 tool_types: list[str] | None = None, enforce_tool_types: bool = True):
        self.static, self.files, self.drop_hosted = list(blocked), block_files, drop_hosted
        self.tool_types, self.enforce_tool_types = set(tool_types or []), enforce_tool_types
        self._src, self._blocked = None, []

    @property
    def blocked(self) -> list[str]:
        texts = self.files.texts() if self.files else []
        if texts is not self._src:
            lines = [l for t in texts for l in t.splitlines() if not l.strip().startswith("#")]
            self._src = texts
            self._blocked = sorted({norm_domain(d) for d in [*self.static, *lines] if norm_domain(d)})
        return self._blocked

    def blocks_host(self, host: str) -> bool:
        host = norm_domain(host)
        return bool(host) and any(host == d or host.endswith("." + d) for d in self.blocked)

    def blocks_ref(self, ref: str) -> bool:
        """A search ref id is either an opaque id (turn0search3) or a URL/bare host. A URL that wraps a blocked
        site (an archive copy, a translation or reader proxy: web.archive.org/web/2024/huggingface.co/...,
        huggingface-co.translate.goog) is blocked too."""
        ref = ref.strip()
        if "://" in ref:
            return self.blocks_host(urlsplit(ref).hostname or "") or self.wraps_blocked(ref)
        first = ref.split("/", 1)[0]
        return ("." in first and self.blocks_host(first)) or ("/" in ref and self.wraps_blocked(ref))

    def wraps_blocked(self, url: str) -> bool:
        url = url.lower()
        return any(re.search(r"(?<![a-z0-9-])" + re.escape(d) + r"(?![a-z0-9-])", url)
                   or d.replace("-", "--").replace(".", "-") + ".translate.goog" in url for d in self.blocked)

    def merge_filters(self, filters) -> dict | None:
        """The filters with our blocklist applied; None means no search may run at all."""
        filters = dict(filters or {})
        allowed = filters.get("allowed_domains")
        if allowed:
            kept = [d for d in allowed if not self.blocks_host(d)]
            if not kept:
                return None
            filters["allowed_domains"] = kept
        blocked = sorted({*map(norm_domain, filters.get("blocked_domains") or []), *self.blocked})
        if blocked:  # the backend rejects empty lists
            filters["blocked_domains"] = blocked
        return filters


def norm_domain(d: str) -> str:
    d = str(d).strip().lower()
    if "://" in d:
        d = urlsplit(d).hostname or ""
    d = d.split("/", 1)[0].removeprefix("*.").rstrip(".")
    return d


def rewrite_tools(tools, policy: Policy, dropped: list | None = None) -> tuple[list, int]:
    """Filter every web_search tool (also inside namespaces); drop one that can't be filtered,
    and (with --allow-tool-type) every tool of another type."""
    out, n = [], 0
    for tool in tools:
        kind = str(tool.get("type", "")) if isinstance(tool, dict) else type(tool).__name__
        if isinstance(tool, dict) and kind.startswith("web_search"):
            filters = policy.merge_filters(tool.get("filters"))
            n += 1
            if filters is None or policy.drop_hosted:
                if dropped is not None:
                    dropped.append(kind)
                continue
            tool = {**tool, "filters": filters}
        elif isinstance(tool, dict) and isinstance(tool.get("tools"), list):
            inner, k = rewrite_tools(tool["tools"], policy, dropped)
            tool, n = {**tool, "tools": inner}, n + k
        if policy.tool_types and kind not in policy.tool_types:
            if dropped is not None:
                dropped.append(kind if policy.enforce_tool_types else f"{kind} (dry run: kept)")
            if policy.enforce_tool_types:
                continue
        out.append(tool)
    return out, n


def tool_types(tools) -> list[str]:
    found = []
    for tool in tools or []:
        if isinstance(tool, dict):
            found.append(str(tool.get("type")))
            if isinstance(tool.get("tools"), list):
                found += [f"{tool.get('type')}/{t}" for t in tool_types(tool["tools"])]
    return found


def fetched_urls(node, found: list | None = None) -> list[str]:
    """Web URLs in a request that the backend would fetch itself (input images and files)."""
    found = [] if found is None else found
    if isinstance(node, dict):
        for k, v in node.items():
            if k in FETCHED_URL_KEYS:
                url = v.get("url") if isinstance(v, dict) else v
                if isinstance(url, str) and not url.startswith("data:"):
                    found.append(url[:200])
            else:
                fetched_urls(v, found)
    elif isinstance(node, list):
        for v in node:
            fetched_urls(v, found)
    return found


def rewrite_responses_body(body: dict, policy: Policy, note: dict | None = None) -> int:
    """Filter a /responses request (or a WebSocket response.create) in place; returns how many
    web_search tools it had. Raises PermissionError for an input the backend would fetch."""
    urls = fetched_urls({k: v for k, v in body.items() if k != "tools"})
    if urls:
        raise PermissionError(f"web URLs in the input are not allowed: {urls[:3]}")
    if note is not None:
        note["keys"] = note.get("keys", []) + sorted(body.keys())[:40]
        if os.environ.get("RELAY_DEBUG_FIELDS"):
            note["fields"] = {k: (v if len(json.dumps(v)) < 400 else f"<{type(v).__name__} {len(json.dumps(v))}B>")
                              for k, v in body.items() if k != "input"}
            if isinstance(body.get("client_metadata"), dict):
                note["client_metadata"] = {k: (str(v)[:120]) for k, v in body["client_metadata"].items()}
            if isinstance(body.get("input"), list):
                note["input_items"] = [(i.get("type"), i.get("role"), len(json.dumps(i))) for i in body["input"]
                                       if isinstance(i, dict)][:12]
    if isinstance(body.get("response"), dict):  # a WebSocket response.create may wrap the request
        return rewrite_responses_body(body["response"], policy, note)
    if not isinstance(body.get("tools"), list):
        return 0
    if note is not None:
        note["tool_types"] = sorted(set(tool_types(body["tools"])))
    dropped: list = []
    body["tools"], n = rewrite_tools(body["tools"], policy, dropped)
    if note is not None and dropped:
        note["tools_dropped"] = dropped
    return n


def rewrite_search_body(body: dict, policy: Policy) -> list[str]:
    """Filter a standalone /alpha/search request in place; returns what was dropped."""
    dropped = []
    if policy.drop_hosted and isinstance(body.get("commands"), dict) and body["commands"].get("screenshot"):
        raise PermissionError("screenshots are disabled")  # an image can't be checked for protected text
    settings = dict(body.get("settings") or {})
    filters = policy.merge_filters(settings.get("filters"))
    if filters is None:
        raise PermissionError("every allowed domain is blocked")
    if filters:
        settings["filters"] = filters
    body["settings"] = settings
    commands = body.get("commands")
    if not isinstance(commands, dict):
        return dropped
    for name in ("search_query", "image_query"):
        kept = []
        for q in commands.get(name) or []:
            domains = q.get("domains")
            if domains:
                ok = [d for d in domains if not policy.blocks_host(d)]
                if not ok:
                    dropped.append(f"{name}:{q.get('q')!r} domains={domains}")
                    continue
                q = {**q, "domains": ok}
            kept.append(q)
        if name in commands:
            commands[name] = kept
    for name in ("open", "find", "screenshot"):
        if name in commands:
            kept = [op for op in commands[name] or [] if not policy.blocks_ref(str(op.get("ref_id", "")))]
            dropped += [f"{name}:{op.get('ref_id')}" for op in commands[name] or [] if op not in kept]
            commands[name] = kept
    # The backend rejects empty command lists, and a request left with no commands at all
    # should tell the model why rather than fail on an empty array.
    for name in [k for k, v in commands.items() if isinstance(v, list) and not v]:
        del commands[name]
    if dropped and not any(isinstance(v, list) for v in commands.values()):
        raise PermissionError(f"{', '.join(policy.blocked)} are blocked for web search and opening")
    return dropped


class Proxy:
    def __init__(self, policy: Policy, log_path: Path, dump_dir: Path | None = None,
                 protector: Protector | None = None, config: str = "", dry_run: bool = False,
                 allow_paths: list[str] | None = None, config_endpoint: bool = True):
        self.policy = policy
        self.config = config
        self.protector = protector
        self.log_path = log_path
        self.dump_dir = dump_dir
        self.dry_run = dry_run
        self.allow_paths = list(allow_paths or [])
        self.config_endpoint = config_endpoint
        self.session: aiohttp.ClientSession | None = None
        self.seen_paths: set = set()

    def log(self, **event):
        event = {"t": round(time.time(), 3), **event}
        with self.log_path.open("a") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        print(json.dumps(event)[:400], file=sys.stderr, flush=True)

    def hosts_in(self, text: str) -> dict:
        """Hosts of every URL in the text; a blocked one here may only be quoted by another page."""
        hosts = sorted({urlsplit(u).hostname or "" for u in URL_RE.findall(text)} - {""})
        return {"hosts": hosts, "blocked_mentioned": [h for h in hosts if self.policy.blocks_host(h)]}

    def search_results(self, text: str) -> dict:
        """The result URLs of a standalone search, and any that are on a blocked domain."""
        try:
            results = json.loads(text).get("results") or []
        except (json.JSONDecodeError, AttributeError):
            return {}
        urls = [r.get("url", "") for r in results if isinstance(r, dict)]
        return {"n_results": len(urls),
                "blocked_results": [u for u in urls if self.policy.blocks_host(urlsplit(u).hostname or "")]}

    async def start(self, app):
        self.session = aiohttp.ClientSession(auto_decompress=True, timeout=aiohttp.ClientTimeout(total=None))

    async def stop(self, app):
        await self.session.close()

    @staticmethod
    def forward_headers(headers) -> dict:
        out = {k: v for k, v in headers.items() if k.lower() not in HOP_HEADERS}
        out["Accept-Encoding"] = "gzip, deflate"
        return out

    @staticmethod
    def upstream(path_qs: str) -> str:
        return (AUTH_UPSTREAM if path_qs.startswith("/oauth/") else UPSTREAM) + path_qs

    async def decode_body(self, request: web.Request) -> bytes:
        raw = await request.read()
        enc = request.headers.get("Content-Encoding", "identity").lower()
        if enc in ("", "identity"):
            return raw
        if enc == "zstd":
            p = await asyncio.create_subprocess_exec("zstd", "-dcq", stdin=asyncio.subprocess.PIPE,
                                                     stdout=asyncio.subprocess.PIPE)
            out, _ = await p.communicate(raw)
            if p.returncode == 0:
                return out
        raise ValueError(f"cannot decode request body encoding {enc!r}")

    def path_allowed(self, path: str) -> bool:
        return not self.allow_paths or any(path == p or path.startswith(p) for p in self.allow_paths)

    async def handle(self, request: web.Request):
        if request.path == HEALTH_PATH:
            return web.Response(text="ok")
        if request.path == CONFIG_PATH and self.config_endpoint:  # codex-filtered checks a running proxy
            return web.Response(text=self.config)
        path = request.path
        if not self.path_allowed(path):
            if not self.dry_run:
                self.log(kind="refused", path=path[:200], method=request.method, reason="path not allowed")
                return web.json_response({"error": {"message": "not found"}}, status=404)
            self.log(kind="would_refuse", path=path[:200], method=request.method, reason="path not allowed")
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await self.handle_ws(request)
        is_search = path.rstrip("/").endswith("/alpha/search")
        is_responses = path.rstrip("/").endswith("/responses")
        try:
            body = await self.decode_body(request)
        except ValueError as e:
            self.log(kind="refused", path=path, reason=str(e))
            return web.json_response({"error": {"message": f"search filter proxy: {e}"}}, status=415)

        note = {}
        if (is_search or is_responses) and body:
            try:
                data = json.loads(body)
                if is_search:
                    note["dropped"] = rewrite_search_body(data, self.policy)
                    note["commands"] = data.get("commands")
                else:
                    note["web_search_tools"] = rewrite_responses_body(data, self.policy, note)
            except PermissionError as e:
                self.log(kind="refused", path=path, reason=str(e))
                return web.json_response({"error": {"message": f"search blocked: {e}"}}, status=403)
            except (json.JSONDecodeError, AttributeError, TypeError) as e:
                self.log(kind="refused", path=path, reason=f"unreadable body: {e}")
                return web.json_response({"error": {"message": "search filter proxy: unreadable body"}},
                                         status=400)
            body = json.dumps(data).encode()

        async with self.session.request(request.method, self.upstream(request.path_qs),
                                        headers=self.forward_headers(request.headers),
                                        data=body or None, allow_redirects=False) as up:
            if is_search:  # a plain JSON reply: read it whole so it can be filtered
                return self.finish_search(up, await up.read(), body, note)
            resp = web.StreamResponse(status=up.status, headers={
                k: v for k, v in up.headers.items() if k.lower() not in HOP_HEADERS})
            await resp.prepare(request)
            sse = SseWatcher(self) if is_responses else None
            async for chunk in up.content.iter_any():
                await resp.write(chunk)
                if sse:
                    sse.feed(chunk)
            await resp.write_eof()
        if is_responses:
            self.log(kind="responses", path=path, status=up.status, **note)
        elif (request.method, path, up.status) not in self.seen_paths:
            self.seen_paths.add((request.method, path, up.status))
            self.log(kind="other", method=request.method, path=path[:200], status=up.status)
        return resp

    def finish_search(self, up, raw: bytes, body: bytes, note: dict) -> web.Response:
        text = filtered = raw.decode("utf-8", "replace")
        withheld = []
        if self.protector and up.status == 200:
            try:
                data = json.loads(text)
                withheld = redact_search_response(data, self.protector, dry_run=self.dry_run, policy=self.policy)
            except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
                self.log(kind="refused", path="alpha/search", reason=f"unreadable search response: {e}")
                return web.json_response({"error": {"message": "search filter proxy: unreadable search "
                                                    "response"}}, status=502)
            if not self.dry_run:
                filtered = json.dumps(data)
        dump = None
        if self.dump_dir:
            dump = f"search_{time.time():.3f}"
            with gzip.open(self.dump_dir / f"{dump}.json.gz", "wt") as f:
                json.dump({"request": json.loads(body) if body else None, "status": up.status,
                           "response": text}, f)
        self.log(kind="standalone_search", status=up.status, **note,
                 **({"would_withhold": withheld} if self.dry_run else {"withheld": withheld}),
                 **({"dump": dump} if dump else {}),
                 **self.search_results(filtered), **self.hosts_in(filtered))
        return web.Response(status=up.status, body=filtered.encode(), headers={
            k: v for k, v in up.headers.items() if k.lower() not in HOP_HEADERS})

    async def handle_ws(self, request: web.Request):
        path = request.path
        try:
            up = await self.session.ws_connect(self.upstream(request.path_qs),
                                               headers=self.forward_headers(request.headers),
                                               max_msg_size=0, autoping=True)
        except aiohttp.WSServerHandshakeError as e:
            self.log(kind="ws_handshake_failed", path=path, status=e.status)
            return web.Response(status=e.status or 502, text=str(e.message))
        hs_headers = {k: v for k, v in (up._response.headers.items() if up._response else [])
                      if k.lower() not in HOP_HEADERS and not k.lower().startswith("sec-websocket")}
        ws = web.WebSocketResponse(max_msg_size=0, autoping=True)
        ws.headers.update(hs_headers)
        await ws.prepare(request)
        self.log(kind="ws_open", path=path)
        watcher = SseWatcher(self)

        async def client_to_up():
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        self.log(kind="refused", path=path, reason="non-JSON websocket message")
                        break
                    if isinstance(data, dict) and data.get("type") == "response.create":
                        note = {}
                        try:
                            n = rewrite_responses_body(data, self.policy, note)
                        except PermissionError as e:
                            self.log(kind="refused", path=path, reason=str(e))
                            break
                        self.log(kind="responses_ws", web_search_tools=n, **note,
                                 **({"hosted_search_removed": n} if n and self.policy.drop_hosted else {}))
                    elif isinstance(data, dict):
                        self.log(kind="ws_message", type=str(data.get("type"))[:80])
                    await up.send_str(json.dumps(data))
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self.log(kind="refused", path=path, reason="binary websocket message")
                    break
                else:
                    break
            await up.close()

        async def up_to_client():
            async for msg in up:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    watcher.event(msg.data)
                    await ws.send_str(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await ws.send_bytes(msg.data)
                else:
                    break
            await ws.close()

        await asyncio.gather(client_to_up(), up_to_client(), return_exceptions=True)
        return ws


class SseWatcher:
    """Logs each finished web_search_call (and the hosts in it) from a response stream."""

    def __init__(self, proxy: Proxy):
        self.proxy, self.buf = proxy, ""

    def feed(self, chunk: bytes):
        self.buf += chunk.decode("utf-8", "replace")
        *lines, self.buf = self.buf.split("\n")
        for line in lines:
            if line.startswith("data:"):
                self.event(line[5:].strip())

    def event(self, data: str):
        if "web_search_call" not in data and "url_citation" not in data:
            return
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            return
        item = ev.get("item") or {}
        if ev.get("type") == "response.output_item.done" and item.get("type") == "web_search_call":
            self.proxy.log(kind="hosted_search", action=item.get("action"), **self.proxy.hosts_in(data))
        elif ev.get("type") == "response.output_item.done" and "url_citation" in data:
            self.proxy.log(kind="citations", **self.proxy.hosts_in(data))


def drop_root(run_as: str, owned: list[Path]):
    """Hand the state files to the host's user, then run as that user (the sidecar starts as root
    because Docker creates a missing bind-mount directory as root)."""
    uid, gid = (int(x) for x in run_as.split(":"))
    for p in owned:
        if p.exists():
            os.chown(p, uid, gid)
    os.setgroups([])
    os.setgid(gid)
    os.setuid(uid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", action="append", default=[], help="domain to block (repeatable)")
    ap.add_argument("--block-file", type=Path, action="append", default=[],
                    help="file with one domain per line (repeatable; re-read when it changes)")
    ap.add_argument("--protect", type=Path, action="append", default=[],
                    help="file with a text to keep out of results, or a folder of such files "
                         "(repeatable; re-read when they change)")
    ap.add_argument("--min-words", type=int, default=10,
                    help="remove a result quoting this many consecutive words of a protected text")
    ap.add_argument("--min-coverage", type=float, default=0.5,
                    help="or containing this share of a protected text's 5-word phrases")
    ap.add_argument("--skip-quoted", action="store_true",
                    help="words inside a quoted span of a protected text do not count")
    ap.add_argument("--stock-df", type=int, default=0,
                    help="a phrase in this many protected texts or more does not count (0: off)")
    ap.add_argument("--min-cov-hits", type=int, default=0,
                    help="the coverage rule also needs this many distinct protected phrases")
    ap.add_argument("--min-quoted", type=int, default=0,
                    help="the run rule needs this many words of a protected text quoted (in --min-words pieces)")
    ap.add_argument("--dry-run", action="store_true", help="log what would be withheld; withhold nothing")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8789)
    ap.add_argument("--tls-cert", type=Path, help="serve HTTPS with this certificate (Codex accepts a plain-HTTP "
                                                  "base URL only on loopback)")
    ap.add_argument("--tls-key", type=Path, help="the certificate's private key")
    ap.add_argument("--allow-path", action="append", default=[],
                    help="forward only paths starting with one of these (repeatable; default: all)")
    ap.add_argument("--allow-tool-type", action="append", default=[],
                    help="a /responses request keeps only tools of these types (repeatable; default: all)")
    ap.add_argument("--no-config-endpoint", action="store_true", help=f"do not serve {CONFIG_PATH}")
    ap.add_argument("--run-as", help="UID:GID: own the log and dumps, and run as this user")
    ap.add_argument("--log", type=Path, default=Path("search_filter.jsonl"))
    ap.add_argument("--dump-dir", type=Path, help="save each standalone search request/response here (gzip)")
    ap.add_argument("--print-config", action="store_true",
                    help="print the settings a proxy started with these options enforces, and exit")
    args = ap.parse_args()
    if not (args.block or args.block_file or args.protect):
        ap.error("give --block, --block-file or --protect")
    config = json.dumps({"block": sorted(args.block),
                         "block_files": [str(f.resolve()) for f in args.block_file],
                         "protect": [str(f.resolve()) for f in args.protect],
                         "min_words": args.min_words if args.protect else None,
                         "min_coverage": args.min_coverage if args.protect else None,
                         **({"skip_quoted": True} if args.skip_quoted else {}),
                         **({"stock_df": args.stock_df} if args.stock_df else {}),
                         **({"min_cov_hits": args.min_cov_hits} if args.min_cov_hits else {}),
                         **({"min_quoted": args.min_quoted} if args.min_quoted else {}),
                         **({"dry_run": True} if args.dry_run else {})})
    if args.print_config:
        print(config)
        return
    if args.dump_dir:
        args.dump_dir.mkdir(parents=True, exist_ok=True)
    ssl_context = None
    if args.tls_cert:
        import ssl
        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        ssl_context.load_cert_chain(args.tls_cert, args.tls_key)
    if args.run_as:
        args.log.touch()
        drop_root(args.run_as, [args.log.parent, args.log, *([args.dump_dir] if args.dump_dir else [])])
    protector = Protector(Sources(args.protect), args.min_words, args.min_coverage, skip_quoted=args.skip_quoted,
                          stock_df=args.stock_df, min_cov_hits=args.min_cov_hits,
                          min_quoted=args.min_quoted) if args.protect else None
    # A dry run withholds nothing and refuses no path or tool type (it logs them); the hosted web_search tool is still
    # removed, so the searches it logs are the ones a real run would filter.
    policy = Policy(args.block, Sources(args.block_file) if args.block_file else None,
                    drop_hosted=protector is not None, tool_types=args.allow_tool_type,
                    enforce_tool_types=not args.dry_run)
    proxy = Proxy(policy, args.log, args.dump_dir, protector, config, dry_run=args.dry_run,
                  allow_paths=args.allow_path, config_endpoint=not args.no_config_endpoint)
    app = web.Application(client_max_size=0)
    app.on_startup.append(proxy.start)
    app.on_cleanup.append(proxy.stop)
    app.router.add_route("*", "/{tail:.*}", proxy.handle)
    n = protector.count() if protector else 0
    proxy.log(kind="start", blocked=policy.blocked, protected_texts=n, dry_run=args.dry_run,
              rule={"min_words": args.min_words, "min_coverage": args.min_coverage, "skip_quoted": args.skip_quoted,
                    "stock_df": args.stock_df, "stock_phrases": protector.n_stock if protector else 0,
                    "min_cov_hits": args.min_cov_hits, "min_quoted": args.min_quoted},
              allow_paths=args.allow_path, allow_tool_types=args.allow_tool_type)
    print(f"blocking {policy.blocked}; protecting {n} texts (min {args.min_words} words or "
          f"{args.min_coverage:.0%} of phrases){' DRY RUN' if args.dry_run else ''} on {args.host}:{args.port}",
          file=sys.stderr)
    # auto_decompress=False: Codex zstd-compresses request bodies; decode_body handles them.
    web.run_app(app, host=args.host, port=args.port, print=None, auto_decompress=False, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
