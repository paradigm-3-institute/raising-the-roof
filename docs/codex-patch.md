# The Codex compaction patch

Codex 0.160.0 and 0.162.1 can count the model's earlier reasoning twice in their estimate of the context size.
When a message arrives in the middle of a session (from a teammate, or the harness's continuation after an agent
stopped), `ContextManager::get_total_token_usage` adds an estimate of all earlier reasoning items to a token count
that already includes them, so Codex can compact well before its compaction point (in the post's self-organized
teams, at about 230k to 350k tokens instead of 450k). Later Codex releases fix the same double counting for resumed
and forked sessions (openai/codex PR #57839).

The patch removes that one addition. It is the same one line for both releases the post patched:

- `codex-patch/compaction-rust-v0.162.1.patch`, on tag `rust-v0.162.1` (commit `092d3acd6bec3e3a14bdc7e7a2810ab628ab759d`)
- `codex-patch/compaction-rust-v0.170.0.patch`, on tag `rust-v0.170.0` (commit `29eb79e9866b3262a777fdf8195c1b9e63684c91`)

```diff
             last_tokens
-                .saturating_add(self.get_non_last_reasoning_items_tokens())
                 .saturating_add(items_after_last_model_generated_tokens)
```

Runs started after the fix used the patched build: the reruns of the 32M self-organized teams of 8, Terra's teams
of 16 and 32, the restarted single agents at 8M, the no-talk pools with the plain prompt, the mixed SwarmChat teams
(Codex 0.162.1), and every GPT-6.2 Sol run (Codex 0.170.0). [experiments.md](experiments.md) shows the build each
run used (`--codex-binary .cache/codex-0.162.1-fix948/bin/codex` or `.cache/codex-0.170.0-fix948/bin/codex`).

## Building it

The patched package is the official release package with `bin/codex` replaced by a binary built from the patched
source. The post's builds followed the `x86_64-unknown-linux-musl` job of Codex's own release workflow
(`.github/workflows/rust-release.yml` at the tag): Rust 1.95.0 with the musl target, Zig 0.14.0, the workflow's
`install-musl-build-tools.sh`, its rusty_v8 artifact and checksum steps, then
`cargo build --target x86_64-unknown-linux-musl --release --bin codex`. A build reproduces the official binary
except for its OpenSSL build timestamp, so the patched binary's hash depends on when it was built.

```bash
# from the repository's root
git clone https://github.com/openai/codex.git codex-src
git -C codex-src checkout rust-v0.162.1
git -C codex-src apply "$PWD/codex-patch/compaction-rust-v0.162.1.patch"
# build codex for x86_64-unknown-linux-musl as rust-release.yml does (above); 0.170.0 needs about 28 GB of memory

# then: the official package of the same release, with the patched binary
install_codex 0.162.1                                   # docs/harness.md
cp -a .cache/codex-0.162.1 .cache/codex-0.162.1-fix948
cp codex-src/codex-rs/target/x86_64-unknown-linux-musl/release/codex .cache/codex-0.162.1-fix948/bin/codex
```

The same for 0.170.0 with its tag and patch. The release's `codex-package.json` still says the release's version,
which is what `run.py` checks.

The patch is a change to Codex's source (Apache-2.0, github.com/openai/codex).
