# The post's runs

Every run of the post, by the run name its job folder has (`jobs/<name>/`) and the data use. Each command is the
one the run was launched with (`python run.py ...`, after `uv sync` and the setup in [harness.md](harness.md)),
shown for one member of the family. A run name says how to fill in the rest:

- `{k}x{b}m`: a team of k agents with b million weighted tokens each (total 8M: `2x4m`, `4x2m`, `8x1m`; total
  32M: `2x16m`, `4x8m`, `8x4m`, `16x2m`, `32x1m`). `--agents k --token-budget b000000`.
- `bare{B}m`: one agent with B million (`--agents 1 --token-budget B000000`).
- `ultra{k}-8m`: Ultra mode with at most k agents and 8M for the whole tree.
- task: `gitzig` (`frontier-swe --task git-to-zig`), `ahc014` / `ahc038` (`ale-bench --problem ahc014`),
  `perf` (`perf-takehome`).
- model: `opus` = `claude-opus-5-5`, `sonnet` = `claude-sonnet-5-5`, `sol` = `gpt-6.1-sol`, `terra` =
  `gpt-6.1-terra`; older and newer: `opus5` = `claude-opus-5`, `sonnet5` = `claude-sonnet-5`, `terra56` =
  `gpt-5.6-terra`, `sol62` = `gpt-6.2-sol`.
- `-r{n}`: the repeat; `-p{nn}`: a member of a no-talk pool.
- prefix `b162-`: Codex 0.162.1 (the message board needs it); `f948-`: the patched Codex 0.162.1 build; `b170-`:
  the patched Codex 0.170.0 build. `f948` (and `fix948` in the CLI folders under `.cache/`) marks runs on the
  patched Codex build ([codex-patch.md](codex-patch.md)).

All runs used `--effort high`, `--resources-tool` and `--trials 1`. The CLI binaries are the ones `install_claude`
and `install_codex` put in `.cache/` ([harness.md](harness.md)); a run's `--codex-binary` names the Codex version
it used. `--ale-judge-cpus` pins ALE-Bench's judge to host CPUs and is specific to the machine (pick CPUs the
agents do not use). Codex runs also passed `--codex-auth` with the login to use (default `~/.codex/auth.json`).

## Main comparison (8M weighted tokens)

Single agent (`bare8m-{task}-{model}-r{n}`):

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 8000000 --resources-tool --timeout 172800 --name bare8m-gitzig-sol-r1
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 8000000 --resources-tool --timeout 864000 --name bare8m-gitzig-opus-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 8000000 --resources-tool --timeout 345600 --name bare8m-ahc014-sol-r4
python run.py ale-bench --problem ahc038 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 8000000 --resources-tool --timeout 864000 --name bare8m-ahc038-opus-r1
python run.py perf-takehome --cli codex --codex-binary .cache/codex-0.160.0/bin/codex --model gpt-6.1-terra --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 8000000 --resources-tool --timeout 864000 --name bare8m-perf-terra-r1
```

For GPT-6.1 Sol on AHC014 and AHC038, the single agent (`bare8m`), the protocol team of 4 (`swarm4x2m`) and the
self-organized teams of 2 and 4 (`b162-{2x4m,4x2m}-board`) count repeats 4 to 6; repeats 1 to 3 had slow judge
feedback and were replaced.

Protocol teams (`swarm{k}x{b}m-{task}-{model}-r{n}`):

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.160.0/bin/codex --model gpt-6.1-sol --effort high --protocol antiherding_matched --trials 1 --timeout 172800 --resources-tool --cpus 8 --memory-mb 32768 --agents 4 --token-budget 2000000 --name swarm4x2m-gitzig-sol-r1
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol antiherding_matched --agents 4 --trials 1 --token-budget 2000000 --timeout 172800 --resources-tool --cpus 8 --memory-mb 32768 --name swarm4x2m-gitzig-opus-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --cli codex --codex-binary .cache/codex-0.160.0/bin/codex --protocol antiherding_matched --model gpt-6.1-sol --effort high --agents 4 --trials 1 --token-budget 2000000 --timeout 345600 --resources-tool --name swarm4x2m-ahc014-sol-r4
python run.py perf-takehome --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol antiherding_matched --agents 4 --trials 1 --token-budget 2000000 --resources-tool --timeout 864000 --name swarm4x2m-perf-opus-r1
```

Self-organized teams, GPT (`b162-{k}x{b}m-board-{task}-{model}-r{n}`):

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol board --agents 4 --token-budget 2000000 --trials 1 --timeout 172800 --resources-tool --cpus 8 --memory-mb 32768 --name b162-4x2m-board-gitzig-sol-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --protocol board --model gpt-6.1-sol --effort high --agents 4 --trials 1 --token-budget 2000000 --timeout 345600 --resources-tool --name b162-4x2m-board-ahc014-sol-r4
python run.py perf-takehome --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol board --agents 2 --token-budget 4000000 --trials 1 --timeout 864000 --resources-tool --name b162-2x4m-board-perf-sol-r1
```

Self-organized teams, Claude (`peers{k}x{b}m-{task}-{model}-r{n}`):

```bash
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol claude_peers --agents 4 --trials 1 --token-budget 2000000 --timeout 345600 --resources-tool --cpus 8 --memory-mb 32768 --fastgrade --name peers4x2m-gitzig-opus-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 0,2,4,6,8,10,12,14 --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol claude_peers --agents 4 --trials 1 --token-budget 2000000 --timeout 864000 --resources-tool --name peers4x2m-ahc014-opus-r1
python run.py perf-takehome --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol claude_peers --agents 4 --trials 1 --token-budget 2000000 --resources-tool --timeout 864000 --name peers4x2m-perf-opus-r1
```

Orchestrated teams (`ultra{k}-8m-{task}-sol-r{n}`):

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol ultra --agents 4 --trials 1 --token-budget 8000000 --resources-tool --cpus 8 --memory-mb 32768 --timeout 172800 --name ultra4-8m-gitzig-sol-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol ultra --agents 4 --trials 1 --token-budget 8000000 --timeout 345600 --resources-tool --name ultra4-8m-ahc014-sol-r1
python run.py perf-takehome --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol ultra --agents 4 --trials 1 --token-budget 8000000 --resources-tool --timeout 864000 --name ultra4-8m-perf-sol-r1
```

## 32M weighted tokens

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 32000000 --resources-tool --timeout 345600 --name bare32m-gitzig-sol-r1
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 32000000 --resources-tool --timeout 864000 --fastgrade --content-filter-max-blocks 15 --name bare32m-gitzig-opus-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 16,18,19,20,21,22,23,24 --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-terra --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 32000000 --resources-tool --timeout 1382400 --name bare32m-ahc014-terra-r1
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol board --agents 8 --token-budget 4000000 --trials 1 --timeout 345600 --resources-tool --cpus 8 --memory-mb 32768 --fastgrade --name b162-8x4m-board-gitzig-sol-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 0,2,4,6,8,10,12,14 --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-terra --effort high --protocol board --agents 8 --trials 1 --token-budget 4000000 --resources-tool --timeout 1382400 --name b162-8x4m-board-ahc014-terra-r1
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol claude_peers --agents 8 --trials 1 --token-budget 4000000 --timeout 345600 --resources-tool --cpus 8 --memory-mb 32768 --fastgrade --name peers8x4m-gitzig-opus-r1
```

GPT-6.1 Terra's 32M git-to-zig runs (`bare32m-gitzig-terra`, `b162-{2x16m,4x8m,8x4m}-board-gitzig-terra`) used
`--timeout 691200`. A few git-to-zig comparisons have 10 repeats (`-r1` to `-r10`).

The reruns on the patched Codex build (`f948-8x4m-board-gitzig-{sol,terra}-r{n}`) and the teams of 16 and 32
(`f948-{16x2m,32x1m}-board-gitzig-terra-r{n}`):

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1-fix948/bin/codex --model gpt-6.1-sol --effort high --protocol board --agents 8 --token-budget 4000000 --trials 1 --timeout 345600 --resources-tool --cpus 8 --memory-mb 32768 --fastgrade --name f948-8x4m-board-gitzig-sol-r2
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1-fix948/bin/codex --model gpt-6.1-terra --effort high --protocol board --agents 32 --token-budget 1000000 --trials 1 --timeout 691200 --resources-tool --cpus 8 --memory-mb 32768 --fastgrade --name f948-32x1m-board-gitzig-terra-r1
```

`f948-8x4m-board-gitzig-sol-r1` is not counted (most of its agents ran on a smaller fallback context window);
repeat 4 replaces it.

## 2M weighted tokens (for λ)

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 2000000 --resources-tool --timeout 172800 --name bare2m-gitzig-sol-r1
python run.py ale-bench --problem ahc038 --ale-no-private-lane --ale-judge-cpus 25,26,27,28,32,33,34,35 --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-sol --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 2000000 --resources-tool --timeout 345600 --name bare2m-ahc038-sol-r1
```

## No-talk pools

Lone agents with the protocol prompt (`nt{k}x{b}m-{task}-{model}-p{nn}`) or the plain prompt
(`ntbare{k}x{b}m-{task}-{model}-p{nn}`); `{k}x{b}m` names the team size the pool stands in for (pools of 6 at 4M,
8 at 2M, 12 at 1M). On git-to-zig each lone agent gets a k-th of a team's resources (`--cpus 8/k --memory-mb
32768/k`).

```bash
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol antiherding_matched --agents 1 --trials 1 --token-budget 2000000 --resources-tool --timeout 172800 --cpus 2 --memory-mb 8192 --name nt4x2m-gitzig-opus-p01
python run.py ale-bench --problem ahc038 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --cli codex --codex-binary .cache/codex-0.160.0/bin/codex --model gpt-6.1-sol --effort high --protocol antiherding_matched --agents 1 --trials 1 --token-budget 4000000 --timeout 345600 --resources-tool --name nt2x4m-ahc038-sol-p01
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1-fix948/bin/codex --model gpt-6.1-sol --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 2000000 --resources-tool --fastgrade --timeout 172800 --cpus 2 --memory-mb 8192 --name ntbare4x2m-gitzig-sol-p01
python run.py perf-takehome --cli codex --codex-binary .cache/codex-0.162.1-fix948/bin/codex --model gpt-6.1-terra --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 4000000 --resources-tool --timeout 864000 --name ntbare2x4m-perf-terra-p01
```

## Matched single agent (`ale8m-{ahc014,ahc038}-{opus,sonnet,sol}-r{n}`)

```bash
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5-5 --effort high --protocol antiherding_matched --agents 1 --trials 1 --token-budget 8000000 --timeout 345600 --resources-tool --name ale8m-ahc014-opus-r1
```

## Restarted single agent (`relay8x{b}m-gitzig-{sol,terra}-r{n}`)

Eight sessions of b million each (8M: `relay8x1m`; 32M: `relay8x4m`):

```bash
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1-fix948/bin/codex --effort high --protocol bare_solo --agents 1 --trials 1 --resources-tool --fastgrade --model gpt-6.1-sol --relay 8 --token-budget 1000000 --timeout 172800 --name relay8x1m-gitzig-sol-r1
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-6.1-terra --effort high --protocol bare_solo --agents 1 --trials 1 --resources-tool --fastgrade --relay 8 --token-budget 4000000 --timeout 691200 --name relay8x4m-gitzig-terra-r1
```

## Mixed teams (appendix "Does model diversity help?")

Protocol teams, half Claude and half GPT-6.1 Sol (`swarm{k}x{b}m-gitzig-mixed`: Claude Sonnet 5.5 and Sol;
`swarm{k}x{b}m-{ahc014,perf}-opussol`: Claude Opus 5.5 and Sol), and self-organized mixed teams on SwarmChat
(`swarmchat{k}x{b}m-gitzig-mixed`):

```bash
python run.py frontier-swe --task git-to-zig --team claude:claude-sonnet-5-5:high,codex:gpt-6.1-sol:high,claude:claude-sonnet-5-5:high,codex:gpt-6.1-sol:high --effort high --claude-binary .cache/claude-2.1.289/claude --codex-binary .cache/codex-0.160.0/bin/codex --team-limit-hold --protocol antiherding_matched --agents 4 --trials 1 --token-budget 2000000 --timeout 172800 --resources-tool --cpus 8 --memory-mb 32768 --name swarm4x2m-gitzig-mixed-r1
python run.py ale-bench --problem ahc014 --ale-no-private-lane --ale-judge-cpus 4,6,8,10,12,14 --team claude:claude-opus-5-5:high,codex:gpt-6.1-sol:high,claude:claude-opus-5-5:high,codex:gpt-6.1-sol:high --effort high --claude-binary .cache/claude-2.1.289/claude --codex-binary .cache/codex-0.160.0/bin/codex --team-limit-hold --protocol antiherding_matched --trials 1 --timeout 864000 --resources-tool --agents 4 --token-budget 2000000 --name swarm4x2m-ahc014-opussol-r1
python run.py frontier-swe --task git-to-zig --team claude:claude-sonnet-5-5:high,codex:gpt-6.1-sol:high,claude:claude-sonnet-5-5:high,codex:gpt-6.1-sol:high --effort high --claude-binary .cache/claude-2.1.289/claude --codex-binary .cache/codex-0.162.1-fix948/bin/codex --team-limit-hold --protocol swarmchat --agents 4 --trials 1 --token-budget 2000000 --timeout 172800 --resources-tool --cpus 8 --memory-mb 32768 --lockstep-gap 250000 --fastgrade --name swarmchat4x2m-gitzig-mixed-r1
```

A team of k alternates the two models, Claude first.

## Older and newer models (appendix "Are models improving at collaborating?")

The same commands as the main comparison with the model swapped, on git-to-zig and AHC038. GPT-5.6 Terra's runs
pin Codex's compaction point (`--codex-compact-limit 450000`); GPT-6.2 Sol needs Codex 0.170.0 and ran on its
patched build (5 repeats):

```bash
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5 --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 8000000 --resources-tool --timeout 864000 --name bare8m-gitzig-opus5-r1
python run.py ale-bench --problem ahc038 --ale-no-private-lane --ale-judge-cpus 25,26,27,28,32,33,34,35 --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5 --effort high --protocol antiherding_matched --agents 4 --trials 1 --token-budget 2000000 --timeout 864000 --resources-tool --name swarm4x2m-ahc038-opus5-r1
python run.py frontier-swe --task git-to-zig --cli claude --claude-binary .cache/claude-2.1.289/claude --model claude-opus-5 --effort high --protocol claude_peers --agents 4 --trials 1 --token-budget 2000000 --timeout 345600 --resources-tool --cpus 8 --memory-mb 32768 --fastgrade --name peers4x2m-gitzig-opus5-r1
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-5.6-terra --effort high --codex-compact-limit 450000 --protocol board --agents 4 --token-budget 2000000 --trials 1 --timeout 172800 --resources-tool --cpus 8 --memory-mb 32768 --name b162-4x2m-board-gitzig-terra56-r1
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.170.0-fix948/bin/codex --model gpt-6.2-sol --effort high --protocol bare_solo --agents 1 --trials 1 --token-budget 8000000 --resources-tool --timeout 172800 --name bare8m-gitzig-sol62-r1
python run.py frontier-swe --task git-to-zig --cli codex --codex-binary .cache/codex-0.170.0-fix948/bin/codex --model gpt-6.2-sol --effort high --protocol board --agents 4 --token-budget 2000000 --trials 1 --timeout 172800 --resources-tool --cpus 8 --memory-mb 32768 --name b170-4x2m-board-gitzig-sol62-r1
```

## BrowseComp (appendix "Replicating OpenAI's multi-agent BrowseComp results")

GPT-5.6 Sol on 253 questions: the first 253 of a seeded random order of all 1,266 question ids
(`random.Random(20261018).sample(range(1, 1267), 1266)`), run as `pilot126-{arm}-{effort}` (the first 126) and
`pilot253-{arm}-{effort}` (the next 127). Every run had the search filter on (`BROWSECOMP_SEARCH_FILTER=on`) and
ran on Codex 0.162.1. `IDS` is the questions' ids, separated by commas. See [browsecomp.md](browsecomp.md).

```bash
export BROWSECOMP_SEARCH_FILTER=on
C="--cli codex --codex-binary .cache/codex-0.162.1/bin/codex --model gpt-5.6-sol --problem $IDS --concurrent 4"
python run.py browsecomp $C --effort high --protocol task_only --agents 1 --token-budget 1000000 --timeout 7200 --name pilot253-solo-high-r1-b2
python run.py browsecomp $C --effort high --protocol ultra --agents 4 --token-budget 1000000 --timeout 7200 --name pilot253-ultra-high-r1-b2a
python run.py browsecomp $C --effort high --protocol antiherding_matched --agents 4 --token-budget 1250000 --timeout 14400 --name pilot253-paper-high-r1-b2
python run.py browsecomp $C --effort high --protocol board --agents 4 --token-budget 1250000 --timeout 14400 --name pilot253-board-high-r1-b2
```

Low effort is the same with `--effort low` (single agent and Ultra). `paper` is the protocol team, `board` the
self-organized team, `solo` the single agent, `ultra` the orchestrated team.
